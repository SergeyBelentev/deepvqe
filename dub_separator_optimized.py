from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from dub_separator import (
    BandSpec,
    DubSeparatorConfig as _BaseDubSeparatorConfig,
    STFTFrontend,
    StereoPairFeatureExtractor,
    ConvNormAct,
    LearnablePositionalBias,
    BranchInputStem,
    SwiGLU,
    FactorizedTokenPosition,
    RefineBlock,
)


@dataclass(slots=True)
class DubSeparatorConfig(_BaseDubSeparatorConfig):
    pretrunk_num_heads: int = 4
    pretrunk_ff_mult: float = 2.0
    gradient_checkpointing: bool = False
    checkpoint_encoder_stages: bool = True
    checkpoint_pretrunk: bool = True
    checkpoint_trunk: bool = True
    checkpoint_decoder: bool = True
    checkpoint_detokenizer: bool = False


class SDPAMultiheadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def _shape(self, x: Tensor) -> Tensor:
        b, l, c = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        qh = self._shape(self.q_proj(q))
        kh = self._shape(self.k_proj(k))
        vh = self._shape(self.v_proj(v))
        out = F.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        b, h, l, d = out.shape
        out = out.transpose(1, 2).reshape(b, l, h * d)
        return self.out_proj(out)


class AxialAttention1D(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = SDPAMultiheadAttention(dim, num_heads, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.attn(y, y, y)
        return x + y


class AxialAttention2D(nn.Module):
    def __init__(self, dim: int, num_heads: int, along_time: bool, along_freq: bool, dropout: float = 0.0) -> None:
        super().__init__()
        self.along_time = along_time
        self.along_freq = along_freq
        self.time_attn = AxialAttention1D(dim, num_heads, dropout) if along_time else None
        self.freq_attn = AxialAttention1D(dim, num_heads, dropout) if along_freq else None

    def forward(self, x: Tensor) -> Tensor:
        b, c, t, f = x.shape
        if self.along_freq and self.freq_attn is not None:
            y = x.permute(0, 2, 3, 1).reshape(b * t, f, c)
            y = self.freq_attn(y)
            x = y.reshape(b, t, f, c).permute(0, 3, 1, 2).contiguous()
        if self.along_time and self.time_attn is not None:
            y = x.permute(0, 3, 2, 1).reshape(b * f, t, c)
            y = self.time_attn(y)
            x = y.reshape(b, f, t, c).permute(0, 3, 2, 1).contiguous()
        return x


class EncoderStage(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride_t: int, stride_f: int, *, axial_t: bool, axial_f: bool, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.block = ConvNormAct(c_in, c_out, stride_t, stride_f)
        self.axial = AxialAttention2D(c_out, num_heads=num_heads, along_time=axial_t, along_freq=axial_f, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        x = self.axial(x)
        return x


class BandEncoderCore(nn.Module):
    PROFILE_STRIDES: Dict[str, Tuple[Tuple[int, int], ...]] = {
        "band0": ((1, 1), (2, 1), (2, 1), (1, 1), (1, 1)),
        "band1": ((1, 2), (2, 1), (2, 1), (1, 1), (1, 1)),
        "band2": ((1, 2), (2, 2), (2, 1), (1, 1), (1, 1)),
        "band3": ((1, 2), (2, 2), (2, 2), (1, 1), (1, 1)),
        "band4plus": ((1, 2), (2, 2), (2, 2), (1, 2), (1, 1)),
    }

    def __init__(self, cfg: DubSeparatorConfig, profile: str) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = cfg.encoder_channels
        strides = self.PROFILE_STRIDES[profile]
        self.use_gradient_checkpointing = bool(cfg.gradient_checkpointing and cfg.checkpoint_encoder_stages)
        self.stages = nn.ModuleList(
            [
                EncoderStage(c1, c1, *strides[0], axial_t=False, axial_f=True, num_heads=cfg.axial_num_heads, dropout=cfg.axial_dropout),
                EncoderStage(c1, c2, *strides[1], axial_t=False, axial_f=True, num_heads=cfg.axial_num_heads, dropout=cfg.axial_dropout),
                EncoderStage(c2, c3, *strides[2], axial_t=True, axial_f=True, num_heads=cfg.axial_num_heads, dropout=cfg.axial_dropout),
                EncoderStage(c3, c4, *strides[3], axial_t=True, axial_f=True, num_heads=cfg.axial_num_heads, dropout=cfg.axial_dropout),
                EncoderStage(c4, c5, *strides[4], axial_t=True, axial_f=True, num_heads=cfg.axial_num_heads, dropout=cfg.axial_dropout),
            ]
        )

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.use_gradient_checkpointing = bool(enabled)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        stages: List[Tensor] = []
        for stage in self.stages:
            if self.use_gradient_checkpointing and self.training and x.requires_grad:
                x = torch_checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
            stages.append(x)
        return x, stages


class QueryBandTokenizer(nn.Module):
    def __init__(self, dim: int, num_tokens: int, num_heads: int, mlp_mult: float) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.attn = SDPAMultiheadAttention(dim, num_heads, dropout=0.0)
        hidden = int(dim * mlp_mult)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden * 2),
            SwiGLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: Tensor, pos_emb: Tensor, stream_emb: Tensor, band_emb: Tensor) -> Tensor:
        b, c, t, f = x.shape
        mem = x.permute(0, 2, 3, 1).reshape(b * t, f, c)
        mem = mem + pos_emb[:f].to(device=x.device, dtype=x.dtype).unsqueeze(0)
        mem = mem + stream_emb.view(1, 1, c) + band_emb.view(1, 1, c)
        mem = self.norm(mem)
        q = self.query.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(b * t, -1, -1)
        q = q + stream_emb.view(1, 1, c) + band_emb.view(1, 1, c)
        tok = self.attn(q, mem, mem)
        tok = tok + self.mlp(tok)
        tok = tok.reshape(b, t, self.num_tokens, c).permute(0, 3, 1, 2).contiguous()
        return tok


class TrunkSelfCrossBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_mult: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = SDPAMultiheadAttention(dim, num_heads, dropout=dropout)
        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = SDPAMultiheadAttention(dim, num_heads, dropout=dropout)
        self.cross_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Sigmoid())
        hidden = int(dim * ff_mult)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden * 2),
            SwiGLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, mix: Tensor, ref: Tensor) -> Tensor:
        sm = self.self_norm(mix)
        mix = mix + self.self_attn(sm, sm, sm)
        q = self.cross_norm_q(mix)
        kv = self.cross_norm_kv(ref)
        c = self.cross_attn(q, kv, kv)
        g = self.cross_gate(mix)
        mix = mix + g * c
        mix = mix + self.ffn(mix)
        return mix


class PerBandPreTrunkBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_mult: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = TrunkSelfCrossBlock(dim=dim, num_heads=num_heads, ff_mult=ff_mult, dropout=dropout)

    def forward(self, mix: Tensor, ref: Tensor) -> Tensor:
        b, c, t, k = mix.shape
        mix_seq = mix.permute(0, 2, 3, 1).reshape(b * t, k, c)
        ref_seq = ref.permute(0, 2, 3, 1).reshape(b * t, k, c)
        out = self.block(mix_seq, ref_seq)
        out = out.reshape(b, t, k, c).permute(0, 3, 1, 2).contiguous()
        return out


class StageSkipTokenizer(nn.Module):
    def __init__(self, dim: int, num_tokens: int, num_heads: int, mlp_mult: float) -> None:
        super().__init__()
        self.tokenizer = QueryBandTokenizer(dim=dim, num_tokens=num_tokens, num_heads=num_heads, mlp_mult=mlp_mult)

    def forward(self, x: Tensor, pos_emb: Tensor, band_emb: Tensor) -> Tensor:
        stream_zero = torch.zeros_like(band_emb)
        return self.tokenizer(x, pos_emb=pos_emb, stream_emb=stream_zero, band_emb=band_emb)


class DecoderBlock(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int, *, upsample_time: bool, axial_time: bool, axial_freq: bool, num_heads: int) -> None:
        super().__init__()
        self.upsample_time = upsample_time
        self.fuse = nn.Conv2d(c_in + c_skip, c_out, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        )
        self.axial = AxialAttention2D(c_out, num_heads=num_heads, along_time=axial_time, along_freq=axial_freq, dropout=0.0)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        if self.upsample_time:
            x = F.interpolate(x, size=(skip.size(2), x.size(3)), mode="bilinear", align_corners=False)
        elif x.size(2) != skip.size(2):
            x = F.interpolate(x, size=(skip.size(2), x.size(3)), mode="bilinear", align_corners=False)
        if x.size(3) != skip.size(3):
            raise ValueError(f"Pseudo-frequency mismatch: {x.shape} vs {skip.shape}")
        x = self.fuse(torch.cat([x, skip], dim=1))
        x = x + self.conv(x)
        x = self.axial(x)
        return x


class ReversePerceiverDetokenizer(nn.Module):
    def __init__(self, dim: int, out_bins: int, num_heads: int) -> None:
        super().__init__()
        self.out_bins = out_bins
        self.query = nn.Parameter(torch.randn(out_bins, dim) * 0.02)
        self.freq_emb = nn.Embedding(out_bins, dim)
        self.norm = nn.LayerNorm(dim)
        self.attn = SDPAMultiheadAttention(dim, num_heads, dropout=0.0)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, t, k = x.shape
        mem = x.permute(0, 2, 3, 1).reshape(b * t, k, c)
        mem = self.norm(mem)
        q = self.query.to(device=x.device, dtype=x.dtype) + self.freq_emb.weight.to(device=x.device, dtype=x.dtype)
        q = q.unsqueeze(0).expand(b * t, -1, -1)
        y = self.attn(q, mem, mem)
        y = y + self.mlp(y)
        y = y.reshape(b, t, self.out_bins, c).permute(0, 3, 1, 2).contiguous()
        return y


class DubSeparator(nn.Module):
    def __init__(self, cfg: DubSeparatorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DubSeparatorConfig()
        c1, c2, c3, c4, c5 = self.cfg.encoder_channels
        self.frontend = STFTFrontend(self.cfg)
        self.features = StereoPairFeatureExtractor(self.cfg)
        self.gradient_checkpointing = bool(self.cfg.gradient_checkpointing)

        self.mix_input_stems = nn.ModuleList([BranchInputStem(11, c1) for _ in self.cfg.bands])
        self.ref_input_stems = nn.ModuleList([BranchInputStem(11, c1) for _ in self.cfg.bands])
        self.band_cores = nn.ModuleList([BandEncoderCore(self.cfg, band.encoder_profile) for band in self.cfg.bands])

        self.band_pos_embs = nn.ModuleList([LearnablePositionalBias(max_len=self.cfg.onesided_bins, dim=self.cfg.trunk_dim) for _ in self.cfg.bands])
        self.band_tokenizers = nn.ModuleList([
            QueryBandTokenizer(dim=self.cfg.trunk_dim, num_tokens=band.num_tokens, num_heads=self.cfg.band_tokenizer_heads, mlp_mult=self.cfg.token_mlp_mult)
            for band in self.cfg.bands
        ])
        self.per_band_pretrunk = PerBandPreTrunkBlock(
            dim=self.cfg.trunk_dim,
            num_heads=self.cfg.pretrunk_num_heads,
            ff_mult=self.cfg.pretrunk_ff_mult,
            dropout=self.cfg.attn_dropout,
        )

        self.mix_stream_emb = nn.Parameter(torch.randn(self.cfg.trunk_dim) * 0.02)
        self.ref_stream_emb = nn.Parameter(torch.randn(self.cfg.trunk_dim) * 0.02)
        self.band_id_emb = nn.Embedding(self.cfg.num_bands, self.cfg.trunk_dim)

        band_ids: List[int] = []
        slot_ids: List[int] = []
        slot_cursor = 0
        for bidx, band in enumerate(self.cfg.bands):
            band_ids.extend([bidx] * band.num_tokens)
            slot_ids.extend(list(range(slot_cursor, slot_cursor + band.num_tokens)))
            slot_cursor += band.num_tokens
        self.register_buffer("trunk_band_ids", torch.tensor(band_ids, dtype=torch.long), persistent=False)
        self.register_buffer("trunk_slot_ids", torch.tensor(slot_ids, dtype=torch.long), persistent=False)
        self.trunk_pos = FactorizedTokenPosition(self.cfg.total_tokens, self.cfg.num_bands, self.cfg.trunk_dim)

        self.trunk_blocks = nn.ModuleList([
            TrunkSelfCrossBlock(self.cfg.trunk_dim, self.cfg.trunk_num_heads, self.cfg.trunk_ff_mult, self.cfg.attn_dropout)
            for _ in range(self.cfg.num_trunk_layers)
        ])

        stage_channels = [c1, c2, c3, c4, c5]
        self.stage_pos_embs = nn.ModuleList([
            nn.ModuleList([LearnablePositionalBias(max_len=self.cfg.onesided_bins, dim=ch) for _ in self.cfg.bands])
            for ch in stage_channels
        ])
        self.stage_band_embs = nn.ModuleList([nn.Embedding(self.cfg.num_bands, ch) for ch in stage_channels])
        self.stage_skip_tokenizers = nn.ModuleList([
            nn.ModuleList([
                StageSkipTokenizer(dim=ch, num_tokens=band.num_tokens, num_heads=self.cfg.stage_tokenizer_heads, mlp_mult=self.cfg.token_mlp_mult)
                for band in self.cfg.bands
            ])
            for ch in stage_channels
        ])

        self.dec5 = DecoderBlock(c_in=c5, c_skip=c5, c_out=c5, upsample_time=False, axial_time=True, axial_freq=True, num_heads=self.cfg.axial_num_heads)
        self.dec4 = DecoderBlock(c_in=c5, c_skip=c4, c_out=c3, upsample_time=False, axial_time=True, axial_freq=True, num_heads=self.cfg.axial_num_heads)
        self.dec3 = DecoderBlock(c_in=c3, c_skip=c3, c_out=c2, upsample_time=True, axial_time=False, axial_freq=True, num_heads=self.cfg.axial_num_heads)
        self.dec2 = DecoderBlock(c_in=c2, c_skip=c2, c_out=c1, upsample_time=True, axial_time=False, axial_freq=True, num_heads=self.cfg.axial_num_heads)
        self.dec1 = DecoderBlock(c_in=c1, c_skip=c1, c_out=self.cfg.head_dim, upsample_time=False, axial_time=False, axial_freq=True, num_heads=self.cfg.axial_num_heads)

        self.detokenizer = ReversePerceiverDetokenizer(self.cfg.head_dim, self.cfg.onesided_bins, self.cfg.detok_num_heads)
        self.refine = RefineBlock(self.cfg.head_dim, self.cfg.refine_kernel_t, self.cfg.refine_kernel_f)

        self.head_crm = nn.Conv2d(self.cfg.head_dim, 4, kernel_size=1)
        self.head_mp = nn.Conv2d(self.cfg.head_dim, 4, kernel_size=1)
        self.head_gate = nn.Conv2d(self.cfg.head_dim, 1, kernel_size=1)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)
        for core in self.band_cores:
            core.set_gradient_checkpointing(bool(enabled and self.cfg.checkpoint_encoder_stages))

    def _maybe_checkpoint(self, module: nn.Module, *args: Tensor, enabled: bool = True) -> Tensor:
        if enabled and self.gradient_checkpointing and self.training and all(a.requires_grad for a in args):
            return torch_checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def _encode_pair_branches(self, mix_feats: List[Tensor], ref_feats: List[Tensor]) -> Tuple[List[Tensor], List[List[Tensor]], List[Tensor]]:
        mix_encoded: List[Tensor] = []
        mix_stage_feats: List[List[Tensor]] = []
        ref_encoded: List[Tensor] = []
        for bidx, (mix_feat, ref_feat) in enumerate(zip(mix_feats, ref_feats)):
            mix_x = self.mix_input_stems[bidx](mix_feat)
            ref_x = self.ref_input_stems[bidx](ref_feat)
            pair_x = torch.cat([mix_x, ref_x], dim=0)
            pair_out, pair_stages = self.band_cores[bidx](pair_x)
            bsz = mix_x.size(0)
            mix_encoded.append(pair_out[:bsz])
            ref_encoded.append(pair_out[bsz:])
            mix_stage_feats.append([stage[:bsz] for stage in pair_stages])
        return mix_encoded, mix_stage_feats, ref_encoded

    def _tokenize_pair_branches(self, mix_encoded_bands: List[Tensor], ref_encoded_bands: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        mix_toks: List[Tensor] = []
        ref_toks: List[Tensor] = []
        mix_stream = self.mix_stream_emb
        ref_stream = self.ref_stream_emb
        for bidx, (mix_x, ref_x) in enumerate(zip(mix_encoded_bands, ref_encoded_bands)):
            pos = self.band_pos_embs[bidx](mix_x.size(-1), mix_x.device)
            band_emb = self.band_id_emb.weight[bidx].to(device=mix_x.device, dtype=mix_x.dtype)
            mix_toks.append(self.band_tokenizers[bidx](mix_x, pos_emb=pos, stream_emb=mix_stream.to(device=mix_x.device, dtype=mix_x.dtype), band_emb=band_emb))
            ref_toks.append(self.band_tokenizers[bidx](ref_x, pos_emb=pos, stream_emb=ref_stream.to(device=ref_x.device, dtype=ref_x.dtype), band_emb=band_emb))
        return mix_toks, ref_toks

    def _run_per_band_pretrunk(self, mix_bands: List[Tensor], ref_bands: List[Tensor]) -> Tuple[List[Tensor], Tensor, Tensor]:
        groups: Dict[int, List[int]] = {}
        for idx, tok in enumerate(mix_bands):
            groups.setdefault(int(tok.size(-1)), []).append(idx)
        mix_out: List[Tensor] = [torch.empty_like(x) for x in mix_bands]
        for _k, indices in groups.items():
            mix_stack = torch.stack([mix_bands[i] for i in indices], dim=1)  # [B,G,C,T,K]
            ref_stack = torch.stack([ref_bands[i] for i in indices], dim=1)
            b, g, c, t, k = mix_stack.shape
            mix_batch = mix_stack.reshape(b * g, c, t, k)
            ref_batch = ref_stack.reshape(b * g, c, t, k)
            mix_batch = self._maybe_checkpoint(self.per_band_pretrunk, mix_batch, ref_batch, enabled=self.cfg.checkpoint_pretrunk)
            mix_stack_out = mix_batch.reshape(b, g, c, t, k)
            for gi, band_idx in enumerate(indices):
                mix_out[band_idx] = mix_stack_out[:, gi].contiguous()
        return mix_out, torch.cat(mix_out, dim=-1), torch.cat(ref_bands, dim=-1)

    def _add_trunk_positions(self, x: Tensor) -> Tensor:
        b, c, t, k = x.shape
        tok_pos, time_pos = self.trunk_pos(band_ids=self.trunk_band_ids, slot_ids=self.trunk_slot_ids, time_steps=t, device=x.device)
        x = x + tok_pos.to(dtype=x.dtype).T.unsqueeze(0).unsqueeze(2)
        x = x + time_pos.to(dtype=x.dtype).T.unsqueeze(0).unsqueeze(-1)
        return x

    def _flatten_tokens(self, x: Tensor) -> Tensor:
        b, c, t, k = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, t * k, c)

    def _unflatten_tokens(self, x: Tensor, t: int, k: int) -> Tensor:
        b, _, c = x.shape
        return x.reshape(b, t, k, c).permute(0, 3, 1, 2).contiguous()

    def _build_stage_skip(self, stage_idx: int, mix_stage_bands: List[List[Tensor]]) -> Tensor:
        pieces: List[Tensor] = []
        for bidx in range(self.cfg.num_bands):
            x = mix_stage_bands[bidx][stage_idx]
            pos = self.stage_pos_embs[stage_idx][bidx](x.size(-1), x.device)
            band_emb = self.stage_band_embs[stage_idx].weight[bidx].to(device=x.device, dtype=x.dtype)
            q = self.stage_skip_tokenizers[stage_idx][bidx](x, pos_emb=pos, band_emb=band_emb)
            pieces.append(q)
        return torch.cat(pieces, dim=-1)

    @staticmethod
    def _scale_crm(raw: Tensor, scale: float) -> Tensor:
        return scale * torch.tanh(raw / max(scale, 1e-6))

    def _build_masks(self, x: Tensor) -> Dict[str, Tensor]:
        crm_raw = self.head_crm(x).float()
        mp_raw = self.head_mp(x).float()
        gate_logits = self.head_gate(x).float()
        gate = torch.sigmoid(gate_logits)

        crm_raw = crm_raw.permute(0, 2, 3, 1).contiguous()
        mp_raw = mp_raw.permute(0, 2, 3, 1).contiguous()
        gate = gate.permute(0, 2, 3, 1).contiguous()
        crm = self._scale_crm(crm_raw, self.cfg.crm_scale)
        crm_complex = torch.complex(torch.stack([crm[..., 0], crm[..., 2]], dim=1), torch.stack([crm[..., 1], crm[..., 3]], dim=1))
        mag = torch.sigmoid(torch.stack([mp_raw[..., 0], mp_raw[..., 2]], dim=1))
        dphi = torch.tanh(torch.stack([mp_raw[..., 1], mp_raw[..., 3]], dim=1)) * math.pi
        mp_complex = mag * torch.complex(torch.cos(dphi), torch.sin(dphi))
        gate_btfs = gate.permute(0, 3, 1, 2).contiguous()
        final_mask = gate_btfs * crm_complex + (1.0 - gate_btfs) * mp_complex
        return {
            "crm_raw": crm_raw,
            "mp_raw": mp_raw,
            "gate_logits": gate_logits,
            "gate": gate_btfs,
            "crm_mask": crm_complex,
            "mp_mask": mp_complex,
            "mask": final_mask,
        }

    def forward(self, mix_waveform: Tensor, ref_waveform: Tensor) -> Dict[str, Tensor | List[Tensor]]:
        if mix_waveform.shape != ref_waveform.shape:
            raise ValueError(f"mix/ref shape mismatch: {tuple(mix_waveform.shape)} vs {tuple(ref_waveform.shape)}")
        length = mix_waveform.size(-1)
        mix_spec = self.frontend.stft(mix_waveform)
        ref_spec = self.frontend.stft(ref_waveform)
        mix_feats, ref_feats, _ = self.features(mix_spec, ref_spec)
        mix_encoded, mix_stages, ref_encoded = self._encode_pair_branches(mix_feats, ref_feats)
        mix_band_tokens, ref_band_tokens = self._tokenize_pair_branches(mix_encoded, ref_encoded)
        pretrunk_mix_bands, z_mix, z_ref = self._run_per_band_pretrunk(mix_band_tokens, ref_band_tokens)
        z_mix = self._add_trunk_positions(z_mix)
        z_ref = self._add_trunk_positions(z_ref)
        b, c, t4, k = z_mix.shape
        mix_seq = self._flatten_tokens(z_mix)
        ref_seq = self._flatten_tokens(z_ref)
        for block in self.trunk_blocks:
            mix_seq = self._maybe_checkpoint(block, mix_seq, ref_seq, enabled=self.cfg.checkpoint_trunk)
        x = self._unflatten_tokens(mix_seq, t4, k)
        skip1 = self._build_stage_skip(0, mix_stages)
        skip2 = self._build_stage_skip(1, mix_stages)
        skip3 = self._build_stage_skip(2, mix_stages)
        skip4 = self._build_stage_skip(3, mix_stages)
        skip5 = self._build_stage_skip(4, mix_stages)
        x = self._maybe_checkpoint(self.dec5, x, skip5, enabled=self.cfg.checkpoint_decoder)
        x = self._maybe_checkpoint(self.dec4, x, skip4, enabled=self.cfg.checkpoint_decoder)
        x = self._maybe_checkpoint(self.dec3, x, skip3, enabled=self.cfg.checkpoint_decoder)
        x = self._maybe_checkpoint(self.dec2, x, skip2, enabled=self.cfg.checkpoint_decoder)
        x = self._maybe_checkpoint(self.dec1, x, skip1, enabled=self.cfg.checkpoint_decoder)
        x_tf = self._maybe_checkpoint(self.detokenizer, x, enabled=self.cfg.checkpoint_detokenizer)
        x_tf = self._maybe_checkpoint(self.refine, x_tf, enabled=self.cfg.checkpoint_detokenizer)
        mask_dict = self._build_masks(x_tf)
        estimated_spec = mix_spec * mask_dict["mask"]
        estimated_wave = self.frontend.istft(estimated_spec, length=length)
        return {
            "estimate_waveform": estimated_wave,
            "estimate_stft": estimated_spec,
            "mix_stft": mix_spec,
            "ref_stft": ref_spec,
            "mask": mask_dict["mask"],
            "crm_mask": mask_dict["crm_mask"],
            "mp_mask": mask_dict["mp_mask"],
            "crm_gate": mask_dict["gate"],
            "crm_gate_logits": mask_dict["gate_logits"],
            "z_mix": z_mix,
            "z_ref": z_ref,
            "pretrunk_mix_bands": pretrunk_mix_bands,
            "decoder_tokens": x,
            "decoder_tf": x_tf,
        }


__all__ = ["BandSpec", "DubSeparatorConfig", "DubSeparator"]
