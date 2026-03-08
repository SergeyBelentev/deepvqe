from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Tuple

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ============================================================
# Configs
# ============================================================


@dataclass(slots=True)
class BandSpec:
    name: str
    f_min_hz: float
    f_max_hz: float
    num_tokens: int
    overlap_ratio: float = 0.20
    encoder_profile: Literal["band0", "band1", "band2", "band3", "band4plus"] = "band4plus"


@dataclass(slots=True)
class DubSeparatorConfig:
    sample_rate: int = 48_000
    n_fft: int = 2048
    hop_length: int = 1024
    win_length: int = 2048
    center: bool = True
    normalized: bool = False
    window_fn: str = "hann"
    eps: float = 1e-8

    encoder_channels: Tuple[int, int, int, int, int] = (32, 96, 160, 192, 192)
    trunk_dim: int = 192
    head_dim: int = 96

    num_trunk_layers: int = 6
    trunk_num_heads: int = 8
    trunk_ff_mult: float = 4.0
    axial_num_heads: int = 4
    axial_dropout: float = 0.0
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

    detok_num_heads: int = 8
    stage_tokenizer_heads: int = 4
    band_tokenizer_heads: int = 4
    token_mlp_mult: float = 2.0
    pretrunk_num_heads: int = 4
    pretrunk_ff_mult: float = 2.0

    crm_scale: float = 3.0
    refine_kernel_t: int = 5
    refine_kernel_f: int = 3

    # 10 single-resolution overlapped bands.
    bands: Tuple[BandSpec, ...] = (
        BandSpec("band0", 0.0, 500.0, 6, encoder_profile="band0"),
        BandSpec("band1", 500.0, 1250.0, 8, encoder_profile="band1"),
        BandSpec("band2", 1250.0, 2500.0, 8, encoder_profile="band2"),
        BandSpec("band3", 2500.0, 5000.0, 10, encoder_profile="band3"),
        BandSpec("band4", 5000.0, 8000.0, 12, encoder_profile="band4plus"),
        BandSpec("band5", 8000.0, 12000.0, 12, encoder_profile="band4plus"),
        BandSpec("band6", 12000.0, 16000.0, 12, encoder_profile="band4plus"),
        BandSpec("band7", 16000.0, 20000.0, 12, encoder_profile="band4plus"),
        BandSpec("band8", 20000.0, 22000.0, 10, encoder_profile="band4plus"),
        BandSpec("band9", 22000.0, 24000.0, 10, encoder_profile="band4plus"),
    )

    @property
    def num_bands(self) -> int:
        return len(self.bands)

    @property
    def total_tokens(self) -> int:
        return sum(b.num_tokens for b in self.bands)

    @property
    def onesided_bins(self) -> int:
        return self.n_fft // 2 + 1


# ============================================================
# Utility ops
# ============================================================


def _make_window(name: str, win_length: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if name == "hann":
        return torch.hann_window(win_length, periodic=True, device=device, dtype=dtype)
    raise ValueError(f"Unsupported window_fn={name!r}")


class STFTFrontend(nn.Module):
    def __init__(self, cfg: DubSeparatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        window = _make_window(cfg.window_fn, cfg.win_length, device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer("window", window, persistent=False)

    def stft(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform: [B, 2, S]
        Returns:
            complex STFT: [B, 2, T, F]
        """
        if waveform.ndim != 3 or waveform.size(1) != 2:
            raise ValueError(f"Expected waveform [B,2,S], got {tuple(waveform.shape)}")

        b, c, s = waveform.shape
        x = waveform.reshape(b * c, s)
        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        spec = torch.stft(
            x,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=window,
            center=self.cfg.center,
            normalized=self.cfg.normalized,
            onesided=True,
            return_complex=True,
        )
        spec = spec.reshape(b, c, spec.size(-2), spec.size(-1))
        spec = spec.permute(0, 1, 3, 2).contiguous()  # [B,2,T,F]
        return spec

    def istft(self, spec: Tensor, *, length: int | None = None) -> Tensor:
        """
        Args:
            spec: complex [B, 2, T, F]
        Returns:
            waveform [B, 2, S]
        """
        if spec.ndim != 4 or spec.size(1) != 2:
            raise ValueError(f"Expected spec [B,2,T,F], got {tuple(spec.shape)}")

        b, c, t, f = spec.shape
        x = spec.permute(0, 1, 3, 2).contiguous().reshape(b * c, f, t)
        window = self.window.to(device=spec.device, dtype=spec.real.dtype)
        wav = torch.istft(
            x,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=window,
            center=self.cfg.center,
            normalized=self.cfg.normalized,
            onesided=True,
            length=length,
        )
        wav = wav.reshape(b, c, wav.size(-1))
        return wav


class BandLayout(nn.Module):
    def __init__(self, cfg: DubSeparatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        freq_hz = torch.linspace(0.0, cfg.sample_rate / 2.0, cfg.onesided_bins)
        self.register_buffer("freq_hz", freq_hz, persistent=False)

        band_meta: List[Dict[str, Any]] = []
        for band in cfg.bands:
            weights = self._build_band_weights(freq_hz, band)
            nz = torch.nonzero(weights > 0, as_tuple=False).flatten()
            if nz.numel() == 0:
                raise ValueError(f"Band {band.name} produced empty support")
            start = int(nz[0].item())
            stop = int(nz[-1].item()) + 1
            crop = weights[start:stop]
            band_meta.append(
                {
                    "name": band.name,
                    "start": start,
                    "stop": stop,
                    "weights": crop,
                    "num_tokens": band.num_tokens,
                    "f_min_hz": band.f_min_hz,
                    "f_max_hz": band.f_max_hz,
                    "profile": band.encoder_profile,
                }
            )
        self.band_meta = band_meta

    @staticmethod
    def _build_band_weights(freq_hz: Tensor, band: BandSpec) -> Tensor:
        f0 = band.f_min_hz
        f1 = band.f_max_hz
        width = max(f1 - f0, 1.0)
        ext = 0.5 * band.overlap_ratio * width  # +/- 10% => total 20% overlap.
        sup0 = max(0.0, f0 - ext)
        sup1 = min(float(freq_hz[-1].item()), f1 + ext)

        w = torch.zeros_like(freq_hz)
        core = (freq_hz >= f0) & (freq_hz <= f1)
        w = torch.where(core, torch.ones_like(w), w)

        if sup0 < f0:
            left = (freq_hz >= sup0) & (freq_hz < f0)
            if left.any():
                alpha = (freq_hz[left] - sup0) / max(f0 - sup0, 1e-6)
                w[left] = 0.5 - 0.5 * torch.cos(math.pi * alpha)

        if f1 < sup1:
            right = (freq_hz > f1) & (freq_hz <= sup1)
            if right.any():
                alpha = 1.0 - (freq_hz[right] - f1) / max(sup1 - f1, 1e-6)
                w[right] = 0.5 - 0.5 * torch.cos(math.pi * alpha)

        return w.clamp(0.0, 1.0)

    def slice_band(self, spec: Tensor, band_idx: int) -> Tuple[Tensor, Tensor]:
        meta = self.band_meta[band_idx]
        start, stop = meta["start"], meta["stop"]
        w = meta["weights"].to(device=spec.device, dtype=spec.real.dtype)
        band_spec = spec[..., start:stop] * w.view(1, 1, 1, -1)
        return band_spec, w


# ============================================================
# Feature extraction
# ============================================================


class StereoPairFeatureExtractor(nn.Module):
    """
    Builds 11-channel features for a band from mix/ref complex STFTs.
    Output for each branch has identical channel layout; only stereo spatial
    channels are branch-specific, while pairwise log channels are shared.
    """

    def __init__(self, cfg: DubSeparatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.band_layout = BandLayout(cfg)

    def _spatial_features(self, spec: Tensor) -> Tensor:
        """
        spec: complex [B,2,T,Fb]
        returns [B,7,T,Fb]
        """
        l = spec[:, 0]
        r = spec[:, 1]

        re_l = l.real
        im_l = l.imag
        re_r = r.real
        im_r = r.imag

        mag_l = l.abs().clamp_min(self.cfg.eps)
        mag_r = r.abs().clamp_min(self.cfg.eps)

        ild = torch.log(mag_l) - torch.log(mag_r)
        ipd = torch.angle(l) - torch.angle(r)
        sin_ipd = torch.sin(ipd)
        cos_ipd = torch.cos(ipd)

        return torch.stack([re_l, im_l, re_r, im_r, ild, sin_ipd, cos_ipd], dim=1)

    def _pairwise_log_features(self, mix_spec: Tensor, ref_spec: Tensor) -> Tensor:
        """
        mix/ref: complex [B,2,T,Fb]
        returns [B,4,T,Fb]
        """
        mix_mag = mix_spec.abs().mean(dim=1).clamp_min(self.cfg.eps)
        ref_mag = ref_spec.abs().mean(dim=1).clamp_min(self.cfg.eps)

        log_mix = torch.log(mix_mag)
        log_ref = torch.log(ref_mag)
        d = log_mix - log_ref
        ad = d.abs()
        return torch.stack([log_mix, log_ref, d, ad], dim=1)

    def forward(self, mix_spec: Tensor, ref_spec: Tensor) -> Tuple[List[Tensor], List[Tensor], List[Dict[str, Any]]]:
        """
        Args:
            mix_spec/ref_spec: complex [B,2,T,F]
        Returns:
            mix_band_features: list[[B,11,T,Fb]]
            ref_band_features: list[[B,11,T,Fb]]
            band_meta list
        """
        mix_bands: List[Tensor] = []
        ref_bands: List[Tensor] = []
        for bidx in range(self.cfg.num_bands):
            mix_band, _ = self.band_layout.slice_band(mix_spec, bidx)
            ref_band, _ = self.band_layout.slice_band(ref_spec, bidx)

            pair = self._pairwise_log_features(mix_band, ref_band)
            mix_sp = self._spatial_features(mix_band)
            ref_sp = self._spatial_features(ref_band)

            mix_bands.append(torch.cat([mix_sp, pair], dim=1))
            ref_bands.append(torch.cat([ref_sp, pair], dim=1))

        return mix_bands, ref_bands, self.band_layout.band_meta


# ============================================================
# Core layers
# ============================================================


class ConvNormAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride_t: int, stride_f: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            c_in,
            c_out,
            kernel_size=3,
            stride=(stride_t, stride_f),
            padding=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out)
        self.act = nn.SiLU()
        self.refine = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, groups=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.refine(x)
        return x


class LearnablePositionalBias(nn.Module):
    def __init__(self, max_len: int, dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(max_len, dim)

    def forward(self, length: int, device: torch.device) -> Tensor:
        idx = torch.arange(length, device=device)
        return self.emb(idx)


class AxialAttention1D(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: [N, L, C]
        y = self.norm(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        return x + y


class AxialAttention2D(nn.Module):
    def __init__(self, dim: int, num_heads: int, along_time: bool, along_freq: bool, dropout: float = 0.0) -> None:
        super().__init__()
        self.along_time = along_time
        self.along_freq = along_freq
        self.time_attn = AxialAttention1D(dim, num_heads, dropout) if along_time else None
        self.freq_attn = AxialAttention1D(dim, num_heads, dropout) if along_freq else None

    def forward(self, x: Tensor) -> Tensor:
        # x: [B,C,T,F]
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


class BranchInputStem(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=c_in)
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.norm(x))


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
        self.stages = nn.ModuleList(
            [
                EncoderStage(c1, c1, *strides[0], axial_t=False, axial_f=True, num_heads=cfg.axial_num_heads, dropout=cfg.axial_dropout),
                EncoderStage(c1, c2, *strides[1], axial_t=False, axial_f=True, num_heads=cfg.axial_num_heads, dropout=cfg.axial_dropout),
                EncoderStage(c2, c3, *strides[2], axial_t=True, axial_f=True, num_heads=cfg.axial_num_heads, dropout=cfg.axial_dropout),
                EncoderStage(c3, c4, *strides[3], axial_t=True, axial_f=True, num_heads=cfg.axial_num_heads, dropout=cfg.axial_dropout),
                EncoderStage(c4, c5, *strides[4], axial_t=True, axial_f=True, num_heads=cfg.axial_num_heads, dropout=cfg.axial_dropout),
            ]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        stages: List[Tensor] = []
        for stage in self.stages:
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
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        hidden = int(dim * mlp_mult)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden * 2),
            SwiGLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: Tensor, pos_emb: Tensor, stream_emb: Tensor, band_emb: Tensor) -> Tensor:
        # x: [B,C,T,F]
        b, c, t, f = x.shape
        mem = x.permute(0, 2, 3, 1).reshape(b * t, f, c)
        mem = mem + pos_emb[:f].to(device=x.device, dtype=x.dtype).unsqueeze(0)
        mem = mem + stream_emb.view(1, 1, c) + band_emb.view(1, 1, c)
        mem = self.norm(mem)

        q = self.query.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(b * t, -1, -1)
        q = q + stream_emb.view(1, 1, c) + band_emb.view(1, 1, c)
        tok, _ = self.attn(q, mem, mem, need_weights=False)
        tok = tok + self.mlp(tok)
        tok = tok.reshape(b, t, self.num_tokens, c).permute(0, 3, 1, 2).contiguous()
        return tok


class SwiGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b


class FactorizedTokenPosition(nn.Module):
    def __init__(self, total_tokens: int, num_bands: int, dim: int) -> None:
        super().__init__()
        self.band_emb = nn.Embedding(num_bands, dim)
        self.slot_emb = nn.Embedding(total_tokens, dim)
        self.time_emb = nn.Embedding(4096, dim)

    def forward(self, *, band_ids: Tensor, slot_ids: Tensor, time_steps: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        token_pos = self.band_emb(band_ids.to(device)) + self.slot_emb(slot_ids.to(device))
        time_pos = self.time_emb(torch.arange(time_steps, device=device))
        return token_pos, time_pos


class TrunkSelfCrossBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_mult: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
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
        y, _ = self.self_attn(sm, sm, sm, need_weights=False)
        mix = mix + y

        q = self.cross_norm_q(mix)
        kv = self.cross_norm_kv(ref)
        c, _ = self.cross_attn(q, kv, kv, need_weights=False)
        g = self.cross_gate(mix)
        mix = mix + g * c

        mix = mix + self.ffn(mix)
        return mix


class PerBandPreTrunkBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_mult: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = TrunkSelfCrossBlock(dim=dim, num_heads=num_heads, ff_mult=ff_mult, dropout=dropout)

    def forward(self, mix: Tensor, ref: Tensor) -> Tensor:
        # mix/ref: [B,C,T,Kb]
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
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B,C,T,K]
        b, c, t, k = x.shape
        mem = x.permute(0, 2, 3, 1).reshape(b * t, k, c)
        mem = self.norm(mem)

        q = self.query.to(device=x.device, dtype=x.dtype)
        q = q + self.freq_emb.weight.to(device=x.device, dtype=x.dtype)
        q = q.unsqueeze(0).expand(b * t, -1, -1)
        y, _ = self.attn(q, mem, mem, need_weights=False)
        y = y + self.mlp(y)
        y = y.reshape(b, t, self.out_bins, c).permute(0, 3, 1, 2).contiguous()
        return y


class SqueezeExcite2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class RefineBlock(nn.Module):
    def __init__(self, channels: int, k_t: int, k_f: int) -> None:
        super().__init__()
        pad_t = k_t // 2
        pad_f = k_f // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size=(k_t, k_f), padding=(pad_t, pad_f), groups=channels)
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.ff = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            ChannelSwiGLU2d(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.se = SqueezeExcite2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        y = self.dw(x)
        y = self.norm(y)
        y = F.silu(y)
        x = x + y
        x = x + self.ff(x)
        x = self.se(x)
        return x


class ChannelSwiGLU2d(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        a, b = x.chunk(2, dim=1)
        return F.silu(a) * b


# ============================================================
# Full model
# ============================================================


class DubSeparator(nn.Module):
    def __init__(self, cfg: DubSeparatorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DubSeparatorConfig()
        c1, c2, c3, c4, c5 = self.cfg.encoder_channels
        self.frontend = STFTFrontend(self.cfg)
        self.features = StereoPairFeatureExtractor(self.cfg)

        # branch-specific input stems, shared band cores.
        self.mix_input_stems = nn.ModuleList([BranchInputStem(11, c1) for _ in self.cfg.bands])
        self.ref_input_stems = nn.ModuleList([BranchInputStem(11, c1) for _ in self.cfg.bands])
        self.band_cores = nn.ModuleList([
            BandEncoderCore(self.cfg, band.encoder_profile) for band in self.cfg.bands
        ])

        # shared band tokenizers between mix/ref, one per band.
        self.band_pos_embs = nn.ModuleList([
            LearnablePositionalBias(max_len=self.cfg.onesided_bins, dim=self.cfg.trunk_dim) for _ in self.cfg.bands
        ])
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

        # mix/ref stream identity + band identity in token space.
        self.mix_stream_emb = nn.Parameter(torch.randn(self.cfg.trunk_dim) * 0.02)
        self.ref_stream_emb = nn.Parameter(torch.randn(self.cfg.trunk_dim) * 0.02)
        self.band_id_emb = nn.Embedding(self.cfg.num_bands, self.cfg.trunk_dim)

        # Trunk positional embeddings.
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
            TrunkSelfCrossBlock(
                dim=self.cfg.trunk_dim,
                num_heads=self.cfg.trunk_num_heads,
                ff_mult=self.cfg.trunk_ff_mult,
                dropout=self.cfg.attn_dropout,
            )
            for _ in range(self.cfg.num_trunk_layers)
        ])

        # Skip tokenizers from mix encoder stages only.
        stage_channels = [c1, c2, c3, c4, c5]
        self.stage_pos_embs = nn.ModuleList([
            nn.ModuleList([
                LearnablePositionalBias(max_len=self.cfg.onesided_bins, dim=ch) for _ in self.cfg.bands
            ])
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

        # Decoder chain, one target only.
        self.dec5 = DecoderBlock(c_in=c5, c_skip=c5, c_out=c5, upsample_time=False, axial_time=True, axial_freq=True, num_heads=self.cfg.axial_num_heads)
        self.dec4 = DecoderBlock(c_in=c5, c_skip=c4, c_out=c3, upsample_time=False, axial_time=True, axial_freq=True, num_heads=self.cfg.axial_num_heads)
        self.dec3 = DecoderBlock(c_in=c3, c_skip=c3, c_out=c2, upsample_time=True, axial_time=False, axial_freq=True, num_heads=self.cfg.axial_num_heads)
        self.dec2 = DecoderBlock(c_in=c2, c_skip=c2, c_out=c1, upsample_time=True, axial_time=False, axial_freq=True, num_heads=self.cfg.axial_num_heads)
        self.dec1 = DecoderBlock(c_in=c1, c_skip=c1, c_out=self.cfg.head_dim, upsample_time=False, axial_time=False, axial_freq=True, num_heads=self.cfg.axial_num_heads)

        self.detokenizer = ReversePerceiverDetokenizer(dim=self.cfg.head_dim, out_bins=self.cfg.onesided_bins, num_heads=self.cfg.detok_num_heads)
        self.refine = RefineBlock(self.cfg.head_dim, self.cfg.refine_kernel_t, self.cfg.refine_kernel_f)

        # Single target micro-heads.
        self.head_crm = nn.Conv2d(self.cfg.head_dim, 4, kernel_size=1)
        self.head_mp = nn.Conv2d(self.cfg.head_dim, 4, kernel_size=1)
        self.head_gate = nn.Conv2d(self.cfg.head_dim, 1, kernel_size=1)

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _encode_branch(self, band_feats: List[Tensor], input_stems: nn.ModuleList) -> Tuple[List[Tensor], List[List[Tensor]]]:
        encoded: List[Tensor] = []
        stage_feats: List[List[Tensor]] = []
        for bidx, feat in enumerate(band_feats):
            x = input_stems[bidx](feat)
            x, stages = self.band_cores[bidx](x)
            encoded.append(x)
            stage_feats.append(stages)
        return encoded, stage_feats

    def _tokenize_branch(self, encoded_bands: List[Tensor], *, stream: Literal["mix", "ref"]) -> Tensor:
        stream_emb = self.mix_stream_emb if stream == "mix" else self.ref_stream_emb
        toks: List[Tensor] = []
        for bidx, x in enumerate(encoded_bands):
            pos = self.band_pos_embs[bidx](x.size(-1), x.device)
            band_emb = self.band_id_emb.weight[bidx].to(device=x.device, dtype=x.dtype)
            tok = self.band_tokenizers[bidx](x, pos_emb=pos, stream_emb=stream_emb.to(device=x.device, dtype=x.dtype), band_emb=band_emb)
            toks.append(tok)
        return torch.cat(toks, dim=-1)

    def _tokenize_branch_bands(self, encoded_bands: List[Tensor], *, stream: Literal["mix", "ref"]) -> List[Tensor]:
        stream_emb = self.mix_stream_emb if stream == "mix" else self.ref_stream_emb
        toks: List[Tensor] = []
        for bidx, x in enumerate(encoded_bands):
            pos = self.band_pos_embs[bidx](x.size(-1), x.device)
            band_emb = self.band_id_emb.weight[bidx].to(device=x.device, dtype=x.dtype)
            tok = self.band_tokenizers[bidx](x, pos_emb=pos, stream_emb=stream_emb.to(device=x.device, dtype=x.dtype), band_emb=band_emb)
            toks.append(tok)
        return toks

    def _run_per_band_pretrunk(self, mix_bands: List[Tensor], ref_bands: List[Tensor]) -> Tuple[List[Tensor], Tensor, Tensor]:
        mix_out: List[Tensor] = []
        ref_out: List[Tensor] = []
        for mix_tok, ref_tok in zip(mix_bands, ref_bands):
            mix_tok = self.per_band_pretrunk(mix_tok, ref_tok)
            mix_out.append(mix_tok)
            ref_out.append(ref_tok)
        return mix_out, torch.cat(mix_out, dim=-1), torch.cat(ref_out, dim=-1)

    def _add_trunk_positions(self, x: Tensor) -> Tensor:
        # x: [B,C,T,K]
        b, c, t, k = x.shape
        tok_pos, time_pos = self.trunk_pos(
            band_ids=self.trunk_band_ids,
            slot_ids=self.trunk_slot_ids,
            time_steps=t,
            device=x.device,
        )
        x = x + tok_pos.to(dtype=x.dtype).T.unsqueeze(0).unsqueeze(2)
        x = x + time_pos.to(dtype=x.dtype).T.unsqueeze(0).unsqueeze(-1)
        return x

    def _flatten_tokens(self, x: Tensor) -> Tensor:
        # [B,C,T,K] -> [B,T*K,C]
        b, c, t, k = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, t * k, c)

    def _unflatten_tokens(self, x: Tensor, t: int, k: int) -> Tensor:
        # [B,T*K,C] -> [B,C,T,K]
        b, s, c = x.shape
        return x.reshape(b, t, k, c).permute(0, 3, 1, 2).contiguous()

    def _build_stage_skip(self, stage_idx: int, mix_stage_bands: List[List[Tensor]]) -> Tensor:
        # mix_stage_bands: list over bands -> list over 5 stages
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
        # x: [B,96,T,F]
        # bf16 autocast-friendly: complex masks are assembled in fp32
        crm_raw = self.head_crm(x).float()
        mp_raw = self.head_mp(x).float()
        gate_logits = self.head_gate(x).float()
        gate = torch.sigmoid(gate_logits)

        crm_raw = crm_raw.permute(0, 2, 3, 1).contiguous()  # [B,T,F,4]
        mp_raw = mp_raw.permute(0, 2, 3, 1).contiguous()
        gate = gate.permute(0, 2, 3, 1).contiguous()  # [B,T,F,1]

        crm = self._scale_crm(crm_raw, self.cfg.crm_scale)

        crm_complex = torch.complex(
            torch.stack([crm[..., 0], crm[..., 2]], dim=1),
            torch.stack([crm[..., 1], crm[..., 3]], dim=1),
        )  # [B,2,T,F]

        mag = torch.sigmoid(torch.stack([mp_raw[..., 0], mp_raw[..., 2]], dim=1))
        dphi = torch.tanh(torch.stack([mp_raw[..., 1], mp_raw[..., 3]], dim=1)) * math.pi
        mp_complex = mag * torch.complex(torch.cos(dphi), torch.sin(dphi))

        gate_btfs = gate.permute(0, 3, 1, 2).contiguous()  # [B,1,T,F]
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

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(self, mix_waveform: Tensor, ref_waveform: Tensor) -> Dict[str, Tensor | List[Tensor]]:
        """
        Args:
            mix_waveform: [B,2,S]
            ref_waveform: [B,2,S]
        Returns:
            dict with estimate waveform, masks and intermediate tensors.
        """
        if mix_waveform.shape != ref_waveform.shape:
            raise ValueError(f"mix/ref shape mismatch: {tuple(mix_waveform.shape)} vs {tuple(ref_waveform.shape)}")

        length = mix_waveform.size(-1)
        mix_spec = self.frontend.stft(mix_waveform)  # [B,2,T,F]
        ref_spec = self.frontend.stft(ref_waveform)  # [B,2,T,F]

        mix_feats, ref_feats, band_meta = self.features(mix_spec, ref_spec)

        mix_encoded, mix_stages = self._encode_branch(mix_feats, self.mix_input_stems)
        ref_encoded, _ = self._encode_branch(ref_feats, self.ref_input_stems)

        mix_band_tokens = self._tokenize_branch_bands(mix_encoded, stream="mix")
        ref_band_tokens = self._tokenize_branch_bands(ref_encoded, stream="ref")
        pretrunk_mix_bands, z_mix, z_ref = self._run_per_band_pretrunk(mix_band_tokens, ref_band_tokens)

        z_mix = self._add_trunk_positions(z_mix)
        z_ref = self._add_trunk_positions(z_ref)

        b, c, t4, k = z_mix.shape
        mix_seq = self._flatten_tokens(z_mix)
        ref_seq = self._flatten_tokens(z_ref)

        for block in self.trunk_blocks:
            mix_seq = block(mix_seq, ref_seq)

        x = self._unflatten_tokens(mix_seq, t4, k)

        skip1 = self._build_stage_skip(0, mix_stages)
        skip2 = self._build_stage_skip(1, mix_stages)
        skip3 = self._build_stage_skip(2, mix_stages)
        skip4 = self._build_stage_skip(3, mix_stages)
        skip5 = self._build_stage_skip(4, mix_stages)

        x = self.dec5(x, skip5)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        x_tf = self.detokenizer(x)
        x_tf = self.refine(x_tf)

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


__all__ = [
    "BandSpec",
    "DubSeparatorConfig",
    "DubSeparator",
]
