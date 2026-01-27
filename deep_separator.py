# stem_separator.py
# PyTorch 2.9.1, CPython 3.11
# Architecture: multi-band encoders -> per-band tokenizers -> 2D trunk MSHA
#              -> per-head decoders + skip tokenizers -> shared detokenizers (2048/4096)
#              -> per-head refine + mask heads (crm/mp + gate) -> shared router (4096<->2048)
#              -> ISTFT(4096)+ISTFT(2048)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utils
# -------------------------

def _complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: complex tensors
    return a * b

def _safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))

def _crm_soft_scale(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # "soft limiter" for complex ratio mask: z / (1 + |z|)
    mag = torch.abs(z)
    return z / (1.0 + mag + eps)

def _make_hann_window(n_fft: int, device: torch.device) -> torch.Tensor:
    return torch.hann_window(n_fft, periodic=True, device=device, dtype=torch.float32)


def _downsample_logits_4096_to_2048(a4096: torch.Tensor) -> torch.Tensor:
    """
    a4096: (B,1,T,2049)
    returns a2048: (B,1,T,1025) using the mapping you described.
    """
    # DC
    dc = a4096[..., :1]  # (B,1,T,1)
    nyq = a4096[..., -1:]  # (B,1,T,1) corresponds to k=2048

    # interior bins: k=1..1023 => avg of (2k,2k+1)
    # indices in 4096 grid: 0..2048
    # build pairs from 2..2047 (inclusive)
    a_mid = a4096[..., 1:2048]  # (B,1,T,2047) -> corresponds to bins 1..2047
    # We need for k=1..1023:
    # a2048(k) = 0.5*(a4096(2k) + a4096(2k+1))
    # bins 2k and 2k+1 lie in [2..2047]
    # Let's index a_mid with offset -1:
    # a4096(bin=j) is a_mid[..., j-1]
    even_bins = a_mid[..., 1::2]   # bins 2,4,6,...,2046 -> length 1023
    odd_bins = a_mid[..., 2::2]    # bins 3,5,7,...,2047 -> length 1022? careful

    # Better: explicitly slice pairs for k=1..1023:
    # bins: 2k => 2..2046 step2 (1023 items)
    # bins: 2k+1 => 3..2047 step2 (1023 items)
    a2k = a4096[..., 2:2047:2]     # (B,1,T,1023)
    a2k1 = a4096[..., 3:2048:2]    # (B,1,T,1023)
    mid = 0.5 * (a2k + a2k1)       # (B,1,T,1023)

    return torch.cat([dc, mid, nyq], dim=-1)  # (B,1,T,1025)


# -------------------------
# Rotary / logRoPE (1D)
# -------------------------

def _build_log_rope_cache(seq_len: int, head_dim: int, device, dtype, base: float = 10000.0):
    """
    Returns sin, cos with shape (seq_len, head_dim//2).
    We use "log positions": p = log1p(i) scaled to [0, seq_len-1] range.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2

    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    if seq_len > 1:
        pos = torch.log1p(pos) / math.log1p(seq_len - 1) * (seq_len - 1)
    # inv_freq shape (half,)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    freqs = torch.einsum("i,j->ij", pos, inv_freq)  # (seq_len, half)
    sin = freqs.sin().to(dtype=dtype)
    cos = freqs.cos().to(dtype=dtype)
    return sin, cos

def _apply_rope(q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q,k: (B, H, L, D)
    sin,cos: (L, D/2)
    """
    B, H, L, D = q.shape
    half = D // 2
    sin = sin.view(1, 1, L, half)
    cos = cos.view(1, 1, L, half)

    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]

    # rotate: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot


# -------------------------
# Attention blocks (SDPA)
# -------------------------

class MHSA(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, use_rope: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        # x: (B, L, C)
        B, L, C = x.shape
        qkv = self.qkv(x)  # (B,L,3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B,H,L,D)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            assert rope_cache is not None, "rope_cache required when use_rope=True"
            sin, cos = rope_cache
            q, k = _apply_rope(q, k, sin, cos)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # (B,H,L,D)

        out = attn.transpose(1, 2).contiguous().view(B, L, C)
        return self.proj(out)


class MHCA(nn.Module):
    """Multi-head Cross-Attention: queries attend to context (keys/values)."""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, use_rope_on_ctx: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.use_rope_on_ctx = use_rope_on_ctx

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        q_in: torch.Tensor,          # (B, Lq, C)
        ctx: torch.Tensor,           # (B, Lk, C)
        rope_cache_ctx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, Lq, C = q_in.shape
        _, Lk, _ = ctx.shape

        q = self.q_proj(q_in).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Lq,D)
        k = self.k_proj(ctx).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)   # (B,H,Lk,D)
        v = self.v_proj(ctx).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)   # (B,H,Lk,D)

        if self.use_rope_on_ctx:
            assert rope_cache_ctx is not None, "rope_cache_ctx required when use_rope_on_ctx=True"
            sin, cos = rope_cache_ctx  # (Lk, D/2)
            # apply RoPE to k (and also q if desired). Here we apply only to k for "ctx positioning".
            # If you want strict RoPE, apply to both q and k with the SAME cache based on positions.
            q, k = _apply_rope(q, k, sin[:Lq], cos[:Lq]) if Lq == Lk else (q, k)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # (B,H,Lq,D)

        out = attn.transpose(1, 2).contiguous().view(B, Lq, C)
        return self.out_proj(out)


class FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * mult
        self.fc1 = nn.Linear(dim, hidden * 2, bias=True)  # SwiGLU
        self.fc2 = nn.Linear(hidden, dim, bias=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        a, b = x.chunk(2, dim=-1)
        x = F.silu(a) * b
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, ffn_dropout: float = 0.0, use_rope: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, num_heads, dropout=attn_dropout, use_rope=use_rope)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mult=4, dropout=ffn_dropout)

    def forward(self, x: torch.Tensor, rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), rope_cache=rope_cache)
        x = x + self.ffn(self.ln2(x))
        return x


# -------------------------
# 2D conv blocks
# -------------------------

class Conv2dBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: Tuple[int, int] = (1, 1), k: int = 3, gn_groups: int = 8):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False)
        g = min(gn_groups, cout)
        self.gn = nn.GroupNorm(g, cout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# -------------------------
# Axial attention blocks (over 4D: B,C,T,F)
# -------------------------

class AxialAttnF(nn.Module):
    """Self-attention along F for each time frame independently."""
    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, ffn_dropout: float = 0.0, use_log_rope: bool = True):
        super().__init__()
        self.block = TransformerBlock(dim, num_heads, attn_dropout, ffn_dropout, use_rope=use_log_rope)
        self.use_log_rope = use_log_rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,F)
        B, C, T, F_ = x.shape
        y = x.permute(0, 2, 3, 1).contiguous().view(B * T, F_, C)  # (B*T, F, C)
        rope_cache = None
        if self.use_log_rope:
            sin, cos = _build_log_rope_cache(F_, self.block.attn.head_dim, device=y.device, dtype=y.dtype)
            rope_cache = (sin, cos)
        y = self.block(y, rope_cache=rope_cache)
        y = y.view(B, T, F_, C).permute(0, 3, 1, 2).contiguous()
        return y


class AxialAttnTF(nn.Module):
    """Self-attention along T then along F."""
    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, ffn_dropout: float = 0.0, use_log_rope_f: bool = True):
        super().__init__()
        # T-attn without RoPE by default; F-attn with logRoPE
        self.block_t = TransformerBlock(dim, num_heads, attn_dropout, ffn_dropout, use_rope=False)
        self.block_f = TransformerBlock(dim, num_heads, attn_dropout, ffn_dropout, use_rope=use_log_rope_f)
        self.use_log_rope_f = use_log_rope_f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,F)
        B, C, T, F_ = x.shape

        # Attend over T for each frequency slot
        y = x.permute(0, 3, 2, 1).contiguous().view(B * F_, T, C)  # (B*F, T, C)
        y = self.block_t(y, rope_cache=None)
        y = y.view(B, F_, T, C).permute(0, 3, 2, 1).contiguous()  # back to (B,C,T,F)

        # Attend over F for each time
        y2 = y.permute(0, 2, 3, 1).contiguous().view(B * T, F_, C)
        rope_cache = None
        if self.use_log_rope_f:
            sin, cos = _build_log_rope_cache(F_, self.block_f.attn.head_dim, device=y2.device, dtype=y2.dtype)
            rope_cache = (sin, cos)
        y2 = self.block_f(y2, rope_cache=rope_cache)
        y2 = y2.view(B, T, F_, C).permute(0, 3, 1, 2).contiguous()
        return y2


# -------------------------
# Perceiver-style tokenizers / detokenizers (per-time)
# -------------------------

class PerTimeTokenizer(nn.Module):
    """
    Tokenize along frequency for each time frame:
      (B,C,T,F) -> (B,C,T,K)
    """
    def __init__(self, dim: int, num_heads: int, K: int, attn_dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.K = K
        self.q = nn.Parameter(torch.randn(K, dim) * 0.02)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_ctx = nn.LayerNorm(dim)
        self.ca = MHCA(dim, num_heads, dropout=attn_dropout, use_rope_on_ctx=False)

    def forward(self, x: torch.Tensor, q_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B,C,T,F)
        B, C, T, F_ = x.shape
        ctx = x.permute(0, 2, 3, 1).contiguous().view(B * T, F_, C)  # (BT, F, C)

        q = self.q
        if q_bias is not None:
            q = q + q_bias  # q_bias: (K,C)
        q = q.unsqueeze(0).expand(B * T, self.K, C)  # (BT,K,C)

        out = self.ca(self.ln_q(q), self.ln_ctx(ctx))  # (BT,K,C)
        out = out.view(B, T, self.K, C).permute(0, 3, 1, 2).contiguous()  # (B,C,T,K)
        return out


class PerTimeDetokenizer(nn.Module):
    """
    Detokenize along frequency for each time frame:
      (B,C,T,K) -> (B,C,T,F_out)
    reverse Perceiver: learned queries over F_out attend to K tokens.
    """
    def __init__(self, dim: int, num_heads: int, F_out: int, attn_dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.F_out = F_out
        self.q = nn.Parameter(torch.randn(F_out, dim) * 0.02)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_ctx = nn.LayerNorm(dim)
        self.ca = MHCA(dim, num_heads, dropout=attn_dropout, use_rope_on_ctx=False)

    def forward(self, x: torch.Tensor, q_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B,C,T,K)
        B, C, T, K = x.shape
        ctx = x.permute(0, 2, 3, 1).contiguous().view(B * T, K, C)  # (BT,K,C)

        q = self.q
        if q_bias is not None:
            q = q + q_bias  # (F_out,C)
        q = q.unsqueeze(0).expand(B * T, self.F_out, C)  # (BT,F,C)

        out = self.ca(self.ln_q(q), self.ln_ctx(ctx))  # (BT,F,C)
        out = out.view(B, T, self.F_out, C).permute(0, 3, 1, 2).contiguous()  # (B,C,T,F_out)
        return out


# -------------------------
# Refine blocks + SE
# -------------------------

class SEBlock2d(nn.Module):
    def __init__(self, channels: int, r: int = 8):
        super().__init__()
        hidden = max(8, channels // r)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,F)
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class RefineBlock(nn.Module):
    def __init__(self, channels: int, k_t: int = 5, k_f: int = 3):
        super().__init__()
        pad_t = k_t // 2
        pad_f = k_f // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size=(k_t, k_f), padding=(pad_t, pad_f), groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.SiLU()

        # SwiGLU (2x channels -> channels)
        self.glu = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)
        self.se = SEBlock2d(channels, r=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.gn(self.pw(self.dw(x))))
        x = x + y

        g = self.glu(x)
        a, b = g.chunk(2, dim=1)
        x = x + (F.silu(a) * b)

        x = self.se(x)
        return x


# -------------------------
# Band specs + config
# -------------------------

@dataclass(frozen=True)
class BandSpec:
    band_id: int
    hz_low: float
    hz_high: float
    n_fft: int  # 2048 or 4096


@dataclass
class SeparatorConfig:
    sample_rate: int = 48_000
    hop: int = 1024
    center: bool = True
    pad_mode: str = "reflect"
    normalized: bool = False

    n_fft_main: int = 4096
    n_fft_sub: int = 2048

    # trunk + attention
    trunk_depth: int = 10
    trunk_heads: int = 8
    axial_heads: int = 8
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0

    # channels
    c_in: int = 7
    c_e1: int = 32
    c_e2: int = 96
    c_e3: int = 160
    c_e4: int = 192
    c_e5: int = 192
    c_trunk: int = 192

    # decoder channels: D5..D1
    c_d5: int = 192
    c_d4: int = 160
    c_d3: int = 96
    c_d2: int = 32
    c_d1: int = 96  # output channels for detokenizers

    # token allocation per band (K_b)
    token_alloc: Dict[int, int] = None

    # heads
    heads: Tuple[str, ...] = ("bass", "drums", "music", "vocals")

    def __post_init__(self):
        if self.token_alloc is None:
            self.token_alloc = {
                0: 10,
                1: 15,
                2: 15,
                3: 15,
                4: 20,
                5: 20,
                6: 20,
                7: 20,
                8: 20,
                9: 15,
            }

    @property
    def K_total(self) -> int:
        return sum(self.token_alloc[b] for b in range(10))

    @property
    def max_K(self) -> int:
        return max(self.token_alloc.values())


# -------------------------
# Feature extractor (stereo -> (B,7,T,F))
# -------------------------

class StereoFeatureExtractor(nn.Module):
    def __init__(self, cfg: SeparatorConfig):
        super().__init__()
        self.cfg = cfg

    def stft(self, wav: torch.Tensor, n_fft: int) -> torch.Tensor:
        """
        wav: (B,2,N)
        returns complex STFT: (B,2,T,F)
        """
        device = wav.device
        win = _make_hann_window(n_fft, device=device)

        # torch.stft expects (B,N) or (N,)
        X = []
        for ch in range(2):
            x = wav[:, ch]
            Xch = torch.stft(
                x,
                n_fft=n_fft,
                hop_length=self.cfg.hop,
                win_length=n_fft,
                window=win,
                center=self.cfg.center,
                pad_mode=self.cfg.pad_mode,
                normalized=self.cfg.normalized,
                onesided=True,
                return_complex=True,
            )  # (B,F,T) complex
            Xch = Xch.transpose(-2, -1).contiguous()  # -> (B,T,F)
            X.append(Xch)
        X = torch.stack(X, dim=1)  # (B,2,T,F)
        return X

    def features_from_stft(self, X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        X: (B,2,T,F) complex
        returns (B,7,T,F):
         [L_re, L_im, R_re, R_im, ILD, sin(IPD), cos(IPD)]
        """
        L = X[:, 0]
        R = X[:, 1]

        Lre, Lim = L.real, L.imag
        Rre, Rim = R.real, R.imag

        magL = torch.abs(L)
        magR = torch.abs(R)
        ild = _safe_log(magL + eps) - _safe_log(magR + eps)  # (B,T,F)

        phL = torch.angle(L)
        phR = torch.angle(R)
        ipd = phL - phR
        sin_ipd = torch.sin(ipd)
        cos_ipd = torch.cos(ipd)

        feat = torch.stack([Lre, Lim, Rre, Rim, ild, sin_ipd, cos_ipd], dim=1)  # (B,7,T,F)
        return feat

    def forward(self, wav: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        returns dict with:
          X4096, X2048: complex (B,2,T,F)
          Feat4096, Feat2048: (B,7,T,F)
        """
        X4096 = self.stft(wav, self.cfg.n_fft_main)
        X2048 = self.stft(wav, self.cfg.n_fft_sub)

        # align T (they should be equal with same hop+center, but in practice can differ by 1)
        T = min(X4096.shape[2], X2048.shape[2])
        X4096 = X4096[:, :, :T]
        X2048 = X2048[:, :, :T]

        F4096 = X4096.shape[-1]
        F2048 = X2048.shape[-1]
        assert F4096 == 2049 and F2048 == 1025, (F4096, F2048)

        Feat4096 = self.features_from_stft(X4096)
        Feat2048 = self.features_from_stft(X2048)
        return {"X4096": X4096, "X2048": X2048, "Feat4096": Feat4096, "Feat2048": Feat2048}


# -------------------------
# Band bin selection
# -------------------------

def _hz_to_bin(hz: float, sr: int, n_fft: int) -> int:
    # onesided rfft bins: k corresponds to k*sr/n_fft
    df = sr / n_fft
    return int(math.floor(hz / df))

def _hz_to_bin_floor(hz: float, sr: int, n_fft: int) -> int:
    return int(math.floor(hz / (sr / n_fft)))

def _hz_to_bin_ceil(hz: float, sr: int, n_fft: int) -> int:
    return int(math.ceil(hz / (sr / n_fft)))

def _build_band_bins(cfg, bands):
    out = {}
    nyq = cfg.sample_rate / 2
    for b in bands:
        F = b.n_fft // 2 + 1
        k0 = _hz_to_bin_floor(b.hz_low, cfg.sample_rate, b.n_fft)

        if b.hz_high >= nyq - 1e-6:
            k1 = F
        else:
            k1 = _hz_to_bin_ceil(b.hz_high, cfg.sample_rate, b.n_fft)

        k0 = max(0, min(F - 1, k0))
        k1 = max(k0 + 1, min(F, k1))
        out[b.band_id] = slice(k0, k1)
    return out


# -------------------------
# Band encoder (5 stages + axial attention)
# -------------------------

class BandEncoder(nn.Module):
    def __init__(
        self,
        cfg: SeparatorConfig,
        stride_schedule: List[Tuple[int, int]],  # len=5: (sT,sF)
    ):
        super().__init__()
        assert len(stride_schedule) == 5

        C1, C2, C3, C4, C5 = cfg.c_e1, cfg.c_e2, cfg.c_e3, cfg.c_e4, cfg.c_e5

        self.e1 = Conv2dBlock(cfg.c_in, C1, stride=stride_schedule[0])
        self.a1 = AxialAttnF(C1, cfg.axial_heads, cfg.attn_dropout, cfg.ffn_dropout, use_log_rope=True)

        self.e2 = Conv2dBlock(C1, C2, stride=stride_schedule[1])
        self.a2 = AxialAttnF(C2, cfg.axial_heads, cfg.attn_dropout, cfg.ffn_dropout, use_log_rope=True)

        self.e3 = Conv2dBlock(C2, C3, stride=stride_schedule[2])
        self.a3 = AxialAttnTF(C3, cfg.axial_heads, cfg.attn_dropout, cfg.ffn_dropout, use_log_rope_f=True)

        self.e4 = Conv2dBlock(C3, C4, stride=stride_schedule[3])
        self.a4 = AxialAttnTF(C4, cfg.axial_heads, cfg.attn_dropout, cfg.ffn_dropout, use_log_rope_f=True)

        self.e5 = Conv2dBlock(C4, C5, stride=stride_schedule[4])
        self.a5 = AxialAttnTF(C5, cfg.axial_heads, cfg.attn_dropout, cfg.ffn_dropout, use_log_rope_f=True)

        self.norm_e5 = nn.GroupNorm(min(8, C5), C5)

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        x: (B,7,T,F_band)
        returns dict stage->tensor after axial:
          1:(B,32,T, F1), 2:(B,96,T/2,F2), 3:(B,160,T/4,F3), 4:(B,192,T/4,F4), 5:(B,192,T/4,F5)
        """
        s1 = self.a1(self.e1(x))
        s2 = self.a2(self.e2(s1))
        s3 = self.a3(self.e3(s2))
        s4 = self.a4(self.e4(s3))
        s5 = self.a5(self.e5(s4))
        s5n = self.norm_e5(s5)
        return {1: s1, 2: s2, 3: s3, 4: s4, 5: s5n}


# -------------------------
# Trunk tokenization: per-band tokenizer + F abs embedding per band
# -------------------------

class BandTokenizer(nn.Module):
    def __init__(self, cfg: SeparatorConfig, band_id: int, K_b: int):
        super().__init__()
        self.band_id = band_id
        self.K_b = K_b
        self.tokenizer = PerTimeTokenizer(cfg.c_trunk, cfg.trunk_heads, K=K_b, attn_dropout=cfg.attn_dropout)

        # learned abs embedding over F within band (we'll generate indices 0..F'-1 at runtime and embed via a small MLP)
        # Using a linear "pos projection" avoids fixed max-F tables.
        self.pos_mlp = nn.Sequential(
            nn.Linear(1, cfg.c_trunk),
            nn.SiLU(),
            nn.Linear(cfg.c_trunk, cfg.c_trunk),
        )

    def forward(self, e5: torch.Tensor) -> torch.Tensor:
        # e5: (B,192,T',F')
        B, C, T, F_ = e5.shape

        # build normalized [0..1] positions and map to embedding
        pos = torch.linspace(0, 1, steps=F_, device=e5.device, dtype=e5.dtype).view(F_, 1)
        pos_emb = self.pos_mlp(pos).view(1, C, 1, F_)  # (1,C,1,F)
        x = e5 + pos_emb

        q = self.tokenizer(x)  # (B,C,T,K_b)
        return q


# -------------------------
# Factorized embedding for trunk pseudo-frequency axis (band + slot)
# -------------------------

class TrunkFPosEmbedding(nn.Module):
    def __init__(self, cfg: SeparatorConfig):
        super().__init__()
        self.band_emb = nn.Embedding(10, cfg.c_trunk)
        self.slot_emb = nn.Embedding(cfg.max_K, cfg.c_trunk)

        # Precompute mapping from K_total index -> (band_id, slot_id)
        band_ids = []
        slot_ids = []
        for b in range(10):
            Kb = cfg.token_alloc[b]
            for s in range(Kb):
                band_ids.append(b)
                slot_ids.append(s)

        self.register_buffer("band_ids", torch.tensor(band_ids, dtype=torch.long), persistent=False)
        self.register_buffer("slot_ids", torch.tensor(slot_ids, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,K_total)
        K = x.shape[-1]
        assert K == self.band_ids.numel()
        pe = self.band_emb(self.band_ids) + self.slot_emb(self.slot_ids)  # (K,C)
        pe = pe.t().unsqueeze(0).unsqueeze(2)  # (1,C,1,K)
        return x + pe


class TrunkTPosEmbedding(nn.Module):
    """
    Adds abs position embedding on T' axis:
      x: (B,C,T,K) -> x + pe_t where pe_t broadcasted over K
    """
    def __init__(self, cfg: SeparatorConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, cfg.c_trunk),
            nn.SiLU(),
            nn.Linear(cfg.c_trunk, cfg.c_trunk),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, K = x.shape
        pos = torch.linspace(0, 1, steps=T, device=x.device, dtype=x.dtype).view(T, 1)  # (T,1)
        pe = self.mlp(pos).t().unsqueeze(0).unsqueeze(-1)  # (1,C,T,1)
        return x + pe


# -------------------------
# Trunk: full 2D MSHA (flatten T*K)
# -------------------------

class TrunkMSHA(nn.Module):
    def __init__(self, cfg: SeparatorConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.c_trunk, cfg.trunk_heads, cfg.attn_dropout, cfg.ffn_dropout, use_rope=False)
            for _ in range(cfg.trunk_depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,K)
        B, C, T, K = x.shape
        y = x.permute(0, 2, 3, 1).contiguous().view(B, T * K, C)  # (B, L, C)
        for blk in self.blocks:
            y = blk(y, rope_cache=None)
        y = y.view(B, T, K, C).permute(0, 3, 1, 2).contiguous()
        return y


# -------------------------
# Skip tokenizers: shared per stage, but band-specific K_b queries
# -------------------------

class StageSkipTokenizer(nn.Module):
    """
    Shared projections per stage; per-band learned query banks sized by K_b.
    (B,Cs,T, F_band') -> (B,Cs,T, K_b)
    """
    def __init__(self, cfg: SeparatorConfig, stage: int, C: int):
        super().__init__()
        self.stage = stage
        self.C = C
        self.num_heads = cfg.trunk_heads
        assert C % self.num_heads == 0

        # shared cross-attn projections:
        self.ca = MHCA(C, cfg.trunk_heads, dropout=cfg.attn_dropout, use_rope_on_ctx=False)
        self.ln_q = nn.LayerNorm(C)
        self.ln_ctx = nn.LayerNorm(C)

        # per-band queries
        self.q_per_band = nn.ParameterDict()
        for b in range(10):
            Kb = cfg.token_alloc[b]
            self.q_per_band[str(b)] = nn.Parameter(torch.randn(Kb, C) * 0.02)

    def forward(self, x: torch.Tensor, band_id: int) -> torch.Tensor:
        # x: (B,C,T,F)
        B, C, T, F_ = x.shape
        ctx = x.permute(0, 2, 3, 1).contiguous().view(B * T, F_, C)  # (BT,F,C)
        q = self.q_per_band[str(band_id)]  # (Kb,C)
        Kb = q.shape[0]
        q = q.unsqueeze(0).expand(B * T, Kb, C)  # (BT,Kb,C)
        out = self.ca(self.ln_q(q), self.ln_ctx(ctx))  # (BT,Kb,C)
        out = out.view(B, T, Kb, C).permute(0, 3, 1, 2).contiguous()  # (B,C,T,Kb)
        return out


# -------------------------
# Per-head decoder (D5..D1) + axial attention + skip fusion
# -------------------------

class DecoderStage(nn.Module):
    def __init__(self, cin: int, cout: int, upsample_t: bool):
        super().__init__()
        self.upsample_t = upsample_t
        self.proj = Conv2dBlock(cin, cout, stride=(1, 1), k=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,K)
        if self.upsample_t:
            # upsample only time axis
            x = F.interpolate(x, scale_factor=(2, 1), mode="nearest")
        return self.proj(x)


class HeadDecoder(nn.Module):
    def __init__(self, cfg: SeparatorConfig):
        super().__init__()
        self.cfg = cfg

        # Fusion convs per stage: concat([dec, skip]) -> 1x1 -> dec
        self.fuse5 = Conv1x1(cfg.c_d5 + cfg.c_e5, cfg.c_d5)
        self.fuse4 = Conv1x1(cfg.c_d5 + cfg.c_e4, cfg.c_d5)  # before D4 we still have 192
        self.fuse3 = Conv1x1(cfg.c_d4 + cfg.c_e3, cfg.c_d4)
        self.fuse2 = Conv1x1(cfg.c_d3 + cfg.c_e2, cfg.c_d3)
        self.fuse1 = Conv1x1(cfg.c_d2 + cfg.c_e1, cfg.c_d2)

        # Decoder stages (time upsample at D3 and D2)
        self.d5 = DecoderStage(cfg.c_trunk, cfg.c_d5, upsample_t=False)  # T/4
        self.d4 = DecoderStage(cfg.c_d5, cfg.c_d4, upsample_t=False)     # T/4
        self.d3 = DecoderStage(cfg.c_d4, cfg.c_d3, upsample_t=True)      # -> T/2
        self.d2 = DecoderStage(cfg.c_d3, cfg.c_d2, upsample_t=True)      # -> T
        self.d1 = DecoderStage(cfg.c_d2, cfg.c_d1, upsample_t=False)     # T

        # Axial attention in decoder:
        # after D5, D4: TF
        self.a5 = AxialAttnTF(cfg.c_d5, cfg.axial_heads, cfg.attn_dropout, cfg.ffn_dropout, use_log_rope_f=True)
        self.a4 = AxialAttnTF(cfg.c_d4, cfg.axial_heads, cfg.attn_dropout, cfg.ffn_dropout, use_log_rope_f=True)
        # after D3, D2, D1: F only
        self.a3 = AxialAttnF(cfg.c_d3, cfg.axial_heads, cfg.attn_dropout, cfg.ffn_dropout, use_log_rope=True)
        self.a2 = AxialAttnF(cfg.c_d2, cfg.axial_heads, cfg.attn_dropout, cfg.ffn_dropout, use_log_rope=True)
        self.a1 = AxialAttnF(cfg.c_d1, cfg.axial_heads, cfg.attn_dropout, cfg.ffn_dropout, use_log_rope=True)

    def forward(self, trunk_out: torch.Tensor, skips: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        trunk_out: (B,192,T/4,K_total)
        skips[s]:  (B,C_s,T_s,K_total) for s=1..5
        returns: (B,96,T,K_total)
        """
        x = trunk_out
        # D5 @ T/4
        skip5 = skips[5]
        x = self.fuse5(torch.cat([x, skip5], dim=1))
        x = self.d5(x)
        x = self.a5(x)

        # D4 @ T/4
        skip4 = skips[4]
        x = self.fuse4(torch.cat([x, skip4], dim=1))
        x = self.d4(x)
        x = self.a4(x)

        # D3: fuse @ T/4, then upsample -> T/2
        skip3 = skips[3]
        x = self.fuse3(torch.cat([x, skip3], dim=1))
        x = self.d3(x)
        x = self.a3(x)

        # D2: fuse @ T/2, then upsample -> T
        skip2 = skips[2]
        x = self.fuse2(torch.cat([x, skip2], dim=1))
        x = self.d2(x)
        x = self.a2(x)

        # D1 @ T
        skip1 = skips[1]
        x = self.fuse1(torch.cat([x, skip1], dim=1))
        x = self.d1(x)
        x = self.a1(x)

        return x

# -------------------------
# Shared detokenizers (K->F) + per-head mask heads
# -------------------------

class MaskHeadPerResolution(nn.Module):
    """
    For a single head and a single resolution:
      D1_tokens (B,96,T,K) -> detok -> refine -> produce crm/mp + gate logits.
    """
    def __init__(self, cfg: SeparatorConfig, detok: PerTimeDetokenizer, F_out: int):
        super().__init__()
        self.detok = detok
        self.refine = RefineBlock(cfg.c_d1, k_t=5, k_f=3)

        # micro heads
        self.head_crm = nn.Conv2d(cfg.c_d1, 4, kernel_size=1)   # Re/Im for L/R
        self.head_mp = nn.Conv2d(cfg.c_d1, 4, kernel_size=1)    # magL,magR,phiL,phiR
        self.head_gate = nn.Conv2d(cfg.c_d1, 2, kernel_size=1)  # 2-class logits (crm vs mp)

        self.F_out = F_out

    def forward(self, x_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_tokens: (B,96,T,K)
        returns:
          M_complex: (B,2,T,F) complex mask in LR domain
          gate_probs: (B,2,T,F) probabilities [p_crm, p_mp]
        """
        feat = self.detok(x_tokens)            # (B,96,T,F_out)
        feat = self.refine(feat)               # (B,96,T,F_out)

        crm_logits = self.head_crm(feat)       # (B,4,T,F)
        mp_logits = self.head_mp(feat)         # (B,4,T,F)
        gate_logits = self.head_gate(feat)     # (B,2,T,F)

        # gate probs (common for L/R)
        gate_probs = torch.softmax(gate_logits, dim=1)  # (B,2,T,F)
        p_crm = gate_probs[:, 0:1]
        p_mp = gate_probs[:, 1:2]

        # cRM mask
        L_re, L_im, R_re, R_im = crm_logits[:, 0], crm_logits[:, 1], crm_logits[:, 2], crm_logits[:, 3]
        Mcrm_L = torch.complex(L_re, L_im)
        Mcrm_R = torch.complex(R_re, R_im)
        Mcrm_L = _crm_soft_scale(Mcrm_L)
        Mcrm_R = _crm_soft_scale(Mcrm_R)

        # Mag+Phase mask
        magL, magR, phiL, phiR = mp_logits[:, 0], mp_logits[:, 1], mp_logits[:, 2], mp_logits[:, 3]
        magL = torch.sigmoid(magL)
        magR = torch.sigmoid(magR)
        dphiL = torch.tanh(phiL) * math.pi
        dphiR = torch.tanh(phiR) * math.pi
        Mmp_L = torch.polar(magL, dphiL)  # complex
        Mmp_R = torch.polar(magR, dphiR)

        # Mix candidates with gate
        M_L = p_crm[:, 0] * Mcrm_L + p_mp[:, 0] * Mmp_L  # (B,T,F)
        M_R = p_crm[:, 0] * Mcrm_R + p_mp[:, 0] * Mmp_R  # same p for both channels

        M = torch.stack([M_L, M_R], dim=1)  # (B,2,T,F) complex
        return M, gate_probs


# -------------------------
# Router: from D1 tokens -> logits on 4096 grid -> derive 2048 gate
# -------------------------

class SharedResolutionRouter(nn.Module):
    """
    router принимает (B,96,T,K_total) и выдаёт a4096 (B,1,T,2049).
    Далее g4096 = sigmoid(a4096), g2048 = 1 - sigmoid(down(a4096)).
    Gate smoothing: small conv on logits.
    """
    def __init__(self, cfg: SeparatorConfig, detok_logits_4096: PerTimeDetokenizer):
        super().__init__()
        self.detok = detok_logits_4096
        self.to_logit = nn.Conv2d(cfg.c_d1, 1, kernel_size=1)

        # smoothing on logits (shared for all heads)
        self.smooth = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_tokens: (B,96,T,K)
        returns g4096 (B,1,T,2049), g2048 (B,1,T,1025)
        """
        feat4096 = self.detok(x_tokens)      # (B,96,T,2049)
        a4096 = self.to_logit(feat4096)      # (B,1,T,2049)
        a4096 = self.smooth(a4096)

        a2048 = _downsample_logits_4096_to_2048(a4096)  # (B,1,T,1025)
        a2048 = self.smooth(a2048)  # reuse same smoother

        g4096 = torch.sigmoid(a4096)
        g2048 = 1.0 - torch.sigmoid(a2048)
        return g4096, g2048


# -------------------------
# Full separator model
# -------------------------

class StemSeparator(nn.Module):
    def __init__(self, cfg: SeparatorConfig):
        super().__init__()
        self.cfg = cfg
        self.fe = StereoFeatureExtractor(cfg)

        # Define bands exactly as in your spec (IDs 0..9)
        self.bands: List[BandSpec] = [
            # main bands (4096)
            BandSpec(0, 0, 500, cfg.n_fft_main),
            BandSpec(2, 500, 2000, cfg.n_fft_main),
            BandSpec(4, 2000, 8000, cfg.n_fft_main),
            BandSpec(6, 8000, 16000, cfg.n_fft_main),
            BandSpec(8, 16000, 24000, cfg.n_fft_main),
            # sub bands (2048)
            BandSpec(1, 250, 1250, cfg.n_fft_sub),
            BandSpec(3, 1250, 5000, cfg.n_fft_sub),
            BandSpec(5, 5000, 12000, cfg.n_fft_sub),
            BandSpec(7, 12000, 20000, cfg.n_fft_sub),
            BandSpec(9, 20000, 24000, cfg.n_fft_sub),
        ]
        self.band_bins = _build_band_bins(cfg, self.bands)

        # Stride schedules per band (E1..E5), as you described.
        # Each tuple = (stride_T, stride_F).
        self.encoder_strides: Dict[int, List[Tuple[int, int]]] = {
            0: [(1, 1), (2, 1), (2, 1), (1, 1), (1, 1)],
            1: [(1, 2), (2, 1), (2, 1), (1, 1), (1, 1)],
            2: [(1, 2), (2, 2), (2, 1), (1, 1), (1, 1)],
            3: [(1, 2), (2, 2), (2, 2), (1, 1), (1, 1)],
            9: [(1, 2), (2, 2), (2, 2), (1, 1), (1, 1)],
        }
        # band4..band8
        for b in [4, 5, 6, 7, 8]:
            self.encoder_strides[b] = [(1, 2), (2, 2), (2, 2), (1, 2), (1, 1)]

        # Encoders per band (0..9)
        self.encoders = nn.ModuleDict({
            str(b): BandEncoder(cfg, self.encoder_strides[b])
            for b in range(10)
        })

        # Per-band tokenizers for trunk input (use E5 output)
        self.band_tokenizers = nn.ModuleDict({
            str(b): BandTokenizer(cfg, band_id=b, K_b=cfg.token_alloc[b])
            for b in range(10)
        })

        # Trunk embeddings and trunk
        self.trunk_pos = TrunkFPosEmbedding(cfg)
        self.trunk_tpos = TrunkTPosEmbedding(cfg)
        self.trunk = TrunkMSHA(cfg)

        # Skip tokenizers per stage (shared per stage)
        # stage -> tokenizer(C_s)
        self.skip_tokenizers = nn.ModuleDict({
            "1": StageSkipTokenizer(cfg, stage=1, C=cfg.c_e1),
            "2": StageSkipTokenizer(cfg, stage=2, C=cfg.c_e2),
            "3": StageSkipTokenizer(cfg, stage=3, C=cfg.c_e3),
            "4": StageSkipTokenizer(cfg, stage=4, C=cfg.c_e4),
            "5": StageSkipTokenizer(cfg, stage=5, C=cfg.c_e5),
        })
        self.decoders = nn.ModuleDict({
            h: HeadDecoder(cfg) for h in cfg.heads
        })

        # Shared detokenizers for mask heads
        self.detok2048 = PerTimeDetokenizer(cfg.c_d1, cfg.trunk_heads, F_out=1025, attn_dropout=cfg.attn_dropout)
        self.detok4096 = PerTimeDetokenizer(cfg.c_d1, cfg.trunk_heads, F_out=2049, attn_dropout=cfg.attn_dropout)

        # Per-head per-resolution mask heads (refine + microheads)
        self.mask_heads_2048 = nn.ModuleDict({
            h: MaskHeadPerResolution(cfg, self.detok2048, F_out=1025) for h in cfg.heads
        })
        self.mask_heads_4096 = nn.ModuleDict({
            h: MaskHeadPerResolution(cfg, self.detok4096, F_out=2049) for h in cfg.heads
        })

        # Shared router (uses detok4096 + 1x1 -> logits)
        self.router = SharedResolutionRouter(cfg, detok_logits_4096=self.detok4096)

    def _select_band_feat(self, Feat4096: torch.Tensor, Feat2048: torch.Tensor, band_id: int) -> torch.Tensor:
        sl = self.band_bins[band_id]
        n_fft = next(b.n_fft for b in self.bands if b.band_id == band_id)
        if n_fft == self.cfg.n_fft_main:
            return Feat4096[..., sl]  # (B,7,T,Fb)
        else:
            return Feat2048[..., sl]

    def _precompute_skips(self, enc_stages: Dict[int, Dict[int, torch.Tensor]]) -> Dict[int, torch.Tensor]:
        """
        Precompute Skip_s once per stage (independent of head).
        Returns:
          skips[s] = (B, C_s, T_s, K_total) for s in 1..5
        """
        skips: Dict[int, torch.Tensor] = {}
        for s in range(1, 6):
            tok = self.skip_tokenizers[str(s)]
            parts = []
            for b in range(10):
                xb = enc_stages[b][s]  # (B,Cs,Ts,Fb)
                qb = tok(xb, band_id=b)  # (B,Cs,Ts,Kb)
                parts.append(qb)
            skips[s] = torch.cat(parts, dim=-1)  # (B,Cs,Ts,K_total)
        return skips

    def _istft(self, S: torch.Tensor, n_fft: int, length: int) -> torch.Tensor:
        """
        S: (B,2,T,F) complex
        length: desired output length (e.g. original N)
        """
        device = S.device
        win = torch.hann_window(n_fft, periodic=True, device=device, dtype=torch.float32)

        out = []
        for ch in range(2):
            X = S[:, ch].transpose(-2, -1).contiguous()  # (B,F,T)
            x = torch.istft(
                X,
                n_fft=n_fft,
                hop_length=self.cfg.hop,
                win_length=n_fft,
                window=win,
                center=self.cfg.center,
                normalized=self.cfg.normalized,
                onesided=True,
                length=length,
                return_complex=False,
            )
            out.append(x)

        return torch.stack(out, dim=1)  # (B,2,length)

    def forward(self, wav: torch.Tensor, return_debug: bool = False) -> Dict[str, torch.Tensor]:
        """
        wav: (B,2,N)
        returns dict: head_name -> stem waveform (B,2,N)
        """
        B, Ch, N = wav.shape
        assert Ch == 2

        # --- NEW: pad waveform so that STFT frame count T is divisible by 4 ---
        hop = self.cfg.hop
        mult = 4  # because you downsample time twice (T -> T/2 -> T/4)

        if not self.cfg.center:
            # For center=False formula depends on n_fft; if you ever need it, we can implement precisely.
            # For now assume center=True as in your setup.
            raise ValueError(
                "This padding logic assumes center=True. Set cfg.center=True or implement center=False case.")

        # With center=True, torch.stft frame count is essentially: T = floor(N/hop) + 1
        T0 = (N // hop) + 1
        Tt = ((T0 + mult - 1) // mult) * mult  # ceil to multiple of 4
        N_pad = max(N, (Tt - 1) * hop)
        pad = N_pad - N
        if pad > 0:
            # zero-pad is simplest and stable; reflect is also possible but needs care with very short segments
            wav = F.pad(wav, (0, pad))

        pack = self.fe(wav)  # STFT now sees padded wav
        X4096 = pack["X4096"]   # (B,2,T,2049) complex
        X2048 = pack["X2048"]   # (B,2,T,1025) complex
        Feat4096 = pack["Feat4096"]
        Feat2048 = pack["Feat2048"]
        T = Feat4096.shape[2]

        # 1) Run band encoders, store stage outputs for skips
        enc_stages: Dict[int, Dict[int, torch.Tensor]] = {}
        for b in range(10):
            feat_b = self._select_band_feat(Feat4096, Feat2048, band_id=b)
            stages = self.encoders[str(b)](feat_b)
            enc_stages[b] = stages  # stages[1..5]

        # 2) Per-band tokenization from E5 -> trunk input tokens
        trunk_parts = []
        for b in range(10):
            e5 = enc_stages[b][5]  # (B,192,T/4,F'_b) by design
            q_b = self.band_tokenizers[str(b)](e5)  # (B,192,T/4,K_b)
            trunk_parts.append(q_b)
        x_trunk = torch.cat(trunk_parts, dim=-1)  # (B,192,T/4,K_total)

        # add factorized (band+slot) embedding on pseudo-frequency axis
        x_trunk = self.trunk_pos(x_trunk)
        x_trunk = self.trunk_tpos(x_trunk)  # abs-pos on T' axis

        # 3) Trunk MSHA
        x_trunk = self.trunk(x_trunk)  # (B,192,T/4,K_total)

        # 4) Shared router gate logits/probs (common for L/R, but computed from tokens)
        # We compute router per head (as you described: router takes tokens after D1),
        # but it's shared weights. We will run it after each head's D1.

        out: Dict[str, torch.Tensor] = {}
        debug: Dict[str, Dict[str, torch.Tensor]] = {} if return_debug else None

        skips = self._precompute_skips(enc_stages)

        for h in self.cfg.heads:
            # 4.1) per-head decoder
            d1_tokens = self.decoders[h](x_trunk, skips)  # (B,96,T,K_total)

            # 4.2) per-resolution masks
            M4096, gate4096 = self.mask_heads_4096[h](d1_tokens)  # (B,2,T,2049)
            M2048, gate2048 = self.mask_heads_2048[h](d1_tokens)  # (B,2,T,1025)

            # apply masks to mixture TF
            S4096 = _complex_mul(X4096, M4096)
            S2048 = _complex_mul(X2048, M2048)

            # 4.3) shared router: resolution gate
            g4096, g2048 = self.router(d1_tokens)  # (B,1,T,2049), (B,1,T,1025)

            S4096_g = S4096 * g4096
            S2048_g = S2048 * g2048

            # 4.4) ISTFT per resolution and sum in time domain
            s4096 = self._istft(S4096_g, n_fft=self.cfg.n_fft_main, length=wav.shape[-1])  # N_pad
            s2048 = self._istft(S2048_g, n_fft=self.cfg.n_fft_sub, length=wav.shape[-1])  # N_pad
            stem = s4096 + s2048
            stem = stem[..., :N]  # back to original length
            out[h] = stem

            if return_debug:
                debug[h] = {
                    "d1_tokens": d1_tokens.detach(),
                    "M4096": M4096.detach(),
                    "M2048": M2048.detach(),
                    "gate4096_crm_mp": gate4096.detach(),
                    "gate2048_crm_mp": gate2048.detach(),
                    "g4096": g4096.detach(),
                    "g2048": g2048.detach(),
                }

        if return_debug:
            out["_debug"] = debug  # type: ignore[assignment]

        return out


# -------------------------
# Quick sanity test
# -------------------------

if __name__ == "__main__":
    cfg = SeparatorConfig()
    model = StemSeparator(cfg).eval()

    with torch.no_grad():
        wav = torch.randn(2, 2, 48_000 * 2)  # (B=2, stereo, 2s)
        y = model(wav, return_debug=False)
        for k, v in y.items():
            print(k, v.shape)  # each stem: (B,2,N)
