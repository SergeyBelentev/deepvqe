from __future__ import annotations
from typing import Final
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


FLOAT_EPS: Final[float] = torch.finfo(torch.float32).eps


def norm2d(ch: int, groups: int = 8) -> nn.Module:
    g = min(groups, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class FE(nn.Module):
    """Feature extraction block.

    The block computes a normalized complex spectrogram magnitude and
    rearranges the tensor into channel-first layout expected by the
    convolutional encoder.
    """

    def __init__(self, c: float = 0.3) -> None:
        super().__init__()
        self.c = c

    def forward(self, x: Tensor) -> Tensor:
        """Normalize input spectrogram.

        Args:
            x: Complex spectrogram of shape ``(B, F, T, 2)``.

        Returns:
            Tensor with shape ``(B, 2, T, F)``.
        """

        x_mag = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        x_c = torch.div(x, x_mag.pow(1 - self.c) + FLOAT_EPS)
        return x_c.permute(0, 3, 2, 1).contiguous()


class ResidualBlock(nn.Module):
    """Simple residual block with padding helper."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pad = nn.ZeroPad2d([1, 1, 3, 0])
        self.conv = nn.Conv2d(channels, channels, kernel_size=(4, 3))
        self.bn = norm2d(channels)
        self.elu = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        """Run a residual step on ``(B, C, T, F)`` features."""

        return self.elu(self.bn(self.conv(self.pad(x)))) + x


class AlignBlockBi(nn.Module):
    """
    Bidirectional delay-aware alignment WITHOUT unfold().

    Idea:
      - pad k/v along time
      - compute attention logits by shifting k over K lags
      - do it in small blocks (k_block) to trade speed vs memory
      - compute ctx similarly (weighted sum of shifted v)

    Shapes:
      x_mic/x_ref: (B, C, T, F)
      ctx:         (B, H, T, F)
      att:         (B, 1, T, K)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        delay_past: int = 25,
        delay_future: int = 25,
        logit_kernel_t: int = 5,
        logit_kernel_k: int = 3,
        *,
        softmax_fp32: bool = True,
        k_block: int = 8,               # 1..K, чем больше -> быстрее, но больше память
        accumulate_fp32: bool = True,   # суммирование ctx в fp32
    ) -> None:
        super().__init__()
        if delay_past < 0 or delay_future < 0:
            raise ValueError("delay_past/delay_future must be >= 0")

        self.delay_past = int(delay_past)
        self.delay_future = int(delay_future)
        self.K = self.delay_past + self.delay_future + 1

        if (logit_kernel_t % 2) != 1 or (logit_kernel_k % 2) != 1:
            raise ValueError("logit_kernel_t/logit_kernel_k must be odd for symmetric padding.")

        self.softmax_fp32 = bool(softmax_fp32)
        self.k_block = int(k_block)
        self.accumulate_fp32 = bool(accumulate_fp32)

        self.pconv_mic = nn.Conv2d(in_channels, hidden_channels, 1)
        self.pconv_ref = nn.Conv2d(in_channels, hidden_channels, 1)
        self.pconv_val = nn.Conv2d(in_channels, hidden_channels, 1)

        pad_t = logit_kernel_t // 2
        pad_k = logit_kernel_k // 2
        self.logit_smoother = nn.Sequential(
            nn.ZeroPad2d([pad_k, pad_k, pad_t, pad_t]),  # pads (K,T) in Conv2d layout
            nn.Conv2d(hidden_channels, 1, (logit_kernel_t, logit_kernel_k)),
        )

    def forward(self, x_mic: Tensor, x_ref: Tensor, return_att: bool = False):
        q = self.pconv_mic(x_mic)  # (B,H,T,F)
        k = self.pconv_ref(x_ref)  # (B,H,T,F)
        v = self.pconv_val(x_ref)  # (B,H,T,F)

        B, H, T, Freq = q.shape
        K = self.K
        kb = max(1, min(self.k_block, K))

        # pad time: (B,H,T+p+f,F)
        k_pad = nn.functional.pad(k, (0, 0, self.delay_past, self.delay_future))
        v_pad = nn.functional.pad(v, (0, 0, self.delay_past, self.delay_future))

        # ---------- logits (B,H,T,K) ----------
        # держим логиты в dtype q (обычно fp32 у тебя), но можно принудить fp32
        logits_dtype = torch.float32 if self.softmax_fp32 else q.dtype
        att_logits = q.new_empty((B, H, T, K), dtype=logits_dtype)

        # block-wise по K, чтобы не делать огромных окон
        q_for = q.to(logits_dtype) if q.dtype != logits_dtype else q
        for s in range(0, K, kb):
            e = min(K, s + kb)
            # stack shifted k: (B,H,T,kb,F)
            k_blk = torch.stack(
                [k_pad[:, :, (s+i):(s+i+T), :] for i in range(e - s)],
                dim=3,
            ).to(logits_dtype)
            # dot over F: (B,H,T,kb)
            att_logits[:, :, :, s:e] = (k_blk * q_for.unsqueeze(3)).sum(dim=-1)

        # smooth logits: Conv2d expects (B,C,T,K) -> OK
        att_logits = self.logit_smoother(att_logits)  # (B,1,T,K)

        if self.softmax_fp32:
            att = torch.softmax(att_logits.float(), dim=-1).to(att_logits.dtype)
        else:
            att = torch.softmax(att_logits, dim=-1)

        # ---------- ctx: sum_i att_i * v_shift_i ----------
        # аккуратно: копим в fp32, потом кастим (у тебя fp32 обучение, но пусть будет надёжно)
        if self.accumulate_fp32:
            ctx = q.new_zeros((B, H, T, Freq), dtype=torch.float32)
            v_for = v_pad.to(torch.float32) if v_pad.dtype != torch.float32 else v_pad
            att_for = att.to(torch.float32) if att.dtype != torch.float32 else att
        else:
            ctx = q.new_zeros((B, H, T, Freq), dtype=q.dtype)
            v_for = v_pad
            att_for = att

        for s in range(0, K, kb):
            e = min(K, s + kb)
            v_blk = torch.stack(
                [v_for[:, :, (s+i):(s+i+T), :] for i in range(e - s)],
                dim=3,  # (B,H,T,kb,F)
            )
            w_blk = att_for[:, 0, :, s:e]  # (B,T,kb)
            ctx = ctx + (v_blk * w_blk[:, None, :, :, None]).sum(dim=3)  # sum over kb

        if self.accumulate_fp32 and q.dtype != torch.float32:
            ctx = ctx.to(q.dtype)

        if return_att:
            return ctx, att
        return ctx


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(4, 3), stride=(1, 2)) -> None:
        super().__init__()
        self.pad = nn.ZeroPad2d([1, 1, 3, 0])
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.bn = norm2d(out_channels)
        self.elu = nn.ELU()
        self.resblock = ResidualBlock(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.resblock(self.elu(self.bn(self.conv(self.pad(x)))))


class Bottleneck(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: Tensor) -> Tensor:
        """x : (B,C,T,F)"""

        y = rearrange(x, "b c t f -> b t (c f)")
        y, _ = self.gru(y)
        y = self.fc(y)
        y = rearrange(y, "b t (c f) -> b c t f", c=x.shape[1])
        return y
    

class SubpixelConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(4, 3)) -> None:
        super().__init__()
        self.pad = nn.ZeroPad2d([1, 1, 3, 0])
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(self.pad(x))
        y = rearrange(y, "b (r c) t f -> b c t (r f)", r=2)
        return y
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(4, 3), is_last: bool = False) -> None:
        super().__init__()
        self.skip_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.resblock = ResidualBlock(in_channels)
        self.deconv = SubpixelConv2d(in_channels, out_channels, kernel_size)
        self.bn = norm2d(out_channels)
        self.elu = nn.ELU()
        self.is_last = is_last

    def forward(self, x: Tensor, x_en: Tensor) -> Tensor:
        y = x + self.skip_conv(x_en)
        y = self.deconv(self.resblock(y))
        if not self.is_last:
            y = self.elu(self.bn(y))
        return y
    

class CCM(nn.Module):
    """Complex convolving mask block."""

    def __init__(self) -> None:
        super().__init__()
        sqrt3 = np.float32(np.sqrt(3.0))
        ccm_basis = np.array([[1, -0.5, -0.5], [0, sqrt3 / 2, -sqrt3 / 2]], dtype=np.float32)
        self.register_buffer("v", torch.from_numpy(ccm_basis))

        self.unfold = nn.Sequential(nn.ZeroPad2d([1, 1, 2, 0]), nn.Unfold(kernel_size=(3, 3)))

    def forward(self, m: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            m: Mask tensor with shape ``(B, 27, T, F)``.
            x: Complex spectrogram with shape ``(B, F, T, 2)``.
        """

        m = rearrange(m, "b (r c) t f -> b r c t f", r=3)
        H_real = torch.sum(self.v[0][None, :, None, None, None] * m, dim=1)  # (B, C/3, T, F)
        H_imag = torch.sum(self.v[1][None, :, None, None, None] * m, dim=1)  # (B, C/3, T, F)

        M_real = rearrange(H_real, "b (m n) t f -> b m n t f", m=3)  # (B,3,3,T,F)
        M_imag = rearrange(H_imag, "b (m n) t f -> b m n t f", m=3)  # (B,3,3,T,F)

        x = x.permute(0, 3, 2, 1).contiguous()  # (B,2,T,F)
        x_unfold = self.unfold(x)
        x_unfold = rearrange(x_unfold, "b (c m n) (t f) -> b c m n t f", m=3, n=3, f=x.shape[-1])

        x_enh_real = torch.sum(M_real * x_unfold[:, 0] - M_imag * x_unfold[:, 1], dim=(1, 2))
        x_enh_imag = torch.sum(M_real * x_unfold[:, 1] + M_imag * x_unfold[:, 0], dim=(1, 2))
        return torch.stack([x_enh_real, x_enh_imag], dim=-1).transpose(1, 2).contiguous()


class DecoderChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.deblock5 = DecoderBlock(128, 128)
        self.deblock4 = DecoderBlock(128, 128)
        self.deblock3 = DecoderBlock(128, 128)
        self.deblock2 = DecoderBlock(128, 64)
        self.deblock1 = DecoderBlock(64, 64, is_last=False)

    def forward(self, z, en5, en4, en3, en2, x1f, x0):
        d5 = self.deblock5(z,  en5)[..., :en4.shape[-1]]
        d4 = self.deblock4(d5, en4)[..., :en3.shape[-1]]
        d3 = self.deblock3(d4, en3)[..., :en2.shape[-1]]
        d2 = self.deblock2(d3, en2)[..., :x1f.shape[-1]]
        d1 = self.deblock1(d2, x1f)[..., :x0.shape[-1]]   # (B,64,T,F_in)
        return d1


# ---------------------------
# Utils
# ---------------------------
class LayerNorm2D(nn.Module):
    """LayerNorm over channels for (B,C,T,F)."""
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        # (B,C,T,F) -> (B,T,F,C) -> LN -> (B,C,T,F)
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class DropPath(nn.Module):
    """Stochastic depth."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


def swiglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=1)
    return a * F.silu(b)


# ---------------------------
# 2D Rotary Positional Embedding (RoPE)
# ---------------------------
class Rotary2D(nn.Module):
    """
    Apply RoPE separately on time axis and freq axis.

    We split head_dim into two even parts:
      - first part uses time positions
      - second part uses freq positions

    q,k expected shape: (B, H, T, F, D)
    """
    def __init__(self, head_dim: int):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError("For 2D RoPE we want head_dim divisible by 4 (so each half is even).")
        self.head_dim = int(head_dim)
        self.dt = head_dim // 2
        self.df = head_dim // 2

        if self.dt % 2 != 0 or self.df % 2 != 0:
            raise ValueError("RoPE parts must be even.")

        self.register_buffer("_t_cache_cos", torch.empty(0), persistent=False)
        self.register_buffer("_t_cache_sin", torch.empty(0), persistent=False)
        self.register_buffer("_f_cache_cos", torch.empty(0), persistent=False)
        self.register_buffer("_f_cache_sin", torch.empty(0), persistent=False)

        self._t_cache_len = 0
        self._f_cache_len = 0

    @staticmethod
    def _build_sin_cos(length: int, dim: int, device, dtype):
        # dim is even; returns cos,sin with shape (length, dim/2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        pos = torch.arange(length, device=device, dtype=torch.float32)
        ang = torch.outer(pos, inv_freq)  # (L, dim/2)
        cos = torch.cos(ang).to(dtype=dtype)
        sin = torch.sin(ang).to(dtype=dtype)
        return cos, sin

    @staticmethod
    def _apply_rotary(x: Tensor, cos: Tensor, sin: Tensor, axis: str) -> Tensor:
        # x: (B,H,T,F,dim) where dim even
        # cos/sin:
        #  - time: (T, dim/2) -> broadcast to (1,1,T,1,dim/2)
        #  - freq: (F, dim/2) -> broadcast to (1,1,1,F,dim/2)
        dim = x.shape[-1]
        x2 = x.view(*x.shape[:-1], dim // 2, 2)
        x1 = x2[..., 0]
        x2v = x2[..., 1]

        if axis == "t":
            cos = cos[None, None, :, None, :]  # (1,1,T,1,dim/2)
            sin = sin[None, None, :, None, :]
        elif axis == "f":
            cos = cos[None, None, None, :, :]  # (1,1,1,F,dim/2)
            sin = sin[None, None, None, :, :]
        else:
            raise ValueError("axis must be 't' or 'f'")

        y1 = x1 * cos - x2v * sin
        y2 = x1 * sin + x2v * cos
        y = torch.stack([y1, y2], dim=-1).flatten(-2)
        return y

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        # q,k: (B,H,T,F,D)
        B, H, T, Freq, D = q.shape
        assert D == self.head_dim

        device = q.device
        dtype = q.dtype

        # cache sin/cos for time
        if T > self._t_cache_len or self._t_cache_cos.device != device or self._t_cache_cos.dtype != dtype:
            cos, sin = self._build_sin_cos(T, self.dt, device, dtype)
            self._t_cache_cos = cos
            self._t_cache_sin = sin
            self._t_cache_len = T
        cos_t = self._t_cache_cos[:T]
        sin_t = self._t_cache_sin[:T]

        # cache sin/cos for freq
        if Freq > self._f_cache_len or self._f_cache_cos.device != device or self._f_cache_cos.dtype != dtype:
            cos, sin = self._build_sin_cos(Freq, self.df, device, dtype)
            self._f_cache_cos = cos
            self._f_cache_sin = sin
            self._f_cache_len = Freq
        cos_f = self._f_cache_cos[:Freq]
        sin_f = self._f_cache_sin[:Freq]

        # split dims
        q_t, q_f = q[..., :self.dt], q[..., self.dt:self.dt + self.df]
        k_t, k_f = k[..., :self.dt], k[..., self.dt:self.dt + self.df]

        # apply time rope on first half
        q_t = self._apply_rotary(q_t, cos_t, sin_t, axis="t")
        k_t = self._apply_rotary(k_t, cos_t, sin_t, axis="t")

        # apply freq rope on second half
        q_f = self._apply_rotary(q_f, cos_f, sin_f, axis="f")
        k_f = self._apply_rotary(k_f, cos_f, sin_f, axis="f")

        q = torch.cat([q_t, q_f], dim=-1)
        k = torch.cat([k_t, k_f], dim=-1)
        return q, k


# ---------------------------
# Full 2D MHSA over TF tokens
# ---------------------------
class MHSA2D(nn.Module):
    """
    Full self-attention over TF grid.
    Input/Output: (B,C,T,F)

    Tokenization is implicit: N = T*F.
    Uses 2D RoPE for position.
    """
    def __init__(self, channels: int, heads: int = 8, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.channels = int(channels)
        self.heads = int(heads)
        if self.channels % self.heads != 0:
            raise ValueError("channels must be divisible by heads.")
        self.head_dim = self.channels // self.heads

        self.qkv = nn.Linear(self.channels, 3 * self.channels, bias=True)
        self.proj = nn.Linear(self.channels, self.channels, bias=True)

        self.attn_drop = float(attn_drop)
        self.proj_drop = nn.Dropout(float(proj_drop))

        self.rope2d = Rotary2D(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B,C,T,F)
        B, C, T, Freq = x.shape
        N = T * Freq

        # (B,C,T,F) -> (B,N,C)
        xt = x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        qkv = self.qkv(xt)  # (B,N,3C)
        qkv = qkv.view(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3,B,H,N,D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,N,D)

        # reshape to 2D grid for RoPE: (B,H,T,F,D)
        q2 = q.view(B, self.heads, T, Freq, self.head_dim)
        k2 = k.view(B, self.heads, T, Freq, self.head_dim)
        q2, k2 = self.rope2d(q2, k2)
        q = q2.view(B, self.heads, N, self.head_dim)
        k = k2.view(B, self.heads, N, self.head_dim)

        # SDPA (quality-first; heavy at large N)
        # out: (B,H,N,D)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().view(B, N, C)  # (B,N,C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # back to (B,C,T,F)
        out = out.view(B, T, Freq, C).permute(0, 3, 1, 2).contiguous()
        return out


# ---------------------------
# Conformer-style conv module on 2D map
# ---------------------------
class ConvModule2D(nn.Module):
    """
    Conformer conv module adapted to (B,C,T,F):
      pw -> GLU -> dwconv(k_t,k_f) -> norm -> SiLU -> pw
    """
    def __init__(self, channels: int, k_t: int = 7, k_f: int = 3, drop: float = 0.0):
        super().__init__()
        C = int(channels)
        self.pw1 = nn.Conv2d(C, 2 * C, kernel_size=1)
        self.dw = nn.Conv2d(C, C, kernel_size=(k_t, k_f), padding=(k_t // 2, k_f // 2), groups=C)
        self.n = nn.GroupNorm(1, C)
        self.pw2 = nn.Conv2d(C, C, kernel_size=1)
        self.drop = nn.Dropout(float(drop))

    def forward(self, x: Tensor) -> Tensor:
        y = self.pw1(x)
        y = swiglu(y)            # GLU-ish with SiLU gate
        y = self.dw(y)
        y = self.n(y)
        y = F.silu(y)
        y = self.pw2(y)
        y = self.drop(y)
        return y


class DWResBlock2D(nn.Module):
    """Depthwise-style residual block on (B,C,T,F)."""
    def __init__(self, channels: int, k_t: int = 5, k_f: int = 3):
        super().__init__()
        self.m1 = ConvModule2D(channels, k_t=k_t, k_f=k_f, drop=0.0)
        self.m2 = ConvModule2D(channels, k_t=k_t, k_f=k_f, drop=0.0)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.m2(self.m1(x))


class FDownsample(nn.Module):
    """Downsample only along F axis: stride=(1,2)."""
    def __init__(self, in_ch: int, out_ch: int, k_t: int = 3, k_f: int = 5):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=(k_t, k_f),
            stride=(1, 2),
            padding=(k_t // 2, k_f // 2),
            groups=in_ch,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.n = nn.GroupNorm(1, out_ch)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.n(x)
        x = self.act(x)
        return x


class FUpsample(nn.Module):
    """Upsample only along F axis by 2 and project channels."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.n = nn.GroupNorm(1, out_ch)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, target_F: int) -> Tensor:
        # (B,C,T,F) -> upsample F only
        x = F.interpolate(x, scale_factor=(1, 2), mode="nearest")
        x = self.pw(x)
        x = self.n(x)
        x = self.act(x)
        # crop to match skip (important when F is odd)
        if x.shape[-1] != target_F:
            x = x[..., :target_F]
        return x


class DepthwiseTFUNet2F(nn.Module):
    """
    Depthwise TF-U-Net, 2 levels, downsample ONLY along F.

    Input:
      d1:     (B,64,T,F)
      y_ccm:  (B,F,T,2)   (we convert to (B,2,T,F) inside)
    Output:
      residual: (B,F,T,2)

    Stability:
      - zero-init out conv
      - learnable gain (sigmoid) starting ~0
    """
    def __init__(
        self,
        d1_ch: int = 64,
        hint_ch: int = 2,
        base_ch: int = 96,
        ch1: int = 128,
        ch2: int = 160,
        k_t: int = 5,
        k_f: int = 3,
        *,
        detach_hint: bool = True,
    ):
        super().__init__()
        self.detach_hint = bool(detach_hint)
        in_ch = int(d1_ch + hint_ch)

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, int(base_ch), kernel_size=1),
            nn.GroupNorm(1, int(base_ch)),
            nn.SiLU(),
        )

        # enc level 0
        self.e0 = DWResBlock2D(int(base_ch), k_t=k_t, k_f=k_f)

        # down1 -> enc1
        self.d1 = FDownsample(int(base_ch), int(ch1), k_t=3, k_f=5)
        self.e1 = DWResBlock2D(int(ch1), k_t=k_t, k_f=k_f)

        # down2 -> enc2
        self.d2 = FDownsample(int(ch1), int(ch2), k_t=3, k_f=5)
        self.e2 = DWResBlock2D(int(ch2), k_t=k_t, k_f=k_f)

        # bottleneck
        self.mid = nn.Sequential(
            DWResBlock2D(int(ch2), k_t=k_t, k_f=k_f),
            DWResBlock2D(int(ch2), k_t=k_t, k_f=k_f),
        )

        # up2 -> dec1
        self.u2 = FUpsample(int(ch2), int(ch1))
        self.dec1_in = nn.Conv2d(int(ch1 + ch1), int(ch1), kernel_size=1)
        self.dec1 = DWResBlock2D(int(ch1), k_t=k_t, k_f=k_f)

        # up1 -> dec0
        self.u1 = FUpsample(int(ch1), int(base_ch))
        self.dec0_in = nn.Conv2d(int(base_ch + base_ch), int(base_ch), kernel_size=1)
        self.dec0 = DWResBlock2D(int(base_ch), k_t=k_t, k_f=k_f)

        # out residual RI
        self.out = nn.Conv2d(int(base_ch), 2, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

        # gain starts near 0 (sigmoid(-6) ~ 0.0025)
        self._gain = nn.Parameter(torch.tensor(-6.0))

    def forward(self, d1: Tensor, y_ccm: Tensor) -> Tensor:
        # d1: (B,64,T,F)
        # y_ccm: (B,F,T,2) -> (B,2,T,F)
        h = y_ccm.permute(0, 3, 2, 1).contiguous()
        if self.detach_hint:
            h = h.detach()

        x = torch.cat([d1, h], dim=1)  # (B,66,T,F)
        x = self.in_proj(x)

        s0 = self.e0(x)          # (B,base,T,F)

        x1 = self.d1(s0)         # (B,ch1,T,F/2)
        s1 = self.e1(x1)         # (B,ch1,T,F/2)

        x2 = self.d2(s1)         # (B,ch2,T,F/4)
        s2 = self.e2(x2)         # (B,ch2,T,F/4)

        m = self.mid(s2)         # (B,ch2,T,F/4)

        # up to F/2, concat skip s1
        u1 = self.u2(m, target_F=s1.shape[-1])               # (B,ch1,T,F/2)
        d1 = torch.cat([u1, s1], dim=1)                      # (B,2*ch1,T,F/2)
        d1 = self.dec1_in(d1)
        d1 = self.dec1(d1)

        # up to F, concat skip s0
        u0 = self.u1(d1, target_F=s0.shape[-1])              # (B,base,T,F)
        d0 = torch.cat([u0, s0], dim=1)                      # (B,2*base,T,F)
        d0 = self.dec0_in(d0)
        d0 = self.dec0(d0)

        r = self.out(d0)  # (B,2,T,F)
        g = torch.sigmoid(self._gain)
        r = r * g

        # back to (B,F,T,2)
        r = r.permute(0, 3, 2, 1).contiguous()
        return r


# ---------------------------
# FFN (SwiGLU) on 2D map
# ---------------------------
class FFN2D(nn.Module):
    def __init__(self, channels: int, mult: int = 4, drop: float = 0.0):
        super().__init__()
        C = int(channels)
        H = int(C * mult)
        self.fc1 = nn.Conv2d(C, 2 * H, kernel_size=1)
        self.fc2 = nn.Conv2d(H, C, kernel_size=1)
        self.drop = nn.Dropout(float(drop))

    def forward(self, x: Tensor) -> Tensor:
        y = self.fc1(x)
        y = swiglu(y)    # -> (B,H,T,F)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return y


# ---------------------------
# TF-Conformer block: (FFN/2) + MHSA2D + Conv + (FFN/2)
# ---------------------------
class TFConformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int = 8,
        ff_mult: int = 4,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        conv_k_t: int = 7,
        conv_k_f: int = 3,
        layer_scale: float = 1e-2,
    ):
        super().__init__()
        C = int(channels)

        self.n1 = LayerNorm2D(C)
        self.ff1 = FFN2D(C, mult=ff_mult, drop=drop)

        self.n2 = LayerNorm2D(C)
        self.mhsa = MHSA2D(C, heads=heads, attn_drop=attn_drop, proj_drop=drop)

        self.n3 = LayerNorm2D(C)
        self.conv = ConvModule2D(C, k_t=conv_k_t, k_f=conv_k_f, drop=drop)

        self.n4 = LayerNorm2D(C)
        self.ff2 = FFN2D(C, mult=ff_mult, drop=drop)

        self.dp = DropPath(drop_path)

        # LayerScale (очень стабилизирует глубокие трансформеры для аудио)
        self.g1 = nn.Parameter(torch.ones(1, C, 1, 1) * layer_scale)
        self.g2 = nn.Parameter(torch.ones(1, C, 1, 1) * layer_scale)
        self.g3 = nn.Parameter(torch.ones(1, C, 1, 1) * layer_scale)
        self.g4 = nn.Parameter(torch.ones(1, C, 1, 1) * layer_scale)

    def forward(self, x: Tensor) -> Tensor:
        # FFN half
        x = x + 0.5 * self.dp(self.g1 * self.ff1(self.n1(x)))
        # MHSA
        x = x + self.dp(self.g2 * self.mhsa(self.n2(x)))
        # Conv module
        x = x + self.dp(self.g3 * self.conv(self.n3(x)))
        # FFN half
        x = x + 0.5 * self.dp(self.g4 * self.ff2(self.n4(x)))
        return x


# ---------------------------
# Trunk
# ---------------------------
class TFConformerTrunk(nn.Module):
    """
    Quality-first trunk for (B,128,T,F5):
      - explicit freq embedding
      - conv positional encoding
      - deep TF-Conformer blocks with full 2D attention
    """
    def __init__(
        self,
        channels: int = 128,
        depth: int = 8,
        heads: int = 8,
        ff_mult: int = 4,
        conv_k_t: int = 9,
        conv_k_f: int = 5,
        drop: float = 0.1,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
        max_freq_bins: int = 64,
    ):
        super().__init__()
        C = int(channels)
        self.max_freq_bins = int(max_freq_bins)

        # learnable freq embedding (лечит "F превратился в batch" проблему и даёт специализацию)
        self.f_emb = nn.Parameter(torch.zeros(1, C, 1, self.max_freq_bins))
        nn.init.normal_(self.f_emb, std=0.02)

        # conv positional encoding (локальная позиционка по T/F)
        self.pos = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C)

        blocks = []
        for i in range(int(depth)):
            # линейный рост drop_path по глубине
            dpr = drop_path * (i / max(1, depth - 1))
            blocks.append(
                TFConformerBlock(
                    channels=C,
                    heads=int(heads),
                    ff_mult=int(ff_mult),
                    attn_drop=float(attn_drop),
                    drop=float(drop),
                    drop_path=float(dpr),
                    conv_k_t=int(conv_k_t),
                    conv_k_f=int(conv_k_f),
                    layer_scale=1e-2,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.out_norm = LayerNorm2D(C)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B,C,T,F)
        B, C, T, Freq = x.shape
        if Freq > self.max_freq_bins:
            raise ValueError(f"Freq bins in trunk ({Freq}) > max_freq_bins ({self.max_freq_bins}). Increase max_freq_bins.")

        x = x + self.f_emb[..., :Freq]
        x = x + self.pos(x)

        x = self.blocks(x)
        x = self.out_norm(x)
        return x



def gn1(ch: int) -> nn.Module:
    return nn.GroupNorm(1, ch)


class HeadAdapterDWGLU(nn.Module):
    """
    Head adapter v2:
      pre -> (residual CBAM gate) -> (DWConv + GLU residual) -> out(27)

    Input:  (B,64,T,F)
    Output: (B,27,T,F)
    """
    def __init__(self, in_ch: int = 64, bottleneck: int = 48, spatial_ks: int = 7, dw_ks: int = 3, se_ratio: int = 4):
        super().__init__()
        bn = int(bottleneck)

        self.pre = nn.Conv2d(in_ch, bn, kernel_size=1)

        # ---------- CBAM-like attention (residual gating) ----------
        hid = max(1, bn // int(se_ratio))
        self.se1 = nn.Conv2d(bn, hid, kernel_size=1)
        self.se2 = nn.Conv2d(hid, bn, kernel_size=1)
        self.se_act = nn.ELU()

        ks = int(spatial_ks)
        pad = ks // 2
        self.sa = nn.Sequential(
            nn.ZeroPad2d([pad, pad, pad, pad]),
            nn.Conv2d(2, 1, kernel_size=ks),
        )

        # ---------- Local residual block: DWConv + GLU ----------
        k = int(dw_ks)
        p = k // 2
        self.dw = nn.Conv2d(bn, bn, kernel_size=k, padding=p, groups=bn)
        self.n1 = gn1(bn)

        # pointwise -> GLU
        self.pw_in = nn.Conv2d(bn, 2 * bn, kernel_size=1)
        self.glu = nn.GLU(dim=1)  # (B,2*bn,T,F) -> (B,bn,T,F)

        self.pw_out = nn.Conv2d(bn, bn, kernel_size=1)
        self.n2 = gn1(bn)

        self.act = nn.GELU()

        # output
        self.out = nn.Conv2d(bn, 27, kernel_size=1)

        # ---- init: zero-init out so early training is stable ----
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

        # (опционально) сделать residual-блок тоже близким к identity на старте:
        nn.init.zeros_(self.pw_out.weight)
        if self.pw_out.bias is not None:
            nn.init.zeros_(self.pw_out.bias)

    def _cbam_gate(self, x: Tensor) -> Tensor:
        # x: (B,bn,T,F)

        # channel gate
        w = x.mean(dim=(2, 3), keepdim=True)   # (B,bn,1,1)
        w = self.se_act(self.se1(w))
        w = torch.sigmoid(self.se2(w))         # (B,bn,1,1)

        # spatial gate
        a = x.mean(dim=1, keepdim=True)        # (B,1,T,F)
        m = x.amax(dim=1, keepdim=True)        # (B,1,T,F)
        s = torch.sigmoid(self.sa(torch.cat([a, m], dim=1)))  # (B,1,T,F)

        # bidirectional residual gating: scale in [1-alpha, 1+alpha]
        alpha = 0.5  # 0.25..1.0
        x = x * (1.0 + alpha * (2.0 * w - 1.0))
        x = x * (1.0 + alpha * (2.0 * s - 1.0))
        return x

    def forward(self, feat: Tensor) -> Tensor:
        # feat: (B,64,T,F)
        x = self.pre(feat)          # (B,bn,T,F)

        # attention first (как ты хочешь), но в residual форме
        x = self._cbam_gate(x)

        # DWConv+GLU residual block
        r = x
        y = self.dw(x)
        y = self.act(self.n1(y))
        y = self.pw_in(y)
        y = self.glu(y)
        y = self.pw_out(y)
        y = self.n2(y)

        x = r + y

        return self.out(x)          # (B,27,T,F)


class DeepVQEConditionalStemSeparator(nn.Module):
    """
    Variant B:
      encoders -> align/fuse (optional) -> en5 -> axial trunk shared -> decoder shared -> shared feat (64ch)
      then per-head:
        HeadAdapter(attn+bottleneck)->27 -> CCM -> (B,F,T,2)
    Output: (B, S_total, F, T, 2)
    """

    def __init__(
        self,
        n_fft: int = 1536,
        num_heads: int = 4,
        *,
        with_ref_head: bool = True,
        delay_past_frames: int = 25,
        delay_future_frames: int = 25,
        align_hidden: int = 64,
        trunk_layers: int = 5,
        trunk_heads: int = 4,
    ):
        super().__init__()
        self.n_fft = int(n_fft)

        self.num_heads_main = int(num_heads)
        self.with_ref_head = bool(with_ref_head)
        self.num_heads_total = self.num_heads_main + (1 if self.with_ref_head else 0)
        self.align_hidden = int(align_hidden)

        self.fe = FE()

        self.enblock1 = EncoderBlock(2, 64)
        self.enblock2 = EncoderBlock(64, 128)
        self.enblock3 = EncoderBlock(128, 128)
        self.enblock4 = EncoderBlock(128, 128)
        self.enblock5 = EncoderBlock(128, 128)

        self.align1 = AlignBlockBi(
            in_channels=64,
            hidden_channels=self.align_hidden,
            delay_past=delay_past_frames,
            delay_future=delay_future_frames,
        )
        self.fuse1 = nn.Conv2d(64 + self.align_hidden, 64, kernel_size=1)

        # dynamic F5
        F_in = self.n_fft // 2 + 1
        with torch.no_grad():
            dummy = torch.zeros(1, 2, 8, F_in)
            y = self.enblock5(self.enblock4(self.enblock3(self.enblock2(self.enblock1(dummy)))))
            self.F5 = int(y.shape[-1])

        self.trunk = TFConformerTrunk(
            channels=128,
            depth=12,
            heads=8,
            ff_mult=4,
            conv_k_t=9,
            conv_k_f=5,
            drop=0.1,
            attn_drop=0.0,
            drop_path=0.1,
            max_freq_bins=self.F5,
        )

        S = self.num_heads_total
        self.decoders = nn.ModuleList([DecoderChain() for _ in range(S)])
        self.adapters = nn.ModuleList([HeadAdapterDWGLU(in_ch=64, bottleneck=64) for _ in range(S)])
        self.ccms = nn.ModuleList([CCM() for _ in range(S)])

        # residual post-filter per head: U-Net gets (d1_i, y_ccm_i) and predicts RI residual
        self.residual_unets = nn.ModuleList(
            [DepthwiseTFUNet2F(d1_ch=64, hint_ch=2, base_ch=96, ch1=128, ch2=160, k_t=5, k_f=3, detach_hint=True)
             for _ in range(S)]
        )

    def forward(self, mix_ri: Tensor, ref_ri: Tensor, ref_valid: Tensor | None = None) -> Tensor:
        # mix_ri/ref_ri: (B,F,T,2)
        B = mix_ri.shape[0]

        if ref_valid is None:
            ref_valid = mix_ri.new_ones((B,), dtype=torch.bool)
        else:
            ref_valid = ref_valid.to(device=mix_ri.device).bool()

        x0 = self.fe(mix_ri)
        r0 = self.fe(ref_ri)

        x1 = self.enblock1(x0)
        r1 = self.enblock1(r0)

        ctx = self.align1(x1, r1, return_att=False)  # (B,align_hidden,T,F')
        mask = ref_valid.float().view(B, 1, 1, 1)  # broadcast
        ctx = ctx * mask  # <- ключевой момент: uncond не даёт градиент в align

        x1f = self.fuse1(torch.cat([x1, ctx], dim=1))

        en2 = self.enblock2(x1f)
        en3 = self.enblock3(en2)
        en4 = self.enblock4(en3)
        en5 = self.enblock5(en4)

        z = self.trunk(en5)  # (B,128,T,F5)

        outs = []
        for i in range(self.num_heads_total):
            d1_i = self.decoders[i](z, en5, en4, en3, en2, x1f, x0)

            m = self.adapters[i](d1_i)
            m = 0.5 * torch.tanh(m)
            y_ccm = self.ccms[i](m, mix_ri)  # (B,F,T,2)

            r = self.residual_unets[i](d1_i, y_ccm)  # (B,F,T,2)

            y = y_ccm + r
            outs.append(y)

        return torch.stack(outs, dim=1)

