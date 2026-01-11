from __future__ import annotations
from typing import Final
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor


FLOAT_EPS: Final[float] = torch.finfo(torch.float32).eps


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
        self.bn = nn.BatchNorm2d(channels)
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
        self.bn = nn.BatchNorm2d(out_channels)
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
        self.bn = nn.BatchNorm2d(out_channels)
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


class DeepVQE(nn.Module):
    """
    Two-input DeepVQE for ref-conditioned cancellation:
      mic: (B,F,T,2)
      ref: (B,F,T,2)
      out: (B,F,T,2)
    """

    def set_return_bg(self, flag: bool = True) -> None:
        self._return_bg = flag

    def __init__(
        self,
        n_fft: int = 1536,
        delay_past_frames: int = 25,
        delay_future_frames: int = 25,
        align_hidden: int = 64,
    ) -> None:
        super().__init__()
        self._return_bg = False
        self.n_fft = n_fft
        self.fe = FE()

        # shared encoders
        self.enblock1 = EncoderBlock(2, 64)
        self.enblock2 = EncoderBlock(64, 128)
        self.enblock3 = EncoderBlock(128, 128)
        self.enblock4 = EncoderBlock(128, 128)
        self.enblock5 = EncoderBlock(128, 128)

        self.align1 = AlignBlockBi(
            in_channels=64,
            hidden_channels=align_hidden,
            delay_past=delay_past_frames,
            delay_future=delay_future_frames
        )
        self.fuse1 = nn.Conv2d(64 + align_hidden, 64, kernel_size=1)

        # ---- dynamic F5 computation ----
        F_in = n_fft // 2 + 1
        with torch.no_grad():
            dummy = torch.zeros(1, 2, 8, F_in)  # (B,2,T,F)
            y = self.enblock1(dummy)
            y = self.enblock2(y)
            y = self.enblock3(y)
            y = self.enblock4(y)
            y = self.enblock5(y)
            F5 = y.shape[-1]
        self.F5 = F5
        # -------------------------------

        self.bottle = Bottleneck(128 * F5, 64 * F5)

        self.deblock5 = DecoderBlock(128, 128)
        self.deblock4 = DecoderBlock(128, 128)
        self.deblock3 = DecoderBlock(128, 128)
        self.deblock2 = DecoderBlock(128, 64)
        self.deblock1 = DecoderBlock(64, 27, is_last=True)
        self.ccm = CCM()

    def _align_ref_ri(self, ref_ri: Tensor, att: Tensor) -> Tensor:
        """
        ref_ri: (B,F,T,2)
        att:    (B,1,T,K)
        out:    aligned ref_ri (B,F,T,2)

        Без unfold(): weighted sum of shifted ref_ri по лагам, block-wise.
        """
        B, Freq, T, _ = ref_ri.shape
        K = self.align1.K
        p = self.align1.delay_past
        q = self.align1.delay_future

        # (B,2,T,F)
        r = ref_ri.permute(0, 3, 2, 1).contiguous()
        r_pad = nn.functional.pad(r, (0, 0, p, q))  # (B,2,T+p+q,F)

        kb = max(1, min(getattr(self.align1, "k_block", 8), K))

        # копим в fp32 (даже если когда-то включишь AMP)
        r_for = r_pad.float()
        att_for = att.float()
        aligned = r_for.new_zeros((B, 2, T, Freq), dtype=torch.float32)

        for s in range(0, K, kb):
            e = min(K, s + kb)
            r_blk = torch.stack(
                [r_for[:, :, (s + i):(s + i + T), :] for i in range(e - s)],
                dim=3,  # (B,2,T,kb,F)
            )
            w_blk = att_for[:, 0, :, s:e]  # (B,T,kb)
            aligned = aligned + (r_blk * w_blk[:, None, :, :, None]).sum(dim=3)

        # back to (B,F,T,2)
        return aligned.to(ref_ri.dtype).permute(0, 3, 2, 1).contiguous()

    def forward(self, mic: Tensor, ref: Tensor) -> Tensor:
        mic0 = self.fe(mic)   # (B,2,T,F)
        ref0 = self.fe(ref)

        mic1 = self.enblock1(mic0)  # (B,64,T,F')
        ref1 = self.enblock1(ref0)

        ref1a, att = self.align1(mic1, ref1, return_att=True)       # (B,align_hidden,T,F')
        ref_ri_aligned = self._align_ref_ri(ref, att)
        mic1f = self.fuse1(torch.cat([mic1, ref1a], 1)) # (B,64,T,F')

        en2 = self.enblock2(mic1f)
        en3 = self.enblock3(en2)
        en4 = self.enblock4(en3)
        en5 = self.enblock5(en4)

        z = self.bottle(en5)

        d5 = self.deblock5(z, en5)[..., :en4.shape[-1]]
        d4 = self.deblock4(d5, en4)[..., :en3.shape[-1]]
        d3 = self.deblock3(d4, en3)[..., :en2.shape[-1]]
        d2 = self.deblock2(d3, en2)[..., :mic1f.shape[-1]]
        d1 = self.deblock1(d2, mic1f)[..., :mic0.shape[-1]]

        bg = self.ccm(d1, ref_ri_aligned)
        out = mic - bg

        # удобно для train: вернуть и out и bg
        if isinstance(getattr(self, "_return_bg", False), bool) and self._return_bg:
            return out, bg
        return out



class DeepVQEStemSeparator(nn.Module):
    """
    Unconditional separator (Phase A/B) on top of DeepVQE backbone.
    Uses the same FE/Encoder/Decoder + CCM, but applies per-head CCM to MIX.

    Input:  mix_ri (B,F,T,2)
    Output: stems_ri (B,S,F,T,2) where S = num_heads
    """

    def __init__(self, n_fft: int = 1536, num_heads: int = 6):
        super().__init__()
        self.n_fft = n_fft
        self.num_heads = int(num_heads)

        self.fe = FE()

        self.enblock1 = EncoderBlock(2, 64)
        self.enblock2 = EncoderBlock(64, 128)
        self.enblock3 = EncoderBlock(128, 128)
        self.enblock4 = EncoderBlock(128, 128)
        self.enblock5 = EncoderBlock(128, 128)

        # dynamic F5 like in your DeepVQE
        F_in = n_fft // 2 + 1
        with torch.no_grad():
            dummy = torch.zeros(1, 2, 8, F_in)
            y = self.enblock5(self.enblock4(self.enblock3(self.enblock2(self.enblock1(dummy)))))
            self.F5 = y.shape[-1]

        self.bottle = Bottleneck(128 * self.F5, 64 * self.F5)

        self.deblock5 = DecoderBlock(128, 128)
        self.deblock4 = DecoderBlock(128, 128)
        self.deblock3 = DecoderBlock(128, 128)
        self.deblock2 = DecoderBlock(128, 64)
        self.deblock1 = DecoderBlock(64, 27, is_last=True)

        self.ccm = CCM()

        # per-head CCM masks: 27 channels per head
        self.head = nn.Conv2d(27, 27 * self.num_heads, kernel_size=1)

    def forward(self, mix_ri: Tensor) -> Tensor:
        # mix_ri: (B,F,T,2)
        x0 = self.fe(mix_ri)            # (B,2,T,F)
        x1 = self.enblock1(x0)          # (B,64,T,F')
        en2 = self.enblock2(x1)
        en3 = self.enblock3(en2)
        en4 = self.enblock4(en3)
        en5 = self.enblock5(en4)

        z = self.bottle(en5)

        d5 = self.deblock5(z, en5)[..., :en4.shape[-1]]
        d4 = self.deblock4(d5, en4)[..., :en3.shape[-1]]
        d3 = self.deblock3(d4, en3)[..., :en2.shape[-1]]
        d2 = self.deblock2(d3, en2)[..., :x1.shape[-1]]
        d1 = self.deblock1(d2, x1)[..., :x0.shape[-1]]   # (B,27,T,F)

        m = self.head(d1)  # (B,27*S,T,F)

        # Apply CCM per head: batch-flatten heads
        B, _, T, Freq = m.shape
        S = self.num_heads

        m2 = rearrange(m, "b (s c) t f -> (b s) c t f", s=S)  # (B*S,27,T,F)
        x2 = mix_ri.repeat_interleave(S, dim=0)               # (B*S,F,T,2)

        y2 = self.ccm(m2, x2)  # (B*S,F,T,2)
        y = rearrange(y2, "(b s) f t r -> b s f t r", b=B, s=S)
        return y


class DeepVQEConditionalStemSeparator(nn.Module):
    """
    Conditional separator:
      input:  mix_ri (B,F,T,2), ref_ri (B,F,T,2)
      output: stems_ri (B,S,F,T,2)
    """

    def __init__(
        self,
        n_fft: int = 1536,
        num_heads: int = 4,
        *,
        delay_past_frames: int = 25,
        delay_future_frames: int = 25,
        align_hidden: int = 64,
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.num_heads = int(num_heads)

        self.fe = FE()

        self.enblock1 = EncoderBlock(2, 64)
        self.enblock2 = EncoderBlock(64, 128)
        self.enblock3 = EncoderBlock(128, 128)
        self.enblock4 = EncoderBlock(128, 128)
        self.enblock5 = EncoderBlock(128, 128)

        self.align1 = AlignBlockBi(
            in_channels=64,
            hidden_channels=align_hidden,
            delay_past=delay_past_frames,
            delay_future=delay_future_frames,
        )
        self.fuse1 = nn.Conv2d(64 + align_hidden, 64, kernel_size=1)

        # dynamic F5 (так же, как у тебя)
        F_in = self.n_fft // 2 + 1
        with torch.no_grad():
            dummy = torch.zeros(1, 2, 8, F_in)
            y = self.enblock5(self.enblock4(self.enblock3(self.enblock2(self.enblock1(dummy)))))
            self.F5 = y.shape[-1]

        self.bottle = Bottleneck(128 * self.F5, 64 * self.F5)

        self.deblock5 = DecoderBlock(128, 128)
        self.deblock4 = DecoderBlock(128, 128)
        self.deblock3 = DecoderBlock(128, 128)
        self.deblock2 = DecoderBlock(128, 64)
        self.deblock1 = DecoderBlock(64, 27, is_last=True)

        self.ccm = CCM()

        # per-head CCM masks: 27 channels per head
        self.head = nn.Conv2d(27, 27 * self.num_heads, kernel_size=1)

    def forward(self, mix_ri: Tensor, ref_ri: Tensor) -> Tensor:
        # mix_ri/ref_ri: (B,F,T,2)
        if ref_ri is None:
            ref_ri = torch.zeros_like(mix_ri)

        x0 = self.fe(mix_ri)  # (B,2,T,F)
        r0 = self.fe(ref_ri)  # (B,2,T,F)

        x1 = self.enblock1(x0)  # (B,64,T,F')
        r1 = self.enblock1(r0)  # (B,64,T,F')

        # ctx: (B,align_hidden,T,F')
        ctx = self.align1(x1, r1, return_att=False)
        x1f = self.fuse1(torch.cat([x1, ctx], dim=1))  # (B,64,T,F')

        en2 = self.enblock2(x1f)
        en3 = self.enblock3(en2)
        en4 = self.enblock4(en3)
        en5 = self.enblock5(en4)

        z = self.bottle(en5)

        d5 = self.deblock5(z, en5)[..., :en4.shape[-1]]
        d4 = self.deblock4(d5, en4)[..., :en3.shape[-1]]
        d3 = self.deblock3(d4, en3)[..., :en2.shape[-1]]
        d2 = self.deblock2(d3, en2)[..., :x1f.shape[-1]]
        d1 = self.deblock1(d2, x1f)[..., :x0.shape[-1]]   # (B,27,T,F)

        m = self.head(d1)  # (B,27*S,T,F)
        B, _, Tt, Freq = m.shape
        S = self.num_heads

        m2 = rearrange(m, "b (s c) t f -> (b s) c t f", s=S)     # (B*S,27,T,F)
        x2 = mix_ri.repeat_interleave(S, dim=0)                  # (B*S,F,T,2)
        y2 = self.ccm(m2, x2)                                    # (B*S,F,T,2)
        y  = rearrange(y2, "(b s) f t r -> b s f t r", b=B, s=S) # (B,S,F,T,2)
        return y
