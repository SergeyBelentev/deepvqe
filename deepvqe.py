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
    Bidirectional delay-aware alignment (memory-efficient):

      For each (t,f) in mic attends to ref frames in window:
        [t - delay_past .. t + delay_future]
      K = delay_past + delay_future + 1

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
    ) -> None:
        super().__init__()
        if delay_past < 0 or delay_future < 0:
            raise ValueError("delay_past/delay_future must be >= 0")

        self.delay_past = int(delay_past)
        self.delay_future = int(delay_future)
        self.K = self.delay_past + self.delay_future + 1  # num candidate lags

        self.pconv_mic = nn.Conv2d(in_channels, hidden_channels, 1)
        self.pconv_ref = nn.Conv2d(in_channels, hidden_channels, 1)
        self.pconv_val = nn.Conv2d(in_channels, hidden_channels, 1)

        # Сглаживание логитов по (T, K) симметричное.
        if (logit_kernel_t % 2) != 1 or (logit_kernel_k % 2) != 1:
            raise ValueError("logit_kernel_t/logit_kernel_k must be odd for symmetric padding.")

        pad_t = logit_kernel_t // 2
        pad_k = logit_kernel_k // 2
        self.logit_smoother = nn.Sequential(
            nn.ZeroPad2d([pad_k, pad_k, pad_t, pad_t]),  # (left,right,top,bottom) for (K,T)
            nn.Conv2d(hidden_channels, 1, (logit_kernel_t, logit_kernel_k)),
        )

    # (если где-то в коде у тебя осталось .D — оставим алиас)
    @property
    def D(self) -> int:
        return self.K

    def _time_windows(self, x: Tensor) -> Tensor:
        """
        x: (B,H,T,F)
        returns: (B,H,T,K,F) view

        ВАЖНО: Tensor.unfold по dimension=2 (time) вернёт (B,H,T,F,K),
        поэтому делаем permute -> (B,H,T,K,F).
        """
        # pad only time dimension
        x_pad = nn.functional.pad(x, (0, 0, self.delay_past, self.delay_future))  # (B,H,T+past+future,F)

        # unfold over time => (B,H,T,F,K)
        x_win = x_pad.unfold(dimension=2, size=self.K, step=1)

        # reorder => (B,H,T,K,F)
        return x_win.permute(0, 1, 2, 4, 3)

    def forward(self, x_mic: Tensor, x_ref: Tensor, return_att: bool = False):
        # Projections
        q = self.pconv_mic(x_mic)  # (B,H,T,F)
        k = self.pconv_ref(x_ref)  # (B,H,T,F)
        v = self.pconv_val(x_ref)  # (B,H,T,F)

        # Windows (view)
        k_win = self._time_windows(k)  # (B,H,T,K,F)
        v_win = self._time_windows(v)  # (B,H,T,K,F)

        # logits: (B,H,T,K)  via matmul over F
        # (B,H,T,K,F) @ (B,H,T,F,1) -> (B,H,T,K,1) -> (B,H,T,K)
        att_logits = torch.matmul(k_win, q.unsqueeze(-1)).squeeze(-1)

        # smooth + collapse H->1
        att_logits = self.logit_smoother(att_logits)      # (B,1,T,K)
        att = torch.softmax(att_logits, dim=-1)           # (B,1,T,K)

        # ctx: weighted sum over K
        # (B,1,T,1,K) @ (B,H,T,K,F) -> (B,H,T,1,F) -> (B,H,T,F)
        ctx = torch.matmul(att.unsqueeze(-2), v_win).squeeze(-2)

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
        Align full-resolution complex ref by attention computed on encoder features.

        ref_ri: (B,F,T,2)
        att:    (B,1,T,K)  where K = delay_past + delay_future + 1
        returns aligned_ref_ri: (B,F,T,2)
        """
        B, Freq, T, _ = ref_ri.shape
        K = self.align1.K
        p = self.align1.delay_past
        q = self.align1.delay_future

        # (B,F,T,2) -> (B,2,T,F)
        r = ref_ri.permute(0, 3, 2, 1).contiguous()

        # pad time, then unfold over time:
        # r_pad: (B,2,T+p+q,F)
        r_pad = nn.functional.pad(r, (0, 0, p, q))

        # unfold => (B,2,T,F,K)  !!! K last
        r_win = r_pad.unfold(dimension=2, size=K, step=1)

        # reorder => (B,2,T,K,F)
        r_win = r_win.permute(0, 1, 2, 4, 3)

        # weighted sum over K:
        # (B,1,T,1,K) @ (B,2,T,K,F) -> (B,2,T,1,F) -> (B,2,T,F)
        aligned = torch.matmul(att.unsqueeze(-2), r_win).squeeze(-2)

        # back to (B,F,T,2)
        return aligned.permute(0, 3, 2, 1).contiguous()

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



