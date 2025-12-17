from __future__ import annotations
from typing import Final
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
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
    
        
class AlignBlock(nn.Module):
    """Delay-aware alignment block used by the original DeepVQE paper."""

    def __init__(self, in_channels: int, hidden_channels: int, delay: int = 100) -> None:
        super().__init__()
        self.pconv_mic = nn.Conv2d(in_channels, hidden_channels, 1)
        self.pconv_ref = nn.Conv2d(in_channels, hidden_channels, 1)
        self.pconv_val = nn.Conv2d(in_channels, hidden_channels, 1)
        self.unfold = nn.Sequential(
            nn.ZeroPad2d([0, 0, delay - 1, 0]),
            nn.Unfold((delay, 1)),
        )
        self.conv = nn.Sequential(nn.ZeroPad2d([1, 1, 4, 0]), nn.Conv2d(hidden_channels, 1, (5, 3)))

    def forward(self, x_mic: Tensor, x_ref: Tensor) -> Tensor:
        """Align reference signal to microphone input.

        Args:
            x_mic: Tensor with shape ``(B, C, T, F)``.
            x_ref: Tensor with shape ``(B, C, T, F)``.

        Returns:
            Tensor with aligned reference of shape ``(B, H, T, F)``.
        """

        q_proj = self.pconv_mic(x_mic)  # (B, H, T, F)
        k_proj = self.pconv_ref(x_ref)  # (B, H, T, F)
        v_proj = self.pconv_val(x_ref)  # (B,H,T,F)
        k_unfold = self.unfold(k_proj)
        k_unfold = k_unfold.view(k_proj.shape[0], k_proj.shape[1], -1, k_proj.shape[2], k_proj.shape[3])
        k_unfold = k_unfold.permute(0, 1, 3, 2, 4)

        att_logits = torch.sum(q_proj.unsqueeze(-2) * k_unfold, dim=-1)  # (B, H, T, D)
        att_logits = self.conv(att_logits)  # (B, 1, T, D)
        att = torch.softmax(att_logits, dim=-1)[..., None]  # (B, 1, T, D, 1)

        v_ctx = self.unfold(v_proj)
        v_ctx = v_ctx.view(v_proj.shape[0], v_proj.shape[1], -1, v_proj.shape[2], v_proj.shape[3])
        v_ctx = v_ctx.permute(0, 1, 3, 2, 4)
        return torch.sum(v_ctx * att, dim=-2)  # (B,H,T,F)



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
        n_fft: int = 1536,          # <-- важно: под 48k fullband
        delay_frames: int = 80,
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

        self.align1 = AlignBlock(in_channels=64, hidden_channels=align_hidden, delay=delay_frames)
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
        self.deblock1 = DecoderBlock(64, 27)
        self.ccm = CCM()

    def forward(self, mic: Tensor, ref: Tensor) -> Tensor:
        mic0 = self.fe(mic)   # (B,2,T,F)
        ref0 = self.fe(ref)

        mic1 = self.enblock1(mic0)  # (B,64,T,F')
        ref1 = self.enblock1(ref0)

        ref1a = self.align1(mic1, ref1)                 # (B,align_hidden,T,F')
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

        bg = self.ccm(d1, mic)
        out = mic - bg

        # удобно для train: вернуть и out и bg
        if isinstance(getattr(self, "_return_bg", False), bool) and self._return_bg:
            return out, bg
        return out



