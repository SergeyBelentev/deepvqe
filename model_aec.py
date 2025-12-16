import torch
import torch.nn as nn
from torch import Tensor


class DeepVQE_AEC(nn.Module):
    """
    Two-input DeepVQE for ref-conditioned cancellation:
      mic: (B,F,T,2)
      ref: (B,F,T,2)
      out: (B,F,T,2)  # enhanced mic STFT => dub-only
    """

    def __init__(self, delay_frames: int = 80, align_hidden: int = 64) -> None:
        super().__init__()
        self.fe = FE()

        # shared encoders
        self.enblock1 = EncoderBlock(2, 64)
        self.enblock2 = EncoderBlock(64, 128)
        self.enblock3 = EncoderBlock(128, 128)
        self.enblock4 = EncoderBlock(128, 128)
        self.enblock5 = EncoderBlock(128, 128)

        self.align1 = AlignBlock(in_channels=64, hidden_channels=align_hidden, delay=delay_frames)
        self.fuse1 = nn.Conv2d(64 + align_hidden, 64, kernel_size=1)

        self.bottle = Bottleneck(128 * 9, 64 * 9)

        self.deblock5 = DecoderBlock(128, 128)
        self.deblock4 = DecoderBlock(128, 128)
        self.deblock3 = DecoderBlock(128, 128)
        self.deblock2 = DecoderBlock(128, 64)
        self.deblock1 = DecoderBlock(64, 27)
        self.ccm = CCM()

    def forward(self, mic: Tensor, ref: Tensor) -> Tensor:
        # mic/ref: (B,F,T,2)

        mic0 = self.fe(mic)   # (B,2,T,F)
        ref0 = self.fe(ref)   # (B,2,T,F)

        mic1 = self.enblock1(mic0)  # (B,64,T,F')
        ref1 = self.enblock1(ref0)  # shared weights

        ref1a = self.align1(mic1, ref1)  # (B,align_hidden,T,F')
        mic1f = self.fuse1(torch.cat([mic1, ref1a], dim=1))  # (B,64,T,F')

        en2 = self.enblock2(mic1f)
        en3 = self.enblock3(en2)
        en4 = self.enblock4(en3)
        en5 = self.enblock5(en4)

        z = self.bottle(en5)

        d5 = self.deblock5(z, en5)[..., : en4.shape[-1]]
        d4 = self.deblock4(d5, en4)[..., : en3.shape[-1]]
        d3 = self.deblock3(d4, en3)[..., : en2.shape[-1]]
        d2 = self.deblock2(d3, en2)[..., : mic1f.shape[-1]]
        d1 = self.deblock1(d2, mic1f)[..., : mic0.shape[-1]]

        out = self.ccm(d1, mic)  # (B,F,T,2)
        return out
