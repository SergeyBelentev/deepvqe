from __future__ import annotations
import argparse
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from deepvqe import DeepVQE


# -----------------------
# audio io (stereo)
# -----------------------
def load_wav_stereo(path: str, sr_expected: int) -> torch.Tensor:
    x, sr = sf.read(path, dtype="float32", always_2d=True)  # (T,C)
    if sr != sr_expected:
        raise RuntimeError(f"SR mismatch for {path}: got {sr}, expected {sr_expected}")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    else:
        x = x[:, :2]
    return torch.from_numpy(x).transpose(0, 1).contiguous()  # (2,T)


def save_wav_stereo(
    path: str,
    y: torch.Tensor,
    sr: int,
    *,
    subtype: str = "PCM_24",
    normalize_if_clip: bool = True,
    peak_target: float = 0.99,
) -> None:
    """
    y: (2,T) float tensor (обычно fp32)
    Пишем без клиппинга. Если пик > 1.0 — мягко нормализуем весь сигнал.
    """
    y = y.detach().cpu().float()

    # (2,T) -> (T,2)
    y = y.transpose(0, 1).contiguous()

    peak = float(y.abs().max().item())
    if normalize_if_clip and peak > 1.0:
        scale = peak_target / peak
        y = y * scale
        print(f"[warn] output peak {peak:.3f} > 1.0 -> scaled by {scale:.6f} to peak~{peak_target}")

    sf.write(path, y.numpy(), sr, subtype=subtype)


# -----------------------
# metrics
# -----------------------
def si_sdr_1d(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    ref_zm = ref - ref.mean()
    est_zm = est - est.mean()
    s_target = (torch.dot(est_zm, ref_zm) / (torch.dot(ref_zm, ref_zm) + eps)) * ref_zm
    e_noise = est_zm - s_target
    ratio = (torch.dot(s_target, s_target) + eps) / (torch.dot(e_noise, e_noise) + eps)
    return float(10.0 * torch.log10(ratio))


# -----------------------
# STFT helper (fp32)
# -----------------------
class STFT(torch.nn.Module):
    def __init__(self, n_fft: int, hop: int, win: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.register_buffer("window", torch.hann_window(win, dtype=torch.float32), persistent=False)

    def stft_ri(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T) fp32
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self.window.to(x.device),
            return_complex=True,
        )
        return torch.view_as_real(X).to(torch.float32)

    def istft_ri(self, X_ri: torch.Tensor, length: int) -> torch.Tensor:
        X = torch.complex(X_ri[..., 0], X_ri[..., 1])
        y = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self.window.to(X.device),
            length=length,
        )
        return y.to(torch.float32)


# -----------------------
# overlap-add helpers
# -----------------------
def make_fade_window(chunk_len: int, fade_len: int, device: torch.device) -> torch.Tensor:
    """
    1D window for overlap-add: linear fade in/out.
    Returns (chunk_len,) float32.
    """
    w = torch.ones(chunk_len, device=device, dtype=torch.float32)
    if fade_len <= 0:
        return w
    fade_len = min(fade_len, chunk_len // 2)
    ramp = torch.linspace(0.0, 1.0, fade_len, device=device, dtype=torch.float32)
    w[:fade_len] = ramp
    w[-fade_len:] = ramp.flip(0)
    return w


@torch.no_grad()
def run_one_chunk(
    model: torch.nn.Module,
    stft_mod: STFT,
    mix_chunk: torch.Tensor,
    ref_chunk: torch.Tensor,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    """
    mix_chunk/ref_chunk: (2,Tc) fp32 on device
    returns out_chunk: (2,Tc) fp32 on device
    """
    if mix_chunk.ndim != 2 or ref_chunk.ndim != 2:
        raise RuntimeError(f"Expected (2,T) chunks, got mix={tuple(mix_chunk.shape)} ref={tuple(ref_chunk.shape)}")
    if mix_chunk.shape[0] != 2 or ref_chunk.shape[0] != 2:
        raise RuntimeError(f"Expected stereo (2,T), got mix={tuple(mix_chunk.shape)} ref={tuple(ref_chunk.shape)}")
    if mix_chunk.shape[1] != ref_chunk.shape[1]:
        raise RuntimeError(f"Chunk length mismatch mix={mix_chunk.shape[1]} ref={ref_chunk.shape[1]}")

    C, Tc = mix_chunk.shape

    # treat channels as batch
    mix_f = mix_chunk.reshape(C, Tc).to(torch.float32)  # (2,Tc)
    ref_f = ref_chunk.reshape(C, Tc).to(torch.float32)

    # STFT always fp32
    mix_ri = stft_mod.stft_ri(mix_f)  # (2,F,Tf,2)
    ref_ri = stft_mod.stft_ri(ref_f)

    # model forward maybe in bf16/fp16 to save VRAM
    with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
        out_ri = model(mix_ri, ref_ri)

    # If model returns (out,bg), take out
    if isinstance(out_ri, (tuple, list)) and len(out_ri) == 2:
        out_ri = out_ri[0]

    # ISTFT fp32
    out_wav = stft_mod.istft_ri(out_ri.to(torch.float32), length=Tc)  # (2,Tc)
    return out_wav


def _match_lengths(
    mix: torch.Tensor,
    ref: torch.Tensor,
    mode: Literal["crop", "pad"],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    mix/ref: (2,Tm)/(2,Tr)
    crop -> both truncated to min length
    pad  -> both padded with zeros to max length
    """
    if mix.ndim != 2 or ref.ndim != 2:
        raise RuntimeError(f"Expected (2,T) tensors, got mix={tuple(mix.shape)} ref={tuple(ref.shape)}")
    if mix.shape[0] != 2 or ref.shape[0] != 2:
        raise RuntimeError(f"Expected stereo (2,T), got mix={tuple(mix.shape)} ref={tuple(ref.shape)}")

    Tm = int(mix.shape[1])
    Tr = int(ref.shape[1])
    if Tm == Tr:
        return mix, ref

    if mode == "crop":
        T = min(Tm, Tr)
        mix = mix[:, :T]
        ref = ref[:, :T]
        return mix, ref

    if mode == "pad":
        T = max(Tm, Tr)
        if Tm < T:
            mix = F.pad(mix, (0, T - Tm))
        if Tr < T:
            ref = F.pad(ref, (0, T - Tr))
        return mix, ref

    raise ValueError(f"Unknown length mode: {mode}")


def _pad_or_trim_to_chunk(x: torch.Tensor, chunk_len: int) -> torch.Tensor:
    """
    x: (2,T)
    returns (2,chunk_len)
    """
    T = int(x.shape[1])
    if T < chunk_len:
        return F.pad(x, (0, chunk_len - T))
    if T > chunk_len:
        return x[:, :chunk_len]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default="pred.wav")
    ap.add_argument("--target", default=None)

    ap.add_argument("--device", default="cuda")

    # chunking
    ap.add_argument("--chunk-sec", type=float, default=8.0)
    ap.add_argument("--overlap-sec", type=float, default=2.0)

    # length handling
    ap.add_argument(
        "--length-mode",
        choices=["crop", "pad"],
        default="crop",
        help="How to handle mix/ref length mismatch: crop to min length or pad to max length",
    )

    # amp for inference (saves VRAM; bf16 usually ok on 4080)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")

    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) or {}

    sr = int(ckpt_args.get("sr", 48000))
    n_fft = int(ckpt_args.get("n_fft", 1536))
    hop = int(ckpt_args.get("hop", 480))
    win = int(ckpt_args.get("win", 1536))
    delay_frames = int(ckpt_args.get("delay_frames", 25))
    align_hidden = int(ckpt_args.get("align_hidden", 64))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print(f"stft: n_fft={n_fft} hop={hop} win={win} | model: delay_frames={delay_frames} align_hidden={align_hidden}")

    model = DeepVQE(n_fft=n_fft, delay_frames=delay_frames, align_hidden=align_hidden).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    stft_mod = STFT(n_fft=n_fft, hop=hop, win=win).to(device)

    mix = load_wav_stereo(args.mix, sr).to(device)  # (2,Tm)
    ref = load_wav_stereo(args.ref, sr).to(device)  # (2,Tr)

    if mix.shape[1] != ref.shape[1]:
        print(f"[warn] length mismatch: mix={mix.shape[1]} ref={ref.shape[1]} | mode={args.length_mode}")
    mix, ref = _match_lengths(mix, ref, mode=args.length_mode)

    C, T = mix.shape
    assert C == 2

    chunk_len = int(round(args.chunk_sec * sr))
    overlap_len = int(round(args.overlap_sec * sr))
    if overlap_len >= chunk_len:
        raise RuntimeError("overlap-sec must be smaller than chunk-sec")
    if chunk_len <= 0:
        raise RuntimeError("chunk-sec too small")
    if overlap_len < 0:
        raise RuntimeError("overlap-sec must be >= 0")

    step = chunk_len - overlap_len
    fade = make_fade_window(chunk_len, overlap_len, device=device)  # (chunk_len,)

    out = torch.zeros((2, T), device=device, dtype=torch.float32)
    wsum = torch.zeros((T,), device=device, dtype=torch.float32)

    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    pos = 0
    while pos < T:
        end = pos + chunk_len
        mix_chunk = mix[:, pos:end]
        ref_chunk = ref[:, pos:end]

        # ensure EXACT chunk_len for both
        mix_chunk = _pad_or_trim_to_chunk(mix_chunk, chunk_len)
        ref_chunk = _pad_or_trim_to_chunk(ref_chunk, chunk_len)

        out_chunk = run_one_chunk(model, stft_mod, mix_chunk, ref_chunk, use_amp, amp_dtype)  # (2,chunk_len)

        valid_len = min(chunk_len, T - pos)
        out[:, pos : pos + valid_len] += out_chunk[:, :valid_len] * fade[:valid_len]
        wsum[pos : pos + valid_len] += fade[:valid_len]

        pos += step

    out = out / (wsum.clamp_min(1e-8)[None, :])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_wav_stereo(args.out, out, sr, subtype='FLOAT')
    print("saved:", args.out)

    if args.target is not None:
        tgt = load_wav_stereo(args.target, sr).to(device)
        # match target length to output for metrics
        if tgt.shape[1] != out.shape[1]:
            Tm = min(tgt.shape[1], out.shape[1])
            tgt = tgt[:, :Tm]
            out_m = out[:, :Tm]
        else:
            out_m = out
        s0 = si_sdr_1d(out_m[0], tgt[0])
        s1 = si_sdr_1d(out_m[1], tgt[1])
        print(f"SI-SDR L={s0:.2f} dB | R={s1:.2f} dB | avg={(s0+s1)/2:.2f} dB")


if __name__ == "__main__":
    main()
