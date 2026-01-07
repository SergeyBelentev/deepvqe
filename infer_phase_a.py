# infer_phase_a_v2.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from deepvqe import DeepVQEStemSeparator


# -----------------------
# STFT helper (fp32, matches training defaults)
# -----------------------
@dataclass
class StftCfg:
    n_fft: int = 1536
    hop: int = 480
    win: int = 1536


class STFT(nn.Module):
    def __init__(self, cfg: StftCfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("window", torch.hann_window(cfg.win, dtype=torch.float32), persistent=False)

    def stft_ri(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,T) float32
        w = self.window.to(device=x.device, dtype=torch.float32)
        X = torch.stft(
            x,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop,
            win_length=self.cfg.win,
            window=w,
            return_complex=True,
        )
        return torch.view_as_real(X).to(torch.float32)  # (N,F,Tf,2)

    def istft_ri(self, X_ri: torch.Tensor, length: int) -> torch.Tensor:
        w = self.window.to(device=X_ri.device, dtype=torch.float32)
        X = torch.complex(X_ri[..., 0].float(), X_ri[..., 1].float())
        y = torch.istft(
            X,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop,
            win_length=self.cfg.win,
            window=w,
            length=length,
        )
        return y.to(torch.float32)


# -----------------------
# Audio I/O
# -----------------------
def read_audio(path: str, sr_expected: int) -> torch.Tensor:
    """
    Returns stereo float32: (2, T)
    """
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != sr_expected:
        raise RuntimeError(f"SR mismatch: got {sr}, expected {sr_expected}. Resample outside this script.")
    # x: (T, ch)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    else:
        x = x[:, :2]
    return torch.from_numpy(x).transpose(0, 1).contiguous()  # (2,T)


def write_audio(path: str, y: np.ndarray, sr: int, fmt: str = "float") -> None:
    """
    y: (T,2) float32
    """
    path = str(path)
    if fmt == "float":
        sf.write(path, y, sr, subtype="FLOAT")
    elif fmt == "pcm16":
        # clip to [-1,1] to be safe
        y16 = np.clip(y, -1.0, 1.0)
        sf.write(path, y16, sr, subtype="PCM_16")
    else:
        raise ValueError("fmt must be one of: float, pcm16")


# -----------------------
# Model loading
# -----------------------
def load_model(ckpt_path: str, device: torch.device, n_fft: int, num_heads: int = 6) -> DeepVQEStemSeparator:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(f"Bad checkpoint: {ckpt_path}")

    model = DeepVQEStemSeparator(n_fft=n_fft, num_heads=num_heads).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


# -----------------------
# OLA inference
# -----------------------
def make_hann_ola(win_len: int, device: torch.device) -> torch.Tensor:
    # nice for 50% overlap
    w = torch.hann_window(win_len, periodic=True, dtype=torch.float32, device=device)
    # avoid exact zeros at edges in case of tiny numeric issues
    return w.clamp_min(1e-8)


@torch.no_grad()
def infer_ola(
    model: DeepVQEStemSeparator,
    stft: STFT,
    x_stereo: torch.Tensor,  # (2,T)
    *,
    sr: int,
    chunk_sec: float,
    overlap: float,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Returns dict head->(2,T) in float32 on CPU.
    Heads: bass, drums, inst, melody, vocals, fx
    """
    assert 0.0 <= overlap < 1.0

    x = x_stereo.to(device, dtype=torch.float32)
    C, T = x.shape
    assert C == 2

    chunk_len = int(round(sr * chunk_sec))
    if chunk_len <= 0:
        raise ValueError("chunk_sec too small")

    hop_len = int(round(chunk_len * (1.0 - overlap)))
    hop_len = max(1, hop_len)

    # pad so that last frame fits
    n_chunks = 1 + max(0, (T - 1) // hop_len)
    total_len = (n_chunks - 1) * hop_len + chunk_len
    pad = max(0, total_len - T)
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, pad))  # (2, T+pad)

    T_pad = x.shape[1]

    # output accumulators (device float32)
    out = torch.zeros((6, 2, T_pad), device=device, dtype=torch.float32)
    wsum = torch.zeros((1, 1, T_pad), device=device, dtype=torch.float32)

    win = make_hann_ola(chunk_len, device=device)  # (chunk_len,)
    win2 = (win * win).view(1, 1, -1)  # power weights for denom

    # loop
    for i in range(n_chunks):
        s = i * hop_len
        e = s + chunk_len
        chunk = x[:, s:e]  # (2, chunk_len)

        # (B*C,T) where B=1
        mix_f = chunk.reshape(2, chunk_len)  # already (C,T)
        mix_ri = stft.stft_ri(mix_f)  # (2,F,Tf,2)

        # AMP optional (по умолчанию off, как у тебя fp32)
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(mix_ri)  # (2,6,F,Tf,2)
        else:
            pred = model(mix_ri)

        # ISTFT each head -> (2,chunk_len)
        # pred layout in your training: (B*C,6,F,Tf,2)
        heads_time: List[torch.Tensor] = []
        for h in range(6):
            y = stft.istft_ri(pred[:, h], length=chunk_len)  # (2,chunk_len)
            heads_time.append(y)

        y6 = torch.stack(heads_time, dim=0)  # (6,2,chunk_len)

        # apply window and OLA
        y6w = y6 * win.view(1, 1, -1)
        out[:, :, s:e] += y6w
        wsum[:, :, s:e] += win2

    out = out / wsum.clamp_min(1e-8)

    # crop padding and move to CPU
    out = out[:, :, :T].detach().cpu()

    names = ["bass", "drums", "inst", "melody", "vocals", "fx"]
    return {names[i]: out[i] for i in range(6)}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", required=True, type=str, help="Path to phase_a_latest.pt (or phase_a_eXXX.pt)")
    ap.add_argument("--inp", required=True, type=str, help="Input audio file (wav/flac/...)")
    ap.add_argument("--outdir", default="out_phase_a", type=str)

    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--n-fft", type=int, default=1536)
    ap.add_argument("--hop", type=int, default=480)
    ap.add_argument("--win", type=int, default=1536)

    ap.add_argument("--chunk-sec", type=float, default=4.0, help="Window size in seconds")
    ap.add_argument("--overlap", type=float, default=0.5, help="Overlap ratio for OLA (0.. <1)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true", help="Enable autocast fp16 on CUDA (optional)")

    ap.add_argument("--format", choices=["float", "pcm16"], default="float")
    ap.add_argument("--write-mix", action="store_true", help="Also write mix.wav as a copy of input")
    ap.add_argument("--write-sum", action="store_true", help="Also write sum.wav = sum(6 heads)")
    ap.add_argument("--write-sum5", action="store_true", help="Also write sum5.wav = sum(first 5 heads)")

    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print("device:", device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load audio
    x = read_audio(args.inp, sr_expected=int(args.sr))  # (2,T)
    T = x.shape[1]
    print(f"[audio] {args.inp} | sr={args.sr} | samples={T} | sec={T/args.sr:.2f}")

    # load model + stft
    model = load_model(args.ckpt, device=device, n_fft=int(args.n_fft), num_heads=6)
    stft = STFT(StftCfg(n_fft=int(args.n_fft), hop=int(args.hop), win=int(args.win))).to(device)

    # infer
    stems = infer_ola(
        model=model,
        stft=stft,
        x_stereo=x,
        sr=int(args.sr),
        chunk_sec=float(args.chunk_sec),
        overlap=float(args.overlap),
        device=device,
        use_amp=bool(args.amp),
    )

    # write
    if args.write_mix:
        write_audio(outdir / "mix.wav", x.transpose(0, 1).numpy(), int(args.sr), fmt=args.format)

    for name, y in stems.items():
        y_np = y.transpose(0, 1).numpy()  # (T,2)
        write_audio(outdir / f"{name}.wav", y_np, int(args.sr), fmt=args.format)

    if args.write_sum or args.write_sum5:
        # sum heads in time domain (already comes from ISTFT)
        sum5 = stems["bass"] + stems["drums"] + stems["inst"] + stems["melody"] + stems["vocals"]
        if args.write_sum5:
            write_audio(outdir / "sum5.wav", sum5.transpose(0, 1).numpy(), int(args.sr), fmt=args.format)
        if args.write_sum:
            sum6 = sum5 + stems["fx"]
            write_audio(outdir / "sum.wav", sum6.transpose(0, 1).numpy(), int(args.sr), fmt=args.format)

    # quick console stats
    def peak(t: torch.Tensor) -> float:
        return float(t.abs().max().item()) if t.numel() else 0.0

    print("[done] wrote:")
    for n in ["bass", "drums", "inst", "melody", "vocals", "fx"]:
        print(f"  {n:7s} peak={peak(stems[n]):.4f}")
    if args.write_sum5:
        print(f"  sum5    peak={peak(sum5):.4f}")
    if args.write_sum:
        sum6 = sum5 + stems["fx"]
        print(f"  sum     peak={peak(sum6):.4f}")
    print(f"  outdir: {outdir}")


if __name__ == "__main__":
    main()
