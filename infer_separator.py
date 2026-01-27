# infer_ola.py
# Overlap-Add inference for StemSeparator
#
# Example:
#   python infer_ola.py --ckpt runs/exp1/ckpt_last.pt --in mix.wav --out out_dir \
#       --segment-sec 8.0 --overlap 0.5 --amp bf16 --tf32 1
#
# Notes:
# - OLA crossfade uses sqrt-Hann window -> accumulate window^2 and divide at the end.
# - Expects model forward: out = model(mix) -> dict[str, (B,2,T)]
# - STEM_ORDER must match training.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch.amp import autocast

# audio i/o
import soundfile as sf

from deep_separator import StemSeparator, SeparatorConfig
from train_phase_a import STEM_ORDER


# -------------------------
# Precision / TF32 helpers
# -------------------------

def set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    try:
        torch.set_float32_matmul_precision("high" if enabled else "highest")
    except Exception:
        pass


def make_autocast(amp: str, device: torch.device):
    amp = amp.lower().strip()
    if device.type != "cuda":
        return autocast("cpu", enabled=False)

    if amp in ("off", "none", "0"):
        return autocast("cuda", enabled=False)
    if amp == "fp16":
        return autocast("cuda", dtype=torch.float16)
    if amp == "bf16":
        return autocast("cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unknown --amp={amp!r}. Use off|fp16|bf16.")


# -------------------------
# Audio helpers
# -------------------------

def _to_stereo(x: np.ndarray) -> np.ndarray:
    # returns (T,2)
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.ndim == 2 and x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    if x.ndim == 2 and x.shape[1] >= 2:
        return x[:, :2]
    raise ValueError(f"Unsupported audio shape: {x.shape}")


def _maybe_resample(x: np.ndarray, sr_in: int, sr_out: int) -> Tuple[np.ndarray, int]:
    if sr_in == sr_out:
        return x, sr_in

    # try torchaudio if available
    try:
        import torchaudio
        xt = torch.from_numpy(x.T).float()  # (C,T)
        yt = torchaudio.functional.resample(xt, sr_in, sr_out)
        y = yt.cpu().numpy().T
        return y, sr_out
    except Exception as e:
        raise RuntimeError(
            f"Input SR={sr_in}, but model SR={sr_out}. "
            f"Install torchaudio for resample or pre-resample externally. Original error: {e}"
        )


# -------------------------
# OLA core
# -------------------------

@torch.no_grad()
def separate_ola(
    model: StemSeparator,
    mix: torch.Tensor,            # (1,2,T) float32
    sr: int,
    segment_sec: float,
    overlap: float,
    device: torch.device,
    amp: str = "bf16",
    window: str = "sqrt_hann",
    pad_mode: str = "zeros",
) -> Dict[str, torch.Tensor]:
    """
    Returns dict stem -> (2,T) on CPU float32.
    """
    assert mix.ndim == 3 and mix.shape[0] == 1 and mix.shape[1] == 2, mix.shape
    assert 0.0 <= overlap < 1.0

    T = int(mix.shape[-1])
    seg = int(round(segment_sec * sr))
    if seg <= 0:
        raise ValueError(f"segment_sec too small: {segment_sec}")

    hop = int(round(seg * (1.0 - overlap)))
    hop = max(1, hop)

    # windowing
    dev = device
    if window == "sqrt_hann":
        w = torch.hann_window(seg, periodic=True, device=dev, dtype=torch.float32).sqrt()
    elif window == "hann":
        w = torch.hann_window(seg, periodic=True, device=dev, dtype=torch.float32)
    elif window in ("rect", "box", "none"):
        w = torch.ones(seg, device=dev, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown window={window!r}")

    # prepare accumulators
    out_acc: Dict[str, torch.Tensor] = {
        s: torch.zeros((2, T), device=dev, dtype=torch.float32) for s in STEM_ORDER
    }
    denom = torch.zeros((T,), device=dev, dtype=torch.float32)

    # pad so we can cover the tail cleanly
    # We do "left=0, right=pad" and handle last chunk with pad inside extraction.
    autocast_ctx = make_autocast(amp, device)

    pos = 0
    while pos < T:
        end = pos + seg
        if end <= T:
            chunk = mix[:, :, pos:end]  # (1,2,seg)
            valid = seg
        else:
            # tail pad
            valid = T - pos
            if pad_mode == "zeros":
                chunk = torch.zeros((1, 2, seg), device=dev, dtype=mix.dtype)
                chunk[:, :, :valid] = mix[:, :, pos:T]
            else:
                raise ValueError(f"Unsupported pad_mode={pad_mode!r}")

        # analysis window
        win = w.view(1, 1, seg)  # (1,1,seg)
        chunk_w = chunk * win

        with autocast_ctx:
            pred = model(chunk_w, return_debug=False)  # dict stem -> (1,2,seg)

        # synthesis window + accumulate
        # Use window^2 for denom so reconstruction is stable.
        w2 = (w * w)  # (seg,)
        if end <= T:
            denom[pos:end] += w2
        else:
            denom[pos:T] += w2[:valid]

        for s in STEM_ORDER:
            y = pred[s].to(dtype=torch.float32)  # (1,2,seg)
            y = (y * win).squeeze(0)             # (2,seg)
            if end <= T:
                out_acc[s][:, pos:end] += y
            else:
                out_acc[s][:, pos:T] += y[:, :valid]

        pos += hop

    # normalize
    denom = denom.clamp_min(1e-8).view(1, T)  # (1,T)
    out = {s: (out_acc[s] / denom).detach().cpu() for s in STEM_ORDER}
    return out


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint path (ckpt_last.pt)")
    p.add_argument("--in", dest="inp", type=str, required=True, help="input wav/flac/ogg (soundfile-supported)")
    p.add_argument("--out", type=str, required=True, help="output directory")
    p.add_argument("--segment-sec", type=float, default=8.0)
    p.add_argument("--overlap", type=float, default=0.5, help="0.. <1.0 (0.5 typical)")
    p.add_argument("--amp", type=str, default="bf16", choices=["off", "fp16", "bf16"])
    p.add_argument("--tf32", type=int, default=1)
    p.add_argument("--window", type=str, default="sqrt_hann", choices=["sqrt_hann", "hann", "rect"])
    p.add_argument("--no-resample", action="store_true", help="error if SR != model SR")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_tf32(bool(args.tf32))

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("cfg", None)

    # Build config
    cfg = SeparatorConfig()
    if isinstance(cfg_dict, dict):
        # best-effort: update dataclass fields if names match
        for k, v in cfg_dict.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # Load model
    model = StemSeparator(cfg).to(device)
    model.eval()
    model.load_state_dict(ckpt["model"], strict=True)

    # Load audio
    x, sr_in = sf.read(args.inp, always_2d=False)
    x = _to_stereo(x).astype(np.float32)  # (T,2)

    if sr_in != int(cfg.sample_rate):
        if args.no_resample:
            raise RuntimeError(f"SR mismatch: input={sr_in}, model={cfg.sample_rate}. Remove --no-resample or pre-resample.")
        x, _ = _maybe_resample(x, sr_in, int(cfg.sample_rate))

    # to torch (1,2,T)
    xt = torch.from_numpy(x.T).unsqueeze(0).to(device=device, dtype=torch.float32)

    # OLA separate
    stems = separate_ola(
        model=model,
        mix=xt,
        sr=int(cfg.sample_rate),
        segment_sec=float(args.segment_sec),
        overlap=float(args.overlap),
        device=device,
        amp=args.amp,
        window=args.window,
    )

    # Save
    for name, y in stems.items():
        y_np = y.transpose(0, 1).numpy()  # (T,2)
        sf.write(str(out_dir / f"{name}.wav"), y_np, int(cfg.sample_rate), subtype="FLOAT")

    # also save sum for sanity
    stem_sum = sum(stems[s] for s in STEM_ORDER)  # (2,T) CPU
    sf.write(str(out_dir / "stem_sum.wav"), stem_sum.transpose(0, 1).numpy(), int(cfg.sample_rate), subtype="FLOAT")

    print(f"[ok] wrote stems to: {out_dir}")


if __name__ == "__main__":
    main()
