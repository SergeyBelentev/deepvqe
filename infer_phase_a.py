# infer_phase_a_v2.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from deepvqe import DeepVQEConditionalStemSeparator


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
def match_length(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    x: (2,Tx) -> (2,T) by crop/pad with zeros
    """
    Tx = x.shape[1]
    if Tx == T:
        return x
    if Tx > T:
        return x[:, :T].contiguous()
    # pad right
    return F.pad(x, (0, T - Tx))


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


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
def _strip_prefix_if_all_keys(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict

def _normalize_checkpoint_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # 1) DDP: "module."
    sd = _strip_prefix_if_all_keys(state_dict, "module.")
    # 2) torch.compile иногда даёт "_orig_mod."
    sd = _strip_prefix_if_all_keys(sd, "_orig_mod.")
    # 3) бывает комбо "module._orig_mod."
    sd = _strip_prefix_if_all_keys(sd, "module._orig_mod.")
    return sd

def load_model(ckpt_path: str, device: torch.device, n_fft: int, num_heads: int = 4):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(f"Bad checkpoint: {ckpt_path}")

    model = DeepVQEConditionalStemSeparator(n_fft=n_fft, num_heads=num_heads).to(device)

    sd = ckpt["model"]
    if not isinstance(sd, dict):
        raise RuntimeError("ckpt['model'] is not a state_dict dict")

    # сначала пробуем как есть (на случай если чекпойнт уже “чистый”)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        sd2 = _normalize_checkpoint_state_dict(sd)
        model.load_state_dict(sd2, strict=True)

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
    model: DeepVQEConditionalStemSeparator,
    stft: STFT,
    x_stereo: torch.Tensor,              # (2,T)
    *,
    ref_stereo: Optional[torch.Tensor] = None,  # (2,T) or None
    sr: int,
    chunk_sec: float,
    overlap: float,
    device: torch.device,
    use_amp: bool = False,
    stitch: str = "ola",
    keep_frac: float = 0.6,
    xfade_ms: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Returns dict head->(2,T) in float32 on CPU.
    Heads: bass, drums, music, vocals
    """
    assert 0.0 <= overlap < 1.0
    if stitch not in ("ola", "crop", "full"):
        raise ValueError("stitch must be 'ola' or 'crop' or 'full'")

    x = x_stereo.to(device, dtype=torch.float32)
    C, T = x.shape
    assert C == 2

    if ref_stereo is None:
        ref = None
    else:
        ref = ref_stereo.to(device, dtype=torch.float32)
        if ref.shape[0] != 2:
            raise ValueError("ref_stereo must be (2,T)")
        if ref.shape[1] != T:
            # на всякий случай: но лучше это делать ещё в main()
            if ref.shape[1] > T:
                ref = ref[:, :T]
            else:
                ref = F.pad(ref, (0, T - ref.shape[1]))


    chunk_len = int(round(sr * chunk_sec))
    if chunk_len <= 0:
        raise ValueError("chunk_sec too small")

    def pred_to_time(pred: torch.Tensor) -> torch.Tensor:
        """
        pred: (2,4,F,Tf,2)
        returns y: (4,2,chunk_len)
        """
        # (4,2,F,Tf,2) -> (8,F,Tf,2)
        p = pred.permute(1, 0, 2, 3, 4).contiguous()
        p = p.view(p.shape[0] * p.shape[1], p.shape[2], p.shape[3], p.shape[4])
        y = stft.istft_ri(p, length=chunk_len)  # (8, chunk_len)
        y = y.view(4, 2, chunk_len)
        return y

    def call_model(mix_ri: torch.Tensor, ref_ri: Optional[torch.Tensor]) -> torch.Tensor:
        return model(mix_ri, ref_ri)


    def run_model_on_chunk(chunk: torch.Tensor, ref_chunk: Optional[torch.Tensor]) -> torch.Tensor:
        """
        chunk: (2,chunk_len) float32 on device
        ref_chunk: (2,chunk_len) float32 on device or None
        returns y: (4,2,chunk_len) float32 on device
        """
        mix_ri = stft.stft_ri(chunk)  # (2,F,Tf,2)
        ref_ri = stft.stft_ri(ref_chunk) if (ref_chunk is not None) else None

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = call_model(mix_ri, ref_ri)  # (2,4,F,Tf,2) expected
        else:
            pred = call_model(mix_ri, ref_ri)

        return pred_to_time(pred)


    # -----------------------
    # Mode 0: full inference (single pass, no windows)
    # -----------------------
    if stitch == "full":
        # x: (2,T) on device
        mix_ri = stft.stft_ri(x)  # (2,F,Tf,2)
        ref_ri = stft.stft_ri(ref) if (ref is not None) else None

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = call_model(mix_ri, ref_ri) # (2,4,F,Tf,2)
        else:
            pred = call_model(mix_ri, ref_ri)

        # pred: (2,4,F,Tf,2) -> time
        # (4,2,F,Tf,2) -> (8,F,Tf,2)
        p = pred.permute(1, 0, 2, 3, 4).contiguous()
        p = p.view(p.shape[0] * p.shape[1], p.shape[2], p.shape[3], p.shape[4])  # (8,F,Tf,2)
        y = stft.istft_ri(p, length=T)  # (8,T)
        y = y.view(4, 2, T).detach().cpu()

        names = ["bass", "drums", "music", "vocals"]
        return {names[i]: y[i] for i in range(4)}


    # -----------------------
    # Mode 1: classic OLA (fixed normalization: wsum += win)
    # -----------------------
    if stitch == "ola":
        hop_len = int(round(chunk_len * (1.0 - overlap)))
        hop_len = max(1, hop_len)

        # pad so that last frame fits
        n_chunks = 1 + max(0, (T - 1) // hop_len)
        total_len = (n_chunks - 1) * hop_len + chunk_len
        pad = max(0, total_len - T)
        if pad > 0:
            x_pad = F.pad(x, (0, pad))
            ref_pad = F.pad(ref, (0, pad)) if (ref is not None) else None
        else:
            x_pad = x
            ref_pad = ref


        T_pad = x_pad.shape[1]

        out = torch.zeros((4, 2, T_pad), device=device, dtype=torch.float32)
        wsum = torch.zeros((1, 1, T_pad), device=device, dtype=torch.float32)

        win = make_hann_ola(chunk_len, device=device)          # (L,)
        win_v = win.view(1, 1, -1)                              # (1,1,L)

        for i in range(n_chunks):
            s = i * hop_len
            e = s + chunk_len
            chunk = x_pad[:, s:e]  # (2,L)
            ref_chunk = ref_pad[:, s:e] if (ref_pad is not None) else None
            y = run_model_on_chunk(chunk, ref_chunk)

            out[:, :, s:e] += y * win_v
            wsum[:, :, s:e] += win_v                            # <<< FIX: sum(win), not sum(win^2)

        out = out / wsum.clamp_min(1e-8)
        out = out[:, :, :T].detach().cpu()

        names = ["bass", "drums", "music", "vocals"]
        return {names[i]: out[i] for i in range(4)}

    # -----------------------
    # Mode 2: crop-stitching (take central 50-75% and stitch)
    # -----------------------
    keep_frac = float(keep_frac)
    if not (0.50 <= keep_frac <= 0.75):
        raise ValueError("--keep-frac must be in [0.50, 0.75] for crop mode")

    keep_len = int(round(chunk_len * keep_frac))
    keep_len = max(1, min(keep_len, chunk_len))

    trim_left = (chunk_len - keep_len) // 2
    trim_right = (chunk_len - keep_len) - trim_left  # may differ by 1 if odd

    # optional crossfade between kept regions
    xfade_len = int(round(sr * float(xfade_ms) / 1000.0))
    xfade_len = max(0, min(xfade_len, keep_len // 2))

    # effective hop in output timeline
    hop_out = keep_len - xfade_len
    hop_out = max(1, hop_out)

    # choose number of chunks so we fully cover T samples in output
    if T <= keep_len:
        n_chunks = 1
    else:
        n_chunks = 1 + math.ceil((T - keep_len) / hop_out)

    out_len = (n_chunks - 1) * hop_out + keep_len
    out = torch.zeros((4, 2, out_len), device=device, dtype=torch.float32)

    if xfade_len > 0:
        wsum = torch.zeros((1, 1, out_len), device=device, dtype=torch.float32)
        fade = torch.linspace(0.0, 1.0, xfade_len, device=device, dtype=torch.float32)
    else:
        wsum = None
        fade = None

    def get_chunk_by_start(src: torch.Tensor, start: int) -> torch.Tensor:
        """
        src: (2,T)
        start can be negative. Returns (2,chunk_len) padded with zeros if needed.
        """
        end = start + chunk_len
        left_pad = max(0, -start)
        right_pad = max(0, end - T)
        s0 = max(0, start)
        e0 = min(T, end)
        ch = src[:, s0:e0]
        if left_pad or right_pad:
            ch = F.pad(ch, (left_pad, right_pad))
        if ch.shape[1] != chunk_len:
            ch = F.pad(ch, (0, max(0, chunk_len - ch.shape[1])))
            ch = ch[:, :chunk_len]
        return ch

    for i in range(n_chunks):
        out_s = i * hop_out
        out_e = out_s + keep_len

        # align: kept center lands exactly at [out_s:out_e]
        in_s = out_s - trim_left  # may be negative
        chunk = get_chunk_by_start(x, in_s)
        ref_chunk = get_chunk_by_start(ref, in_s) if (ref is not None) else None

        y = run_model_on_chunk(chunk, ref_chunk)
        y_keep = y[:, :, trim_left:trim_left + keep_len]  # (4,2,keep_len)

        if xfade_len <= 0:
            out[:, :, out_s:out_e] = y_keep
            continue

        # build per-chunk weights:
        # - first chunk: no fade-in
        # - last chunk: no fade-out
        w = torch.ones((keep_len,), device=device, dtype=torch.float32)
        if i > 0:
            w[:xfade_len] = fade
        if i < n_chunks - 1:
            w[-xfade_len:] = torch.flip(fade, dims=[0])
        wv = w.view(1, 1, -1)

        out[:, :, out_s:out_e] += y_keep * wv
        wsum[:, :, out_s:out_e] += wv  # type: ignore[index]

    if wsum is not None:
        out = out / wsum.clamp_min(1e-8)

    out = out[:, :, :T].detach().cpu()
    names = ["bass", "drums", "music", "vocals"]
    return {names[i]: out[i] for i in range(4)}



def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", required=True, type=str, help="Path to phase_a_latest.pt (or phase_a_eXXX.pt)")
    ap.add_argument("--inp", required=True, type=str, help="Input audio file (wav/flac/...)")
    ap.add_argument("--ref", default="", type=str, help="(conditional) Reference audio file (same SR). If empty -> zeros.")
    ap.add_argument("--ref-gain-db", type=float, default=0.0, help="Optional gain (dB) applied to ref before STFT.")
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

    ap.add_argument(
        "--stitch",
        choices=["ola", "crop", "full"],
        default="ola",
        help="Stitching mode: 'full' = single-pass on entire audio, 'ola' = overlap-add, 'crop' = central crop stitch",
    )
    ap.add_argument(
        "--keep-frac",
        type=float,
        default=0.6,
        help="(crop) Central fraction of chunk to keep. Must be in [0.50, 0.75].",
    )
    ap.add_argument(
        "--xfade-ms",
        type=float,
        default=0.0,
        help="(crop) Optional crossfade length in milliseconds between kept regions (0 = off).",
    )


    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print("device:", device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load audio
    x = read_audio(args.inp, sr_expected=int(args.sr))  # (2,T)
    T = x.shape[1]
    print(f"[audio] {args.inp} | sr={args.sr} | samples={T} | sec={T/args.sr:.2f}")

    ref = None
    if args.ref:
        ref = read_audio(args.ref, sr_expected=int(args.sr))  # (2,Tr)
        ref = match_length(ref, T)
        g = db_to_lin(float(args.ref_gain_db))
        if g != 1.0:
            ref = ref * float(g)
        print(f"[ref] {args.ref} | gain_db={args.ref_gain_db}")


    # load model + stft
    model = load_model(args.ckpt, device=device, n_fft=int(args.n_fft), num_heads=4)
    stft = STFT(StftCfg(n_fft=int(args.n_fft), hop=int(args.hop), win=int(args.win))).to(device)

    # infer
    stems = infer_ola(
        model=model,
        stft=stft,
        x_stereo=x,
        ref_stereo=ref,
        sr=int(args.sr),
        chunk_sec=float(args.chunk_sec),
        overlap=float(args.overlap),
        device=device,
        use_amp=bool(args.amp),
        stitch=str(args.stitch),
        keep_frac=float(args.keep_frac),
        xfade_ms=float(args.xfade_ms),
    )

    # write
    if args.write_mix:
        write_audio(outdir / "mix.wav", x.transpose(0, 1).numpy(), int(args.sr), fmt=args.format)

    for name, y in stems.items():
        y_np = y.transpose(0, 1).numpy()  # (T,2)
        write_audio(outdir / f"{name}.wav", y_np, int(args.sr), fmt=args.format)

    if args.write_sum or args.write_sum5:
        # sum heads in time domain (already comes from ISTFT)
        sum5 = stems["bass"] + stems["drums"] + stems["music"] + stems["vocals"]
        if args.write_sum:
            write_audio(outdir / "sum.wav", sum5.transpose(0, 1).numpy(), int(args.sr), fmt=args.format)

    # quick console stats
    def peak(t: torch.Tensor) -> float:
        return float(t.abs().max().item()) if t.numel() else 0.0

    print("[done] wrote:")
    for n in ["bass", "drums", "music", "vocals"]:
        print(f"  {n:7s} peak={peak(stems[n]):.4f}")
    if args.write_sum:
        print(f"  sum     peak={peak(sum5):.4f}")
    print(f"  outdir: {outdir}")


if __name__ == "__main__":
    main()
