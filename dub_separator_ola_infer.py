from __future__ import annotations

import argparse
import math
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.amp import autocast
from tqdm.auto import tqdm

from dub_separator_optimized import BandSpec, DubSeparator, DubSeparatorConfig


# ============================================================
# Utils
# ============================================================


def build_amp_dtype(amp_mode: str) -> torch.dtype | None:
    amp_mode = amp_mode.lower()
    if amp_mode == "none":
        return None
    if amp_mode == "bf16":
        return torch.bfloat16
    if amp_mode == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported amp mode: {amp_mode}")


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def make_autocast(device: torch.device, amp_mode: str):
    dtype = build_amp_dtype(amp_mode)
    enabled = dtype is not None and device.type == "cuda"
    return autocast(device_type=device.type if device.type in {"cuda", "cpu"} else "cuda", dtype=dtype, enabled=enabled)



def _ensure_stereo(x: np.ndarray, path: str) -> np.ndarray:
    if x.ndim == 1:
        x = np.stack([x, x], axis=-1)
    if x.ndim != 2:
        raise RuntimeError(f"Expected 1D/2D audio from {path}, got shape {x.shape}")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] >= 2:
        x = x[:, :2]
    return x.astype(np.float32, copy=False)



def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return _ensure_stereo(wav, str(path)), int(sr)



def trim_pair_to_min(mix: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(mix.shape[0], ref.shape[0])
    return mix[:n], ref[:n]



def relative_to_best_effort(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except Exception:
        return Path(path.name)


# ============================================================
# Checkpoint / config
# ============================================================



def _bandspec_from_any(x: Any) -> BandSpec:
    if isinstance(x, BandSpec):
        return x
    if isinstance(x, dict):
        return BandSpec(**x)
    raise TypeError(f"Cannot convert band spec from {type(x)!r}")



def config_from_payload(obj: Any) -> DubSeparatorConfig:
    if isinstance(obj, DubSeparatorConfig):
        return obj
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported model_cfg payload type: {type(obj)!r}")

    cfg_fields = {f.name for f in fields(DubSeparatorConfig)}
    data: Dict[str, Any] = {}
    for k, v in obj.items():
        if k not in cfg_fields:
            continue
        if k == "bands":
            data[k] = tuple(_bandspec_from_any(b) for b in v)
        else:
            data[k] = v
    return DubSeparatorConfig(**data)



def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[DubSeparator, Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(payload, dict) and "model" in payload:
        cfg = config_from_payload(payload.get("model_cfg", {}))
        model = DubSeparator(cfg)
        model.load_state_dict(payload["model"], strict=True)
        meta = {
            "epoch": int(payload.get("epoch", 0)),
            "global_step": int(payload.get("global_step", 0)),
            "has_train_payload": True,
        }
        return model.to(device), meta

    if isinstance(payload, dict):
        # raw state_dict fallback
        model = DubSeparator(DubSeparatorConfig())
        model.load_state_dict(payload, strict=True)
        meta = {"epoch": 0, "global_step": 0, "has_train_payload": False}
        return model.to(device), meta

    raise RuntimeError(f"Unsupported checkpoint payload type: {type(payload)!r}")


# ============================================================
# File discovery
# ============================================================



def discover_pairs(root: Path) -> List[Tuple[Path, Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Input path not found: {root}")

    mix_files = sorted(p for p in root.rglob("*mix.wav") if p.is_file())
    pairs: List[Tuple[Path, Path]] = []
    missing: List[Path] = []

    for mix_path in mix_files:
        stem = mix_path.name[:-len("mix.wav")]
        ref_path = mix_path.with_name(f"{stem}ref.wav")
        if ref_path.exists():
            pairs.append((mix_path, ref_path))
        else:
            missing.append(mix_path)

    if not pairs:
        raise RuntimeError(f"No mix/ref wav pairs found under: {root}")
    if missing:
        print(f"warning: skipped {len(missing)} mix files without matching ref.wav")
    return pairs


# ============================================================
# OLA inference
# ============================================================



def make_ola_window(num_samples: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if num_samples <= 1:
        return torch.ones(num_samples, device=device, dtype=dtype)
    # smooth fade in/out, better than rectangular window for chunk stitching
    return torch.hann_window(num_samples, periodic=False, device=device, dtype=dtype)



def build_chunk_starts(total_samples: int, chunk_samples: int, hop_samples: int) -> List[int]:
    if total_samples <= chunk_samples:
        return [0]
    starts = list(range(0, max(total_samples - chunk_samples, 0) + 1, hop_samples))
    last = total_samples - chunk_samples
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.inference_mode()
def ola_infer_pair(
    *,
    model: DubSeparator,
    mix_np: np.ndarray,
    ref_np: np.ndarray,
    device: torch.device,
    amp_mode: str,
    segment_sec: float,
    overlap_sec: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    mix_np, ref_np = trim_pair_to_min(mix_np, ref_np)
    total_samples = int(mix_np.shape[0])
    sr = int(model.cfg.sample_rate)

    chunk_samples = int(round(segment_sec * sr))
    overlap_samples = int(round(overlap_sec * sr))
    if chunk_samples <= 0:
        raise ValueError("segment-sec must produce at least 1 sample")
    if overlap_samples < 0:
        raise ValueError("overlap-sec must be >= 0")
    if overlap_samples >= chunk_samples:
        raise ValueError("overlap-sec must be smaller than segment-sec")

    hop_samples = chunk_samples - overlap_samples
    starts = build_chunk_starts(total_samples, chunk_samples, hop_samples)

    out_acc = torch.zeros((1, 2, total_samples), device=device, dtype=torch.float32)
    weight_acc = torch.zeros((1, 1, total_samples), device=device, dtype=torch.float32)
    window = make_ola_window(chunk_samples, device=device, dtype=torch.float32).view(1, 1, -1)

    # avoid zeros at boundaries so single-chunk edges still normalize correctly
    if chunk_samples > 1:
        window[..., 0] = 1e-3
        window[..., -1] = 1e-3

    for start in starts:
        end = min(start + chunk_samples, total_samples)
        current_len = end - start

        mix_chunk = np.zeros((chunk_samples, 2), dtype=np.float32)
        ref_chunk = np.zeros((chunk_samples, 2), dtype=np.float32)
        mix_chunk[:current_len] = mix_np[start:end]
        ref_chunk[:current_len] = ref_np[start:end]

        mix_t = torch.from_numpy(mix_chunk.T.copy()).unsqueeze(0).to(device)
        ref_t = torch.from_numpy(ref_chunk.T.copy()).unsqueeze(0).to(device)

        with make_autocast(device, amp_mode):
            outputs = model(mix_t, ref_t)
        est = outputs["estimate_waveform"].float()[..., :current_len]
        w = window[..., :current_len]

        out_acc[..., start:end] += est * w
        weight_acc[..., start:end] += w

    estimate = out_acc / weight_acc.clamp_min(1e-6)
    estimate_np = estimate.squeeze(0).transpose(0, 1).detach().cpu().numpy()

    meta = {
        "num_chunks": len(starts),
        "total_samples": total_samples,
        "segment_samples": chunk_samples,
        "overlap_samples": overlap_samples,
        "hop_samples": hop_samples,
        "duration_sec": total_samples / float(sr),
    }
    return estimate_np, meta


# ============================================================
# CLI
# ============================================================



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OLA inference for DubSeparator")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to train checkpoint or raw model state_dict")
    p.add_argument("--input", type=Path, required=True, help="Input folder containing *_mix.wav/*_ref.wav pairs (recursive)")
    p.add_argument("--out-dir", type=Path, required=True, help="Output folder for estimated wav files")
    p.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16", "fp16"], help="AMP mode")
    p.add_argument("--tf32", action="store_true", help="Enable TF32 on CUDA")
    p.add_argument("--segment-sec", type=float, default=4.0, help="Chunk size in seconds for OLA")
    p.add_argument("--overlap-sec", type=float, default=1.0, help="Overlap size in seconds for OLA")
    p.add_argument("--suffix", type=str, default="_dub.wav", help="Output suffix replacing _mix.wav")
    p.add_argument("--flat", action="store_true", help="Do not mirror input subfolders; save all outputs directly under out-dir")
    p.add_argument("--limit", type=int, default=0, help="Optional max number of pairs to process")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return p



def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model, meta = load_model(args.checkpoint, device=device)
    model.eval()

    pairs = discover_pairs(args.input)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== model ===")
    print(f"checkpoint        : {args.checkpoint}")
    print(f"device            : {device}")
    print(f"amp               : {args.amp}")
    print(f"tf32              : {bool(args.tf32)}")
    print(f"sample_rate       : {model.cfg.sample_rate}")
    print(f"segment_sec       : {args.segment_sec}")
    print(f"overlap_sec       : {args.overlap_sec}")
    print(f"n_fft / hop       : {model.cfg.n_fft} / {model.cfg.hop_length}")
    print(f"train payload     : {meta['has_train_payload']}")
    if meta["has_train_payload"]:
        print(f"epoch/global_step : {meta['epoch']} / {meta['global_step']}")

    print("=== files ===")
    print(f"input root        : {args.input}")
    print(f"pairs found       : {len(pairs)}")
    print(f"output root       : {args.out_dir}")

    progress = tqdm(pairs, desc="infer", dynamic_ncols=True)
    for mix_path, ref_path in progress:
        rel_dir = relative_to_best_effort(mix_path.parent, args.input)
        out_name = mix_path.name.replace("_mix.wav", args.suffix)
        if out_name == mix_path.name:
            out_name = mix_path.stem + args.suffix
        out_path = (args.out_dir / out_name) if args.flat else (args.out_dir / rel_dir / out_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            progress.set_postfix({"skip": out_path.name})
            continue

        mix_np, mix_sr = read_audio(mix_path)
        ref_np, ref_sr = read_audio(ref_path)
        if mix_sr != model.cfg.sample_rate or ref_sr != model.cfg.sample_rate:
            raise RuntimeError(
                f"Sample-rate mismatch for {mix_path.name}: mix/ref = {mix_sr}/{ref_sr}, expected {model.cfg.sample_rate}"
            )

        estimate_np, info = ola_infer_pair(
            model=model,
            mix_np=mix_np,
            ref_np=ref_np,
            device=device,
            amp_mode=args.amp,
            segment_sec=args.segment_sec,
            overlap_sec=args.overlap_sec,
        )

        sf.write(str(out_path), estimate_np, samplerate=model.cfg.sample_rate)
        progress.set_postfix({
            "chunks": info["num_chunks"],
            "dur_s": f"{info['duration_sec']:.1f}",
            "out": out_path.name,
        })

    print("done")


if __name__ == "__main__":
    main()
