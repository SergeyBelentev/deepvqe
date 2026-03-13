
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, fields
from pathlib import Path
import pathlib
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.amp import autocast
from tqdm.auto import tqdm

from dub_separator_optimized import BandSpec, DubSeparator, DubSeparatorConfig

if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath   # type: ignore[attr-defined]


def build_amp_dtype(amp_mode: str) -> torch.dtype | None:
    amp_mode = amp_mode.lower()
    if amp_mode == "none":
        return None
    if amp_mode == "bf16":
        return torch.bfloat16
    if amp_mode == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported amp mode: {amp_mode}")


def make_autocast(device: torch.device, amp_mode: str):
    dtype = build_amp_dtype(amp_mode)
    enabled = dtype is not None and device.type == "cuda"
    return autocast(
        device_type=device.type if device.type in {"cuda", "cpu"} else "cuda",
        dtype=dtype,
        enabled=enabled,
    )


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


def chunks(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


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
        model = DubSeparator(DubSeparatorConfig())
        model.load_state_dict(payload, strict=True)
        meta = {"epoch": 0, "global_step": 0, "has_train_payload": False}
        return model.to(device), meta

    raise RuntimeError(f"Unsupported checkpoint payload type: {type(payload)!r}")


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


def make_ola_window(num_samples: int) -> np.ndarray:
    if num_samples <= 1:
        return np.ones((num_samples,), dtype=np.float32)
    win = torch.hann_window(num_samples, periodic=False).cpu().numpy().astype(np.float32, copy=False)
    win[0] = max(float(win[0]), 1e-3)
    win[-1] = max(float(win[-1]), 1e-3)
    return win


def build_chunk_starts(total_samples: int, chunk_samples: int, hop_samples: int) -> List[int]:
    if total_samples <= chunk_samples:
        return [0]
    starts = list(range(0, max(total_samples - chunk_samples, 0) + 1, hop_samples))
    last = total_samples - chunk_samples
    if starts[-1] != last:
        starts.append(last)
    return starts


@dataclass
class PairState:
    mix_path: Path
    ref_path: Path
    out_path: Path
    mix_np: np.ndarray
    ref_np: np.ndarray
    total_samples: int
    starts: List[int]
    out_acc: np.ndarray
    weight_acc: np.ndarray
    duration_sec: float


def load_pair_states(
    pair_batch: Sequence[Tuple[Path, Path]],
    *,
    model_sr: int,
    input_root: Path,
    out_dir: Path,
    suffix: str,
    flat: bool,
    overwrite: bool,
    segment_sec: float,
    overlap_sec: float,
) -> Tuple[List[PairState], int]:
    chunk_samples = int(round(segment_sec * model_sr))
    overlap_samples = int(round(overlap_sec * model_sr))
    hop_samples = chunk_samples - overlap_samples

    states: List[PairState] = []
    skipped_count = 0

    for mix_path, ref_path in pair_batch:
        rel_dir = relative_to_best_effort(mix_path.parent, input_root)
        out_name = mix_path.name.replace("_mix.wav", suffix)
        if out_name == mix_path.name:
            out_name = mix_path.stem + suffix
        out_path = (out_dir / out_name) if flat else (out_dir / rel_dir / out_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            skipped_count += 1
            continue

        mix_np, mix_sr = read_audio(mix_path)
        ref_np, ref_sr = read_audio(ref_path)
        if mix_sr != model_sr or ref_sr != model_sr:
            raise RuntimeError(
                f"Sample-rate mismatch for {mix_path.name}: mix/ref = {mix_sr}/{ref_sr}, expected {model_sr}"
            )

        mix_np, ref_np = trim_pair_to_min(mix_np, ref_np)
        total_samples = int(mix_np.shape[0])
        starts = build_chunk_starts(total_samples, chunk_samples, hop_samples)

        states.append(
            PairState(
                mix_path=mix_path,
                ref_path=ref_path,
                out_path=out_path,
                mix_np=mix_np,
                ref_np=ref_np,
                total_samples=total_samples,
                starts=starts,
                out_acc=np.zeros((2, total_samples), dtype=np.float32),
                weight_acc=np.zeros((total_samples,), dtype=np.float32),
                duration_sec=total_samples / float(model_sr),
            )
        )

    return states, skipped_count


@torch.inference_mode()
def process_states_batched(
    *,
    model: DubSeparator,
    states: Sequence[PairState],
    device: torch.device,
    amp_mode: str,
    segment_sec: float,
    overlap_sec: float,
    batch_size: int,
) -> Dict[str, Any]:
    sr = int(model.cfg.sample_rate)
    chunk_samples = int(round(segment_sec * sr))
    overlap_samples = int(round(overlap_sec * sr))
    hop_samples = chunk_samples - overlap_samples
    window = make_ola_window(chunk_samples)

    work_items: List[Tuple[int, int, int]] = []
    for state_idx, st in enumerate(states):
        for start in st.starts:
            end = min(start + chunk_samples, st.total_samples)
            work_items.append((state_idx, start, end - start))

    total_chunks = len(work_items)

    for batch_items in chunks(work_items, batch_size):
        bs = len(batch_items)
        mix_batch = np.zeros((bs, chunk_samples, 2), dtype=np.float32)
        ref_batch = np.zeros((bs, chunk_samples, 2), dtype=np.float32)

        for bi, (state_idx, start, current_len) in enumerate(batch_items):
            st = states[state_idx]
            mix_batch[bi, :current_len] = st.mix_np[start:start + current_len]
            ref_batch[bi, :current_len] = st.ref_np[start:start + current_len]

        mix_t = torch.from_numpy(mix_batch).permute(0, 2, 1).contiguous().to(device, non_blocking=True)
        ref_t = torch.from_numpy(ref_batch).permute(0, 2, 1).contiguous().to(device, non_blocking=True)

        with make_autocast(device, amp_mode):
            outputs = model(mix_t, ref_t)

        est = outputs["estimate_waveform"].float().detach().cpu().numpy()

        for bi, (state_idx, start, current_len) in enumerate(batch_items):
            st = states[state_idx]
            w = window[:current_len]
            st.out_acc[:, start:start + current_len] += est[bi, :, :current_len] * w[None, :]
            st.weight_acc[start:start + current_len] += w

    for st in states:
        estimate = st.out_acc / np.clip(st.weight_acc[None, :], 1e-6, None)
        estimate_np = estimate.T.astype(np.float32, copy=False)
        sf.write(str(st.out_path), estimate_np, samplerate=sr)

    return {
        "num_files": len(states),
        "num_chunks": total_chunks,
        "segment_samples": chunk_samples,
        "overlap_samples": overlap_samples,
        "hop_samples": hop_samples,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batched OLA inference for DubSeparator")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to train checkpoint or raw model state_dict")
    p.add_argument("--input", type=Path, required=True, help="Input folder containing *_mix.wav/*_ref.wav pairs (recursive)")
    p.add_argument("--out-dir", type=Path, required=True, help="Output folder for estimated wav files")
    p.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16", "fp16"], help="AMP mode")
    p.add_argument("--tf32", action="store_true", help="Enable TF32 on CUDA")
    p.add_argument("--segment-sec", type=float, default=4.0, help="Chunk size in seconds for OLA")
    p.add_argument("--overlap-sec", type=float, default=1.0, help="Overlap size in seconds for OLA")
    p.add_argument("--batch-size", type=int, default=4, help="Model batch size in chunks")
    p.add_argument("--file-batch-size", type=int, default=16, help="How many file-pairs to load/process together")
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

    # Determine all pairs at startup.
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
    print(f"batch_size        : {args.batch_size}")
    print(f"file_batch_size   : {args.file_batch_size}")
    print(f"n_fft / hop       : {model.cfg.n_fft} / {model.cfg.hop_length}")
    print(f"train payload     : {meta['has_train_payload']}")
    if meta["has_train_payload"]:
        print(f"epoch/global_step : {meta['epoch']} / {meta['global_step']}")

    print("=== files ===")
    print(f"input root        : {args.input}")
    print(f"pairs found       : {len(pairs)}")
    print(f"output root       : {args.out_dir}")

    file_progress = tqdm(total=len(pairs), desc="files", dynamic_ncols=True)
    total_chunks = 0
    processed_files = 0

    for pair_batch in chunks(pairs, args.file_batch_size):
        states, skipped_count = load_pair_states(
            pair_batch,
            model_sr=int(model.cfg.sample_rate),
            input_root=args.input,
            out_dir=args.out_dir,
            suffix=args.suffix,
            flat=bool(args.flat),
            overwrite=bool(args.overwrite),
            segment_sec=float(args.segment_sec),
            overlap_sec=float(args.overlap_sec),
        )

        if skipped_count:
            file_progress.update(skipped_count)
            file_progress.set_postfix({"skip": skipped_count})

        if not states:
            continue

        info = process_states_batched(
            model=model,
            states=states,
            device=device,
            amp_mode=args.amp,
            segment_sec=float(args.segment_sec),
            overlap_sec=float(args.overlap_sec),
            batch_size=int(args.batch_size),
        )

        processed_files += info["num_files"]
        total_chunks += info["num_chunks"]
        file_progress.update(info["num_files"])
        file_progress.set_postfix(
            {
                "chunk_bs": args.batch_size,
                "file_bs": args.file_batch_size,
                "chunks": total_chunks,
                "last_files": info["num_files"],
            }
        )

    file_progress.close()

    print("=== done ===")
    print(f"processed files   : {processed_files}")
    print(f"processed chunks  : {total_chunks}")


if __name__ == "__main__":
    main()
