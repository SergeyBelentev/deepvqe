from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from dub_separator_optimized import DubSeparator, DubSeparatorConfig


# ============================================================
# Indexing
# ============================================================


class SegmentIndexError(RuntimeError):
    pass


class StopTraining(Exception):
    pass


def _relpath_str(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)



def _info_frames_and_sr(path: Path) -> Tuple[int, int, int]:
    info = sf.info(str(path))
    return int(info.frames), int(info.samplerate), int(info.channels)



def _scan_mix_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*_mix.wav") if p.is_file())



def index_speech_segments(root: Path, expected_sr: int) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    mix_files = _scan_mix_files(root)
    if not mix_files:
        raise SegmentIndexError(f"No *_mix.wav files found under speech root: {root}")

    for mix_path in mix_files:
        stem = mix_path.name[:-len("_mix.wav")]
        ref_path = mix_path.with_name(f"{stem}_ref.wav")
        target_path = mix_path.with_name(f"{stem}_target.wav")
        if not ref_path.exists() or not target_path.exists():
            raise SegmentIndexError(
                f"Speech triplet incomplete for {mix_path}: expected {ref_path.name} and {target_path.name}"
            )

        mix_frames, mix_sr, mix_ch = _info_frames_and_sr(mix_path)
        ref_frames, ref_sr, ref_ch = _info_frames_and_sr(ref_path)
        tgt_frames, tgt_sr, tgt_ch = _info_frames_and_sr(target_path)

        if mix_sr != expected_sr or ref_sr != expected_sr or tgt_sr != expected_sr:
            raise SegmentIndexError(
                f"Sample-rate mismatch for speech segment {mix_path}: mix/ref/target sr = "
                f"{mix_sr}/{ref_sr}/{tgt_sr}, expected {expected_sr}"
            )
        if mix_ch < 1 or ref_ch < 1 or tgt_ch < 1:
            raise SegmentIndexError(f"Invalid channel count for speech segment {mix_path}")

        min_frames = min(mix_frames, ref_frames, tgt_frames)
        entries.append(
            {
                "kind": "speech",
                "id": stem,
                "mix_path": str(mix_path),
                "ref_path": str(ref_path),
                "target_path": str(target_path),
                "mix_frames": mix_frames,
                "ref_frames": ref_frames,
                "target_frames": tgt_frames,
                "trimmed_frames": min_frames,
                "relpath": _relpath_str(mix_path.parent, root),
            }
        )
    return entries



def index_nospeech_segments(root: Path, expected_sr: int) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    mix_files = _scan_mix_files(root)
    if not mix_files:
        raise SegmentIndexError(f"No *_mix.wav files found under nospeech root: {root}")

    for mix_path in mix_files:
        stem = mix_path.name[:-len("_mix.wav")]
        ref_path = mix_path.with_name(f"{stem}_ref.wav")
        if not ref_path.exists():
            raise SegmentIndexError(
                f"Nospeech pair incomplete for {mix_path}: expected {ref_path.name}"
            )

        mix_frames, mix_sr, mix_ch = _info_frames_and_sr(mix_path)
        ref_frames, ref_sr, ref_ch = _info_frames_and_sr(ref_path)

        if mix_sr != expected_sr or ref_sr != expected_sr:
            raise SegmentIndexError(
                f"Sample-rate mismatch for nospeech segment {mix_path}: mix/ref sr = "
                f"{mix_sr}/{ref_sr}, expected {expected_sr}"
            )
        if mix_ch < 1 or ref_ch < 1:
            raise SegmentIndexError(f"Invalid channel count for nospeech segment {mix_path}")

        min_frames = min(mix_frames, ref_frames)
        entries.append(
            {
                "kind": "nospeech",
                "id": stem,
                "mix_path": str(mix_path),
                "ref_path": str(ref_path),
                "target_path": None,
                "mix_frames": mix_frames,
                "ref_frames": ref_frames,
                "target_frames": 0,
                "trimmed_frames": min_frames,
                "relpath": _relpath_str(mix_path.parent, root),
            }
        )
    return entries



def summarize_index(entries: Sequence[Dict[str, Any]], sample_rate: int) -> Dict[str, Any]:
    total_frames = sum(int(e["trimmed_frames"]) for e in entries)
    hours = total_frames / float(sample_rate) / 3600.0
    return {
        "count": len(entries),
        "hours": hours,
        "min_sec": (min(int(e["trimmed_frames"]) for e in entries) / sample_rate) if entries else 0.0,
        "max_sec": (max(int(e["trimmed_frames"]) for e in entries) / sample_rate) if entries else 0.0,
        "mean_sec": (total_frames / max(len(entries), 1) / sample_rate) if entries else 0.0,
    }


def save_full_index_json(
    path: Path,
    *,
    speech_entries: Sequence[Dict[str, Any]],
    nospeech_entries: Sequence[Dict[str, Any]],
    sample_rate: int,
    speech_root: Optional[Path],
    nospeech_root: Optional[Path],
) -> None:
    payload = {
        "version": 1,
        "sample_rate": int(sample_rate),
        "speech_root": str(speech_root) if speech_root is not None else None,
        "nospeech_root": str(nospeech_root) if nospeech_root is not None else None,
        "speech_entries": list(speech_entries),
        "nospeech_entries": list(nospeech_entries),
        "speech_summary": summarize_index(speech_entries, sample_rate),
        "nospeech_summary": summarize_index(nospeech_entries, sample_rate),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_loaded_entry(entry: Dict[str, Any], *, kind: str) -> Dict[str, Any]:
    required = ["id", "mix_path", "ref_path", "trimmed_frames"]
    for key in required:
        if key not in entry:
            raise SegmentIndexError(f"Loaded {kind} index entry is missing key: {key}")

    norm = {
        "kind": kind,
        "id": str(entry["id"]),
        "mix_path": str(entry["mix_path"]),
        "ref_path": str(entry["ref_path"]),
        "target_path": str(entry["target_path"]) if entry.get("target_path") is not None else None,
        "mix_frames": int(entry.get("mix_frames", entry["trimmed_frames"])),
        "ref_frames": int(entry.get("ref_frames", entry["trimmed_frames"])),
        "target_frames": int(entry.get("target_frames", 0)),
        "trimmed_frames": int(entry["trimmed_frames"]),
        "relpath": str(entry.get("relpath", "")),
    }
    return norm


def load_full_index_json(path: Path, *, expected_sr: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sr = int(payload.get("sample_rate", -1))
    if sr != int(expected_sr):
        raise SegmentIndexError(f"Index JSON sample_rate={sr} does not match expected sample_rate={expected_sr}: {path}")

    speech_entries = [_normalize_loaded_entry(e, kind="speech") for e in payload.get("speech_entries", [])]
    nospeech_entries = [_normalize_loaded_entry(e, kind="nospeech") for e in payload.get("nospeech_entries", [])]

    if not speech_entries and not nospeech_entries:
        raise SegmentIndexError(f"Loaded index JSON contains no entries: {path}")

    meta = {
        "version": int(payload.get("version", 1)),
        "sample_rate": sr,
        "speech_root": payload.get("speech_root"),
        "nospeech_root": payload.get("nospeech_root"),
        "speech_summary": payload.get("speech_summary"),
        "nospeech_summary": payload.get("nospeech_summary"),
    }
    return speech_entries, nospeech_entries, meta


# ============================================================
# Audio loading and dataset
# ============================================================



def _ensure_stereo(x: np.ndarray, path: str) -> np.ndarray:
    if x.ndim == 1:
        x = np.stack([x, x], axis=-1)
    if x.ndim != 2:
        raise SegmentIndexError(f"Expected 1D/2D audio from {path}, got shape {x.shape}")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] >= 2:
        x = x[:, :2]
    return x.astype(np.float32, copy=False)



def _read_audio(path: str) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = _ensure_stereo(wav, path)
    return wav, sr



def _trim_triplet_to_min(mix: np.ndarray, ref: np.ndarray, target: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = [mix, ref]
    if target is not None:
        arrays.append(target)
    min_len = min(x.shape[0] for x in arrays)
    mix = mix[:min_len]
    ref = ref[:min_len]
    if target is None:
        target = np.zeros((min_len, 2), dtype=np.float32)
    else:
        target = target[:min_len]
    return mix, ref, target


class MixedSegmentDataset(Dataset[Dict[str, Tensor]]):
    def __init__(
        self,
        *,
        speech_entries: Sequence[Dict[str, Any]],
        nospeech_entries: Sequence[Dict[str, Any]],
        sample_rate: int,
        segment_sec: float,
        epoch_steps: int,
        batch_size: int,
        speech_prob: float,
        nospeech_prob: float,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.speech_entries = list(speech_entries)
        self.nospeech_entries = list(nospeech_entries)
        self.sample_rate = int(sample_rate)
        self.segment_samples = int(round(segment_sec * sample_rate))
        self.epoch_steps = int(epoch_steps)
        self.batch_size = int(batch_size)
        self.virtual_length = self.epoch_steps * self.batch_size
        self.seed = int(seed)

        if self.segment_samples <= 0:
            raise ValueError("segment_sec must produce at least 1 sample")
        if not self.speech_entries and not self.nospeech_entries:
            raise ValueError("Both speech and nospeech indexes are empty")
        if speech_prob < 0 or nospeech_prob < 0:
            raise ValueError("speech_prob and nospeech_prob must be >= 0")
        total_prob = speech_prob + nospeech_prob
        if total_prob <= 0:
            raise ValueError("speech_prob + nospeech_prob must be > 0")
        self.speech_prob = speech_prob / total_prob
        self.nospeech_prob = nospeech_prob / total_prob

    def __len__(self) -> int:
        return self.virtual_length

    def _rng(self, index: int) -> random.Random:
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = worker_info.seed if worker_info is not None else self.seed
        return random.Random(worker_seed + index * 9973)

    def _pick_entry(self, rng: random.Random) -> Dict[str, Any]:
        want_speech = rng.random() < self.speech_prob
        if want_speech and self.speech_entries:
            return self.speech_entries[rng.randrange(len(self.speech_entries))]
        if (not want_speech) and self.nospeech_entries:
            return self.nospeech_entries[rng.randrange(len(self.nospeech_entries))]
        if self.speech_entries:
            return self.speech_entries[rng.randrange(len(self.speech_entries))]
        return self.nospeech_entries[rng.randrange(len(self.nospeech_entries))]

    def _load_triplet(self, entry: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mix, mix_sr = _read_audio(entry["mix_path"])
        ref, ref_sr = _read_audio(entry["ref_path"])
        if mix_sr != self.sample_rate or ref_sr != self.sample_rate:
            raise SegmentIndexError(
                f"Runtime sample-rate mismatch for {entry['id']}: mix/ref = {mix_sr}/{ref_sr}, expected {self.sample_rate}"
            )

        target_path = entry.get("target_path")
        if target_path:
            target, tgt_sr = _read_audio(target_path)
            if tgt_sr != self.sample_rate:
                raise SegmentIndexError(
                    f"Runtime sample-rate mismatch for {entry['id']}: target={tgt_sr}, expected {self.sample_rate}"
                )
        else:
            target = None

        mix, ref, target = _trim_triplet_to_min(mix, ref, target)
        return mix, ref, target

    def _crop_or_pad(
        self,
        mix: np.ndarray,
        ref: np.ndarray,
        target: np.ndarray,
        rng: random.Random,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        length = mix.shape[0]
        seg = self.segment_samples
        valid = np.zeros((seg,), dtype=np.float32)

        if length >= seg:
            start = rng.randint(0, length - seg)
            end = start + seg
            mix_out = mix[start:end]
            ref_out = ref[start:end]
            tgt_out = target[start:end]
            valid[:] = 1.0
            valid_len = seg
            return mix_out, ref_out, tgt_out, valid, valid_len

        offset = rng.randint(0, seg - length)
        mix_out = np.zeros((seg, 2), dtype=np.float32)
        ref_out = np.zeros((seg, 2), dtype=np.float32)
        tgt_out = np.zeros((seg, 2), dtype=np.float32)
        mix_out[offset:offset + length] = mix
        ref_out[offset:offset + length] = ref
        tgt_out[offset:offset + length] = target
        valid[offset:offset + length] = 1.0
        return mix_out, ref_out, tgt_out, valid, int(length)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        rng = self._rng(index)
        entry = self._pick_entry(rng)
        mix, ref, target = self._load_triplet(entry)
        mix, ref, target, valid, valid_len = self._crop_or_pad(mix, ref, target, rng)

        mix_t = torch.from_numpy(mix.T.copy())
        ref_t = torch.from_numpy(ref.T.copy())
        tgt_t = torch.from_numpy(target.T.copy())
        valid_t = torch.from_numpy(valid.copy())
        is_speech = 1 if entry["kind"] == "speech" else 0

        return {
            "mix": mix_t,
            "ref": ref_t,
            "target": tgt_t,
            "valid_mask": valid_t,
            "valid_samples": torch.tensor(valid_len, dtype=torch.long),
            "is_speech": torch.tensor(is_speech, dtype=torch.bool),
        }



def collate_segments(batch: Sequence[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    return {
        "mix": torch.stack([x["mix"] for x in batch], dim=0),
        "ref": torch.stack([x["ref"] for x in batch], dim=0),
        "target": torch.stack([x["target"] for x in batch], dim=0),
        "valid_mask": torch.stack([x["valid_mask"] for x in batch], dim=0),
        "valid_samples": torch.stack([x["valid_samples"] for x in batch], dim=0),
        "is_speech": torch.stack([x["is_speech"] for x in batch], dim=0),
    }


# ============================================================
# Losses
# ============================================================


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        *,
        resolutions: Sequence[Tuple[int, int, int]] | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.resolutions = list(resolutions or [
            (512, 128, 512),
            (1024, 256, 1024),
            (2048, 512, 2048),
        ])
        for n_fft, _, win_length in self.resolutions:
            win = torch.hann_window(win_length, periodic=True)
            self.register_buffer(f"window_{n_fft}_{win_length}", win, persistent=False)

    def _window(self, n_fft: int, win_length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        win = getattr(self, f"window_{n_fft}_{win_length}")
        return win.to(device=device, dtype=dtype)

    def _mag(self, x: Tensor, n_fft: int, hop: int, win_length: int) -> Tensor:
        # x: [B, C, S] -> [B*C, F, T]
        b, c, s = x.shape
        z = x.reshape(b * c, s)
        win = self._window(n_fft, win_length, x.device, x.dtype)
        spec = torch.stft(
            z,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_length,
            window=win,
            center=True,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return spec.abs().clamp_min(self.eps)

    def forward(self, estimate: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        sc_losses: List[Tensor] = []
        logmag_losses: List[Tensor] = []
        for n_fft, hop, win_length in self.resolutions:
            est_mag = self._mag(estimate, n_fft, hop, win_length)
            tgt_mag = self._mag(target, n_fft, hop, win_length)

            diff = est_mag - tgt_mag
            num = torch.linalg.vector_norm(diff, ord=2, dim=(1, 2))
            den = torch.linalg.vector_norm(tgt_mag, ord=2, dim=(1, 2)).clamp_min(self.eps)
            sc = (num / den).mean()
            logmag = (torch.log(est_mag) - torch.log(tgt_mag)).abs().mean()

            sc_losses.append(sc)
            logmag_losses.append(logmag)

        return torch.stack(sc_losses).mean(), torch.stack(logmag_losses).mean()



def masked_l1_loss(estimate: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    # estimate/target: [B,2,S], valid_mask: [B,S]
    w = valid_mask.unsqueeze(1)
    denom = w.sum().clamp_min(1.0) * estimate.size(1)
    return ((estimate - target).abs() * w).sum() / denom



def apply_valid_mask(x: Tensor, valid_mask: Tensor) -> Tensor:
    return x * valid_mask.unsqueeze(1)


# ============================================================
# Checkpointing
# ============================================================



def save_checkpoint(
    path: Path,
    *,
    model: DubSeparator,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "model_cfg": asdict(model.cfg),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, tmp_path)
    except RuntimeError:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        torch.save(payload, tmp_path, _use_new_zipfile_serialization=False)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: Path,
    *,
    model: DubSeparator,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
) -> Tuple[int, int]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    return epoch, global_step


# ============================================================
# Utils
# ============================================================



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)



def human_hours(hours: float) -> str:
    return f"{hours:.2f} h"



def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def grad_norm_l2(module: nn.Module) -> float:
    total_sq = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total_sq += float(g.float().pow(2).sum().item())
    return math.sqrt(max(total_sq, 0.0))


def safe_output_stats(outputs: Dict[str, Tensor]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    crm_gate = outputs.get("crm_gate")
    if crm_gate is not None:
        stats["crm_gate_mean"] = float(crm_gate.detach().mean().item())
    mask = outputs.get("mask")
    if mask is not None:
        stats["mask_abs_mean"] = float(mask.detach().abs().mean().item())
    est = outputs.get("estimate_waveform")
    if est is not None:
        stats["est_abs_mean"] = float(est.detach().abs().mean().item())
    return stats


def update_ema_dict(ema: Dict[str, float], current: Dict[str, float], decay: float) -> Dict[str, float]:
    for k, v in current.items():
        if not math.isfinite(v):
            continue
        if k not in ema:
            ema[k] = float(v)
        else:
            ema[k] = decay * ema[k] + (1.0 - decay) * float(v)
    return ema



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
    return autocast(device_type=device.type if device.type in {"cuda", "cpu"} else "cuda", dtype=dtype, enabled=enabled)



def compute_losses(
    *,
    outputs: Dict[str, Tensor],
    batch: Dict[str, Tensor],
    mrstft: MultiResolutionSTFTLoss,
    lambda_l1: float,
    lambda_sc: float,
    lambda_logmag: float,
) -> Tuple[Tensor, Dict[str, float]]:
    # Losses are always computed in fp32, outside autocast.
    estimate = outputs["estimate_waveform"].float()
    target = batch["target"].float()
    valid_mask = batch["valid_mask"].float()
    is_speech = batch["is_speech"].bool()

    total_loss = estimate.new_zeros((), dtype=torch.float32)
    logs: Dict[str, float] = {
        "loss_total": 0.0,
        "loss_speech": 0.0,
        "loss_nospeech": 0.0,
        "l1_speech": 0.0,
        "sc_speech": 0.0,
        "logmag_speech": 0.0,
        "l1_nospeech": 0.0,
        "logmag_nospeech": 0.0,
        "num_speech": float(is_speech.sum().item()),
        "num_nospeech": float((~is_speech).sum().item()),
    }

    if is_speech.any():
        idx = is_speech
        est_s = apply_valid_mask(estimate[idx], valid_mask[idx])
        tgt_s = apply_valid_mask(target[idx], valid_mask[idx])
        l1_s = masked_l1_loss(estimate[idx], target[idx], valid_mask[idx])
        sc_s, logmag_s = mrstft(est_s, tgt_s)
        speech_loss = lambda_l1 * l1_s + lambda_sc * sc_s + lambda_logmag * logmag_s
        total_loss = total_loss + speech_loss
        logs.update(
            {
                "loss_speech": float(speech_loss.detach().item()),
                "l1_speech": float(l1_s.detach().item()),
                "sc_speech": float(sc_s.detach().item()),
                "logmag_speech": float(logmag_s.detach().item()),
            }
        )

    if (~is_speech).any():
        idx = ~is_speech
        est_n = apply_valid_mask(estimate[idx], valid_mask[idx])
        tgt_n = apply_valid_mask(target[idx], valid_mask[idx])
        l1_n = masked_l1_loss(estimate[idx], target[idx], valid_mask[idx])
        _, logmag_n = mrstft(est_n, tgt_n)
        nospeech_loss = lambda_l1 * l1_n + lambda_logmag * logmag_n
        total_loss = total_loss + nospeech_loss
        logs.update(
            {
                "loss_nospeech": float(nospeech_loss.detach().item()),
                "l1_nospeech": float(l1_n.detach().item()),
                "logmag_nospeech": float(logmag_n.detach().item()),
            }
        )

    logs["loss_total"] = float(total_loss.detach().item())
    return total_loss, logs


# ============================================================
# Training loop
# ============================================================



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DubSeparator on speech + nospeech segments")
    p.add_argument("--speech-root", type=Path, required=True, help="Path to segs_voice root")
    p.add_argument("--nospeech-root", type=Path, required=True, help="Path to segs root")
    p.add_argument("--out-dir", type=Path, required=True, help="Directory for checkpoints and logs")
    p.add_argument("--index-json", type=Path, default=None, help="Load full speech/nospeech index from JSON instead of re-indexing WAV files")
    p.add_argument("--save-index-json", type=Path, default=None, help="Where to save full speech/nospeech index JSON after fresh indexing (default: <out-dir>/index_full.json)")

    p.add_argument("--segment-sec", type=float, default=4.0, help="Fixed model input segment length in seconds")
    p.add_argument("--batch", type=int, default=2, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16", "fp16"], help="AMP mode")
    p.add_argument("--tf32", action="store_true", help="Enable TF32 on CUDA")
    p.add_argument("--save-every-epoch", type=int, default=1, help="Save numbered checkpoint every N epochs")
    p.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume from")

    p.add_argument("--loss-l1", type=float, default=1.0, help="Waveform L1 coefficient")
    p.add_argument("--loss-mr-sc", type=float, default=1.0, help="MR spectral convergence coefficient (speech only)")
    p.add_argument("--loss-mr-logmag", type=float, default=1.0, help="MR log-magnitude coefficient")

    p.add_argument("--epoch-size", type=int, default=1000, help="Number of optimizer steps per epoch")
    p.add_argument("--speech-prob", type=float, default=0.8, help="Sampling probability weight for speech examples")
    p.add_argument("--nospeech-prob", type=float, default=0.2, help="Sampling probability weight for nospeech examples")

    p.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Global random seed")
    p.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    p.add_argument("--grad-checkpoint", action="store_true", help="Enable activation checkpointing in the optimized model")
    return p



def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    cfg = DubSeparatorConfig()
    cfg.gradient_checkpointing = bool(args.grad_checkpoint)
    sample_rate = cfg.sample_rate

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== indexing ===")
    if args.index_json is not None:
        speech_entries, nospeech_entries, loaded_meta = load_full_index_json(args.index_json, expected_sr=sample_rate)
        speech_stats = summarize_index(speech_entries, sample_rate)
        nospeech_stats = summarize_index(nospeech_entries, sample_rate)
        print(f"loaded index json: {args.index_json}")
    else:
        speech_entries = index_speech_segments(args.speech_root, sample_rate)
        nospeech_entries = index_nospeech_segments(args.nospeech_root, sample_rate)
        speech_stats = summarize_index(speech_entries, sample_rate)
        nospeech_stats = summarize_index(nospeech_entries, sample_rate)

        index_json_path = args.save_index_json if args.save_index_json is not None else (args.out_dir / "index_full.json")
        save_full_index_json(
            index_json_path,
            speech_entries=speech_entries,
            nospeech_entries=nospeech_entries,
            sample_rate=sample_rate,
            speech_root=args.speech_root,
            nospeech_root=args.nospeech_root,
        )
        print(f"saved full index json: {index_json_path}")

    print(f"speech   : {speech_stats['count']} items, {human_hours(speech_stats['hours'])}, "
          f"min/mean/max = {speech_stats['min_sec']:.2f}/{speech_stats['mean_sec']:.2f}/{speech_stats['max_sec']:.2f} sec")
    print(f"nospeech : {nospeech_stats['count']} items, {human_hours(nospeech_stats['hours'])}, "
          f"min/mean/max = {nospeech_stats['min_sec']:.2f}/{nospeech_stats['mean_sec']:.2f}/{nospeech_stats['max_sec']:.2f} sec")

    index_dump = {
        "speech": speech_stats,
        "nospeech": nospeech_stats,
        "sample_rate": sample_rate,
        "segment_sec": args.segment_sec,
        "speech_root": str(args.speech_root),
        "nospeech_root": str(args.nospeech_root),
        "index_json": str(args.index_json) if args.index_json is not None else None,
    }
    (args.out_dir / "index_summary.json").write_text(json.dumps(index_dump, indent=2, ensure_ascii=False), encoding="utf-8")

    dataset = MixedSegmentDataset(
        speech_entries=speech_entries,
        nospeech_entries=nospeech_entries,
        sample_rate=sample_rate,
        segment_sec=args.segment_sec,
        epoch_steps=args.epoch_size,
        batch_size=args.batch,
        speech_prob=args.speech_prob,
        nospeech_prob=args.nospeech_prob,
        seed=args.seed,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_segments,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=(args.num_workers * 2) if args.num_workers > 0 else None,
    )

    model = DubSeparator(cfg).to(device)
    model.set_gradient_checkpointing(bool(args.grad_checkpoint))
    try:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=(device.type == "cuda"))
    except TypeError:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mrstft = MultiResolutionSTFTLoss().to(device)

    use_scaler = args.amp.lower() == "fp16" and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else None

    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        start_epoch, global_step = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        print(f"Resumed from {args.resume} at epoch={start_epoch}, global_step={global_step}")

    print("=== training ===")
    print(f"device            : {device}")
    print(f"params trainable  : {count_parameters(model):,}")
    print(f"segment_sec       : {args.segment_sec}")
    print(f"batch             : {args.batch}")
    print(f"epoch_size steps  : {args.epoch_size}")
    print(f"amp               : {args.amp}")
    print(f"tf32              : {bool(args.tf32)}")
    print(f"loss coeffs       : l1={args.loss_l1} mr_sc={args.loss_mr_sc} mr_logmag={args.loss_mr_logmag}")
    print(f"speech/nospeech   : {args.speech_prob} / {args.nospeech_prob}")
    print(f"grad checkpoint   : {bool(args.grad_checkpoint)}")

    stop_requested = {"flag": False}

    def _handle_signal(signum, frame):  # type: ignore[no-untyped-def]
        stop_requested["flag"] = True
        print(f"\nSignal {signum} received. Will stop after current step and save checkpoint.")

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    epoch = start_epoch
    try:
        while True:
            epoch += 1
            model.train()
            epoch_start = time.time()
            running: Dict[str, float] = {
                "loss_total": 0.0,
                "loss_speech": 0.0,
                "loss_nospeech": 0.0,
                "l1_speech": 0.0,
                "sc_speech": 0.0,
                "logmag_speech": 0.0,
                "l1_nospeech": 0.0,
                "logmag_nospeech": 0.0,
                "num_speech": 0.0,
                "num_nospeech": 0.0,
            }

            ema: Dict[str, float] = {}
            progress = tqdm(loader, total=args.epoch_size, desc=f"epoch {epoch:05d}", dynamic_ncols=True, leave=True)
            for step_idx, batch in enumerate(progress, start=1):
                if stop_requested["flag"]:
                    raise StopTraining

                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)

                with make_autocast(device, args.amp):
                    outputs = model(batch["mix"], batch["ref"])

                with autocast(device_type=device.type if device.type in {"cuda", "cpu"} else "cuda", enabled=False):
                    loss, logs = compute_losses(
                        outputs=outputs,
                        batch=batch,
                        mrstft=mrstft,
                        lambda_l1=args.loss_l1,
                        lambda_sc=args.loss_mr_sc,
                        lambda_logmag=args.loss_mr_logmag,
                    )

                should_log = (step_idx % args.log_every == 0) or (step_idx == 1) or (step_idx == args.epoch_size)

                if use_scaler:
                    assert scaler is not None
                    scaler.scale(loss).backward()
                    if should_log:
                        scaler.unscale_(optimizer)
                        grad_norm = grad_norm_l2(model)
                    else:
                        grad_norm = float("nan")
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = grad_norm_l2(model) if should_log else float("nan")
                    optimizer.step()

                global_step += 1
                for k in running:
                    running[k] += logs.get(k, 0.0)

                if should_log:
                    extra_stats = safe_output_stats(outputs)
                    step_stats = {**logs, **extra_stats}
                    step_stats["grad_norm"] = float(grad_norm)
                    step_stats["lr"] = float(optimizer.param_groups[0]["lr"])
                    total_in_batch = logs.get("num_speech", 0.0) + logs.get("num_nospeech", 0.0)
                    step_stats["speech_frac"] = float(logs.get("num_speech", 0.0) / total_in_batch) if total_in_batch > 0 else 0.0
                    ema = update_ema_dict(ema, step_stats, decay=0.95)

                    progress.set_postfix({
                        "loss": f"{logs['loss_total']:.4f}",
                        "ema": f"{ema.get('loss_total', logs['loss_total']):.4f}",
                        "sp": f"{logs.get('loss_speech', 0.0):.3f}",
                        "ns": f"{logs.get('loss_nospeech', 0.0):.3f}",
                        "l1": f"{(logs.get('l1_speech', 0.0) + logs.get('l1_nospeech', 0.0)):.3f}",
                        "sc": f"{logs.get('sc_speech', 0.0):.3f}",
                        "logm": f"{(logs.get('logmag_speech', 0.0) + logs.get('logmag_nospeech', 0.0)):.3f}",
                        "gate": f"{extra_stats.get('crm_gate_mean', float('nan')):.3f}",
                        "|M|": f"{extra_stats.get('mask_abs_mean', float('nan')):.3f}",
                        "gn": f"{grad_norm:.2f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "sf": f"{step_stats['speech_frac']:.2f}",
                    })

                    avg_loss = running["loss_total"] / step_idx
                    print(
                        f"epoch={epoch:05d} step={step_idx:05d}/{args.epoch_size:05d} global_step={global_step:08d} "
                        f"loss={logs['loss_total']:.6f} avg={avg_loss:.6f} "
                        f"speech={int(logs['num_speech'])} nospeech={int(logs['num_nospeech'])} "
                        f"grad_norm={grad_norm:.4f} lr={optimizer.param_groups[0]['lr']:.3e} "
                        f"crm_gate={extra_stats.get('crm_gate_mean', float('nan')):.4f} mask_abs={extra_stats.get('mask_abs_mean', float('nan')):.4f}"
                    )
            progress.close()

            elapsed = time.time() - epoch_start
            denom = float(args.epoch_size)
            print(
                f"[epoch {epoch:05d}] "
                f"loss={running['loss_total']/denom:.6f} "
                f"speech_loss={running['loss_speech']/denom:.6f} "
                f"nospeech_loss={running['loss_nospeech']/denom:.6f} "
                f"l1_speech={running['l1_speech']/denom:.6f} "
                f"sc_speech={running['sc_speech']/denom:.6f} "
                f"logmag_speech={running['logmag_speech']/denom:.6f} "
                f"l1_nospeech={running['l1_nospeech']/denom:.6f} "
                f"logmag_nospeech={running['logmag_nospeech']/denom:.6f} "
                f"speech_count/step={running['num_speech']/denom:.3f} "
                f"nospeech_count/step={running['num_nospeech']/denom:.3f} "
                f"ema_loss={ema.get('loss_total', 0.0):.6f} "
                f"ema_gate={ema.get('crm_gate_mean', float('nan')):.6f} "
                f"ema_mask_abs={ema.get('mask_abs_mean', float('nan')):.6f} "
                f"ema_grad_norm={ema.get('grad_norm', float('nan')):.6f} "
                f"time={elapsed:.1f}s"
            )

            last_ckpt = args.out_dir / "last.pt"
            save_checkpoint(
                last_ckpt,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                args=args,
            )
            if args.save_every_epoch > 0 and epoch % args.save_every_epoch == 0:
                numbered = args.out_dir / f"epoch_{epoch:05d}.pt"
                save_checkpoint(
                    numbered,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    args=args,
                )

    except StopTraining:
        print("Stopping requested. Saving interrupt checkpoint...")
        save_checkpoint(
            args.out_dir / "interrupt.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            args=args,
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt. Saving interrupt checkpoint...")
        save_checkpoint(
            args.out_dir / "interrupt.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            args=args,
        )
        raise


if __name__ == "__main__":
    main()
