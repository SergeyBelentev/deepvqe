from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random
import signal
import time
from collections import OrderedDict
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

import os
import pathlib

if os.name != "nt":
    pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[attr-defined]

# Optional backends
try:
    import torchaudio  # type: ignore
    import torchaudio.functional as TAF  # type: ignore

    HAS_TORCHAUDIO = True
except Exception:
    torchaudio = None
    TAF = None
    HAS_TORCHAUDIO = False

try:
    from pedalboard import (  # type: ignore
        Compressor,
        Gain,
        HighpassFilter,
        Limiter,
        LowpassFilter,
        PeakFilter,
        Pedalboard,
        Reverb,
    )

    HAS_PEDALBOARD = True
except Exception:
    HAS_PEDALBOARD = False


AUDIO_EXTS = {".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a", ".aac"}


class DatasetIndexError(RuntimeError):
    pass


class StopTraining(Exception):
    pass


def _ensure_stereo_np(x: np.ndarray, path: str) -> np.ndarray:
    if x.ndim == 1:
        x = np.stack([x, x], axis=-1)
    if x.ndim != 2:
        raise DatasetIndexError(f"Expected 1D/2D audio from {path}, got shape {x.shape}")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] >= 2:
        x = x[:, :2]
    return x.astype(np.float32, copy=False)


def _load_audio_any(path: Path) -> Tuple[np.ndarray, int]:
    path_str = str(path)

    try:
        wav, sr = sf.read(path_str, dtype="float32", always_2d=True)
        wav = _ensure_stereo_np(wav, path_str)
        return wav, int(sr)
    except Exception:
        pass

    if HAS_TORCHAUDIO:
        try:
            wav_t, sr = torchaudio.load(path_str)  # [C, T]
            if wav_t.dim() != 2:
                raise DatasetIndexError(f"torchaudio.load returned shape {tuple(wav_t.shape)} for {path}")
            if wav_t.size(0) == 1:
                wav_t = wav_t.repeat(2, 1)
            elif wav_t.size(0) >= 2:
                wav_t = wav_t[:2]
            wav = wav_t.transpose(0, 1).contiguous().cpu().numpy().astype(np.float32, copy=False)
            return wav, int(sr)
        except Exception as exc:
            raise DatasetIndexError(
                f"Could not decode audio file: {path}. soundfile and torchaudio both failed."
            ) from exc

    raise DatasetIndexError(f"Could not decode audio file: {path}. Install torchaudio for mp3/ogg support.")


def _audio_info_any(path: Path) -> Tuple[int, int, int]:
    path_str = str(path)

    try:
        info = sf.info(path_str)
        return int(info.frames), int(info.samplerate), int(info.channels)
    except Exception:
        pass

    if HAS_TORCHAUDIO:
        try:
            info = torchaudio.info(path_str)
            channels = int(getattr(info, "num_channels", 2))
            frames = int(getattr(info, "num_frames", 0))
            sr = int(getattr(info, "sample_rate", 0))
            if frames > 0 and sr > 0:
                return frames, sr, channels
        except Exception:
            pass

    wav, sr = _load_audio_any(path)
    return int(wav.shape[0]), int(sr), int(wav.shape[1])


def _resample_np(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32, copy=False)

    if HAS_TORCHAUDIO and TAF is not None:
        x = torch.from_numpy(audio.T.copy())
        y = TAF.resample(x, sr_in, sr_out)
        return y.transpose(0, 1).contiguous().cpu().numpy().astype(np.float32, copy=False)

    x = torch.from_numpy(audio.T.copy()).unsqueeze(0)
    new_len = max(1, int(round(audio.shape[0] * float(sr_out) / float(sr_in))))
    y = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
    return y.squeeze(0).transpose(0, 1).contiguous().numpy().astype(np.float32, copy=False)


def _read_audio_segment(
    path: Path,
    *,
    start_frame: int,
    num_frames: int,
    target_sr: int,
) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    try:
        info = sf.info(str(path))
        src_frames = int(info.frames)
        src_sr = int(info.samplerate)

        left_pad = max(0, -start_frame)
        read_start = max(0, start_frame)
        read_end = min(src_frames, start_frame + num_frames)
        read_count = max(0, read_end - read_start)

        if read_count > 0:
            with sf.SoundFile(str(path), "r") as f:
                f.seek(read_start)
                wav = f.read(read_count, dtype="float32", always_2d=True)
            wav = _ensure_stereo_np(wav, str(path))
        else:
            wav = np.zeros((0, 2), dtype=np.float32)

        right_pad = max(0, (start_frame + num_frames) - src_frames)
        if left_pad > 0 or right_pad > 0:
            out = np.zeros((num_frames, 2), dtype=np.float32)
            if wav.shape[0] > 0:
                out[left_pad:left_pad + wav.shape[0]] = wav
            wav = out

        if src_sr != target_sr:
            wav = _resample_np(wav, src_sr, target_sr)
            want = max(1, int(round(num_frames * float(target_sr) / float(src_sr))))
            if wav.shape[0] < want:
                pad = np.zeros((want - wav.shape[0], 2), dtype=np.float32)
                wav = np.concatenate([wav, pad], axis=0)
            elif wav.shape[0] > want:
                wav = wav[:want]
        return wav.astype(np.float32, copy=False)
    except Exception:
        pass

    full, src_sr = _load_audio_any(path)
    if src_sr != target_sr:
        full = _resample_np(full, src_sr, target_sr)

    src_frames = full.shape[0]
    left_pad = max(0, -start_frame)
    read_start = max(0, start_frame)
    read_end = min(src_frames, start_frame + num_frames)
    wav = full[read_start:read_end]
    right_pad = max(0, (start_frame + num_frames) - src_frames)
    if left_pad > 0 or right_pad > 0:
        out = np.zeros((num_frames, 2), dtype=np.float32)
        if wav.shape[0] > 0:
            out[left_pad:left_pad + wav.shape[0]] = wav
        wav = out
    return wav.astype(np.float32, copy=False)


def _scan_audio_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def _duration_sec(frames: int, sr: int) -> float:
    return float(frames) / float(max(sr, 1))


def index_ref_files(root: Path) -> List[Dict[str, Any]]:
    files = _scan_audio_files(root)
    if not files:
        raise DatasetIndexError(f"No audio files found under ref root: {root}")

    entries: List[Dict[str, Any]] = []
    for path in files:
        frames, sr, ch = _audio_info_any(path)
        if frames <= 0 or sr <= 0:
            continue
        entries.append(
            {
                "path": str(path),
                "frames": int(frames),
                "sample_rate": int(sr),
                "channels": int(ch),
                "duration_sec": _duration_sec(frames, sr),
            }
        )
    if not entries:
        raise DatasetIndexError(f"No valid audio files indexed under ref root: {root}")
    return entries


def index_tts_files(root: Path) -> List[Dict[str, Any]]:
    files = _scan_audio_files(root)
    if not files:
        raise DatasetIndexError(f"No audio files found under TTS root: {root}")

    entries: List[Dict[str, Any]] = []
    for path in files:
        frames, sr, ch = _audio_info_any(path)
        if frames <= 0 or sr <= 0:
            continue
        entries.append(
            {
                "path": str(path),
                "frames": int(frames),
                "sample_rate": int(sr),
                "channels": int(ch),
                "duration_sec": _duration_sec(frames, sr),
            }
        )
    if not entries:
        raise DatasetIndexError(f"No valid audio files indexed under TTS root: {root}")
    return entries


def summarize_index(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    durations = [float(e["duration_sec"]) for e in entries]
    total_hours = sum(durations) / 3600.0
    return {
        "count": len(entries),
        "hours": total_hours,
        "min_sec": min(durations) if durations else 0.0,
        "max_sec": max(durations) if durations else 0.0,
        "mean_sec": (sum(durations) / max(len(durations), 1)) if durations else 0.0,
    }


def save_index_json(
    path: Path,
    *,
    ref_entries: Sequence[Dict[str, Any]],
    tts_entries: Sequence[Dict[str, Any]],
    sample_rate: int,
    ref_root: Optional[Path],
    tts_root: Optional[Path],
) -> None:
    payload = {
        "version": 2,
        "sample_rate": int(sample_rate),
        "ref_root": str(ref_root) if ref_root is not None else None,
        "tts_root": str(tts_root) if tts_root is not None else None,
        "ref_entries": list(ref_entries),
        "tts_entries": list(tts_entries),
        "ref_summary": summarize_index(ref_entries),
        "tts_summary": summarize_index(tts_entries),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_index_json(path: Path, *, expected_sr: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sr = int(payload.get("sample_rate", -1))
    if sr != int(expected_sr):
        raise DatasetIndexError(f"Index JSON sample_rate={sr} does not match expected sample_rate={expected_sr}: {path}")

    ref_entries = list(payload.get("ref_entries", []))
    tts_entries = list(payload.get("tts_entries", []))
    if not ref_entries or not tts_entries:
        raise DatasetIndexError(f"Loaded index JSON contains empty ref or tts entries: {path}")

    meta = {
        "version": int(payload.get("version", 2)),
        "sample_rate": sr,
        "ref_root": payload.get("ref_root"),
        "tts_root": payload.get("tts_root"),
        "ref_summary": payload.get("ref_summary"),
        "tts_summary": payload.get("tts_summary"),
    }
    return ref_entries, tts_entries, meta


class WeightedPicker:
    def __init__(self, entries: Sequence[Dict[str, Any]], weights: Sequence[float]) -> None:
        if len(entries) != len(weights):
            raise ValueError("entries and weights length mismatch")
        if not entries:
            raise ValueError("empty entries")
        self.entries = list(entries)
        self.cum: List[float] = []
        total = 0.0
        for w in weights:
            total += max(float(w), 1e-6)
            self.cum.append(total)
        self.total = total

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        x = rng.random() * self.total
        idx = bisect.bisect_left(self.cum, x)
        if idx >= len(self.entries):
            idx = len(self.entries) - 1
        return self.entries[idx]


def rms_db(audio: np.ndarray, eps: float = 1e-8) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + eps))
    return 20.0 * math.log10(max(rms, eps))


def apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return (audio * (10.0 ** (gain_db / 20.0))).astype(np.float32, copy=False)


def trim_silence_tts(audio: np.ndarray, threshold: float = 0.002, pad_samples: int = 480) -> np.ndarray:
    env = np.max(np.abs(audio), axis=1)
    idx = np.where(env > threshold)[0]
    if idx.size == 0:
        return audio
    start = max(0, int(idx[0]) - pad_samples)
    end = min(audio.shape[0], int(idx[-1]) + pad_samples + 1)
    return audio[start:end]


def random_eq_fft(audio: np.ndarray, sr: int, rng: random.Random, *, role: str) -> np.ndarray:
    x = audio.T
    n = x.shape[1]
    spec = np.fft.rfft(x, axis=1)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    if role == "ref":
        low_gain = rng.uniform(-1.5, 1.0)
        high_gain = rng.uniform(-2.5, 1.5)
        bell_gain = rng.uniform(-2.0, 2.0)
    else:
        low_gain = rng.uniform(-1.0, 1.5)
        high_gain = rng.uniform(-1.0, 3.0)
        bell_gain = rng.uniform(-2.5, 3.0)

    low_fc = rng.uniform(120.0, 320.0)
    high_fc = rng.uniform(2500.0, 6500.0)
    bell_fc = rng.uniform(900.0, 3500.0)
    bell_q = rng.uniform(0.6, 1.6)

    low_curve = 1.0 / (1.0 + (freqs / max(low_fc, 1.0)) ** 2)
    high_curve = 1.0 - 1.0 / (1.0 + (freqs / max(high_fc, 1.0)) ** 2)
    logf = np.log(np.maximum(freqs, 20.0))
    bell_curve = np.exp(-0.5 * ((logf - math.log(bell_fc)) / max(0.15, 1.0 / bell_q)) ** 2)

    curve_db = low_gain * low_curve + high_gain * high_curve + bell_gain * bell_curve
    curve = (10.0 ** (curve_db / 20.0)).astype(np.float32)

    spec *= curve[None, :]
    out = np.fft.irfft(spec, n=n, axis=1).T
    return out.astype(np.float32, copy=False)


def simple_compressor(audio: np.ndarray, sr: int, rng: random.Random, *, role: str) -> np.ndarray:
    if role == "ref":
        threshold_db = rng.uniform(-26.0, -16.0)
        ratio = rng.uniform(1.6, 3.0)
        attack_ms = rng.uniform(8.0, 25.0)
        release_ms = rng.uniform(60.0, 180.0)
    else:
        threshold_db = rng.uniform(-28.0, -14.0)
        ratio = rng.uniform(2.0, 4.5)
        attack_ms = rng.uniform(3.0, 15.0)
        release_ms = rng.uniform(50.0, 160.0)

    eps = 1e-8
    mono = np.mean(np.abs(audio), axis=1)
    attack = math.exp(-1.0 / max(1.0, attack_ms * 1e-3 * sr))
    release = math.exp(-1.0 / max(1.0, release_ms * 1e-3 * sr))

    env = np.zeros_like(mono, dtype=np.float32)
    state = 0.0
    for i, x in enumerate(mono):
        coeff = attack if x > state else release
        state = coeff * state + (1.0 - coeff) * float(x)
        env[i] = state

    env_db = 20.0 * np.log10(np.maximum(env, eps))
    over_db = np.maximum(env_db - threshold_db, 0.0)
    gain_reduction_db = -over_db * (1.0 - 1.0 / ratio)
    gain = np.power(10.0, gain_reduction_db / 20.0).astype(np.float32)
    out = audio * gain[:, None]
    return out.astype(np.float32, copy=False)


def simple_reverb(audio: np.ndarray, sr: int, rng: random.Random, *, role: str) -> np.ndarray:
    wet = rng.uniform(0.01, 0.05 if role == "ref" else 0.08)
    if wet <= 0.0:
        return audio

    delays_ms = [rng.uniform(8.0, 20.0), rng.uniform(20.0, 45.0), rng.uniform(45.0, 80.0)]
    decays = [rng.uniform(0.25, 0.55), rng.uniform(0.15, 0.40), rng.uniform(0.08, 0.25)]

    out = audio.copy()
    for d_ms, decay in zip(delays_ms, decays):
        d = max(1, int(round(d_ms * 1e-3 * sr)))
        delayed = np.zeros_like(audio)
        delayed[d:] = audio[:-d] * decay
        out += wet * delayed

    return out.astype(np.float32, copy=False)


def soft_saturation(audio: np.ndarray, rng: random.Random, *, role: str) -> np.ndarray:
    drive = rng.uniform(1.02, 1.12 if role == "ref" else 1.20)
    x = audio * drive
    y = np.tanh(x)
    return y.astype(np.float32, copy=False)


def pedalboard_process(audio: np.ndarray, sr: int, rng: random.Random, *, role: str) -> np.ndarray:
    if not HAS_PEDALBOARD:
        raise RuntimeError("pedalboard backend requested but pedalboard is not installed")

    if role == "ref":
        board = Pedalboard(
            [
                Gain(gain_db=rng.uniform(-5.0, 0.0)),
                HighpassFilter(cutoff_frequency_hz=rng.uniform(40.0, 90.0)),
                LowpassFilter(cutoff_frequency_hz=rng.uniform(9000.0, 16000.0)),
                PeakFilter(
                    cutoff_frequency_hz=rng.uniform(800.0, 3200.0),
                    q=rng.uniform(0.6, 1.4),
                    gain_db=rng.uniform(-2.5, 2.0),
                ),
                Compressor(
                    threshold_db=rng.uniform(-26.0, -16.0),
                    ratio=rng.uniform(1.6, 3.0),
                    attack_ms=rng.uniform(8.0, 25.0),
                    release_ms=rng.uniform(60.0, 180.0),
                ),
                Reverb(
                    room_size=rng.uniform(0.05, 0.18),
                    damping=rng.uniform(0.3, 0.7),
                    wet_level=rng.uniform(0.01, 0.05),
                    dry_level=1.0,
                    width=1.0,
                    freeze_mode=0.0,
                ),
                Limiter(threshold_db=rng.uniform(-1.8, -0.6), release_ms=rng.uniform(30.0, 120.0)),
            ]
        )
    else:
        board = Pedalboard(
            [
                HighpassFilter(cutoff_frequency_hz=rng.uniform(50.0, 110.0)),
                LowpassFilter(cutoff_frequency_hz=rng.uniform(10000.0, 18000.0)),
                PeakFilter(
                    cutoff_frequency_hz=rng.uniform(1000.0, 4500.0),
                    q=rng.uniform(0.7, 1.6),
                    gain_db=rng.uniform(-2.5, 3.5),
                ),
                Compressor(
                    threshold_db=rng.uniform(-28.0, -14.0),
                    ratio=rng.uniform(2.0, 4.5),
                    attack_ms=rng.uniform(3.0, 15.0),
                    release_ms=rng.uniform(50.0, 160.0),
                ),
                Reverb(
                    room_size=rng.uniform(0.04, 0.20),
                    damping=rng.uniform(0.3, 0.8),
                    wet_level=rng.uniform(0.01, 0.08),
                    dry_level=1.0,
                    width=1.0,
                    freeze_mode=0.0,
                ),
                Limiter(threshold_db=rng.uniform(-1.5, -0.3), release_ms=rng.uniform(20.0, 80.0)),
            ]
        )

    x = audio.T.astype(np.float32, copy=False)
    y = board(x, sr)
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)
    return y.T.astype(np.float32, copy=False)


def native_process(audio: np.ndarray, sr: int, rng: random.Random, *, role: str) -> np.ndarray:
    x = audio.astype(np.float32, copy=False)
    if role == "ref":
        x = apply_gain_db(x, rng.uniform(-5.0, 0.0))
    x = random_eq_fft(x, sr, rng, role=role)
    x = simple_compressor(x, sr, rng, role=role)
    x = simple_reverb(x, sr, rng, role=role)
    x = soft_saturation(x, rng, role=role)
    return x.astype(np.float32, copy=False)


def process_branch(audio: np.ndarray, sr: int, rng: random.Random, *, role: str, backend: str) -> np.ndarray:
    if backend == "pedalboard":
        return pedalboard_process(audio, sr, rng, role=role)
    if backend == "native":
        return native_process(audio, sr, rng, role=role)
    if HAS_PEDALBOARD:
        try:
            return pedalboard_process(audio, sr, rng, role=role)
        except Exception:
            return native_process(audio, sr, rng, role=role)
    return native_process(audio, sr, rng, role=role)


class RuntimeSyntheticDubDataset(Dataset[Dict[str, Tensor]]):
    def __init__(
        self,
        *,
        ref_entries: Sequence[Dict[str, Any]],
        tts_entries: Sequence[Dict[str, Any]],
        sample_rate: int,
        segment_sec: float,
        epoch_steps: int,
        batch_size: int,
        seed: int = 42,
        fx_backend: str = "auto",
        max_shift_ms: float = 250.0,
        tts_cache_size: int = 128,
        min_pause_ms: float = 40.0,
        max_pause_ms: float = 180.0,
    ) -> None:
        super().__init__()
        self.ref_entries = list(ref_entries)
        self.tts_entries = list(tts_entries)
        self.sample_rate = int(sample_rate)
        self.segment_sec = float(segment_sec)
        self.segment_samples = int(round(self.segment_sec * self.sample_rate))
        self.epoch_steps = int(epoch_steps)
        self.batch_size = int(batch_size)
        self.virtual_length = self.epoch_steps * self.batch_size
        self.seed = int(seed)
        self.fx_backend = str(fx_backend).lower()
        self.max_shift_ms = float(max_shift_ms)
        self.tts_cache_size = int(tts_cache_size)
        self.min_pause_samples = int(round(min_pause_ms * 1e-3 * self.sample_rate))
        self.max_pause_samples = int(round(max_pause_ms * 1e-3 * self.sample_rate))

        if self.segment_samples <= 0:
            raise ValueError("segment_sec must produce at least 1 sample")
        if not self.ref_entries:
            raise ValueError("ref_entries is empty")
        if not self.tts_entries:
            raise ValueError("tts_entries is empty")
        if self.fx_backend not in {"auto", "pedalboard", "native"}:
            raise ValueError("fx_backend must be one of: auto, pedalboard, native")

        ref_weights = []
        for e in self.ref_entries:
            dur_tgt = float(e["duration_sec"]) * float(self.sample_rate)
            usable = max(1.0, dur_tgt - self.segment_samples + 1)
            ref_weights.append(usable)
        self.ref_picker = WeightedPicker(self.ref_entries, ref_weights)

        tts_weights = [max(float(e["duration_sec"]), 0.05) for e in self.tts_entries]
        self.tts_picker = WeightedPicker(self.tts_entries, tts_weights)

        self._tts_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return self.virtual_length

    def _rng(self, index: int) -> random.Random:
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = worker_info.seed if worker_info is not None else self.seed
        return random.Random(worker_seed + index * 10007)

    def _get_tts_audio(self, path: str) -> np.ndarray:
        if path in self._tts_cache:
            audio = self._tts_cache.pop(path)
            self._tts_cache[path] = audio
            return audio

        wav, sr = _load_audio_any(Path(path))
        if sr != self.sample_rate:
            wav = _resample_np(wav, sr, self.sample_rate)
        wav = trim_silence_tts(wav)
        if wav.shape[0] == 0:
            wav = np.zeros((1, 2), dtype=np.float32)

        self._tts_cache[path] = wav
        while len(self._tts_cache) > self.tts_cache_size:
            self._tts_cache.popitem(last=False)
        return wav

    def _sample_ref_pair(self, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
        entry = self.ref_picker.sample(rng)
        path = Path(entry["path"])
        src_sr = int(entry["sample_rate"])
        src_frames = int(entry["frames"])

        seg_src = max(1, int(round(self.segment_samples * float(src_sr) / float(self.sample_rate))))
        max_start = max(0, src_frames - seg_src)
        start_src = rng.randint(0, max_start) if max_start > 0 else 0

        ref_input = _read_audio_segment(path, start_frame=start_src, num_frames=seg_src, target_sr=self.sample_rate)

        max_shift_src = max(1, int(round(self.max_shift_ms * 1e-3 * src_sr)))
        shift_src = rng.randint(-max_shift_src, max_shift_src)
        mix_start_src = start_src + shift_src

        ref_for_mix = _read_audio_segment(path, start_frame=mix_start_src, num_frames=seg_src, target_sr=self.sample_rate)

        if ref_input.shape[0] < self.segment_samples:
            pad = np.zeros((self.segment_samples - ref_input.shape[0], 2), dtype=np.float32)
            ref_input = np.concatenate([ref_input, pad], axis=0)
        elif ref_input.shape[0] > self.segment_samples:
            ref_input = ref_input[:self.segment_samples]

        if ref_for_mix.shape[0] < self.segment_samples:
            pad = np.zeros((self.segment_samples - ref_for_mix.shape[0], 2), dtype=np.float32)
            ref_for_mix = np.concatenate([ref_for_mix, pad], axis=0)
        elif ref_for_mix.shape[0] > self.segment_samples:
            ref_for_mix = ref_for_mix[:self.segment_samples]

        return ref_input.astype(np.float32, copy=False), ref_for_mix.astype(np.float32, copy=False)

    def _assemble_target_dry(self, rng: random.Random) -> np.ndarray:
        seg = self.segment_samples
        out = np.zeros((seg, 2), dtype=np.float32)
        cursor = rng.randint(0, max(0, int(0.35 * self.sample_rate)))
        cursor = min(cursor, seg)

        while cursor < seg:
            entry = self.tts_picker.sample(rng)
            wav = self._get_tts_audio(entry["path"])
            if wav.shape[0] <= 0:
                break

            remaining = seg - cursor
            if remaining <= 0:
                break

            if wav.shape[0] > remaining:
                start = rng.randint(0, wav.shape[0] - remaining)
                chunk = wav[start:start + remaining]
            else:
                chunk = wav

            take = min(chunk.shape[0], remaining)
            out[cursor:cursor + take] = chunk[:take]
            cursor += take
            if cursor >= seg:
                break

            pause = rng.randint(self.min_pause_samples, max(self.min_pause_samples, self.max_pause_samples))
            cursor += pause

        return out.astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        rng = self._rng(index)

        ref_input, ref_for_mix = self._sample_ref_pair(rng)
        target_dry = self._assemble_target_dry(rng)

        ref_mix = process_branch(ref_for_mix, self.sample_rate, rng, role="ref", backend=self.fx_backend)
        target_proc = process_branch(target_dry, self.sample_rate, rng, role="target", backend=self.fx_backend)

        ref_db = rms_db(ref_mix)
        tgt_db = rms_db(target_proc)
        desired_delta_db = rng.uniform(-1.5, 4.0)
        align_gain_db = (ref_db - tgt_db) + desired_delta_db
        target_proc = apply_gain_db(target_proc, align_gain_db)
        target_proc = apply_gain_db(target_proc, rng.uniform(-1.0, 1.0))

        mix = ref_mix + target_proc

        mix = np.clip(mix, -1.25, 1.25).astype(np.float32, copy=False)
        target_proc = np.clip(target_proc, -1.25, 1.25).astype(np.float32, copy=False)
        ref_input = np.clip(ref_input, -1.25, 1.25).astype(np.float32, copy=False)

        valid = np.ones((self.segment_samples,), dtype=np.float32)

        return {
            "mix": torch.from_numpy(mix.T.copy()),
            "ref": torch.from_numpy(ref_input.T.copy()),
            "target": torch.from_numpy(target_proc.T.copy()),
            "valid_mask": torch.from_numpy(valid),
            "valid_samples": torch.tensor(self.segment_samples, dtype=torch.long),
        }


def collate_segments(batch: Sequence[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    return {
        "mix": torch.stack([x["mix"] for x in batch], dim=0),
        "ref": torch.stack([x["ref"] for x in batch], dim=0),
        "target": torch.stack([x["target"] for x in batch], dim=0),
        "valid_mask": torch.stack([x["valid_mask"] for x in batch], dim=0),
        "valid_samples": torch.stack([x["valid_samples"] for x in batch], dim=0),
    }


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, *, resolutions: Sequence[Tuple[int, int, int]] | None = None, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)
        self.resolutions = list(resolutions or [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)])
        for n_fft, _, win_length in self.resolutions:
            win = torch.hann_window(win_length, periodic=True)
            self.register_buffer(f"window_{n_fft}_{win_length}", win, persistent=False)

    def _window(self, n_fft: int, win_length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        win = getattr(self, f"window_{n_fft}_{win_length}")
        return win.to(device=device, dtype=dtype)

    def _mag(self, x: Tensor, n_fft: int, hop: int, win_length: int) -> Tensor:
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
    w = valid_mask.unsqueeze(1)
    denom = w.sum().clamp_min(1.0) * estimate.size(1)
    return ((estimate - target).abs() * w).sum() / denom


def apply_valid_mask(x: Tensor, valid_mask: Tensor) -> Tensor:
    return x * valid_mask.unsqueeze(1)


def save_checkpoint(path: Path, *, model: DubSeparator, optimizer: torch.optim.Optimizer, scaler: Optional[GradScaler], epoch: int, global_step: int, args: argparse.Namespace) -> None:
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
    strict_model: bool = True,
    reset_optimizer: bool = False,
    reset_scaler: bool = False,
    reset_global_step: bool = False,
    reset_epoch: bool = False,
) -> Tuple[int, int, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=strict_model)

    resume_info: Dict[str, Any] = {
        "optimizer_loaded": False,
        "scaler_loaded": False,
        "optimizer_reset": False,
        "scaler_reset": False,
        "epoch_reset": False,
        "global_step_reset": False,
        "messages": [],
    }

    if reset_optimizer:
        resume_info["optimizer_reset"] = True
        resume_info["messages"].append("optimizer reset requested by flag")
    else:
        opt_state = ckpt.get("optimizer")
        if opt_state is None:
            resume_info["optimizer_reset"] = True
            resume_info["messages"].append("checkpoint has no optimizer state; using fresh optimizer")
        else:
            try:
                optimizer.load_state_dict(opt_state)
                resume_info["optimizer_loaded"] = True
                resume_info["messages"].append("optimizer state loaded")
            except Exception as exc:
                resume_info["optimizer_reset"] = True
                resume_info["messages"].append(f"optimizer state load failed; using fresh optimizer ({exc})")

    if scaler is not None:
        if reset_scaler:
            resume_info["scaler_reset"] = True
            resume_info["messages"].append("GradScaler reset requested by flag")
        else:
            scaler_state = ckpt.get("scaler")
            if scaler_state is None:
                resume_info["scaler_reset"] = True
                resume_info["messages"].append("checkpoint has no GradScaler state; using fresh scaler")
            else:
                try:
                    scaler.load_state_dict(scaler_state)
                    resume_info["scaler_loaded"] = True
                    resume_info["messages"].append("GradScaler state loaded")
                except Exception as exc:
                    resume_info["scaler_reset"] = True
                    resume_info["messages"].append(f"GradScaler state load failed; using fresh scaler ({exc})")

    epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    if reset_epoch:
        epoch = 0
        resume_info["epoch_reset"] = True
        resume_info["messages"].append("epoch reset to 0")
    if reset_global_step:
        global_step = 0
        resume_info["global_step_reset"] = True
        resume_info["messages"].append("global_step reset to 0")

    return epoch, global_step, resume_info


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


def compute_losses(*, outputs: Dict[str, Tensor], batch: Dict[str, Tensor], mrstft: MultiResolutionSTFTLoss, lambda_l1: float, lambda_sc: float, lambda_logmag: float) -> Tuple[Tensor, Dict[str, float]]:
    estimate = outputs["estimate_waveform"].float()
    target = batch["target"].float()
    valid_mask = batch["valid_mask"].float()

    est_v = apply_valid_mask(estimate, valid_mask)
    tgt_v = apply_valid_mask(target, valid_mask)

    l1 = masked_l1_loss(estimate, target, valid_mask)
    sc, logmag = mrstft(est_v, tgt_v)
    total = lambda_l1 * l1 + lambda_sc * sc + lambda_logmag * logmag

    logs = {
        "loss_total": float(total.detach().item()),
        "l1": float(l1.detach().item()),
        "sc": float(sc.detach().item()),
        "logmag": float(logmag.detach().item()),
    }
    return total, logs


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DubSeparator on runtime-synthesized ref + TTS data")
    p.add_argument("--ref-root", type=Path, required=True, help="Path to folder with long Japanese vocal ref files")
    p.add_argument("--tts-root", type=Path, required=True, help="Path to recursively scanned TTS audio tree (wav/mp3/ogg/...)")
    p.add_argument("--out-dir", type=Path, required=True, help="Directory for checkpoints and logs")
    p.add_argument("--index-json", type=Path, default=None, help="Load ref/TTS index from JSON instead of re-indexing")
    p.add_argument("--save-index-json", type=Path, default=None, help="Where to save ref/TTS index JSON after fresh indexing")

    p.add_argument("--segment-sec", type=float, default=4.0, help="Fixed model input segment length in seconds")
    p.add_argument("--batch", type=int, default=2, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16", "fp16"], help="AMP mode")
    p.add_argument("--tf32", action="store_true", help="Enable TF32 on CUDA")
    p.add_argument("--save-every-epoch", type=int, default=1, help="Save numbered checkpoint every N epochs")
    p.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume from")

    p.add_argument("--loss-l1", type=float, default=1.0, help="Waveform L1 coefficient")
    p.add_argument("--loss-mr-sc", type=float, default=1.0, help="MR spectral convergence coefficient")
    p.add_argument("--loss-mr-logmag", type=float, default=1.0, help="MR log-magnitude coefficient")

    p.add_argument("--epoch-size", type=int, default=1000, help="Number of optimizer steps per epoch")

    p.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Global random seed")
    p.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")

    p.add_argument("--grad-checkpoint", action="store_true", help="Enable activation checkpointing in the optimized model")
    p.add_argument("--reset-optimizer", action="store_true", help="Do not load optimizer state from checkpoint")
    p.add_argument("--reset-scaler", action="store_true", help="Do not load GradScaler state from checkpoint")
    p.add_argument("--reset-global-step", action="store_true", help="Reset global_step to 0 after loading checkpoint")
    p.add_argument("--reset-epoch", action="store_true", help="Reset epoch counter to 0 after loading checkpoint")
    p.add_argument("--non-strict-resume", action="store_true", help="Load model weights with strict=False")

    p.add_argument("--fx-backend", type=str, default="auto", choices=["auto", "pedalboard", "native"], help="DSP/effects backend")
    p.add_argument("--max-shift-ms", type=float, default=250.0, help="Max absolute shift applied to mix-side ref chunk")
    p.add_argument("--tts-cache-size", type=int, default=128, help="Per-worker TTS clip cache size")
    p.add_argument("--min-pause-ms", type=float, default=40.0, help="Minimum silence pause when stitching short TTS clips")
    p.add_argument("--max-pause-ms", type=float, default=180.0, help="Maximum silence pause when stitching short TTS clips")
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
        ref_entries, tts_entries, meta = load_index_json(args.index_json, expected_sr=sample_rate)
        ref_stats = summarize_index(ref_entries)
        tts_stats = summarize_index(tts_entries)
        print(f"loaded index json: {args.index_json}")
    else:
        ref_entries = index_ref_files(args.ref_root)
        tts_entries = index_tts_files(args.tts_root)
        ref_stats = summarize_index(ref_entries)
        tts_stats = summarize_index(tts_entries)

        index_json_path = args.save_index_json if args.save_index_json is not None else (args.out_dir / "index_runtime_synth.json")
        save_index_json(
            index_json_path,
            ref_entries=ref_entries,
            tts_entries=tts_entries,
            sample_rate=sample_rate,
            ref_root=args.ref_root,
            tts_root=args.tts_root,
        )
        print(f"saved index json: {index_json_path}")

    print(f"ref  : {ref_stats['count']} files, {human_hours(ref_stats['hours'])}, min/mean/max = {ref_stats['min_sec']:.2f}/{ref_stats['mean_sec']:.2f}/{ref_stats['max_sec']:.2f} sec")
    print(f"tts  : {tts_stats['count']} files, {human_hours(tts_stats['hours'])}, min/mean/max = {tts_stats['min_sec']:.2f}/{tts_stats['mean_sec']:.2f}/{tts_stats['max_sec']:.2f} sec")
    print(f"fx backend      : {args.fx_backend} (pedalboard_available={HAS_PEDALBOARD})")
    print(f"torchaudio      : {HAS_TORCHAUDIO}")

    index_dump = {
        "ref": ref_stats,
        "tts": tts_stats,
        "sample_rate": sample_rate,
        "segment_sec": args.segment_sec,
        "ref_root": str(args.ref_root),
        "tts_root": str(args.tts_root),
        "index_json": str(args.index_json) if args.index_json is not None else None,
        "fx_backend": args.fx_backend,
    }
    (args.out_dir / "index_summary.json").write_text(json.dumps(index_dump, indent=2, ensure_ascii=False), encoding="utf-8")

    dataset = RuntimeSyntheticDubDataset(
        ref_entries=ref_entries,
        tts_entries=tts_entries,
        sample_rate=sample_rate,
        segment_sec=args.segment_sec,
        epoch_steps=args.epoch_size,
        batch_size=args.batch,
        seed=args.seed,
        fx_backend=args.fx_backend,
        max_shift_ms=args.max_shift_ms,
        tts_cache_size=args.tts_cache_size,
        min_pause_ms=args.min_pause_ms,
        max_pause_ms=args.max_pause_ms,
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
        start_epoch, global_step, resume_info = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            strict_model=not args.non_strict_resume,
            reset_optimizer=bool(args.reset_optimizer),
            reset_scaler=bool(args.reset_scaler),
            reset_global_step=bool(args.reset_global_step),
            reset_epoch=bool(args.reset_epoch),
        )
        print(f"Resumed from {args.resume} at epoch={start_epoch}, global_step={global_step}")
        for msg in resume_info.get("messages", []):
            print(f"[resume] {msg}")

    print("=== training ===")
    print(f"device            : {device}")
    print(f"params trainable  : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"segment_sec       : {args.segment_sec}")
    print(f"batch             : {args.batch}")
    print(f"epoch_size steps  : {args.epoch_size}")
    print(f"amp               : {args.amp}")
    print(f"tf32              : {bool(args.tf32)}")
    print(f"loss coeffs       : l1={args.loss_l1} mr_sc={args.loss_mr_sc} mr_logmag={args.loss_mr_logmag}")
    print(f"grad checkpoint   : {bool(args.grad_checkpoint)}")
    print(f"reset optimizer   : {bool(args.reset_optimizer)}")
    print(f"reset scaler      : {bool(args.reset_scaler)}")
    print(f"reset global_step : {bool(args.reset_global_step)}")
    print(f"reset epoch       : {bool(args.reset_epoch)}")
    print(f"strict resume     : {not bool(args.non_strict_resume)}")

    stop_requested = {"flag": False}

    def _handle_signal(signum, frame):
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

            running: Dict[str, float] = {"loss_total": 0.0, "l1": 0.0, "sc": 0.0, "logmag": 0.0}
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
                    ema = update_ema_dict(ema, step_stats, decay=0.95)

                    progress.set_postfix({
                        "loss": f"{logs['loss_total']:.4f}",
                        "ema": f"{ema.get('loss_total', logs['loss_total']):.4f}",
                        "l1": f"{logs['l1']:.3f}",
                        "sc": f"{logs['sc']:.3f}",
                        "logm": f"{logs['logmag']:.3f}",
                        "gate": f"{extra_stats.get('crm_gate_mean', float('nan')):.3f}",
                        "|M|": f"{extra_stats.get('mask_abs_mean', float('nan')):.3f}",
                        "gn": f"{grad_norm:.2f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    })

                    avg_loss = running["loss_total"] / step_idx
                    print(
                        f"epoch={epoch:05d} step={step_idx:05d}/{args.epoch_size:05d} global_step={global_step:08d} "
                        f"loss={logs['loss_total']:.6f} avg={avg_loss:.6f} "
                        f"grad_norm={grad_norm:.4f} lr={optimizer.param_groups[0]['lr']:.3e} "
                        f"crm_gate={extra_stats.get('crm_gate_mean', float('nan')):.4f} "
                        f"mask_abs={extra_stats.get('mask_abs_mean', float('nan')):.4f}"
                    )

            progress.close()

            elapsed = time.time() - epoch_start
            denom = float(args.epoch_size)
            print(
                f"[epoch {epoch:05d}] "
                f"loss={running['loss_total']/denom:.6f} "
                f"l1={running['l1']/denom:.6f} "
                f"sc={running['sc']/denom:.6f} "
                f"logmag={running['logmag']/denom:.6f} "
                f"ema_loss={ema.get('loss_total', 0.0):.6f} "
                f"ema_gate={ema.get('crm_gate_mean', float('nan')):.6f} "
                f"ema_mask_abs={ema.get('mask_abs_mean', float('nan')):.6f} "
                f"ema_grad_norm={ema.get('grad_norm', float('nan')):.6f} "
                f"time={elapsed:.1f}s"
            )

            last_ckpt = args.out_dir / "last.pt"
            save_checkpoint(last_ckpt, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, global_step=global_step, args=args)
            if args.save_every_epoch > 0 and epoch % args.save_every_epoch == 0:
                numbered = args.out_dir / f"epoch_{epoch:05d}.pt"
                save_checkpoint(numbered, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, global_step=global_step, args=args)

    except StopTraining:
        print("Stopping requested. Saving interrupt checkpoint...")
        save_checkpoint(args.out_dir / "interrupt.pt", model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, global_step=global_step, args=args)
    except KeyboardInterrupt:
        print("KeyboardInterrupt. Saving interrupt checkpoint...")
        save_checkpoint(args.out_dir / "interrupt.pt", model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, global_step=global_step, args=args)
        raise


if __name__ == "__main__":
    main()
