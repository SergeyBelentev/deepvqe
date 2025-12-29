# train.py
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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


def random_crop_same(signals: List[torch.Tensor], length: int) -> List[torch.Tensor]:
    max_len = max(s.shape[-1] for s in signals)
    if max_len < length:
        pad = length - max_len
        signals = [F.pad(s, (0, pad)) for s in signals]
        max_len = length

    start = 0 if max_len == length else torch.randint(0, max_len - length + 1, (1,)).item()

    out = []
    for s in signals:
        if s.shape[-1] < start + length:
            s = F.pad(s, (0, start + length - s.shape[-1]))
        out.append(s[..., start: start + length])
    return out


def apply_ref_timevarying_gain(
    ref_f: torch.Tensor,  # (N,T)
    *,
    sr: int,
    prob: float,
    max_db: float,
    knot_sec: float = 0.5,
    smooth_ms: float = 40.0,
    row_mask: torch.Tensor | None = None,  # bool (N,)
) -> torch.Tensor:
    """
    Плавный time-varying gain только для REF (AGC/mastering drift).

    - узлы каждые knot_sec
    - линейная интерполяция -> Hann-сглаживание (похоже на spline)
    - применяется с вероятностью prob только к строкам row_mask
    - величина в пределах ±max_db

    ref_f: (N,T), где N = B*C (каналы слиты в batch)
    """
    if prob <= 0.0 or max_db <= 0.0:
        return ref_f

    device = ref_f.device
    N, T = ref_f.shape

    if row_mask is None:
        eligible = torch.ones((N,), device=device, dtype=torch.bool)
    else:
        eligible = row_mask.to(device=device, dtype=torch.bool).view(-1)
        if eligible.numel() != N:
            raise RuntimeError(f"row_mask length mismatch: got {eligible.numel()} expected {N}")

    idx = torch.nonzero(eligible, as_tuple=False).flatten()
    if idx.numel() == 0:
        return ref_f

    sel = torch.rand((idx.numel(),), device=device) < float(prob)
    apply_idx = idx[sel]
    if apply_idx.numel() == 0:
        return ref_f

    M = int(apply_idx.numel())
    knot_samp = max(1, int(round(knot_sec * sr)))
    K = (T - 1) // knot_samp + 2  # >=2

    knots_db = (torch.rand((M, K), device=device, dtype=torch.float32) * 2.0 - 1.0) * float(max_db)

    db_curve = F.interpolate(
        knots_db[:, None, :], size=T, mode="linear", align_corners=True
    ).squeeze(1)  # (M,T)

    if smooth_ms > 0.0:
        L = int(round(smooth_ms * 1e-3 * sr))
        if L >= 3:
            if L % 2 == 0:
                L += 1
            w = torch.hann_window(L, device=device, dtype=torch.float32)
            w = w / w.sum().clamp_min(1e-12)
            db_curve = F.conv1d(db_curve[:, None, :], w[None, None, :], padding=L // 2).squeeze(1)

    gain = torch.pow(ref_f.new_tensor(10.0), db_curve / 20.0)  # (M,T)

    ref_f = ref_f.clone()
    ref_f[apply_idx] = ref_f[apply_idx] * gain
    return ref_f


# -----------------------
# STFT helper (fp32 only)
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
        w = self.window.to(device=x.device, dtype=torch.float32)
        X = torch.stft(
            x,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop,
            win_length=self.cfg.win,
            window=w,
            return_complex=True,
        )
        return torch.view_as_real(X).to(torch.float32)

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
# losses
# -----------------------
def corr_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    num = (x * y).sum(dim=-1).abs()
    den = torch.sqrt((x * x).sum(dim=-1) * (y * y).sum(dim=-1) + eps)
    return (num / (den + eps)).mean()


def complex_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).abs().mean()


class MRSTFTLoss(nn.Module):
    """
    MRSTFTLoss with silence-robustness:
      - soft gate by target RMS (dBFS): near-silence => near-zero weight
      - spectral convergence denominator floor
      - optional clipping of SC/log terms
    """

    def __init__(
        self,
        cfgs: List[Tuple[int, int, int]],
        eps: float = 1e-7,
        gate_db_low: float = -55.0,
        gate_db_high: float = -40.0,
        gate_detach: bool = True,
        sc_denom_floor: float = 1.0,
        sc_clip: float | None = 10.0,
        log_clip: float | None = 10.0,
        mag_floor: float = 1e-5,
        gate_floor: float = 0.05,
    ):
        super().__init__()
        self.cfgs = cfgs
        self.eps = float(eps)

        self.gate_db_low = float(gate_db_low)
        self.gate_db_high = float(gate_db_high)
        self.gate_detach = bool(gate_detach)

        self.sc_denom_floor = float(sc_denom_floor)
        self.sc_clip = sc_clip
        self.log_clip = log_clip
        self.mag_floor = float(mag_floor)

        self.gate_floor = float(gate_floor)

        for i, (_, _, win) in enumerate(cfgs):
            self.register_buffer(
                f"window_{i}",
                torch.hann_window(win, dtype=torch.float32),
                persistent=False,
            )

    @staticmethod
    def _rms_db(x: torch.Tensor, eps: float) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1) + eps)  # (B,)
        return 20.0 * torch.log10(rms + eps)

    def _gate(self, y: torch.Tensor) -> torch.Tensor:
        db = self._rms_db(y, self.eps)  # (B,)
        w = (db - self.gate_db_low) / max(1e-6, (self.gate_db_high - self.gate_db_low))
        w = torch.clamp(w, 0.0, 1.0)
        w = self.gate_floor + (1.0 - self.gate_floor) * w
        if self.gate_detach:
            w = w.detach()
        return w  # (B,)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.float()
        y = y.float()

        B = x.shape[0]
        w_gate = self._gate(y)  # (B,)

        total = x.new_zeros(())
        for i, (n_fft, hop, win) in enumerate(self.cfgs):
            w = getattr(self, f"window_{i}").to(device=x.device, dtype=torch.float32)

            X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=w, return_complex=True)
            Y = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, window=w, return_complex=True)

            Xmag = torch.abs(X).clamp_min(self.mag_floor)
            Ymag = torch.abs(Y).clamp_min(self.mag_floor)

            diff = (Ymag - Xmag).reshape(B, -1)
            ref = Ymag.reshape(B, -1)

            diff_norm = torch.linalg.vector_norm(diff, dim=1)  # (B,)
            ref_norm = torch.linalg.vector_norm(ref, dim=1)    # (B,)

            denom = torch.maximum(ref_norm, ref_norm.new_full(ref_norm.shape, self.sc_denom_floor))
            sc = diff_norm / (denom + self.eps)  # (B,)
            if self.sc_clip is not None:
                sc = sc.clamp_max(float(self.sc_clip))

            logX = torch.log(Xmag)
            logY = torch.log(Ymag)
            log_abs = torch.abs(logX - logY)
            log_l1 = log_abs.mean(dim=(1, 2))  # (B,)
            if self.log_clip is not None:
                log_l1 = log_l1.clamp_max(float(self.log_clip))

            per = (sc + log_l1) * w_gate
            total = total + per.mean()

        return total / max(1, len(self.cfgs))


# -----------------------
# dataset
# -----------------------
def _sf_num_frames_and_sr(path: str) -> tuple[int, int]:
    info = sf.info(path)
    return int(info.frames), int(info.samplerate)


def _load_wav_stereo_segment(path: str, sr_expected: int, start: int, length: int) -> torch.Tensor:
    """
    Read exactly [start : start+length) from file (stereo), pad zeros if out of bounds.
    Returns: (2, length) float32
    """
    if length <= 0:
        return torch.zeros((2, 0), dtype=torch.float32)

    with sf.SoundFile(path, "r") as f:
        if f.samplerate != sr_expected:
            raise RuntimeError(f"SR mismatch for {path}: got {f.samplerate}, expected {sr_expected}")

        frames = int(f.frames)
        # if start beyond EOF -> silence
        if start >= frames:
            x = np.zeros((0, f.channels), dtype=np.float32)
        else:
            start_clamped = max(0, start)
            f.seek(start_clamped)
            x = f.read(frames=length, dtype="float32", always_2d=True)

    # ensure stereo
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    else:
        x = x[:, :2]

    # pad to exact length
    if x.shape[0] < length:
        pad = length - x.shape[0]
        x = np.vstack([x, np.zeros((pad, 2), dtype=np.float32)])

    return torch.from_numpy(x).transpose(0, 1).contiguous()  # (2,T)


class AecDataset(Dataset):
    """
    CSV: mix_path, ref_path, target_path (target_path can be 'None' -> zero target)

    - short example (min_len < long_threshold_sec): behaves like old code (random_crop_same over full signals)
    - long example: creates multiple virtual items (sliding windows), so long files contribute more segments.
    """

    def __init__(
        self,
        manifest_path: str,
        sr: int,
        segment_sec: float,
        *,
        long_threshold_sec: float = 6.0,
        long_hop_sec: float | None = None,
        long_jitter_sec: float = 0.0,
    ):
        self.manifest_path = manifest_path
        self.sr = sr
        self.seg_len = int(round(sr * segment_sec))

        self.long_threshold_len = int(round(sr * long_threshold_sec))
        self.long_hop_len = int(round(sr * (long_hop_sec if long_hop_sec is not None else segment_sec)))
        self.long_hop_len = max(1, self.long_hop_len)

        self.long_jitter_len = int(round(sr * long_jitter_sec))
        self.long_jitter_len = max(0, self.long_jitter_len)

        self.items: List[tuple[str, str, str]] = []
        self.index: List[tuple[int, int]] = []  # (base_item_idx, start_sample) ; start_sample=-1 means "short mode"
        self._load_manifest_data()
        self._build_index()

    def _load_manifest_data(self) -> None:
        with open(self.manifest_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",", skipinitialspace=False)
            for row in reader:
                if not row:
                    continue
                if row[0].strip().startswith("#"):
                    continue
                if len(row) < 3:
                    raise RuntimeError(f"Bad row (need 3 cols): {row}")
                self.items.append((row[0].strip(), row[1].strip(), row[2].strip()))

    def _min_aligned_len(self, mix_p: str, ref_p: str, tgt_p: str) -> int:
        """
        For long-window indexing we prefer the region where all signals exist (avoid padding).
        """
        mix_n, mix_sr = _sf_num_frames_and_sr(mix_p)
        ref_n, ref_sr = _sf_num_frames_and_sr(ref_p)
        if mix_sr != self.sr or ref_sr != self.sr:
            raise RuntimeError(f"SR mismatch in manifest: {mix_p} sr={mix_sr}, {ref_p} sr={ref_sr}, expected {self.sr}")

        if tgt_p != "None":
            tgt_n, tgt_sr = _sf_num_frames_and_sr(tgt_p)
            if tgt_sr != self.sr:
                raise RuntimeError(f"SR mismatch in manifest: {tgt_p} sr={tgt_sr}, expected {self.sr}")
            return min(mix_n, ref_n, tgt_n)

        return min(mix_n, ref_n)

    def _build_index(self) -> None:
        self.index.clear()

        for i, (mix_p, ref_p, tgt_p) in enumerate(self.items):
            min_len = self._min_aligned_len(mix_p, ref_p, tgt_p)

            # short mode: behave like old logic
            if min_len < self.long_threshold_len:
                self.index.append((i, -1))
                continue

            # long mode: build sliding windows over the whole aligned region
            if min_len <= self.seg_len:
                # still just one window
                self.index.append((i, 0))
                continue

            max_start = min_len - self.seg_len
            starts = list(range(0, max_start + 1, self.long_hop_len))

            # ensure tail coverage (may create overlap)
            if starts[-1] != max_start:
                starts.append(max_start)

            for s in starts:
                self.index.append((i, int(s)))

        if not self.index:
            raise RuntimeError("Dataset index is empty. Check manifest or files.")

        print(
            f"[AecDataset] base_items={len(self.items)} virtual_items={len(self.index)} | "
            f"seg_len={self.seg_len} long_thr={self.long_threshold_len} hop={self.long_hop_len} jitter={self.long_jitter_len}"
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        base_i, start = self.index[idx]
        mix_p, ref_p, tgt_p = self.items[base_i]
        has_speech = (tgt_p != "None")

        # short mode: keep previous behavior (loads full audio)
        if start < 0:
            mix = load_wav_stereo(mix_p, self.sr)
            ref = load_wav_stereo(ref_p, self.sr)
            if not has_speech:
                max_len = max(mix.shape[-1], ref.shape[-1])
                tgt = mix.new_zeros((2, max_len))
            else:
                tgt = load_wav_stereo(tgt_p, self.sr)

            mix, ref, tgt = random_crop_same([mix, ref, tgt], length=self.seg_len)
            return mix, ref, tgt, has_speech

        # long mode: windowed reading (reads only segment)
        # optional jitter for diversity
        if self.long_jitter_len > 0:
            # symmetric jitter
            j = int(torch.randint(-self.long_jitter_len, self.long_jitter_len + 1, (1,)).item())
        else:
            j = 0

        # clamp to valid range based on aligned length
        min_len = self._min_aligned_len(mix_p, ref_p, tgt_p)
        max_start = max(0, min_len - self.seg_len)
        start_j = int(max(0, min(max_start, start + j)))

        mix = _load_wav_stereo_segment(mix_p, self.sr, start_j, self.seg_len)
        ref = _load_wav_stereo_segment(ref_p, self.sr, start_j, self.seg_len)

        if not has_speech:
            tgt = mix.new_zeros((2, self.seg_len))
        else:
            tgt = _load_wav_stereo_segment(tgt_p, self.sr, start_j, self.seg_len)

        return mix, ref, tgt, has_speech



def collate(batch):
    mix = torch.stack([b[0] for b in batch], dim=0)  # (B,2,T)
    ref = torch.stack([b[1] for b in batch], dim=0)
    tgt = torch.stack([b[2] for b in batch], dim=0)
    has_speech = torch.tensor([bool(b[3]) for b in batch], dtype=torch.bool)  # (B,)
    return mix, ref, tgt, has_speech


# -----------------------
# resume helpers
# -----------------------
def _move_optimizer_state_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)


def _get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "torch": torch.random.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    if "torch" in state and state["torch"] is not None:
        torch.random.set_rng_state(state["torch"])
    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "cuda_all" in state and state["cuda_all"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_all"])


def _assert_resume_compat(args_now: argparse.Namespace, ckpt_args: Dict[str, Any]) -> None:
    """
    Жёсткие параметры, которые ДОЛЖНЫ совпасть, иначе веса/формы разъедутся.
    """
    keys = [
        "sr",
        "n_fft",
        "hop",
        "win",
        "delay_past_frames",
        "delay_future_frames",
        "align_hidden",
    ]
    mism = []
    for k in keys:
        if k in ckpt_args:
            v_old = ckpt_args[k]
            v_new = getattr(args_now, k)
            if v_old != v_new:
                mism.append((k, v_old, v_new))
    if mism:
        lines = ["Resume arg mismatch (these MUST match to load weights safely):"]
        for k, old, new in mism:
            lines.append(f"  - {k}: ckpt={old} vs now={new}")
        raise RuntimeError("\n".join(lines))


def _setup_tf32(enable: bool, verbose: bool = True) -> None:
    if not torch.cuda.is_available():
        if verbose:
            print("TF32: cuda not available -> ignored")
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(enable)
    torch.backends.cudnn.allow_tf32 = bool(enable)
    if verbose:
        print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"TF32 cudnn : {torch.backends.cudnn.allow_tf32}")


def _resolve_delays(args: argparse.Namespace) -> None:
    """
    Поддержка:
      - --delay-frames (deprecated): выставляет симметрично past=future
      - --delay-past-frames / --delay-future-frames: явная настройка
    Внутри args будут гарантированно int past/future.
    """
    past = args.delay_past_frames
    fut = args.delay_future_frames
    sym = args.delay_frames

    if past is None and fut is None:
        if sym is not None:
            past = int(sym)
            fut = int(sym)
        else:
            past = 25
            fut = 25
    else:
        if past is None:
            past = 25
        if fut is None:
            fut = 25

    if past <= 0 or fut <= 0:
        raise RuntimeError(f"delay frames must be > 0: past={past} future={fut}")

    args.delay_past_frames = int(past)
    args.delay_future_frames = int(fut)


# -----------------------
# train
# -----------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", required=True, help="CSV: mix_path,ref_path,target_path (target_path can be 'None')")
    ap.add_argument("--save-dir", default="ckpt_48k")
    ap.add_argument("--epochs", type=int, default=50, help="TOTAL epochs to train to (if resuming: must be > ckpt epoch)")
    ap.add_argument("--save-every-epochs", type=int, default=1)

    # Dataset args
    ap.add_argument("--long-threshold-sec", type=float, default=6.0,
                    help="If min(mix/ref/tgt) longer than this -> use sliding windows over the whole file")
    ap.add_argument("--long-hop-sec", type=float, default=None,
                    help="Hop between windows for long examples. Default: segment-sec (no overlap). Use smaller for overlap.")
    ap.add_argument("--long-jitter-sec", type=float, default=0.0,
                    help="Random jitter added to each long-window start (seconds). Clamped to valid range.")

    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--segment-sec", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)

    # audio / stft
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--n-fft", type=int, default=1536)
    ap.add_argument("--hop", type=int, default=480)
    ap.add_argument("--win", type=int, default=1536)

    # augment: random loudness/gain
    ap.add_argument(
        "--random-gain-db",
        type=float,
        default=0.0,
        help=(
            "Uniform random gain in dB applied equally to mix/ref/target per (B*C) row. "
            "Range: [-random_gain_db, +random_gain_db]."
        ),
    )

    # augment: time-varying gain drift for REF only (separately for no-speech vs speech)
    ap.add_argument("--ref-tv-gain-knot-sec", type=float, default=0.5, help="Knot spacing in seconds (REF drift)")
    ap.add_argument("--ref-tv-gain-smooth-ms", type=float, default=40.0, help="Smoothing window in ms (REF drift)")

    ap.add_argument("--ref-tv-gain-prob-nospeech", type=float, default=0.2, help="Prob apply REF drift on no-speech segments")
    ap.add_argument("--ref-tv-gain-db-nospeech", type=float, default=4.0, help="±dB range for REF drift on no-speech")

    ap.add_argument("--ref-tv-gain-prob-speech", type=float, default=0.05, help="Prob apply REF drift on speech segments")
    ap.add_argument("--ref-tv-gain-db-speech", type=float, default=2.0, help="±dB range for REF drift on speech")

    # model alignment (bidirectional)
    ap.add_argument("--delay-past-frames", type=int, default=None, help="Align search: how many frames into the past")
    ap.add_argument("--delay-future-frames", type=int, default=None, help="Align search: how many frames into the future")
    ap.add_argument("--delay-frames", type=int, default=None, help="(deprecated) sets past=future=delay_frames")
    ap.add_argument("--align-hidden", type=int, default=64)

    # augment: random ref shift, but quantized to hop (frame-aligned)
    ap.add_argument("--ref-shift-ms", type=float, default=0.0, help="±ms, quantized to hop")

    # loss weights
    ap.add_argument("--w-out-l1", type=float, default=1.0)
    ap.add_argument("--w-bg-l1", type=float, default=1.0)
    ap.add_argument("--w-bg-stft", type=float, default=0.2)
    ap.add_argument("--w-mrstft", type=float, default=0.2)
    ap.add_argument("--w-leak", type=float, default=0.5)

    # mrstft loss settings
    ap.add_argument("--mrstft-gate-db-low", type=float, default=-55.0)
    ap.add_argument("--mrstft-gate-db-high", type=float, default=-40.0)
    ap.add_argument("--mrstft-gate-floor", type=float, default=0.05)

    # amp
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--amp-cache", action="store_true", help="Enable autocast cache (default: off)")

    # tf32
    ap.add_argument("--tf32", action="store_true", help="Enable TF32 for matmul/conv (fp32 speedup on Ampere/Ada)")
    ap.add_argument("--cudnn-benchmark", action="store_true", help="Enable cudnn benchmark (speed, less deterministic)")

    # resume
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt")
    ap.add_argument("--resume-nonstrict", action="store_true", help="Allow non-strict model load (not recommended)")
    ap.add_argument("--reset-opt", action="store_true", help="Do NOT load optimizer state")
    ap.add_argument("--reset-scaler", action="store_true", help="Do NOT load GradScaler state")

    args = ap.parse_args()
    _resolve_delays(args)

    # --- seeds ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # TF32 / cudnn benchmark
    if device.type == "cuda":
        _setup_tf32(args.tf32, verbose=True)
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
        if args.cudnn_benchmark:
            print("cudnn.benchmark=True")

    # --- data ---
    ds = AecDataset(
        args.manifest,
        sr=args.sr,
        segment_sec=args.segment_sec,
        long_threshold_sec=args.long_threshold_sec,
        long_hop_sec=args.long_hop_sec,
        long_jitter_sec=args.long_jitter_sec,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        drop_last=(len(ds) >= args.batch),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate,
    )
    print(f"dataset: {len(ds)} items | batch={args.batch} | batches/epoch={len(dl)}")
    if len(dl) == 0:
        raise RuntimeError("DataLoader has 0 batches. Use --batch 1 or add more data.")

    # --- model ---
    model = DeepVQE(
        n_fft=args.n_fft,
        delay_past_frames=args.delay_past_frames,
        delay_future_frames=args.delay_future_frames,
        align_hidden=args.align_hidden,
    ).to(device)
    model.train()
    if hasattr(model, "set_return_bg"):
        model.set_return_bg(True)

    print(
        f"model: n_fft={args.n_fft} | delay_past_frames={args.delay_past_frames} "
        f"| delay_future_frames={args.delay_future_frames} | align_hidden={args.align_hidden}"
    )

    # --- stft / loss ---
    stft = STFT(StftCfg(n_fft=args.n_fft, hop=args.hop, win=args.win)).to(device)

    mrstft = MRSTFTLoss(
        cfgs=[
            (1024, 240, 1024),
            (2048, 480, 2048),
            (4096, 960, 4096),
        ],
        gate_db_low=args.mrstft_gate_db_low,
        gate_db_high=args.mrstft_gate_db_high,
        gate_floor=args.mrstft_gate_floor,
    ).to(device)

    # --- opt / amp ---
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    autocast_cache = bool(args.amp_cache)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ref shift quantization to hop + clamp by align window
    max_shift_frames = int(round((args.ref_shift_ms * 1e-3 * args.sr) / args.hop))
    max_shift_frames = max(0, max_shift_frames)
    max_allowed = max(0, min(args.delay_past_frames, args.delay_future_frames) - 1)
    max_shift_frames = min(max_shift_frames, max_allowed)

    # --- resume state ---
    start_epoch = 1
    micro_step = 0

    if args.resume is not None:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {}) or {}

        _assert_resume_compat(args, ckpt_args)

        strict = not args.resume_nonstrict
        model.load_state_dict(ckpt["model"], strict=strict)
        print(f"Resumed model from {ckpt_path} (strict={strict})")

        ckpt_epoch = int(ckpt.get("epoch", 0) or 0)
        if args.epochs <= ckpt_epoch:
            raise RuntimeError(
                f"--epochs must be > ckpt epoch when resuming. got --epochs {args.epochs}, ckpt epoch {ckpt_epoch}"
            )
        start_epoch = ckpt_epoch + 1

        if (not args.reset_opt) and ("opt" in ckpt) and (ckpt["opt"] is not None):
            opt.load_state_dict(ckpt["opt"])
            _move_optimizer_state_to_device(opt, device)
            print("Resumed optimizer state")

        if (not args.reset_scaler) and (ckpt.get("scaler") is not None) and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
            print("Resumed GradScaler state")

        micro_step = int(ckpt.get("micro_step", 0) or 0)

        rng_state = ckpt.get("rng_state", None)
        if isinstance(rng_state, dict):
            _set_rng_state(rng_state)
            print("Resumed RNG state")

        print(f"Resume: ckpt_epoch={ckpt_epoch} -> start_epoch={start_epoch} | micro_step={micro_step}")

    # -----------------------
    # train loop
    # -----------------------
    for epoch in range(start_epoch, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        run_loss = 0.0

        pbar = tqdm(dl, desc=f"Epoch {epoch}", dynamic_ncols=True)
        for mix, ref, tgt, has_speech in pbar:
            mix = mix.to(device, non_blocking=True)  # (B,2,T)
            ref = ref.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            B, C, T = mix.shape
            mix_f = mix.reshape(B * C, T).float()
            ref_f = ref.reshape(B * C, T).float()
            tgt_f = tgt.reshape(B * C, T).float()

            # augment: random gain (same for mix/ref/tgt row-wise)
            if args.random_gain_db and args.random_gain_db > 0:
                db = (torch.rand((mix_f.shape[0], 1), device=device) * 2.0 - 1.0) * float(args.random_gain_db)
                gain = torch.pow(mix_f.new_tensor(10.0), db / 20.0)
                mix_f = mix_f * gain
                ref_f = ref_f * gain
                tgt_f = tgt_f * gain

            # per-row speech/nospeech masks (B*C,)
            speech_rows = has_speech.to(device=device).repeat_interleave(C)
            nospeech_rows = ~speech_rows

            # REF-only time-varying drift
            if args.ref_tv_gain_prob_nospeech > 0 and args.ref_tv_gain_db_nospeech > 0:
                ref_f = apply_ref_timevarying_gain(
                    ref_f,
                    sr=args.sr,
                    prob=float(args.ref_tv_gain_prob_nospeech),
                    max_db=float(args.ref_tv_gain_db_nospeech),
                    knot_sec=float(args.ref_tv_gain_knot_sec),
                    smooth_ms=float(args.ref_tv_gain_smooth_ms),
                    row_mask=nospeech_rows,
                )

            if args.ref_tv_gain_prob_speech > 0 and args.ref_tv_gain_db_speech > 0:
                ref_f = apply_ref_timevarying_gain(
                    ref_f,
                    sr=args.sr,
                    prob=float(args.ref_tv_gain_prob_speech),
                    max_db=float(args.ref_tv_gain_db_speech),
                    knot_sec=float(args.ref_tv_gain_knot_sec),
                    smooth_ms=float(args.ref_tv_gain_smooth_ms),
                    row_mask=speech_rows,
                )

            # augment: shift REF (frame-aligned)
            if max_shift_frames > 0:
                sh = int(torch.randint(-max_shift_frames, max_shift_frames + 1, (1,), device=device).item())
                shift_samp = sh * args.hop
                if shift_samp > 0:
                    ref_f = torch.cat(
                        [ref_f.new_zeros((ref_f.shape[0], shift_samp)), ref_f[:, : T - shift_samp]],
                        dim=1,
                    )
                elif shift_samp < 0:
                    s = -shift_samp
                    ref_f = torch.cat([ref_f[:, s:], ref_f.new_zeros((ref_f.shape[0], s))], dim=1)

            # STFT fp32
            with torch.amp.autocast(device_type="cuda", enabled=False):
                mix_ri = stft.stft_ri(mix_f)  # (B*2,F,Tf,2)
                ref_ri = stft.stft_ri(ref_f)
                tgt_ri = stft.stft_ri(tgt_f)
                bg_true_ri = mix_ri - tgt_ri
                bg_true_wav = mix_f - tgt_f

            # forward
            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp,
                dtype=amp_dtype,
                cache_enabled=autocast_cache,
            ):
                out = model(mix_ri, ref_ri)

            if not (isinstance(out, (tuple, list)) and len(out) == 2):
                raise RuntimeError("Model must return (out_ri, bg_ri). Call model.set_return_bg(True).")
            out_ri, bg_ri = out

            # losses fp32
            with torch.amp.autocast(device_type="cuda", enabled=False):
                out_wav = stft.istft_ri(out_ri, length=T)
                bg_wav = stft.istft_ri(bg_ri, length=T)

                loss_out = F.l1_loss(out_wav, tgt_f)
                loss_bg = F.l1_loss(bg_wav, bg_true_wav)
                loss_bg_stft = complex_l1(bg_ri, bg_true_ri)
                loss_mr = mrstft(out_wav, tgt_f) if args.w_mrstft > 0 else out_wav.new_tensor(0.0)
                loss_leak = corr_loss(out_wav, bg_true_wav)

                loss = (
                    args.w_out_l1 * loss_out
                    + args.w_bg_l1 * loss_bg
                    + args.w_bg_stft * loss_bg_stft
                    + args.w_mrstft * loss_mr
                    + args.w_leak * loss_leak
                ) / max(1, args.grad_accum)

            # backward
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_step += 1

            if micro_step % max(1, args.grad_accum) == 0:
                if args.grad_clip > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            run_loss += float(loss.detach().cpu()) * max(1, args.grad_accum)

            pbar.set_postfix(
                total=f"{(run_loss / max(1, (pbar.n + 1))):.6f}",
                lo=f"{float(loss_out.detach().cpu()):.4f}",
                lb=f"{float(loss_bg.detach().cpu()):.4f}",
                lmr=f"{float(loss_mr.detach().cpu()):.4f}" if args.w_mrstft > 0 else "0.0000",
                ll=f"{float(loss_leak.detach().cpu()):.4f}",
            )

        avg = run_loss / max(1, len(dl))
        print(f"epoch {epoch:03d} | total {avg:.6f}")

        # сохраняем args без deprecated delay_frames
        args_to_save = vars(args).copy()
        args_to_save.pop("delay_frames", None)

        ckpt_out = {
            "epoch": epoch,
            "micro_step": micro_step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": (scaler.state_dict() if scaler.is_enabled() else None),
            "args": args_to_save,
            "rng_state": _get_rng_state(),
        }

        if epoch % args.save_every_epochs == 0:
            torch.save(ckpt_out, str(Path(args.save_dir) / f"deepvqe_aec48k_e{epoch:03d}.pt"))
        torch.save(ckpt_out, str(Path(args.save_dir) / "deepvqe_latest.pt"))

    print("done.")


if __name__ == "__main__":
    main()
