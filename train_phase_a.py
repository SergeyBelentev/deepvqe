# train_phase_a_v2.py
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from deepvqe import DeepVQEStemSeparator


# -----------------------
# I/O helpers
# -----------------------
def _num_frames_and_sr(path: str) -> tuple[int, int, int]:
    info = sf.info(path)
    return int(info.frames), int(info.samplerate), int(info.channels)


def _read_stereo_segment(path: str, sr_expected: int, start: int, length: int) -> torch.Tensor:
    """
    Returns (2, length) float32, pads with zeros if needed.
    """
    if length <= 0:
        return torch.zeros((2, 0), dtype=torch.float32)

    with sf.SoundFile(path, "r") as f:
        if int(f.samplerate) != int(sr_expected):
            raise RuntimeError(f"SR mismatch for {path}: got {f.samplerate}, expected {sr_expected}")

        frames = int(f.frames)
        ch = int(f.channels)

        if start >= frames:
            x = np.zeros((0, ch), dtype=np.float32)
        else:
            start_clamped = max(0, start)
            f.seek(start_clamped)
            x = f.read(frames=length, dtype="float32", always_2d=True)

    # ensure 2ch
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    else:
        x = x[:, :2]

    if x.shape[0] < length:
        pad = length - x.shape[0]
        x = np.vstack([x, np.zeros((pad, 2), dtype=np.float32)])

    return torch.from_numpy(x).transpose(0, 1).contiguous()  # (2,T)


def _peak(x: torch.Tensor) -> float:
    return float(x.abs().max().item()) if x.numel() else 0.0


def _clip_safe_scale_all(
    signals: List[torch.Tensor],
    peak_ref: torch.Tensor,
    peak_target: float = 0.98,
    eps: float = 1e-12,
) -> Tuple[List[torch.Tensor], float]:
    """
    Apply one global scale to all signals so that peak(peak_ref) <= 1.0 (or <= peak_target when >1).
    We keep the same behavior as before: scale only if peak > 1.0 (avoid unnecessary renorm).
    """
    p = _peak(peak_ref)
    if p <= 1.0:
        return signals, 1.0
    scale = float(peak_target / max(p, eps))
    return [s * scale for s in signals], scale


# -----------------------
# STFT helper (fp32)
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
# Dataset
# -----------------------
@dataclass
class TrackItem:
    kind: str  # "vocal" or "novocal"
    full: str
    bass: str
    drums: str
    instruments: str
    vocals: Optional[str] = None
    melody: Optional[str] = None


def _is_file(p: Path) -> bool:
    return p.exists() and p.is_file()


def scan_root_to_items(root: str) -> List[TrackItem]:
    root_p = Path(root)
    if not root_p.exists():
        raise FileNotFoundError(f"--root not found: {root_p}")

    items: List[TrackItem] = []
    for d in sorted(root_p.iterdir()):
        if not d.is_dir():
            continue

        full = d / "full.wav"
        bass = d / "bass.wav"
        drums = d / "drums.wav"
        inst = d / "instruments.wav"
        vocals = d / "vocals.wav"
        melody = d / "melody.wav"

        if not (_is_file(full) and _is_file(bass) and _is_file(drums) and _is_file(inst)):
            continue

        if _is_file(vocals) and not _is_file(melody):
            items.append(
                TrackItem(
                    kind="vocal",
                    full=str(full),
                    bass=str(bass),
                    drums=str(drums),
                    instruments=str(inst),
                    vocals=str(vocals),
                    melody=None,
                )
            )
        elif _is_file(melody) and not _is_file(vocals):
            items.append(
                TrackItem(
                    kind="novocal",
                    full=str(full),
                    bass=str(bass),
                    drums=str(drums),
                    instruments=str(inst),
                    vocals=None,
                    melody=str(melody),
                )
            )

    if not items:
        raise RuntimeError("scan_root_to_items found 0 valid tracks. Check folder layout / filenames.")
    print(f"[scan] tracks={len(items)}")
    return items


def load_manifest_csv(path: str) -> List[TrackItem]:
    items: List[TrackItem] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        need = {"kind", "full", "bass", "drums", "instruments", "vocals", "melody"}
        if set(r.fieldnames or []) != need:
            raise RuntimeError(f"Bad manifest header. Need exactly: {sorted(need)}; got: {r.fieldnames}")
        for row in r:
            kind = row["kind"].strip()
            vocals = row["vocals"].strip() or None
            melody = row["melody"].strip() or None
            items.append(
                TrackItem(
                    kind=kind,
                    full=row["full"].strip(),
                    bass=row["bass"].strip(),
                    drums=row["drums"].strip(),
                    instruments=row["instruments"].strip(),
                    vocals=vocals,
                    melody=melody,
                )
            )
    if not items:
        raise RuntimeError("Manifest is empty")
    return items


class StemPhaseADataset(Dataset):
    """
    Returns:
      mix, bass, drums, inst, melody, vocals, is_vocal
    where mix is either:
      - stem_sum (default)
      - full.wav (if mix_mode=full)

    Note:
      For mix_mode=full we still read stems and can later form fx_target = full - sum(stems).
    """

    def __init__(
        self,
        items: List[TrackItem],
        *,
        sr: int,
        segment_sec: float,
        long_threshold_sec: float = 6.0,
        long_hop_sec: Optional[float] = None,
        long_jitter_sec: float = 0.0,
        mix_mode: str = "stem_sum",
    ):
        self.items = items
        self.sr = int(sr)
        self.seg_len = int(round(self.sr * float(segment_sec)))

        self.long_threshold_len = int(round(self.sr * float(long_threshold_sec)))
        self.long_hop_len = int(round(self.sr * float(long_hop_sec if long_hop_sec is not None else segment_sec)))
        self.long_hop_len = max(1, self.long_hop_len)
        self.long_jitter_len = int(round(self.sr * float(long_jitter_sec)))
        self.long_jitter_len = max(0, self.long_jitter_len)

        if mix_mode not in ("stem_sum", "full"):
            raise ValueError("mix_mode must be one of: stem_sum, full")
        self.mix_mode = mix_mode

        self.index: List[Tuple[int, int]] = []
        self._build_index()

    def _min_len(self, it: TrackItem) -> int:
        paths = [it.bass, it.drums, it.instruments]
        if self.mix_mode == "full":
            paths.append(it.full)

        if it.kind == "vocal":
            if not it.vocals:
                raise RuntimeError("vocal item missing vocals")
            paths.append(it.vocals)
        else:
            if not it.melody:
                raise RuntimeError("novocal item missing melody")
            paths.append(it.melody)

        mins = None
        for p in paths:
            n, sr, _ch = _num_frames_and_sr(p)
            if sr != self.sr:
                raise RuntimeError(f"SR mismatch: {p} sr={sr}, expected {self.sr}")
            mins = n if mins is None else min(mins, n)
        return int(mins or 0)

    def _build_index(self) -> None:
        self.index.clear()
        for i, it in enumerate(self.items):
            n = self._min_len(it)
            if n <= 0:
                continue
            if n < self.long_threshold_len:
                self.index.append((i, -1))
                continue
            if n <= self.seg_len:
                self.index.append((i, 0))
                continue

            max_start = n - self.seg_len
            starts = list(range(0, max_start + 1, self.long_hop_len))
            if starts[-1] != max_start:
                starts.append(max_start)
            for s in starts:
                self.index.append((i, int(s)))

        if not self.index:
            raise RuntimeError("Dataset index is empty. Check audio files / sr / lengths.")

        print(
            f"[StemPhaseADataset] tracks={len(self.items)} virtual_items={len(self.index)} "
            f"| seg_len={self.seg_len} long_thr={self.long_threshold_len} hop={self.long_hop_len} jitter={self.long_jitter_len} "
            f"| mix_mode={self.mix_mode}"
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        item_i, start = self.index[idx]
        it = self.items[item_i]
        is_vocal = (it.kind == "vocal")

        if start < 0:
            n = self._min_len(it)
            if n <= self.seg_len:
                start_j = 0
            else:
                start_j = int(torch.randint(0, n - self.seg_len + 1, (1,)).item())
        else:
            if self.long_jitter_len > 0:
                j = int(torch.randint(-self.long_jitter_len, self.long_jitter_len + 1, (1,)).item())
            else:
                j = 0
            n = self._min_len(it)
            max_start = max(0, n - self.seg_len)
            start_j = int(max(0, min(max_start, start + j)))

        bass = _read_stereo_segment(it.bass, self.sr, start_j, self.seg_len)
        drums = _read_stereo_segment(it.drums, self.sr, start_j, self.seg_len)
        inst = _read_stereo_segment(it.instruments, self.sr, start_j, self.seg_len)

        if is_vocal:
            vocals = _read_stereo_segment(it.vocals, self.sr, start_j, self.seg_len)  # type: ignore[arg-type]
            melody = torch.zeros_like(vocals)
        else:
            melody = _read_stereo_segment(it.melody, self.sr, start_j, self.seg_len)  # type: ignore[arg-type]
            vocals = torch.zeros_like(melody)

        stem_sum = bass + drums + inst + melody + vocals

        if self.mix_mode == "full":
            full = _read_stereo_segment(it.full, self.sr, start_j, self.seg_len)
            mix = full
            peak_ref = full
            signals, _ = _clip_safe_scale_all([mix, bass, drums, inst, melody, vocals], peak_ref=peak_ref)
            mix, bass, drums, inst, melody, vocals = signals
        else:
            mix = stem_sum
            peak_ref = stem_sum
            signals, _ = _clip_safe_scale_all([mix, bass, drums, inst, melody, vocals], peak_ref=peak_ref)
            mix, bass, drums, inst, melody, vocals = signals

        return mix, bass, drums, inst, melody, vocals, bool(is_vocal)


def collate(batch):
    mix = torch.stack([b[0] for b in batch], dim=0)      # (B,2,T)
    bass = torch.stack([b[1] for b in batch], dim=0)
    drums = torch.stack([b[2] for b in batch], dim=0)
    inst = torch.stack([b[3] for b in batch], dim=0)
    melody = torch.stack([b[4] for b in batch], dim=0)
    vocals = torch.stack([b[5] for b in batch], dim=0)
    is_vocal = torch.tensor([b[6] for b in batch], dtype=torch.bool)  # (B,)
    return mix, bass, drums, inst, melody, vocals, is_vocal


# -----------------------
# Loss
# -----------------------
def l1_ri(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).abs().mean()


# -----------------------
# Training
# -----------------------
def main():
    ap = argparse.ArgumentParser()

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--root", type=str)
    g.add_argument("--manifest", type=str)

    ap.add_argument("--save-dir", default="ckpt_phase_a")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--save-every-epochs", type=int, default=1)

    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--segment-sec", type=float, default=4.0)
    ap.add_argument("--long-threshold-sec", type=float, default=6.0)
    ap.add_argument("--long-hop-sec", type=float, default=None)
    ap.add_argument("--long-jitter-sec", type=float, default=0.0)

    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--n-fft", type=int, default=1536)
    ap.add_argument("--hop", type=int, default=480)
    ap.add_argument("--win", type=int, default=1536)
    ap.add_argument("--num-heads", type=int, default=6)  # bass,drums,inst,melody,vocals,fx

    # Phase A modes
    ap.add_argument("--mix-mode", choices=["stem_sum", "full"], default="stem_sum")
    ap.add_argument("--fx-mode", choices=["zero", "residual"], default="zero")

    # loss weights
    ap.add_argument("--w-stem", type=float, default=1.0)
    ap.add_argument("--w-mix", type=float, default=0.5)
    ap.add_argument("--w-fx", type=float, default=1.0, help="Weight for fx head (present always: zero or residual).")
    ap.add_argument("--w-missing", type=float, default=1.0, help="Weight for missing heads (MUST be ~1.0 for sanity).")

    ap.add_argument("--limit-items", type=int, default=0)
    ap.add_argument("--dump-audio-every-epochs", type=int, default=0)
    ap.add_argument("--dump-dir", type=str, default="dumps_phase_a")

    args = ap.parse_args()

    # compatibility
    if args.mix_mode == "stem_sum" and args.fx_mode == "residual":
        raise RuntimeError("fx-mode=residual makes sense only with mix-mode=full.")
    if args.mix_mode == "full" and args.fx_mode == "zero":
        # allowed, but then full mastering mismatch has nowhere to go -> harder training
        print("[warn] mix-mode=full with fx-mode=zero: you are forcing fx=0 even if full != stem_sum.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print("device:", device)

    if args.root:
        items = scan_root_to_items(args.root)
    else:
        items = load_manifest_csv(args.manifest)

    if args.limit_items and args.limit_items > 0:
        items = items[: int(args.limit_items)]
        print(f"[info] limit-items={len(items)}")

    ds = StemPhaseADataset(
        items,
        sr=args.sr,
        segment_sec=args.segment_sec,
        long_threshold_sec=args.long_threshold_sec,
        long_hop_sec=args.long_hop_sec,
        long_jitter_sec=args.long_jitter_sec,
        mix_mode=args.mix_mode,
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
    print(f"dataset: {len(ds)} segments | batch={args.batch} | batches/epoch={len(dl)}")

    if args.num_heads != 6:
        raise RuntimeError("This script assumes num_heads=6: bass,drums,inst,melody,vocals,fx")

    model = DeepVQEStemSeparator(n_fft=args.n_fft, num_heads=args.num_heads).to(device)
    model.train()

    stft = STFT(StftCfg(n_fft=args.n_fft, hop=args.hop, win=args.win)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        run = {"stem": 0.0, "mix": 0.0, "total": 0.0}
        pbar = tqdm(dl, desc=f"Epoch {epoch}", dynamic_ncols=True)

        for mix, bass, drums, inst, melody, vocals, is_vocal in pbar:
            mix = mix.to(device, non_blocking=True)        # (B,2,T)
            bass = bass.to(device, non_blocking=True)
            drums = drums.to(device, non_blocking=True)
            inst = inst.to(device, non_blocking=True)
            melody = melody.to(device, non_blocking=True)
            vocals = vocals.to(device, non_blocking=True)
            is_vocal = is_vocal.to(device, non_blocking=True)  # (B,)

            B, C, T = mix.shape  # C=2

            def flat(x: torch.Tensor) -> torch.Tensor:
                return x.reshape(B * C, T).float()

            mix_f = flat(mix)
            bass_f = flat(bass)
            drums_f = flat(drums)
            inst_f = flat(inst)
            melody_f = flat(melody)
            vocals_f = flat(vocals)

            stem_sum_f = bass_f + drums_f + inst_f + melody_f + vocals_f

            if args.fx_mode == "residual":
                fx_f = (mix_f - stem_sum_f)
            else:
                fx_f = torch.zeros_like(mix_f)

            # STFT targets (fp32, no grad)
            with torch.no_grad():
                mix_ri = stft.stft_ri(mix_f)  # (B*C,F,Tf,2)

                tgt = torch.stack(
                    [
                        stft.stft_ri(bass_f),
                        stft.stft_ri(drums_f),
                        stft.stft_ri(inst_f),
                        stft.stft_ri(melody_f),
                        stft.stft_ri(vocals_f),
                        stft.stft_ri(fx_f),
                    ],
                    dim=1,
                )  # (B*C,6,F,Tf,2)

            pred = model(mix_ri)  # (B*C,6,F,Tf,2)

            # head weights per track, then expand to B*C
            w = torch.ones((B, 6), device=device, dtype=torch.float32)
            # missing: melody on vocal tracks
            w[is_vocal, 3] = float(args.w_missing)
            # missing: vocals on novocal tracks
            w[~is_vocal, 4] = float(args.w_missing)
            # fx is ALWAYS present as a supervised head (zero or residual)
            w[:, 5] = float(args.w_fx)

            w_bc = w.repeat_interleave(C, dim=0)  # (B*C,6)

            diff = (pred.float() - tgt.float()).abs().mean(dim=(2, 3, 4))  # (B*C,6)
            loss_stem = (diff * w_bc).sum() / (w_bc.sum().clamp_min(1e-8))

            # IMPORTANT FIX: include fx in mixture sum
            mix_hat = pred.sum(dim=1)  # (B*C,F,Tf,2)
            loss_mix = l1_ri(mix_hat, mix_ri)

            loss = float(args.w_stem) * loss_stem + float(args.w_mix) * loss_mix

            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            opt.zero_grad(set_to_none=True)

            run["stem"] += float(loss_stem.detach().cpu())
            run["mix"] += float(loss_mix.detach().cpu())
            run["total"] += float(loss.detach().cpu())

            denom = max(1, pbar.n + 1)
            pbar.set_postfix(
                total=f"{run['total']/denom:.6f}",
                stem=f"{run['stem']/denom:.6f}",
                mix=f"{run['mix']/denom:.6f}",
            )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
        }
        if epoch % int(args.save_every_epochs) == 0:
            torch.save(ckpt, str(Path(args.save_dir) / f"phase_a_e{epoch:03d}.pt"))
        torch.save(ckpt, str(Path(args.save_dir) / "phase_a_latest.pt"))

        if args.dump_audio_every_epochs and (epoch % int(args.dump_audio_every_epochs) == 0):
            model.eval()
            with torch.no_grad():
                mix, bass, drums, inst, melody, vocals, is_vocal = next(iter(dl))
                mix = mix.to(device)
                B, C, T = mix.shape

                mix_f = mix.reshape(B * C, T).float()
                mix_ri = stft.stft_ri(mix_f)
                pred = model(mix_ri)  # (B*C,6,F,Tf,2)

                def unflat(y: torch.Tensor) -> torch.Tensor:
                    return y.reshape(B, C, T)

                names = ["bass", "drums", "inst", "melody", "vocals", "fx"]
                for head, name in enumerate(names):
                    y = stft.istft_ri(pred[:, head], length=T)  # (B*C,T)
                    y_st = unflat(y)[0].detach().cpu().transpose(0, 1).numpy()  # (T,2)
                    sf.write(str(dump_dir / f"e{epoch:03d}_ex0_{name}.wav"), y_st, args.sr, subtype="FLOAT")

                mix_w = unflat(stft.istft_ri(mix_ri, length=T))[0].cpu().transpose(0, 1).numpy()
                sf.write(str(dump_dir / f"e{epoch:03d}_ex0_mix.wav"), mix_w, args.sr, subtype="FLOAT")

                sum_ri = pred.sum(dim=1)
                sum_w = unflat(stft.istft_ri(sum_ri, length=T))[0].cpu().transpose(0, 1).numpy()
                sf.write(str(dump_dir / f"e{epoch:03d}_ex0_sum.wav"), sum_w, args.sr, subtype="FLOAT")

                print(f"[dump] wrote wavs to {dump_dir}")
            model.train()

    print("done.")


if __name__ == "__main__":
    main()
