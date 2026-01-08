from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
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


# -----------------------
# STFT helper (fp32) — same behavior as your train
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
            center=True,
            pad_mode="reflect",
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
            center=True,
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


def str_or_none(p: Path) -> str|None:
    if _is_file(p):
        return str(p)
    else:
        return


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

        if not (_is_file(full)):
            print('skip', d, 'missing full')
            continue


        if _is_file(vocals) and not _is_file(melody):
            items.append(
                TrackItem(
                    kind="vocal", # TODO delete vocal kind
                    full=str(full),
                    bass=str_or_none(bass),
                    drums=str_or_none(drums),
                    instruments=str_or_none(inst),
                    vocals=str_or_none(vocals),
                    melody=None,
                )
            )
        elif _is_file(melody) and not _is_file(vocals):
            items.append(
                TrackItem(
                    kind="novocal",
                    full=str(full),
                    bass=str_or_none(bass),
                    drums=str_or_none(drums),
                    instruments=str_or_none(inst),
                    vocals=None,
                    melody=str_or_none(melody),
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


class StemDataset(Dataset):
    """
    Returns RAW (unscaled) segments:
      full, bass, drums, inst, melody, vocals
    Scaling + mode selection happens in train loop.
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
    ):
        self.items = items
        self.sr = int(sr)
        self.seg_len = int(round(self.sr * float(segment_sec)))

        self.long_threshold_len = int(round(self.sr * float(long_threshold_sec)))
        self.long_hop_len = int(round(self.sr * float(long_hop_sec if long_hop_sec is not None else segment_sec)))
        self.long_hop_len = max(1, self.long_hop_len)
        self.long_jitter_len = int(round(self.sr * float(long_jitter_sec)))
        self.long_jitter_len = max(0, self.long_jitter_len)

        self.index: List[Tuple[int, int]] = []
        self._build_index()

    def _min_len(self, it: TrackItem) -> int:
        paths = [it.full, it.bass, it.drums, it.instruments]
        paths = list(filter(lambda x: x is not None, paths))
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
            n, sr, _ = _num_frames_and_sr(p)
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
            f"[StemDataset] tracks={len(self.items)} virtual_items={len(self.index)} "
            f"| seg_len={self.seg_len} long_thr={self.long_threshold_len} hop={self.long_hop_len} jitter={self.long_jitter_len}"
        )

    def __len__(self) -> int:
        return len(self.index)

    def load_segment(self, full, path: str|None, start_j):
        if path:
            return _read_stereo_segment(path, self.sr, start_j, self.seg_len)
        else:
            return torch.zeros_like(full)

    def __getitem__(self, idx: int):
        item_i, start = self.index[idx]
        it = self.items[item_i]

        if start < 0:
            n = self._min_len(it)
            if n <= self.seg_len:
                start_j = 0
            else:
                start_j = int(torch.randint(0, n - self.seg_len + 1, (1,)).item())
        else:
            j = int(torch.randint(-self.long_jitter_len, self.long_jitter_len + 1, (1,)).item()) if self.long_jitter_len > 0 else 0
            n = self._min_len(it)
            max_start = max(0, n - self.seg_len)
            start_j = int(max(0, min(max_start, start + j)))

        full = _read_stereo_segment(it.full, self.sr, start_j, self.seg_len)

        bass = self.load_segment(full, it.bass, start_j)
        drums = self.load_segment(full, it.drums, start_j)
        inst = self.load_segment(full, it.instruments, start_j)
        vocals = self.load_segment(full, it.vocals, start_j)
        melody = self.load_segment(full, it.melody, start_j)

        return full, bass, drums, inst, melody, vocals


def collate(batch):
    full = torch.stack([b[0] for b in batch], dim=0)     # (B,2,T)
    bass = torch.stack([b[1] for b in batch], dim=0)
    drums = torch.stack([b[2] for b in batch], dim=0)
    inst = torch.stack([b[3] for b in batch], dim=0)
    melody = torch.stack([b[4] for b in batch], dim=0)
    vocals = torch.stack([b[5] for b in batch], dim=0)
    return full, bass, drums, inst, melody, vocals


# -----------------------
# Utils
# -----------------------
def l1_ri(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).abs().mean()


def linear_ramp(epoch: int, start: int, end: int, v0: float, v1: float) -> float:
    if end <= start:
        return float(v1 if epoch >= end else v0)
    if epoch <= start:
        return float(v0)
    if epoch >= end:
        return float(v1)
    t = (epoch - start) / (end - start)
    return float(v0 + (v1 - v0) * t)


def apply_clip_safe_scale(
    peak_ref: torch.Tensor,          # (B,2,T)
    signals: List[torch.Tensor],     # each (B,2,T)
    peak_target: float = 0.98,
    eps: float = 1e-12,
) -> List[torch.Tensor]:
    p = peak_ref.abs().amax(dim=(1, 2))  # (B,)
    scale = torch.ones_like(p)
    mask = p > 1.0
    scale[mask] = float(peak_target) / (p[mask].clamp_min(eps))
    s = scale[:, None, None]
    return [x * s for x in signals]


def save_ckpt(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_ckpt(path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(f"Bad checkpoint format: {path}")
    return ckpt


def _as_uint8_tensor(x) -> Optional[torch.ByteTensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").to(torch.uint8)  # type: ignore[return-value]
    # если вдруг попалась list[int]
    try:
        t = torch.tensor(x, dtype=torch.uint8)
        return t  # type: ignore[return-value]
    except Exception:
        return None


def restore_rng(ckpt: Dict[str, Any]) -> None:
    st = _as_uint8_tensor(ckpt.get("rng_state_torch"))
    if st is not None:
        torch.set_rng_state(st)

    if torch.cuda.is_available():
        cuda_states = ckpt.get("rng_state_cuda")
        if isinstance(cuda_states, list):
            fixed = []
            ok = True
            for s in cuda_states:
                t = _as_uint8_tensor(s)
                if t is None:
                    ok = False
                    break
                fixed.append(t)
            if ok:
                try:
                    torch.cuda.set_rng_state_all(fixed)
                except Exception:
                    pass

    np_state = ckpt.get("rng_state_numpy")
    if np_state is not None:
        try:
            np.random.set_state(np_state)
        except Exception:
            pass


def capture_rng() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["rng_state_torch"] = torch.get_rng_state()
    if torch.cuda.is_available():
        out["rng_state_cuda"] = torch.cuda.get_rng_state_all()
    out["rng_state_numpy"] = np.random.get_state()
    return out


# -----------------------
# Training
# -----------------------
def main():
    ap = argparse.ArgumentParser()

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--root", type=str)
    g.add_argument("--manifest", type=str)

    ap.add_argument("--save-dir", default="ckpt_phase_ab_4stem")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--save-every-epochs", type=int, default=1)

    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--reset-opt", action="store_true")
    ap.add_argument("--reset-rng", action="store_true")

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

    # HEADS: bass, drums, music(inst+melody), vocals
    ap.add_argument("--num-heads", type=int, default=4)

    # A->B schedule for INPUT mix
    # Mode A input: mix_in = stem_sum
    # Mode B input: mix_in = full
    ap.add_argument("--ab-start-epoch", type=int, default=1)
    ap.add_argument("--ab-end-epoch", type=int, default=10)
    ap.add_argument("--ab-prob-start", type=float, default=0.0)
    ap.add_argument("--ab-prob-end", type=float, default=0.7)  # ВАЖНО: я бы не делал 1.0 сразу

    # losses
    ap.add_argument("--w-stem", type=float, default=1.0)
    ap.add_argument("--w-mix", type=float, default=0.5)

    # optional per-head weights
    ap.add_argument("--w-bass", type=float, default=1.0)
    ap.add_argument("--w-drums", type=float, default=1.0)
    ap.add_argument("--w-music", type=float, default=1.0)
    ap.add_argument("--w-vocals", type=float, default=1.0)

    ap.add_argument("--limit-items", type=int, default=0)
    ap.add_argument("--dump-audio-every-epochs", type=int, default=0)
    ap.add_argument("--dump-dir", type=str, default="dumps_phase_ab_4stem")

    args = ap.parse_args()

    if args.num_heads != 4:
        raise RuntimeError("This script assumes num_heads=4: bass, drums, music(inst+melody), vocals")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print("device:", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    items = scan_root_to_items(args.root) if args.root else load_manifest_csv(args.manifest)
    if args.limit_items and args.limit_items > 0:
        items = items[: int(args.limit_items)]
        print(f"[info] limit-items={len(items)}")

    ds = StemDataset(
        items,
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
    print(f"dataset: {len(ds)} segments | batch={args.batch} | batches/epoch={len(dl)}")

    model = DeepVQEStemSeparator(n_fft=args.n_fft, num_heads=args.num_heads).to(device)
    stft = STFT(StftCfg(n_fft=args.n_fft, hop=args.hop, win=args.win)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    global_step = 0

    if args.resume:
        ckpt = load_ckpt(args.resume, device=device)
        model.load_state_dict(ckpt["model"], strict=True)
        if (not args.reset_opt) and ("opt" in ckpt):
            try:
                opt.load_state_dict(ckpt["opt"])
            except Exception as e:
                print(f"[warn] failed to load optimizer state: {e}")
        if (not args.reset_rng):
            restore_rng(ckpt)

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(f"[resume] loaded {args.resume}")
        print(f"[resume] will start from epoch={start_epoch}")

    if start_epoch > args.epochs:
        print(f"[info] nothing to do: start_epoch={start_epoch} > --epochs={args.epochs}")
        return

    model.train()

    # per-head weights (order: bass, drums, music, vocals)
    head_w = torch.tensor(
        [args.w_bass, args.w_drums, args.w_music, args.w_vocals],
        dtype=torch.float32,
        device=device,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        p_full = linear_ramp(
            epoch=epoch,
            start=int(args.ab_start_epoch),
            end=int(args.ab_end_epoch),
            v0=float(args.ab_prob_start),
            v1=float(args.ab_prob_end),
        )
        p_full = float(np.clip(p_full, 0.0, 1.0))

        run = {"stem": 0.0, "mix": 0.0, "total": 0.0}
        pbar = tqdm(dl, desc=f"Epoch {epoch}", dynamic_ncols=True)

        for full, bass, drums, inst, melody, vocals in pbar:
            full = full.to(device, non_blocking=True)      # (B,2,T)
            bass = bass.to(device, non_blocking=True)
            drums = drums.to(device, non_blocking=True)
            inst = inst.to(device, non_blocking=True)
            melody = melody.to(device, non_blocking=True)
            vocals = vocals.to(device, non_blocking=True)

            B, C, T = full.shape

            # merge
            music = inst + melody                       # (B,2,T)
            stem_sum = bass + drums + music + vocals    # (B,2,T)  <-- THIS is the canonical target mix

            # choose input mode per example
            use_full = (torch.rand((B,), device=device) < p_full)  # (B,)
            mix_in = torch.where(use_full[:, None, None], full, stem_sum)

            # clip-safe scale based on chosen input, apply to all signals + stem_sum
            full, bass, drums, music, vocals, stem_sum, mix_in = apply_clip_safe_scale(
                peak_ref=mix_in,
                signals=[full, bass, drums, music, vocals, stem_sum, mix_in],
                peak_target=0.98,
            )

            # flatten (B,2,T) -> (B*C,T)
            def flat(x: torch.Tensor) -> torch.Tensor:
                return x.reshape(B * C, T).float()

            mix_f = flat(mix_in)
            bass_f = flat(bass)
            drums_f = flat(drums)
            music_f = flat(music)
            vocals_f = flat(vocals)
            stem_sum_f = flat(stem_sum)

            with torch.no_grad():
                mix_ri = stft.stft_ri(mix_f)                # input (B*C,F,Tf,2)
                tgt = torch.stack(
                    [
                        stft.stft_ri(bass_f),
                        stft.stft_ri(drums_f),
                        stft.stft_ri(music_f),
                        stft.stft_ri(vocals_f),
                    ],
                    dim=1,
                )  # (B*C,4,F,Tf,2)
                stem_sum_ri = stft.stft_ri(stem_sum_f)      # TARGET MIX ALWAYS = stem_sum

            pred = model(mix_ri)  # (B*C,4,F,Tf,2)

            # stem loss (weighted per head)
            diff = (pred.float() - tgt.float()).abs().mean(dim=(2, 3, 4))  # (B*C,4)
            loss_stem = (diff * head_w[None, :]).sum(dim=1).mean() / head_w.mean().clamp_min(1e-8)

            # mixture consistency: sum(pred_stems) must match stem_sum (not full!)
            mix_hat = pred.sum(dim=1)  # (B*C,F,Tf,2)
            loss_mix = l1_ri(mix_hat, stem_sum_ri)

            loss = float(args.w_stem) * loss_stem + float(args.w_mix) * loss_mix

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            global_step += 1

            run["stem"] += float(loss_stem.detach().cpu())
            run["mix"] += float(loss_mix.detach().cpu())
            run["total"] += float(loss.detach().cpu())

            denom = max(1, pbar.n + 1)
            pbar.set_postfix(
                total=f"{run['total']/denom:.6f}",
                stem=f"{run['stem']/denom:.6f}",
                mix=f"{run['mix']/denom:.6f}",
                pFull=f"{p_full:.2f}",
            )

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            **capture_rng(),
        }
        if epoch % int(args.save_every_epochs) == 0:
            save_ckpt(save_dir / f"phase_ab_4stem_e{epoch:03d}.pt", ckpt)
        save_ckpt(save_dir / "phase_ab_4stem_latest.pt", ckpt)

        if args.dump_audio_every_epochs and (epoch % int(args.dump_audio_every_epochs) == 0):
            model.eval()
            with torch.no_grad():
                full, bass, drums, inst, melody, vocals = next(iter(dl))
                full = full.to(device)
                bass = bass.to(device)
                drums = drums.to(device)
                inst = inst.to(device)
                melody = melody.to(device)
                vocals = vocals.to(device)

                B, C, T = full.shape
                music = inst + melody
                stem_sum = bass + drums + music + vocals

                use_full = (torch.rand((B,), device=device) < p_full)
                mix_in = torch.where(use_full[:, None, None], full, stem_sum)

                full, bass, drums, music, vocals, stem_sum, mix_in = apply_clip_safe_scale(
                    peak_ref=mix_in,
                    signals=[full, bass, drums, music, vocals, stem_sum, mix_in],
                    peak_target=0.98,
                )

                mix_f = mix_in.reshape(B * C, T).float()
                mix_ri = stft.stft_ri(mix_f)
                pred = model(mix_ri)  # (B*C,4,F,Tf,2)

                def unflat(y: torch.Tensor) -> torch.Tensor:
                    return y.reshape(B, C, T)

                names = ["bass", "drums", "music", "vocals"]
                for head, name in enumerate(names):
                    y = stft.istft_ri(pred[:, head], length=T)  # (B*C,T)
                    y_st = unflat(y)[0].detach().cpu().transpose(0, 1).numpy()  # (T,2)
                    sf.write(str(dump_dir / f"e{epoch:03d}_ex0_{name}.wav"), y_st, args.sr, subtype="FLOAT")

                # refs
                sf.write(str(dump_dir / f"e{epoch:03d}_ex0_mix_in.wav"), mix_in[0].cpu().transpose(0, 1).numpy(), args.sr, subtype="FLOAT")
                sf.write(str(dump_dir / f"e{epoch:03d}_ex0_stem_sum.wav"), stem_sum[0].cpu().transpose(0, 1).numpy(), args.sr, subtype="FLOAT")
                sf.write(str(dump_dir / f"e{epoch:03d}_ex0_full.wav"), full[0].cpu().transpose(0, 1).numpy(), args.sr, subtype="FLOAT")

                sum_ri = pred.sum(dim=1)
                sum_w = unflat(stft.istft_ri(sum_ri, length=T))[0].cpu().transpose(0, 1).numpy()
                sf.write(str(dump_dir / f"e{epoch:03d}_ex0_sum.wav"), sum_w, args.sr, subtype="FLOAT")

                (dump_dir / f"e{epoch:03d}_ex0_mode.txt").write_text(
                    f"epoch={epoch}\npFull={p_full:.4f}\nuse_full_ex0={bool(use_full[0].item())}\n",
                    encoding="utf-8",
                )

                print(f"[dump] wrote wavs to {dump_dir}")
            model.train()

    print("done.")


if __name__ == "__main__":
    main()
