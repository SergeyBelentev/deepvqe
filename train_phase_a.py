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
from torch.utils.data import Dataset, DataLoader, get_worker_info
from tqdm import tqdm
from deepvqe import DeepVQEStemSeparator
import os, random
import boto3
from botocore.config import Config as BotoConfig
from boto3.s3.transfer import TransferConfig
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import json


def ddp_setup():
    # torchrun выставит эти env vars
    if "RANK" not in os.environ:
        return False, 0, 1, 0  # not distributed

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True, rank, world, local_rank


def seed_worker(worker_id: int) -> None:
    # seed, который DataLoader назначил воркеру
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)

    # очень важно против oversubscription
    torch.set_num_threads(1)
    # если используешь torch.set_num_interop_threads — делай это в main, не в воркерах

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
# Multi-Resolution STFT loss (time-domain)
# -----------------------
def _parse_int_csv(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


class MRSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss:
      loss = mean_over_resolutions( sc_weight * spectral_convergence + logmag_weight * logmag_l1 )

    Returns:
      per-example loss vector: (N,)
    """

    def __init__(
        self,
        n_ffts: List[int],
        hops: List[int],
        wins: List[int],
        *,
        sc_weight: float = 1.0,
        logmag_weight: float = 1.0,
        eps: float = 1e-7,
        chunk: int = 0,   # 0 -> no chunking
    ) -> None:
        super().__init__()
        if not (len(n_ffts) == len(hops) == len(wins)) or len(n_ffts) == 0:
            raise ValueError("MRSTFTLoss: n_ffts/hops/wins must have same non-zero length")

        self.cfgs = list(zip([int(x) for x in n_ffts], [int(x) for x in hops], [int(x) for x in wins]))
        self.sc_weight = float(sc_weight)
        self.logmag_weight = float(logmag_weight)
        self.eps = float(eps)
        self.chunk = int(chunk)

        # windows are buffers so .to(device) moves them once (no per-step copies)
        for i, (_, _, win) in enumerate(self.cfgs):
            self.register_buffer(
                f"window_{i}",
                torch.hann_window(int(win), dtype=torch.float32),
                persistent=False,
            )

    def _one_res_loss(self, x: torch.Tensor, y: torch.Tensor, i: int) -> torch.Tensor:
        """
        x,y: (N,T) float32
        returns: (N,) for one resolution
        """
        n_fft, hop, win = self.cfgs[i]
        w = getattr(self, f"window_{i}").to(device=x.device, dtype=torch.float32)

        X = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=w,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )
        Y = torch.stft(
            y,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=w,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )

        magX = X.abs()
        magY = Y.abs()

        # spectral convergence per-example
        # ||Y - X||_F / ||Y||_F
        diff = (magY - magX).reshape(magY.shape[0], -1)
        denom = magY.reshape(magY.shape[0], -1)

        sc = torch.linalg.vector_norm(diff, dim=1) / torch.linalg.vector_norm(denom, dim=1).clamp_min(self.eps)

        # log-mag L1 per-example
        logmag = (torch.log(magY + self.eps) - torch.log(magX + self.eps)).abs().mean(dim=(1, 2))

        return (self.sc_weight * sc) + (self.logmag_weight * logmag)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x,y: (N,T) any float dtype -> internally float32
        returns: (N,) per-example loss
        """
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(f"MRSTFTLoss expects (N,T), got x={tuple(x.shape)} y={tuple(y.shape)}")
        if x.shape != y.shape:
            raise ValueError(f"MRSTFTLoss shape mismatch: x={tuple(x.shape)} y={tuple(y.shape)}")

        x = x.float()
        y = y.float()

        N = x.shape[0]
        chunk = self.chunk if self.chunk and self.chunk > 0 else N

        outs: List[torch.Tensor] = []
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            xc = x[s:e]
            yc = y[s:e]

            loss_c = xc.new_zeros((e - s,), dtype=torch.float32)
            for i in range(len(self.cfgs)):
                loss_c = loss_c + self._one_res_loss(xc, yc, i)
            loss_c = loss_c / float(len(self.cfgs))
            outs.append(loss_c)

        return torch.cat(outs, dim=0)



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


STEM_ORDER = ["bass", "drums", "music", "vocals"]
STEM_SET = set(STEM_ORDER)

_STEM_ALIASES = {
    "bass": "bass",
    "drums": "drums",
    "music": "music",
    "instrumental": "music",
    "instruments": "music",
    "inst": "music",
    "melody": "music",
    "vocals": "vocals",
    "vocal": "vocals",
    "voice": "vocals",
}


def _norm_stem_name(x: str) -> str:
    k = x.strip().lower()
    if k not in _STEM_ALIASES:
        raise ValueError(f"Unknown stem name: {x!r}. Allowed: {sorted(set(_STEM_ALIASES.keys()))}")
    return _STEM_ALIASES[k]


@dataclass(frozen=True)
class Recipe:
    prob: float
    type: str                      # "unconditional" (пока)
    mix_in: str                    # "stem_sum" | "full"
    stem_sum_mode: str             # "fixed" | "random" | "random_within_track"
    stem_count: int                # 1..4
    required_stems: Tuple[str, ...]
    available_stems: Tuple[str, ...]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Recipe":
        prob = float(d.get("prob", 0.0))
        typ = str(d.get("type", "unconditional"))
        mix_in = str(d.get("mix_in", "stem_sum"))
        mode = str(d.get("stem_sum_mode", "fixed"))
        stem_count = int(d.get("stem_count", 4))

        req = tuple(_norm_stem_name(x) for x in (d.get("required_stems", []) or []))
        avl = tuple(_norm_stem_name(x) for x in (d.get("available_stems", []) or []))

        if stem_count < 1 or stem_count > 4:
            raise ValueError(f"stem_count must be 1..4, got {stem_count}")

        if any(s not in STEM_SET for s in req + avl):
            raise ValueError(f"Bad stems in recipe: req={req} avl={avl}")

        if len(set(req)) != len(req):
            raise ValueError(f"required_stems has duplicates: {req}")

        # stem_count must cover required
        if len(req) > stem_count:
            raise ValueError(f"stem_count={stem_count} < len(required_stems)={len(req)}")

        # mode validation
        if mode not in ("fixed", "random", "random_within_track"):
            raise ValueError(f"stem_sum_mode must be one of fixed|random|random_within_track, got {mode!r}")

        if mix_in not in ("stem_sum", "full"):
            raise ValueError(f"mix_in must be one of stem_sum|full, got {mix_in!r}")

        # full only makes sense for fixed (иначе targets не соответствуют входу)
        if mix_in == "full" and mode != "fixed":
            raise ValueError("mix_in='full' requires stem_sum_mode='fixed' (otherwise input/targets mismatch).")

        return Recipe(
            prob=prob,
            type=typ,
            mix_in=mix_in,
            stem_sum_mode=mode,
            stem_count=stem_count,
            required_stems=req,
            available_stems=avl,
        )

    def pick_active_stems(self, rng: np.random.Generator) -> Tuple[str, ...]:
        """
        Возвращает какие стемы "включены" в этом сэмпле (и должны быть в mix_in, и быть ненулевыми в таргете).
        Остальные таргеты = 0 (тишина).
        """
        active = list(self.required_stems)
        need = self.stem_count - len(active)
        if need > 0:
            pool = [s for s in self.available_stems if s not in active]
            if len(pool) < need:
                raise ValueError(
                    f"Not enough available_stems to reach stem_count. need={need} pool={pool} req={active}"
                )
            pick = rng.choice(pool, size=need, replace=False).tolist()
            active.extend(pick)
        # фиксируем порядок стабильным (не обязательно, но удобнее)
        active = sorted(active, key=lambda s: STEM_ORDER.index(s))
        return tuple(active)


class RecipeBook:
    """
    JSON:
    {
      "1": [ {recipe}, {recipe}, ... ],
      "2": [ ... ],
      ...
    }
    """

    def __init__(self, plans: Dict[int, List[Recipe]]) -> None:
        if not plans:
            raise ValueError("RecipeBook: empty plans")
        self.plans = {int(k): v for k, v in plans.items()}
        self.epochs_sorted = sorted(self.plans.keys())

        # validate probs per epoch
        for e in self.epochs_sorted:
            rs = self.plans[e]
            s = sum(r.prob for r in rs)
            if s <= 0:
                raise ValueError(f"Epoch {e}: sum(prob) must be >0, got {s}")

    @staticmethod
    def from_json_path(path: str) -> "RecipeBook":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("RecipeBook JSON must be an object {epoch: [recipes]}")
        plans: Dict[int, List[Recipe]] = {}
        for k, v in data.items():
            e = int(k)
            if not isinstance(v, list):
                raise ValueError(f"Epoch {k}: value must be list of recipes")
            plans[e] = [Recipe.from_dict(x) for x in v]
        return RecipeBook(plans)

    def plan_for_epoch(self, epoch: int) -> List[Recipe]:
        # берём последний определённый epoch <= текущего
        e = int(epoch)
        cand = [x for x in self.epochs_sorted if x <= e]
        use = cand[-1] if cand else self.epochs_sorted[0]
        return self.plans[use]

    @staticmethod
    def _choose_weighted(rng: np.random.Generator, recipes: List[Recipe]) -> Recipe:
        w = np.array([max(0.0, r.prob) for r in recipes], dtype=np.float64)
        w = w / w.sum()
        i = int(rng.choice(len(recipes), p=w))
        return recipes[i]

    def sample_recipe(self, rng: np.random.Generator, epoch: int) -> Recipe:
        rs = self.plan_for_epoch(epoch)
        return self._choose_weighted(rng, rs)


class FlexibleMixDataset(Dataset):
    """
    Генерирует один тренировочный пример:
      mix_in: (2,T)
      tgt:    (4,2,T) в порядке STEM_ORDER
      present_mask: (4,) 1 если стем включен в mix_in, иначе 0

    Реализует:
      - stem_sum_mode=fixed: все стемы из одной и той же позиции трека (как раньше)
      - stem_sum_mode=random: каждый стем из независимого трека/сегмента (гипотеза 1)
      - stem_count/required/available: частичные миксы (гипотеза 2/3)
    """

    def __init__(
        self,
        items: List["TrackItem"],
        *,
        sr: int,
        segment_sec: float,
        recipe_book: RecipeBook,
        long_threshold_sec: float = 6.0,
        long_hop_sec: Optional[float] = None,
        long_jitter_sec: float = 0.0,
        epoch_size: int = 200_000,   # сколько __getitem__ даём "виртуально"
    ):
        super().__init__()
        self.items = items
        self.sr = int(sr)
        self.seg_len = int(round(self.sr * float(segment_sec)))

        self.long_threshold_len = int(round(self.sr * float(long_threshold_sec)))
        self.long_hop_len = int(round(self.sr * float(long_hop_sec if long_hop_sec is not None else segment_sec)))
        self.long_hop_len = max(1, self.long_hop_len)
        self.long_jitter_len = int(round(self.sr * float(long_jitter_sec)))
        self.long_jitter_len = max(0, self.long_jitter_len)

        self.epoch_size = int(epoch_size)
        self.book = recipe_book
        self._epoch = 1

        # кеш минимальной длины по доступным стемам (как у тебя)
        self._min_len_cache = {i: self._min_len(it) for i, it in enumerate(self.items)}

        # пулы треков, где можно брать сегменты
        self.valid_any: List[int] = []
        self.valid_vocals: List[int] = []
        for i, it in enumerate(self.items):
            n = self._min_len_cache[i]
            if n >= self.seg_len and n > 0:
                self.valid_any.append(i)
                if it.vocals:  # только треки с vocals.wav
                    self.valid_vocals.append(i)

        if not self.valid_any:
            raise RuntimeError("FlexibleMixDataset: no valid tracks with length >= segment")
        if not self.valid_vocals:
            # можно, но тогда recipes с required vocals не сработают
            print("[warn] FlexibleMixDataset: no vocal tracks (vocals.wav). 'vocals' recipes will fail.")

        # для fixed-режима мы хотим как раньше иметь list возможных starts
        self.track_starts: Dict[int, List[int]] = {}
        for i in self.valid_any:
            n = self._min_len_cache[i]
            self.track_starts[i] = self._build_starts_for_track(n)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.epoch_size

    def _min_len(self, it: "TrackItem") -> int:
        # минимальная длина среди full/bass/drums/instruments и vocals/melody (как у тебя)
        paths = [it.full, it.bass, it.drums, it.instruments]
        paths = [p for p in paths if p is not None]
        if it.vocals:
            paths.append(it.vocals)
        if it.melody:
            paths.append(it.melody)

        mins = None
        for p in paths:
            n, sr, _ = _num_frames_and_sr(p)
            if sr != self.sr:
                # этот трек фактически невалиден
                return 0
            mins = n if mins is None else min(mins, n)
        return int(mins or 0)

    def _build_starts_for_track(self, n: int) -> List[int]:
        # повторяем логику твоего индексатора: для длинных треков — fixed starts, иначе -1 (рандом)
        if n <= 0:
            return []
        if n < self.long_threshold_len:
            return [-1]
        if n <= self.seg_len:
            return [0]
        max_start = n - self.seg_len
        starts = list(range(0, max_start + 1, self.long_hop_len))
        if starts[-1] != max_start:
            starts.append(max_start)
        return starts

    def _random_start_from_starts(self, rng: np.random.Generator, track_i: int) -> int:
        starts = self.track_starts.get(track_i)
        n = self._min_len_cache[track_i]
        if not starts:
            # fallback: uniform
            if n <= self.seg_len:
                return 0
            return int(rng.integers(0, n - self.seg_len + 1))
        base = int(rng.choice(starts))
        if base < 0:
            if n <= self.seg_len:
                return 0
            return int(rng.integers(0, n - self.seg_len + 1))
        if self.long_jitter_len > 0:
            j = int(rng.integers(-self.long_jitter_len, self.long_jitter_len + 1))
        else:
            j = 0
        max_start = max(0, n - self.seg_len)
        return int(max(0, min(max_start, base + j)))

    def _rng_for_item(self, idx: int) -> np.random.Generator:
        # детерминизм на уровне воркера: worker seed + epoch + idx
        wi = get_worker_info()
        if wi is None:
            seed = (torch.initial_seed() % (2**32))
        else:
            seed = (wi.seed % (2**32))
        # смешиваем с epoch и idx
        seed = (seed + 1000003 * int(self._epoch) + 9176 * int(idx)) % (2**32)
        return np.random.default_rng(seed)

    def _pick_track_for_stem(self, rng: np.random.Generator, stem: str) -> int:
        if stem == "vocals":
            if not self.valid_vocals:
                raise RuntimeError("No vocal tracks in dataset, but recipe requires vocals.")
            return int(rng.choice(self.valid_vocals))
        return int(rng.choice(self.valid_any))

    def _load_stem(self, track_i: int, start: int, stem: str) -> torch.Tensor:
        it = self.items[track_i]
        # full / bass / drums / instruments / melody / vocals — у тебя уже есть _read_stereo_segment
        if stem == "bass":
            return _read_stereo_segment(it.bass, self.sr, start, self.seg_len) if it.bass else torch.zeros((2, self.seg_len), dtype=torch.float32)
        if stem == "drums":
            return _read_stereo_segment(it.drums, self.sr, start, self.seg_len) if it.drums else torch.zeros((2, self.seg_len), dtype=torch.float32)
        if stem == "vocals":
            return _read_stereo_segment(it.vocals, self.sr, start, self.seg_len) if it.vocals else torch.zeros((2, self.seg_len), dtype=torch.float32)
        if stem == "music":
            inst = _read_stereo_segment(it.instruments, self.sr, start, self.seg_len) if it.instruments else torch.zeros((2, self.seg_len), dtype=torch.float32)
            mel = _read_stereo_segment(it.melody, self.sr, start, self.seg_len) if it.melody else torch.zeros((2, self.seg_len), dtype=torch.float32)
            return inst + mel
        raise ValueError(f"Unknown stem: {stem}")

    def _load_full_and_all_stems_fixed(self, track_i: int, start: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        it = self.items[track_i]
        full = _read_stereo_segment(it.full, self.sr, start, self.seg_len)
        stems = {
            "bass": self._load_stem(track_i, start, "bass"),
            "drums": self._load_stem(track_i, start, "drums"),
            "music": self._load_stem(track_i, start, "music"),
            "vocals": self._load_stem(track_i, start, "vocals"),
        }
        return full, stems

    def __getitem__(self, idx: int):
        rng = self._rng_for_item(idx)
        recipe = self.book.sample_recipe(rng, epoch=self._epoch)
        if recipe.type != "unconditional":
            raise ValueError(f"Unsupported recipe type: {recipe.type}")

        active = recipe.pick_active_stems(rng)  # tuple[str]

        # targets (4,2,T) и mask (4,)
        tgt = {s: torch.zeros((2, self.seg_len), dtype=torch.float32) for s in STEM_ORDER}
        present = {s: 0.0 for s in STEM_ORDER}
        full = None

        if recipe.stem_sum_mode == "fixed":
            # один трек/позиция на все стемы
            track_i = int(rng.choice(self.valid_any))
            start = self._random_start_from_starts(rng, track_i)
            full, stems = self._load_full_and_all_stems_fixed(track_i, start)
            for s in active:
                tgt[s] = stems[s]
                present[s] = 1.0

            if recipe.mix_in == "full":
                mix_in = full
            else:
                mix_in = tgt["bass"] + tgt["drums"] + tgt["music"] + tgt["vocals"]

        elif recipe.stem_sum_mode == "random":
            # каждый активный стем из отдельного (случайного) трека/позиции
            for s in active:
                tr = self._pick_track_for_stem(rng, s)
                st = self._random_start_from_starts(rng, tr)
                tgt[s] = self._load_stem(tr, st, s)
                present[s] = 1.0
            mix_in = tgt["bass"] + tgt["drums"] + tgt["music"] + tgt["vocals"]

        elif recipe.stem_sum_mode == "random_within_track":
            # один трек общий, но время для каждого стема рандомное (мягче, чем random)
            track_i = int(rng.choice(self.valid_any))
            for s in active:
                st = self._random_start_from_starts(rng, track_i)
                tgt[s] = self._load_stem(track_i, st, s)
                present[s] = 1.0
            mix_in = tgt["bass"] + tgt["drums"] + tgt["music"] + tgt["vocals"]

        else:
            raise ValueError(f"Bad stem_sum_mode: {recipe.stem_sum_mode}")

        # clip-safe scale по mix_in, применяем ко всем таргетам и mix_in
        mix_in_b = mix_in.unsqueeze(0)  # (1,2,T)
        signals = [mix_in_b] + [tgt[s].unsqueeze(0) for s in STEM_ORDER]
        scaled = apply_clip_safe_scale(peak_ref=mix_in_b, signals=signals, peak_target=0.98)
        mix_in_b = scaled[0]
        for i, s in enumerate(STEM_ORDER):
            tgt[s] = scaled[i + 1].squeeze(0)

        mix_in = mix_in_b.squeeze(0)

        tgt_tensor = torch.stack([tgt[s] for s in STEM_ORDER], dim=0)  # (4,2,T)
        present_mask = torch.tensor([present[s] for s in STEM_ORDER], dtype=torch.float32)  # (4,)

        return mix_in, tgt_tensor, present_mask



def collate(batch):
    mix = torch.stack([b[0] for b in batch], dim=0)          # (B,2,T)
    tgt = torch.stack([b[1] for b in batch], dim=0)          # (B,4,2,T)
    pm  = torch.stack([b[2] for b in batch], dim=0)          # (B,4)
    return mix, tgt, pm


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


def load_ckpt(path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(f"Bad checkpoint format: {path}")
    return ckpt


def atomic_save_ckpt(path: Path, payload: Dict[str, Any]) -> None:
    """
    Пишем через tmp + os.replace, чтобы 'latest' никогда не был наполовину записан.
    Это особенно важно на spot/при падениях.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(str(tmp), str(path))


def make_s3_client(
    *,
    region: str = "",
    endpoint_url: str = "",
    access_key_id: str = "",
    secret_access_key: str = "",
    session_token: str = "",
    profile: str = "",
):
    """
    Если access_key_id/secret_access_key заданы — используем их.
    Иначе boto3 использует стандартную credential chain:
      env vars -> shared config/profile -> EC2/ECS role (IMDS) -> ...
    """
    cfg = BotoConfig(retries={"max_attempts": 10, "mode": "adaptive"})

    # Session нужен, чтобы корректно поддержать profile и/или явные креды.
    if profile:
        sess = boto3.Session(profile_name=profile, region_name=region or None)
        return sess.client("s3", endpoint_url=(endpoint_url or None), config=cfg)

    # Если дали ключи — используем их явно
    if access_key_id and secret_access_key:
        sess = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=(session_token or None),
            region_name=(region or None),
        )
        return sess.client("s3", endpoint_url=(endpoint_url or None), config=cfg)

    # иначе обычная цепочка boto3 (env/role/etc.)
    sess = boto3.Session(region_name=(region or None))
    return sess.client("s3", endpoint_url=(endpoint_url or None), config=cfg)



def upload_to_s3(
    s3,
    *,
    local_path: Path,
    bucket: str,
    key: str,
) -> None:
    # Мультипарт + параллелизм для больших чекпоинтов
    tcfg = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )
    s3.upload_file(str(local_path), bucket, key, Config=tcfg)


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
    ap.add_argument("--recipe-json", type=str, required=True)
    ap.add_argument("--epoch-size", type=int, default=200000)

    ap.add_argument("--save-dir", default="ckpt_phase_ab_4stem")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--save-every-epochs", type=int, default=1)

    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--reset-opt", action="store_true")
    ap.add_argument("--reset-rng", action="store_true")

    # Enable TF32
    ap.add_argument("--enable-tf32", action="store_true")

    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--segment-sec", type=float, default=4.0)
    ap.add_argument("--long-threshold-sec", type=float, default=6.0)
    ap.add_argument("--long-hop-sec", type=float, default=None)
    ap.add_argument("--long-jitter-sec", type=float, default=0.0)

    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--pin-memory", action="store_true", default=True)
    ap.add_argument("--persistent-workers", action="store_true", default=True)
    ap.add_argument("--in-order", action="store_true", default=True)

    ap.add_argument("--n-fft", type=int, default=1536)
    ap.add_argument("--hop", type=int, default=480)
    ap.add_argument("--win", type=int, default=1536)

    # HEADS: bass, drums, music(inst+melody), vocals
    ap.add_argument("--num-heads", type=int, default=4)

    # losses
    ap.add_argument("--w-stem", type=float, default=1.0)
    ap.add_argument("--w-mix", type=float, default=0.5)

    # ---- MR-STFT (time-domain) over ALL heads ----
    ap.add_argument("--w-mrstft", type=float, default=0.0)
    ap.add_argument("--mr-nffts", type=str, default="512,1024,2048")
    ap.add_argument("--mr-hops", type=str, default="120,240,480")
    ap.add_argument("--mr-wins", type=str, default="512,1024,2048")
    ap.add_argument("--mr-sc-weight", type=float, default=1.0)
    ap.add_argument("--mr-logmag-weight", type=float, default=1.0)
    ap.add_argument("--mr-eps", type=float, default=1e-7)
    ap.add_argument("--mr-chunk", type=int, default=20)  # <= важно для памяти
    ap.add_argument("--mr-every", type=int, default=1)  # считать раз в N шагов (1 = каждый шаг)

    # optional per-head weights
    ap.add_argument("--w-bass", type=float, default=1.0)
    ap.add_argument("--w-drums", type=float, default=1.0)
    ap.add_argument("--w-music", type=float, default=1.0)
    ap.add_argument("--w-vocals", type=float, default=1.0)
    ap.add_argument("--silence-weight", type=float, default=0.35)

    ap.add_argument("--limit-items", type=int, default=0)
    ap.add_argument("--dump-audio-every-epochs", type=int, default=0)
    ap.add_argument("--dump-dir", type=str, default="dumps_phase_ab_4stem")

    # S3
    ap.add_argument("--s3-bucket", type=str, default="")
    ap.add_argument("--s3-prefix", type=str, default="deepvqe_ckpt")  # папка в бакете
    ap.add_argument("--s3-region", type=str, default="")             # можно пусто на EC2 с ролью
    ap.add_argument("--s3-endpoint-url", type=str, default="")       # если MinIO/CEPH, иначе пусто
    ap.add_argument("--s3-upload-latest-only", action="store_true")  # опционально
    ap.add_argument("--s3-access-key-id", type=str, default="")
    ap.add_argument("--s3-secret-access-key", type=str, default="")


    args = ap.parse_args()

    is_ddp, rank, world_size, local_rank = ddp_setup()
    is_main = (rank == 0)

    s3 = None
    if is_main and args.s3_bucket:
        s3 = make_s3_client(
            region=args.s3_region,
            endpoint_url=args.s3_endpoint_url,
            access_key_id=args.s3_access_key_id,
            secret_access_key=args.s3_secret_access_key,
        )


    if args.enable_tf32:
        print('Activated tf32')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if args.num_heads != 4:
        raise RuntimeError("This script assumes num_heads=4: bass, drums, music(inst+melody), vocals")

    device = torch.device(f"cuda:{local_rank}" if (is_ddp and torch.cuda.is_available()) else args.device)
    print("device:", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    items = scan_root_to_items(args.root) if args.root else load_manifest_csv(args.manifest)
    if args.limit_items and args.limit_items > 0:
        items = items[: int(args.limit_items)]
        print(f"[info] limit-items={len(items)}")

    book = RecipeBook.from_json_path(args.recipe_json)

    ds = FlexibleMixDataset(
        items,
        sr=args.sr,
        segment_sec=args.segment_sec,
        recipe_book=book,
        long_threshold_sec=args.long_threshold_sec,
        long_hop_sec=args.long_hop_sec,
        long_jitter_sec=args.long_jitter_sec,
        epoch_size=args.epoch_size,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    sampler = None
    if is_ddp:
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=(len(ds) >= args.batch),
        )

    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=(len(ds) >= args.batch),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda") and args.pin_memory,
        persistent_workers=(args.num_workers > 0) and args.persistent_workers,
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=g,
        collate_fn=collate,
        in_order=args.in_order,
    )
    print(f"dataset: {len(ds)} segments | batch={args.batch} | batches/epoch={len(dl)}")

    model = DeepVQEStemSeparator(n_fft=args.n_fft, num_heads=args.num_heads).to(device)

    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    stft = STFT(StftCfg(n_fft=args.n_fft, hop=args.hop, win=args.win)).to(device)

    # MR-STFT loss module
    mrstft = None
    if float(args.w_mrstft) > 0.0:
        n_ffts = _parse_int_csv(args.mr_nffts)
        hops = _parse_int_csv(args.mr_hops)
        wins = _parse_int_csv(args.mr_wins)
        mrstft = MRSTFTLoss(
            n_ffts=n_ffts,
            hops=hops,
            wins=wins,
            sc_weight=float(args.mr_sc_weight),
            logmag_weight=float(args.mr_logmag_weight),
            eps=float(args.mr_eps),
            chunk=int(args.mr_chunk),
        ).to(device)


    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    global_step = 0

    raw_model = model.module if isinstance(model, DDP) else model

    if args.resume:
        ckpt = load_ckpt(args.resume, device=device)

        sd = _normalize_checkpoint_state_dict(ckpt["model"])
        raw_model.load_state_dict(sd, strict=True)

        if (not args.reset_opt) and ("opt" in ckpt):
            try:
                opt.load_state_dict(ckpt["opt"])
            except Exception as e:
                print(f"[warn] failed to load optimizer state: {e}")

        if not args.reset_rng:
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
        ds.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)

        run = {"stem": 0.0, "mix": 0.0, "mr": 0.0, "total": 0.0}
        pbar = tqdm(dl, desc=f"Epoch {epoch}", dynamic_ncols=True) if is_main else dl

        for mix_in, tgt, present_mask in pbar:
            # mix_in: (B,2,T)
            # tgt:    (B,4,2,T)
            # present_mask: (B,4)

            mix_in = mix_in.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            present_mask = present_mask.to(device, non_blocking=True)

            B, C, T = mix_in.shape
            N = B * C  # как у тебя: канал считаем как batch

            def flat_bc(x: torch.Tensor) -> torch.Tensor:
                return x.reshape(N, T).float()

            mix_f = flat_bc(mix_in)  # (N,T)

            # targets per head -> (N,T) each
            bass_f = flat_bc(tgt[:, 0])  # (N,T)
            drums_f = flat_bc(tgt[:, 1])
            music_f = flat_bc(tgt[:, 2])
            vocals_f = flat_bc(tgt[:, 3])

            # mask expand to channels (N,4)
            pm_n = present_mask.repeat_interleave(C, dim=0)  # (N,4)

            with torch.no_grad():
                mix_ri = stft.stft_ri(mix_f)  # (N,F,Tf,2)
                tgt_ri = torch.stack(
                    [
                        stft.stft_ri(bass_f),
                        stft.stft_ri(drums_f),
                        stft.stft_ri(music_f),
                        stft.stft_ri(vocals_f),
                    ],
                    dim=1,
                )  # (N,4,F,Tf,2)

            pred = model(mix_ri)  # (N,4,F,Tf,2)

            # --- stem loss (L1 in RI) ---
            diff = (pred.float() - tgt_ri.float()).abs().mean(dim=(2, 3, 4))  # (N,4)

            # weights: head_w * (present + silence_weight * absent)
            sil_w = float(args.silence_weight)
            w = head_w[None, :] * (pm_n * 1.0 + (1.0 - pm_n) * sil_w)  # (N,4)

            # normalize per-sample to not depend on how many stems included
            loss_stem = ((diff * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-8)).mean()

            # --- mixture consistency: sum(pred) must reconstruct mix_in (which IS sum of active stems) ---
            mix_hat = pred.sum(dim=1)  # (N,F,Tf,2)
            loss_mix = l1_ri(mix_hat, mix_ri)

            loss = float(args.w_stem) * loss_stem + float(args.w_mix) * loss_mix

            # --- MR-STFT (time-domain) ---
            mr_loss = None
            if (mrstft is not None) and (int(args.mr_every) > 0) and ((global_step % int(args.mr_every)) == 0):
                S = pred.shape[1]  # 4
                pred_ri_flat = pred.reshape(N * S, pred.shape[2], pred.shape[3], 2)
                pred_wav = stft.istft_ri(pred_ri_flat, length=T)  # (N*S,T)

                tgt_wav = torch.stack([bass_f, drums_f, music_f, vocals_f], dim=1).reshape(N * S, T)  # (N*S,T)

                mr_vec = mrstft(pred_wav, tgt_wav)  # (N*S,)

                # weights for MR: same idea as above, but flattened (N*S,)
                w_mr = w.reshape(N * S).float()
                mr_loss = (mr_vec * w_mr).sum() / w_mr.sum().clamp_min(1e-8)

                loss = loss + float(args.w_mrstft) * mr_loss

            if mr_loss is not None:
                run["mr"] += float(mr_loss.detach().cpu())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            global_step += 1

            run["stem"] += float(loss_stem.detach().cpu())
            run["mix"] += float(loss_mix.detach().cpu())
            run["total"] += float(loss.detach().cpu())

            if is_main:
                denom = max(1, pbar.n + 1)
                pbar.set_postfix(
                    total=f"{run['total'] / denom:.6f}",
                    stem=f"{run['stem'] / denom:.6f}",
                    mix=f"{run['mix'] / denom:.6f}",
                    mr=f"{run['mr'] / denom:.6f}",
                )

        raw_model = model.module if isinstance(model, DDP) else model
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": raw_model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            **capture_rng(),
        }

        latest_path = None
        epoch_path = None
        if is_main:
            latest_path = save_dir / "phase_ab_4stem_latest.pt"
            atomic_save_ckpt(latest_path, ckpt)

            epoch_path = save_dir / f"phase_ab_4stem_e{epoch:03d}.pt"
            if epoch % int(args.save_every_epochs) == 0:
                atomic_save_ckpt(epoch_path, ckpt)

        if is_ddp:
            dist.barrier()

        # --- UPLOAD (только rank0) ---
        if is_main and s3 is not None:
            prefix = args.s3_prefix.strip("/")

            # грузим latest всегда
            key_latest = f"{prefix}/phase_ab_4stem_latest.pt"
            upload_to_s3(s3, local_path=latest_path, bucket=args.s3_bucket, key=key_latest)

            # грузим epoch-овый чекпоинт по расписанию (если не включен "только latest")
            if (not args.s3_upload_latest_only) and (epoch % int(args.save_every_epochs) == 0):
                key_epoch = f"{prefix}/phase_ab_4stem_e{epoch:03d}.pt"
                upload_to_s3(s3, local_path=epoch_path, bucket=args.s3_bucket, key=key_epoch)

        if is_ddp:
            dist.barrier()

    if is_ddp:
        dist.destroy_process_group()

    print("done.")


if __name__ == "__main__":
    main()
