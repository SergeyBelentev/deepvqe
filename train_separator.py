# train_separator.py
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import os
import datetime
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from deep_separator import StemSeparator, SeparatorConfig

from train_phase_a import (
    scan_root_to_items,
    load_manifest_csv,
    RecipeBook,
    FlexibleMixDataset,
    collate,
    STEM_ORDER,
)


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
        return autocast("cpu", enabled=False), None, False

    if amp in ("off", "none", "0"):
        return autocast("cuda", enabled=False), None, False
    if amp == "fp16":
        return autocast("cuda", dtype=torch.float16), torch.float16, True
    if amp == "bf16":
        return autocast("cuda", dtype=torch.bfloat16), torch.bfloat16, False
    raise ValueError(f"Unknown --amp={amp!r}. Use off|fp16|bf16.")


# -------------------------
# Multi-Resolution STFT loss (SC + LogMag)
# -------------------------

class MultiResolutionSTFTLoss(nn.Module):
    """
    MR-STFT loss = mean over resolutions of:
      SC = || |Sx|-|Sy| ||_F / (|| |Sy| ||_F + eps)
      LogMag = mean | log(|Sx|+eps) - log(|Sy|+eps) |
    Stereo is handled by flattening channel into batch.
    """
    def __init__(
        self,
        n_ffts=(1024, 2048, 4096),
        hops=(256, 512, 1024),
        win_lengths=None,
        center=True,
        normalized=False,
        eps=1e-8,
        w_sc=1.0,
        w_lm=1.0,
    ):
        super().__init__()
        assert len(n_ffts) == len(hops)
        self.n_ffts = tuple(map(int, n_ffts))
        self.hops = tuple(map(int, hops))
        self.win_lengths = tuple(map(int, (win_lengths or n_ffts)))
        self.center = bool(center)
        self.normalized = bool(normalized)
        self.eps = float(eps)
        self.w_sc = float(w_sc)
        self.w_lm = float(w_lm)

    def _stft_mag(self, x: torch.Tensor, n_fft: int, hop: int, win_length: int) -> torch.Tensor:
        # x: (B,2,T)
        B, C, T = x.shape
        win = torch.hann_window(win_length, periodic=True, device=x.device, dtype=torch.float32)
        x2 = x.reshape(B * C, T)
        S = torch.stft(
            x2,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_length,
            window=win,
            center=self.center,
            normalized=self.normalized,
            onesided=True,
            return_complex=True,
        )  # (B*C, F, TT)
        mag = S.abs().reshape(B, C, S.shape[-2], S.shape[-1])  # (B,C,F,TT)
        return mag

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x,y: (B,2,T)
        sc_all, lm_all = [], []
        for n_fft, hop, wl in zip(self.n_ffts, self.hops, self.win_lengths):
            mx = self._stft_mag(x, n_fft, hop, wl)  # (B,C,F,TT)
            my = self._stft_mag(y, n_fft, hop, wl)

            diff = mx - my  # (B,C,F,TT)

            # SC per (B,C)
            diff_fro = torch.linalg.vector_norm(diff.flatten(2), dim=2)  # (B,C)
            my_fro   = torch.linalg.vector_norm(my.flatten(2), dim=2).clamp_min(self.eps)  # (B,C)
            sc = (diff_fro / my_fro).mean()  # scalar

            # LogMag per (B,C)
            lmx = (mx + self.eps).log()
            lmy = (my + self.eps).log()
            lm = (lmx - lmy).abs().mean(dim=(2, 3)).mean()  # scalar

            sc_all.append(sc)
            lm_all.append(lm)

        sc_m = torch.stack(sc_all).mean()
        lm_m = torch.stack(lm_all).mean()
        total = self.w_sc * sc_m + self.w_lm * lm_m
        return total, {"mr_sc": sc_m.detach(), "mr_logmag": lm_m.detach()}


# -------------------------
# Loss pack
# -------------------------

def rms(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(x.pow(2).mean(dim=(1, 2)).clamp_min(eps))


def cosine_abs(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    av = a.reshape(a.shape[0], -1)
    bv = b.reshape(b.shape[0], -1)
    num = (av * bv).sum(dim=1)
    den = (av.norm(dim=1) * bv.norm(dim=1)).clamp_min(eps)
    return (num / den).abs()


@torch.no_grad()
def _mean_where(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mf = m.float()
    return (x * mf).sum() / (mf.sum() + eps)


class LossComputer(nn.Module):
    def __init__(
        self,
        sr: int,
        mr_w_sc: float = 1.0,
        mr_w_lm: float = 1.0,
        silence_rms_thr: float = 1e-3,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = float(eps)
        self.silence_rms_thr = float(silence_rms_thr)

        self.mr = MultiResolutionSTFTLoss(
            n_ffts=(1024, 2048, 4096),
            hops=(256, 512, 1024),
            win_lengths=(1024, 2048, 4096),
            center=True,
            normalized=False,
            eps=eps,
            w_sc=mr_w_sc,
            w_lm=mr_w_lm,
        )

    def forward(
        self,
        pred_stems: torch.Tensor,     # (B,4,2,T)
        tgt_stems: torch.Tensor,      # (B,4,2,T)
        present_mask: torch.Tensor,   # (B,4) float 0/1
        mix_target: torch.Tensor,     # (B,2,T)
        weights: Dict[str, float],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, H, C, T = pred_stems.shape
        assert H == 4 and C == 2

        mix_scale = mix_target.abs().mean(dim=(1, 2), keepdim=True).clamp_min(self.eps)  # (B,1,1)
        pm = present_mask.view(B, H, 1, 1)  # (B,4,1,1)

        l1_per = (pred_stems - tgt_stems).abs().mean(dim=(2, 3))  # (B,4)
        l1_per_norm = l1_per / mix_scale.squeeze(-1)  # (B,4)
        l1_head = (l1_per_norm * present_mask).sum() / (present_mask.sum() + self.eps)

        mix_pred = pred_stems.sum(dim=1)  # (B,2,T)
        l1_mix = (mix_pred - mix_target).abs().mean(dim=(1, 2)) / mix_scale.squeeze(-1).squeeze(-1)
        l1_mix = l1_mix.mean()
        mr_mix, mr_mix_stat = self.mr(mix_pred, mix_target)

        tgt_rms = torch.stack([rms(tgt_stems[:, i]) for i in range(4)], dim=1)  # (B,4)
        silence_mask = (present_mask <= 0.0) | (tgt_rms < self.silence_rms_thr)  # (B,4) bool
        pred_abs = pred_stems.abs().mean(dim=(2, 3))  # (B,4)
        silence_loss = _mean_where(pred_abs, silence_mask)

        mix_tgt_present = (tgt_stems * pm).sum(dim=1)  # (B,2,T)

        leak_terms = []
        for i in range(4):
            sel_i = present_mask[:, i] > 0.5
            if not bool(sel_i.any()):
                continue
            other = mix_tgt_present - tgt_stems[:, i] * pm[:, i]
            ok = sel_i & (rms(other) > self.silence_rms_thr) & (rms(tgt_stems[:, i]) > self.silence_rms_thr)
            if bool(ok.any()):
                leak_terms.append(cosine_abs(pred_stems[ok, i], other[ok]).mean())

        leak_loss = torch.stack(leak_terms).mean() if leak_terms else pred_stems.sum() * 0.0

        mr_head_vals = []
        mr_sc_vals = []
        mr_lm_vals = []

        mr_mask = (present_mask > 0.5) & (tgt_rms >= self.silence_rms_thr)  # (B,4) bool

        for i in range(4):
            sel = mr_mask[:, i]
            if not bool(sel.any()):
                continue
            li, stat = self.mr(pred_stems[sel, i], tgt_stems[sel, i])
            mr_head_vals.append(li)
            mr_sc_vals.append(stat["mr_sc"])
            mr_lm_vals.append(stat["mr_logmag"])

        if mr_head_vals:
            mr_head = torch.stack(mr_head_vals).mean()
            mr_sc = torch.stack(mr_sc_vals).mean()
            mr_lm = torch.stack(mr_lm_vals).mean()
        else:
            mr_head = pred_stems.sum() * 0.0
            mr_sc = pred_stems.sum() * 0.0
            mr_lm = pred_stems.sum() * 0.0

        total = (
            weights["w_l1_head"] * l1_head +
            weights["w_mr_head"] * mr_head +
            weights["w_l1_mix"] * l1_mix +
            weights["w_mr_mix"] * mr_mix +
            weights["w_silence"] * silence_loss +
            weights["w_leak"] * leak_loss
        )

        stats = {
            "loss_total": total.detach(),
            "l1_head": l1_head.detach(),
            "mr_head": mr_head.detach(),
            "mr_sc_head": mr_sc.detach(),
            "mr_lm_head": mr_lm.detach(),
            "l1_mix": l1_mix.detach(),
            "mr_mix": mr_mix.detach(),
            "mr_sc_mix": mr_mix_stat["mr_sc"].detach(),
            "mr_lm_mix": mr_mix_stat["mr_logmag"].detach(),
            "silence": silence_loss.detach(),
            "leak": leak_loss.detach(),
        }
        return total, stats


# -------------------------
# Checkpointing
# -------------------------

def save_ckpt(
    path: Path,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    sched: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    step: int,
    cfg: SeparatorConfig,
    extra: Optional[Dict[str, Any]] = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "epoch": int(epoch),
        "step": int(step),
        "cfg": asdict(cfg),
        "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
        "opt": opt.state_dict(),
        "sched": (sched.state_dict() if sched is not None else None),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "rng": {
            "py": random.getstate(),
            "np": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": (torch.cuda.random.get_rng_state_all() if torch.cuda.is_available() else None),
        },
        "extra": extra or {},
    }
    torch.save(obj, str(path))


def load_ckpt(
    path: Path,
    model: nn.Module,
    opt: Optional[torch.optim.Optimizer] = None,
    sched: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = "cpu",
) -> Tuple[int, int, Dict[str, Any]]:
    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)

    sd = ckpt["model"]
    (model.module if hasattr(model, "module") else model).load_state_dict(sd, strict=True)

    if opt is not None and ckpt.get("opt") is not None:
        opt.load_state_dict(ckpt["opt"])
    if sched is not None and ckpt.get("sched") is not None:
        sched.load_state_dict(ckpt["sched"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    rng = ckpt.get("rng", None)
    if rng is not None:
        try:
            random.setstate(rng["py"])
            np.random.set_state(rng["np"])
            torch.random.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.random.set_rng_state_all(rng["cuda"])
        except Exception:
            pass

    epoch = int(ckpt.get("epoch", 1))
    step = int(ckpt.get("step", 0))
    extra = ckpt.get("extra", {}) or {}
    return epoch, step, extra


# -------------------------
# Scheduler
# -------------------------

def make_scheduler(opt, total_steps: int, warmup_steps: int):
    def lr_lambda(s: int):
        if s < warmup_steps:
            return float(s) / float(max(1, warmup_steps))
        t = (s - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)


# -------------------------
# Train/Val split by songs (items)
# -------------------------

def split_items_train_val(
    items: List[Any],
    *,
    val_frac: float,
    val_count: int,
    seed: int,
) -> Tuple[List[Any], List[Any]]:
    n = len(items)
    if n == 0:
        return [], []
    if val_count > 0:
        nv = int(val_count)
    else:
        nv = int(round(float(val_frac) * n))

    # guarantee: val uses whole songs, and train has at least 1 song if possible
    nv = max(0, nv)
    if n >= 2:
        nv = min(nv, n - 1)
    else:
        nv = min(nv, n)

    if nv == 0:
        return list(items), []

    idx = list(range(n))
    rng = random.Random(int(seed))
    rng.shuffle(idx)
    val_idx = set(idx[:nv])

    train_items = [items[i] for i in range(n) if i not in val_idx]
    val_items = [items[i] for i in range(n) if i in val_idx]
    return train_items, val_items


# -------------------------
# S3 helpers (optional)
# -------------------------

def build_s3_client(
    *,
    region: str = "",
    endpoint_url: str = "",
    access_key_id: str = "",
    secret_access_key: str = "",
    session_token: str = "",
    profile: str = "",
):
    try:
        import boto3
        from botocore.config import Config as BotocoreConfig
    except Exception as e:
        raise RuntimeError(f"S3 requested but boto3/botocore not available: {e}")

    cfg = BotocoreConfig(
        retries={"max_attempts": 10, "mode": "adaptive"},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    endpoint = endpoint_url or None

    if profile:
        sess = boto3.session.Session(profile_name=profile, region_name=(region or None))
        return sess.client("s3", endpoint_url=endpoint, config=cfg)

    if access_key_id and secret_access_key:
        sess = boto3.session.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=(session_token or None),
            region_name=(region or None),
        )
        return sess.client("s3", endpoint_url=endpoint, config=cfg)

    sess = boto3.session.Session(region_name=(region or None))
    return sess.client("s3", endpoint_url=endpoint, config=cfg)


def s3_upload_file(s3, local_path: Path, bucket: str, key: str) -> None:
    # не 1-в-1 с твоим кодом, но логика та же: multipart + concurrency
    from boto3.s3.transfer import TransferConfig

    local_path = Path(local_path)
    cfg = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )
    s3.upload_file(str(local_path), bucket, key, Config=cfg)


def join_s3_key(prefix: str, name: str) -> str:
    p = (prefix or "").strip("/")
    return f"{p}/{name}" if p else name


# -------------------------
# Validation runner
# -------------------------

def _safe_item_name(flags: Any, fallback: str) -> str:
    # batch[6] is "flags" in твоём collate; структура может отличаться.
    # Пытаемся вытащить что-то стабильное для имени файла.
    try:
        if isinstance(flags, dict):
            for k in ("uid", "id", "key", "name", "kind", "folder"):
                if k in flags and flags[k]:
                    return str(flags[k])
        if isinstance(flags, (list, tuple)) and len(flags) > 0:
            # часто flags = list[dict] по batch
            f0 = flags[0]
            if isinstance(f0, dict):
                for k in ("uid", "id", "key", "name", "kind", "folder"):
                    if k in f0 and f0[k]:
                        return str(f0[k])
    except Exception:
        pass
    return fallback


def _save_audio_bundle(
    out_dir: Path,
    *,
    sr: int,
    mix: torch.Tensor,        # (2,T) cpu
    mix_target: torch.Tensor, # (2,T) cpu
    pred: torch.Tensor,       # (4,2,T) cpu
    tgt: torch.Tensor,        # (4,2,T) cpu
    pm: torch.Tensor,         # (4,) cpu
):
    try:
        import soundfile as sf
    except Exception as e:
        raise RuntimeError(f"Saving validation audio requested but soundfile is missing: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)

    def _w(name: str, x: torch.Tensor):
        x_np = x.transpose(0, 1).contiguous().numpy()  # (T,2)
        sf.write(str(out_dir / name), x_np, sr, subtype="FLOAT")

    _w("mix_in.wav", mix)
    _w("mix_target.wav", mix_target)

    for i, stem in enumerate(STEM_ORDER):
        present = int(pm[i].item() > 0.5)
        _w(f"pred_{stem}_p{present}.wav", pred[i])
        _w(f"gt_{stem}_p{present}.wav", tgt[i])


@torch.no_grad()
def run_validation(
    *,
    epoch: int,
    model: nn.Module,
    loss_comp: LossComputer,
    dl_val: DataLoader,
    device: torch.device,
    autocast_ctx,
    weights: Dict[str, float],
    max_batches: int,
    save_dir: Optional[Path],
    save_n_audio: int,
    save_audio: bool,
    sr: int,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()

    sums: Dict[str, float] = {}
    n = 0

    # audio save folder per epoch
    epoch_dir = None
    if save_dir is not None:
        epoch_dir = Path(save_dir) / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

    for bidx, batch in enumerate(tqdm(dl_val, desc=f"val epoch {epoch}", dynamic_ncols=True)):
        mix = batch[0].to(device, non_blocking=True)         # (B,2,T)
        tgt = batch[3].to(device, non_blocking=True)         # (B,4,2,T)
        pm  = batch[4].to(device, non_blocking=True)         # (B,4)
        mix_target = batch[5].to(device, non_blocking=True)  # (B,2,T)
        flags = batch[6] if len(batch) > 6 else None

        with autocast_ctx:
            out = model(mix, return_debug=False)  # dict head -> (B,2,T)
            pred = torch.stack([out[s] for s in STEM_ORDER], dim=1)  # (B,4,2,T)
            loss, stats = loss_comp(
                pred_stems=pred,
                tgt_stems=tgt,
                present_mask=pm,
                mix_target=mix_target,
                weights=weights,
            )

        # accumulate means
        for k, v in stats.items():
            sums[k] = sums.get(k, 0.0) + float(v.item())
        n += 1

        # optional: save first N audio examples
        if save_audio and epoch_dir is not None and bidx < int(save_n_audio):
            name = _safe_item_name(flags, fallback=f"sample_{bidx:04d}")
            sample_dir = epoch_dir / name
            # only first sample in batch
            _save_audio_bundle(
                sample_dir,
                sr=sr,
                mix=mix[0].detach().cpu(),
                mix_target=mix_target[0].detach().cpu(),
                pred=pred[0].detach().cpu(),
                tgt=tgt[0].detach().cpu(),
                pm=pm[0].detach().cpu(),
            )

        if int(max_batches) > 0 and n >= int(max_batches):
            break

    means = {k: (v / max(1, n)) for k, v in sums.items()}

    if was_training:
        model.train()
    return means

# DDP Helpers
def ddp_enabled_from_env() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def ddp_setup(backend: str = "nccl", timeout_sec: int = 1800) -> Dict[str, int]:
    """
    Инициализация DDP через env:// (torchrun задаёт RANK/WORLD_SIZE/LOCAL_RANK).
    """
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=datetime.timedelta(seconds=int(timeout_sec)),
        )

    return {"world_size": world_size, "rank": rank, "local_rank": local_rank}

def ddp_cleanup() -> None:
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank: int) -> bool:
    return rank == 0

def ddp_barrier() -> None:
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()

def ddp_broadcast_object(obj, src: int = 0):
    """
    Удобно для broadcast списков/словарей (python objects) через dist.broadcast_object_list.
    """
    import torch.distributed as dist
    if not dist.is_initialized():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]

def ddp_reduce_stats(stats: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Усредняет скаляры stats по всем ранкам (1 all_reduce на вектор).
    Вызывай НЕ каждый итерационный шаг, а только когда реально логируешь.
    """
    import torch.distributed as dist
    if not dist.is_initialized():
        return {k: float(v.item()) for k, v in stats.items()}

    keys = sorted(stats.keys())
    vec = torch.stack([stats[k].detach().float() for k in keys], dim=0)
    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    vec /= dist.get_world_size()

    return {k: float(vec[i].item()) for i, k in enumerate(keys)}

def item_uid_for_split(it: Any) -> str:
    """
    Стабильный uid для split'а по песням.
    """
    try:
        if isinstance(it, dict):
            for k in ("uid", "id", "key", "folder", "name", "full"):
                v = it.get(k, None)
                if v:
                    return str(v)
        # на случай объектов
        for k in ("uid", "id", "key", "folder", "name", "full"):
            if hasattr(it, k):
                v = getattr(it, k)
                if v:
                    return str(v)
    except Exception:
        pass
    return repr(it)

# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=str, help="dataset root to scan (folders with full.wav, stems)")
    src.add_argument("--manifest", type=str, help="manifest.csv with columns kind/full/bass/drums/instruments/vocals/melody")

    p.add_argument("--recipes", type=str, required=True, help="RecipeBook JSON path")
    p.add_argument("--out", type=str, required=True, help="output dir for checkpoints")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--epoch-size", type=int, default=200_000)
    p.add_argument("--segment-sec", type=float, default=8.0)

    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--warmup-steps", type=int, default=2_000)

    p.add_argument("--amp", type=str, default="bf16", choices=["off", "fp16", "bf16"])
    p.add_argument("--tf32", type=int, default=1, help="1 to enable TF32 on matmul/conv")

    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--clip-grad", type=float, default=1.0)

    p.add_argument("--save-every-step", type=int, default=200_000)
    p.add_argument("--save-every-epoch", type=int, default=1)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--resume", type=str, default="")

    # loss weights
    p.add_argument("--w-l1-head", type=float, default=1.0)
    p.add_argument("--w-mr-head", type=float, default=1.0)
    p.add_argument("--w-l1-mix", type=float, default=0.5)
    p.add_argument("--w-mr-mix", type=float, default=0.5)
    p.add_argument("--w-silence", type=float, default=0.2)
    p.add_argument("--w-leak", type=float, default=0.2)
    p.add_argument("--silence-rms", type=float, default=1e-3)

    # --- validation split by songs (items)
    p.add_argument("--val-frac", type=float, default=0.02, help="fraction of songs(items) held out for validation")
    p.add_argument("--val-count", type=int, default=0, help="override val size by number of songs (0 = use val-frac)")
    p.add_argument("--val-seed", type=int, default=12345, help="seed for train/val song split")

    # --- validation runner controls
    p.add_argument("--val-epoch-size", type=int, default=1, help="validation dataset epoch_size (segments)")
    p.add_argument("--val-batch", type=int, default=1)
    p.add_argument("--val-num-workers", type=int, default=0)
    p.add_argument("--val-max-batches", type=int, default=200, help="max val batches per epoch (0 = full loader)")
    p.add_argument("--val-save-dir", type=str, default="val", help="subdir under --out to store val artifacts")
    p.add_argument("--val-save-audio", type=int, default=1, help="1 to save example wavs for first N batches")
    p.add_argument("--val-save-n-audio", type=int, default=4, help="how many validation batches to dump audio for")

    # --- S3 for checkpoints / validation artifacts
    p.add_argument("--s3-bucket", type=str, default="")
    p.add_argument("--s3-prefix", type=str, default="")
    p.add_argument("--s3-region", type=str, default="")
    p.add_argument("--s3-endpoint-url", type=str, default="")
    p.add_argument("--s3-access-key-id", type=str, default="")
    p.add_argument("--s3-secret-access-key", type=str, default="")
    p.add_argument("--s3-session-token", type=str, default="")
    p.add_argument("--s3-profile", type=str, default="")

    p.add_argument("--upload-ckpt-s3", type=int, default=0, help="1 to upload checkpoints to S3")
    p.add_argument("--upload-val-s3", type=int, default=0, help="1 to upload validation artifacts to S3 (tar.gz)")

    # --- DDP
    p.add_argument("--ddp", type=int, default=0, help="1 to force DDP (requires torchrun env vars)")
    p.add_argument("--ddp-backend", type=str, default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--ddp-timeout-sec", type=int, default=1800)
    p.add_argument("--ddp-broadcast-buffers", type=int, default=0)
    p.add_argument("--ddp-gradient-as-bucket-view", type=int, default=1)

    return p.parse_args()


# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- DDP init (torchrun sets env)
    force_ddp = bool(int(args.ddp))
    env_ddp = ddp_enabled_from_env()
    use_ddp = env_ddp or force_ddp

    if force_ddp and not env_ddp:
        raise RuntimeError("DDP forced (--ddp=1) but WORLD_SIZE=1. Run with torchrun.")

    ddp_info = {"world_size": 1, "rank": 0, "local_rank": 0}
    if use_ddp:
        ddp_info = ddp_setup(backend=args.ddp_backend, timeout_sec=int(args.ddp_timeout_sec))

    rank = int(ddp_info["rank"])
    world_size = int(ddp_info["world_size"])
    local_rank = int(ddp_info["local_rank"])
    main_proc = is_main_process(rank)

    # --- device: one process -> one GPU (LOCAL_RANK)
    if torch.cuda.is_available():
        if use_ddp:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    set_tf32(bool(args.tf32))

    base_seed = 1234
    seed = base_seed + (rank if use_ddp else 0)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # dataset items (each item ~= song)
    if args.root:
        items = scan_root_to_items(args.root)
    else:
        items = load_manifest_csv(args.manifest)

    # split by full songs
    if use_ddp:
        if main_proc:
            train_items0, val_items0 = split_items_train_val(
                items,
                val_frac=float(args.val_frac),
                val_count=int(args.val_count),
                seed=int(args.val_seed),
            )
            val_ids = [item_uid_for_split(x) for x in val_items0]
        else:
            val_ids = None

        val_ids = ddp_broadcast_object(val_ids, src=0)
        val_id_set = set(val_ids or [])

        train_items = [it for it in items if item_uid_for_split(it) not in val_id_set]
        val_items = [it for it in items if item_uid_for_split(it) in val_id_set]
    else:
        train_items, val_items = split_items_train_val(
            items,
            val_frac=float(args.val_frac),
            val_count=int(args.val_count),
            seed=int(args.val_seed),
        )

    if len(train_items) == 0:
        raise RuntimeError("Train set is empty after split. Reduce --val-frac/--val-count or check dataset.")
    if len(val_items) == 0 and main_proc:
        print("[warn] validation set is empty (val_frac/val_count resulted in 0). Validation will be skipped.")

    book = RecipeBook.from_json_path(args.recipes)
    cfg = SeparatorConfig()

    # train dataset
    ds = FlexibleMixDataset(
        train_items,
        sr=cfg.sample_rate,
        segment_sec=float(args.segment_sec),
        recipe_book=book,
        epoch_size=int(args.epoch_size),
    )

    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=base_seed,
            drop_last=True,
        )

    dl = DataLoader(
        ds,
        batch_size=int(args.batch),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=(int(args.num_workers) > 0),
        collate_fn=collate,
        drop_last=True,
        prefetch_factor=2 if int(args.num_workers) > 0 else None,
    )

    # val dataset (segments sampled ONLY from held-out songs)
    dl_val = None
    if main_proc and len(val_items) > 0:
        ds_val = FlexibleMixDataset(
            val_items,
            sr=cfg.sample_rate,
            segment_sec=float(args.segment_sec),
            recipe_book=book,
            epoch_size=int(args.val_epoch_size),
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=int(args.val_batch),
            shuffle=False,
            num_workers=int(args.val_num_workers),
            pin_memory=True,
            persistent_workers=(int(args.val_num_workers) > 0),
            collate_fn=collate,
            drop_last=False,
            prefetch_factor=2 if int(args.val_num_workers) > 0 else None,
        )

    # model
    model = StemSeparator(cfg).to(device)
    model.train()

    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=bool(int(args.ddp_broadcast_buffers)),
            gradient_as_bucket_view=bool(int(args.ddp_gradient_as_bucket_view)),
        )

    # optimizer (fused if possible)
    opt = None
    try:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            betas=(0.9, 0.95),
            weight_decay=float(args.wd),
            fused=(device.type == "cuda"),
        )
    except TypeError:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            betas=(0.9, 0.95),
            weight_decay=float(args.wd),
        )
    except Exception:
        # fallback hard
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            betas=(0.9, 0.95),
            weight_decay=float(args.wd),
        )

    steps_per_epoch = max(1, (len(dl) // max(1, int(args.grad_accum))))
    total_steps = int(args.epochs) * steps_per_epoch
    sched = make_scheduler(opt, total_steps=total_steps, warmup_steps=int(args.warmup_steps))

    autocast_ctx, amp_dtype, use_scaler = make_autocast(args.amp, device)
    scaler = GradScaler("cuda", enabled=use_scaler)

    loss_comp = LossComputer(
        sr=cfg.sample_rate,
        mr_w_sc=1.0,
        mr_w_lm=1.0,
        silence_rms_thr=float(args.silence_rms),
    ).to(device)

    weights = {
        "w_l1_head": float(args.w_l1_head),
        "w_mr_head": float(args.w_mr_head),
        "w_l1_mix": float(args.w_l1_mix),
        "w_mr_mix": float(args.w_mr_mix),
        "w_silence": float(args.w_silence),
        "w_leak": float(args.w_leak),
    }

    # S3 init (optional)
    s3 = None
    s3_bucket = (args.s3_bucket or "").strip()
    s3_prefix = (args.s3_prefix or "").strip()
    upload_ckpt_s3 = bool(int(args.upload_ckpt_s3))
    upload_val_s3 = bool(int(args.upload_val_s3))

    if (upload_ckpt_s3 or upload_val_s3) and not s3_bucket:
        raise RuntimeError("S3 upload enabled but --s3-bucket is empty.")

    if main_proc and (upload_ckpt_s3 or upload_val_s3) and s3_bucket:
        s3 = build_s3_client(
            region=args.s3_region,
            endpoint_url=args.s3_endpoint_url,
            access_key_id=args.s3_access_key_id,
            secret_access_key=args.s3_secret_access_key,
            session_token=args.s3_session_token,
            profile=args.s3_profile,
        )

    def maybe_upload(local_path: Path, key_name: str) -> None:
        if not main_proc:
            return
        if s3 is None or not s3_bucket:
            return
        key = join_s3_key(s3_prefix, key_name)
        s3_upload_file(s3, local_path, bucket=s3_bucket, key=key)

    # resume
    start_epoch = 1
    global_step = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_file():
            start_epoch, global_step, extra = load_ckpt(
                ckpt_path, model, opt=opt, sched=sched, scaler=scaler, map_location="cpu"
            )
            print(f"[resume] from {ckpt_path} epoch={start_epoch} step={global_step} extra={extra}")
        else:
            raise FileNotFoundError(f"--resume not found: {ckpt_path}")

    # train loop
    opt.zero_grad(set_to_none=True)
    t0 = time.time()

    val_root = out_dir / str(args.val_save_dir)

    for epoch in range(start_epoch, int(args.epochs) + 1):
        ds.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if dl_val is not None:
            # чтобы валидация тоже была воспроизводимой по эпохам
            try:
                dl_val.dataset.set_epoch(epoch)  # type: ignore[attr-defined]
            except Exception:
                pass

        pbar = tqdm(dl, desc=f"epoch {epoch}/{args.epochs}", dynamic_ncols=True) if main_proc else dl
        ema: Dict[str, float] = {}

        accum = int(args.grad_accum)
        for it, batch in enumerate(pbar):
            mix = batch[0].to(device, non_blocking=True)         # (B,2,T)
            tgt = batch[3].to(device, non_blocking=True)         # (B,4,2,T)
            pm  = batch[4].to(device, non_blocking=True)         # (B,4)
            mix_target = batch[5].to(device, non_blocking=True)  # (B,2,T)

            with autocast_ctx:
                out = model(mix, return_debug=False)
                pred = torch.stack([out[s] for s in STEM_ORDER], dim=1)  # (B,4,2,T)

            with torch.autocast(device_type="cuda", enabled=False):
                loss, stats = loss_comp(
                    pred_stems=pred.float(),
                    tgt_stems=tgt.float(),
                    present_mask=pm.float(),
                    mix_target=mix_target.float(),
                    weights=weights,
                )
                loss_scaled = loss / float(accum)

            do_step = ((it + 1) % accum == 0)
            sync_ctx = nullcontext()
            if use_ddp and not do_step:
                sync_ctx = model.no_sync()  # type: ignore[union-attr]
            with sync_ctx:
                if use_scaler:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

            if do_step:
                if use_scaler:
                    scaler.unscale_(opt)
                if float(args.clip_grad) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad))

                if use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

                if main_proc and (global_step % int(args.log_every)) == 0:
                    stats_f = ddp_reduce_stats(stats) if use_ddp else {k: float(v.item()) for k, v in stats.items()}

                    for k, x in stats_f.items():
                        ema[k] = x if k not in ema else (0.9 * ema[k] + 0.1 * x)

                    lr = opt.param_groups[0]["lr"]
                    pbar.set_postfix({
                        "lr": f"{lr:.2e}",
                        "loss": f"{ema.get('loss_total', stats_f['loss_total']):.4f}",
                        "l1h": f"{ema.get('l1_head', stats_f['l1_head']):.3f}",
                        "mrh": f"{ema.get('mr_head', stats_f['mr_head']):.3f}",
                        "mix": f"{ema.get('l1_mix', stats_f['l1_mix']):.3f}",
                        "sil": f"{ema.get('silence', stats_f['silence']):.3f}",
                        "leak": f"{ema.get('leak', stats_f['leak']):.3f}",
                    })

                # step ckpt
                if main_proc and int(args.save_every_step) > 0 and (global_step % int(args.save_every_step)) == 0:
                    ckpt_step = out_dir / f"ckpt_step_{global_step:08d}.pt"
                    save_ckpt(
                        ckpt_step,
                        model=model,
                        opt=opt,
                        sched=sched,
                        scaler=scaler if use_scaler else None,
                        epoch=epoch,
                        step=global_step,
                        cfg=cfg,
                        extra={"segment_sec": float(args.segment_sec), "amp": args.amp, "tf32": int(args.tf32)},
                    )
                    save_ckpt(
                        out_dir / "ckpt_last.pt",
                        model=model,
                        opt=opt,
                        sched=sched,
                        scaler=scaler if use_scaler else None,
                        epoch=epoch,
                        step=global_step,
                        cfg=cfg,
                        extra={"segment_sec": float(args.segment_sec), "amp": args.amp, "tf32": int(args.tf32)},
                    )

                    if upload_ckpt_s3:
                        maybe_upload(ckpt_step, key_name=f"ckpt/ckpt_step_{global_step:08d}.pt")
                        maybe_upload(out_dir / "ckpt_last.pt", key_name="ckpt/ckpt_last.pt")

        # end epoch: save epoch ckpt (by epoch counter, not by global_step)
        if main_proc and int(args.save_every_epoch) > 0 and (epoch % int(args.save_every_epoch)) == 0:
            ckpt_epoch = out_dir / f"ckpt_epoch_{epoch:04d}.pt"
            save_ckpt(
                ckpt_epoch,
                model=model,
                opt=opt,
                sched=sched,
                scaler=scaler if use_scaler else None,
                epoch=epoch,
                step=global_step,
                cfg=cfg,
                extra={"segment_sec": float(args.segment_sec), "amp": args.amp, "tf32": int(args.tf32)},
            )
            if upload_ckpt_s3:
                maybe_upload(ckpt_epoch, key_name=f"ckpt/ckpt_epoch_{epoch:04d}.pt")

        if main_proc:
            # always update last
            save_ckpt(
                out_dir / "ckpt_last.pt",
                model=model,
                opt=opt,
                sched=sched,
                scaler=scaler if use_scaler else None,
                epoch=epoch,
                step=global_step,
                cfg=cfg,
                extra={"segment_sec": float(args.segment_sec), "amp": args.amp, "tf32": int(args.tf32)},
            )
            if upload_ckpt_s3:
                maybe_upload(out_dir / "ckpt_last.pt", key_name="ckpt/ckpt_last.pt")

        # --- validation after epoch
        if use_ddp:
            ddp_barrier()

        if main_proc and dl_val is not None:
            save_audio = bool(int(args.val_save_audio))
            save_n_audio = int(args.val_save_n_audio)

            val_means = run_validation(
                epoch=epoch,
                model=model,
                loss_comp=loss_comp,
                dl_val=dl_val,
                device=device,
                autocast_ctx=autocast_ctx,
                weights=weights,
                max_batches=int(args.val_max_batches),
                save_dir=val_root,
                save_n_audio=save_n_audio,
                save_audio=save_audio,
                sr=int(cfg.sample_rate),
            )

            # write metrics.json
            epoch_dir = val_root / f"epoch_{epoch:04d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = epoch_dir / "metrics.json"
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        "val_batches": int(min(int(args.val_max_batches), len(dl_val)) if int(args.val_max_batches) > 0 else len(dl_val)),
                        "metrics": val_means,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            # upload validation artifacts (tar.gz) if requested
            if upload_val_s3:
                base_name = str(epoch_dir)
                archive_path = Path(shutil.make_archive(base_name, "gztar", root_dir=str(epoch_dir)))
                maybe_upload(archive_path, key_name=f"val/epoch_{epoch:04d}.tar.gz")

        if use_ddp:
            ddp_barrier()  # rank0 закончил валидацию — отпускаем остальных


    dt = time.time() - t0
    print(f"[done] time={dt/3600:.2f}h, steps={global_step}, out={out_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        ddp_cleanup()
