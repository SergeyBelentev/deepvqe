# train_separator.py
# Train script for StemSeparator (stem_separator.py)
# - AMP: fp16/bf16/off
# - TF32 on/off
# - resume checkpoints
# - tqdm logging
# - segment-sec dataset control (default 8s)
#
# Usage examples:
#   python train_separator.py --root /path/to/ds --recipes recipes.json --out runs/exp1
#   python train_separator.py --manifest manifest.csv --recipes recipes.json --out runs/exp1 --amp bf16 --tf32 1
#   python train_separator.py --root /path/to/ds --recipes recipes.json --out runs/exp1 --resume runs/exp1/ckpt_last.pt

from __future__ import annotations

import argparse
import math
import time
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
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
    # affects matmul/conv on Ampere+; attention SDPA may also benefit indirectly
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    # optional: matmul precision hint (PyTorch 2.x)
    try:
        torch.set_float32_matmul_precision("high" if enabled else "highest")
    except Exception:
        pass


def make_autocast(amp: str, device: torch.device):
    amp = amp.lower().strip()
    if device.type != "cuda":
        return autocast("cpu", enabled=False), None, False  # на CPU amp не нужен

    if amp in ("off", "none", "0"):
        return autocast("cuda", enabled=False), None, False
    if amp == "fp16":
        return autocast("cuda", dtype=torch.float16), torch.float16, True
    if amp == "bf16":
        return autocast("cuda", dtype=torch.bfloat16), torch.bfloat16, False
    raise ValueError(...)


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
        self.n_ffts = tuple(int(x) for x in n_ffts)
        self.hops = tuple(int(x) for x in hops)
        self.win_lengths = tuple(int(x) for x in (win_lengths or n_ffts))
        self.center = bool(center)
        self.normalized = bool(normalized)
        self.eps = float(eps)
        self.w_sc = float(w_sc)
        self.w_lm = float(w_lm)

        self.register_buffer("_dummy", torch.tensor(0.0), persistent=False)

    def _stft_mag(self, x: torch.Tensor, n_fft: int, hop: int, win_length: int) -> torch.Tensor:
        # x: (B,2,T) float
        B, C, T = x.shape
        dev = x.device
        win = torch.hann_window(win_length, periodic=True, device=dev, dtype=torch.float32)
        x2 = x.reshape(B * C, T)  # (B*C, T)
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
        )  # (B*C, F, TT) complex64/complex32
        mag = S.abs()  # (B*C, F, TT)
        return mag

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x,y: (B,2,T)
        sc_all = []
        lm_all = []
        for n_fft, hop, wl in zip(self.n_ffts, self.hops, self.win_lengths):
            mx = self._stft_mag(x, n_fft, hop, wl)
            my = self._stft_mag(y, n_fft, hop, wl)

            diff = (mx - my)
            sc = diff.norm(p="fro") / (my.norm(p="fro") + self.eps)

            lmx = (mx + self.eps).log()
            lmy = (my + self.eps).log()
            lm = (lmx - lmy).abs().mean()

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
    # x: (B,2,T) -> (B,)
    return torch.sqrt(x.pow(2).mean(dim=(1, 2)).clamp_min(eps))


def cosine_abs(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # a,b: (B,2,T) -> (B,)
    av = a.reshape(a.shape[0], -1)
    bv = b.reshape(b.shape[0], -1)
    num = (av * bv).sum(dim=1)
    den = (av.norm(dim=1) * bv.norm(dim=1)).clamp_min(eps)
    return (num / den).abs()


@torch.no_grad()
def _mean_where(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: (B,4) or (B,) ; m same shape bool/float
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

        # normalize by mix amplitude (stabilizes scale; matches "L1 on stem / mix")
        mix_scale = mix_target.abs().mean(dim=(1, 2), keepdim=True).clamp_min(self.eps)  # (B,1,1)

        # reconstruction per head only when present (unconditional partial mixes)
        pm = present_mask.view(B, H, 1, 1)  # (B,4,1,1)
        l1_per = (pred_stems - tgt_stems).abs().mean(dim=(2, 3))  # (B,4)
        l1_per_norm = l1_per / mix_scale.squeeze(-1)  # (B,4)
        l1_head = (l1_per_norm * present_mask).sum() / (present_mask.sum() + self.eps)

        # mix consistency: sum heads should match mix_target (stem_sum)
        mix_pred = pred_stems.sum(dim=1)  # (B,2,T)
        l1_mix = (mix_pred - mix_target).abs().mean(dim=(1, 2)) / mix_scale.squeeze(-1).squeeze(-1)
        l1_mix = l1_mix.mean()
        mr_mix, mr_mix_stat = self.mr(mix_pred, mix_target)

        # silence loss:
        #   1) stems that are absent in recipe (present_mask==0)
        #   2) or stems that are actually quiet by RMS in target
        tgt_rms = torch.stack([rms(tgt_stems[:, i]) for i in range(4)], dim=1)  # (B,4)
        silence_mask = (present_mask <= 0.0) | (tgt_rms < self.silence_rms_thr)  # (B,4) bool
        # penalize absolute energy of prediction for silence stems
        pred_abs = pred_stems.abs().mean(dim=(2, 3))  # (B,4)
        silence_loss = _mean_where(pred_abs, silence_mask)

        # leak loss (simple but effective):
        # minimize similarity of pred_i to sum(target_other)
        leak_terms = []
        for i in range(4):
            other = tgt_stems.sum(dim=1) - tgt_stems[:, i]  # (B,2,T)
            sim = cosine_abs(pred_stems[:, i], other)       # (B,)
            # ignore cases where other is ~0
            other_ok = (rms(other) > self.silence_rms_thr)
            if other_ok.any():
                leak_terms.append(sim[other_ok].mean())
        leak_loss = torch.stack(leak_terms).mean() if leak_terms else pred_stems.sum() * 0.0

        # MR per head, only when present AND target is not silent
        mr_head_vals = []
        mr_sc_vals = []
        mr_lm_vals = []

        mr_mask = (present_mask > 0.5) & (tgt_rms >= self.silence_rms_thr)  # (B,4) bool

        for i in range(4):
            sel = mr_mask[:, i]
            if not bool(sel.any()):
                continue
            li, stat = self.mr(pred_stems[sel, i], tgt_stems[sel, i])  # (B',2,T)
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

        # total
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
# Schedulers
# -------------------------

def make_scheduler(opt, total_steps: int, warmup_steps: int):
    # cosine with linear warmup
    def lr_lambda(s: int):
        if s < warmup_steps:
            return float(s) / float(max(1, warmup_steps))
        t = (s - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)


# -------------------------
# Main
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

    p.add_argument("--save-every", type=int, default=1)
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

    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_tf32(bool(args.tf32))

    # reproducibility baseline (dataset also seeds per-worker)
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # dataset
    if args.root:
        items = scan_root_to_items(args.root)
    else:
        items = load_manifest_csv(args.manifest)

    book = RecipeBook.from_json_path(args.recipes)

    cfg = SeparatorConfig()
    ds = FlexibleMixDataset(
        items,
        sr=cfg.sample_rate,
        segment_sec=float(args.segment_sec),
        recipe_book=book,
        epoch_size=int(args.epoch_size),
    )

    dl = DataLoader(
        ds,
        batch_size=int(args.batch),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate,
        drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # model
    model = StemSeparator(cfg).to(device)
    model.train()

    # optimizer
    fused_ok = (device.type == "cuda") and hasattr(torch.optim, "AdamW")
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(0.9, 0.95),
        weight_decay=float(args.wd),
        fused=True if fused_ok else False,  # if not supported, PyTorch will error; so keep guarded
    )

    # total steps
    steps_per_epoch = len(dl) // max(1, int(args.grad_accum))
    total_steps = int(args.epochs) * max(1, steps_per_epoch)
    sched = make_scheduler(opt, total_steps=total_steps, warmup_steps=int(args.warmup_steps))

    # amp
    autocast_ctx, amp_dtype, use_scaler = make_autocast(args.amp, device)
    scaler = GradScaler("cuda", enabled=use_scaler)

    # losses
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

    # training loop
    opt.zero_grad(set_to_none=True)
    t0 = time.time()

    for epoch in range(start_epoch, int(args.epochs) + 1):
        ds.set_epoch(epoch)
        pbar = tqdm(dl, desc=f"epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        # simple EMA for nicer logs
        ema: Dict[str, float] = {}

        accum = int(args.grad_accum)
        for it, batch in enumerate(pbar):
            # batch:
            # mix, ref, ref_tgt, tgt, pm, mix_target, flags
            mix = batch[0].to(device, non_blocking=True)         # (B,2,T)
            tgt = batch[3].to(device, non_blocking=True)         # (B,4,2,T)
            pm  = batch[4].to(device, non_blocking=True)         # (B,4)
            mix_target = batch[5].to(device, non_blocking=True)  # (B,2,T)

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
                loss_scaled = loss / float(accum)

            if use_scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            do_step = ((it + 1) % accum == 0)
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

                # logging
                if (global_step % int(args.log_every)) == 0:
                    for k, v in stats.items():
                        x = float(v.item())
                        ema[k] = x if k not in ema else (0.9 * ema[k] + 0.1 * x)

                    lr = opt.param_groups[0]["lr"]
                    pbar.set_postfix({
                        "lr": f"{lr:.2e}",
                        "loss": f"{ema.get('loss_total', float(stats['loss_total'])):.4f}",
                        "l1h": f"{ema.get('l1_head', float(stats['l1_head'])):.3f}",
                        "mrh": f"{ema.get('mr_head', float(stats['mr_head'])):.3f}",
                        "mix": f"{ema.get('l1_mix', float(stats['l1_mix'])):.3f}",
                        "sil": f"{ema.get('silence', float(stats['silence'])):.3f}",
                        "leak": f"{ema.get('leak', float(stats['leak'])):.3f}",
                    })

                # checkpointing
                if (global_step % int(args.save_every)) == 0:
                    save_ckpt(
                        out_dir / f"ckpt_step_{global_step:08d}.pt",
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

        # end epoch
        save_ckpt(
            out_dir / f"ckpt_epoch_{epoch:04d}.pt",
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

    dt = time.time() - t0
    print(f"[done] time={dt/3600:.2f}h, steps={global_step}, out={out_dir}")


if __name__ == "__main__":
    main()
