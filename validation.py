# val_full_tracks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import soundfile as sf
import torch
from torch import nn


# -----------------------------
# Audio loading (full track)
# -----------------------------

def _read_stereo_full(path: str) -> tuple[torch.Tensor, int]:
    """
    returns: wav (2,T) float32 torch (CPU), sr
    """
    x, sr = sf.read(path, always_2d=True)  # (T,C) float
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] >= 2:
        x = x[:, :2]
    else:
        raise RuntimeError(f"bad audio channels: {x.shape}")
    xt = torch.from_numpy(x.astype(np.float32)).transpose(0, 1).contiguous()  # (2,T)
    return xt, int(sr)


def _load_track_full(
    it: Any,  # TrackItem-like: has .full .bass .drums .instruments .vocals? .melody?
    *,
    sr: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      mix: (2,T) float32 CPU
      tgt: (4,2,T) float32 CPU  in STEM_ORDER ["bass","drums","music","vocals"]
    No gain, no recipe, no scaling.
    Crops all streams to minimal common length among those that exist (like your _min_len).
    """
    waves: dict[str, torch.Tensor] = {}

    # full mandatory
    full, sr_full = _read_stereo_full(it.full)
    if sr_full != sr:
        raise RuntimeError(f"SR mismatch for {it.full}: {sr_full} != {sr}")
    waves["full"] = full

    lens = [waves["full"].shape[1]]

    # helper
    def _read_opt(p: Optional[str], key: str) -> None:
        if not p:
            return
        w, sr_w = _read_stereo_full(p)
        if sr_w != sr:
            raise RuntimeError(f"SR mismatch for {p}: {sr_w} != {sr}")
        waves[key] = w
        lens.append(int(w.shape[1]))

    _read_opt(getattr(it, "bass", None), "bass")
    _read_opt(getattr(it, "drums", None), "drums")
    _read_opt(getattr(it, "instruments", None), "instruments")
    _read_opt(getattr(it, "vocals", None), "vocals")
    _read_opt(getattr(it, "melody", None), "melody")

    T = int(min(lens))
    for k in list(waves.keys()):
        waves[k] = waves[k][:, :T].contiguous()

    bass = waves.get("bass", torch.zeros((2, T), dtype=torch.float32))
    drums = waves.get("drums", torch.zeros((2, T), dtype=torch.float32))
    inst = waves.get("instruments", torch.zeros((2, T), dtype=torch.float32))
    mel = waves.get("melody", torch.zeros((2, T), dtype=torch.float32))
    music = inst + mel
    vocals = waves.get("vocals", torch.zeros((2, T), dtype=torch.float32))

    tgt = torch.stack([bass, drums, music, vocals], dim=0)  # (4,2,T)
    mix = waves["full"]  # (2,T)
    return mix, tgt


# -----------------------------
# Metrics: SDR / SI-SDR
# -----------------------------

def _sdr(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred,tgt: (2,T) or (B,2,T) float
    returns: scalar tensor (mean over batch and channels)
    SDR = 10 log10(||t||^2 / ||t - p||^2)
    """
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        tgt = tgt.unsqueeze(0)
    e = tgt - pred
    num = (tgt ** 2).sum(dim=-1)  # (B,2)
    den = (e ** 2).sum(dim=-1).clamp_min(eps)
    sdr = 10.0 * torch.log10((num.clamp_min(eps) / den))
    return sdr.mean()


def _si_sdr(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred,tgt: (2,T) or (B,2,T) float
    SI-SDR:
      s = <p,t>/<t,t> * t
      e = p - s
      10 log10(||s||^2 / ||e||^2)
    """
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        tgt = tgt.unsqueeze(0)
    dot = (pred * tgt).sum(dim=-1)                       # (B,2)
    tt  = (tgt * tgt).sum(dim=-1).clamp_min(eps)         # (B,2)
    alpha = (dot / tt).unsqueeze(-1)                     # (B,2,1)
    s = alpha * tgt
    e = pred - s
    num = (s * s).sum(dim=-1)                            # (B,2)
    den = (e * e).sum(dim=-1).clamp_min(eps)
    sisdr = 10.0 * torch.log10((num.clamp_min(eps) / den))
    return sisdr.mean()


def _rms2ch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # x: (2,T)
    return torch.sqrt(x.pow(2).mean().clamp_min(eps))


# -----------------------------
# DDP helpers
# -----------------------------

def _ddp_is_init() -> bool:
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def _ddp_allreduce_sum_count(x_sum: torch.Tensor, x_cnt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    import torch.distributed as dist
    if not dist.is_initialized():
        return x_sum, x_cnt
    dist.all_reduce(x_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(x_cnt, op=dist.ReduceOp.SUM)
    return x_sum, x_cnt


# -----------------------------
# One-chunk forward (pred + tf)
# -----------------------------

@torch.no_grad()
def _forward_chunk_with_tf(
    model: nn.Module,
    chunk: torch.Tensor,          # (1,2,seg_len) on device
    *,
    autocast_ctx,
    STEM_ORDER: list[str],
) -> tuple[torch.Tensor, dict]:
    """
    returns:
      pred_seg: (1,4,2,seg_len) float32 on device (caller may cast)
      tf_pack: dict from model["_tf"]
    """
    with autocast_ctx:
        out = model(chunk, return_debug=False, return_tf=True)
        tf_pack = out.pop("_tf")
        pred = torch.stack([out[s] for s in STEM_ORDER], dim=1)  # (1,4,2,seg)
    return pred, tf_pack


# -----------------------------
# Main validation
# -----------------------------

@torch.no_grad()
def run_validation_full_tracks(
    *,
    epoch: int,
    model: nn.Module,
    loss_comp: nn.Module,
    val_items: list[Any],              # list[TrackItem]
    device: torch.device,
    autocast_ctx,
    weights: Dict[str, float],
    seg_len: int,
    sr: int,
    silence_rms_thr: float,
    use_ddp: bool,
    rank: int,
    world_size: int,
    STEM_ORDER: list[str],             # ["bass","drums","music","vocals"]
    compute_losses: bool = True,       # includes TF-mix when True
) -> Dict[str, float]:
    """
    Full-track deterministic validation:
      - all tracks, full length
      - sequential chunks of seg_len (no overlap)
      - no RecipeBook, no gains, no clip-safe scaling
      - DDP: split tracks by rank, then all-reduce
      - metrics: SDR + SI-SDR per stem + macro avg
      - optional loss pack: uses loss_comp and tf_pack from model(return_tf=True)
    """
    was_training = model.training
    model.eval()

    # --- loss accumulators (segment-averaged)
    loss_keys = [
        "loss_total",
        "l1_head",
        "mr_head",
        "mr_sc_head",
        "mr_lm_head",
        "tf_ri_mix",
        "tf_lm_mix",
        "tf_ri_4096",
        "tf_lm_4096",
        "tf_ri_2048",
        "tf_lm_2048",
        "silence",
        "leak",
    ]
    loss_sums: Dict[str, float] = {k: 0.0 for k in loss_keys}
    loss_count = 0

    # --- metric accumulators per stem
    H = len(STEM_ORDER)
    sdr_sum = torch.zeros((H,), dtype=torch.float64, device=device)
    sisdr_sum = torch.zeros((H,), dtype=torch.float64, device=device)
    met_cnt = torch.zeros((H,), dtype=torch.float64, device=device)

    # iterate shard
    for ti in range(rank, len(val_items), world_size):
        it = val_items[ti]
        mix_cpu, tgt_cpu = _load_track_full(it, sr=sr)  # (2,T), (H,2,T) CPU
        T = int(mix_cpu.shape[1])

        # full prediction for metrics (CPU)
        pred_full_cpu = torch.zeros((H, 2, T), dtype=torch.float32, device="cpu")

        pos = 0
        while pos < T:
            end = min(T, pos + seg_len)
            valid = end - pos

            # build padded chunk
            chunk = torch.zeros((1, 2, seg_len), device=device, dtype=torch.float32)
            chunk[:, :, :valid] = mix_cpu[:, pos:end].to(device=device, dtype=torch.float32)

            # forward once (pred + tf_pack)
            pred_seg, tf_pack = _forward_chunk_with_tf(
                model,
                chunk,
                autocast_ctx=autocast_ctx,
                STEM_ORDER=STEM_ORDER,
            )  # pred_seg: (1,H,2,seg)

            # stash to full buffer
            pred_full_cpu[:, :, pos:end] = pred_seg[0].to(dtype=torch.float32).detach().cpu()[:, :, :valid]

            if compute_losses:
                # build targets for this segment
                tgt_seg = torch.zeros((1, H, 2, seg_len), device=device, dtype=torch.float32)
                tgt_seg[:, :, :, :valid] = tgt_cpu[:, :, pos:end].to(device=device, dtype=torch.float32)

                # present mask from target RMS (segment-wise)
                pm_seg = torch.zeros((1, H), device=device, dtype=torch.float32)
                for si in range(H):
                    seg_rms = torch.sqrt(tgt_seg[0, si].pow(2).mean()).item()
                    pm_seg[0, si] = 1.0 if seg_rms >= float(silence_rms_thr) else 0.0

                # mix_target = stem_sum
                mt_seg = tgt_seg.sum(dim=1)  # (1,2,seg)

                # compute loss pack with TF mix
                with torch.autocast(device_type="cuda", enabled=False):
                    loss, stats = loss_comp(
                        pred_stems=pred_seg.float(),
                        tgt_stems=tgt_seg.float(),
                        present_mask=pm_seg.float(),
                        mix_target=mt_seg.float(),
                        weights=weights,
                        tf_pack=tf_pack,  # <-- TF LOSS MIX INCLUDED
                    )

                for k in loss_keys:
                    loss_sums[k] += float(stats[k].item())
                loss_count += 1

            pos += seg_len

        # --- metrics per stem over whole track
        for si in range(H):
            tgt_s = tgt_cpu[si]          # (2,T) CPU
            pred_s = pred_full_cpu[si]   # (2,T) CPU

            if float(_rms2ch(tgt_s).item()) < float(silence_rms_thr):
                continue

            t = tgt_s.to(device=device, dtype=torch.float32)
            p = pred_s.to(device=device, dtype=torch.float32)

            sdr_sum[si] += _sdr(p, t).to(dtype=torch.float64)
            sisdr_sum[si] += _si_sdr(p, t).to(dtype=torch.float64)
            met_cnt[si] += 1.0

    # --- DDP all-reduce
    if use_ddp and _ddp_is_init():
        import torch.distributed as dist

        sdr_sum, met_cnt = _ddp_allreduce_sum_count(sdr_sum, met_cnt)
        sisdr_sum, _ = _ddp_allreduce_sum_count(sisdr_sum, met_cnt.clone())

        if compute_losses:
            keys = loss_keys
            vec = torch.tensor([loss_sums[k] for k in keys] + [float(loss_count)], device=device, dtype=torch.float64)
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
            loss_count_g = float(vec[-1].item())
            for i, k in enumerate(keys):
                loss_sums[k] = float(vec[i].item())
            loss_count = int(loss_count_g)

    # finalize
    out: Dict[str, float] = {}

    if compute_losses and loss_count > 0:
        for k in loss_keys:
            out[k] = float(loss_sums[k]) / float(loss_count)

    sdr_mean = sdr_sum / met_cnt.clamp_min(1.0)
    sisdr_mean = sisdr_sum / met_cnt.clamp_min(1.0)

    for i, s in enumerate(STEM_ORDER):
        out[f"sdr_{s}"] = float(sdr_mean[i].item())
        out[f"sisdr_{s}"] = float(sisdr_mean[i].item())
        out[f"cnt_{s}"] = float(met_cnt[i].item())

    m = (met_cnt > 0).to(dtype=torch.float64)
    denom = float(m.sum().clamp_min(1.0).item())
    out["sdr_avg"] = float((sdr_mean * m).sum().item() / denom)
    out["sisdr_avg"] = float((sisdr_mean * m).sum().item() / denom)

    if was_training:
        model.train()
    return out
