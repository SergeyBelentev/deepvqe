# val_full_tracks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch import nn
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, Future
from collections import deque


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

    full, sr_full = _read_stereo_full(it.full)
    if sr_full != sr:
        raise RuntimeError(f"SR mismatch for {it.full}: {sr_full} != {sr}")
    waves["full"] = full

    lens = [int(waves["full"].shape[1])]

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
    mix = waves["full"]                                     # (2,T)
    return mix, tgt


# -----------------------------
# Metrics: SDR / SI-SDR
# -----------------------------

def _sdr(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        tgt = tgt.unsqueeze(0)
    e = tgt - pred
    num = (tgt ** 2).sum(dim=-1)
    den = (e ** 2).sum(dim=-1).clamp_min(eps)
    sdr = 10.0 * torch.log10((num.clamp_min(eps) / den))
    return sdr.mean()


def _si_sdr(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        tgt = tgt.unsqueeze(0)
    dot = (pred * tgt).sum(dim=-1)
    tt = (tgt * tgt).sum(dim=-1).clamp_min(eps)
    alpha = (dot / tt).unsqueeze(-1)
    s = alpha * tgt
    e = pred - s
    num = (s * s).sum(dim=-1)
    den = (e * e).sum(dim=-1).clamp_min(eps)
    sisdr = 10.0 * torch.log10((num.clamp_min(eps) / den))
    return sisdr.mean()


def _rms2ch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
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
# One-batch forward (pred + tf)
# -----------------------------

@torch.no_grad()
def _forward_batch_with_tf(
    model: nn.Module,
    batch_chunks: torch.Tensor,   # (B,2,seg_len) on device
    *,
    autocast_ctx,
    STEM_ORDER: list[str],
) -> tuple[torch.Tensor, dict]:
    with autocast_ctx:
        out = model(batch_chunks, return_debug=False, return_tf=True)
        tf_pack = out.pop("_tf")
        pred = torch.stack([out[s] for s in STEM_ORDER], dim=1)  # (B,H,2,seg)
    return pred, tf_pack


# -----------------------------
# Internal: track state
# -----------------------------

@dataclass
class _TrackState:
    mix: torch.Tensor          # (2,T) CPU
    tgt: torch.Tensor          # (H,2,T) CPU
    pred_full: torch.Tensor    # (H,2,T) CPU
    T: int
    pos: int                  # next segment start


def _num_segs(T: int, seg_len: int) -> int:
    return (T + seg_len - 1) // seg_len


# -----------------------------
# Main validation (batched + IO workers)
# -----------------------------

@torch.no_grad()
def run_validation_full_tracks(
    *,
    epoch: int,
    model: nn.Module,
    loss_comp: nn.Module,
    val_items: list[Any],
    device: torch.device,
    autocast_ctx,
    weights: Dict[str, float],
    seg_len: int,
    sr: int,
    silence_rms_thr: float,
    use_ddp: bool,
    rank: int,
    world_size: int,
    STEM_ORDER: list[str],
    batch_size: int,

    # NEW: parallel IO
    io_workers: int = 0,          # set = train num_workers
    prefetch_tracks: int = 0,     # default: 2*io_workers
    compute_losses: bool = True,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()

    H = len(STEM_ORDER)
    B = int(batch_size)
    if B <= 0:
        raise ValueError(f"batch_size must be >0, got {batch_size}")

    io_workers = int(io_workers)
    if io_workers < 0:
        raise ValueError(f"io_workers must be >=0, got {io_workers}")
    if prefetch_tracks <= 0:
        prefetch_tracks = max(1, 2 * io_workers) if io_workers > 0 else 1
    prefetch_tracks = int(prefetch_tracks)

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

    sdr_sum = torch.zeros((H,), dtype=torch.float64, device=device)
    sisdr_sum = torch.zeros((H,), dtype=torch.float64, device=device)
    met_cnt = torch.zeros((H,), dtype=torch.float64, device=device)

    # shard indices for this rank
    shard_indices = list(range(rank, len(val_items), world_size))

    # precompute total segments for tqdm (approx exact: need lengths; we compute after load)
    # We'll update tqdm.total dynamically as we load (supported by tqdm).
    pbar = tqdm(total=0, desc=f"val full epoch {epoch} rank{rank}", dynamic_ncols=True)

    # futures queue: holds loaded (mix,tgt,it_index)
    futs: deque[Future] = deque()
    loaded_tracks = 0

    def submit(ex: ThreadPoolExecutor, it: Any, idx: int) -> None:
        futs.append(ex.submit(_load_track_full, it, sr=sr))

    # consume loaded into states gradually
    states: List[_TrackState] = []

    if io_workers > 0:
        with ThreadPoolExecutor(max_workers=io_workers) as ex:
            # kick initial prefetch
            ptr = 0
            while ptr < len(shard_indices) and len(futs) < prefetch_tracks:
                ti = shard_indices[ptr]
                submit(ex, val_items[ti], ti)
                ptr += 1

            while futs:
                # get next loaded
                fut = futs.popleft()
                mix_cpu, tgt_cpu = fut.result()

                T = int(mix_cpu.shape[1])
                pred_full_cpu = torch.zeros((H, 2, T), dtype=torch.float32, device="cpu")
                states.append(_TrackState(mix=mix_cpu, tgt=tgt_cpu, pred_full=pred_full_cpu, T=T, pos=0))

                loaded_tracks += 1
                pbar.total += _num_segs(T, seg_len)
                pbar.set_postfix_str(f"tracks_loaded={loaded_tracks}/{len(shard_indices)}")

                # submit more
                while ptr < len(shard_indices) and len(futs) < prefetch_tracks:
                    ti = shard_indices[ptr]
                    submit(ex, val_items[ti], ti)
                    ptr += 1

                if ptr >= len(shard_indices) and not futs:
                    break
    else:
        # sequential load
        for ti in shard_indices:
            it = val_items[ti]
            mix_cpu, tgt_cpu = _load_track_full(it, sr=sr)
            T = int(mix_cpu.shape[1])
            pred_full_cpu = torch.zeros((H, 2, T), dtype=torch.float32, device="cpu")
            states.append(_TrackState(mix=mix_cpu, tgt=tgt_cpu, pred_full=pred_full_cpu, T=T, pos=0))

            loaded_tracks += 1
            pbar.total += _num_segs(T, seg_len)
            pbar.set_postfix_str(f"tracks_loaded={loaded_tracks}/{len(shard_indices)}")

    # --------
    # Batched segment inference (same as before)
    # --------
    active = [i for i in range(len(states)) if states[i].pos < states[i].T]

    while active:
        batch_chunks = torch.zeros((0, 2, seg_len), device=device, dtype=torch.float32)
        meta: List[Tuple[int, int, int]] = []  # (state_index, pos, valid)

        ai = 0
        while len(meta) < B and active:
            if ai >= len(active):
                ai = 0
                if len(meta) == 0:
                    break

            si = active[ai]
            st = states[si]
            if st.pos >= st.T:
                active.pop(ai)
                continue

            pos = st.pos
            end = min(st.T, pos + seg_len)
            valid = end - pos

            chunk = torch.zeros((1, 2, seg_len), device=device, dtype=torch.float32)
            chunk[:, :, :valid] = st.mix[:, pos:end].to(device=device, dtype=torch.float32)

            batch_chunks = torch.cat([batch_chunks, chunk], dim=0)
            meta.append((si, pos, valid))

            st.pos = end
            ai += 1

        if len(meta) == 0:
            break

        pred_b, tf_pack = _forward_batch_with_tf(
            model,
            batch_chunks,
            autocast_ctx=autocast_ctx,
            STEM_ORDER=STEM_ORDER,
        )

        for bi, (si, pos, valid) in enumerate(meta):
            st = states[si]
            end = pos + valid
            st.pred_full[:, :, pos:end] = pred_b[bi].to(dtype=torch.float32).detach().cpu()[:, :, :valid]

        if compute_losses:
            bsz = len(meta)
            tgt_b = torch.zeros((bsz, H, 2, seg_len), device=device, dtype=torch.float32)
            pm_b = torch.zeros((bsz, H), device=device, dtype=torch.float32)

            for bi, (si, pos, valid) in enumerate(meta):
                st = states[si]
                tgt_b[bi, :, :, :valid] = st.tgt[:, :, pos:pos+valid].to(device=device, dtype=torch.float32)
                for h in range(H):
                    seg_rms = torch.sqrt(tgt_b[bi, h].pow(2).mean()).item()
                    pm_b[bi, h] = 1.0 if seg_rms >= float(silence_rms_thr) else 0.0

            mt_b = tgt_b.sum(dim=1)

            with torch.autocast(device_type="cuda", enabled=False):
                loss, stats = loss_comp(
                    pred_stems=pred_b.float(),
                    tgt_stems=tgt_b.float(),
                    present_mask=pm_b.float(),
                    mix_target=mt_b.float(),
                    weights=weights,
                    tf_pack=tf_pack,
                )

            for k in loss_keys:
                loss_sums[k] += float(stats[k].item())
            loss_count += 1

        pbar.update(len(meta))
        active = [i for i in active if states[i].pos < states[i].T]

    pbar.close()

    # --------
    # Metrics per stem over whole tracks
    # --------
    for st in states:
        for si in range(H):
            tgt_s = st.tgt[si]
            pred_s = st.pred_full[si]

            if float(_rms2ch(tgt_s).item()) < float(silence_rms_thr):
                continue

            t = tgt_s.to(device=device, dtype=torch.float32)
            p = pred_s.to(device=device, dtype=torch.float32)

            sdr_sum[si] += _sdr(p, t).to(dtype=torch.float64)
            sisdr_sum[si] += _si_sdr(p, t).to(dtype=torch.float64)
            met_cnt[si] += 1.0

    # --------
    # DDP all-reduce
    # --------
    if use_ddp and _ddp_is_init():
        import torch.distributed as dist

        sdr_sum, met_cnt = _ddp_allreduce_sum_count(sdr_sum, met_cnt)
        sisdr_sum, _ = _ddp_allreduce_sum_count(sisdr_sum, met_cnt.clone())

        if compute_losses:
            vec = torch.tensor([loss_sums[k] for k in loss_keys] + [float(loss_count)],
                               device=device, dtype=torch.float64)
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
            loss_count_g = float(vec[-1].item())
            for i, k in enumerate(loss_keys):
                loss_sums[k] = float(vec[i].item())
            loss_count = int(loss_count_g)

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
