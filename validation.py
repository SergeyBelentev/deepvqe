from __future__ import annotations
from typing import Optional, Dict

import numpy as np
import soundfile as sf
import torch
from torch import nn

from train_phase_a import TrackItem


def _read_stereo_full(path: str) -> tuple[torch.Tensor, int]:
    """
    returns: wav (2,T) float32 torch, sr
    """
    x, sr = sf.read(path, always_2d=True)  # (T,C)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] >= 2:
        x = x[:, :2]
    else:
        raise RuntimeError(f"bad audio channels: {x.shape}")
    xt = torch.from_numpy(x.astype(np.float32)).transpose(0, 1).contiguous()  # (2,T)
    return xt, int(sr)

def _load_track_full(
    it: TrackItem,
    *,
    sr: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      mix: (2,T)
      tgt: (4,2,T) for STEM_ORDER
    No gain, no recipe, no scaling.
    Crops everything to the minimal common length across available streams
    (same idea as _min_len_cache in dataset).
    """
    waves: dict[str, torch.Tensor] = {}

    # full is mandatory
    full, sr_full = _read_stereo_full(it.full)
    if sr_full != sr:
        raise RuntimeError(f"SR mismatch for {it.full}: {sr_full} != {sr}")
    waves["full"] = full

    # stems (optional -> zeros)
    def read_or_zeros(p: Optional[str], T: int) -> torch.Tensor:
        if not p:
            return torch.zeros((2, T), dtype=torch.float32)
        w, sr_w = _read_stereo_full(p)
        if sr_w != sr:
            raise RuntimeError(f"SR mismatch for {p}: {sr_w} != {sr}")
        return w

    # read available first (to decide min length)
    # but to mimic your _min_len, we include only existing paths in min computation
    lens = [waves["full"].shape[1]]

    if it.bass:
        w, sr_w = _read_stereo_full(it.bass)
        if sr_w != sr: raise RuntimeError(f"SR mismatch for {it.bass}: {sr_w} != {sr}")
        waves["bass"] = w; lens.append(w.shape[1])
    if it.drums:
        w, sr_w = _read_stereo_full(it.drums)
        if sr_w != sr: raise RuntimeError(f"SR mismatch for {it.drums}: {sr_w} != {sr}")
        waves["drums"] = w; lens.append(w.shape[1])
    if it.instruments:
        w, sr_w = _read_stereo_full(it.instruments)
        if sr_w != sr: raise RuntimeError(f"SR mismatch for {it.instruments}: {sr_w} != {sr}")
        waves["instruments"] = w; lens.append(w.shape[1])
    if it.vocals:
        w, sr_w = _read_stereo_full(it.vocals)
        if sr_w != sr: raise RuntimeError(f"SR mismatch for {it.vocals}: {sr_w} != {sr}")
        waves["vocals"] = w; lens.append(w.shape[1])
    if it.melody:
        w, sr_w = _read_stereo_full(it.melody)
        if sr_w != sr: raise RuntimeError(f"SR mismatch for {it.melody}: {sr_w} != {sr}")
        waves["melody"] = w; lens.append(w.shape[1])

    T = int(min(lens))
    # crop all loaded
    for k in list(waves.keys()):
        waves[k] = waves[k][:, :T].contiguous()

    # build stems
    bass = waves.get("bass", torch.zeros((2, T), dtype=torch.float32))
    drums = waves.get("drums", torch.zeros((2, T), dtype=torch.float32))
    inst = waves.get("instruments", torch.zeros((2, T), dtype=torch.float32))
    mel  = waves.get("melody", torch.zeros((2, T), dtype=torch.float32))
    music = inst + mel
    vocals = waves.get("vocals", torch.zeros((2, T), dtype=torch.float32))

    tgt = torch.stack([bass, drums, music, vocals], dim=0)  # (4,2,T)
    mix = waves["full"]  # (2,T)
    return mix, tgt


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
    SI-SDR (scale-invariant):
      s = <p,t>/<t,t> * t
      e = p - s
      10 log10(||s||^2 / ||e||^2)
    """
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        tgt = tgt.unsqueeze(0)
    # (B,2,T)
    dot = (pred * tgt).sum(dim=-1)                       # (B,2)
    tt  = (tgt * tgt).sum(dim=-1).clamp_min(eps)         # (B,2)
    alpha = (dot / tt).unsqueeze(-1)                     # (B,2,1)
    s = alpha * tgt
    e = pred - s
    num = (s * s).sum(dim=-1)                            # (B,2)
    den = (e * e).sum(dim=-1).clamp_min(eps)
    sisdr = 10.0 * torch.log10((num.clamp_min(eps) / den))
    return sisdr.mean()


@torch.no_grad()
def _forward_chunk_with_tf(
    model: nn.Module,
    chunk: torch.Tensor,          # (1,2,seg_len) on device
    *,
    autocast_ctx,
    STEM_ORDER,
) -> tuple[torch.Tensor, dict]:
    """
    returns:
      pred_seg: (1,4,2,seg_len) float32 (on device)
      tf_pack: dict from model["_tf"]
    """
    with autocast_ctx:
        out = model(chunk, return_debug=False, return_tf=True)  # dict stem->(1,2,seg) + "_tf"
        tf_pack = out.pop("_tf")
        pred = torch.stack([out[s] for s in STEM_ORDER], dim=1)  # (1,4,2,seg)
    return pred, tf_pack


@torch.no_grad()
def _separate_full_track_sequential(
    model: nn.Module,
    mix: torch.Tensor,     # (2,T) float32 CPU or GPU
    STEM_ORDER,
    *,
    device: torch.device,
    autocast_ctx,
    seg_len: int,
) -> torch.Tensor:
    """
    Returns pred_stems: (4,2,T) float32 on CPU
    No overlap, sequential chunks of seg_len.
    """
    assert mix.ndim == 2 and mix.shape[0] == 2
    T = int(mix.shape[1])
    out = torch.zeros((len(STEM_ORDER), 2, T), dtype=torch.float32, device="cpu")

    mix_d = mix.to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,2,T)

    pos = 0
    while pos < T:
        end = min(T, pos + seg_len)
        valid = end - pos

        chunk = torch.zeros((1, 2, seg_len), device=device, dtype=torch.float32)
        chunk[:, :, :valid] = mix_d[:, :, pos:end]

        with autocast_ctx:
            pred_dict = model(chunk, return_debug=False, return_tf=True)  # dict stem -> (B,2,seg)
            tf_pack = pred_dict.pop("_tf")
            # если у тебя forward всегда добавляет _tf при return_tf=False — тогда просто pred_dict.pop("_tf", None)

        pred = torch.stack([pred_dict[s] for s in STEM_ORDER], dim=1)  # (1,4,2,seg)
        pred = pred[0].to(dtype=torch.float32).detach().cpu()          # (4,2,seg)

        out[:, :, pos:end] = pred[:, :, :valid]
        pos += seg_len

    return out

def _ddp_allreduce_sum_count(x_sum: torch.Tensor, x_cnt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    import torch.distributed as dist
    if not dist.is_initialized():
        return x_sum, x_cnt
    dist.all_reduce(x_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(x_cnt, op=dist.ReduceOp.SUM)
    return x_sum, x_cnt

@torch.no_grad()
def run_validation_full_tracks(
    *,
    epoch: int,
    model: nn.Module,
    loss_comp,
    val_items: list[TrackItem],
    device: torch.device,
    autocast_ctx,
    weights: Dict[str, float],
    seg_len: int,
    sr: int,
    silence_rms_thr: float,
    use_ddp: bool,
    rank: int,
    world_size: int,
    STEM_ORDER,
) -> Dict[str, float]:
    """
    - прогоняем все треки целиком
    - режем на sequential 8s
    - без RecipeBook/gain
    - DDP: делим треки по rank, потом all_reduce метрики
    """
    was_training = model.training
    model.eval()

    # sums for losses (segment-averaged)
    loss_sums = {
        "loss_total": 0.0,
        "l1_head": 0.0,
        "mr_head": 0.0,
        "mr_sc_head": 0.0,
        "mr_lm_head": 0.0,
        "tf_ri_mix": 0.0,
        "tf_lm_mix": 0.0,
        "tf_ri_4096": 0.0,
        "tf_lm_4096": 0.0,
        "tf_ri_2048": 0.0,
        "tf_lm_2048": 0.0,
        "silence": 0.0,
        "leak": 0.0,
    }
    loss_count = 0

    # SDR metrics per stem
    # We accumulate sums and counts to later all_reduce.
    sdr_sum = torch.zeros((4,), dtype=torch.float64, device=device)
    sisdr_sum = torch.zeros((4,), dtype=torch.float64, device=device)
    met_cnt = torch.zeros((4,), dtype=torch.float64, device=device)

    # iterate shard
    for ti in range(rank, len(val_items), world_size):
        it = val_items[ti]

        mix_cpu, tgt_cpu = _load_track_full(it, sr=sr)  # (2,T), (4,2,T) CPU float32
        T = int(mix_cpu.shape[1])

        # predict full track sequentially
        pred_cpu = _separate_full_track_sequential(
            model=model,
            mix=mix_cpu,
            device=device,
            autocast_ctx=autocast_ctx,
            seg_len=seg_len,
            STEM_ORDER=STEM_ORDER,
        )  # (4,2,T) CPU

        # --- metrics per stem (skip if tgt is (near) silence)
        for si in range(4):
            tgt_s = tgt_cpu[si]   # (2,T)
            pred_s = pred_cpu[si] # (2,T)

            # RMS-based presence gate (same idea as your silence_rms_thr)
            rms_val = float(torch.sqrt(tgt_s.pow(2).mean()).item())
            if rms_val < float(silence_rms_thr):
                continue

            # compute on device for speed
            t = tgt_s.to(device=device, dtype=torch.float32)
            p = pred_s.to(device=device, dtype=torch.float32)

            sdr_sum[si] += _sdr(p, t).to(dtype=torch.float64)
            sisdr_sum[si] += _si_sdr(p, t).to(dtype=torch.float64)
            met_cnt[si] += 1.0

        # --- losses (optional): считаем по сегментам последовательно, без TF mix (router targets зависят от tf_pack)
        # Если тебе loss-метрики в валидаторе не нужны — можно удалить блок ниже.
        # Тут мы считаем только l1/mr/silence/leak (tf_pack=None).
        pos = 0
        while pos < T:
            end = min(T, pos + seg_len)
            valid = end - pos

            mix_seg = torch.zeros((1, 2, seg_len), dtype=torch.float32, device=device)
            tgt_seg = torch.zeros((1, 4, 2, seg_len), dtype=torch.float32, device=device)
            pm_seg  = torch.zeros((1, 4), dtype=torch.float32, device=device)
            mt_seg  = torch.zeros((1, 2, seg_len), dtype=torch.float32, device=device)

            mix_seg[:, :, :valid] = mix_cpu[:, pos:end].to(device=device)
            tgt_seg[:, :, :, :valid] = tgt_cpu[:, :, pos:end].to(device=device)
            # present mask: 1 если rms(tgt_stem_seg) >= thr
            # (можно и по целому треку, но по сегментам корректнее)
            for si in range(4):
                seg_rms = torch.sqrt(tgt_seg[0, si].pow(2).mean()).item()
                pm_seg[0, si] = 1.0 if seg_rms >= float(silence_rms_thr) else 0.0
            mt_seg[:, :, :valid] = tgt_seg.sum(dim=1)[:, :, :valid]  # mix_target = stem_sum (для consistency)

            # pred from already computed full pred (чтобы не гонять модель второй раз)
            pred_seg = torch.zeros((1, 4, 2, seg_len), dtype=torch.float32, device=device)
            pred_seg[:, :, :, :valid] = pred_cpu[:, :, pos:end].to(device=device).unsqueeze(0)

            loss, stats = loss_comp(
                pred_stems=pred_seg,
                tgt_stems=tgt_seg,
                present_mask=pm_seg,
                mix_target=mt_seg,
                weights=weights,
                tf_pack=None,  # важно: без TF-mix
            )

            for k in loss_sums.keys():
                loss_sums[k] += float(stats[k].item())
            loss_count += 1

            pos += seg_len

    # --- all_reduce metrics and losses
    if use_ddp:
        import torch.distributed as dist

        sdr_sum, met_cnt = _ddp_allreduce_sum_count(sdr_sum, met_cnt)
        sisdr_sum, _ = _ddp_allreduce_sum_count(sisdr_sum, met_cnt.clone())

        # losses: pack into tensors for 1 all_reduce
        keys = sorted(loss_sums.keys())
        vec = torch.tensor([loss_sums[k] for k in keys] + [float(loss_count)], device=device, dtype=torch.float64)
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        loss_count_g = float(vec[-1].item())
        for i, k in enumerate(keys):
            loss_sums[k] = float(vec[i].item())
        loss_count = int(loss_count_g)

    # finalize
    out: Dict[str, float] = {}
    # losses mean
    if loss_count > 0:
        for k, v in loss_sums.items():
            out[k] = float(v) / float(loss_count)

    # sdr/sisdr per stem and macro averages
    stem_names = STEM_ORDER
    sdr_mean = sdr_sum / met_cnt.clamp_min(1.0)
    sisdr_mean = sisdr_sum / met_cnt.clamp_min(1.0)

    for i, s in enumerate(stem_names):
        out[f"sdr_{s}"] = float(sdr_mean[i].item())
        out[f"sisdr_{s}"] = float(sisdr_mean[i].item())
        out[f"cnt_{s}"] = float(met_cnt[i].item())

    # macro avg over stems with cnt>0
    m = (met_cnt > 0).to(dtype=torch.float64)
    denom = float(m.sum().clamp_min(1.0).item())
    out["sdr_avg"] = float((sdr_mean * m).sum().item() / denom)
    out["sisdr_avg"] = float((sisdr_mean * m).sum().item() / denom)

    if was_training:
        model.train()
    return out
