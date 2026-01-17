# infer_phase_a_v2.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from deepvqe import DeepVQEConditionalStemSeparator


# -----------------------
# STFT helper (fp32, matches training defaults)
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
        # x: (N,T) float32
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
# Audio I/O
# -----------------------
def match_length(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    x: (2,Tx) -> (2,T) by crop/pad with zeros
    """
    Tx = x.shape[1]
    if Tx == T:
        return x
    if Tx > T:
        return x[:, :T].contiguous()
    # pad right
    return F.pad(x, (0, T - Tx))


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def read_audio(path: str, sr_expected: int) -> torch.Tensor:
    """
    Returns stereo float32: (2, T)
    """
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != sr_expected:
        raise RuntimeError(f"SR mismatch: got {sr}, expected {sr_expected}. Resample outside this script.")
    # x: (T, ch)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    else:
        x = x[:, :2]
    return torch.from_numpy(x).transpose(0, 1).contiguous()  # (2,T)


def write_audio(path: str, y: np.ndarray, sr: int, fmt: str = "float") -> None:
    """
    y: (T,2) float32
    """
    path = str(path)
    if fmt == "float":
        sf.write(path, y, sr, subtype="FLOAT")
    elif fmt == "pcm16":
        # clip to [-1,1] to be safe
        y16 = np.clip(y, -1.0, 1.0)
        sf.write(path, y16, sr, subtype="PCM_16")
    else:
        raise ValueError("fmt must be one of: float, pcm16")


# -----------------------
# Model loading
# -----------------------
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

def load_model(ckpt_path: str, device: torch.device, n_fft: int, num_heads: int = 4):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(f"Bad checkpoint: {ckpt_path}")

    model = DeepVQEConditionalStemSeparator(n_fft=n_fft, num_heads=num_heads).to(device)

    sd = ckpt["model"]
    if not isinstance(sd, dict):
        raise RuntimeError("ckpt['model'] is not a state_dict dict")

    # сначала пробуем как есть (на случай если чекпойнт уже “чистый”)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        sd2 = _normalize_checkpoint_state_dict(sd)
        model.load_state_dict(sd2, strict=True)

    model.eval()
    return model


# -----------------------
# OLA inference
# -----------------------
def make_hann_ola(win_len: int, device: torch.device) -> torch.Tensor:
    # nice for 50% overlap
    w = torch.hann_window(win_len, periodic=True, dtype=torch.float32, device=device)
    # avoid exact zeros at edges in case of tiny numeric issues
    return w.clamp_min(1e-8)


# -----------------------
# Residual U-Net debug tap (forward_hook)
# -----------------------
class ResidualUnetTap:
    """
    Hooks every module in model.residual_unets and captures:
      - inputs (tuple): we expect (d1_i, y_ccm_i) per your design
      - output: what the U-Net returns (often residual in RI)
    Works per forward() call: call .reset() before model(...)
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.last_out: Dict[int, Any] = {}
        self.last_inp: Dict[int, Any] = {}
        self.handles = []

        if not hasattr(model, "residual_unets"):
            raise RuntimeError("Model has no attribute 'residual_unets' (nothing to tap).")

        for i, m in enumerate(getattr(model, "residual_unets")):
            self.handles.append(m.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, i: int):
        def hook(mod, inp, out):
            self.last_inp[i] = inp
            self.last_out[i] = out
        return hook

    def reset(self) -> None:
        self.last_inp.clear()
        self.last_out.clear()

    def close(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()


def _pick_tensor(x: Any) -> Optional[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
        return x[0]
    return None


def _to_ri_layout(t: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Try to coerce tensor to RI layout (N,F,Tf,2) needed by istft_ri().
    Supports common variants:
      - (N,F,Tf,2)  -> ok
      - (N,2,Tf,F)  -> permute to (N,F,Tf,2)
    Returns None if it's not RI-like.
    """
    if not isinstance(t, torch.Tensor):
        return None
    if t.ndim != 4:
        return None

    # already (N,F,Tf,2)
    if t.shape[-1] == 2:
        return t

    # maybe (N,2,Tf,F)
    if t.shape[1] == 2:
        return t.permute(0, 3, 2, 1).contiguous()

    return None


def _ri_to_stereo(stft: STFT, ri: torch.Tensor, *, length: int) -> Optional[torch.Tensor]:
    """
    ri: (N,F,Tf,2) where N==2 (stereo channels treated as batch).
    returns: (2,length)
    """
    ri = _to_ri_layout(ri)
    if ri is None:
        return None
    if ri.shape[0] != 2:
        # your infer uses N=2 (stereo). If not, we skip.
        return None

    wav = stft.istft_ri(ri.float(), length=length)  # (2,length)
    return wav



@torch.no_grad()
def infer_ola(
    model: DeepVQEConditionalStemSeparator,
    stft: STFT,
    x_stereo: torch.Tensor,              # (2,T)
    *,
    ref_stereo: Optional[torch.Tensor] = None,  # (2,T) or None
    sr: int,
    chunk_sec: float,
    overlap: float,
    device: torch.device,
    use_amp: bool = False,
    stitch: str = "ola",
    keep_frac: float = 0.6,
    xfade_ms: float = 0.0,
    unet_tap: Optional[ResidualUnetTap] = None,
    unet_dump_what: str = "both",
) -> tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    """
    Returns dict head->(2,T) in float32 on CPU.
    Heads: bass, drums, music, vocals
    """
    assert 0.0 <= overlap < 1.0
    if stitch not in ("ola", "crop", "full"):
        raise ValueError("stitch must be 'ola' or 'crop' or 'full'")

    x = x_stereo.to(device, dtype=torch.float32)
    C, T = x.shape
    assert C == 2

    if ref_stereo is None:
        ref = None
    else:
        ref = ref_stereo.to(device, dtype=torch.float32)
        if ref.shape[0] != 2:
            raise ValueError("ref_stereo must be (2,T)")
        if ref.shape[1] != T:
            # на всякий случай: но лучше это делать ещё в main()
            if ref.shape[1] > T:
                ref = ref[:, :T]
            else:
                ref = F.pad(ref, (0, T - ref.shape[1]))


    chunk_len = int(round(sr * chunk_sec))
    if chunk_len <= 0:
        raise ValueError("chunk_sec too small")

    # debug buffers (time-domain) for U-Net input/output
    dbg_out = None   # (S,2,T_pad)
    dbg_yccm = None  # (S,2,T_pad)


    def pred_to_time(pred: torch.Tensor, length: int) -> torch.Tensor:
        """
        pred: (2,S,F,Tf,2)
        returns y: (S,2,length)
        """
        S = pred.shape[1]

        # (2,S,F,Tf,2) -> (S,2,F,Tf,2) -> (S*2,F,Tf,2)
        p = pred.permute(1, 0, 2, 3, 4).contiguous()
        p = p.view(S * 2, p.shape[2], p.shape[3], p.shape[4])

        y = stft.istft_ri(p, length=length)  # (S*2, length)
        y = y.view(S, 2, length)
        return y

    def call_model(mix_ri: torch.Tensor, ref_ri: Optional[torch.Tensor]) -> torch.Tensor:
        # mix_ri: (N,F,Tf,2)
        N = mix_ri.shape[0]

        if ref_ri is None:
            ref_ri = torch.zeros_like(mix_ri)
            ref_valid = torch.zeros((N,), device=mix_ri.device, dtype=torch.bool)
        else:
            ref_valid = torch.ones((N,), device=mix_ri.device, dtype=torch.bool)

        if unet_tap is not None:
            unet_tap.reset()

        return model(mix_ri, ref_ri, ref_valid=ref_valid)

    def run_model_on_chunk(chunk: torch.Tensor, ref_chunk: Optional[torch.Tensor]) -> tuple[
        torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        chunk: (2,chunk_len) float32 on device
        ref_chunk: (2,chunk_len) float32 on device or None
        returns:
          y: (S,2,chunk_len) float32 on device
          dbg: dict with optional keys:
               - "unet_out": (S,2,chunk_len)
               - "unet_yccm": (S,2,chunk_len)
        """
        mix_ri = stft.stft_ri(chunk)  # (2,F,Tf,2)
        ref_ri = stft.stft_ri(ref_chunk) if (ref_chunk is not None) else None

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = call_model(mix_ri, ref_ri)  # (2,4,F,Tf,2) expected
        else:
            pred = call_model(mix_ri, ref_ri)

        y_time = pred_to_time(pred, length=chunk_len)

        dbg = None
        if unet_tap is not None and (unet_dump_what in ("out", "yccm", "both")):
            # determine how many heads U-Net has
            n_unet = 0
            if hasattr(model, "residual_unets"):
                n_unet = len(getattr(model, "residual_unets"))
            n_unet = int(max(0, n_unet))

            # build per-head tensors (time-domain)
            outs = []
            yccms = []

            for hi in range(n_unet):
                # U-Net output
                out_t = _pick_tensor(unet_tap.last_out.get(hi, None))
                out_wav = _ri_to_stereo(stft, out_t, length=chunk_len) if out_t is not None else None

                # U-Net 2nd input (expected y_ccm_i)
                inp = unet_tap.last_inp.get(hi, None)
                yccm_t = None
                if isinstance(inp, (tuple, list)) and len(inp) >= 2 and isinstance(inp[1], torch.Tensor):
                    yccm_t = inp[1]
                yccm_wav = _ri_to_stereo(stft, yccm_t, length=chunk_len) if yccm_t is not None else None

                if out_wav is None and yccm_wav is None:
                    continue

                # both are (2,L). Make them consistent.
                if out_wav is None:
                    out_wav = torch.zeros((2, chunk_len), device=device, dtype=torch.float32)
                if yccm_wav is None:
                    yccm_wav = torch.zeros((2, chunk_len), device=device, dtype=torch.float32)

                outs.append(out_wav)
                yccms.append(yccm_wav)

            if outs or yccms:
                dbg = {}
                if unet_dump_what in ("out", "both") and outs:
                    dbg["unet_out"] = torch.stack(outs, dim=0)  # (S_unet,2,L)
                if unet_dump_what in ("yccm", "both") and yccms:
                    dbg["unet_yccm"] = torch.stack(yccms, dim=0)  # (S_unet,2,L)

        return y_time, dbg

    # -----------------------
    # Mode 0: full inference (single pass, no windows)
    # -----------------------
    if stitch == "full":
        # x: (2,T) on device
        mix_ri = stft.stft_ri(x)  # (2,F,Tf,2)
        ref_ri = stft.stft_ri(ref) if (ref is not None) else None

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = call_model(mix_ri, ref_ri) # (2,4,F,Tf,2)
        else:
            pred = call_model(mix_ri, ref_ri)

        # pred: (2,4,F,Tf,2) -> time
        # (4,2,F,Tf,2) -> (8,F,Tf,2)
        S = pred.shape[1]

        p = pred.permute(1, 0, 2, 3, 4).contiguous()  # (S,2,F,Tf,2)
        p = p.view(S * 2, p.shape[2], p.shape[3], p.shape[4])

        y = stft.istft_ri(p, length=T)  # (S*2,T)
        y = y.view(S, 2, T).detach().cpu()

        names = ["bass", "drums", "music", "vocals"]
        if S == 5:
            names += ["ref"]

        stems = {names[i] if i < len(names) else f"head{i}": y[i] for i in range(S)}

        dbg_ret = None
        if unet_tap is not None and (unet_dump_what in ("out", "yccm", "both")):
            dbg_ret = {}

            # map unet heads to names (best effort)
            n_unet = len(getattr(model, "residual_unets")) if hasattr(model, "residual_unets") else 0
            unet_names = names[:n_unet]

            if unet_dump_what in ("out", "both"):
                tmp = {}
                for hi in range(n_unet):
                    out_t = _pick_tensor(unet_tap.last_out.get(hi, None))
                    out_wav = _ri_to_stereo(stft, out_t, length=T) if out_t is not None else None
                    if out_wav is not None:
                        tmp[unet_names[hi]] = out_wav.detach().cpu()
                if tmp:
                    dbg_ret["unet_out"] = tmp

            if unet_dump_what in ("yccm", "both"):
                tmp = {}
                for hi in range(n_unet):
                    inp = unet_tap.last_inp.get(hi, None)
                    yccm_t = None
                    if isinstance(inp, (tuple, list)) and len(inp) >= 2 and isinstance(inp[1], torch.Tensor):
                        yccm_t = inp[1]
                    yccm_wav = _ri_to_stereo(stft, yccm_t, length=T) if yccm_t is not None else None
                    if yccm_wav is not None:
                        tmp[unet_names[hi]] = yccm_wav.detach().cpu()
                if tmp:
                    dbg_ret["unet_yccm"] = tmp

        return stems, dbg_ret

    # -----------------------
    # Mode 1: classic OLA (fixed normalization: wsum += win)
    # -----------------------
    if stitch == "ola":
        hop_len = int(round(chunk_len * (1.0 - overlap)))
        hop_len = max(1, hop_len)

        # pad so that last frame fits
        n_chunks = 1 + max(0, (T - 1) // hop_len)
        total_len = (n_chunks - 1) * hop_len + chunk_len
        pad = max(0, total_len - T)
        if pad > 0:
            x_pad = F.pad(x, (0, pad))
            ref_pad = F.pad(ref, (0, pad)) if (ref is not None) else None
        else:
            x_pad = x
            ref_pad = ref


        T_pad = x_pad.shape[1]

        out = None
        wsum = torch.zeros((1, 1, T_pad), device=device, dtype=torch.float32)

        win = make_hann_ola(chunk_len, device=device)  # (L,)
        win_v = win.view(1, 1, -1)  # (1,1,L)

        for i in range(n_chunks):
            s = i * hop_len
            e = s + chunk_len
            chunk = x_pad[:, s:e]
            ref_chunk = ref_pad[:, s:e] if (ref_pad is not None) else None

            y, dbg = run_model_on_chunk(chunk, ref_chunk)  # y: (S,2,L), dbg optional

            if out is None:
                S = y.shape[0]
                out = torch.zeros((S, 2, T_pad), device=device, dtype=torch.float32)

                if unet_tap is not None:
                    # allocate debug buffers lazily after we know S_unet
                    pass

            out[:, :, s:e] += y * win_v
            wsum[:, :, s:e] += win_v
            if dbg is not None:
                # allocate debug buffers on first use
                if (dbg_out is None) and ("unet_out" in dbg):
                    Su = dbg["unet_out"].shape[0]
                    dbg_out = torch.zeros((Su, 2, T_pad), device=device, dtype=torch.float32)
                if (dbg_yccm is None) and ("unet_yccm" in dbg):
                    Su = dbg["unet_yccm"].shape[0]
                    dbg_yccm = torch.zeros((Su, 2, T_pad), device=device, dtype=torch.float32)

                if dbg_out is not None and ("unet_out" in dbg):
                    dbg_out[:, :, s:e] += dbg["unet_out"] * win_v
                if dbg_yccm is not None and ("unet_yccm" in dbg):
                    dbg_yccm[:, :, s:e] += dbg["unet_yccm"] * win_v


        out = out / wsum.clamp_min(1e-8)
        out = out[:, :, :T].detach().cpu()
        dbg_ret = None
        if (dbg_out is not None) or (dbg_yccm is not None):
            dbg_ret = {}
            if dbg_out is not None:
                dbg_out = (dbg_out / wsum.clamp_min(1e-8))[:, :, :T].detach().cpu()
                dbg_ret["unet_out"] = dbg_out
            if dbg_yccm is not None:
                dbg_yccm = (dbg_yccm / wsum.clamp_min(1e-8))[:, :, :T].detach().cpu()
                dbg_ret["unet_yccm"] = dbg_yccm


        names = ["bass", "drums", "music", "vocals"]
        if out.shape[0] == 5:
            names = names + ["ref"]  # 5-я голова

        stems = {names[i] if i < len(names) else f"head{i}": out[i] for i in range(out.shape[0])}

        if dbg_ret is not None:
            # convert dbg tensors -> dict[name]->(2,T)
            names_u = names[
                : (dbg_ret["unet_out"].shape[0] if "unet_out" in dbg_ret else dbg_ret["unet_yccm"].shape[0])]
            dbg_dict: Dict[str, torch.Tensor] = {}

            if "unet_out" in dbg_ret:
                for i, nm in enumerate(names_u):
                    dbg_dict[f"unet_out_{nm}"] = dbg_ret["unet_out"][i]
            if "unet_yccm" in dbg_ret:
                for i, nm in enumerate(names_u):
                    dbg_dict[f"unet_yccm_{nm}"] = dbg_ret["unet_yccm"][i]

            return stems, dbg_dict

        return stems, None

    # -----------------------
    # Mode 2: crop-stitching (take central 50-75% and stitch)
    # -----------------------
    keep_frac = float(keep_frac)
    if not (0.50 <= keep_frac <= 0.75):
        raise ValueError("--keep-frac must be in [0.50, 0.75] for crop mode")

    keep_len = int(round(chunk_len * keep_frac))
    keep_len = max(1, min(keep_len, chunk_len))

    trim_left = (chunk_len - keep_len) // 2
    trim_right = (chunk_len - keep_len) - trim_left  # may differ by 1 if odd

    # optional crossfade between kept regions
    xfade_len = int(round(sr * float(xfade_ms) / 1000.0))
    xfade_len = max(0, min(xfade_len, keep_len // 2))

    # effective hop in output timeline
    hop_out = keep_len - xfade_len
    hop_out = max(1, hop_out)

    # choose number of chunks so we fully cover T samples in output
    if T <= keep_len:
        n_chunks = 1
    else:
        n_chunks = 1 + math.ceil((T - keep_len) / hop_out)

    out_len = (n_chunks - 1) * hop_out + keep_len
    out = torch.zeros((4, 2, out_len), device=device, dtype=torch.float32)

    if xfade_len > 0:
        wsum = torch.zeros((1, 1, out_len), device=device, dtype=torch.float32)
        fade = torch.linspace(0.0, 1.0, xfade_len, device=device, dtype=torch.float32)
    else:
        wsum = None
        fade = None

    def get_chunk_by_start(src: torch.Tensor, start: int) -> torch.Tensor:
        """
        src: (2,T)
        start can be negative. Returns (2,chunk_len) padded with zeros if needed.
        """
        end = start + chunk_len
        left_pad = max(0, -start)
        right_pad = max(0, end - T)
        s0 = max(0, start)
        e0 = min(T, end)
        ch = src[:, s0:e0]
        if left_pad or right_pad:
            ch = F.pad(ch, (left_pad, right_pad))
        if ch.shape[1] != chunk_len:
            ch = F.pad(ch, (0, max(0, chunk_len - ch.shape[1])))
            ch = ch[:, :chunk_len]
        return ch

    for i in range(n_chunks):
        out_s = i * hop_out
        out_e = out_s + keep_len

        # align: kept center lands exactly at [out_s:out_e]
        in_s = out_s - trim_left  # may be negative
        chunk = get_chunk_by_start(x, in_s)
        ref_chunk = get_chunk_by_start(ref, in_s) if (ref is not None) else None

        y, dbg = run_model_on_chunk(chunk, ref_chunk)
        y_keep = y[:, :, trim_left:trim_left + keep_len]  # (4,2,keep_len)
        dbg_keep_out = None
        dbg_keep_yccm = None
        if dbg is not None:
            if "unet_out" in dbg:
                dbg_keep_out = dbg["unet_out"][:, :, trim_left:trim_left + keep_len]
            if "unet_yccm" in dbg:
                dbg_keep_yccm = dbg["unet_yccm"][:, :, trim_left:trim_left + keep_len]


        if xfade_len <= 0:
            out[:, :, out_s:out_e] = y_keep
            continue

        # build per-chunk weights:
        # - first chunk: no fade-in
        # - last chunk: no fade-out
        w = torch.ones((keep_len,), device=device, dtype=torch.float32)
        if i > 0:
            w[:xfade_len] = fade
        if i < n_chunks - 1:
            w[-xfade_len:] = torch.flip(fade, dims=[0])
        wv = w.view(1, 1, -1)

        out[:, :, out_s:out_e] += y_keep * wv
        wsum[:, :, out_s:out_e] += wv  # type: ignore[index]

    if wsum is not None:
        out = out / wsum.clamp_min(1e-8)

    out = out[:, :, :T].detach().cpu()
    names = ["bass", "drums", "music", "vocals"]
    return {names[i]: out[i] for i in range(4)}



def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", required=True, type=str, help="Path to phase_a_latest.pt (or phase_a_eXXX.pt)")
    ap.add_argument("--inp", required=True, type=str, help="Input audio file (wav/flac/...)")
    ap.add_argument("--ref", default="", type=str, help="(conditional) Reference audio file (same SR). If empty -> zeros.")
    ap.add_argument("--ref-gain-db", type=float, default=0.0, help="Optional gain (dB) applied to ref before STFT.")
    ap.add_argument("--outdir", default="out_phase_a", type=str)

    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--n-fft", type=int, default=1536)
    ap.add_argument("--hop", type=int, default=480)
    ap.add_argument("--win", type=int, default=1536)

    ap.add_argument("--chunk-sec", type=float, default=4.0, help="Window size in seconds")
    ap.add_argument("--overlap", type=float, default=0.5, help="Overlap ratio for OLA (0.. <1)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true", help="Enable autocast fp16 on CUDA (optional)")

    ap.add_argument("--format", choices=["float", "pcm16"], default="float")
    ap.add_argument("--write-mix", action="store_true", help="Also write mix.wav as a copy of input")
    ap.add_argument("--write-sum", action="store_true", help="Also write sum.wav = sum(6 heads)")
    ap.add_argument("--write-sum5", action="store_true", help="Also write sum5.wav = sum(first 5 heads)")

    ap.add_argument(
        "--stitch",
        choices=["ola", "crop", "full"],
        default="ola",
        help="Stitching mode: 'full' = single-pass on entire audio, 'ola' = overlap-add, 'crop' = central crop stitch",
    )
    ap.add_argument(
        "--keep-frac",
        type=float,
        default=0.6,
        help="(crop) Central fraction of chunk to keep. Must be in [0.50, 0.75].",
    )
    ap.add_argument(
        "--xfade-ms",
        type=float,
        default=0.0,
        help="(crop) Optional crossfade length in milliseconds between kept regions (0 = off).",
    )

    ap.add_argument("--dump-unet", action="store_true", help="Dump residual U-Net debug wavs (y_ccm input and/or U-Net output)")
    ap.add_argument("--dump-unet-what", choices=["out", "yccm", "both"], default="both", help="What to dump: U-Net output, y_ccm input, or both")


    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print("device:", device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load audio
    x = read_audio(args.inp, sr_expected=int(args.sr))  # (2,T)
    T = x.shape[1]
    print(f"[audio] {args.inp} | sr={args.sr} | samples={T} | sec={T/args.sr:.2f}")

    ref = None
    if args.ref:
        ref = read_audio(args.ref, sr_expected=int(args.sr))  # (2,Tr)
        ref = match_length(ref, T)
        g = db_to_lin(float(args.ref_gain_db))
        if g != 1.0:
            ref = ref * float(g)
        print(f"[ref] {args.ref} | gain_db={args.ref_gain_db}")


    # load model + stft
    model = load_model(args.ckpt, device=device, n_fft=int(args.n_fft), num_heads=4)
    stft = STFT(StftCfg(n_fft=int(args.n_fft), hop=int(args.hop), win=int(args.win))).to(device)

    # infer
    tap = None
    if args.dump_unet:
        tap = ResidualUnetTap(model)

    stems, dbg = infer_ola(
        model=model,
        stft=stft,
        x_stereo=x,
        ref_stereo=ref,
        sr=int(args.sr),
        chunk_sec=float(args.chunk_sec),
        overlap=float(args.overlap),
        device=device,
        use_amp=bool(args.amp),
        stitch=str(args.stitch),
        keep_frac=float(args.keep_frac),
        xfade_ms=float(args.xfade_ms),
        unet_tap=tap,
        unet_dump_what=str(args.dump_unet_what),
    )

    if tap is not None:
        tap.close()

    # write
    if args.write_mix:
        write_audio(outdir / "mix.wav", x.transpose(0, 1).numpy(), int(args.sr), fmt=args.format)

    for name, y in stems.items():
        y_np = y.transpose(0, 1).numpy()  # (T,2)
        write_audio(outdir / f"{name}.wav", y_np, int(args.sr), fmt=args.format)

    if dbg is not None:
        for name, y in dbg.items():
            y_np = y.transpose(0, 1).numpy()  # (T,2)
            write_audio(outdir / f"{name}.wav", y_np, int(args.sr), fmt=args.format)


    if args.write_sum or args.write_sum5:
        # sum heads in time domain (already comes from ISTFT)
        sum5 = stems["bass"] + stems["drums"] + stems["music"] + stems["vocals"]
        if args.write_sum:
            write_audio(outdir / "sum.wav", sum5.transpose(0, 1).numpy(), int(args.sr), fmt=args.format)

    # quick console stats
    def peak(t: torch.Tensor) -> float:
        return float(t.abs().max().item()) if t.numel() else 0.0

    print("[done] wrote:")
    for n in ["bass", "drums", "music", "vocals"]:
        print(f"  {n:7s} peak={peak(stems[n]):.4f}")
    if args.write_sum:
        print(f"  sum     peak={peak(sum5):.4f}")
    print(f"  outdir: {outdir}")


if __name__ == "__main__":
    main()
