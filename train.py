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
        out.append(s[..., start : start + length])
    return out


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
    def __init__(self, cfgs: List[Tuple[int, int, int]], eps: float = 1e-7):
        super().__init__()
        self.cfgs = cfgs
        self.eps = eps
        for i, (_, _, win) in enumerate(cfgs):
            self.register_buffer(f"window_{i}", torch.hann_window(win, dtype=torch.float32), persistent=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.float()
        y = y.float()
        total = 0.0
        for i, (n_fft, hop, win) in enumerate(self.cfgs):
            w = getattr(self, f"window_{i}").to(device=x.device, dtype=torch.float32)
            X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=w, return_complex=True)
            Y = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, window=w, return_complex=True)
            Xmag = torch.abs(X) + self.eps
            Ymag = torch.abs(Y) + self.eps

            diff = (Ymag - Xmag).reshape(x.shape[0], -1)
            ref = Ymag.reshape(x.shape[0], -1)
            sc = (torch.linalg.vector_norm(diff, dim=1) / (torch.linalg.vector_norm(ref, dim=1) + self.eps)).mean()
            log_l1 = F.l1_loss(torch.log(Xmag), torch.log(Ymag))
            total = total + (sc + log_l1)
        return total / max(1, len(self.cfgs))


# -----------------------
# dataset
# -----------------------
class AecDataset(Dataset):
    # CSV: mix_path, ref_path, target_path (target_path can be 'None' -> zero target)
    def __init__(self, manifest_path: str, sr: int, segment_sec: float):
        self.manifest_path = manifest_path
        self.sr = sr
        self.seg_len = int(sr * segment_sec)
        self.items: List[Tuple[str, str, str]] = []
        self._load_manifest_data()

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

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        mix_p, ref_p, tgt_p = self.items[idx]
        mix = load_wav_stereo(mix_p, self.sr)  # (2,T)
        ref = load_wav_stereo(ref_p, self.sr)  # (2,T)

        if tgt_p == "None":
            max_len = max(mix.shape[-1], ref.shape[-1])
            tgt = mix.new_zeros((2, max_len))
        else:
            tgt = load_wav_stereo(tgt_p, self.sr)

        mix, ref, tgt = random_crop_same([mix, ref, tgt], length=self.seg_len)
        return mix, ref, tgt


def collate(batch):
    mix = torch.stack([b[0] for b in batch], dim=0)  # (B,2,T)
    ref = torch.stack([b[1] for b in batch], dim=0)
    tgt = torch.stack([b[2] for b in batch], dim=0)
    return mix, ref, tgt


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
    keys = ["sr", "n_fft", "hop", "win", "delay_frames", "align_hidden"]
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


# -----------------------
# train
# -----------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", required=True, help="CSV: mix_path,ref_path,target_path (target_path can be 'None')")
    ap.add_argument("--save-dir", default="ckpt_48k")
    ap.add_argument("--epochs", type=int, default=50, help="TOTAL epochs to train to (if resuming: must be > ckpt epoch)")
    ap.add_argument("--save-every-epochs", type=int, default=1)

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

    # model alignment
    ap.add_argument("--delay-frames", type=int, default=25)
    ap.add_argument("--align-hidden", type=int, default=64)

    # augment: random ref shift, but quantized to hop (frame-aligned)
    ap.add_argument("--ref-shift-ms", type=float, default=0.0, help="±ms, quantized to hop")

    # loss weights
    ap.add_argument("--w-out-l1", type=float, default=1.0)
    ap.add_argument("--w-bg-l1", type=float, default=1.0)
    ap.add_argument("--w-bg-stft", type=float, default=0.2)
    ap.add_argument("--w-mrstft", type=float, default=0.2)
    ap.add_argument("--w-leak", type=float, default=0.5)

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
    ds = AecDataset(args.manifest, sr=args.sr, segment_sec=args.segment_sec)
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
    model = DeepVQE(n_fft=args.n_fft, delay_frames=args.delay_frames, align_hidden=args.align_hidden).to(device)
    model.train()
    if hasattr(model, "set_return_bg"):
        model.set_return_bg(True)

    # --- stft / loss ---
    stft = STFT(StftCfg(n_fft=args.n_fft, hop=args.hop, win=args.win)).to(device)

    mrstft = MRSTFTLoss([
        (1024, 240, 1024),
        (2048, 480, 2048),
        (4096, 960, 4096),
    ]).to(device)

    # --- opt / amp ---
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # autocast cache: default OFF (helps VRAM stability)
    autocast_cache = bool(args.amp_cache)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ref shift quantization to hop
    max_shift_frames = int(round((args.ref_shift_ms * 1e-3 * args.sr) / args.hop))
    max_shift_frames = min(max_shift_frames, max(0, args.delay_frames - 1))

    # --- resume state ---
    start_epoch = 1
    micro_step = 0  # counts minibatches; used for grad_accum phase continuity

    if args.resume is not None:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
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
        for mix, ref, tgt in pbar:
            mix = mix.to(device, non_blocking=True)  # (B,2,T)
            ref = ref.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            B, C, T = mix.shape
            mix_f = mix.reshape(B * C, T).float()
            ref_f = ref.reshape(B * C, T).float()
            tgt_f = tgt.reshape(B * C, T).float()

            # augment: shift REF input only (simulate unknown delay), quantized to hop
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

            # STFT in fp32
            with torch.amp.autocast(device_type="cuda", enabled=False):
                mix_ri = stft.stft_ri(mix_f)  # (B*2,F,Tf,2)
                ref_ri = stft.stft_ri(ref_f)
                tgt_ri = stft.stft_ri(tgt_f)
                bg_true_ri = mix_ri - tgt_ri
                bg_true_wav = mix_f - tgt_f

            # forward (optional AMP)
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

            # optimizer step every grad_accum micro-steps
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

        ckpt_out = {
            "epoch": epoch,
            "micro_step": micro_step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": (scaler.state_dict() if scaler.is_enabled() else None),
            "args": vars(args),
            "rng_state": _get_rng_state(),
        }

        if epoch % args.save_every_epochs == 0:
            torch.save(ckpt_out, str(Path(args.save_dir) / f"deepvqe_aec48k_e{epoch:03d}.pt"))
        torch.save(ckpt_out, str(Path(args.save_dir) / "deepvqe_latest.pt"))

    print("done.")


if __name__ == "__main__":
    main()
