import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from deepvqe import DeepVQE


# -----------------------
# audio io (stereo)
# -----------------------
def load_wav_stereo(path: str, sr_expected: int) -> torch.Tensor:
    """
    Returns: (2,T) float32
    """
    x, sr = sf.read(path, dtype="float32", always_2d=True)  # (T,C)
    if sr != sr_expected:
        raise RuntimeError(f"SR mismatch for {path}: got {sr}, expected {sr_expected}")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    else:
        x = x[:, :2]
    return torch.from_numpy(x).transpose(0, 1).contiguous()  # (2,T)


def random_crop_same(signals: List[torch.Tensor], length: int) -> List[torch.Tensor]:
    """
    signals: list of tensors (..., T)
    Crops/pads all to the same last-dim length.
    """
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
    hop: int = 480   # 10ms @48k
    win: int = 1536  # 32ms @48k


class STFT(nn.Module):
    def __init__(self, cfg: StftCfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("window", torch.hann_window(cfg.win, dtype=torch.float32), persistent=False)

    def stft_ri(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T) float32 -> (B,F,Tf,2) float32
        """
        if x.dtype != torch.float32:
            raise RuntimeError(f"Model STFT expects float32 input, got {x.dtype}")
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
        """
        X_ri: (B,F,Tf,2) -> (B,T) float32
        """
        if X_ri.dtype != torch.float32:
            X_ri = X_ri.to(torch.float32)
        w = self.window.to(device=X_ri.device, dtype=torch.float32)
        X = torch.complex(X_ri[..., 0], X_ri[..., 1])
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
def si_sdr_loss(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    est/ref: (B,T)
    Returns: negative SI-SDR (minimize)
    """
    ref_zm = ref - ref.mean(dim=-1, keepdim=True)
    est_zm = est - est.mean(dim=-1, keepdim=True)

    s_target = (
        (torch.sum(est_zm * ref_zm, dim=-1, keepdim=True) / (torch.sum(ref_zm**2, dim=-1, keepdim=True) + eps))
        * ref_zm
    )
    e_noise = est_zm - s_target

    ratio = (torch.sum(s_target**2, dim=-1) + eps) / (torch.sum(e_noise**2, dim=-1) + eps)
    si_sdr = 10.0 * torch.log10(ratio)
    return -si_sdr.mean()


def mag_l1_loss(est_ri: torch.Tensor, ref_ri: torch.Tensor) -> torch.Tensor:
    """
    est/ref: (B,F,T,2)
    Magnitude-only (phase-invariant). Keep weight small.
    """
    est_ri = est_ri.float()
    ref_ri = ref_ri.float()
    est_mag = torch.sqrt(est_ri[..., 0] ** 2 + est_ri[..., 1] ** 2 + 1e-12)
    ref_mag = torch.sqrt(ref_ri[..., 0] ** 2 + ref_ri[..., 1] ** 2 + 1e-12)
    return F.l1_loss(est_mag, ref_mag)


class MRSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT loss on waveform:
      loss = mean_k( spectral_convergence_k + logmag_l1_k )

    x, y: (B,T) float32
    """
    def __init__(self, cfgs: List[Tuple[int, int, int]], eps: float = 1e-7):
        super().__init__()
        self.cfgs = cfgs
        self.eps = eps
        for i, (_, _, win) in enumerate(cfgs):
            self.register_buffer(f"window_{i}", torch.hann_window(win, dtype=torch.float32), persistent=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if y.dtype != torch.float32:
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


def _shift_ref_frames(x: torch.Tensor, shift: int) -> torch.Tensor:
    """
    x: (B,F,T,2). shift along T in STFT frames.
    shift > 0: delay (pad left with zeros)
    shift < 0: advance (pad right with zeros)
    """
    if shift == 0:
        return x
    B, Fq, T, C = x.shape
    if shift > 0:
        pad = x.new_zeros((B, Fq, shift, C))
        return torch.cat([pad, x[:, :, : T - shift, :]], dim=2)
    s = -shift
    pad = x.new_zeros((B, Fq, s, C))
    return torch.cat([x[:, :, s:, :], pad], dim=2)


def shiftinv_bg_loss(bg_ri: torch.Tensor, ref_ri: torch.Tensor, max_shift: int, tau: float = 0.5) -> torch.Tensor:
    """
    bg should match ref up to an unknown shift in [-max_shift, +max_shift] frames.
    Uses softmin to be differentiable.
    """
    bg = bg_ri.float()
    ref = ref_ri.float()

    losses = []
    for s in range(-max_shift, max_shift + 1):
        ref_s = _shift_ref_frames(ref, s)
        l = (bg - ref_s).abs().mean(dim=(1, 2, 3))  # (B,)
        losses.append(l)

    L = torch.stack(losses, dim=0)  # (S,B)
    best = -tau * torch.logsumexp(-L / tau, dim=0)
    return best.mean()


def shiftinv_ortho_loss(out_ri: torch.Tensor, ref_ri: torch.Tensor, max_shift: int, tau: float = 0.1, eps: float = 1e-8) -> torch.Tensor:
    """
    Penalize correlation between out and ref for ANY shift in [-max_shift, +max_shift].
    Works for +ref and -ref (phase flip) because it uses absolute dot.
    Uses softmax over shifts as a smooth max.
    """
    out = out_ri.float()
    ref = ref_ri.float()

    out_r, out_i = out[..., 0], out[..., 1]  # (B,F,T)
    out_e = (out_r * out_r + out_i * out_i).sum(dim=(1, 2)) + eps  # (B,)

    corrs = []
    for s in range(-max_shift, max_shift + 1):
        ref_s = _shift_ref_frames(ref, s)
        rr, ri = ref_s[..., 0], ref_s[..., 1]
        ref_e = (rr * rr + ri * ri).sum(dim=(1, 2)) + eps

        # Re(out * conj(ref)) = out_r*rr + out_i*ri
        dot = (out_r * rr + out_i * ri).sum(dim=(1, 2)).abs()
        corr = dot / torch.sqrt(out_e * ref_e)  # (B,)
        corrs.append(corr)

    C = torch.stack(corrs, dim=0)  # (S,B)
    w = torch.softmax(C / tau, dim=0)
    return (w * C).sum(dim=0).mean()


# -----------------------
# dataset
# -----------------------
class AecDataset(Dataset):
    """
    Manifest CSV/TSV: mix_path, ref_path, target_path
    """
    def __init__(self, manifest_path: str, sr: int, segment_sec: float):
        self.sr = sr
        self.seg_len = int(sr * segment_sec)
        self.items: List[Tuple[str, str, str]] = []

        with open(manifest_path, "r", newline="", encoding="utf-8") as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            reader = csv.reader(f, dialect)
            for row in reader:
                if not row:
                    continue
                if row[0].strip().startswith("#"):
                    continue
                if len(row) < 3:
                    raise RuntimeError(f"Bad row (need 3 cols): {row}")
                mix, ref, tgt = row[0].strip(), row[1].strip(), row[2].strip()
                self.items.append((mix, ref, tgt))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        mix_p, ref_p, tgt_p = self.items[idx]
        mix = load_wav_stereo(mix_p, self.sr)  # (2,T)
        ref = load_wav_stereo(ref_p, self.sr)
        tgt = load_wav_stereo(tgt_p, self.sr)
        mix, ref, tgt = random_crop_same([mix, ref, tgt], length=self.seg_len)
        return mix, ref, tgt


def collate(batch):
    mix = torch.stack([b[0] for b in batch], dim=0)  # (B,2,T)
    ref = torch.stack([b[1] for b in batch], dim=0)
    tgt = torch.stack([b[2] for b in batch], dim=0)
    return mix, ref, tgt


# -----------------------
# ref shift augmentation (time-domain, zero-pad)
# -----------------------
def apply_time_shift_1d(x: torch.Tensor, shift: int) -> torch.Tensor:
    """
    x: (T,)
    shift in samples:
      shift>0: delay (pad left)
      shift<0: advance (pad right)
    """
    T = x.shape[0]
    if shift == 0:
        return x
    if shift > 0:
        s = min(shift, T)
        return torch.cat([x.new_zeros((s,)), x[: T - s]], dim=0)
    s = min(-shift, T)
    return torch.cat([x[s:], x.new_zeros((s,))], dim=0)


def apply_ref_shift_batch(ref_f: torch.Tensor, max_shift_samples: int) -> torch.Tensor:
    """
    ref_f: (B,T) float32
    Applies independent random shift per item.
    """
    if max_shift_samples <= 0:
        return ref_f
    B, T = ref_f.shape
    out = []
    for b in range(B):
        sh = int(torch.randint(-max_shift_samples, max_shift_samples + 1, (1,)).item())
        out.append(apply_time_shift_1d(ref_f[b], sh))
    return torch.stack(out, dim=0)


# -----------------------
# train
# -----------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", required=True, help="CSV/TSV: mix_path,ref_path,target_path")
    ap.add_argument("--save-dir", default="ckpt_48k")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--save-every-epochs", type=int, default=5)

    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--segment-sec", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)

    # audio / stft (model input/output)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--n-fft", type=int, default=1536)
    ap.add_argument("--hop", type=int, default=480)
    ap.add_argument("--win", type=int, default=1536)

    # model alignment
    ap.add_argument("--delay-frames", type=int, default=25)
    ap.add_argument("--align-hidden", type=int, default=64)

    # shift-robust losses (in STFT frames)
    ap.add_argument("--shift-frames", type=int, default=None, help="default: =delay-frames")
    ap.add_argument("--bg-w", type=float, default=0.5)
    ap.add_argument("--leak-w", type=float, default=1.0)
    ap.add_argument("--bg-tau", type=float, default=0.5)
    ap.add_argument("--leak-tau", type=float, default=0.1)

    # optional: random shift of REF input (time-domain) to force robustness
    ap.add_argument("--ref-shift-ms", type=float, default=0.0, help="random ref time shift in ms (±), default 0")

    # spectral losses (keep small; they are phase-invariant)
    ap.add_argument("--mag-loss", type=float, default=0.1)
    ap.add_argument("--mrstft-weight", type=float, default=0.2)

    # amp
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")

    args = ap.parse_args()

    if args.shift_frames is None:
        args.shift_frames = int(args.delay_frames)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = AecDataset(args.manifest, sr=args.sr, segment_sec=args.segment_sec)

    drop_last = (len(ds) >= args.batch)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        drop_last=drop_last,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
    )

    print(f"dataset: {len(ds)} items | batch={args.batch} | drop_last={drop_last} | batches/epoch={len(dl)}")
    if len(dl) == 0:
        raise RuntimeError(
            f"DataLoader has 0 batches (len(ds)={len(ds)}, batch={args.batch}). "
            f"Use --batch 1 or add more data."
        )

    model = DeepVQE(
        delay_frames=args.delay_frames,
        align_hidden=args.align_hidden,
        n_fft=args.n_fft,
    ).to(device)
    model.train()

    # ask model to return (out, bg) if supported
    if hasattr(model, "set_return_bg"):
        model.set_return_bg(True)

    stft = STFT(StftCfg(n_fft=args.n_fft, hop=args.hop, win=args.win)).to(device)

    mr_cfgs = [
        (1024, 240, 1024),
        (2048, 480, 2048),
        (4096, 960, 4096),
    ]
    mrstft = MRSTFTLoss(mr_cfgs).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    # torch>=2.0 preferred AMP API (avoids deprecation warnings)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    max_ref_shift_samples = int(args.ref_shift_ms * 1e-3 * args.sr)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        running_si = 0.0
        running_bg = 0.0
        running_leak = 0.0

        opt.zero_grad(set_to_none=True)

        for mix, ref, tgt in dl:
            # mix/ref/tgt: (B,2,T)
            mix = mix.to(device, non_blocking=True)
            ref = ref.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            B, C, T = mix.shape
            assert C == 2

            # flatten stereo -> (B*2,T)
            mix_f = mix.reshape(B * C, T).to(torch.float32)
            ref_f = ref.reshape(B * C, T).to(torch.float32)
            tgt_f = tgt.reshape(B * C, T).to(torch.float32)

            # optional: random misalignment of REF input (forces robustness)
            if max_ref_shift_samples > 0:
                ref_f = apply_ref_shift_batch(ref_f, max_ref_shift_samples)

            # STFT in FP32 only
            with torch.amp.autocast(device_type="cuda", enabled=False):
                mix_ri = stft.stft_ri(mix_f)  # (B*2,F,Tf,2)
                ref_ri = stft.stft_ri(ref_f)
                tgt_ri = stft.stft_ri(tgt_f)

            # model forward (optional AMP)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                out = model(mix_ri, ref_ri)

            # support both signatures: out only OR (out,bg)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                out_ri, bg_ri = out
            else:
                out_ri, bg_ri = out, None

            # losses in FP32
            with torch.amp.autocast(device_type="cuda", enabled=False):
                out_wav = stft.istft_ri(out_ri.to(torch.float32), length=T)  # (B*2,T)

                loss_time = si_sdr_loss(out_wav, tgt_f)
                loss_mag1 = mag_l1_loss(out_ri, tgt_ri)

                loss_mr = mrstft(out_wav, tgt_f) if args.mrstft_weight > 0 else out_wav.new_tensor(0.0)

                # shift-robust constraints
                if bg_ri is None:
                    # if your model doesn't return bg, you can still train leakage loss
                    loss_bg = out_wav.new_tensor(0.0)
                else:
                    loss_bg = shiftinv_bg_loss(bg_ri, ref_ri, max_shift=args.shift_frames, tau=args.bg_tau)

                loss_leak = shiftinv_ortho_loss(out_ri, ref_ri, max_shift=args.shift_frames, tau=args.leak_tau)

                loss = (
                    loss_time
                    + args.mag_loss * loss_mag1
                    + args.mrstft_weight * loss_mr
                    + args.bg_w * loss_bg
                    + args.leak_w * loss_leak
                )

                loss = loss / max(1, args.grad_accum)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running += float(loss.detach().cpu()) * max(1, args.grad_accum)
            running_si += float((-loss_time).detach().cpu())
            running_bg += float(loss_bg.detach().cpu())
            running_leak += float(loss_leak.detach().cpu())

            global_step += 1

            if global_step % max(1, args.grad_accum) == 0:
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

        avg = running / max(1, len(dl))
        avg_si = running_si / max(1, len(dl))
        avg_bg = running_bg / max(1, len(dl))
        avg_leak = running_leak / max(1, len(dl))

        print(
            f"epoch {epoch:03d} | total {avg:.4f} | si_sdr {avg_si:.2f} dB | "
            f"bg {avg_bg:.4f} | leak {avg_leak:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": (scaler.state_dict() if scaler.is_enabled() else None),
            "args": vars(args),
            "mr_cfgs": mr_cfgs,
        }
        if epoch % args.save_every_epochs == 0:
            torch.save(ckpt, str(Path(args.save_dir) / f"deepvqe_aec48k_e{epoch:03d}.pt"))
        # also keep a rolling "latest"
        torch.save(ckpt, str(Path(args.save_dir) / "deepvqe_latest.pt"))

    print("done.")


if __name__ == "__main__":
    main()
