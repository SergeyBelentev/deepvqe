import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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
    Returns: (C,T) float32, C=2
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
    """
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

            X = torch.stft(
                x, n_fft=n_fft, hop_length=hop, win_length=win, window=w,
                return_complex=True
            )
            Y = torch.stft(
                y, n_fft=n_fft, hop_length=hop, win_length=win, window=w,
                return_complex=True
            )

            Xmag = torch.abs(X) + self.eps
            Ymag = torch.abs(Y) + self.eps

            # Spectral Convergence: ||Y - X|| / ||Y||
            diff = (Ymag - Xmag).reshape(x.shape[0], -1)
            ref = Ymag.reshape(x.shape[0], -1)
            sc = (torch.linalg.vector_norm(diff, dim=1) / (torch.linalg.vector_norm(ref, dim=1) + self.eps)).mean()

            # Log-magnitude L1
            log_l1 = F.l1_loss(torch.log(Xmag), torch.log(Ymag))

            total = total + (sc + log_l1)

        return total / max(1, len(self.cfgs))


# -----------------------
# STFT helper for model IO (fp32 only)
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
# dataset
# -----------------------
class AecDataset(Dataset):
    """
    Manifest CSV/TSV: mix_path, ref_path, target_path
    All wavs stereo (mono auto-duplicated to stereo).
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
# train
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV/TSV: mix_path,ref_path,target_path")
    ap.add_argument("--save-dir", default="ckpt_48k")
    ap.add_argument("--epochs", type=int, default=50)

    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--segment-sec", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--grad-accum", type=int, default=1)

    # audio / stft (model input/output)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--n-fft", type=int, default=1536)
    ap.add_argument("--hop", type=int, default=480)   # 10ms @48k
    ap.add_argument("--win", type=int, default=1536)

    # align/model
    ap.add_argument("--delay-frames", type=int, default=25)  # ~250ms @ hop=10ms
    ap.add_argument("--align-hidden", type=int, default=64)

    # losses
    ap.add_argument("--mag-loss", type=float, default=0.3)          # one-res STFT mag loss (model-domain)
    ap.add_argument("--mrstft-weight", type=float, default=0.5)     # multi-res waveform STFT loss weight

    # amp
    ap.add_argument("--amp", action="store_true", help="enable autocast for model forward")
    ap.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = AecDataset(args.manifest, sr=args.sr, segment_sec=args.segment_sec)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
    )

    model = DeepVQE(
        delay_frames=args.delay_frames,
        align_hidden=args.align_hidden,
        n_fft=args.n_fft,
    ).to(device)
    model.train()

    stft = STFT(StftCfg(n_fft=args.n_fft, hop=args.hop, win=args.win)).to(device)

    # Multi-res configs (48k): ~5ms / 10ms / 20ms hops with growing FFT
    mr_cfgs = [
        (1024, 240, 1024),
        (2048, 480, 2048),
        (4096, 960, 4096),
    ]
    mrstft = MRSTFTLoss(mr_cfgs).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    step = 0
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        opt.zero_grad(set_to_none=True)

        for mix, ref, tgt in dl:
            mix = mix.to(device, non_blocking=True)  # (B,2,T)
            ref = ref.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            B, C, T = mix.shape
            assert C == 2

            # flatten stereo -> (B*2,T)
            mix_f = mix.reshape(B * C, T).to(torch.float32)
            ref_f = ref.reshape(B * C, T).to(torch.float32)
            tgt_f = tgt.reshape(B * C, T).to(torch.float32)

            # model IO STFT in FP32 only
            with torch.cuda.amp.autocast(enabled=False):
                mix_ri = stft.stft_ri(mix_f)
                ref_ri = stft.stft_ri(ref_f)
                tgt_ri = stft.stft_ri(tgt_f)

            # model forward (optional AMP)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                out_ri = model(mix_ri, ref_ri)

            # ISTFT + losses in FP32
            with torch.cuda.amp.autocast(enabled=False):
                out_wav = stft.istft_ri(out_ri.to(torch.float32), length=T)

                loss_time = si_sdr_loss(out_wav, tgt_f)
                loss_mag1 = mag_l1_loss(out_ri.to(torch.float32), tgt_ri)

                if args.mrstft_weight > 0:
                    loss_mr = mrstft(out_wav, tgt_f)
                else:
                    loss_mr = out_wav.new_tensor(0.0)

                loss = loss_time + args.mag_loss * loss_mag1 + args.mrstft_weight * loss_mr
                loss = loss / max(1, args.grad_accum)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running += float(loss.detach().cpu()) * max(1, args.grad_accum)
            step += 1

            if step % max(1, args.grad_accum) == 0:
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
        print(f"epoch {epoch:03d} | loss {avg:.4f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": (scaler.state_dict() if scaler.is_enabled() else None),
            "args": vars(args),
            "mr_cfgs": mr_cfgs,
        }
        torch.save(ckpt, str(Path(args.save_dir) / f"deepvqe_aec48k_e{epoch:03d}.pt"))

    print("done.")


if __name__ == "__main__":
    main()
