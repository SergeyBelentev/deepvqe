import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Импортируй DeepVQE из твоего файла модели:
from deepvqe import DeepVQE


# -----------------------
# audio io
# -----------------------
def load_wav_mono(path: str, sr_expected: int) -> torch.Tensor:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != sr_expected:
        raise RuntimeError(f"SR mismatch for {path}: got {sr}, expected {sr_expected}")
    x = x.mean(axis=1)  # mono
    return torch.from_numpy(x)  # (T,)


def random_crop_same(signals: List[torch.Tensor], length: int) -> List[torch.Tensor]:
    max_len = max(s.numel() for s in signals)
    if max_len < length:
        pad = length - max_len
        signals = [F.pad(s, (0, pad)) for s in signals]
        max_len = length

    if max_len == length:
        start = 0
    else:
        start = torch.randint(0, max_len - length + 1, (1,)).item()

    out = []
    for s in signals:
        if s.numel() < start + length:
            s = F.pad(s, (0, start + length - s.numel()))
        out.append(s[start : start + length])
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

    s_target = (torch.sum(est_zm * ref_zm, dim=-1, keepdim=True) / (torch.sum(ref_zm ** 2, dim=-1, keepdim=True) + eps)) * ref_zm
    e_noise = est_zm - s_target

    ratio = (torch.sum(s_target ** 2, dim=-1) + eps) / (torch.sum(e_noise ** 2, dim=-1) + eps)
    si_sdr = 10.0 * torch.log10(ratio)
    return -si_sdr.mean()


def mag_l1_loss(est_ri: torch.Tensor, ref_ri: torch.Tensor) -> torch.Tensor:
    """
    est/ref: (B,F,T,2)
    """
    est_mag = torch.sqrt(est_ri[..., 0] ** 2 + est_ri[..., 1] ** 2 + 1e-12)
    ref_mag = torch.sqrt(ref_ri[..., 0] ** 2 + ref_ri[..., 1] ** 2 + 1e-12)
    return F.l1_loss(est_mag, ref_mag)


# -----------------------
# STFT helper
# -----------------------
@dataclass
class StftCfg:
    n_fft: int = 1536
    hop: int = 768
    win: int = 1536


class STFT(nn.Module):
    def __init__(self, cfg: StftCfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("window", torch.hann_window(cfg.win), persistent=False)

    def stft_ri(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T) -> (B,F,Tf,2)  where F=n_fft//2+1
        """
        w = self.window.to(x.device)
        X = torch.stft(
            x,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop,
            win_length=self.cfg.win,
            window=w,
            return_complex=True,
        )
        return torch.view_as_real(X)  # (B,F,Tf,2)

    def istft_ri(self, X_ri: torch.Tensor, length: int) -> torch.Tensor:
        """
        X_ri: (B,F,Tf,2) -> (B,T)
        """
        w = self.window.to(X_ri.device)
        X = torch.complex(X_ri[..., 0], X_ri[..., 1])
        y = torch.istft(
            X,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop,
            win_length=self.cfg.win,
            window=w,
            length=length,
        )
        return y


# -----------------------
# dataset
# -----------------------
class AecDataset(Dataset):
    """
    Manifest CSV/TSV with 3 columns:
      mix_path, ref_path, target_path
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

        mix = load_wav_mono(mix_p, self.sr)
        ref = load_wav_mono(ref_p, self.sr)
        tgt = load_wav_mono(tgt_p, self.sr)

        mix, ref, tgt = random_crop_same([mix, ref, tgt], length=self.seg_len)
        return mix, ref, tgt


def collate(batch):
    mix = torch.stack([b[0] for b in batch], dim=0)  # (B,T)
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
    ap.add_argument("--batch", type=int, default=2)          # 48k fullband: стартуй маленько
    ap.add_argument("--segment-sec", type=float, default=2.0) # 48k: начинай с 2с
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)    # Windows-friendly
    ap.add_argument("--grad-clip", type=float, default=5.0)

    # 48k + STFT
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--n-fft", type=int, default=1536)
    ap.add_argument("--hop", type=int, default=768)
    ap.add_argument("--win", type=int, default=1536)

    # align/model
    ap.add_argument("--delay-frames", type=int, default=80)
    ap.add_argument("--align-hidden", type=int, default=64)

    # losses
    ap.add_argument("--mag-loss", type=float, default=0.1)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # dataset/loader
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

    # model + stft
    model = DeepVQE(
        delay_frames=args.delay_frames,
        align_hidden=args.align_hidden,
        # важно: модель должна принимать n_fft и считать F5 динамически
        n_fft=args.n_fft,
    ).to(device)
    model.train()

    stft = STFT(StftCfg(n_fft=args.n_fft, hop=args.hop, win=args.win)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # perf flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for mix, ref, tgt in dl:
            mix = mix.to(device, non_blocking=True)
            ref = ref.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            mix_ri = stft.stft_ri(mix)  # (B,F,Tf,2)
            ref_ri = stft.stft_ri(ref)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                out_ri = model(mix_ri, ref_ri)  # (B,F,Tf,2)
                out_wav = stft.istft_ri(out_ri, length=mix.shape[-1])  # (B,T)

                loss_time = si_sdr_loss(out_wav, tgt)

                tgt_ri = stft.stft_ri(tgt)
                loss_mag = mag_l1_loss(out_ri, tgt_ri)

                loss = loss_time + args.mag_loss * loss_mag

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(opt)
            scaler.update()

            running += float(loss.detach().cpu())

        avg = running / max(1, len(dl))
        print(f"epoch {epoch:03d} | loss {avg:.4f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, str(Path(args.save_dir) / f"deepvqe_aec48k_e{epoch:03d}.pt"))

    print("done.")


if __name__ == "__main__":
    main()
