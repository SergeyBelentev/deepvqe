# train.py
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
        # x: (B,T) fp32 -> (B,F,Tf,2) fp32
        w = self.window.to(device=x.device, dtype=torch.float32)
        X = torch.stft(
            x, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop, win_length=self.cfg.win,
            window=w, return_complex=True
        )
        return torch.view_as_real(X).to(torch.float32)

    def istft_ri(self, X_ri: torch.Tensor, length: int) -> torch.Tensor:
        w = self.window.to(device=X_ri.device, dtype=torch.float32)
        X = torch.complex(X_ri[..., 0].float(), X_ri[..., 1].float())
        y = torch.istft(
            X, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop, win_length=self.cfg.win,
            window=w, length=length
        )
        return y.to(torch.float32)


# -----------------------
# losses
# -----------------------
def corr_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x,y: (B,T)
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    num = (x * y).sum(dim=-1).abs()
    den = torch.sqrt((x * x).sum(dim=-1) * (y * y).sum(dim=-1) + eps)
    return (num / (den + eps)).mean()


def complex_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: (B,F,T,2)
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

    def _load_manifest_data(self):
        with open(self.manifest_path, "r", newline="", encoding="utf-8") as f:
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
                self.items.append((row[0].strip(), row[1].strip(), row[2].strip()))

    # CSV/TSV: mix_path, ref_path, target_path
    def __init__(self, manifest_path: str, sr: int, segment_sec: float):
        self.manifest_path = manifest_path
        self.sr = sr
        self.seg_len = int(sr * segment_sec)
        self.items: List[Tuple[str, str, str]] = []
        self._load_manifest_data()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        mix_p, ref_p, tgt_p = self.items[idx]
        mix = load_wav_stereo(mix_p, self.sr)  # (2,T)
        ref = load_wav_stereo(ref_p, self.sr)

        if tgt_p == 'None':
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
# train
# -----------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", required=True, help="CSV/TSV: mix_path,ref_path,target_path")
    ap.add_argument("--save-dir", default="ckpt_48k")
    ap.add_argument("--epochs", type=int, default=50)
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

    # augment: random ref shift, but quantized to hop (so it’s frame-aligned)
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

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = AecDataset(args.manifest, sr=args.sr, segment_sec=args.segment_sec)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        drop_last=(len(ds) >= args.batch),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
    )
    print(f"dataset: {len(ds)} items | batch={args.batch} | batches/epoch={len(dl)}")
    if len(dl) == 0:
        raise RuntimeError("DataLoader has 0 batches. Use --batch 1 or add more data.")

    model = DeepVQE(n_fft=args.n_fft, delay_frames=args.delay_frames, align_hidden=args.align_hidden).to(device)
    model.train()
    if hasattr(model, "set_return_bg"):
        model.set_return_bg(True)

    stft = STFT(StftCfg(n_fft=args.n_fft, hop=args.hop, win=args.win)).to(device)

    mrstft = MRSTFTLoss([
        (1024, 240, 1024),
        (2048, 480, 2048),
        (4096, 960, 4096),
    ]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ref shift quantization to hop
    max_shift_frames = int(round((args.ref_shift_ms * 1e-3 * args.sr) / args.hop))
    max_shift_frames = min(max_shift_frames, max(0, args.delay_frames - 1))

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)

        run_loss = 0.0
        for mix, ref, tgt in tqdm(dl, desc=f'Epoch {epoch}'):
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
                    ref_f = torch.cat([ref_f.new_zeros((ref_f.shape[0], shift_samp)), ref_f[:, : T - shift_samp]], dim=1)
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
            with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
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

                # leak: out should be decorrelated from background (true bg in mic)
                loss_leak = corr_loss(out_wav, bg_true_wav)

                loss = (
                    args.w_out_l1 * loss_out
                    + args.w_bg_l1 * loss_bg
                    + args.w_bg_stft * loss_bg_stft
                    + args.w_mrstft * loss_mr
                    + args.w_leak * loss_leak
                ) / max(1, args.grad_accum)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

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

            run_loss += float(loss.detach().cpu()) * max(1, args.grad_accum)

        avg = run_loss / max(1, len(dl))
        print(f"epoch {epoch:03d} | total {avg:.6f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": (scaler.state_dict() if scaler.is_enabled() else None),
            "args": vars(args),
        }

        # save every epoch by default
        if epoch % args.save_every_epochs == 0:
            torch.save(ckpt, str(Path(args.save_dir) / f"deepvqe_aec48k_e{epoch:03d}.pt"))
        torch.save(ckpt, str(Path(args.save_dir) / "deepvqe_latest.pt"))

    print("done.")


if __name__ == "__main__":
    main()
