import argparse
import numpy as np
import soundfile as sf
import torch


def load_stereo(path, sr_expected):
    x, sr = sf.read(path, dtype="float32", always_2d=True)  # (T,C)
    if sr != sr_expected:
        raise RuntimeError(f"SR mismatch: {path} got {sr}, expected {sr_expected}")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    else:
        x = x[:, :2]
    return torch.from_numpy(x).T.contiguous()  # (2,T)


def rms(x: torch.Tensor, eps=1e-12):
    return torch.sqrt(torch.mean(x * x) + eps)


def fit_ab(mix, ref, tgt):
    """
    Find scalars a,b per channel s.t. mix ≈ a*ref + b*tgt (least squares).
    mix/ref/tgt: (T,)
    """
    X = torch.stack([ref, tgt], dim=1)  # (T,2)
    # solve (X^T X) theta = X^T y
    XtX = X.T @ X
    Xty = X.T @ mix
    theta = torch.linalg.solve(XtX, Xty)  # (2,)
    return theta[0].item(), theta[1].item()


def proj_gain(x, ref, eps=1e-12):
    # best alpha in least squares: x ≈ alpha*ref
    return (x @ ref) / ((ref @ ref) + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--sr", type=int, default=48000)
    args = ap.parse_args()

    mix = load_stereo(args.mix, args.sr)
    ref = load_stereo(args.ref, args.sr)
    tgt = load_stereo(args.target, args.sr)

    T = min(mix.shape[1], ref.shape[1], tgt.shape[1])
    mix, ref, tgt = mix[:, :T], ref[:, :T], tgt[:, :T]

    print("RMS mix/ref/tgt (L,R):",
          f"{rms(mix[0]):.4f}/{rms(mix[1]):.4f}",
          f"{rms(ref[0]):.4f}/{rms(ref[1]):.4f}",
          f"{rms(tgt[0]):.4f}/{rms(tgt[1]):.4f}")

    # Check linear consistency: mix ≈ a*ref + b*tgt
    for ch, name in [(0, "L"), (1, "R")]:
        a, b = fit_ab(mix[ch], ref[ch], tgt[ch])
        recon = a * ref[ch] + b * tgt[ch]
        err = rms(mix[ch] - recon) / (rms(mix[ch]) + 1e-12)
        print(f"[{name}] fit mix≈a*ref+b*tgt: a={a:.4f} b={b:.4f} rel_err={err:.6f}")

    if args.pred:
        pred = load_stereo(args.pred, args.sr)[:, :T]
        for ch, name in [(0, "L"), (1, "R")]:
            # similarity to mix
            diff_mix = rms(pred[ch] - mix[ch]) / (rms(mix[ch]) + 1e-12)

            # how much ref is present in pred (projection)
            alpha = proj_gain(pred[ch], ref[ch])
            ref_part = alpha * ref[ch]
            ref_ratio = (rms(ref_part) / (rms(pred[ch]) + 1e-12)).item()

            print(f"[{name}] pred vs mix rel_rms_diff={diff_mix:.6f} | "
                  f"ref_leak_alpha={alpha:.4f} ref_energy_ratio={ref_ratio:.4f}")

    print("done.")


if __name__ == "__main__":
    main()
