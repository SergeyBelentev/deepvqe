from __future__ import annotations

import argparse
from dataclasses import replace

import torch

from dub_separator import BandSpec, DubSeparator, DubSeparatorConfig


def _count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _tensor_stats(x: torch.Tensor) -> str:
    xr = x.float()
    finite = torch.isfinite(xr).all().item()
    return (
        f"shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}, "
        f"finite={finite}, mean={xr.mean().item():.6f}, std={xr.std().item():.6f}, "
        f"min={xr.min().item():.6f}, max={xr.max().item():.6f}"
    )


def build_tiny_config() -> DubSeparatorConfig:
    base = DubSeparatorConfig()
    tiny_bands = (
        BandSpec(name="band0", f_min_hz=0.0, f_max_hz=8000.0, num_tokens=2, encoder_profile="band0"),
        BandSpec(name="band1", f_min_hz=8000.0, f_max_hz=16000.0, num_tokens=2, encoder_profile="band2"),
        BandSpec(name="band2", f_min_hz=16000.0, f_max_hz=24000.0, num_tokens=2, encoder_profile="band4plus"),
    )
    return replace(
        base,
        encoder_channels=(8, 16, 24, 32, 32),
        trunk_dim=32,
        head_dim=16,
        num_trunk_layers=1,
        trunk_num_heads=4,
        axial_num_heads=4,
        detok_num_heads=4,
        stage_tokenizer_heads=4,
        band_tokenizer_heads=4,
        bands=tiny_bands,
    )



def make_mock_waveforms(
    *,
    batch_size: int,
    sample_rate: int,
    seconds: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_samples = int(sample_rate * seconds)
    t = torch.linspace(0.0, seconds, num_samples, device=device, dtype=dtype)
    t = t.unsqueeze(0).unsqueeze(0)

    ref_l = 0.20 * torch.sin(2.0 * torch.pi * 220.0 * t)
    ref_r = 0.18 * torch.sin(2.0 * torch.pi * 224.0 * t + 0.05)
    ref_l = ref_l + 0.08 * torch.sin(2.0 * torch.pi * 660.0 * t)
    ref_r = ref_r + 0.07 * torch.sin(2.0 * torch.pi * 670.0 * t + 0.10)

    env = torch.exp(-1.8 * t)
    ambience_l = 0.03 * torch.randn_like(ref_l) * env
    ambience_r = 0.03 * torch.randn_like(ref_r) * env
    ref = torch.cat([ref_l + ambience_l, ref_r + ambience_r], dim=1)

    dub_env = 0.5 * (1.0 + torch.sin(2.0 * torch.pi * 2.0 * t))
    dub_l = 0.22 * torch.sin(2.0 * torch.pi * 170.0 * t + 0.20) * dub_env
    dub_r = 0.20 * torch.sin(2.0 * torch.pi * 176.0 * t + 0.28) * dub_env
    dub_l = dub_l + 0.05 * torch.sin(2.0 * torch.pi * 2100.0 * t) * dub_env
    dub_r = dub_r + 0.05 * torch.sin(2.0 * torch.pi * 2050.0 * t + 0.15) * dub_env
    dub = torch.cat([dub_l, dub_r], dim=1)

    mix = ref.clone()
    mix[:, 0] = mix[:, 0] * 0.95
    mix[:, 1] = mix[:, 1] * 1.05
    mix = mix + dub

    if batch_size > 1:
        mix = mix.repeat(batch_size, 1, 1) + 0.005 * torch.randn(batch_size, 2, num_samples, device=device, dtype=dtype)
        ref = ref.repeat(batch_size, 1, 1) + 0.005 * torch.randn(batch_size, 2, num_samples, device=device, dtype=dtype)

    return mix.clamp(-1.0, 1.0), ref.clamp(-1.0, 1.0)



def print_output_summary(outputs: dict[str, torch.Tensor | list[torch.Tensor]]) -> None:
    print("\n=== outputs ===")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key:>18}: {_tensor_stats(value)}")
        elif isinstance(value, list):
            print(f"{key:>18}: list(len={len(value)})")
        else:
            print(f"{key:>18}: {type(value).__name__}")

    est = outputs["estimate_waveform"]
    mask = outputs["mask"]
    gate = outputs["crm_gate"]
    print("\n=== quick view ===")
    print("estimate first channel, first 12 samples:")
    print(est[0, 0, :12].detach().cpu())
    print("\nmask magnitude stats:")
    print(_tensor_stats(mask.abs()))
    print("\ncrm gate first frame, first 12 bins:")
    print(gate[0, 0, 0, :12].detach().cpu())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check for DubSeparator")
    parser.add_argument("--full", action="store_true", help="Run full default config")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seconds", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg = DubSeparatorConfig() if args.full else build_tiny_config()
    model = DubSeparator(cfg).to(device)
    model.eval()

    total_params, trainable_params = _count_params(model)
    print("=== model ===")
    print(f"device          : {device}")
    print(f"config          : {'full' if args.full else 'tiny'}")
    print(f"bands           : {[(b.name, b.num_tokens) for b in cfg.bands]}")
    print(f"n_fft / hop     : {cfg.n_fft} / {cfg.hop_length}")
    print(f"trunk layers    : {cfg.num_trunk_layers}")
    print(f"total tokens    : {cfg.total_tokens}")
    print(f"encoder_channels: {cfg.encoder_channels}")
    print(f"trunk_dim       : {cfg.trunk_dim}")
    print(f"head_dim        : {cfg.head_dim}")
    print(f"params total    : {total_params:,}")
    print(f"params trainable: {trainable_params:,}")

    mix, ref = make_mock_waveforms(
        batch_size=args.batch_size,
        sample_rate=cfg.sample_rate,
        seconds=args.seconds,
        device=device,
        dtype=torch.float32,
    )

    print("\n=== inputs ===")
    print(f"mix: {_tensor_stats(mix)}")
    print(f"ref: {_tensor_stats(ref)}")

    outputs = model(mix, ref)
    print_output_summary(outputs)

    est = outputs["estimate_waveform"]
    print("\n=== sanity assertions ===")
    print(f"estimate shape matches input: {est.shape == mix.shape}")
    print(f"estimate finite            : {torch.isfinite(est).all().item()}")
    print(f"estimate differs from mix  : {(est - mix).abs().mean().item():.6f} mean abs diff")


if __name__ == "__main__":
    main()
