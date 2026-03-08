
from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


def _load_python_module(module_path: str, module_name: str = "loaded_model_module"):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def convert_in_proj_to_qkv(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    new_sd: OrderedDict[str, torch.Tensor] = OrderedDict()

    for key, value in state_dict.items():
        if key.endswith(".in_proj_weight"):
            prefix = key[:-len(".in_proj_weight")]
            if value.ndim != 2:
                raise ValueError(f"Expected 2D in_proj_weight for {key}, got shape={tuple(value.shape)}")
            if value.shape[0] % 3 != 0:
                raise ValueError(f"Expected first dim divisible by 3 for {key}, got shape={tuple(value.shape)}")
            dim = value.shape[0] // 3
            new_sd[f"{prefix}.q_proj.weight"] = value[:dim].clone()
            new_sd[f"{prefix}.k_proj.weight"] = value[dim:2 * dim].clone()
            new_sd[f"{prefix}.v_proj.weight"] = value[2 * dim:3 * dim].clone()
            continue

        if key.endswith(".in_proj_bias"):
            prefix = key[:-len(".in_proj_bias")]
            if value.ndim != 1:
                raise ValueError(f"Expected 1D in_proj_bias for {key}, got shape={tuple(value.shape)}")
            if value.shape[0] % 3 != 0:
                raise ValueError(f"Expected dim divisible by 3 for {key}, got shape={tuple(value.shape)}")
            dim = value.shape[0] // 3
            new_sd[f"{prefix}.q_proj.bias"] = value[:dim].clone()
            new_sd[f"{prefix}.k_proj.bias"] = value[dim:2 * dim].clone()
            new_sd[f"{prefix}.v_proj.bias"] = value[2 * dim:3 * dim].clone()
            continue

        new_sd[key] = value

    return new_sd


def _normalize_model_cfg(module, cfg_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Rebuild nested config objects expected by the target DubSeparatorConfig.
    Most importantly:
      bands: list[dict] -> tuple[BandSpec]
    """
    cfg = dict(cfg_payload)

    # Rebuild BandSpec entries if needed.
    band_spec_cls = getattr(module, "BandSpec", None)
    bands = cfg.get("bands")
    if band_spec_cls is not None and bands is not None:
        rebuilt_bands = []
        for b in bands:
            if isinstance(b, dict):
                rebuilt_bands.append(band_spec_cls(**b))
            else:
                rebuilt_bands.append(b)
        cfg["bands"] = tuple(rebuilt_bands)

    # Normalize tuple-like fields if they were serialized as lists.
    for key in ("encoder_channels",):
        if key in cfg and isinstance(cfg[key], list):
            cfg[key] = tuple(cfg[key])

    return cfg


def build_model_for_validation(model_module_path: str, ckpt: dict[str, Any]):
    module = _load_python_module(model_module_path, module_name="dub_separator_target")
    if not hasattr(module, "DubSeparator") or not hasattr(module, "DubSeparatorConfig"):
        raise RuntimeError("Target module must define DubSeparator and DubSeparatorConfig")

    cfg_payload = ckpt.get("model_cfg")
    if cfg_payload is None:
        raise RuntimeError("Checkpoint has no 'model_cfg'; cannot validate automatically")

    cfg_cls = module.DubSeparatorConfig
    model_cls = module.DubSeparator

    if isinstance(cfg_payload, dict):
        cfg_payload = _normalize_model_cfg(module, cfg_payload)
        cfg = cfg_cls(**cfg_payload)
    else:
        raw = vars(cfg_payload)
        raw = _normalize_model_cfg(module, raw)
        cfg = cfg_cls(**raw)

    model = model_cls(cfg)
    return model


def maybe_strip_optimizer(ckpt: dict[str, Any], strip_optimizer: bool) -> None:
    if strip_optimizer:
        ckpt.pop("optimizer", None)
        ckpt.pop("scaler", None)
        ckpt.pop("scheduler", None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert old nn.MultiheadAttention checkpoints to SDPA q_proj/k_proj/v_proj layout."
    )
    parser.add_argument("--src", required=True, help="Source checkpoint path")
    parser.add_argument("--dst", required=True, help="Destination checkpoint path")
    parser.add_argument(
        "--model-file",
        default=None,
        help="Path to the target dub_separator Python file for strict validation after conversion",
    )
    parser.add_argument(
        "--strip-optimizer",
        action="store_true",
        help="Remove optimizer/scaler/scheduler state from the converted checkpoint",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip strict model.load_state_dict validation even if --model-file is provided",
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise RuntimeError("Checkpoint has no 'model' key")

    ckpt["model"] = convert_in_proj_to_qkv(ckpt["model"])
    maybe_strip_optimizer(ckpt, args.strip_optimizer)

    if args.model_file and not args.no_validate:
        model = build_model_for_validation(args.model_file, ckpt)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Validation failed after conversion.\n"
                f"Missing keys: {missing}\n"
                f"Unexpected keys: {unexpected}"
            )
        model.load_state_dict(ckpt["model"], strict=True)
        print("Validation: strict=True load_state_dict passed.")

    torch.save(ckpt, dst)
    print(f"Saved converted checkpoint: {dst}")


if __name__ == "__main__":
    main()
