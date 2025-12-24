import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import diffusers


def parse_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str.lower())
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return dtype


def resolve_base_model_dir(model_dir: Path, base_model_dir: str | None) -> str:
    if base_model_dir:
        return base_model_dir

    model_index_path = model_dir / "model_index.json"
    if not model_index_path.exists():
        raise FileNotFoundError(
            f"model_index.json not found in {model_dir}. Provide --base_model_dir explicitly."
        )

    with model_index_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    for key in ("base_model", "diffusers_path", "_name_or_path"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value

    raise ValueError(
        "Unable to infer base model path from model_index.json. "
        "Provide --base_model_dir explicitly."
    )


def load_transformer_state(model_dir: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    transformer = diffusers.SD3Transformer2DModel.from_pretrained(
        model_dir,
        subfolder="transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        device_map=None,
    )
    return {k: v.detach().cpu().float() for k, v in transformer.state_dict().items()}


def normalize_ema_state(ema_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("transformer.") for k in ema_state.keys()):
        ema_state = {k[len("transformer."):]: v for k, v in ema_state.items() if k.startswith("transformer.")}
    return {k: v.detach().cpu().float() for k, v in ema_state.items()}


def compare_state_dicts(
    base_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    name: str,
    atol: float = 1e-6,
) -> bool:
    shared_keys = sorted(set(base_state) & set(target_state))
    if not shared_keys:
        raise ValueError(f"No shared parameters found when comparing {name} weights.")

    missing_in_target = sorted(set(base_state) - set(target_state))
    missing_in_base = sorted(set(target_state) - set(base_state))
    if missing_in_target:
        print(f"[WARN] {name} is missing {len(missing_in_target)} keys found in base model.")
    if missing_in_base:
        print(f"[WARN] {name} has {len(missing_in_base)} extra keys not in base model.")

    max_abs_diff = 0.0
    changed_keys = 0
    for key in shared_keys:
        diff = (target_state[key] - base_state[key]).abs().max().item()
        if diff > atol:
            changed_keys += 1
        max_abs_diff = max(max_abs_diff, diff)

    total_keys = len(shared_keys)
    print(
        f"[CHECK] {name}: {changed_keys}/{total_keys} parameters differ "
        f"(max_abs_diff={max_abs_diff:.6f}, atol={atol})."
    )
    return changed_keys > 0


def main():
    parser = argparse.ArgumentParser(description="Validate sd3_light weights by comparing to base init.")
    parser.add_argument("--model_dir", required=True, help="Path to saved sd3_light model directory.")
    parser.add_argument("--base_model_dir", default=None, help="Path to base diffusers model directory.")
    parser.add_argument("--ema_shadow", default=None, help="Path to ema_shadow.pt (optional).")
    parser.add_argument("--dtype", default="float16", help="float16|bfloat16|float32")

    # legacy args preserved for compatibility with sample-test.sh
    parser.add_argument("--prompt", default="a photo of a cute cat")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", default="sd3_light_validation_outputs")
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    if not torch.cuda.is_available() and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32

    model_dir = Path(args.model_dir)
    base_model_dir = resolve_base_model_dir(model_dir, args.base_model_dir)

    print("[1] Loading base transformer weights...")
    base_state = load_transformer_state(base_model_dir, dtype)

    print("[2] Loading trained transformer weights...")
    trained_state = load_transformer_state(str(model_dir), dtype)

    changed = compare_state_dicts(base_state, trained_state, name="trained")
    if not changed:
        raise RuntimeError("Trained weights are identical to base weights.")

    if args.ema_shadow:
        print("[3] Loading EMA shadow weights...")
        ema_state = torch.load(args.ema_shadow, map_location="cpu")
        if not isinstance(ema_state, dict) or not ema_state:
            raise ValueError(f"Invalid ema_shadow at {args.ema_shadow}: empty or not a dict")
        ema_state = normalize_ema_state(ema_state)
        ema_changed = compare_state_dicts(base_state, ema_state, name="ema")
        if not ema_changed:
            raise RuntimeError("EMA weights are identical to base weights.")
    else:
        print("[3] ema_shadow not provided, skipping EMA comparison.")

    print("✅ sd3_light weight comparison complete.")


if __name__ == "__main__":
    main()
