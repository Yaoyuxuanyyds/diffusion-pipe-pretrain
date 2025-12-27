import argparse
from pathlib import Path

import torch
import diffusers

from models.sd3_light import LightSD3Pipeline


def build_min_config(model_dir, dtype):
    return {
        "model": {
            "dtype": dtype,
            "transformer_dtype": dtype,
            "diffusers_path": str(model_dir),
        }
    }


def make_diffusers_sd_from_transformer(pipe):
    return {f"transformer.{k}": v.detach().to("cpu") for k, v in pipe.transformer.state_dict().items()}


def main():
    parser = argparse.ArgumentParser(description="Test sd3_light save/load flow.")
    parser.add_argument("--model_dir", type=Path, required=True, help="Path to a diffusers SD3-Light directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the model.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print("[TEST] Loading sd3_light pipeline...")
    pipe = LightSD3Pipeline.load_from_pretrained(
        args.model_dir,
        dtype=dtype,
        transformer_dtype=dtype,
        extra_model_config=build_min_config(args.model_dir, dtype)["model"],
    )

    # Ensure original_name is present (save_full_model depends on it)
    original_name_count = sum(
        1 for p in pipe.transformer.parameters() if getattr(p, "original_name", None)
    )
    print(f"[TEST] original_name set on {original_name_count} transformer params.")

    # Make a deterministic change to verify save/load consistency
    with torch.no_grad():
        for p in pipe.transformer.parameters():
            p.add_(0.001)
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[TEST] Saving model...")
    diffusers_sd = make_diffusers_sd_from_transformer(pipe)
    pipe.save_model(args.output_dir, diffusers_sd=diffusers_sd)

    print("[TEST] Reloading saved model...")
    reloaded = diffusers.StableDiffusion3Pipeline.from_pretrained(
        args.output_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        device_map=None,
    )

    print("[TEST] Comparing transformer weights...")
    reloaded_sd = reloaded.transformer.state_dict()
    original_sd = pipe.transformer.state_dict()

    if original_sd.keys() != reloaded_sd.keys():
        missing = set(original_sd.keys()) - set(reloaded_sd.keys())
        extra = set(reloaded_sd.keys()) - set(original_sd.keys())
        raise RuntimeError(
            f"State dict key mismatch: missing={len(missing)} extra={len(extra)}"
        )

    for k in original_sd.keys():
        torch.testing.assert_close(
            original_sd[k].cpu(),
            reloaded_sd[k].cpu(),
            rtol=0,
            atol=0,
            msg=f"Mismatch at key: {k}",
        )

    print("[TEST] ✅ sd3_light save/load flow verified.")


if __name__ == "__main__":
    main()
