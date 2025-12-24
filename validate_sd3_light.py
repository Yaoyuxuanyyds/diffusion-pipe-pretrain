import argparse
from pathlib import Path

import torch

from models.sd3_light import LightSD3Pipeline


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


def build_prompt_embeddings(pipe: LightSD3Pipeline, prompt: str):
    prompt_embed, pooled_prompt_embed = pipe._encode_clip_prompts(
        [prompt], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer
    )
    prompt_2_embed, pooled_prompt_2_embed = pipe._encode_clip_prompts(
        [prompt], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2
    )
    t5_prompt_embed = pipe._encode_t5_prompts([prompt])
    return prompt_embed, pooled_prompt_embed, prompt_2_embed, pooled_prompt_2_embed, t5_prompt_embed


@torch.no_grad()
def run_loss_check(pipe: LightSD3Pipeline, prompt: str, height: int, width: int, device: torch.device):
    pipe.eval()
    vae = pipe.get_vae()
    vae.to(device)

    image = torch.rand((1, 3, height, width), device=device) * 2 - 1
    latents = pipe.get_call_vae_fn(vae)(image)

    (
        prompt_embed,
        pooled_prompt_embed,
        prompt_2_embed,
        pooled_prompt_2_embed,
        t5_prompt_embed,
    ) = build_prompt_embeddings(pipe, prompt)

    inputs = {
        "latents": latents,
        "prompt_embed": prompt_embed,
        "pooled_prompt_embed": pooled_prompt_embed,
        "prompt_2_embed": prompt_2_embed,
        "pooled_prompt_2_embed": pooled_prompt_2_embed,
        "t5_prompt_embed": t5_prompt_embed,
        "mask": torch.ones((1, height, width), device=device),
    }

    model_inputs, label = pipe.prepare_inputs(inputs)
    output = pipe.transformer(*model_inputs)
    loss_fn = pipe.get_loss_fn()
    loss = loss_fn(output, label)

    loss_value = float(loss.detach().cpu())
    if not torch.isfinite(loss):
        raise RuntimeError("Loss is NaN/Inf, model weights may be invalid.")

    return loss_value, pipe.get_loss_breakdown()


@torch.no_grad()
def run_sampling(
    pipe: LightSD3Pipeline,
    prompt: str,
    height: int,
    width: int,
    steps: int,
    seed: int,
    out_path: Path,
    device: torch.device,
):
    pipe.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe.diffusers_pipeline(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=5.0,
        generator=generator,
    ).images[0]
    image.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Validate sd3_light weights and EMA with loss + sampling.")
    parser.add_argument("--model_dir", required=True, help="Path to saved sd3_light model directory.")
    parser.add_argument("--ema_shadow", default=None, help="Path to ema_shadow.pt (optional).")
    parser.add_argument("--prompt", default="a photo of a cute cat", help="Prompt for embeddings/sampling.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=12, help="Sampling steps.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", default="float16", help="float16|bfloat16|float32")
    parser.add_argument("--expected_blocks", type=int, default=15, help="Optional transformer block count check.")
    parser.add_argument("--output_dir", default="sd3_light_validation_outputs")
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Loading sd3_light pipeline...")
    pipe = LightSD3Pipeline.load_from_pretrained(
        model_dir=args.model_dir,
        dtype=dtype,
        extra_model_config={"num_layers": args.expected_blocks} if args.expected_blocks else None,
    )
    pipe.to(device)

    if args.expected_blocks is not None:
        actual_blocks = len(pipe.transformer.transformer_blocks)
        if actual_blocks != args.expected_blocks:
            raise RuntimeError(
                f"Transformer block count mismatch: expected {args.expected_blocks}, got {actual_blocks}"
            )
        print(f"[CHECK] Transformer blocks OK: {actual_blocks}")

    print("[2] Running loss check on base weights...")
    base_loss, base_breakdown = run_loss_check(pipe, args.prompt, args.height, args.width, device)
    print(f"Base loss: {base_loss:.6f}")
    if base_breakdown:
        print(f"Base loss breakdown: {base_breakdown}")

    base_sample_path = output_dir / "sample_base.png"
    print("[3] Sampling image with base weights...")
    run_sampling(
        pipe,
        args.prompt,
        args.height,
        args.width,
        args.steps,
        args.seed,
        base_sample_path,
        device,
    )
    print(f"Base sample saved to: {base_sample_path}")

    if args.ema_shadow:
        print("[4] Loading EMA shadow and re-checking loss...")
        pipe.load_ema_shadow(args.ema_shadow, strict=False)
        ema_loss, ema_breakdown = run_loss_check(pipe, args.prompt, args.height, args.width, device)
        print(f"EMA loss: {ema_loss:.6f}")
        if ema_breakdown:
            print(f"EMA loss breakdown: {ema_breakdown}")

        ema_sample_path = output_dir / "sample_ema.png"
        print("[5] Sampling image with EMA weights...")
        run_sampling(
            pipe,
            args.prompt,
            args.height,
            args.width,
            args.steps,
            args.seed,
            ema_sample_path,
            device,
        )
        print(f"EMA sample saved to: {ema_sample_path}")
    else:
        print("[4] ema_shadow not provided, skipping EMA checks.")

    print("✅ sd3_light validation complete.")


if __name__ == "__main__":
    main()
