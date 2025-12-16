

import argparse
import torch
import diffusers
import numpy as np

from models.sd3_light import LightSD3Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe/outputs/sd3_light_pretrain/test1/20251215_09-42-42/step20")
    parser.add_argument("--ema_shadow", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe/outputs/sd3_light_pretrain/test1/20251215_09-42-42/step20/ema_shadow.pt")
    parser.add_argument("--expected_blocks", type=int, default=15)
    args = parser.parse_args()

    print("=" * 80)
    print("[1] Load pipeline via LightSD3Pipeline.load_from_pretrained")
    print("=" * 80)

    pipe = LightSD3Pipeline.load_from_pretrained(
        model_dir=args.model_dir,
        dtype=torch.float32,
        extra_model_config={"num_layers": args.expected_blocks},
    )

    transformer = pipe.transformer

    # ------------------------------------------------------------------
    # 1. 验证 transformer block 数
    # ------------------------------------------------------------------
    print("\n[CHECK-1] Transformer block count")

    num_blocks = len(transformer.transformer_blocks)
    print(f"Expected blocks: {args.expected_blocks}")
    print(f"Actual blocks:   {num_blocks}")

    assert num_blocks == args.expected_blocks, (
        f"❌ Block count mismatch: expected {args.expected_blocks}, got {num_blocks}"
    )
    print("✅ Block count correct")

    # ------------------------------------------------------------------
    # 2. 列出所有 trainable parameters
    # ------------------------------------------------------------------
    print("\n[CHECK-2] Trainable parameters (requires_grad=True)")

    trainable = {
        name: p
        for name, p in transformer.named_parameters()
        if p.requires_grad
    }

    print(f"Trainable parameter count: {len(trainable)}")
    print("First 20 trainable parameters:")
    for i, k in enumerate(trainable.keys()):
        if i >= 20:
            break
        print(" ", k)

    # ------------------------------------------------------------------
    # 3. pos_embed 是 parameter 还是 buffer？
    # ------------------------------------------------------------------
    print("\n[CHECK-3] pos_embed inspection (SD3-correct)")

    pos_params = [
        name for name, _ in transformer.named_parameters()
        if name.startswith("pos_embed.")
    ]

    pos_buffers = [
        name for name, _ in transformer.named_buffers()
        if name.startswith("pos_embed.")
    ]

    print("pos_embed parameters:", pos_params)
    print("pos_embed buffers:", pos_buffers)

    # 必须包含可训练的 projection
    assert "pos_embed.proj.weight" in pos_params
    assert "pos_embed.proj.bias" in pos_params

    # 必须包含 buffer positional grid
    assert "pos_embed.pos_embed" in pos_buffers

    print("✅ pos_embed structure is SD3-correct")


    # ------------------------------------------------------------------
    # 4. 加载 EMA shadow
    # ------------------------------------------------------------------
    print("\n[CHECK-4] Load EMA shadow")

    ema_sd = torch.load(args.ema_shadow, map_location="cpu")

    assert isinstance(ema_sd, dict), "❌ EMA shadow is not a dict"
    assert len(ema_sd) > 0, "❌ EMA shadow is empty"

    print(f"EMA shadow entries: {len(ema_sd)}")
    print("First 20 EMA keys:")
    for i, k in enumerate(ema_sd.keys()):
        if i >= 20:
            break
        print(" ", k)

    # ------------------------------------------------------------------
    # 5. EMA keys ⊆ trainable parameters？
    # ------------------------------------------------------------------
    print("\n[CHECK-5] EMA keys vs trainable parameters")

    trainable_keys = set(trainable.keys())
    ema_keys = set(ema_sd.keys())

    missing_in_model = ema_keys - trainable_keys
    missing_in_ema = trainable_keys - ema_keys

    print("EMA keys not in trainable parameters:", missing_in_model)
    print("Trainable parameters missing in EMA:", missing_in_ema)

    assert len(missing_in_model) == 0, "❌ EMA contains non-trainable params"
    assert len(missing_in_ema) == 0, "❌ Some trainable params not tracked by EMA"

    print("✅ EMA tracking matches trainable parameters exactly")

    # ------------------------------------------------------------------
    # 6. 验证 load_ema_shadow 是否真的替换参数
    # ------------------------------------------------------------------
    print("\n[CHECK-6] Verify EMA actually replaces weights")

    # 随机选 5 个参数做数值对比
    test_keys = list(trainable_keys)[:5]

    before = {
        k: transformer.state_dict()[k].clone()
        for k in test_keys
    }

    missing, unexpected = pipe.load_ema_shadow(args.ema_shadow, strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    after = {
        k: transformer.state_dict()[k]
        for k in test_keys
    }

    diffs = []
    for k in test_keys:
        diff = (before[k] - after[k]).abs().mean().item()
        diffs.append(diff)
        print(f"Param {k}: mean |Δ| = {diff:.6e}")

    assert any(d > 0 for d in diffs), (
        "❌ EMA load did not change any parameters"
    )

    print("✅ EMA parameters successfully loaded and differ from normal weights")

    print("\n" + "=" * 80)
    print("🎉 ALL EMA / SD3-LIGHT VERIFICATIONS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
