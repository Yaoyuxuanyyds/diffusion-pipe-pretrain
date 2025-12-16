import os
import torch
import random
import numpy as np
from models import sd3_light


def main():
    config = {
        "model": {
            "type": "sd3_light",
            "diffusers_path": "/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/SD3",
            "dtype": torch.bfloat16,
            "transformer_dtype": torch.bfloat16,
            "num_layers": 15,
        }
    }

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    save_dir = (
        "/inspire/hdd/project/chineseculture/public/yuxuan/"
        "diffusion-pipe/outputs/sd3_light_pretrain/sd3_light-layer15_init"
    )

    if os.path.exists(save_dir):
        raise RuntimeError(f"{save_dir} already exists. Refusing to overwrite.")

    # 1️⃣ 构建 bootstrap pipeline（VAE / text encoder = pretrained）
    pipe = sd3_light.LightSD3Pipeline(config)

    # 2️⃣ 随机初始化 denoiser + 冻结非 denoiser + 保存
    pipe.build_random_init_and_save(save_dir)

    # 3️⃣ 记录初始化随机种子（可选但很好）
    with open(os.path.join(save_dir, "init_seed.txt"), "w") as f:
        f.write(str(seed))

    print(f"[OK] Saved random-initialized SD3-Light to {save_dir}")


if __name__ == "__main__":
    main()
