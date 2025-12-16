import os
from pathlib import Path

# 要生成的脚本数量：生成从 start_idx 到 N（包含）
N = 8         # 比如生成 0~10 共 11 个；你按需改
start_idx = 0   # 如果想从 1 开始，就改成 1

# 输出目录（放生成的 .sh 文件）
out_dir = Path("/inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe/scripts/cache/extra")
out_dir.mkdir(parents=True, exist_ok=True)

template = """#!/usr/bin/env bash
set -euo pipefail
source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-pipe
cd /inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe

NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config /inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe/settings/cache/train_configs/extra/sd3_cache_25-{idx}.toml --cache_only
"""

for i in range(start_idx, N + 1):
    idx_str = f"{i:02d}"  # 0000, 0001, 0002, ...
    script_name = out_dir / f"run_cache_{idx_str}.sh"

    content = template.format(idx=idx_str)

    with open(script_name, "w", encoding="utf-8") as f:
        f.write(content)

    # 设置可执行权限（Linux）
    os.chmod(script_name, 0o755)

    print(f"生成: {script_name}")
