#!/usr/bin/env bash
set -euo pipefail
source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-pipe
cd /inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe

NCCL_P2P_DISABLE="0" NCCL_IB_DISABLE="0" deepspeed --num_gpus=4 train.py --deepspeed --config /inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe/settings/cache/train_configs/sd3_cache_0011.toml --cache_only --regenerate_cache
