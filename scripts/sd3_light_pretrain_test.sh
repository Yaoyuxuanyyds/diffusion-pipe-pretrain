set -euo pipefail


# =============== 阶段 0：环境 ===============
source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-pipe

# 进入项目目录
cd /inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe

NCCL_P2P_DISABLE="0" NCCL_IB_DISABLE="0" deepspeed --num_gpus=8 train.py --deepspeed --config /inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe/settings/train/pretrain_sd3_light_test.toml --trust_cache