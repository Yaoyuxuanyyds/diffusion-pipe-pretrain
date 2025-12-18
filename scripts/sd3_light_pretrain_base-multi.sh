set -euo pipefail


# =============== 阶段 0：环境 ===============
source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-pipe

# 进入项目目录
cd /inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe

export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=eth0   # 按实际网卡修改

export WORLD_SIZE=$((PET_NNODES * 8))
export RANK=$((PET_NODE_RANK * 8))

deepspeed \
  --num_nodes ${PET_NNODES} \
  --num_gpus 8 \
  train.py \
  --deepspeed \
  --config /inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe/settings/train/pretrain_sd3_light_base-multi.toml \
  --trust_cache
