set -euo pipefail

source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-pipe

cd /inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe


unset RANK || true
unset WORLD_SIZE || true
unset LOCAL_RANK || true

export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=eth0   # 如有多网卡建议打开

# ===== rendezvous 关键 =====
if [ "${PET_NODE_RANK}" -eq 0 ]; then
  export MASTER_ADDR=$(hostname -I | awk '{print $1}')
fi
export MASTER_PORT=29500

torchrun \
  --nnodes ${PET_NNODES} \
  --node_rank ${PET_NODE_RANK} \
  --nproc_per_node 8 \
  --master_addr ${MASTER_ADDR} \
  --master_port ${MASTER_PORT} \
  train.py \
  --deepspeed \
  --config /inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe/settings/train/pretrain_sd3_light_base-multi-1.toml \
  --trust_cache
