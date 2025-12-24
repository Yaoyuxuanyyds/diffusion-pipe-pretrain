# SD3_Light 预训练指南（Diffusion-Pipe-Pretrain）

本文档以 **SD3_Light** 为核心，介绍模型、数据集、预训练流程，并给出常用的启动命令示例（含多卡 cache、单机多卡、以及多机多卡）。

## SD3_Light 模型简介
**SD3_Light** 是 Stable Diffusion 3 系列的轻量化变体，强调在较低显存和算力预算下实现高质量生成。该仓库提供了 SD3_Light 的训练与预训练配置，配合 Deepspeed 实现高效并行训练。

主要特点：
- 轻量化架构，适合单机/多机多卡预训练。
- 与本仓库的数据缓存机制紧密配合，降低训练阶段的显存压力。
- 通过配置文件统一管理训练参数，易于复现与扩展。

## sd3_streaming_dataset 简介
**sd3_streaming_dataset** 是面向 SD3_Light 预训练的数据集组织方式，支持流式读取与缓存机制结合使用。其优势包括：
- 数据流式加载，支持大规模数据集训练。
- 自动缓存潜变量与文本编码，减少重复计算。
- 适配分布式训练场景，减少 I/O 压力。

## 预训练概览
预训练过程分为两部分：
1. **缓存阶段（cache）**：预先计算并缓存 latent 与文本编码，后续训练直接读取缓存。
2. **训练阶段（train）**：加载缓存后的数据执行模型训练。可通过 `--trust_cache` 信任已生成的缓存。

建议：
- 数据量大时先完成缓存再训练。
- 多机多卡时务必设置正确的 rendezvous（MASTER_ADDR/MASTER_PORT）。

## 基本使用方法
下面给出常用命令示例，请根据实际路径与硬件配置调整。

### 1. 多卡 cache（仅缓存）
```
NCCL_P2P_DISABLE="0" NCCL_IB_DISABLE="0" \
  deepspeed --num_gpus=4 train.py --deepspeed \
  --config /inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe/settings/cache/train_configs/sd3_cache_0000.toml \
  --cache_only
```

### 2. 单机多卡基础 SD3_Light 预训练
```
NCCL_P2P_DISABLE="0" NCCL_IB_DISABLE="0" \
  deepspeed --num_gpus=8 train.py --deepspeed \
  --config /inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe/settings/train/pretrain_sd3_light_base.toml \
  --trust_cache
```

### 3. 多机多卡基础 SD3_Light + 文本重建（Text Reconstruction）预训练
```
RANK || true
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
  --config /inspire/hdd/project/chineseculture/public/yuxuan/diffusion-pipe/settings/train/pretrain_sd3_light_txt-multi.toml \
  --trust_cache
```

## 备注
- `NCCL_P2P_DISABLE` / `NCCL_IB_DISABLE` 建议按集群网络与显卡拓扑调整。
- `--trust_cache` 会跳过缓存一致性校验，请确保缓存与原始数据一致。
- 配置文件中可自定义数据路径、batch size、优化器设置等。
