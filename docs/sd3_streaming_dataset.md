# SD3 light pretrain: streaming dataset & cache design

## 目标
- 替换旧的 `mp.Pool + HF dataset.iter` 缓存方案，提供一套可控、低内存、易于调试的管线。
- 仅在 `train_type = "sd3_light_pretrain"` 时启用，不影响其他训练类型。
- 利用 PyTorch `DataLoader + DistributedSampler` 做并行预取，避免长生命周期的 `Manager`/fork 进程导致的 COW 内存上涨。

## 代码落点与职责划分
- **`utils/cache.py`**
  - `ShardCacheWriter`：主进程追加写入，定长分片 + manifest（无 SQLite/Manager）。
  - `ShardCache`：读端 LRU，按需加载 shard，限制常驻内存。
  - `Cache`：兼容旧接口的轻量封装，复用新实现。
- **`utils/dataset.py`**
  - `SD3LightManifestBuilder`：扫描数据目录生成元数据 shard（路径、caption、是否视频）。
  - `SD3LightPretrainDataset`：仅存 `ShardCache` 句柄，`__getitem__` 调 `model.get_preprocess_media_file_fn`。
  - `SD3LightPretrainDataLoader`：`DataLoader + DistributedSampler` 构造大 batch，再用现有 `split_batch` 做 micro-batch，仍支持 `_broadcast_target`、`set_eval_quantile`。
- **`train.py`**
  - 当 `training_type == "sd3_light_pretrain"` 且 `sd3_streaming_dataset=True` 时，走新 manifest+streaming 路径；否则保持旧路径。
  - 现有优化器/调度器/EMA/日志逻辑不变。

## 核心组件
1) **Manifest 构建 (`SD3LightManifestBuilder`)**
   - 按目录扫描媒体文件（图片/视频），读取同名 `.txt` 文本作为 caption（可选前缀）。
   - 使用新的 `ShardCacheWriter` 将元数据按 shard 写入 `cache/sd3_stream_<model_name>`，每个 shard 默认 512 条。
   - Manifest 中包含 fingerprint（基于 dataset 配置 + model name），用于校验/自动失效。

2) **轻量缓存 (`ShardCacheWriter` / `ShardCache`)**
   - 仅依赖简单的 `.pt` 分片与 `manifest.json`，无 SQLite/Manager。
   - 读端带 LRU（默认 2 个 shard）减少磁盘读取，同时保持低峰值内存。
   - 提供兼容层 `Cache` 让旧路径依然可运行。

3) **训练集 (`SD3LightPretrainDataset`)**
   - 只持有 `ShardCache` 句柄；`__getitem__` 读取一条元数据并调用 `model.get_preprocess_media_file_fn` 做按需预处理。
   - 无额外全局状态，workers 生命周期可控。

4) **数据加载器 (`SD3LightPretrainDataLoader`)**
   - 基于 `torch.utils.data.DataLoader + DistributedSampler`，每个 step 直接拉取一个“梯度累积大 batch”，再用现有 `split_batch` 切分为 micro-batch。
   - 兼容已有训练循环接口：`set_eval_quantile/reset/sync_epoch/state_dict` 等。
   - 仅使用 PyTorch worker；不再创建 `Manager`/`Pool`，减少跨进程共享导致的 COW。

## 训练流程（sd3_light_pretrain）
1. 通过 `SD3LightManifestBuilder.build()` 生成/校验 manifest（fingerprint 不一致时自动清空）。
2. 构造 `SD3LightPretrainDataset`（持有 `ShardCache` 与模型的 `preprocess` 函数）。
3. 创建 `SD3LightPretrainDataLoader`：
   - `batch_size = micro_batch_size_per_gpu * gradient_accumulation_steps`
   - 使用 `DistributedSampler` 分发样本并乱序。
   - 每个 batch 经 `model.prepare_inputs` + `_broadcast_target`，然后按梯度累积切成 micro-batch。
4. 训练循环保持原有 `get_data_iterator_for_step`/`sync_epoch` 逻辑，避免额外内存占用。

## 为什么能抑制内存线性上升
- **无 Manager/Pool/共享大对象**：只用 PyTorch DataLoader worker，生命周期简单。
- **分片持久化 + LRU 读取**：元数据按 shard 存盘，读取时最多缓存少量 shard。
- **无 keep_in_memory/select**：不再在主进程保留大块 Arrow/Tensor。
- **按需预处理**：不缓存全量张量，避免 COW 复制；显存回收与 CPU 内存稳定。

## 扩展与调优
- `sd3_shard_size`：控制 manifest 分片大小（默认 512）。
- `num_dataloader_workers` / `dataloader_prefetch_per_worker`：平衡 CPU 吞吐与主机内存。
- 如需在缓存阶段增加校验/过滤，可在 `SD3LightManifestBuilder.build` 中扩展。

## 启用方式与配置示例
```toml
training_type = "sd3_light_pretrain"
sd3_streaming_dataset = true          # 走新管线，默认为 true
sd3_shard_size = 512                  # 可调分片大小
num_dataloader_workers = 4
dataloader_prefetch_per_worker = 2
```

## 兼容性与回退
- 仅 sd3_light_pretrain 走新路径；其他训练类型沿用原有 DatasetManager/缓存。
- 老接口仍可用（`Cache` 兼容层）；如需完全禁用新方案，设 `sd3_streaming_dataset=false`。

## 验证建议
- 观察主进程与 worker 的 RSS：manifest 构建后应稳定，训练中不随 step 线性上升。
- 压测 shard LRU：调小 `max_shards_in_memory`（构造 `ShardCache` 时）应仍能正常迭代，但磁盘 IO 增加。
- 大批次/高累积情况下，检查 `split_batch` 后的 micro-batch 流程是否与旧逻辑一致。

## 仅单尺寸 resize 的支持
- 在 streaming 模式下（sd3_light_pretrain），manifest 会读取 `resolutions` 的第一个元素作为目标尺寸，并写入每条样本的 `target_size`。例如：
  ```toml
  training_type = "sd3_light_pretrain"
  sd3_streaming_dataset = true
  resolutions = [256]            # 或 [[256, 256]]

  [[directory]]
  path = "/data/cc12m/group_000"
  num_repeats = 1
  ```
- 读取时 `SD3LightPretrainDataset` 将把 `target_size` 传给 `PreprocessMediaFile` 作为 `size_bucket`，进行居中裁剪 + resize（取整到 16 的倍数），并返回 `pixel_values`、`mask`、`caption`。
- 构建 manifest 时即调用 VAE 与文本编码器，将 `latents`、各类 `prompt_embed` 以及 `mask`（无则为空张量）写入分片；训练阶段直接读取成品特征，避免在 step 内重复编码。
- 当前只支持**单一尺寸**；如需多尺寸分桶，请使用旧管线（`sd3_streaming_dataset=false`）。

## 缓存阶段批量编码
- `caching_batch_size` 控制 manifest 构建时的编码批大小；同一个 batch 的 `pixel_values` 会 stack 后统一送入 VAE，再批量跑所有文本编码器，结果拆分写入 shard。
- 示例：
  ```toml
  caching_batch_size = 8
  ```
- 若未显式设置 `sd3_shard_size`，默认与 `caching_batch_size` 相同，以减少已编码批次被再次拆分的开销。
- `caching_device` 可设为 `"auto"`（默认，优先 CUDA）或显式 `"cuda"` / `"cpu"`；编码阶段会把 VAE、文本编码器迁移到该设备并在 CUDA 上启用 amp。
