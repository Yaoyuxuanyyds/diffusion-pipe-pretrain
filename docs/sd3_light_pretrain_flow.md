# SD3-Light 预训练数据与代码链路

下文梳理 `sd3_light` 预训练（`train_type = "sd3_light_pretrain"`）从数据缓存到梯度回传的主要调用链与数据流向，帮助快速定位关键步骤。

## 1. 数据缓存与准备
- **缓存入口**：`train.py` 构建完模型后会创建 `DatasetManager` 并调用 `dataset_manager.cache()`，触发多进程缓存。缓存流程依赖模型提供的 VAE / 文本编码器调用函数，以 GPU worker 方式完成。数据集配置解析与注册见 `train.py` 中 `Dataset` 创建与注册逻辑。【F:train.py†L495-L551】【F:utils/dataset.py†L1123-L1146】
- **缓存细节**：`_cache_fn` 负责遍历所有子数据集，调用 `cache_metadata`、`cache_latents`、`cache_text_embeddings`。其中 `latents_map_fn` 会把图像/遮罩送到 VAE，`text_embedding_map_fn` 会单独处理每个文本编码器并把结果写入缓存文件，以便训练阶段直接加载。完成后主进程再把缓存重新加载为 CPU 张量，避免训练时额外 I/O。【F:utils/dataset.py†L1009-L1117】【F:utils/dataset.py†L1181-L1204】

## 2. 训练时的数据加载
- **管线 DataLoader**：`PipelineDataLoader` 包装了 `torch.utils.data.DataLoader`，会调用模型的 `prepare_inputs` 将缓存好的特征转成噪声输入、时间步等，并把目标 `target` 广播到管线首尾阶段以兼容流水线并行。【F:utils/dataset.py†L1274-L1379】
- **批次拆分**：同一大批次会被 `split_batch` 切成多个 micro-batch，用于梯度累积或流水线并行。返回值格式是 `(features, label)`，其中 `features` 和 `label` 仅包含张量。【F:utils/dataset.py†L1248-L1256】

## 3. 模型前向准备与随机噪声
- **特征组装**：`models/sd3_light.py::LightSD3Pipeline.prepare_inputs` 读取缓存的 `latents`、三路文本编码器输出（两路 CLIP + 一路 T5）、可选掩码，先执行可配置的“无条件文本 dropout”，随后按 SD3 约定拼接文本嵌入，插值掩码，并根据设定的时间步采样方式（对数正态/均匀、可选 shift）采样 `t`。最终生成 `(noisy_latents, t*1000, prompt_embeds, pooled_prompt_embeds)` 特征，以及 `(target, mask)` 作为监督信号。【F:models/sd3_light.py†L454-L520】
- **无条件 dropout**：当 `enable_uncond_text_dropout` 为 `true` 时，三路文本编码器各自以 `uncond_text_dropout_prob`（默认 0.464）独立采样伯努利掩码，将对应编码器的 token/pooled 向量置零，实现约 10% 的无条件训练步率。【F:models/sd3_light.py†L463-L482】

## 4. 训练主循环与前向/反向
- **迭代驱动**：训练循环中，Deepspeed `model_engine.train_batch(iterator)` 消费 `PipelineDataLoader` 迭代器完成一次前向与反向计算；循环内按步数更新 EMA、日志、评估与 checkpoint 触发器。`model_engine.reset_activation_shape()` 在每步前重置形状缓存以适配动态尺寸。【F:train.py†L967-L1036】
- **损失计算**：模型默认使用 `BasePipeline.get_loss_fn` 提供的 MSE（或可选 pseudo-Huber）损失，对输出与 `target` 做逐元素比较并按掩码加权后求平均；该损失函数传递给 Deepspeed 管线，参与反向传播与梯度累积。【F:models/base.py†L260-L278】

## 5. 参数更新与保存
- **优化与调度**：`train.py` 基于配置构建优化器与学习率调度器，Deepspeed 负责梯度同步、梯度裁剪、optimizer step。梯度释放/EMA 配置在主循环中按设定频率更新并在保存模型时写出。`Saver` 处理 checkpoint 与最终模型导出（包括 EMA 权重）。【F:train.py†L497-L560】【F:train.py†L889-L952】【F:train.py†L1037-L1044】

以上链路覆盖了从缓存生成、输入准备、模型前向/反向到损失与权重更新的核心路径，可结合具体配置或调试需求在对应节点插桩。
