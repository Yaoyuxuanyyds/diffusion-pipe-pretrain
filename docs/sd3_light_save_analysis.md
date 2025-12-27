# SD3-Light 预训练保存逻辑审查（重点：`Saver.save_full_model`）

本文聚焦 `sd3_light` 预训练中**模型保存**的代码路径，尤其是 `utils/saver.py` 的 `save_full_model`，并结合其调用的 `models/sd3_light.py::LightSD3Pipeline.save_model` 分析潜在风险与可能导致保存模型不可用的问题。

---

## 1. 入口与整体流程（训练保存）

训练过程中保存模型的入口位于 `utils/saver.py::Saver.save_model`：

1. 训练过程中满足保存条件时（步数/epoch/手动信号），调用 `Saver.save_model(name)`。
2. 非 LoRA 情况下会执行 `save_full_model(name)`。
3. 若启用 EMA，会额外保存 `ema_shadow.pt` 到同一目录。

该流程说明 **全模型保存**由 `save_full_model` + `LightSD3Pipeline.save_model` 组合完成。

---

## 2. `Saver.save_full_model` 逻辑逐步说明

文件：`utils/saver.py`

核心步骤如下：

1. **目录与同步**
   - 仅 `dp0 + stage0` 创建 `save_dir/tmp`。
   - `dist.barrier()` 让所有 pipeline stage 同步。

2. **按 pipeline stage 保存参数分片**
   - 每个 pipeline stage 的 `dp0` 遍历 `self.pipeline_model.parameters()`。
   - 仅保存**包含 `original_name` 属性**的参数到 `partial_state_dict`。
   - 依配置可转换 `save_dtype`。
   - 保存为 `tmp/state_dict_{stage_id}.bin`。

3. **汇总所有分片并交给 `model.save_model`**
   - `stage0 + dp0` 读取所有 `state_dict_*.bin` 合并为完整 `state_dict`。
   - 调用 `self.model.save_model(save_dir, state_dict)`。
   - 复制 config 并清理 tmp。

这一设计的关键是：**Deepspeed pipeline 模型的分片参数，需要通过 `original_name` 拼回完整模型。**

---

## 3. `LightSD3Pipeline.save_model` 逻辑

文件：`models/sd3_light.py`

主要步骤：

1. **确保冻结组件不是 meta tensor**
   - 如果 pipeline 内存在 meta tensor，重新从 `diffusers_path` 加载完整 SD3 pipeline 作为 `save_pipe`。
   - 否则直接使用当前 `self.diffusers_pipeline`。

2. **用 `diffusers_sd` 覆盖 transformer 权重**
   - 传入的 `diffusers_sd` 是 `Saver.save_full_model` 合并出的参数字典。
   - 抽取 `transformer.*` 前缀，并去掉前缀再 `load_state_dict` 到 `save_pipe.transformer`。
   - 未匹配的 key 会输出 warning。

3. **保存 diffusers 目录**
   - `save_pipe.save_pretrained(save_dir, safe_serialization=True)`。

**结论：**保存后的模型是否正确，关键取决于 `diffusers_sd` 是否正确、完整地反映训练后的 transformer 权重。

---

## 4. 潜在问题分析（可能导致保存模型不可用）

### ⚠️ 问题 1：`original_name` 未设置导致 state_dict 为空

`save_full_model` 的参数遍历中：

```python
for p in self.pipeline_model.parameters():
    if not hasattr(p, "original_name"):
        continue
    partial_state_dict[p.original_name] = p.detach().to("cpu")
```

**风险点：**
- `sd3_light` 中未看到任何显式设置 `p.original_name` 的逻辑。
- `BasePipeline.configure_adapter` 中会设置 `original_name`，但 SD3-Light 不支持 LoRA，因此该路径通常不会执行。

**结果：**
- 若 `original_name` 未赋值，所有参数都会被跳过，最终 `diffusers_sd` 为空。
- `save_model` 会尝试加载空 dict，导致 transformer 没有被覆盖。

**潜在后果：**
- 保存模型可能只是初始权重（或某些 stage 当前持有的子集），而不是训练后的完整权重。
- 尤其在 pipeline 并行/分片训练场景，这种情况可能导致保存模型无法正确推理。

**建议重点确认：**
- `sd3_light` 是否在某处隐式设置 `original_name`；
- 若没有，保存将不包含任何参数（核心风险）。

---

### ⚠️ 问题 2：保存依赖 pipeline stage 合并，但未覆盖 buffers

`save_full_model` 仅保存 `parameters()`，不包含 `buffers`（如 layernorm running stats）。

**潜在影响：**
- 对多数 transformer 来说 buffers 影响较小，但如果 SD3-Light 中存在影响推理的 buffer，该逻辑会丢失。
- 此类问题不会报错，但可能导致推理表现异常。

---

### ⚠️ 问题 3：`diffusers_sd` key 命名不一致导致加载不完整

`save_model` 中：

```python
if k.startswith("transformer."):
    tr_sd[k[len("transformer."):]] = v
else:
    tr_sd[k] = v
```

**风险点：**
- 如果 `original_name` 不是以 `transformer.` 开头，但仍包含其它前缀或层级差异，则会造成 `missing/unexpected`。
- `strict=False` 会掩盖此类错误，仅打印 warning。

**潜在后果：**
- 可能只加载了部分参数，导致推理结果异常或模型不一致。

---

### ✅ 目前逻辑较稳健的部分

- 处理 meta tensor 的逻辑能避免冻结组件意外保存为 meta tensor。
- `safe_serialization=True` 能避免保存为传统 `.bin` 带来的不一致问题。
- 每个 stage 单独保存分片，理论上可支持 pipeline 并行训练。

---

## 5. 结论与建议

### ✅ 结论
当前保存流程整体设计合理，但**强依赖 `original_name` 属性**。而 `sd3_light` 中未见此属性设置逻辑，这极有可能导致 `save_full_model` 保存的 state_dict 为空或不完整，从而产生**保存模型无法正常推理**或**权重未更新**的问题。

### ✅ 建议重点排查/改进方向

1. **确认 `original_name` 是否在 SD3-Light 训练流程中被设置**
   - 若没有，建议在 SD3-Light 初始化或训练前显式为 `self.transformer` 参数赋值（与 `sdxl` 等模型一致）。

2. **增加保存前校验**
   - 当 `diffusers_sd` 为空或 key 数量异常时，应明确报错或至少警告。

3. **检查 key prefix 是否一致**
   - 建议统一约定 `original_name` 使用 `transformer.` 前缀，避免 `missing/unexpected`。

---

## 6. 参考代码位置

- `utils/saver.py::Saver.save_full_model`
- `models/sd3_light.py::LightSD3Pipeline.save_model`
- `train.py::EMAWeights`（保存 EMA 与 `original_name` 的依赖逻辑）

---

如需我继续补充修复建议或提供 patch 示例，可直接说明。
