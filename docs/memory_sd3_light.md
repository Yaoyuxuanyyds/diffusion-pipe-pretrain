# sd3_light 预训练内存占用与优化说明

## 背景
在 `sd3_light` 预训练中，数据量巨大且往往会在 DataLoader 中开启多个 `num_workers`（如 4-8）以提升数据吞吐。原有实现中：

- 每个 worker 会预取多个批次的缓存结果（latent、文本向量等），当批次体积较大时会在 CPU 内存中同时驻留。
- 预取批次数与 worker 数量相乘，导致内存峰值随 `num_workers` 线性放大，甚至触发 OOM / worker 被 kill。

因此需要在保证吞吐的前提下，控制每个 worker 的预取深度。

## 问题原因梳理
1. **过度预取**：PyTorch `DataLoader` 默认 `prefetch_factor=2`，每个 worker 预先准备 `prefetch_factor * batch_per_request` 个批次。对于包含高分辨率 latent 的批次，这会让多个批次同时驻留在主机内存。
2. **多 worker 叠加**：当 `num_workers` 为 4-8 时，默认的预取策略意味着同时缓存 8-16 个批次的 decoded/loaded 数据，直接推高系统内存。
3. **批次解耦不足**：虽然缓存文件存储在磁盘（参见 `utils/cache.py`），但预取阶段会把完整批次解码成 `torch.Tensor` 后放入 worker 队列，导致 CPU 内存占用，而非仅靠文件 mmap。

## 改进策略
1. **显式控制 worker 预取批次**：为 `PipelineDataLoader` 增加 `prefetch_batches_per_worker`，默认限制为 1，使每个 worker 同时只保留 1 个待消费批次，显著降低内存峰值。
2. **配置化**：通过训练配置项 `dataloader_prefetch_per_worker` 调整预取深度，兼顾不同场景的吞吐需求。
3. **兼容现有入口**：`train.py` 和评估 DataLoader 均读取上述配置，保持行为一致，避免训练/评估切换时内存激增。

## 最小验证代码块
下面的示例创建一个伪造的小批次数据集，分别在默认（2）与受控（1）预取下启动 2 个 worker。运行时可结合 `psutil.Process(os.getpid()).memory_info().rss` 观测内存，确保受控模式的 RSS 明显下降。

```python
import torch
from torch.utils.data import Dataset
from utils import dataset as dataset_util

class TinyDataset(Dataset):
    def __len__(self):
        return 16
    def __getitem__(self, idx):
        return {'x': torch.zeros((4, 3, 256, 256), dtype=torch.float16), 'mask': None}

# 模拟模型接口
class DummyModel:
    name = 'dummy'
    def prepare_inputs(self, batch, timestep_quantile=None):
        return (batch, (torch.zeros(1), None))

    @property
    def grid(self):
        class _G:
            def get_data_parallel_rank(self): return 0
            def get_data_parallel_world_size(self): return 1
            def get_model_parallel_rank(self): return 0
            def get_model_parallel_world_size(self): return 1
            def get_pipe_parallel_rank(self): return 0
            def get_pipe_parallel_world_size(self): return 1
            @property
            def pp_group(self): return [0]
        return _G()
    def gradient_accumulation_steps(self): return 1
    def is_pipe_parallel(self): return False

model = DummyModel()
ds = TinyDataset()

# 默认 DataLoader，预取 2*workers 批次
loader_default = dataset_util.PipelineDataLoader(ds, model, 1, model, num_dataloader_workers=2, prefetch_batches_per_worker=2)
print('default prefetch_factor =', loader_default.dataloader.prefetch_factor)

# 受控预取 DataLoader
loader_limited = dataset_util.PipelineDataLoader(ds, model, 1, model, num_dataloader_workers=2, prefetch_batches_per_worker=1)
print('limited prefetch_factor =', loader_limited.dataloader.prefetch_factor)
```

观察 `prefetch_factor` 输出应分别为 2 与 1，且在实际训练中将受控版本与原版对比可看到系统内存峰值下降。
