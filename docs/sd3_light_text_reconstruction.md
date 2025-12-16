# sd3_light 文本特征重建损失说明

## 需求背景

在 `sd3_light` 预训练中，原本仅对 MMDiT 的噪声预测（predicted velocity）进行监督。为提升文本条件对生成质量的约束力，本次改动新增了 **文本特征重建损失**：

- 取 MMDiT 最后一层输出的文本 tokens；
- 与某一“目标层”的文本 tokens 做逐 token 的 MSE 重建；
- 以权重 `gamma` 融合到原有的 denoise loss 中。

## 配置项

在 `model` 配置中新增两项参数：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `text_recon_gamma` | 文本重建损失权重；`0` 表示关闭。 | `0.0` |
| `text_recon_target_layer` | 选取哪一层的文本 tokens 作为重建目标。`0` 代表使用输入给 MMDiT（经过 `context_embedder` 后）的文本 tokens，`k (>0)` 代表使用第 `k` 个 Transformer block 输出的文本 tokens。 | `0` |

示例片段：

```toml
[model]
text_recon_gamma = 0.5
text_recon_target_layer = 0
```

## 实现要点

1. **流水线层级捕获文本 tokens**
   - `InitialLayer` 在前向时记录输入给 MMDiT 的文本 tokens，当 `text_recon_target_layer = 0` 时作为重建目标。
   - `TransformerLayer` 接受当前层索引，若命中 `text_recon_target_layer`，则更新重建目标为本层输出的文本 tokens（`detach` 后参与监督）。
   - `FinalLayer` 在启用文本重建时返回 `(predicted_noise, final_text_tokens, target_tokens)`，否则保持原有输出。

2. **定制损失函数**
   - `LightSD3Pipeline.get_loss_fn` 现在在开启文本重建时，同时计算：
     - 原有的 denoise loss（支持 mask 与 pseudo-huber）；
     - 文本 tokens 间的逐 token MSE（若存在有效目标 tokens）。
   - 最终损失：`loss = denoise_loss + gamma * text_recon_loss`。

## 兼容性与性能

- 当 `text_recon_gamma = 0` 时，前向输出与损失逻辑与旧版完全一致，不额外开销。
- 文本重建路径仅在启用时多带一份文本 tokens（单路张量），对显存与通讯影响可控。

## 最小验证片段

无需真实权重即可快速验证损失组合逻辑：

```python
import torch
from models.sd3_light import LightSD3Pipeline

# 构造一个“未初始化”的 pipeline 实例，仅用于获取 loss_fn
pipe = LightSD3Pipeline.__new__(LightSD3Pipeline)
pipe.config = {}
pipe.model_config = {"text_recon_gamma": 0.5, "text_recon_target_layer": 0}

loss_fn = pipe.get_loss_fn()

pred = torch.tensor([1.0, 2.0])
target = torch.zeros_like(pred)
mask = torch.tensor([])

final_tokens = torch.tensor([[1.0, 1.0]])
target_tokens = torch.tensor([[0.0, 0.0]])

loss = loss_fn((pred, final_tokens, target_tokens), (target, mask))

# 期望：base MSE (=2.5) + 0.5 * text MSE (=1.0)
expected = torch.mean((pred - target) ** 2) + 0.5 * torch.mean((final_tokens - target_tokens) ** 2)
assert torch.allclose(loss, expected)
```

如上断言通过即表明文本重建损失路径生效，并与 `gamma` 正确组合。
