下面给你一版**简洁版文档说明（归一化坐标方案）**，专门针对“长宽可变 + horizon 可变”的 (S\times H \to A) 策略网络。

---

# 文档：可变尺寸环境下的 (S\times H \to A) 策略（归一化坐标版）

## 1. 目标

学习策略：

[
\pi_\theta(a \mid s,h), \quad s=(x,y)
]

要求：

* 支持 **不同网格尺寸**（rows, cols 可变）
* 支持 **不同 horizon**
* 具备 **空间泛化能力**

---

## 2. 核心思想

把“绝对坐标”改成“相对位置”：

[
\tilde{x} = \frac{x}{R-1}, \quad
\tilde{y} = \frac{y}{C-1}, \quad
\tilde{h} = \frac{h}{H-1}
]

其中：

* (R)：行数（rows）
* (C)：列数（cols）
* (H)：horizon

这样输入始终落在 ([0,1]) 区间，实现**尺寸无关**。

注意这里的 H 就取训练到这一步的时候的 Horizon

---

## 3. 输入特征

基础输入：

[
\phi(s,h) = [\tilde{x}, \tilde{y}, \tilde{h}]
]

可选增强（推荐）：

[
[\tilde{x}, \tilde{y}, \tilde{h},\ 1-\tilde{x},\ 1-\tilde{y}, 1-\tilde{h}]
]

含义：

* 距离上下边界
* 距离左右边界

---

## 4. 网络结构

简单 MLP：

[
\phi(s,h)
\to \text{Linear}
\to \text{ReLU}
\to \text{Linear}
\to \text{ReLU}
\to \text{Linear}
\to \mathbb{R}^{|A|}
]

输出 logits：

[
\ell(s,h) \in \mathbb{R}^{|A|}
]

策略：

[
\pi_\theta(a\mid s,h)=\text{softmax}(\ell(s,h))
]

---

## 5. 推荐超参数

* 输入维度：3（或 5）
* hidden size：64
* 层数：2 层
* 激活：ReLU
* 输出：动作数 (|A|)

即：

[
3 \to 64 \to 64 \to |A|
]

---

## 6. PyTorch 实现（核心版）

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

class CoordPolicy(nn.Module):
    def __init__(self, n_actions, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # x,y,h
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x, y, h, rows, cols, horizon):
        # 归一化
        x_norm = x.float() / (rows - 1)
        y_norm = y.float() / (cols - 1)
        h_norm = h.float() / (horizon - 1)

        inp = torch.stack([x_norm, y_norm, h_norm], dim=-1)
        logits = self.net(inp)
        return logits

    def act(self, x, y, h, rows, cols, horizon):
        logits = self.forward(x, y, h, rows, cols, horizon)
        dist = Categorical(logits=logits)
        return dist.sample()
```

---

## 7. 为什么这个方法适合“长宽可变”

因为网络学的是：

* “靠左/靠右/靠中间”
* “靠上/靠下”
* “早期/后期”

而不是：

* 第 37 列
* 第 82 行

所以：

* 5×100 → 7×200 → 3×60 都能自然迁移
* 不依赖 embedding 表大小
* 不需要重新定义网络结构

---

## 8. 使用建议

### 适合场景

* 不同尺寸环境之间泛化
* 空间结构比较“平滑”
* reward/policy依赖相对位置

### 不适合场景

* 特定绝对位置非常重要（比如某几个固定格子）
* reward 非常离散、不规则

---

## 9. 可选增强

如果想更强泛化，可以加：

* 边界特征：
  [
  1-\tilde{x},\ 1-\tilde{y}, 1-\tilde{h}
  ]

---

## 10. 一句话总结

把策略从：

> “第几个格子 + 第几步”

变成：

> “在地图中的相对位置 + 当前进度”

即：

[
(x,y,h) \Rightarrow \left(\frac{x}{R},\frac{y}{C},\frac{h}{H}\right)
\Rightarrow \text{MLP} \Rightarrow \pi(a)
]

这是最简单、最稳定、最适合**可变尺寸泛化**的方案。
