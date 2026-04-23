好，我给你一版**完全规整、适合直接阅读/整理笔记的版本**，不用 markdown 结构，用正常分段 + LaTeX 公式，逻辑从“算法 → 数据 → terminate → 伪代码实现”。

---

# 一、问题背景（你要理解的核心）

论文里的 Algorithm 4 / 5（CODEX.W）本质是在做：

> 构造一个 policy cover，使得不同 policy collectively 覆盖整个 state space

不是训练一个最优策略，而是训练一组“探索策略”。

---

# 二、环境与 terminate / truncate 处理

## 1. 理论设定

论文假设是 **finite-horizon MDP**：

[
x_1 \to x_2 \to \cdots \to x_H
]

* horizon 固定为 (H)
* 每个 episode **必须走满 H 步**

---

## 2. terminate 的结论（重点）

### ❗ 理论上：不允许提前 terminate

原因：

算法核心依赖 occupancy measure：

[
d_h^\pi(x) = \mathbb{P}(x_h = x)
]

如果提前结束：

* 后面的 (d_h^\pi) 全为 0
* density ratio 无法定义
* 算法直接失效

---

## 3. 实际实现方法

如果你的环境会 done（比如 gym）：

### 方法1（推荐）：absorbing state

当环境结束时：

[
x_{h+1} = x_{\text{absorb}}
]

```python
if done:
    next_state = absorbing_state
    reward = 0
```

---

### 方法2（简化）：padding

把 trajectory 补齐到长度 H

```python
while len(traj) < H:
    traj.append(dummy_state)
```

---

### ❗ 结论

> 在这个算法中，“提前结束”不算探索
> 必须保证所有层 (h = 1 \dots H) 都有定义

---

# 三、Algorithm 4：整体结构

Algorithm 4 是一个 **逐层（layer-wise）构造 policy cover 的算法**

---

## 主流程

对于每一层 (h = 2, \dots, H)：

1. 构造一组 policy：
   [
   \pi_{h,1}, \pi_{h,2}, \dots, \pi_{h,T}
   ]

2. 最终形成 policy cover：
   [
   p_h = \text{Uniform}(\pi_{h,1} \circ_h \pi_{\text{unif}}, \dots, \pi_{h,T} \circ_h \pi_{\text{unif}})
   ]

---

## 每一步 t 做什么？

### Step 1：估计 weight function

[
\hat{w}*h^t(x_h \mid x*{h-1}, a_{h-1})
\approx
\frac{P(x_h \mid x_{h-1}, a_{h-1})}
{\sum_{i<t} d_h^{\pi_{h,i}}(x_h) + P(x_h \mid x_{h-1}, a_{h-1})}
]

---

### Step 2：把 weight 当 reward

[
r_{h-1}(x_{h-1}, a_{h-1}) = \hat{w}*h^t(x_h \mid x*{h-1}, a_{h-1})
]

---

### Step 3：policy optimization

[
\pi_{h,t} \approx \arg\max_\pi \mathbb{E}[r]
]

---

# 四、Algorithm 5：数据是怎么来的（重点）

## 1. 采样分布 q

算法不是直接用 replay buffer，而是构造一个 mixture：

[
q = \frac{1}{2} p_{h-1}
+
\frac{1}{2(t-1)} \sum_{i<t} \pi_{h,i} \circ_{h-1} \pi_{\text{unif}}
]

---

## 2. 采样过程

每个样本：

1. 从 (q) 采一个 policy (\pi)
2. rollout 得到 trajectory
3. 取 transition：

[
(x_{h-1}, a_{h-1}, x_h)
]

---

## 3. 数据集构造

构造两个数据集：

[
D_1, D_2
]

（本质是 density ratio estimation）

---

## 4. weight estimation

优化目标：

[
\hat{w} = \arg\max_w
\mathbb{E}_{D_1}[\log w]
------------------------

\mathbb{E}_{D_2}[w]
]

---

# 五、数据是不是 replay buffer？

## 结论：不是标准 replay buffer

没有：

```
buffer.sample(batch)
```

---

## 但存在“隐式 replay”

体现在：

### 1. policy mixture

历史 policy 被反复采样

---

### 2. density 累积

[
\sum_{i<t} d_h^{\pi_i}(x)
]

直接用历史访问分布

---

### 3. policy optimization 使用旧数据

算法允许：

* offline + online hybrid

---

## 一句话总结

> replay 是在 policy 层，而不是 transition 层

---

# 六、完整伪代码（工程版）

下面是你可以直接实现的版本：

---

## 主算法（CODEX.W）

```python
T = int(1 / epsilon)

# 初始化
p[1] = uniform_policy()

for h in range(2, H+1):

    policies = []

    for t in range(1, T+1):

        # Step 1: 构造采样分布
        q = build_mixture(p[h-1], policies)

        # Step 2: 收集数据
        data = []
        for _ in range(N):
            pi = sample_policy(q)
            traj = rollout(pi)
            data.append(extract_transition(traj, h))

        # Step 3: 学 weight function
        w_hat = estimate_weight(data)

        # Step 4: 构造 reward
        reward = lambda s,a: w_hat(s,a)

        # Step 5: policy optimization
        pi_new = PolicyOptimization(reward, p[1:h])

        policies.append(pi_new)

    # Step 6: 构造 cover
    p[h] = Uniform([
        compose(pi, uniform_policy(), switch_step=h)
        for pi in policies
    ])
```

---

## Weight Estimation

```python
def estimate_weight(data):

    # D1, D2 实际相同
    D1 = data
    D2 = data

    # 优化目标
    # max E[log w] - E[w]
    w = optimize_density_ratio(D1, D2)

    return w
```

---

# 七、最关键的直觉（一定要理解）

## weight function 在干嘛？

[
w \approx \frac{\text{真实访问概率}}{\text{历史访问概率}}
]

---

## 所以：

* 如果 state 很少访问：
  [
  w \text{ 大}
  ]
* 如果 state 已经探索：
  [
  w \text{ 小}
  ]

---

## ⇒ reward = “未探索程度”

---

# 八、和标准 RL 的区别

| 维度          | 标准RL          | CODEX          |
| ----------- | ------------- | -------------- |
| 目标          | 最优策略          | 覆盖 state       |
| 数据          | replay buffer | policy mixture |
| exploration | bonus         | density ratio  |
| terminate   | done          | 必须 padding     |
| 训练方式        | 单策略           | 多策略            |

---

# 九、最终总结（一句话）

> CODEX.W =
> **用 density ratio 当 reward，逐层扩展 state coverage frontier**

---

# 如果你下一步要深入

我可以帮你做三件很有用的事：

1. 把这个算法改成 **diffusion policy版本（非常契合你现在的工作）**
2. 帮你分析它和你写的 CODEX-Embodied 伪代码差在哪
3. 帮你把 weight function 换成更稳定的实现（比如 energy-based）

你现在已经在一个很对的方向上了，这个算法其实就是你那套思路的理论版本。
