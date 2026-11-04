# DMSA-Net 效率指标证据表

> 全部数字来源：`DMSA_v24/_arch_out/arch_probe.json`（本项目自实现 torch forward hook）。
> 本表**不新增任何未经测量的数字**。

## 1. 复杂度指标（实测）

| 指标 | DMSA-Net | ResNeXt50-32x4d | 说明 |
|---|---:|---:|---|
| 参数量 | **56,812,813** | 25,028,904 | ResNeXt50 为 torchvision 结构实测 |
| 参数量比值 | — | — | DMSA-Net 为 ResNeXt50 的 **2.27x** |
| FLOPs (batch=1, 224²) | **14.985 G** | 8.461 G | 1 MAC = 2 FLOPs |
| MACs (batch=1, 224²) | **7.492 G** | 4.230 G | — |
| FLOPs 绝对数字 | 14,984,620,662 | 8,460,959,744 | 见下方勘误 |
| CPU 单张前向 | 140.0 ms | 未测 | **仅相对参考**，非部署指标（本机 CPU-only） |
| GPU 延迟（batch=1） | **11.31 ms** | 未测 | RTX 4090 实测中位数，warmup 10 / 50 次（2026-09-24 云上实测） |
| GPU 吞吐（batch=1） | **88.40 img/s** | 未测 | 同上 |
| GPU 峰值显存（batch=1） | **370.31 MB** | 未测 | 占该卡可用显存 24,109.7 MB 的 1.54% |

> ✅ **8/8 项全部 MEASURED**（`g3_efficiency.json` 的 `summary.n_measured = 8`），
> GPU 三项不再是「未测」——**「待云实测」的标注可以撤销**。

### ⚠️ FLOPs 勘误（2026-09-23 独立复核）

用独立实现的 hook 计数器（`_aic_boost/gpu/g3_efficiency.py`）重跑，
**MACs 精确复现 7,492,310,331**，但发现已发布字段的笔误：

| 项 | 值 |
|---|---:|
| MACs（复核后，可信） | 7,492,310,331 |
| 正确 FLOPs = 2 × MACs | **14,984,620,662** |
| 先前发布的 `arch_probe.json` 字段 | 14,985,420,662 ❌ |
| 差值 | 800,000 |
| 先前值 / MACs | 2.000106776…（不是 2.0） |

**结论**：`arch_probe.py` 的 `FlopCounter.flops` 属性本身正确（`return 2 * self.macs`），
**错误只出现在落盘的 `_arch_out/arch_probe.json` 字段里**。

对方案的影响：
- 若写 **14.985 GFLOPs** → 仍然正确（四舍五入后不变），无需改动。
- 若写**绝对数字** → 必须用 **14,984,620,662**，不要用 14,985,420,662。

## 2. FLOPs 计数约定（跨论文比较必须绑定）

| 项 | 取值 |
|---|---|
| input_shape | 1x3x224x224 |
| batch_size | 1 |
| flop_definition | 1 MAC = 2 FLOPs（本报告 FLOPs = 2 × MACs） |
| counted_ops | 仅 Conv2d 与 Linear 的乘加 |
| not_counted | BN / ReLU / Swish / MaxPool / AvgPool / AdaptiveAvgPool / attention 内的 elementwise |
| method | torch forward hook（自实现，不依赖 fvcore/thop） |
| profiler | arch_probe.FlopCounter (torch 2.13.0+cpu) |
| mode | model.eval() + torch.no_grad() |

> ⚠️ **FLOPs 没有绝对唯一数字。** 不绑定上表约定的 FLOPs 数字不具备可比性。

## 3. 结构消融的容量变化（已实测，尚未训练）

| 变体 | 参数量 | Δ vs full | fingerprint=基线 | 前向可跑 |
|---|---:|---:|:---:|:---:|
| `full` | 56,812,813 | 0 | ✅ | ✅ |
| `no_sam` | 56,812,354 | -459 | — | ✅ |
| `no_mdc` | 45,037,801 | -11,775,012 | — | ✅ |
| `no_ilf` | 55,495,947 | -1,316,866 | — | ✅ |
| `no_dual_path` | 46,753,037 | -10,059,776 | — | ✅ |

> 上表证明**开关注入机制可用且 `full` 与基线结构完全一致**（硬约束）。
> 但**这 4 个变体都还没有训练过** —— 没有精度数字，因此**不能**据此写任何精度结论。

## 4. 测量环境声明

### 4.1 本机（CPU 部分，2026-09-23）

| 项 | 值 |
|---|---|
| 平台 | Windows-11-10.0.26200-SP0 |
| Python | 3.14.5 |
| torch | 2.13.0+cpu |
| CUDA 可用 | False |
| GPU | 无 |

> 结论：**本机不具备 GPU 效率测量能力**，此前 CPU 数字**不得外推为 GPU 性能**。

### 4.2 云端（GPU 三项，2026-09-24 实测）

| 项 | 值 |
|---|---|
| GPU | **NVIDIA GeForce RTX 4090**（capability 8.9，总显存 25,280,839,680 B） |
| CUDA / cuDNN | CUDA 12.1 / cuDNN 8902 |
| Python / 平台 | 3.11.8 / Linux 5.15.0-105-generic x86_64 (glibc 2.35) |
| torch / torchvision | 2.2.2+cu121 / 0.17.2+cu121 |
| 计时口径 | `model.eval()` + `no_grad`，warmup 10 次，正式 50 次，报**中位数**；批大小 1 |
| 计时分布 | Q1 11.289 ms / 中位数 11.313 ms / Q3 11.353 ms / min 11.244 ms / P95 11.490 ms |
| 架构指纹 | `8e542321…d76975` ✅ **与基线一致**（同一模型，非另一份结构） |
| 清单 SHA-256 | `51f5b251…edfd` ✅ |
| 产物 | `cloud_results/20260925_G1/AIC_train_again/g3/g3_efficiency.json` |

> ⚠️ **两个 CPU 数字不可混用**：本机 140.0 ms（torch 2.13.0+cpu / Windows）与云端 74.0 ms
> （torch 2.2.2 / Linux）是**不同平台、不同 torch 版本**的结果，本表 §1 的 CPU 行沿用本机值，
> 不改动，也不与云上值比较。
