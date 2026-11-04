# 阶段 3.1–3.2 / 4.1–4.2 架构探针报告

- torch `2.13.0+cpu`／CUDA `None`／可用 `False`

## 3.1 参数量与 fingerprint

| 指标 | 实测 | 基线常量 | 一致 |
|---|---|---|---|
| 参数量 | 56,812,813 | 56,812,813 | ✅ |
| fingerprint | `8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975` | `8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975` | ✅ |

> 论文写的 25,689,412 已失效；以本表实测为准。

## 3.3 参数归因

| 模块 | 参数量 | 占比 |
|---|---|---|
| edpb_fusion_conv | 21,299,200 | 37.49% |
| edpb_path2_mdc | 10,630,763 | 18.71% |
| edpb_path1 | 7,581,952 | 13.35% |
| edpb_path2_conv1_3 | 6,867,072 | 12.09% |
| ilf_fusion34 | 5,246,977 | 9.24% |
| downsample | 2,776,576 | 4.89% |
| ilf_fusion23 | 1,312,769 | 2.31% |
| head_fc | 1,051,140 | 1.85% |
| edpb_fusion_bn | 36,352 | 0.06% |
| backbone_other | 9,553 | 0.02% |
| spatial_attention | 459 | 0.00% |

> **EDPB `fusion_conv`+`fusion_bn` 合计 21,335,552（37.55%）**—— 论文 Fig.4 与正文均未单独列出此项，
> 它是「参数量对不上」的主因。

## 3.2 FLOPs / 吞吐

- FLOPs(batch=1, 224×224) = **14.985 GFLOPs**（MACs 7.492 G）
- CPU 前向 140.0 ms/img（**仅相对参考**，GPU 吞吐须在云环境重测）

**FLOPs 计数约定**（M4；跨论文比较必须绑定这些，否则同一模型会出现两个 FLOPs）：

- 输入 `1x3x224x224`，batch = 1
- 定义：1 MAC = 2 FLOPs（本报告 FLOPs = 2 × MACs）
- 计入：仅 Conv2d 与 Linear 的乘加
- **不计入**：BN / ReLU / Swish / MaxPool / AvgPool / AdaptiveAvgPool / attention 内的 elementwise
- 工具：torch forward hook（自实现，不依赖 fvcore/thop）；model.eval() + torch.no_grad()

| Top 模块 | MACs |
|---|---|
| `layer1.0.fusion_conv` | 0.206 G |
| `layer1.1.fusion_conv` | 0.206 G |
| `layer1.2.fusion_conv` | 0.206 G |
| `layer2.0.fusion_conv` | 0.206 G |
| `layer2.1.fusion_conv` | 0.206 G |
| `layer2.2.fusion_conv` | 0.206 G |

## 4.1 开关注入与 4.2 不变量

- `full` 变体 fingerprint 与基线一致：**✅**

| 变体 | 参数量 | Δ | 前向可跑 | fp=基线 |
|---|---|---|---|---|
| full | 56,812,813 | +0 | ✅ | ✅ |
| no_sam | 56,812,354 | -459 | ✅ | — |
| no_mdc | 45,037,801 | -11,775,012 | ✅ | — |
| no_ilf | 55,495,947 | -1,316,866 | ✅ | — |
| no_dual_path | 46,753,037 | -10,059,776 | ✅ | — |

> 纪律：`full` 必须与基线 fp 一致，否则本注入不得用于正式消融。
> 关闭某个创新点时参数量会变化（如 no_mdc 去掉空洞卷积分支），
> 因此 §5.2 要求**容量匹配对照**必须同时报 params/FLOPs。