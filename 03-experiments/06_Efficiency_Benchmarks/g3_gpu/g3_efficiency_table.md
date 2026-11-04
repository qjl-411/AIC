# DMSA-Net 效率指标实测

- 设备：`NVIDIA GeForce RTX 4090`（torch 2.2.2+cu121）
- 架构指纹：`8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975`（与基线一致 ✅）
- FLOPs 口径：输入 1x3x224x224，batch=1；1 MAC = 2 FLOPs；只统计 Conv2d 与 Linear；BN / 激活 / 池化不计入。

| 指标 | 数值 | 状态 |
|---|---:|---|
| 参数量 | 56,812,813 | ✅ 实测 |
| 对照 ResNeXt50-32x4d 参数量 | 25,028,904 | ✅ 实测 |
| 参数量比值（DMSA / ResNeXt） | 2.27× | ✅ |
| FLOPs (batch=1, 224²) | 14.985 G | ✅ 实测 |
| MACs | 7.492 G | ✅ 实测 |
| CPU 单图延迟 | 74.0 ms | ✅ 实测（仅相对） |
| **GPU 延迟 (bs=1)** | **11.31 ms** | ✅ 实测 |
| **GPU 吞吐 (bs=1)** | **88.4 img/s** | ✅ 实测 |
| **GPU 峰值显存 (bs=1)** | **370.3 MB** | ✅ 实测 |
| **GPU 延迟 (bs=1)** | **11.31 ms** | ✅ 实测 |
| **GPU 吞吐 (bs=1)** | **88.4 img/s** | ✅ 实测 |
| **GPU 峰值显存 (bs=1)** | **370.3 MB** | ✅ 实测 |

> GPU 三项只在有 CUDA 的机器上可测。本脚本不会用 CPU 数值外推 GPU 性能——这是项目铁律。若在 GPU 机器上运行，请把 --device cuda:0 且确认 torch.cuda.is_available() 为 True。