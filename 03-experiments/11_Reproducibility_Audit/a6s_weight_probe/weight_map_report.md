# A6-s 预训练权重静态映射报告（v2.6 三列版）

- torch `2.13.0+cpu` / torchvision `0.28.0+cpu`
- 目标模型参数量：**56,812,813**（`parameters()`；`state_dict()` 含 buffer 为 56,923,706）

## 三列总账（**不要只看一个数**）

| 列 | 定义 | 参数量 | 占比 | key 数 |
|---|---|---:|---:|---:|
| **A** exact-key | 同名+同shape | 2,786,112 | 4.90% | 30 |
| **B** semantic-mapped | 显式映射后shape相符 | 2,786,112 | 4.90% | 69 |
| **C** actually-loaded | 真正写入+前向 | 2,786,112 | 4.90% | 69 |

- **结论（与直觉相反）**：A → B 的参数量增量为 **0**，即**语义映射一个参数都没多搬**。4.90% 不是被命名空间低估的，而是**真实的结构上界**。原因见下节宽度对照。
- 前向验证：通过 [1, 4]
- ⚠️ 用 torchvision **随机初始化** 权重验证机制可行性；本机离线未下载 ImageNet，故本列**不**代表迁移后的精度

## 为什么语义映射没有提高比例（宽度对照，**推翻 E5 的前提**）

GPT 第五轮 E5 的判断是「4.90% 由'同名 key'算出，故系统性低估，建映射表即可提高」。**实测不成立** —— 建立显式映射后比例纹丝不动。

根因：DMSA 的瓶颈**宽度本身**与 ResNeXt50 不同。

| 位置 | ResNeXt50 | DMSA-Net |
|---|---|---|
| `layer1.0.conv1.weight` | [128, 64] | `layer1.0.path1_conv1.weight` → [64, 64] |
| `layer1.0.conv2.weight` | [128, 4] | `layer1.0.path2_conv2.0.weight` → [64, 1] |
| `layer1.0.conv3.weight` | [256, 128] | `layer1.0.path1_conv3.weight` → [128, 64] |
| `layer1.1.conv1.weight` | [128, 256] | `layer1.1.path1_conv1.weight` → [64, 256] |
| `layer1.1.conv2.weight` | [128, 4] | `layer1.1.path2_conv2.0.weight` → [64, 1] |
| `layer1.1.conv3.weight` | [256, 128] | `layer1.1.path1_conv3.weight` → [128, 64] |
| `layer1.2.conv1.weight` | [128, 256] | `layer1.2.path1_conv1.weight` → [64, 256] |
| `layer1.2.conv2.weight` | [128, 4] | `layer1.2.path2_conv2.0.weight` → [64, 1] |
| `layer1.2.conv3.weight` | [256, 128] | `layer1.2.path1_conv3.weight` → [128, 64] |
| `layer2.0.conv1.weight` | [256, 256] | `layer2.0.path1_conv1.weight` → [128, 256] |
| `layer2.0.conv2.weight` | [256, 8] | `layer2.0.path2_conv2.0.weight` → [128, 1] |
| `layer2.0.conv3.weight` | [512, 256] | `layer2.0.path1_conv3.weight` → [256, 128] |

即：**ResNeXt50 的瓶颈扩张比是 4×（64→128→256），DMSA `path1` 是 2×（64→64→128）**。
名字不同**且**形状不同 —— 语义映射能解决前者，解决不了后者。

> 这对本项目的意义：**不能再指望「补一张映射表」把预训练迁移做起来**。若真要做，只能 (a) 把 path1 加宽到与 ResNeXt50 一致（改变架构、使 fingerprint 失效、需重跑全部实验），或 (b) 接受从头训练。

## 列 D｜代码事实（决定性判据）

- 扫描文件：smoke_test.py, train_5fold.py, final_train.py, final_test.py
- 判定：**NO_IMAGENET_LOAD_PATH**
- 命中 `pretrained/weights=` 或 `models.xxx(` 的行数：**0**

> 未发现任何加载 ImageNet 预训练权重的构造式调用。
> （另有 4 处 `load_state_dict`，但那是加载本项目自己的 checkpoint，不构成预训练加载。）

## 语义映射表

| ResNeXt50 key | DMSA-Net key | shape 一致 |
|---|---|---|
| `conv1.weight` | `conv1.weight` | ✅ [64, 3, 7, 7] |
| `bn1.weight` | `bn1.weight` | ✅ [64] |
| `bn1.bias` | `bn1.bias` | ✅ [64] |
| `bn1.running_mean` | `bn1.running_mean` | ✅ [64] |
| `bn1.running_var` | `bn1.running_var` | ✅ [64] |
| `bn1.num_batches_tracked` | `bn1.num_batches_tracked` | ✅ [] |
| `layer1.0.bn1.num_batches_tracked` | `layer1.0.path1_bn1.num_batches_tracked` | ✅ [] |
| `layer1.0.bn2.num_batches_tracked` | `layer1.0.path2_conv2.1.num_batches_tracked` | ✅ [] |
| `layer1.0.bn3.num_batches_tracked` | `layer1.0.path1_bn3.num_batches_tracked` | ✅ [] |
| `layer1.0.downsample.0.weight` | `layer1.0.downsample.0.weight` | ✅ [256, 64, 1, 1] |
| `layer1.0.downsample.1.weight` | `layer1.0.downsample.1.weight` | ✅ [256] |
| `layer1.0.downsample.1.bias` | `layer1.0.downsample.1.bias` | ✅ [256] |
| `layer1.0.downsample.1.running_mean` | `layer1.0.downsample.1.running_mean` | ✅ [256] |
| `layer1.0.downsample.1.running_var` | `layer1.0.downsample.1.running_var` | ✅ [256] |
| `layer1.0.downsample.1.num_batches_tracked` | `layer1.0.downsample.1.num_batches_tracked` | ✅ [] |
| `layer1.1.bn1.num_batches_tracked` | `layer1.1.path1_bn1.num_batches_tracked` | ✅ [] |
| `layer1.1.bn2.num_batches_tracked` | `layer1.1.path2_conv2.1.num_batches_tracked` | ✅ [] |
| `layer1.1.bn3.num_batches_tracked` | `layer1.1.path1_bn3.num_batches_tracked` | ✅ [] |
| `layer1.2.bn1.num_batches_tracked` | `layer1.2.path1_bn1.num_batches_tracked` | ✅ [] |
| `layer1.2.bn2.num_batches_tracked` | `layer1.2.path2_conv2.1.num_batches_tracked` | ✅ [] |
| `layer1.2.bn3.num_batches_tracked` | `layer1.2.path1_bn3.num_batches_tracked` | ✅ [] |
| `layer2.0.bn1.num_batches_tracked` | `layer2.0.path1_bn1.num_batches_tracked` | ✅ [] |
| `layer2.0.bn2.num_batches_tracked` | `layer2.0.path2_conv2.1.num_batches_tracked` | ✅ [] |
| `layer2.0.bn3.num_batches_tracked` | `layer2.0.path1_bn3.num_batches_tracked` | ✅ [] |
| `layer2.0.downsample.0.weight` | `layer2.0.downsample.0.weight` | ✅ [512, 256, 1, 1] |
| `layer2.0.downsample.1.weight` | `layer2.0.downsample.1.weight` | ✅ [512] |
| `layer2.0.downsample.1.bias` | `layer2.0.downsample.1.bias` | ✅ [512] |
| `layer2.0.downsample.1.running_mean` | `layer2.0.downsample.1.running_mean` | ✅ [512] |
| `layer2.0.downsample.1.running_var` | `layer2.0.downsample.1.running_var` | ✅ [512] |
| `layer2.0.downsample.1.num_batches_tracked` | `layer2.0.downsample.1.num_batches_tracked` | ✅ [] |
| `layer2.1.bn1.num_batches_tracked` | `layer2.1.path1_bn1.num_batches_tracked` | ✅ [] |
| `layer2.1.bn2.num_batches_tracked` | `layer2.1.path2_conv2.1.num_batches_tracked` | ✅ [] |
| `layer2.1.bn3.num_batches_tracked` | `layer2.1.path1_bn3.num_batches_tracked` | ✅ [] |
| `layer2.2.bn1.num_batches_tracked` | `layer2.2.path1_bn1.num_batches_tracked` | ✅ [] |
| `layer2.2.bn2.num_batches_tracked` | `layer2.2.path2_conv2.1.num_batches_tracked` | ✅ [] |
| `layer2.2.bn3.num_batches_tracked` | `layer2.2.path1_bn3.num_batches_tracked` | ✅ [] |
| `layer2.3.bn1.num_batches_tracked` | `layer2.3.path1_bn1.num_batches_tracked` | ✅ [] |
| `layer2.3.bn2.num_batches_tracked` | `layer2.3.path2_conv2.1.num_batches_tracked` | ✅ [] |
| `layer2.3.bn3.num_batches_tracked` | `layer2.3.path1_bn3.num_batches_tracked` | ✅ [] |
| `layer3.0.bn1.num_batches_tracked` | `layer3.0.path1_bn1.num_batches_tracked` | ✅ [] |
| `layer3.0.bn3.num_batches_tracked` | `layer3.0.path1_bn3.num_batches_tracked` | ✅ [] |
| `layer3.0.downsample.0.weight` | `layer3.0.downsample.0.weight` | ✅ [1024, 512, 1, 1] |
| `layer3.0.downsample.1.weight` | `layer3.0.downsample.1.weight` | ✅ [1024] |
| `layer3.0.downsample.1.bias` | `layer3.0.downsample.1.bias` | ✅ [1024] |
| `layer3.0.downsample.1.running_mean` | `layer3.0.downsample.1.running_mean` | ✅ [1024] |
| `layer3.0.downsample.1.running_var` | `layer3.0.downsample.1.running_var` | ✅ [1024] |
| `layer3.0.downsample.1.num_batches_tracked` | `layer3.0.downsample.1.num_batches_tracked` | ✅ [] |
| `layer3.1.bn1.num_batches_tracked` | `layer3.1.path1_bn1.num_batches_tracked` | ✅ [] |
| `layer3.1.bn3.num_batches_tracked` | `layer3.1.path1_bn3.num_batches_tracked` | ✅ [] |
| `layer3.2.bn1.num_batches_tracked` | `layer3.2.path1_bn1.num_batches_tracked` | ✅ [] |
| `layer3.2.bn3.num_batches_tracked` | `layer3.2.path1_bn3.num_batches_tracked` | ✅ [] |
| `layer3.3.bn1.num_batches_tracked` | `layer3.3.path1_bn1.num_batches_tracked` | ✅ [] |
| `layer3.3.bn3.num_batches_tracked` | `layer3.3.path1_bn3.num_batches_tracked` | ✅ [] |
| `layer3.4.bn1.num_batches_tracked` | `layer3.4.path1_bn1.num_batches_tracked` | ✅ [] |
| `layer3.4.bn3.num_batches_tracked` | `layer3.4.path1_bn3.num_batches_tracked` | ✅ [] |
| `layer3.5.bn1.num_batches_tracked` | `layer3.5.path1_bn1.num_batches_tracked` | ✅ [] |
| `layer3.5.bn3.num_batches_tracked` | `layer3.5.path1_bn3.num_batches_tracked` | ✅ [] |
| `layer4.0.bn1.num_batches_tracked` | `layer4.0.path1_bn1.num_batches_tracked` | ✅ [] |
| `layer4.0.bn3.num_batches_tracked` | `layer4.0.path1_bn3.num_batches_tracked` | ✅ [] |
| `layer4.0.downsample.0.weight` | `layer4.0.downsample.0.weight` | ✅ [2048, 1024, 1, 1] |
| `layer4.0.downsample.1.weight` | `layer4.0.downsample.1.weight` | ✅ [2048] |
| `layer4.0.downsample.1.bias` | `layer4.0.downsample.1.bias` | ✅ [2048] |
| `layer4.0.downsample.1.running_mean` | `layer4.0.downsample.1.running_mean` | ✅ [2048] |
| `layer4.0.downsample.1.running_var` | `layer4.0.downsample.1.running_var` | ✅ [2048] |
| `layer4.0.downsample.1.num_batches_tracked` | `layer4.0.downsample.1.num_batches_tracked` | ✅ [] |
| `layer4.1.bn1.num_batches_tracked` | `layer4.1.path1_bn1.num_batches_tracked` | ✅ [] |
| `layer4.1.bn3.num_batches_tracked` | `layer4.1.path1_bn3.num_batches_tracked` | ✅ [] |
| `layer4.2.bn1.num_batches_tracked` | `layer4.2.path1_bn1.num_batches_tracked` | ✅ [] |
| `layer4.2.bn3.num_batches_tracked` | `layer4.2.path1_bn3.num_batches_tracked` | ✅ [] |
| `layer1.0.conv1.weight` | `layer1.0.path1_conv1.weight` | ❌ [128, 64, 1, 1] vs [64, 64, 1, 1] |
| `layer1.0.bn1.weight` | `layer1.0.path1_bn1.weight` | ❌ [128] vs [64] |
| `layer1.0.bn1.bias` | `layer1.0.path1_bn1.bias` | ❌ [128] vs [64] |
| `layer1.0.bn1.running_mean` | `layer1.0.path1_bn1.running_mean` | ❌ [128] vs [64] |
| `layer1.0.bn1.running_var` | `layer1.0.path1_bn1.running_var` | ❌ [128] vs [64] |
| `layer1.0.conv2.weight` | `layer1.0.path2_conv2.0.weight` | ❌ [128, 4, 3, 3] vs [64, 1, 3, 3] |
| `layer1.0.bn2.weight` | `layer1.0.path2_conv2.1.weight` | ❌ [128] vs [64] |
| `layer1.0.bn2.bias` | `layer1.0.path2_conv2.1.bias` | ❌ [128] vs [64] |
| `layer1.0.bn2.running_mean` | `layer1.0.path2_conv2.1.running_mean` | ❌ [128] vs [64] |
| `layer1.0.bn2.running_var` | `layer1.0.path2_conv2.1.running_var` | ❌ [128] vs [64] |
| `layer1.0.conv3.weight` | `layer1.0.path1_conv3.weight` | ❌ [256, 128, 1, 1] vs [128, 64, 1, 1] |
| `layer1.0.bn3.weight` | `layer1.0.path1_bn3.weight` | ❌ [256] vs [128] |
| `layer1.0.bn3.bias` | `layer1.0.path1_bn3.bias` | ❌ [256] vs [128] |
| `layer1.0.bn3.running_mean` | `layer1.0.path1_bn3.running_mean` | ❌ [256] vs [128] |
| `layer1.0.bn3.running_var` | `layer1.0.path1_bn3.running_var` | ❌ [256] vs [128] |
| `layer1.1.conv1.weight` | `layer1.1.path1_conv1.weight` | ❌ [128, 256, 1, 1] vs [64, 256, 1, 1] |
| `layer1.1.bn1.weight` | `layer1.1.path1_bn1.weight` | ❌ [128] vs [64] |
| `layer1.1.bn1.bias` | `layer1.1.path1_bn1.bias` | ❌ [128] vs [64] |
| `layer1.1.bn1.running_mean` | `layer1.1.path1_bn1.running_mean` | ❌ [128] vs [64] |
| `layer1.1.bn1.running_var` | `layer1.1.path1_bn1.running_var` | ❌ [128] vs [64] |

## 判定

### `CLAIM_UNSUPPORTED_NO_LOAD_PATH`

训练/建模脚本中**不存在**加载 ImageNet 预训练权重的代码路径。「无缝加载 ResNeXt50」在实现层不成立 —— 此判定**不依赖** A/B 两列的匹配率。

> **判定规则（事前写死）**：列 D 为决定性判据；只要代码里没有 ImageNet 加载路径，「无缝加载」即不成立，**与 A/B 两列的比例无关**。

## 对论文的直接影响

- 现基线 `_init_weights()` 为 kaiming 从头初始化，且无预训练加载路径，
  **代码事实支持「本模型从头训练，未使用预训练」**。
- 若仍要保留「双路径解决知识迁移权衡」这一贡献，
  必须先真正实现映射表加载（列 B/C 已给出上界与可行性），
  再训练对照 —— 那是 A6-t（可选训练，见路线 §5.4）。
- 论文 Table 1 的 8 个对比模型是否用了预训练，PDF 中完全未报告；
  若基线用而 DMSA 不用，「+5.0 个百分点」是配置不对等的对比。