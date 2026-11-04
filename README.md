# DMSA-Net：阿尔茨海默病 MRI 病程分期

> **2026 年第八届全球校园人工智能算法精英大赛 · 算法创新赛（赛题 1：算法模型创新）**
> 团队：脑影溯析 ｜ 团队编号：AIC-2026-95340465 ｜ 作品：DMSA-Net — AD-MRI 病程分期

本仓库收录 DMSA-Net 的全部可公开材料：**算法代码**、**临床辅助分析平台（NeuroEvo-AD / 脑影溯析）**、**实验原始数据与模型产物**、**参赛文档**。

核心主张只有一句：**这里的每个数字都能被指到具体文件，不跑代码也能核对。**

---

## 一、项目速览

| 项 | 内容 |
|---|---|
| 任务 | 结构 MRI 四分类病程分期：No Impairment / Very Mild / Mild / Moderate Impairment |
| 数据集 | 公开数据集 **Alzheimer's MRI Dataset**（`Combined Dataset/{train}/<四类>`）；原始 128×128 灰度，训练按 224×224 RGB 读取 |
| 划分协议 | P-CV v2 **分组五折**（按源图分组，抑制同源泄漏）；训练 10,240 张，每折 8,192 训练 / 2,048 验证（每类 512） |
| 最终训练 | 全量 10,240 张，固定 **126 epochs** |
| 封存测试 | **1,279 张**，一次性评测，不参与任何调参 |
| 运行环境 | PyTorch 2.2.2+cu121 / Python 3.11.8 / CUDA 12.1 / NVIDIA RTX 4090 |

---

## 二、核心结果

### 2.1 封存测试集（1,279 张，`evaluated_at: 2026-08-26`）

| 指标 | 数值 |
|---|---|
| Accuracy | **98.98 %**（1,266 / 1,279） |
| Precision（macro） | 99.32 % |
| Recall（macro） | 99.26 % |
| F1（macro） | **99.29 %** |
| Cohen's Kappa | 98.32 % |
| Test Loss | 0.0607 |

分类型表现：

| 类别 | 样本数 | 精确率 | 召回率 | F1 |
|---|---:|---:|---:|---:|
| No Impairment | 640 | 98.76 % | 99.38 % | 99.07 % |
| Very Mild Impairment | 448 | 99.10 % | 98.21 % | 98.65 % |
| Mild Impairment | 179 | 99.44 % | 99.44 % | 99.44 % |
| Moderate Impairment | 12 | 100.00 % | 100.00 % | 100.00 % |

> 数据出处：`03-experiments/01_Sealed_Test_Evaluation/test_metrics.json`
> 该文件同时记录了模型权重 SHA-256 与测试清单 SHA-256，可核验「评的就是这一份权重、这一份数据」。
> **注意**：Moderate Impairment 仅 12 张样本，其 100 % 指标不具备统计意义，不应单独引用。

### 2.2 效率（RTX 4090，实测，非估算）

| 指标 | DMSA-Net | 对照 ResNeXt50-32x4d |
|---|---:|---:|
| 参数量 | 56,812,813 | 25,028,904（DMSA 为 2.27×） |
| FLOPs（1×3×224×224, bs=1） | 14.985 G | — |
| GPU 单图延迟 | **11.31 ms** | — |
| GPU 吞吐 | **88.4 img/s** | — |
| GPU 峰值显存 | **370.3 MB** | — |

> 数据出处：`03-experiments/06_Efficiency_Benchmarks/g3_gpu/g3_efficiency_table.md`
> FLOPs 口径：1 MAC = 2 FLOPs，仅统计 Conv2d 与 Linear，不含 BN / 激活 / 池化。

### 2.3 模块消融（方向性结论，非显著性结论）

| 变体 | 参数量 | val 准确率（2 折均值） | Δ vs full |
|---|---:|---:|---:|
| `full` | 56,812,813 | 99.170 % | — |
| `no_sam`（去空间注意力） | 56,812,354 | 98.877 % | −0.293 pp |
| `no_mdc`（去多尺度空洞卷积） | 45,037,801 | 98.828 % | −0.342 pp |
| `no_ilf`（去层间融合） | 55,495,947 | 26.147 % | −73.022 pp |

> 数据出处：`03-experiments/04_Ablation_Studies/g1_ablation_table_with_delta.md`
> **效力边界（原文声明，请一并引用）**：统计单位仅 **2 折（fold 1 / 5）**，n=2 不做显著性检验，
> 因此只能表述为「方向上变化 X pp」，**不得**写成「显著提升/下降」。
> `no_ilf` 各折 best epoch 均为 2、准确率接近随机（≈25 %），结论只能是「**移除 ILF 后训练不收敛**」，
> 不能直接主张「ILF 机制不可或缺」——该表已明确标注这一区分。

---

## 三、仓库结构

```
.
├── 01-algorithm-code/     # 算法代码（A/B/C/D 四组，含复现手册与验收报告）
├── 02-platform/           # 平台代码 NeuroEvo-AD（FastAPI + React 19 + three.js）
├── 03-experiments/        # 实验数据与模型产物（11 个分类目录，逐目录带 README）
├── 04-docs/               # 参赛文档（技术报告 / 佐证材料 / 答辩 PPT / 作品简介）
├── 05-demo/               # 演示视频说明（视频文件未入库，见第五节）
└── 目录结构.md            # 原始提交包的完整目录树（逐文件列出）
```

| 目录 | 内容 | 体积 |
|---|---|---:|
| `01-algorithm-code/` | 48 个脚本：`A01–A05` 训练与评测入口、`B06–B21` 数据与泄漏审计、`C22–C38` 补充实验、`D39–D42` 运行手册 | 4.9 MB |
| `02-platform/` | `NeuroEvo_AD/`：`apps/api`（FastAPI，含 DMSA-Net 推理与文献库）、`apps/web`（React 19 + Vite + three.js）、`deploy/`、`docs/` | 65 MB |
| `03-experiments/` | 封存测试、五折 CV、最终训练、消融、多种子、效率、ROC/校准、权重清单、图表、原始日志、复现审计 | 222 MB |
| `04-docs/` | `technical-report/`、`evidence/`、`defense-slides/`、`intro/` | 108 MB |

**建议入口**（按目的选一条）：

- 只想核结论 → `03-experiments/数据索引与验证指南.md`（把文档里每个数字指到具体文件）
- 想跑算法 → `01-algorithm-code/README.md` 与 `D39_RUNBOOK_GPU_Package.md`
- 想跑平台 → `02-platform/README.md`（含权重挂载与前端启动）
- 想核数据来源与泄漏 → `03-experiments/11_Reproducibility_Audit/`

---

## 四、快速开始

### 4.1 复现训练与评测（需 NVIDIA GPU）

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 pillow tqdm
```

数据准备：按 `Combined Dataset/train/<四类>` 放置公开数据集影像，路径由脚本顶部 `DATA_ROOT` 或 `--data-root` 指定。
划分清单 `pcv_v2_folds_seed42_FROZEN.csv` 已随包提供（`01-algorithm-code/_frozen_manifests_keep_original_name/`），可据此校验划分一致性。

```bash
python A01_train_final_model.py        # 最终全量训练（126 epochs）
python A02_evaluate_sealed_test.py     # 封存测试集评测（仅一次）
python A03_train_five_fold_cv.py       # 五折交叉验证
python A04_smoke_test_pipeline.py      # 冒烟测试：先验证接线
```

### 4.2 只核数据，不跑训练（纯 CPU）

```bash
python B19_run_audit_pipeline.py --root ..    # 一键执行审计全流程
python B20_test_acceptance_criteria.py        # 93 项执行前验收（仅需 NumPy + Pillow）
```

### 4.3 启动平台

```bash
# 后端
cd 02-platform/NeuroEvo_AD/apps/api
pip install -r requirements.txt
uvicorn app.main:app --port 8000

# 前端
cd 02-platform/NeuroEvo_AD/apps/web
npm install && npm run dev
```

平台加载的模型权重与环境变量配置见 `02-platform/README.md`。

---

## 五、未入库的大文件

以下两个文件超过 **GitHub 单文件 100 MB 硬限制**，未纳入本仓库。SHA-256 一并给出，可用于校验你手上副本是否与实验一致。

| 文件 | 大小 | SHA-256 | 说明 |
|---|---:|---|---|
| `final_model_epoch126.pth` | 217.5 MB | `ba96575888e35764e3ff196674e52757d9a5f9c90742a38aa8f1d1f9a8457658` | 最终模型权重，即封存测试所用权重 |
| `AIC-2026-95340465-…-AD-MRI病程分期.mp4` | 142.9 MB | `47b8489ac85fab8e4a362edfe1d887d5552459b2aad00d2c2c0a7145488651db` | 作品演示视频 |

权重的配套元信息已入库，可在不下载权重的情况下核对：

- `03-experiments/08_Model_Checkpoints_and_Artifacts/checkpoint.sha256.txt` — 权重 SHA-256
- `03-experiments/08_Model_Checkpoints_and_Artifacts/model_manifest.json` — 权重哈希、参数量、架构指纹
- `03-experiments/03_Final_Training_Run/` — 训练协议、逐轮历史、代码快照，可据此重新训练出同构权重

另有一类内容按原项目 `.gitignore` 声明排除：

- `02-platform/NeuroEvo_AD/apps/api/data/pdfs/` — 336 篇文献全文 PDF，约 899 MB。
  这是第三方文献的本地副本，元数据库 `neuroevo_papers.db`（902 篇）已入库，全文可用 `LOCALIZE_LITERATURE_PDFS.sh` 重新下载。

---

## 六、复现与可信边界

这一节列出**已知的、尚未消除的限制**，不做粉饰。判断本项目结论的适用范围时请一并考虑。

1. **消融只有 2 折，无置信区间**。n=2 不做显著性检验，`g1` 表只能给方向性结论。项目自身铁律要求「统计单位 = 源图、CI = cluster bootstrap」，该表两项都不满足。
2. **消融参照与实验有 3 处未对齐**：entry script 不同（`train_5fold.py` vs `smoke_test.py`）、device 不同（RTX 3060 vs 4090）、`deterministic=false`。Δ 中含硬件与非确定性噪声，无法完全归因到模块语义。
3. **`no_mdc` / `no_ilf` 的容量不可比**。`no_mdc` ΔParams = −11,775,012，其中 MDC 自身仅 10,623,003，余下 1,152,009 来自相邻层通道变化，故不可归因单一模块。
4. **`no_sam` 两折准确率完全相同**（均 2025/2048），两折 best epoch / elapsed 不同说明非复制错误，但 `fold_1` 的 `metrics.csv` 已缺失，无法逐 epoch 复核，标注存疑。
5. **数据集本体未随包提供**。划分清单已提供，可校验划分一致性；但「两份图片是否逐字节同一批」未验证。
6. **平台与实验共用同一份权重**（同一 SHA-256），不存在两套模型；但平台的临床报告生成链路只做工程验证，**未做临床可用性验证**。

---

## 七、材料清单

| 文件 | 位置 |
|---|---|
| 作品方案 / 技术报告（docx） | `04-docs/technical-report/2026AIC-算法创新赛-DMSA-net网络算法作品方案.docx` |
| 技术报告（PDF） | `04-docs/technical-report/2026AIC-算法创新赛-DMSA-net网络算法作品技术报告.pdf` |
| 佐证材料（docx / PDF ×2） | `04-docs/evidence/` |
| 答辩 PPT（pptx / PDF） | `04-docs/defense-slides/` |
| 作品简介 | `04-docs/intro/脑影溯析.pdf` |
| 原始提交包完整目录树 | `目录结构.md` |

---

## 八、说明

- 部分目录保留原提交包的中文文件名与 `SHA256SUMS.txt` 校验清单，**文件名与内容均未改动**，可直接在目录内执行 `sha256sum -c SHA256SUMS.txt` 自检。
- 本仓库顶层目录已按材料类型重组（原 `-01/-02/-03` 长目录名改为英文短名），目录内部结构与文件哈希保持不变。
- 训练与评测在云端 GPU 完成，仓库中的 `*.log` 为原始控制台输出，未做裁剪。

---

*团队：脑影溯析 ｜ AIC-2026-95340465*
