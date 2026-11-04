# 01 算法代码 · 使用说明

本目录是 DMSA-Net（阿尔茨海默病 MRI 四分类病程分期网络）的**全部算法代码**：从数据检查、网络结构、五折交叉验证、最终训练、封存测试，到复现审计与补充实验（效率 / 多种子 / 消融 / 对照五折）。

代码按功能分组编号，可直接定位：

| 前缀 | 分组 | 说明 |
|---|---|---|
| `A01–A05` | 基线与训练 / 评测入口 | 复现主实验**只需要这一组** |
| `B06–B21` | 复现与数据审计 | 核数据来源、重复、泄漏、参数量等结论（多数纯 CPU） |
| `C22–C38` | 补充实验（GPU 包） | 效率、多种子、消融、对照五折等提分实验 |
| `D39–D41` | 说明文档 | 运行手册、验收报告 |

---

## 一、运行环境（实测记录，非估算）

以下为本实验**实际运行**时的环境记录（取自实验数据包中的 `environment.json`）：

| 项 | 实测值 |
|---|---|
| 操作系统 | Linux（Ubuntu 22.04，glibc 2.35） |
| Python | 3.11.8 |
| PyTorch | 2.2.2+cu121 |
| TorchVision | 0.17.2+cu121 |
| NumPy | 1.26.4 |
| CUDA / cuDNN | 12.1 / 8902 |
| GPU | NVIDIA GeForce RTX 4090（24 GB）；五折训练用 RTX 3060（12 GB）亦可 |
| 磁盘 | 建议 ≥ 50 GB（数据集 + 各折权重） |

**安装依赖**（CUDA 12.1）：

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 pillow tqdm
```

- **A 组 / C 组**：需要 GPU（无 GPU 时可先跑 `A04_smoke_test_pipeline.py` 验证接线）。
- **B 组**：多数只用 NumPy + Pillow，**纯 CPU 即可**，用于核对数据与结论。

---

## 二、代码清单（用途 / 核心能力）

### A 组 · 基线与训练评测入口（复现主实验只需这一组）

| 文件 | 用途 | 核心能力 | 硬件 | 主要产出 |
|---|---|---|---|---|
| `A01_train_final_model.py` | 最终全量训练 | 在完整 10,240 张训练集上固定训练 **126 epochs**；内含网络结构定义（Swish 激活、空间注意力、多尺度空洞卷积、增强双路径瓶颈、层间融合） | GPU | 最终权重、逐轮训练历史 |
| `A02_evaluate_sealed_test.py` | 封存测试单次评价 | 加载最终权重，对 **1,279 张封存测试集评测且仅评测一次**；评测前做数据完整性校验，输出混淆矩阵与 accuracy / F1 / Kappa | GPU | `test_metrics.json`、混淆矩阵 |
| `A03_train_five_fold_cv.py` | 五折交叉验证 | P-CV v2 分组五折（每折 8,192 训练 / 2,048 验证，每类 512），200 epochs；内置 `STRICT_BASELINE_CONFIG` 防误改基线超参 | GPU | 各折 metrics、cv_summary、权重 |
| `A04_smoke_test_pipeline.py` | 冒烟测试 | 用最小 batch 跑通全流程，验证环境与模块接线（含共享网络结构实现） | CPU / GPU | 通过 / 失败 |
| `A05_check_dataset_preparation.py` | 数据准备检查 | 校验数据集解压结果与划分是否与 manifest 一致（sha256 校验） | CPU | 检查报告 |

> ⚠️ `A05` 的 `main()` 会触发解压，日常只需调用其中的只读检查函数即可。

### B 组 · 复现与数据审计（核结论用，多数纯 CPU）

| 文件 | 用途 | 核心能力 |
|---|---|---|
| `B06_audit_dataset_inventory.py` | 阶段 0.0 输入盘点 | 逐项检查各任务的**前置输入**是否齐备，缺失直接标记 `blocked_missing_inputs` |
| `B07_audit_dataset_verification.py` | 数据集到货验证 | 验证到手数据集与声明是否一致（帧几何、类别分布、文件哈希） |
| `B08_audit_duplicate_detector_power.py` | 重复检测器判别力实测 | 实测 pHash 等检测器在真实数据上的可用工作点（含阴性结论边界） |
| `B09_audit_source_group_mapping.py` | 源图映射 / 谱系候选 | 生成同源分组候选，用于分析组内冗余 |
| `B10_audit_detector_calibration.py` | 阈值与变换集标定 | 用合成样本对检测器阈值做三态划分与留出集标定 |
| `B11_audit_cross_dataset_leakage.py` | 训练集 ↔ 测试集近重复审计 | 审计跨集近重复（对齐 SSIM），给出泄漏判定 |
| `B12_audit_pretrained_weight_probe.py` | 预训练权重静态映射探测 | **0 GPU**，判断模型是否真的加载了 ImageNet 预训练权重 |
| `B13_probe_architecture_fingerprint.py` | 架构指纹与 FLOPs 探针 | 实测**参数量 / FLOPs / 吞吐**，并支持双路径塌缩等结构变体对照 |
| `B14_audit_intra_group_leakage.py` | 结构泄漏审计 + ESS/DEFF | 纯 manifest 计算，**0 图像、0 GPU、0 重训** |
| `B15_lib_deduplication.py` | 去重公共库 | pHash、汉明距离、变换集等基础算子 |
| `B16_lib_provenance_audit.py` | 产物溯源公共库 | 代码指纹与配置哈希，标记过期产物（纯标准库） |
| `B17_lib_statistics.py` | 源图级统计公共库 | 混淆矩阵、per-class P/R/F1、指标聚合 |
| `B18_stats_stage2_analysis.py` | 源图级指标 + 三列对比 + cluster bootstrap | 以**源图**为统计单位给出置信区间 |
| `B19_run_audit_pipeline.py` | **一键执行**审计全流程 | `python B19_run_audit_pipeline.py --root ..`（可用 `--only 0,1` 只跑指定阶段） |
| `B20_test_acceptance_criteria.py` | 执行前验收测试（93 项） | 只依赖 NumPy + Pillow，跑实验前先自检 |
| `B21_log_sealed_test_access.py` | 封存测试访问日志 | 记录测试集访问，证明其未被用于调参或模型选择 |

### C 组 · 补充实验（GPU 包，对应提分项）

| 文件 | 用途 | 核心能力 |
|---|---|---|
| `C22_lib_gpu_runtime.py` | 共享运行时库 | 数据根目录解析（`--data-root`，默认 `/home/featurize/work/Combined_Dataset`）、输出目录、基线定位 |
| `C23_lib_provenance_gpu.py` | 产物溯源库 | 代码指纹 / 配置哈希，标记过期产物 |
| `C24_train_final_model_gpu.py` | 最终训练（GPU 版） | 同 A01，云端路径与断点续训增强版 |
| `C25_smoke_test_gpu.py` | 冒烟测试（GPU 版） | 上机前先跑通 |
| `C26_probe_architecture_gpu.py` | 架构探针（GPU 版） | 参数量 / FLOPs / 吞吐实测 |
| `C27_make_control_manifest.py` | 生成对照五折清单 | 生成「忽略 group_id」的图像级随机划分清单，**事前冻结** |
| `C28_build_cv_matrix.py` | 对照五折 + 跨种子矩阵 | 量化分组协议压掉的乐观偏差 |
| `C29_run_g0_preflight.py` | G0 一键编排 | 运行前预检查与编排 |
| `C30_run_g12_baseline_training.py` | G1 消融 + G2 多种子一键启动 | 含运行前预检查 |
| `C31_run_g1_ablation.py` | G1 模块消融矩阵 | `no_dual_path` / `no_ilf` 等结构变体对照 |
| `C32_run_g2_multi_seed.py` | G2 多随机种子重复 | 检验结果对随机种子的敏感性 |
| `C33_run_g3_efficiency.py` | G3 效率实测 | 延迟 / 吞吐 / 显存 / FLOPs（GPU 三项补齐） |
| `C34_run_g4_full_cv.py` | G4 一站式完整档 | 对照五折 + 跨种子 + 完整消融（≈105 h） |
| `C35_run_g5_robustness.py` | G5 最省补完档 | 只补最便宜的两块（≈15.75 h） |
| `C36_check_training_status.py` | 运行态体检 | 查看进程、子进程与进度 |
| `C37_verify_acceptance.py` | G0–G3 产物验收器 | `python C37_verify_acceptance.py <输出根目录>`，退出码 0 为通过 |
| `C38_verify_g5_outputs.py` | G5 产物验收器 | 只读、数个数、给退出码 |

### D 组 · 文档

| 文件 | 内容 |
|---|---|
| `D39_RUNBOOK_GPU_Package.md` | GPU 包运行手册（三段式实验步骤、后台运行命令、排错） |
| `D40_Acceptance_Report_G0_G3.md` | G0–G3 验收报告 |
| `D41_README_Baseline_Folder.txt` | 基线目录说明 |

---

## 三、跑之前要配置什么

| 配置项 | 在哪里改 | 说明 |
|---|---|---|
| 数据根目录 | `A02` / `A03` 顶部常量 `DATA_ROOT`；C 组用命令行 `--data-root` | 默认 `/home/featurize/data/dmsa_pcv2/Combined Dataset`（**路径含空格，注意引号**）；C 组默认 `/home/featurize/work/Combined_Dataset` |
| 输出目录 | `A03` 的 `OUTPUT_DIR`；`A02` 的 `FINAL_RUN_DIR` | 分别指向「各折产物目录」与「最终训练产物目录」 |
| 设备 | `A02` / `A03` 的 `DEVICE` | 默认 `cuda:0` |
| 数据划分清单 | `_frozen_manifests_keep_original_name/pcv_v2_folds_seed42.csv` | 五折用；**文件名不可改**（代码硬引用 + sha256 完整性锚点） |
| 封存测试清单 | `_frozen_manifests_keep_original_name/pcv_test_manifest_SEALED.csv` | 封存测试用，同上 |
| 超参 | 脚本内已固定 | batch 16 / lr 1e-3 / 学习率衰减 γ=0.96 / seed 42 / epochs 200（五折）或 126（最终）；`A03` 的 `STRICT_BASELINE_CONFIG=True` 会阻止误改 |

**数据集**：使用公开的 Alzheimer's MRI Dataset，目录结构为 `Combined Dataset/{train,test}/<四分类目录>`。原始影像为 128×128 灰度，训练时按 224×224 RGB 读取（Resize → ToTensor → Normalize(0.5)）。数据集本体不在本包内（体量大），划分清单已随包提供，可据此校验划分是否一致。

---

## 四、推荐运行顺序

### 复现主实验（最小路径）

```bash
# 0. 安装依赖（见上文）
# 1. 准备数据：按 DATA_ROOT 放好 Combined Dataset
# 2. 检查数据（只读检查即可）
python A05_check_dataset_preparation.py
# 3. 冒烟测试：验证环境与接线
python A04_smoke_test_pipeline.py
# 4. 五折交叉验证（可只跑单折：改 A03 的 FOLD_TO_RUN）
python A03_train_five_fold_cv.py
# 5. 最终全量训练（126 epochs，产出最终权重）
python A01_train_final_model.py
# 6. 封存测试（只跑一次，得到最终指标）
python A02_evaluate_sealed_test.py
```

### 补充实验（需 GPU，小时～百小时级）

```bash
python C29_run_g0_preflight.py            # G0 预检查
python C33_run_g3_efficiency.py           # 效率三项
python C30_run_g12_baseline_training.py   # G1 消融 + G2 多种子
python C37_verify_acceptance.py <输出根目录>   # 产物验收（退出码 0 为通过）
# 需要完整档：python C34_run_g4_full_cv.py（≈105 h）
# 预算有限：python C35_run_g5_robustness.py（≈15.75 h）→ python C38_verify_g5_outputs.py
```

长时间任务建议后台运行（关机 / 断网 / SSH 断开均不影响），命令见 `D39_RUNBOOK_GPU_Package.md`：

```bash
nohup python -u C35_run_g5_robustness.py --quiet > g5_console.log 2>&1 &
```

### 数据与结论审计（纯 CPU）

```bash
python B20_test_acceptance_criteria.py    # 执行前 93 项自检
python B19_run_audit_pipeline.py --root .. # 一键跑完审计全流程
```

---

## 五、跑完怎么验（验收口径）

| 验收项 | 期望值 | 出处 |
|---|---|---|
| 架构指纹 | `8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975` | `A01` 内 `EXPECTED_ARCHITECTURE_FINGERPRINT` |
| 封存测试 | accuracy **0.9898** / F1-macro **0.9929** / Kappa **0.9832** | 实验数据包 `01_Sealed_Test_Evaluation/test_metrics.json` |
| 权重 sha256 | 与 `checkpoint.sha256.txt`、`model_manifest.json` 三处一致 | 实验数据包 `08_Model_Checkpoints_and_Artifacts/` |
| 产物验收 | `C37` / `C38` 退出码 0 | 见 `D40_Acceptance_Report_G0_G3.md` |

---

## 六、常见问题

- **没有 GPU**：`A04` 冒烟与全部 B 组审计可在 CPU 上跑；训练与效率实测需要 GPU。
- **路径含空格**：`DATA_ROOT` 默认值含空格，命令行传参时加引号。
- **想快速确认能跑通**：先 `python A04_smoke_test_pipeline.py`，再决定是否投入完整训练。
- **数据集不在包内**：属公开数据集，请自行获取；包内 `_frozen_manifests_keep_original_name/` 提供了带 sha256 的划分清单，可用来校验你拿到的数据划分是否与本实验一致。

---

## 一、本目录结构（完整）

```
AIC-2026-95340465-算法模型创新-DMSA-Net：阿尔茨海默病MRI病程分层辨识网络-01（算法代码）/
├── _frozen_manifests_keep_original_name/
│   ├── pcv_test_manifest_SEALED.csv  (153.2 KB)
│   ├── pcv_v2_folds_imgsplit_seed42_FROZEN.csv  (1.8 MB)
│   ├── pcv_v2_folds_imgsplit_seed42_FROZEN.csv.sha256  (0.1 KB)
│   ├── pcv_v2_folds_imgsplit_seed42_FROZEN.report.json  (1.6 KB)
│   └── pcv_v2_folds_seed42.csv  (1.8 MB)
├── A01_train_final_model.py  (53.5 KB)
├── A02_evaluate_sealed_test.py  (17.3 KB)
├── A03_train_five_fold_cv.py  (5.8 KB)
├── A04_smoke_test_pipeline.py  (52.3 KB)
├── A05_check_dataset_preparation.py  (12.5 KB)
├── B06_audit_dataset_inventory.py  (8.6 KB)
├── B07_audit_dataset_verification.py  (24.6 KB)
├── B08_audit_duplicate_detector_power.py  (11.5 KB)
├── B09_audit_source_group_mapping.py  (20.9 KB)
├── B10_audit_detector_calibration.py  (10.6 KB)
├── B11_audit_cross_dataset_leakage.py  (6.9 KB)
├── B12_audit_pretrained_weight_probe.py  (23.0 KB)
├── B13_probe_architecture_fingerprint.py  (31.1 KB)
├── B14_audit_intra_group_leakage.py  (15.0 KB)
├── B15_lib_deduplication.py  (22.7 KB)
├── B16_lib_provenance_audit.py  (8.2 KB)
├── B17_lib_statistics.py  (23.7 KB)
├── B18_stats_stage2_analysis.py  (18.9 KB)
├── B19_run_audit_pipeline.py  (6.1 KB)
├── B20_test_acceptance_criteria.py  (49.0 KB)
├── B21_log_sealed_test_access.py  (3.5 KB)
├── B22_dedup_config.json  (1.9 KB)
├── C22_lib_gpu_runtime.py  (32.0 KB)
├── C23_lib_provenance_gpu.py  (8.2 KB)
├── C24_train_final_model_gpu.py  (53.5 KB)
├── C25_smoke_test_gpu.py  (52.3 KB)
├── C26_probe_architecture_gpu.py  (31.1 KB)
├── C27_make_control_manifest.py  (14.2 KB)
├── C28_build_cv_matrix.py  (33.1 KB)
├── C29_run_g0_preflight.py  (13.9 KB)
├── C30_run_g12_baseline_training.py  (32.4 KB)
├── C31_run_g1_ablation.py  (39.5 KB)
├── C32_run_g2_multi_seed.py  (25.8 KB)
├── C33_run_g3_efficiency.py  (22.1 KB)
├── C34_run_g4_full_cv.py  (57.0 KB)
├── C35_run_g5_robustness.py  (38.8 KB)
├── C36_check_training_status.py  (12.7 KB)
├── C37_verify_acceptance.py  (4.8 KB)
├── C38_verify_g5_outputs.py  (8.5 KB)
├── D39_RUNBOOK_GPU_Package.md  (89.1 KB)
├── D40_Acceptance_Report_G0_G3.md  (26.1 KB)
├── D41_README_Baseline_Folder.txt  (0.7 KB)
├── D42_g1_eta_helper.sh  (3.2 KB)
└── README.md  (0.0 KB)
```

## 附：文件清单

| 文件 / 子目录 | 大小 | 作用或来源 |
|---|---:|---|
| `A01_train_final_model.py` | 53.5 KB | 原始数据文件 |
| `A02_evaluate_sealed_test.py` | 17.3 KB | 原始数据文件 |
| `A03_train_five_fold_cv.py` | 5.8 KB | 原始数据文件 |
| `A04_smoke_test_pipeline.py` | 52.3 KB | 原始数据文件 |
| `A05_check_dataset_preparation.py` | 12.5 KB | 原始数据文件 |
| `B06_audit_dataset_inventory.py` | 8.6 KB | 原始数据文件 |
| `B07_audit_dataset_verification.py` | 24.6 KB | 原始数据文件 |
| `B08_audit_duplicate_detector_power.py` | 11.5 KB | 原始数据文件 |
| `B09_audit_source_group_mapping.py` | 20.9 KB | 原始数据文件 |
| `B10_audit_detector_calibration.py` | 10.6 KB | 原始数据文件 |
| `B11_audit_cross_dataset_leakage.py` | 6.9 KB | 原始数据文件 |
| `B12_audit_pretrained_weight_probe.py` | 23.0 KB | 原始数据文件 |
| `B13_probe_architecture_fingerprint.py` | 31.1 KB | 原始数据文件 |
| `B14_audit_intra_group_leakage.py` | 15.0 KB | 原始数据文件 |
| `B15_lib_deduplication.py` | 22.7 KB | 原始数据文件 |
| `B16_lib_provenance_audit.py` | 8.2 KB | 原始数据文件 |
| `B17_lib_statistics.py` | 23.7 KB | 原始数据文件 |
| `B18_stats_stage2_analysis.py` | 18.9 KB | 原始数据文件 |
| `B19_run_audit_pipeline.py` | 6.1 KB | 原始数据文件 |
| `B20_test_acceptance_criteria.py` | 49.0 KB | 原始数据文件 |
| `B21_log_sealed_test_access.py` | 3.5 KB | 原始数据文件 |
| `B22_dedup_config.json` | 1.9 KB | 原始数据文件 |
| `C22_lib_gpu_runtime.py` | 32.0 KB | 原始数据文件 |
| `C23_lib_provenance_gpu.py` | 8.2 KB | 原始数据文件 |
| `C24_train_final_model_gpu.py` | 53.5 KB | 原始数据文件 |
| `C25_smoke_test_gpu.py` | 52.3 KB | 原始数据文件 |
| `C26_probe_architecture_gpu.py` | 31.1 KB | 原始数据文件 |
| `C27_make_control_manifest.py` | 14.2 KB | 原始数据文件 |
| `C28_build_cv_matrix.py` | 33.1 KB | 原始数据文件 |
| `C29_run_g0_preflight.py` | 13.9 KB | 原始数据文件 |
| `C30_run_g12_baseline_training.py` | 32.4 KB | 原始数据文件 |
| `C31_run_g1_ablation.py` | 39.5 KB | 原始数据文件 |
| `C32_run_g2_multi_seed.py` | 25.8 KB | 原始数据文件 |
| `C33_run_g3_efficiency.py` | 22.1 KB | 原始数据文件 |
| `C34_run_g4_full_cv.py` | 57.0 KB | 原始数据文件 |
| `C35_run_g5_robustness.py` | 38.8 KB | 原始数据文件 |
| `C36_check_training_status.py` | 12.7 KB | 原始数据文件 |
| `C37_verify_acceptance.py` | 4.8 KB | 原始数据文件 |
| `C38_verify_g5_outputs.py` | 8.5 KB | 原始数据文件 |
| `D39_RUNBOOK_GPU_Package.md` | 89.1 KB | 原始数据文件 |
| `D40_Acceptance_Report_G0_G3.md` | 26.1 KB | 原始数据文件 |
| `D41_README_Baseline_Folder.txt` | 0.7 KB | 原始数据文件 |
| `D42_g1_eta_helper.sh` | 3.2 KB | 原始数据文件 |
| `_frozen_manifests_keep_original_name/` | 5 个文件 | 见该目录内 README |
