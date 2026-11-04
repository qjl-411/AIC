# 03 实验数据与模型产物 · 说明

本目录是本项目**全部实验结果与产物**，按**数据类型**分类存放：

| 目录 | 内容 |
|---|---|
| `01_Sealed_Test_Evaluation` | 封存测试（1,279 张）结果：总体指标、混淆矩阵、逐样本预测、所用代码与清单、完整性校验 |
| `02_Five_Fold_Cross_Validation` | P-CV v2 五折交叉验证：每折逐轮指标、汇总、划分快照、最优混淆矩阵图、环境记录 |
| `03_Final_Training_Run` | 最终全量训练（epoch 126）：训练协议、逐轮历史、全量样本清单、所用代码快照 |
| `04_Ablation_Studies` | 模块消融（g1）与鲁棒性（g5）：逐折结果、汇总表、归档包 |
| `05_Multi_Seed_Validation` | 多随机种子验证（g2：seed 43 / 44） |
| `06_Efficiency_Benchmarks` | 效率实测：DMSA-Net 与基线 ResNeXt50 的延迟 / 吞吐 / 显存对比，含归因 |
| `07_ROC_Calibration_Confidence` | ROC 与 AUC、校准指标（ECE / Brier / NLL）、预测置信度分布，含曲线底层数据 |
| `08_Model_Checkpoints_and_Artifacts` | 最终模型权重（217.5 MB）及配套清单、训练协议、封存指标、生产验证 |
| `09_Figures_and_Plots` | 全部图表（方案配图 + 补充实验图表）及其生成脚本 |
| `10_Raw_Console_Logs` | 训练原始控制台日志 |
| `11_Reproducibility_Audit` | 复现审计产物：数据清单、检测器标定、同源映射、泄漏审计、预训练核查、架构指纹 |

**每个分类目录内都有 `README.md`**，列出该目录所有文件的作用与来源；根目录的
**`数据索引与验证指南.md`** 则把「文档里的每一个数字」直接指到具体文件，**不跑代码也能核对**。

全部文件均登记在 `SHA256SUMS.txt` 中，可验证内容未被改动。

---

## 这些数据是怎么得来的（产出链路）

下表把每一类数据追溯到**由哪个脚本、在什么环境、按什么协议**产出——便于判断数据的可信边界。

| 数据类别 | 由什么产出 | 运行环境 | 关键协议 / 说明 |
|---|---|---|---|
| `02_Five_Fold_Cross_Validation` | `A03_train_five_fold_cv.py` | RTX 3060 / torch 2.2.2 | P-CV v2 分组五折，每折 8,192 训练 / 2,048 验证（每类 512），200 epochs，seed 42 |
| `03_Final_Training_Run`（训练历史、协议） | `A01_train_final_model.py` | RTX 4090 | 完整 10,240 张固定训练 **126 epochs**；轮次数由五折对齐轮次分析事前确定 |
| `01_Sealed_Test_Evaluation`（指标、混淆矩阵、逐样本预测） | `A02_evaluate_sealed_test.py` | RTX 4090 | 对封存测试集 **1,279 张只评测一次**；评测前做完整性校验，测试集不参与任何调参与模型选择 |
| `08_Model_Checkpoints_and_Artifacts`（权重） | `A01_train_final_model.py` 产出 | RTX 4090 | epoch 126；sha256 与 `checkpoint.sha256.txt`、`model_manifest.json` 三处一致，可自行校验 |
| `06_Efficiency_Benchmarks`（延迟 / 吞吐 / 显存） | `C33_run_g3_efficiency.py` 与 b2 脚本 | RTX 4090（GPU 实测） | 与基线 ResNeXt50 同口径对照；跨脚本不可直接比较的原因见 `b2_efficiency_attribution.md` |
| `07_ROC_Calibration_Confidence`（ROC / AUC / 校准 / 置信度） | `b1_roc_calibration.py` 计算、`b1_make_figures.py` 作图 | 本机 | 曲线底层数据随包提供（`b1_roc_curves.json`），可自行重画 |
| `04_Ablation_Studies`（g1 消融 / g5 鲁棒性） | `C31_run_g1_ablation.py`、`C35_run_g5_robustness.py` | 云端 GPU | 结构变体（如 `no_dual_path`）与基线对照；参照未完全对齐时结论仅作参考，已在表中标注 |
| `05_Multi_Seed_Validation`（多种子） | `C32_run_g2_multi_seed.py` | 云端 GPU | seed 43 / 44 重复实验，用于观察结果对随机种子的敏感性 |
| `11_Reproducibility_Audit`（参数量 / FLOPs / 泄漏 / 同源） | `B13_probe_architecture_fingerprint.py`、`B14`、`B11` 等 | 本机（纯 CPU 可跑） | 参数量 56,812,813、FLOPs 14,984,620,662 为实测值；审计类结论均落盘 json/md |
| `09_Figures_and_Plots`（图表） | `b1_make_figures.py` 及方案配图脚本 | 本机 | 生成脚本随包提供，可复现作图 |
| `10_Raw_Console_Logs` | 训练过程控制台输出 | 云端 | 留痕用，可追溯每一轮训练 |

> **边界说明（诚实披露）**：消融与多种子实验受算力约束只跑了部分折 / 部分种子，其结论强度弱于主指标；
> 主指标（封存测试 98.98%、五折 99.14%）为完整实验结果。详情见 `04_`、`05_` 目录内的 README。

## 本目录结构（完整）

```
AIC-2026-95340465-算法模型创新-DMSA-Net：阿尔茨海默病MRI病程分层辨识网络-03（实验数据与模型产物）/
├── 01_Sealed_Test_Evaluation/
│   ├── README.md  (0.9 KB)
│   ├── pre_evaluation_integrity.json  (0.8 KB)
│   ├── sealed_test_code_used.py  (17.3 KB)
│   ├── sealed_test_manifest_used.csv  (153.2 KB)
│   ├── test_confusion_matrix.csv  (0.2 KB)
│   ├── test_confusion_matrix.png  (78.9 KB)
│   ├── test_environment.json  (0.4 KB)
│   ├── test_metrics.json  (1.6 KB)
│   └── test_predictions.csv  (182.4 KB)
├── 02_Five_Fold_Cross_Validation/
│   ├── Fold1/
│   │   ├── best_confusion_matrix.png  (78.2 KB)
│   │   ├── best_metrics.json  (0.2 KB)
│   │   ├── cv_summary.csv  (0.2 KB)
│   │   ├── cv_summary.json  (0.9 KB)
│   │   ├── environment.json  (0.4 KB)
│   │   ├── metrics.csv  (36.5 KB)
│   │   ├── model_info.json  (0.2 KB)
│   │   ├── run_config.json  (0.7 KB)
│   │   └── split_snapshot.csv  (1.5 MB)
│   ├── Fold2/
│   │   ├── best_confusion_matrix.png  (77.5 KB)
│   │   ├── best_metrics.json  (0.2 KB)
│   │   ├── cv_summary.csv  (0.2 KB)
│   │   ├── cv_summary.json  (0.9 KB)
│   │   ├── environment.json  (0.4 KB)
│   │   ├── metrics.csv  (36.4 KB)
│   │   ├── model_info.json  (0.2 KB)
│   │   ├── run_config.json  (0.7 KB)
│   │   └── split_snapshot.csv  (1.5 MB)
│   ├── Fold3/
│   │   ├── best_confusion_matrix.png  (78.1 KB)
│   │   ├── best_metrics.json  (0.2 KB)
│   │   ├── cv_summary.csv  (0.2 KB)
│   │   ├── cv_summary.json  (0.9 KB)
│   │   ├── environment.json  (0.4 KB)
│   │   ├── metrics.csv  (36.6 KB)
│   │   ├── model_info.json  (0.2 KB)
│   │   ├── run_config.json  (0.7 KB)
│   │   └── split_snapshot.csv  (1.5 MB)
│   ├── Fold4/
│   │   ├── best_confusion_matrix.png  (78.4 KB)
│   │   ├── best_metrics.json  (0.2 KB)
│   │   ├── cv_summary.csv  (0.2 KB)
│   │   ├── cv_summary.json  (0.9 KB)
│   │   ├── environment.json  (0.4 KB)
│   │   ├── metrics.csv  (36.5 KB)
│   │   ├── model_info.json  (0.2 KB)
│   │   ├── run_config.json  (0.7 KB)
│   │   └── split_snapshot.csv  (1.5 MB)
│   ├── Fold5/
│   │   ├── best_confusion_matrix.png  (78.9 KB)
│   │   ├── best_metrics.json  (0.2 KB)
│   │   ├── cv_summary.csv  (0.2 KB)
│   │   ├── cv_summary.json  (0.9 KB)
│   │   ├── environment.json  (0.4 KB)
│   │   ├── metrics.csv  (36.6 KB)
│   │   ├── model_info.json  (0.2 KB)
│   │   ├── run_config.json  (0.7 KB)
│   │   └── split_snapshot.csv  (1.5 MB)
│   └── README.md  (0.5 KB)
├── 03_Final_Training_Run/
│   ├── FINAL_MODEL_READY.json  (0.3 KB)
│   ├── README.md  (0.8 KB)
│   ├── class_to_idx.json  (0.1 KB)
│   ├── environment.json  (0.4 KB)
│   ├── final_train_code_used.py  (53.5 KB)
│   ├── final_training_protocol.json  (0.6 KB)
│   ├── full_train_snapshot.csv  (1.5 MB)
│   ├── model_info.json  (0.2 KB)
│   ├── pcv_v2_folds_used.csv  (1.8 MB)
│   └── train_history.csv  (8.5 KB)
├── 04_Ablation_Studies/
│   ├── g1/
│   │   ├── no_ilf/
│   │   │   ├── fold_1/
│   │   │   │   ├── best_metrics.json  (1.1 KB)
│   │   │   │   └── metrics.csv  (13.2 KB)
│   │   │   └── fold_5/
│   │   │       ├── best_metrics.json  (1.1 KB)
│   │   │       └── metrics.csv  (13.3 KB)
│   │   ├── no_mdc/
│   │   │   ├── fold_1/
│   │   │   │   ├── best_metrics.json  (1.1 KB)
│   │   │   │   └── metrics.csv  (24.4 KB)
│   │   │   └── fold_5/
│   │   │       ├── best_metrics.json  (1.1 KB)
│   │   │       └── metrics.csv  (24.5 KB)
│   │   ├── no_sam/
│   │   │   ├── fold_1/
│   │   │   └── fold_5/
│   │   │       ├── best_metrics.json  (1.1 KB)
│   │   │       └── metrics.csv  (24.6 KB)
│   │   ├── g1_ablation_per_fold.csv  (0.9 KB)
│   │   ├── g1_ablation_summary.json  (2.4 KB)
│   │   ├── g1_ablation_table.csv  (0.3 KB)
│   │   ├── g1_ablation_table.md  (1.1 KB)
│   │   ├── g1_dryrun.json  (1.6 KB)
│   │   ├── g1_progress.log  (1.4 MB)
│   │   ├── g1_run.log  (78.4 MB)
│   │   └── g1_run_config.json  (1.2 KB)
│   ├── g5_robustness/
│   │   ├── abl/
│   │   │   └── runs/
│   │   │       ├── grouped__no_dual_path__seed42__fold1/
│   │   │       │   ├── best_metrics.json  (1.3 KB)
│   │   │       │   └── metrics.csv  (19.5 KB)
│   │   │       └── grouped__no_dual_path__seed42__fold5/
│   │   │           ├── best_metrics.json  (1.2 KB)
│   │   │           └── metrics.csv  (24.4 KB)
│   │   ├── seed/
│   │   │   └── runs/
│   │   │       ├── grouped__full__seed43__fold1/
│   │   │       │   ├── best_metrics.json  (1.2 KB)
│   │   │       │   └── metrics.csv  (24.7 KB)
│   │   │       ├── grouped__full__seed43__fold5/
│   │   │       │   ├── best_metrics.json  (1.2 KB)
│   │   │       │   └── metrics.csv  (24.5 KB)
│   │   │       ├── grouped__full__seed44__fold1/
│   │   │       │   ├── best_metrics.json  (1.2 KB)
│   │   │       │   └── metrics.csv  (24.4 KB)
│   │   │       └── grouped__full__seed44__fold5/
│   │   │           ├── best_metrics.json  (1.2 KB)
│   │   │           └── metrics.csv  (24.3 KB)
│   │   ├── g5_ablation_merged.json  (2.2 KB)
│   │   ├── g5_per_run.csv  (1.7 KB)
│   │   ├── g5_precheck.json  (2.6 KB)
│   │   ├── g5_progress.log  (1.9 MB)
│   │   ├── g5_report.md  (3.6 KB)
│   │   └── g5_summary.json  (8.3 KB)
│   ├── AIC_g1_0925_archive.tar.gz  (14.8 MB)
│   ├── README.md  (0.5 KB)
│   ├── add_delta_vs_full.py  (9.9 KB)
│   └── g1_ablation_table_with_delta.md  (3.0 KB)
├── 05_Multi_Seed_Validation/
│   ├── _variants/
│   │   ├── final_train_seed4.py  (53.5 KB)
│   │   ├── final_train_seed43.py  (53.5 KB)
│   │   ├── final_train_seed44.py  (53.5 KB)
│   │   └── pcv_v2_folds_seed42.csv  (1.8 MB)
│   ├── dmsa_final_train_seed43_epoch126/
│   │   ├── FINAL_MODEL_READY.json  (0.3 KB)
│   │   ├── class_to_idx.json  (0.1 KB)
│   │   ├── environment.json  (0.4 KB)
│   │   ├── final_train_code_used.py  (53.5 KB)
│   │   ├── final_training_protocol.json  (0.6 KB)
│   │   ├── full_train_snapshot.csv  (1.5 MB)
│   │   ├── model_info.json  (0.2 KB)
│   │   ├── pcv_v2_folds_used.csv  (1.8 MB)
│   │   └── train_history.csv  (8.5 KB)
│   ├── dmsa_final_train_seed44_epoch126/
│   │   ├── FINAL_MODEL_READY.json  (0.3 KB)
│   │   ├── class_to_idx.json  (0.1 KB)
│   │   ├── environment.json  (0.4 KB)
│   │   ├── final_train_code_used.py  (53.5 KB)
│   │   ├── final_training_protocol.json  (0.6 KB)
│   │   ├── full_train_snapshot.csv  (1.5 MB)
│   │   ├── model_info.json  (0.2 KB)
│   │   ├── pcv_v2_folds_used.csv  (1.8 MB)
│   │   └── train_history.csv  (8.5 KB)
│   ├── README.md  (0.9 KB)
│   ├── g2_dryrun.json  (1.1 KB)
│   ├── g2_multiseed_summary.json  (1.9 KB)
│   ├── g2_multiseed_table.csv  (0.5 KB)
│   ├── g2_run.log  (372.6 KB)
│   ├── log_seed4.txt  (0.4 KB)
│   ├── log_seed43.txt  (11.0 MB)
│   ├── log_seed44.txt  (11.0 MB)
│   ├── progress_seed43.log  (211.9 KB)
│   └── progress_seed44.log  (215.8 KB)
├── 06_Efficiency_Benchmarks/
│   ├── g3_gpu/
│   │   ├── g3_efficiency.json  (3.3 KB)
│   │   ├── g3_efficiency_table.md  (1.2 KB)
│   │   ├── g3_latency_table.csv  (0.3 KB)
│   │   └── g3_progress.log  (0.6 KB)
│   ├── README.md  (0.6 KB)
│   ├── b2_efficiency_attribution.md  (2.2 KB)
│   ├── b2_efficiency_baseline.json  (0.4 KB)
│   ├── b2_efficiency_dmsa.json  (2.7 KB)
│   └── b2_efficiency_table.md  (4.7 KB)
├── 07_ROC_Calibration_Confidence/
│   ├── README.md  (0.5 KB)
│   ├── b1_auc_summary.json  (1.1 KB)
│   ├── b1_calibration.json  (2.7 KB)
│   ├── b1_confidence.json  (0.3 KB)
│   └── b1_roc_curves.json  (120.6 KB)
├── 08_Model_Checkpoints_and_Artifacts/
│   ├── .gitignore  (0.2 KB)
│   ├── README.md  (1.3 KB)
│   ├── checkpoint.sha256.txt  (0.1 KB)
│   ├── class_to_idx.json  (0.1 KB)
│   ├── cv_aligned_epoch_mean.csv  (18.0 KB)
│   ├── cv_aligned_epoch_mean_detailed.csv  (24.2 KB)
│   ├── cv_fold_best.json  (1.6 KB)
│   ├── final_model_epoch126.pth  (217.5 MB)
│   ├── final_train_history.csv  (7.8 KB)
│   ├── final_training_protocol.json  (0.6 KB)
│   ├── model_info.json  (0.2 KB)
│   ├── model_manifest.json  (2.0 KB)
│   ├── model_performance.json  (5.2 KB)
│   ├── production_verification.json  (4.1 KB)
│   ├── sealed_test_confusion_matrix.csv  (0.2 KB)
│   └── sealed_test_metrics.json  (1.6 KB)
├── 09_Figures_and_Plots/
│   ├── Boost_Figures/
│   │   ├── fig1_roc_curves.png  (161.2 KB)
│   │   ├── fig2_reliability.png  (188.0 KB)
│   │   └── fig3_confidence_gap.png  (92.2 KB)
│   ├── Plot_Scripts/
│   │   ├── b1_make_figures.py  (8.5 KB)
│   │   ├── b1_roc_calibration.py  (13.1 KB)
│   │   └── b2_efficiency_report.py  (10.1 KB)
│   ├── Scheme_Figures/
│   │   ├── 01_architecture.png  (281.6 KB)
│   │   ├── 02_bottleneck.png  (209.4 KB)
│   │   ├── 03_cv_results.png  (115.8 KB)
│   │   ├── 04_loss_curve.png  (166.2 KB)
│   │   ├── 05_confusion_matrix.png  (155.5 KB)
│   │   ├── 06_model_comparison.png  (98.8 KB)
│   │   ├── 07_spatial_attention.png  (124.2 KB)
│   │   ├── 08_platform.png  (211.0 KB)
│   │   └── 09_multi_scale.png  (94.0 KB)
│   └── README.md  (0.3 KB)
├── 10_Raw_Console_Logs/
│   ├── README.md  (0.3 KB)
│   ├── g12_precheck.json  (3.4 KB)
│   └── g12_run.log  (78.8 MB)
└── 11_Reproducibility_Audit/
    ├── a0_dataset_inventory/
    │   ├── dataset_verify.json  (7.2 KB)
    │   ├── dataset_verify.md  (3.6 KB)
    │   ├── detector_power.json  (3.9 KB)
    │   ├── detector_power.md  (2.1 KB)
    │   ├── inventory.json  (5.4 KB)
    │   └── inventory.md  (1.0 KB)
    ├── a1cal_calibration/
    │   ├── calibration.csv  (0.6 KB)
    │   ├── calibration_holdout.csv  (0.5 KB)
    │   ├── suggested_thresholds.json  (0.9 KB)
    │   └── transform_set_report.csv  (0.1 KB)
    ├── a1v2_source_mapping/
    ├── a6s_weight_probe/
    │   ├── weight_map_report.json  (36.5 KB)
    │   └── weight_map_report.md  (12.0 KB)
    ├── arch_probe/
    │   ├── arch_probe.json  (5.0 KB)
    │   └── arch_probe.md  (2.7 KB)
    ├── leakage_audit/
    │   ├── ess_deff.csv  (0.6 KB)
    │   ├── leakage_audit.json  (2.4 KB)
    │   └── leakage_audit.md  (2.2 KB)
    ├── stage2_stats/
    │   ├── stage2_report.json  (4.9 KB)
    │   ├── stage2_report.md  (1.9 KB)
    │   └── three_column_table.csv  (0.5 KB)
    ├── README.md  (0.7 KB)
    └── provenance_report.json  (3.4 KB)
```

## 分类索引

| 目录 | 内容 |
|---|---|
| `01_Sealed_Test_Evaluation/` | 封存测试（1,279 张）评测结果：总体指标、混淆矩阵、逐样本预测、所用代码与清单、完整性校验。98.98% 的唯一来源。 |
| `02_Five_Fold_Cross_Validation/` | P-CV v2 五折交叉验证：每折逐轮指标、汇总、划分快照、最优混淆矩阵图与环境记录。 |
| `03_Final_Training_Run/` | 最终全量训练（epoch 126）过程记录：训练协议、逐轮历史、全量样本清单、所用代码快照。 |
| `04_Ablation_Studies/` | 模块消融实验（g1：no_dual_path / no_ilf 等变体）与 g5 鲁棒性实验：逐折结果、汇总表、归档包。 |
| `05_Multi_Seed_Validation/` | 多随机种子验证（g2：seed 43 / 44），说明结果对随机种子的敏感性。 |
| `06_Efficiency_Benchmarks/` | 效率实测：DMSA-Net 与基线 ResNeXt50 的延迟、吞吐、显存对比（RTX 4090），含归因说明。 |
| `07_ROC_Calibration_Confidence/` | ROC 曲线与 AUC、校准指标（ECE / Brier / NLL）、预测置信度分布，含曲线底层数据。 |
| `08_Model_Checkpoints_and_Artifacts/` | 最终模型权重（217.5 MB）及配套清单：架构指纹、训练协议、逐轮历史、封存指标、生产验证。 |
| `09_Figures_and_Plots/` | 全部图表：方案配图、补充实验图表及其生成脚本。 |
| `10_Raw_Console_Logs/` | 训练原始控制台日志（g12 等），用于追溯训练过程。 |
| `11_Reproducibility_Audit/` | 复现审计产物：数据清单、检测器标定、同源映射、泄漏审计、预训练权重核查、架构指纹、阶段二统计。 |
