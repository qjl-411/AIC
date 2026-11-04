# 08_Model_Checkpoints_and_Artifacts

最终模型权重（217.5 MB）及配套清单：架构指纹、训练协议、逐轮历史、封存指标、生产验证。

## 内容

| 文件 / 子目录 | 大小 | 作用或来源 |
|---|---:|---|
| `.gitignore` | 0.2 KB | 原始数据文件 |
| `checkpoint.sha256.txt` | 0.1 KB | 权重 sha256 校验值 |
| `class_to_idx.json` | 0.1 KB | 类别与索引映射 |
| `cv_aligned_epoch_mean.csv` | 18.0 KB | 五折对齐轮次均值分析 |
| `cv_aligned_epoch_mean_detailed.csv` | 24.2 KB | 五折对齐轮次均值分析（明细） |
| `cv_fold_best.json` | 1.6 KB | 五折各自最优结果 |
| `final_model_epoch126.pth` | 217.5 MB | 最终模型权重（epoch 126，56,812,813 参数，即封存测试所用） |
| `final_train_history.csv` | 7.8 KB | 最终训练逐轮历史 |
| `final_training_protocol.json` | 0.6 KB | 最终训练协议（轮次来源、划分依据） |
| `model_info.json` | 0.2 KB | 模型结构信息 |
| `model_manifest.json` | 2.0 KB | 模型清单（权重 sha256、参数量、架构指纹、封存指标） |
| `model_performance.json` | 5.2 KB | 模型性能记录 |
| `production_verification.json` | 4.1 KB | 平台生产环境验证记录 |
| `sealed_test_confusion_matrix.csv` | 0.2 KB | 封存测试混淆矩阵 |
| `sealed_test_metrics.json` | 1.6 KB | 封存测试指标 |