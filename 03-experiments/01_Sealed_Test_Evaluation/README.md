# 01_Sealed_Test_Evaluation

封存测试（1,279 张）评测结果：总体指标、混淆矩阵、逐样本预测、所用代码与清单、完整性校验。98.98% 的唯一来源。

## 内容

| 文件 / 子目录 | 大小 | 作用或来源 |
|---|---:|---|
| `pre_evaluation_integrity.json` | 0.8 KB | 评测前完整性校验 |
| `sealed_test_code_used.py` | 17.3 KB | 封存测试实际执行的代码快照 |
| `sealed_test_manifest_used.csv` | 153.2 KB | 封存测试所用清单（1,279 张） |
| `test_confusion_matrix.csv` | 0.2 KB | 封存测试混淆矩阵（数值） |
| `test_confusion_matrix.png` | 78.9 KB | 封存测试混淆矩阵（图） |
| `test_environment.json` | 0.4 KB | 封存测试运行环境记录 |
| `test_metrics.json` | 1.6 KB | 封存测试总体指标（accuracy / f1 / kappa） |
| `test_predictions.csv` | 182.4 KB | 封存测试 1,279 张逐样本预测（98.98% 的底层证据） |