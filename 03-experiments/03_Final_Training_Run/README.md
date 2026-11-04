# 03_Final_Training_Run

最终全量训练（epoch 126）过程记录：训练协议、逐轮历史、全量样本清单、所用代码快照。

## 内容

| 文件 / 子目录 | 大小 | 作用或来源 |
|---|---:|---|
| `FINAL_MODEL_READY.json` | 0.3 KB | 最终模型就绪标记 |
| `class_to_idx.json` | 0.1 KB | 类别与索引映射 |
| `environment.json` | 0.4 KB | 运行环境记录（Python / torch / 显卡） |
| `final_train_code_used.py` | 53.5 KB | 最终训练实际执行的代码快照 |
| `final_training_protocol.json` | 0.6 KB | 最终训练协议（轮次来源、划分依据） |
| `full_train_snapshot.csv` | 1.5 MB | 最终训练所用全量样本清单 |
| `model_info.json` | 0.2 KB | 模型结构信息 |
| `pcv_v2_folds_used.csv` | 1.8 MB | 原始数据文件 |
| `train_history.csv` | 8.5 KB | 最终训练逐轮 loss / accuracy |