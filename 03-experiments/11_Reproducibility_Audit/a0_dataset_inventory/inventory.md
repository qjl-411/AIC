# 阶段 0.0 输入盘点

- 根目录：`E:\Code\NeuroEvo\DMSA_netV2`
- 被阻塞的零成本任务数：**3**

| 任务 | 状态 | 缺失输入 | 后果 |
|---|---|---|---|
| 阶段 0.0c 数据集到货验证（v2.7 新增） | READY | - | - |
| 阶段 0.1 A1 源图映射 | READY | - | - |
| 阶段 0.2 A2 train<->test 审计 | READY | - | - |
| 阶段 0.3 基线冻结 | READY | - | - |
| 阶段 0.4 五折产物盘点 | BLOCKED | 五折 metrics.csv, best_model_state.pth x5, final_model_state.pth | 阶段 1.4/1.5 的预测侧分析只能改为重训；§5.5 固定 epoch 对照只能读折内日志 |
| 阶段 0.5 协议溯源（v1_val_fold / 论文 0.9990） | BLOCKED | 旧 split 文件, 旧 checkpoint, 旧训练日志 | 只能写'基于当前材料暂不可复现'，不得升级为'不可复现' |
| 阶段 3.4 / 5.4 A6-s 静态权重映射 | READY | - | - |
| 阶段 1.9 subject-level 可得性判定 | BLOCKED | 发布者 patient/subject ID 或元数据 | 只能报告原图级独立，并显式写 limitation |