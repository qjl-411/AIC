# 阶段 0.0b 数据集到货验证报告

- 数据根目录：`E:\Code\NeuroEvo\DMSA_netV2\Alzheimer's MRI Dataset\Alzheimer's MRI Dataset\Combined Dataset`
- 判定：**ARRIVED_AND_RECONCILED**
- 数据目录只读，本脚本未做任何写操作

## 1 结构与逐类计数

| split | Mild Impairment | Moderate Impairment | No Impairment | Very Mild Impairment | 合计 |
|---|---|---|---|---|---|
| train | 2560 | 2560 | 2560 | 2560 | 10240 |
| test | 179 | 12 | 640 | 448 | 1279 |

## 2 帧几何（与声明对账）

- 实测尺寸：`{'128x128': 11519}`
- 实测模式：`{'L': 11519}`
- 论文/代码声明：`(224, 224) × 3ch (RGB)`
- **一致：否**

## 3 SHA-256 逐文件对账

| split | manifest | 行数 | 磁盘文件 | OK | 缺失 | SHA 不符 | 磁盘多余 | 判定 |
|---|---|---|---|---|---|---|---|---|
| train | `pcv_v2_folds_seed42.csv` | 10240 | 10240 | 10240 | 0 | 0 | 0 | **EXACT_MATCH** |
| test | `pcv_test_manifest_SEALED.csv` | 1279 | 1279 | 1279 | 0 | 0 | 0 | **EXACT_MATCH** |

## 4 SHA-only 唯一数（**故意演示『用错工具』会得到的数字**）

| split | 类别 | 文件数 | 唯一 SHA | 组内冗余 | 多成员组 |
|---|---|---|---|---|---|
| train | Mild Impairment | 2560 | 2435 | 125 | 125 |
| train | Moderate Impairment | 2560 | 2560 | 0 | 0 |
| train | No Impairment | 2560 | 2560 | 0 | 0 |
| train | Very Mild Impairment | 2560 | 2503 | 57 | 57 |
| test | Mild Impairment | 179 | 179 | 0 | 0 |
| test | Moderate Impairment | 12 | 12 | 0 | 0 |
| test | No Impairment | 640 | 640 | 0 | 0 |
| test | Very Mild Impairment | 448 | 448 | 0 | 0 |

## 5 train ↔ test 跨集 exact 交集

| test 类别 | test 张数 | 在 train 中有完全相同副本 |
|---|---|---|
| Mild Impairment | 179 | 0 |
| Moderate Impairment | 12 | 0 |
| No Impairment | 640 | 0 |
| Very Mild Impairment | 448 | 0 |
| **合计** | **1279** | **0** |

## 6 manifest 自述分组 vs 磁盘 SHA 实测（交叉验证）

- `group_reason` 分布：`{'singleton': 9328, 'exact_duplicate': 364, 'near_duplicate_component': 548}`
- 组件数 **9659**，最大组件 **64**
- 规模分布：`{'1': 9328, '2': 277, '3': 25, '4': 10, '5': 6, '6': 5, '8': 3, '9': 1, '12': 1, '35': 1, '39': 1, '64': 1}`

| 类别 | manifest 自述 exact 组内冗余 | 磁盘 SHA 实测组内冗余 | 一致 |
|---|---|---|---|
| Mild Impairment | 125 | 125 | ✓ |
| Moderate Impairment | 0 | 0 | ✓ |
| No Impairment | 0 | 0 | ✓ |
| Very Mild Impairment | 57 | 57 | ✓ |

**manifest 的 exact 分组可被磁盘 SHA 独立复现**

## 7 复用基线自带校验器（`data_prepare_check.py`）

- 模块 SHA 已在 `MANIFEST.sha256` 登记；`VERIFY_ALL_HASHES=True`
- 它期望的原始 ZIP：`/home/featurize/work/DMSA_PCV2_20260824/input/Alzheimer_MRI.zip`（**用户已确认拿不到**）
- 跑完全部校验：**是**

```
[PASS] Fold 1: train=8192, val=2048, train/class=2048, val/class=512, group_overlap=0
[PASS] Fold 2: train=8192, val=2048, train/class=2048, val/class=512, group_overlap=0
[PASS] Fold 3: train=8192, val=2048, train/class=2048, val/class=512, group_overlap=0
[PASS] Fold 4: train=8192, val=2048, train/class=2048, val/class=512, group_overlap=0
[PASS] Fold 5: train=8192, val=2048, train/class=2048, val/class=512, group_overlap=0
[PASS] train/test exact SHA-256 overlap = 0
```

> ⚠️ **只调用了它的校验函数，没有调用 `main()`** —— `main()` 会 `extract_dataset(zip, combined_root)`，往数据目录里解压。
> 我们既没有那个 ZIP，也不允许任何脚本改动已到货的数据目录。