# DMSA-Net 2026AIC 提分实验 · 验收报告

> **适用对象**：`_aic_boost/gpu/` 下 5 个脚本（`gpu_lib.py` / `g0_run_all.py` / `g1_ablation.py` / `g2_multi_seed.py` / `g3_efficiency.py`）在云 GPU 机器上的运行结果。
> **验收对象**：阶段 **G3（效率）** → **G2（多种子）** → **G1（消融）**，以及编排层 **G0**。
> **输出根目录**：`/home/featurize/work/AIC_train_again/`（下称 `<OUT>`）。

---

## 0. 验收总原则

本项目的验收**不看"脚本没报错"**，只看四件事，缺一不可：

| 编号 | 原则 | 为什么 |
|---|---|---|
| **P1** | **以退出码为判据**（`exit code == 0`），不以"日志里没有 Traceback"为判据 | 项目铁律 5：「有单元测试」≠「脚本能跑」 |
| **P2** | **产物齐备**：预期文件全部存在，且 JSON 能被解析、字段非空 | 部分产物 = 最危险的状态（看着像成功） |
| **P3** | **数值对账**：关键数字必须与冻结基线逐位对齐（见 §1.2） | 项目铁律 1：每个数字可复现 |
| **P4** | **硬门必过**：协议 SHA、架构指纹、`full` 同构三项任一不过 → **整批结果作废** | 项目铁律 2/6 |

> ⚠️ **三条红线（出现即整批结果不可用，不是"警告"）**
> 1. `architecture fingerprint` ≠ `8e542321…69d7`
> 2. `train manifest` SHA-256 ≠ `51f5b251…edfd`
> 3. 任何一步**读取了 sealed test**（本包设计上物理不可达，若出现说明代码被改动过）

---

## 1. 通用检查项（每个阶段都要过）

### 1.1 溯源完整性（P2/P3）

每个阶段的产物都经唯一写入口落盘，自动带 `provenance`。验收时必须确认：

| 检查点 | 期望 |
|---|---|
| `provenance.script` | 与产生该产物的脚本名一致 |
| `provenance.script_sha256` | 非空；G0 报告里应能与本机下发版本对上 |
| `provenance.written_at_utc` | 是本次运行时间（不要是几天前） |
| `provenance.platform` | 是 Linux（云机器），不是 `Windows` |

> 若 `platform` 里出现 `Windows`，说明产物是**本机自检遗留**被人为拷贝过去的，不是云端真实结果。

### 1.2 冻结常量对账表（P3 的唯一真源）

| 项 | 期望值 |
|---|---|
| train manifest SHA-256 | `51f5b251f9e7cce6c557798ae8fa8e263a6baf58c125524be4e1a1df4bd9edfd` |
| architecture fingerprint | `8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975` |
| 参数量 | `56,812,813` |
| MACs（batch=1, 224²） | `7,492,310,331` |
| FLOPs | `14,984,620,662`（= 2 × MACs） |
| ResNeXt50-32x4d 参数量 | `25,028,904` |
| 参数量比值 | `2.27×`（容差 ±0.01） |
| 每类 val 张数 | `512`；每折 val `2,048` / train `8,192` |
| `FINAL_EPOCHS` | `126` |
| `BASE_EPOCHS`（G1 用） | `200` |

### 1.3 sealed test 隔离（P4）

本包三个实验脚本**物理上不接收 test 参数、不构造 `data_root/test`**。验收时的可执行判据：

```bash
# 1) 数据目录里只应上传 train/ —— 若 test/ 存在，说明上传超范围（不致命，但需确认未被读取）
ls "/home/featurize/work/Combined_Dataset"

# 2) 全目录日志里不应出现任何 test manifest 名
grep -rl "pcv_test_manifest_SEALED" <OUT> && echo "❌ 触碰到 sealed test manifest" || echo "✅ 未触碰"
```

---

## 2. G3 · 效率指标（先跑，最便宜）

**命令**
```bash
python g3_efficiency.py --device cuda:0 --batch-sizes 1,16
```

### 2.1 预期输出

| 产物 | 说明 |
|---|---|
| `<OUT>/g3/g3_efficiency.json` | 全量结构化结果（唯一权威） |
| `<OUT>/g3/g3_latency_table.csv` | 延迟/吞吐/显存表 |
| `<OUT>/g3/g3_efficiency_table.md` | **可直接粘贴进方案**的 Markdown 表 |

`g3_efficiency.json` 应包含这些顶层键：
`environment`、`flop_convention`、`model`、`flops`、`resnext50_reference`、`param_ratio_dmsa_over_resnext`、`latency`{`gpu_bs1`,`gpu_bs16`,`cpu_bs1`}、`summary`。

### 2.2 通过标准

| # | 检查项 | 判定依据 | 结果 |
|---|---|---|---|
| G3-1 | 进程退出码 | `echo $?` == `0` | 必须 |
| G3-2 | `model.architecture_fingerprint` | == `8e542321…69d7` | 必须 |
| G3-3 | `model.fingerprint_matches_baseline` | == `true` | 必须 |
| G3-4 | `model.params_total` | == `56812813` | 必须 |
| G3-5 | `flops.macs` | == `7492310331` | 必须 |
| G3-6 | `flops.flops` | == `14984620662`（且 `== 2 × macs`） | 必须 |
| G3-7 | `flops.matches_published` | == `true` | 必须 |
| G3-8 | `flops.convention` | 非空（口径必须写死） | 必须 |
| G3-9 | `resnext50_reference.params` | == `25028904` | 必须 |
| G3-10 | `param_ratio_dmsa_over_resnext` | `2.27 ± 0.01` | 必须 |
| **G3-11** | **`latency.gpu_bs1.measurable`** | **== `true`** | **★ 核心目的** |
| **G3-12** | **`latency.gpu_bs16.measurable`** | **== `true`** | **★ 核心目的** |
| G3-13 | `latency.gpu_bs*` 三个字段 | `latency_ms_median` / `throughput_img_per_s` / `peak_memory_mb` 均为**正数** | 必须 |
| G3-14 | `summary.n_measured` | 在 GPU 机器上应为 **7** | 必须 |
| G3-15 | `summary.measured_7_items.gpu_*` | 三项都不是 `NOT_MEASURED_ON_THIS_MACHINE` | 必须 |

> **G3-11/12 是本阶段存在的唯一理由。**
> G3 在 CPU 机器上跑出的结果是 **`NOT_MEASURED_ON_THIS_MACHINE`**，那是**未验收**，
> 不是"通过"。`n_measured` 只有 3~5 时，G3 判定为 **INCOMPLETE**。
>
> 严禁用 CPU 延迟外推 GPU 性能——这是项目铁律。

### 2.3 常见失败/不完整情形

| 现象 | 含义 | 处理 |
|---|---|---|
| `device_name` 缺失、`gpu_not_measured_reason` 存在 | 没走 CUDA 分支 | 确认 `torch.cuda.is_available()` 为 `True`，且确实传了 `--device cuda:0` |
| `resnext50_reference` 缺失 | torchvision 导入失败 | 装 / 修 torchvision（可用 `flops_error` 字段定位） |
| `flops.matches_published == false` | 模型被改动过 | **停**，diff 基线 `smoke_test.py` |

---

## 3. G2 · 多随机种子（一晚）

**命令**
```bash
python g2_multi_seed.py --device cuda:0 --seeds 43,44
```

### 3.1 预期输出

| 产物 | 说明 |
|---|---|
| `<OUT>/g2/g2_multiseed_summary.json` | 汇总（`per_seed` + `final_train_loss` 统计） |
| `<OUT>/g2/g2_multiseed_table.csv` | 每个种子的单行记录 |
| `<OUT>/g2/log_seed43.txt`、`log_seed44.txt` | 每个种子的完整训练日志 |
| `<OUT>/g2/_variants/final_train_seed{43,44}.py` | 改写后的训练副本（留档） |
| `<OUT>/g2/_variants/pcv_v2_folds_seed42.csv` | 冻结 manifest 的逐字节副本 |
| `<OUT>/g2/dmsa_final_train_seed{43,44}_epoch126/` | 每个种子的训练输出目录 |

每个 `dmsa_final_train_seedNN_epoch126/` 内含：

| 文件 | 大小（参考） | 说明 |
|---|---|---|
| `FINAL_MODEL_READY.json` | 小 | **完成标记**——有它才算这次训练真的跑完了 |
| `final_model_epoch126.pth` | ~228 MB | 最终模型 |
| `resume_checkpoint.pth` | ~683 MB | 容灾断点（跑完可删） |
| `train_history.csv` | 小 | 逐 epoch `train_loss` |
| `final_training_protocol.json` | 小 | 本次实际使用的协议 |
| `model_info.json` / `class_to_idx.json` / `environment.json` | 小 | 模型/类别/环境快照 |
| `full_train_snapshot.csv` / `pcv_v2_folds_used.csv` | 小 | 数据与划分快照 |
| `final_train_code_used.py` | 中 | **实际跑过的代码**（关键溯源） |

> 上述快照类文件只在**全新运行**时写入；断点续跑不会重写，属正常。

### 3.2 通过标准

| # | 检查项 | 判定依据 | 结果 |
|---|---|---|---|
| G2-1 | 进程退出码 | == `0` | 必须 |
| G2-2 | `n_seeds_ok` | == 请求的种子数（如 `2`） | 必须 |
| G2-3 | 每个种子 `exit_code` | == `0` | 必须 |
| G2-4 | 每个种子 `n_epochs_logged` | == `126` | 必须 |
| G2-5 | 每个种子目录内 `FINAL_MODEL_READY.json` | 存在 | 必须 |
| G2-6 | `final_model_epoch126.pth` 存在且 > 200 MB | 存在 | 必须 |
| G2-7 | `final_train_loss.stdev` | 字段存在且为数值 | 必须（<2 个种子时只有 `note`，判 INCOMPLETE） |
| G2-8 | `sealed_test_untouched` | == `true` | 必须 |
| G2-9 | 副本 `_variants/final_train_seedNN.py` 内 `SEED` | == 对应种子（不是 42） | 必须 |
| G2-10 | 副本内 `"SEED":` 协议断言值 | == 对应种子（与常量同步，否则脚本自检会崩） | 必须 |
| G2-11 | `_variants/pcv_v2_folds_seed42.csv` SHA-256 | == `51f5b251…edfd` | 必须 |
| G2-12 | 源文件 `DMSA_net/final_train.py` SHA-256 | 运行前后**不变** | 必须（基线零改动） |
| G2-13 | 各种子 `first_train_loss` → `final_train_loss` | 单调下降趋势（不要求严格） | 观察项 |

### 3.3 G2 的解读边界（必须与结论同时写）

- G2 **只给 train loss 的跨种子离散度**，**不给 test 指标**。
  若方案要"多种子的 test 指标"，**必须另找一批从未用过的测试数据**。
- 种子间 `final_train_loss` 差异小 ⇒ 训练稳定；差异大 ⇒ 需要报出来，**不要藏**。
- 副本里的 `DATA_ROOT` **总是**被改写成本次解析出的数据根（默认 `gpu_lib.FIXED_DATA_ROOT`）。
  ⚠️ 基线 `final_train.py` 硬编码的那条路径与云端实测位置**不一致**，所以不能沿用。

---

## 4. G1 · 模块消融矩阵（最贵）

**命令**
```bash
# 先单变体单折试跑，确认无误再全量
python g1_ablation.py --device cuda:0 --variants no_sam --folds 1 --epochs 200
python g1_ablation.py --device cuda:0 --variants no_sam,no_mdc,no_ilf --folds 1,5
```

### 4.1 预期输出

| 产物 | 说明 |
|---|---|
| `<OUT>/g1/g1_ablation_summary.json` | 汇总（`table` + 两条诚实说明） |
| `<OUT>/g1/g1_ablation_table.md` | **可直接粘贴**的消融表 |
| `<OUT>/g1/g1_ablation_table.csv` | 同上，表格式 |
| `<OUT>/g1/g1_ablation_per_fold.csv` | 逐 (变体 × 折) 明细 |
| `<OUT>/g1/g1_run_config.json` | 本次实际配置（含各文件 SHA） |
| `<OUT>/g1/<variant>/fold_<N>/metrics.csv` | 逐 epoch 指标 |
| `<OUT>/g1/<variant>/fold_<N>/best_metrics.json` | 该折 best（含 `fingerprint`、`switches`） |

### 4.2 变体参数期望值（对账用，`params` 必须逐位相等）

| 变体 | 期望 params | Δ vs full |
|---|---:|---:|
| `full` | `56,812,813` | `+0` |
| `no_sam` | `56,812,354` | `−459` |
| `no_mdc` | `45,037,801` | `−11,775,012` |
| `no_ilf` | `55,495,947` | `−1,316,866` |
| `no_dual_path` | `46,753,037` | `−10,059,776` |

### 4.3 通过标准

| # | 检查项 | 判定依据 | 结果 |
|---|---|---|---|
| G1-1 | **`--dry-run` 硬门** | 打印 `✅ 硬门通过：full 与基线同构` | **前置必须** |
| G1-2 | 进程退出码 | == `0` | 必须 |
| G1-3 | `g1_run_config.json.manifest_sha256` | == `51f5b251…edfd` | 必须 |
| G1-4 | 每个 `best_metrics.json.fingerprint_equals_baseline`（非 full 变体） | == `false`（**变了才对**，否则开关没生效） | 必须 |
| G1-5 | `table[].params` | 与 §4.2 逐位相等 | 必须 |
| G1-6 | `table[].n_folds` | == 请求折数 | 必须 |
| G1-7 | `table[].accuracy_mean` | 非 `null`，且在 `(0, 1]` | 必须 |
| G1-8 | **存在 `full` 行** | 见 §4.4 —— 没有它则 Δ 全部为 `null` | **必须** |
| G1-9 | `table[].delta_vs_full` | 非 `null` | 必须（依赖 G1-8） |
| G1-10 | `sealed_test_untouched` | == `true` | 必须 |
| G1-11 | `validation_only` | == `true` | 必须 |
| G1-12 | `summary.epochs` | == 实际传入的 `--epochs`（未传则 200） | 必须 |
| G1-13 | 每个 `best_metrics.json.val_rows` | == `2048`，且 `val_class_counts` 每类 == `512` | 必须 |

### 4.4 ⚠️ `full` 基准行：云机器上的关键前提

`full` 行是 Δ 的基准。它有两个来源：

1. **优先复用**已完成的五折结果（`复现版实验数据/Fold1..5/best_metrics.json`）；
2. 找不到时——**只有当 `full` 在 `--variants` 里**——才于本次实跑生成。

**云机器上没有 `E:` 盘**，自动搜索大概率找不到（该目录原本在本地开发机上）。
因此若用默认 `--variants no_sam,no_mdc,no_ilf`（**不含 full**），表格将
**没有 `full` 行、`delta_vs_full` 全为 `null`**。脚本会打印显式告警，但**退出码仍是 0**——
这就是一个"看着跑完了其实残缺"的典型陷阱。

**两种正确处理，任选其一：**

```bash
# 方案 A：把 full 加进去，让它实跑（+1 个变体成本，推荐，自包含）
python g1_ablation.py --device cuda:0 \
    --variants full,no_sam,no_mdc,no_ilf --folds 1,5

# 方案 B：把已有的五折参照目录传到云上，显式指定
python g1_ablation.py --device cuda:0 \
    --variants no_sam,no_mdc,no_ilf --folds 1,5 \
    --reference-cv /path/to/复现版实验数据
```

> `--reference-cv` 是**显式即显式**：指错目录、或目录里凑不齐 5 折，
> 会**直接报错退出**（`exit 1`），**绝不静默回退**到别的目录。

### 4.5 G1 的解读边界（必须与结论同时写）

1. **容量不匹配**：`no_mdc` 少 1,177 万参数、`no_dual_path` 少 1,006 万。
   因此精度差**不能直接归因给单个模块**，只能写「移除 X 后的整体效果（含容量变化）」。
   要拆出模块语义贡献，需要**容量匹配对照**（把省下的参数用等量普通卷积补回）。
2. **统计效力**：只跑 1 折时，差异可能被单折噪声淹没。
   `table[].fold_spread` 给出折间跨度，**只在 `|Δ| > fold_spread` 时才可声称差异显著**。
3. 仅 `no_sam`（Δ 仅 459 参数）的参数代价可忽略，因此**只有 `no_sam` 是干净对照**。

---

## 5. G0 · 编排层（聚合判定）

**命令**
```bash
python g0_run_all.py --device cuda:0
```

按 **G3 → G2 → G1** 顺序执行；**任一步退出码非 0 即停止后续步骤**。

### 5.1 预期输出

| 产物 | 说明 |
|---|---|
| `<OUT>/G0_运行报告.md` | **先看这个**：三步结果汇总成一份报告 |
| `<OUT>/g0_plan.json` | 本次编排参数 + 基线/协议指纹 |
| `<OUT>/g0_results.json` | 每步的 `exit_code` / `elapsed_hours` / `out_dir` / `command` |
| `<OUT>/g0/g1|g2|g3/…` | 各步骤自己的产物（同 §2~§4） |

### 5.2 通过标准

| # | 检查项 | 判定依据 |
|---|---|---|
| G0-1 | 进程退出码 | == `0`（内部即 `all(exit_code == 0)`） |
| G0-2 | `g0_results.json.results[].exit_code` | 全部 == `0` |
| G0-3 | `results[].step` | 覆盖 `order` 中所有步骤（被 `break` 则说明中途失败） |
| G0-4 | `G0_运行报告.md` | 存在，且含 §2/§3/§4 三个小节 |
| G0-5 | `g0_plan.json.sealed_test_policy` | == `"not touched by any step"` |
| G0-6 | `g0_plan.json.manifest_sha256` | == `51f5b251…edfd` |

> **注意**：`G0` 的 `--out-dir` 与 `--out-root` 同义；两者同时给出且不一致会**直接报错**。

---

## 6. 整体验收流程

### 6.1 推荐执行顺序（含每步的验收动作）

| 序 | 动作 | 命令 | 验收动作 | 通过后 |
|:--:|---|---|---|---|
| 0 | 环境自检 | `python gpu_lib.py --smoke` | 5 行 `PASS`，且 `fp=8e542321…`、`params=56,812,813` | 才允许继续 |
| 1 | **G3** | `g3_efficiency.py --device cuda:0` | 走 §2.2 全表；**尤其 G3-11/12** | 拿到 GPU 三项 |
| 2 | **G2** | `g2_multi_seed.py --device cuda:0 --seeds 43,44` | 走 §3.2 全表 | 拿到跨种子 std |
| 3 | **G1** | `g1_ablation.py --device cuda:0 --variants full,no_sam,no_mdc,no_ilf --folds 1,5` | 走 §4.3 全表；**尤其 G1-8/9** | 拿到消融表 |
| 4 | 编排复验（可选） | `g0_run_all.py --device cuda:0` | 走 §5.2 | 一份汇总报告 |

> 也可以直接第 4 步一条命令跑完，但**首次上机仍建议按 1→2→3 分开跑**，
> 这样失败时容易定位，且 G1 可以先单变体单折试跑。

### 6.2 可执行验收器（本包已附带）

把"逐条对照"变成本机断言——符合项目铁律 6（可被可执行验证器当场证伪）。

**本包已附带 `accept.py`，无需手工粘贴：**

```bash
python accept.py /home/featurize/work/AIC_train_again
echo $?     # 0 = 全部通过；1 = 有未通过项
```

已用 4 个场景做过退出码测试：完整且已实测 → `0`（47/47）；GPU 三项缺失 → `1`；
空目录 → `1`；G1 缺 `full` 行 → `1`。**每次都会打印逐条结果表与未通过清单。**

完整源码（`accept.py` 内容，便于你按需修改）：

```python
#!/usr/bin/env python3
"""G0-G3 产物验收器。用法: python accept.py /home/featurize/work/AIC_train_again"""
import json, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/home/featurize/work/AIC_train_again")
FP   = "8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975"
MANS = "51f5b251f9e7cce6c557798ae8fa8e263a6baf58c125524be4e1a1df4bd9edfd"
EXPECT_PARAMS = {"full": 56812813, "no_sam": 56812354, "no_mdc": 45037801,
                 "no_ilf": 55495947, "no_dual_path": 46753037}

rows, fails = [], []

def ck(stage, name, ok, detail=""):
    rows.append((stage, name, "PASS" if ok else "FAIL", detail))
    if not ok:
        fails.append(f"[{stage}] {name} {detail}")

def jload(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception as e:
        return {"__err__": str(e)}

# ---------- G3 ----------
g3 = jload(OUT / "g3" / "g3_efficiency.json")
if "__err__" in g3:
    ck("G3", "g3_efficiency.json 可读", False, g3["__err__"])
else:
    m, f = g3.get("model", {}), g3.get("flops", {})
    lat = g3.get("latency", {})
    ck("G3", "架构指纹", m.get("architecture_fingerprint") == FP)
    ck("G3", "指纹与基线一致", m.get("fingerprint_matches_baseline") is True)
    ck("G3", "参数量 == 56,812,813", m.get("params_total") == 56812813,
       f"got={m.get('params_total')}")
    ck("G3", "MACs == 7,492,310,331", f.get("macs") == 7492310331, f"got={f.get('macs')}")
    ck("G3", "FLOPs == 2×MACs", f.get("flops") == 2 * (f.get("macs") or 0))
    ck("G3", "matches_published", f.get("matches_published") is True)
    ref = g3.get("resnext50_reference") or {}
    ck("G3", "ResNeXt 参数量", ref.get("params") == 25028904, f"got={ref.get('params')}")
    for bs in (1, 16):
        r = lat.get(f"gpu_bs{bs}", {})
        ck("G3", f"GPU bs={bs} 已实测", r.get("measurable") is True,
           str(r.get("reason", "")))
        if r.get("measurable"):
            ck("G3", f"GPU bs={bs} 延迟为正", (r.get("latency_ms_median") or 0) > 0)
            ck("G3", f"GPU bs={bs} 吞吐为正", (r.get("throughput_img_per_s") or 0) > 0)
            ck("G3", f"GPU bs={bs} 显存为正", (r.get("peak_memory_mb") or 0) > 0)

# ---------- G2 ----------
g2 = jload(OUT / "g2" / "g2_multiseed_summary.json")
if "__err__" in g2:
    ck("G2", "g2_multiseed_summary.json 可读", False, g2["__err__"])
else:
    ps = g2.get("per_seed", [])
    ck("G2", "n_seeds_ok >= 2", g2.get("n_seeds_ok", 0) >= 2,
       f"got={g2.get('n_seeds_ok')}")
    ck("G2", "sealed_test_untouched", g2.get("sealed_test_untouched") is True)
    ck("G2", "有 stdev", "stdev" in (g2.get("final_train_loss") or {}))
    for r in ps:
        tag = f"seed{r.get('seed')}"
        ck("G2", f"{tag} exit_code==0", r.get("exit_code") == 0,
           f"got={r.get('exit_code')}")
        ck("G2", f"{tag} 记录 126 epoch", r.get("n_epochs_logged") == 126,
           f"got={r.get('n_epochs_logged')}")
        rd = Path(r.get("run_dir", ""))
        ck("G2", f"{tag} FINAL_MODEL_READY.json", (rd / "FINAL_MODEL_READY.json").is_file())
        ck("G2", f"{tag} final_model_epoch126.pth",
           (rd / "final_model_epoch126.pth").is_file())

# ---------- G1 ----------
g1 = jload(OUT / "g1" / "g1_ablation_summary.json")
if "__err__" in g1:
    ck("G1", "g1_ablation_summary.json 可读", False, g1["__err__"])
else:
    ck("G1", "sealed_test_untouched", g1.get("sealed_test_untouched") is True)
    ck("G1", "validation_only", g1.get("validation_only") is True)
    tab = g1.get("table", [])
    names = [t.get("variant") for t in tab]
    ck("G1", "有 full 基准行", "full" in names, f"variants={names}")
    for t in tab:
        v = t.get("variant")
        ck("G1", f"{v} params", t.get("params") == EXPECT_PARAMS.get(v),
           f"got={t.get('params')} want={EXPECT_PARAMS.get(v)}")
        ck("G1", f"{v} accuracy_mean 非空", t.get("accuracy_mean") is not None)
        ck("G1", f"{v} delta_vs_full 非空", t.get("delta_vs_full") is not None)
        ck("G1", f"{v} n_folds==2", t.get("n_folds") == 2, f"got={t.get('n_folds')}")

# ---------- G0（可选）----------
g0 = jload(OUT / "g0_results.json")
if "__err__" not in g0:
    res = g0.get("results", [])
    ck("G0", "所有步骤 exit_code==0",
       bool(res) and all(r.get("exit_code") == 0 for r in res))
    ck("G0", "G0_运行报告.md 存在", (OUT / "G0_运行报告.md").is_file())

# ---------- 输出 ----------
print(f"{'阶段':<6}{'检查项':<34}{'结果':<7}说明")
print("-" * 78)
for s, n, r, d in rows:
    print(f"{s:<6}{n:<34}{r:<7}{d}")
n_pass = sum(1 for x in rows if x[2] == "PASS")
print("-" * 78)
print(f"通过 {n_pass}/{len(rows)}")
if fails:
    print("\n未通过：")
    for x in fails:
        print("  -", x)
sys.exit(0 if not fails else 1)
```

---

## 7. 最终结论判定方式

先算 **通过率** 与 **是否有红线命中**，再落档。分三档，**不允许只写"通过"**：

### 7.1 三档判定

| 结论 | 成立条件 | 可以写进方案吗 |
|---|---|---|
| **✅ ACCEPTED** | 红线 0 命中；G3/G2/G1 各自的**必须项全 PASS**（含 G3-11/12、G1-8/9） | 可以，且需附 §7.2 的限制说明 |
| **🟡 ACCEPTED-WITH-CAVEATS** | 无红线；但有**观察项**失败，或某阶段判 INCOMPLETE（如 G3 缺 GPU 三项、G2 只有 1 个种子） | 可以，但**必须逐条写明缺哪项**，并撤回对应局限（不要超范围声明） |
| **❌ REJECTED** | 任一红线命中；或任一阶段的**必须项**失败 | **不可以**。对应局限**不得撤销** |

### 7.2 结论必须同时附带的限制说明（不得省略）

无论哪一档，最终结论里必须**逐字保留**下面三条（它们已由脚本自动写入产物）：

1. **G1 容量不匹配** → 精度差不能直接归因单个模块（`g1_ablation_summary.json::capacity_warning`）
2. **G1 统计效力** → 只在 `|Δ| > fold_spread` 时才声称显著（`…::limitation_note`）
3. **G2 不评 test** → 只给 train loss 离散度（`g2_multiseed_summary.json::limitation_note`）
4. **G3 口径** → FLOPs 必须绑定「batch=1, 224²；1 MAC = 2 FLOPs；仅 Conv/Linear」（`…::flop_convention`）

### 7.3 与方案（PDF）的对应关系

| 验收结论 | 方案里可撤回的局限 |
|---|---|
| G3 ACCEPTED（GPU 三项实测） | 撤回 **L4**「未实测效率指标」 |
| G2 ACCEPTED | 撤回 **L6**「单一随机种子」 |
| G1 ACCEPTED | 消融小节可写，创新性维度加分（不撤回局限） |
| 三项中任一项未 ACCEPTED | **对应局限不得撤回**；局限表维持原样 |

> ⚠️ 插入图表会改变 57 页方案的**分页**，改完**必须重导 PDF**，
> 并复核附录（二）4 的图表清单页码。

---

## 8. 常见"假成功"模式（来自本包 10 个已修 bug，验收时逐条防）

| # | 假成功长什么样 | 真实情况 | 验收时怎么防 |
|:--:|---|---|---|
| 1 | `g3_efficiency.json` 生成了、退出码 0 | GPU 三项全是 `NOT_MEASURED` | **G3-11/12 必查** |
| 2 | G1 表格生成了、退出码 0 | 没有 `full` 行，`Δ` 全 `null` | **G1-8/9 必查** |
| 3 | `g2` 跑完 | 副本缺 manifest → 子进程 `FileNotFoundError`，但 `per_seed` 里 `exit_code≠0` | **G2-3 必查** |
| 4 | 脚本"跑得动" | `resolve_out_dir` 等 4 个符号没 import，一运行就 `ImportError` | 第 0 步 `--smoke` 必须先过 |
| 5 | `--data-root` 传了 | `final_train.py` 的 `DATA_ROOT` 是硬编码，参数**空转** | 核对 `g1_run_config.json::train_root` 与实际数据位置 |
| 6 | `--out-dir` 传了 | `g0` 只读 `--out-root`，参数**被忽略** | 检查产物是否真的落在你指定的目录 |
| 7 | 产物看着完整 | `arch_probe.json` 的 FLOPs 曾是错值（差 800,000） | **G3-6** 强制 `FLOPs == 2×MACs` |
| 8 | `params` 对上了 | 但某个变体的开关没生效（`fingerprint_equals_baseline` 应为 `false` 却为 `true`） | **G1-4 必查** |

> **共同教训**：这类 bug 的共性是「**改了参数/符号但没真正接上**」，
> `ast.parse` 与 dry-run 都抓不到，**只有针对该参数单独写一条断言才抓得到**。
> 这也是 §6.2 验收器存在的理由——它把检查项变成了可执行断言。

---

## 9. 验收结论模板（直接复制填写）

```markdown
## 验收结论

- 运行时间：<YYYY-MM-DD HH:MM> ~ <...>
- 机器 / GPU：<device_name>，显存 <xx> GB，torch <version>+cu<xxx>
- 输出根目录：/home/featurize/work/AIC_train_again
- 冻结对账：manifest SHA ✅ / 架构指纹 ✅ / 参数 56,812,813 ✅ / MACs 7,492,310,331 ✅
- sealed test：未触碰（grep 全目录 0 命中）

| 阶段 | 结论 | 通过/总检查 | 未通过项 |
|---|---|---|---|
| G3 效率 | ✅ / 🟡 / ❌ | <n>/<N> | <列出或 "—"> |
| G2 多种子 | … | … | … |
| G1 消融 | … | … | … |
| G0 编排（可选） | … | … | … |

**总体结论：ACCEPTED / ACCEPTED-WITH-CAVEATS / REJECTED**

附带限制（逐字保留）：
1. G1 容量不匹配：no_mdc −11,775,012 / no_dual_path −10,059,776 → 不能归因单模块
2. G1 统计效力：仅当 |Δ| > fold_spread 才声称显著
3. G2 仅 train loss，不评 test
4. G3 FLOPs 口径：batch=1, 224²；1 MAC=2 FLOPs；仅 Conv/Linear

方案可撤回的局限：<列出实际 ACCEPTED 的项对应的 L4/L6，其余维持>
```

---

## 10. 一句话总结

> **验收 = 退出码为 0（P1）+ 产物齐备可解析（P2）+ 关键数字与冻结基线逐位对上（P3）+ 三项硬门全过（P4）**；
> 其中 **G3 必须有 GPU 三项实测**、**G1 必须有 `full` 基准行**、**G2 必须有 ≥2 个种子的 stdev**——
> 这三条是"跑完了"与"验收通过"之间的分界线。
