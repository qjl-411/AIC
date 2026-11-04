# GPU 实验脚本 · 运行手册

> 面向：需要把 `04_需用户本人操作事项反馈.md` §5 的 ★★/★★★ 补充实验跑出来的人。
> 本目录的脚本**已在 CPU 上全部验证接线可用**，你租到 GPU 机器后可直接运行。
> **2026-09-27 更新**：本手册改为**三段式**。第一段（G3 / G2 / G1）**已跑完**；
> **第二段（`g4_run.py`）= 一次跑完「对照五折 + 跨种子 + 完整消融矩阵」，正式步骤见 §5（≈105 h）**；
> **第三段（`g5_run.py`）= 最省补完**（消融补 `no_dual_path`×2折 + 多种子 seed43/44×2折，6 run ≈15.75 h ≈¥38），正式步骤见 **§6**。
> 数据根目录已按云端实测位置定为 `/home/featurize/work/Combined_Dataset`；上传布局确认为「全部平铺同一目录」。

---

## 0. 一分钟速览

### 0.1 三段实验的分工（先看这张表）

| 段 | 脚本 | 状态 | 覆盖的提分项 |
|---|---|---|---|
| **第一段** | `g3_efficiency.py` → `g12_run.py`（= g2 + g1） | ✅ **已跑完**（2026-09-25，serial 总 19:48:24，exit 0/0） | ★ 效率 GPU 三项；★★ 多种子（**仅 train_loss**）；★★ 消融（**3 变体 × 2 折，参照未对齐**） |
| **第二段** | `g4_run.py`（**单文件**） | ⬜ **待跑**（≈105 h） | ★★★ 对照五折；★★ 消融补齐（**4 变体 × 5 折，参照已对齐**）；★★ 多种子（**补 val/acc**） |
| **第三段** | `g5_run.py`（**单文件**） | ⬜ **待跑**（**≈15.75 h ≈¥38**） | ★★ 消融**补最后一个变体**（`no_dual_path` × 2 折）；★★ 多种子**补 val/acc**（seed 43/44 × 2 折） |

> **三段的关系**：第二段（g4，≈105 h）是**完整档**；第三段（g5，≈15.75 h）是**省钱档** ——
> 它**只补 g4 里最便宜的两块、且只跑 2 折**，**不覆盖 ★★★ 对照五折、也不把消融补到 5 折**。
> 预算紧 → 先跑 **g5（§6）**；预算够 → 直接跑 **g4（§5）**，g5 的两块被 g4 完全包含、不必重复。

**为什么还要第二段**（一句话）：第一段留了三个洞 ——
① ★★★**对照五折从未跑过**；
② 消融**缺 `no_dual_path`**，且 `full` 参照来自**另一个脚本（`train_5fold.py` ≠ `smoke_test.py`）、另一张卡（RTX 3060 ≠ 4090）、另一个 `data_root`（空格路径 ≠ 下划线路径）**，所以 Δ 只能叫「参考性对照」、不可归因；
③ 多种子**只落了 `train_loss`，没有 val/acc**，不足以撤销 L6。
**`g4_run.py` 一次把这三个洞补齐。**

### 0.2 第二段（g4）速览

| 项 | 值 |
|---|---|
| 脚本 | `g4_run.py`（**单文件**，40 run 一次跑完，无需另开终端看进度） |
| 任务构成 | A1 分组 `full`×5 ｜ **B 4 变体 × 5 折（20 run）** ｜ A2 图像级 `full`×5 ｜ A3/A4 分组 `full`×seed43/44×5 |
| 时长 | **≈ 105.0 h（≈ 4.4 天）**，按 G1 实测单 run 2.63 h 外推（**不是估算**） |
| 上传 | 只需**新增 2 个文件**：`g4_run.py` + `pcv_v2_folds_imgsplit_seed42_FROZEN.csv` |
| 命令（前台） | `python g4_run.py --plan` → `python g4_run.py` |
| 命令（**后台，推荐**） | `nohup python -u g4_run.py --quiet > g4_console.log 2>&1 &` — 关机/断网/SSH 断都不影响，见 **§5.9** |
| → | **正式步骤见 §5** |

### 0.3 第三段（g5）速览

| 项 | 值 |
|---|---|
| 脚本 | `g5_run.py`（**单文件**，6 run 一次跑完；内部 **import 复用** `g4_run.py` 的训练/探测函数） |
| 任务构成 | 阶段 1 `ABL`：`no_dual_path` × 折 {1,5}（2 run）｜阶段 2 `SEED`：`full` × seed 43/44 × 折 {1,5}（4 run） |
| 时长 | **≈ 15.75 h（≈ 0.7 天）**，按 G1 实测单 run 2.625 h 外推（**不是估算**） |
| 费用 | **≈ ¥38**（单价 ¥2.4/h 为**反推值**，不是平台报价） |
| 上传 | 只需**新增 1 个文件**：`g5_run.py`（其余云端已有） |
| 命令（前台） | `python g5_run.py --plan` → `python g5_run.py --dry-run --device cpu` → `python g5_run.py` |
| 命令（**后台，推荐**） | `nohup python -u g5_run.py --quiet > g5_console.log 2>&1 &` — 关机/断网/SSH 断都不影响，见 **§6.9** |
| → | **正式步骤见 §6** |

### 0.4 第一段脚本清单（已跑完，留档备查）

| 脚本 | 作用 | 成本 | 产出 |
|---|---|---|---|
| `gpu_lib.py` | 共享库（协议守卫 / 落盘 / 设备探测） | — | 被其他脚本 import |
| `g3_efficiency.py` | ★ 补齐 GPU 延迟/吞吐/显存 | **~0.2 h** | 效率表（可粘贴） |
| `g2_multi_seed.py` | ★★ 多随机种子重复 | **8–18 h** | 跨种子 std |
| `g1_ablation.py` | ★★ 模块消融矩阵 | **16–32 h** | 消融表（可粘贴） |
| `g0_run_all.py` | 一键编排上面三个 | 合计 | 汇总报告 |
| `accept.py` | **跑完后验收产物**（退出码 0/1 + 逐条结果表） | 秒级 | 验收结论 |
| `验收报告_G0-G3.md` | 各阶段预期输出 / 通过标准 / 最终结论判定方式 | — | 验收依据 |

**最低建议投入**：先跑 `g3`（20 分钟），再跑 `g2`（一晚）。合计约 ¥20–40。
跑完后 **必跑** `python accept.py /home/featurize/work/AIC_train_again`。

---

## 1. 上传清单

**全部平铺到同一个目录**（本包就是 `Again_train/` 的原样结构），文件名一个都别改：

```
# —— 第一段（已跑完；若只是补跑第二段，这些云端已经有了）——
gpu_lib.py  g0_run_all.py  g1_ablation.py  g2_multi_seed.py  g3_efficiency.py  g12_run.py
accept.py
RUNBOOK.md  验收报告_G0-G3.md
smoke_test.py  final_train.py  pcv_v2_folds_seed42.csv
arch_probe.py  lib_provenance.py

# —— 第二段（g4）新增，只多这两个 ——
g4_run.py  pcv_v2_folds_imgsplit_seed42_FROZEN.csv

# —— 第三段（g5）新增，只多这一个（g5 复用 g4_run.py，故 g4_run.py 须在位）——
g5_run.py
```

> ⚠️ **第三段（g5）的依赖**：`g5_run.py` 会 `import g4_run` 复用其 `train_one_run` / `variant_probe` /
> `load_arch_probe`，并 `import gpu_lib` / `arch_probe` / `smoke_test`。
> ⇒ 上传 `g5_run.py` 时必须保证 **`g4_run.py` 与 `gpu_lib.py` / `arch_probe.py` / `lib_provenance.py` /
> `smoke_test.py` / `pcv_v2_folds_seed42.csv` 都在同一目录**（第一/二段已传过的即满足）。
> `g5_run.py` **不 import** `g1/g2/g3/g12/g0`，也**不引用 `final_train.py`**。

> ⚠️ 第二段的**两个新文件必须与 `g4_run.py` 同目录**：
> `pcv_v2_folds_imgsplit_seed42_FROZEN.csv` 是 `g4_run.py` 硬编码 SHA 回校的对照清单，
> 放错位置或缺席 → 预检直接 `FAIL` 拒绝开跑（**不会静默回退**，这是刻意的）。
> `g4_run.py` **不 import** `g1/g2/g3/g12/g0`，也**不引用 `final_train.py`**（只在注释里提到），
> 它只依赖 `gpu_lib.py` + `smoke_test.py` + `arch_probe.py` + 两份 manifest。

脚本的搜索顺序（**先搜自己所在的目录**，这就是平铺能用的原因）：
- **基线目录**：按 `smoke_test.py` + `final_train.py` + `pcv_v2_folds_seed42.csv`
  三者同时存在来判定。自动搜索失败时用 `--baseline-dir` 显式指定。
- **`arch_probe.py`**：**首选脚本自身目录**，其次才是 `DMSA_v24/` 之类的子目录。
  G1 / g4 / g5 需要它，见下方 §4。

**两个路径已按你的要求定死，不用传参**：

| 用途 | 路径 |
|---|---|
| 数据 | `/home/featurize/work/Combined_Dataset`（用户实测确认） |
| 输出 | `/home/featurize/work/AIC_train_again` |

跑之前先确认数据在位（一条命令）：

```bash
ls "/home/featurize/work/Combined_Dataset/train" | head
```

要换位置，`--data-root` / `--out-root` 随时可覆盖。

> ⚠️ 基线 `final_train.py:1115` / `smoke_test.py:1480` 里硬编码的 `DATA_ROOT` 写的是
> `.../data/dmsa_pcv2/Combined Dataset`，**与上表不一致**。本包一律以
> `gpu_lib.FIXED_DATA_ROOT` 为准；G2 会把每个副本里的 `DATA_ROOT` 强制改写成该值。
>
> 🔴 **路径里没有空格！** 云端目录名是 `Combined_Dataset`（**下划线**），不是
> `Combined Dataset`（空格）。上游 Kaggle 数据集与本地副本用的是**带空格**的名字
> （`Alzheimer's MRI Dataset/.../Combined Dataset/`），**两边不一致**——照抄本地名字会得到
> `ls: cannot access '...': No such file or directory`。
> 用 `ls -d /home/featurize/work/*/` 或 tab 补全核对真实名字，**不要手打**。

**核对云端文件是不是最新包**（一条命令）：

```bash
grep -n "^PKG_VERSION" gpu_lib.py
# 期望： PKG_VERSION = "2026-09-24.3"
```

每个脚本启动时也会打印一行 `包版本      : <版本>`。
🔴 **报错时先把这一行对一下**——如果脚本打印的版本、或它打印的 `data root`，
与你手上的包不一致，说明**云端跑的是旧文件**，重传即可，不必查代码。

### 1.1 最小上传集（已按代码依赖闭包核实）

| 放在哪 | 内容 | 为什么 |
|---|---|---|
| 上传目录（**全部平铺**） | `gpu_lib.py`、`g0_run_all.py`、`g1_ablation.py`、`g2_multi_seed.py`、`g3_efficiency.py`、`g12_run.py` + `accept.py` | `g0/g1/g2/g3/g12` 都 `import gpu_lib`；`accept.py` 验收用。⚠️ 少了 `g12_run.py` 就跑不了「G2+G1 一键启动」（G3 单独跑即可，不再有 `g23_run.py`） |
| 同上 | `smoke_test.py`、`final_train.py`、`pcv_v2_folds_seed42.csv` | 三者是 `find_baseline_dir()` 的判定条件；`smoke_test` 是核心实现真源，`final_train` 是 G2 模板，csv 是冻结 manifest |
| 同上 | `arch_probe.py`、`lib_provenance.py` | **G1 与 g4** 的消融注入真源；`arch_probe` 只 `from lib_provenance import write_artifact`，两者**必须同目录** |
| **第二段专属（新增）** | `g4_run.py`、`pcv_v2_folds_imgsplit_seed42_FROZEN.csv` | g4 单文件脚本 + 它的**对照清单**（脚本内**硬编码 SHA 回校** `d8230219…c168b8`）；缺失或不符 → 预检 FAIL、拒绝开跑 |
| **第三段专属（新增）** | `g5_run.py` | g5 单文件脚本（最省补完）；**`import g4_run`** ⇒ `g4_run.py` 须同时在位；只**读** `g1/` 已有产物、**不写** g1 |
| 同上（可选） | `RUNBOOK.md`、`验收报告_G0-G3.md` | 只供人阅读，脚本不读 |
| 数据 | `Combined Dataset/**train**/` | 见下 |

> **目录布局：平铺（推荐）与嵌套都支持。** `find_baseline_dir()` 与
> `load_arch_probe()` 都会**先搜脚本自身目录**，所以上面这些文件与 5 个脚本
> **放在同一层**即可；若你习惯 `DMSA_net/`、`DMSA_v24/` 的子目录布局，也仍能找到。

**只需要传 `train/`，不用传 `test/`。** `final_train.py` 自己的文档里写明：
「本文件只读取 `Combined Dataset/train/` 与 `pcv_v2_folds_seed42.csv`；
本文件【不会读取 test/，也不会读取 sealed test manifest】」。
G1/G3 同理（G3 完全不用数据）。**所以 sealed test manifest 也不必上传**——
不把它放上机器，等于多一层物理保障。

依赖闭包已逐文件核实：`smoke_test.py` / `final_train.py` / `lib_provenance.py`
三者**均无本地模块 import**；`arch_probe.py` **只** import `lib_provenance`。

---

## 2. 环境自检

### 2.0 先补齐依赖（只需一次）

镜像通常自带 `torch` / `torchvision` / `numpy` / `matplotlib`，但**缺 `scikit-learn`**
（`smoke_test.py:100` 会 `from sklearn.metrics import ...`），少数镜像还缺 `tqdm`：

```bash
pip install scikit-learn tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 🔴 **不要 `pip install torch` / `torchvision`**，也不要用 `requirements.txt` 一把装——
> 那会把镜像里**带 CUDA 的 torch 换成 CPU 版，直接废掉 GPU**。本包只需要上面这两个包。

### 2.1 冒烟自检

```bash
python gpu_lib.py --smoke
```

期望看到 5 行 `PASS`：

```
[1/5] manifest 校验 PASS  rows=10,240  per_class={...2560...}
[2/5] 架构校验 PASS  params=56,812,813  fp=8e542321b288…
[3/5] 真实数据加载 PASS  batch=(2, 3, 224, 224)
[4/5] 前向 PASS  logits=(2, 4)
[5/5] 反传+更新 PASS
```

> 若打印 `[3/5] 跳过真实数据加载（未找到 train/）`，说明**数据根目录没找对**；
> 程序会把试过的路径逐条列出来，对照 §1 的表核对即可。
> 云端会用 `/home/featurize/work/Combined_Dataset/train`，本机 Windows 走本地数据集，
> **两边都应是 5 行 PASS**。

### 2.2 实验脚本的 dry-run（三段都适用）

**第一段（G3 / G2 / G1）：**

```bash
python g3_efficiency.py --dry-run --skip-cpu
python g2_multi_seed.py --dry-run
python g1_ablation.py --dry-run     # 会验证 4 个变体都能构造+前向
```

`g1 --dry-run` 必须打印 `✅ 硬门通过：full 与基线同构`，否则**禁止**用其结果做消融。

**第二段（g4）：**

```bash
python g4_run.py --plan  --device cpu     # 只预检 + 报总时长，不训练
python g4_run.py --dry-run --device cpu   # CPU 上逐个构造 5 个变体，比对参数量与指纹
```

`g4 --dry-run` 必须打印 `✅ dry-run 通过：5/5 个变体参数与指纹均符合文档值。`
期望参数量（对不上会直接报错，不会静默继续）：

| 变体 | params | Δ vs full |
|---|---:|---:|
| `full` | 56,812,813 | +0（= 基线指纹 `8e542321…d76975` ✅） |
| `no_sam` | 56,812,354 | −459 |
| `no_mdc` | 45,037,801 | −11,775,012 |
| `no_ilf` | 55,495,947 | −1,316,866 |
| `no_dual_path` | 46,753,037 | −10,059,776 |

**第三段（g5）：**

```bash
python g5_run.py --plan  --device cpu     # 只预检 + 报总时长/费用，不训练
python g5_run.py --dry-run --device cpu   # CPU 上构造 full + no_dual_path，比对参数量与指纹
```

`g5 --dry-run` 必须打印 `✅ dry-run 通过：`full` 与基线同构，待跑变体均可构造并前向。`
本段只涉及 **2 个变体**：`full`（fp = 基线硬门）与 `no_dual_path`（params=46,753,037）。

> ⚠️ **本机（Windows）自检请加 `--num-workers 0`**，否则 DataLoader 走 `spawn`、
> 子进程 import 不到动态加载的 `dmsa_baseline_smoke` 会崩。**云端 Linux 用 fork，不受影响**（保持默认 6）。
> 原因详见 §10.1。`g4_run.py` / `g5_run.py` 检测到 `os.name == "nt"` 且 `--num-workers > 0` 时会主动打印提示。

---

## 3. 第一段实验（G3 / G2 / G1）· 正式运行

> ✅ **本段已于 2026-09-25 跑完**（`g12_run.log`：serial 总 **19:48:24**，G2/G1 `exit_code` 均为 0；G3 秒级完成）。
> 本节留档，供复跑或核对产物用。**接下来要跑的是第二段（g4），见 §5。**

> **路径已定死，不需要你再传任何路径参数。**
> 输出根目录 = `/home/featurize/work/AIC_train_again`
> 数据根目录 = `/home/featurize/work/Combined_Dataset`（用户实测确认）
> ⚠️ 基线 `final_train.py:1115` / `smoke_test.py:1480` 里写的是
> `.../data/dmsa_pcv2/Combined Dataset`，**与实测位置不一致**；本包不采用它。
> G2 会把副本里的 `DATA_ROOT` 强制改写成本包的值。
> 想换位置再用 `--out-root` / `--data-root` 覆盖即可。

### 3.0 推荐跑法：先单独跑 G3，再用 `g12_run.py` 一键跑 G2 + G1

> ⛔ **一条铁律（2026-09-25 两次事故换来的）**：
> `g1_ablation.py` / `g2_multi_seed.py` 是 `g12_run.py` **调度的子任务**，
> **绝不能手动单独再启动它们**。手动启动的进程不在调度链上，会与正在跑的那条链
> 抢 GPU、并往**同一个输出目录**写（后启动的还会触发覆盖清理，删掉前者已落盘的产物）。
>
> **启动前先体检**（只读，随时可跑）：
> ```bash
> python check_running.py            # 期望：✅ 单链运行 / ✅ 没有发现任务进程
> python check_running.py --strict   # 有异常时退出码 1
> ```
> 它会列出所有任务进程（PID / PPID / 运行时长 / 完整命令行），
> 标出**追不到总调度的孤儿进程**，并给出单杀命令示例。

> 用户实际的执行顺序是 **G3（分钟级）→ G2 + G1（小时级）**。
> G3 只要几分钟，先独占整卡跑完，再把卡完整交给 G2/G1，最省事也最不容易 OOM。

```bash
# ① 先单独跑 G3（几分钟，独占整卡）
python g3_efficiency.py --device cuda:0 --batch-sizes 1,16

# ② 看一眼 G2+G1 的预检查结论（不启动任何训练，几秒钟）
python g12_run.py --plan --seeds 43,44 --variants no_sam,no_mdc,no_ilf --folds 1,5

# ③ 确认无误后一键跑（默认串行：G2 跑完 → 再 G1 跑完，各独占整卡）
python g12_run.py --seeds 43,44 --variants no_sam,no_mdc,no_ilf --folds 1,5

# ④ 有两张卡，想真并行：
python g12_run.py --parallel --g2-device cuda:0 --g1-device cuda:1

# ⑤ 想反过来先跑 G1：
python g12_run.py --order g1,g2
```

**为什么默认顺序是 G2 → G1**（用 `--order g1,g2` 可换）：

| | 工作量口径 | 说明 |
|---|---|---|
| G2 多种子 | `n_seed × 126 epoch × 640 step` | 全量 10,240 张 final 训练 |
| G1 消融 | `n_run × 200 epoch × 512 step`（`n_run = 变体数 × 折数`） | 折内 8,192 张训练 + 2,048 张 val |

G2 工作量较小 → 先跑完能先拿到**一套完整可用**的产物；即便后面中断，
也不会出现"两套都跑一半"。G1 的 val 前向不计入 step，所以它的实际墙钟时间
会比"按 step 数外推"更长一些。

**会占用 GPU 的方式（先说结论）：**

| 策略 | 谁在卡上 | 显存 | 影响 |
|---|---|---|---|
| **serial（默认）** | 同一时刻**只有一个**进程 | G2 独占跑完 → 结束 → G1 独占跑完 | 不会 OOM，互不拖慢 |
| **parallel（各占一卡）** | 两个进程在**不同**卡上 | 各算各的 | 真并行，最快 |
| **parallel（同一张卡）** | 两个进程挤**同一**卡 | **两者相加** | 放得下也会算力争抢，两个都变慢 |

**预检查查什么、结论怎么读：**

| 检查项 | FAIL / WARN 会怎样 |
|---|---|
| 输出目录不重叠（`<root>/g2` vs `<root>/g1`） | 直接中止（exit 2），未启动任何训练 |
| 产物文件名无交集（`g2_*` / `g1_*` + `metrics.csv`/`best_metrics.json`） | 同上 |
| 输出区与数据集互不嵌套 | 同上（这条是"覆盖"功能的安全底线） |
| `train/` 与四个类目录存在 | 同上（提醒目录名**带空格**） |
| **G1 的 full 基准**（五折参照 / `--variants` 含 full） | WARN：既无参照又不跑 full → 表格**没有 Δ vs full**（最容易"跑完了但结果是残的"） |
| 随机种子 / 数据读取 | 结论：无 RNG 耦合；只读共享，仅 IO 竞争（各 6 个 worker） |
| 磁盘空间（G2 每 seed 存 GB 级权重） | < 2 GB → FAIL 中止；< 8 GB → WARN |
| **GPU 显存实测**（真建一次模型跑一个 batch） | 给出 `SERIAL_REQUIRED` / `PARALLEL_OK` / `PARALLEL_WITH_CAVEATS` |

> 显存不是靠"参数量 × 4 字节"估的——激活值才是大头，估不准。
> 预检查会真的建一次模型、跑一个 batch（前向+反传+Adam）测峰值（几秒），
> 再乘 1.15 安全系数下结论。G1 的消融变体容量都 ≤ `full`，
> 所以按 `full` 探测得到的是**上界**，用它下结论是安全的。
> 判定"只能串行"时你强行 `--parallel` **会被拒绝**，除非加 `--force`（后果自负）。

**显存判据：什么情况会中止、什么情况只是警告**（2026-09-24.4 修订）

| 情况 | 结果 |
|---|---|
| 连**一个**任务都放不下（`单任务峰值×1.15 > 卡总容量×0.85`） | ❌ FAIL，中止（这卡真跑不了） |
| 串行放得下，**同卡并行**放不下 | ⚠️ WARN，结论 `SERIAL_REQUIRED`，**不中止**（这就是默认执行方式） |
| 同卡并行也放得下 | ⚠️ WARN，结论 `PARALLEL_WITH_CAVEATS`，仍推荐串行 |

> 🔴 **上一版（.3）的 bug**：把"同卡并行放不下"当成了硬冲突 → 预检查 exit 2，任务根本起不来。
> 而且判据里**单位不一致**（`_probe_peak_mb` 返回字节、`mem_get_info` 是 MB），
> 导致 `放不下` 恒成立——**任何机器都会被中止**。本机无 GPU 时这条路径从不执行，所以测不出来。
> 已修：单位统一到 MB，且只有"单任务放不下"才算 FAIL。
>
> 若看到 `当前空闲 0 MB`：通常是**卡上还挂着别的进程/Jupyter 内核**，或容器上报不准
> （探测本身仍能分配 5 GB，说明卡是好的）。串行不受影响，但建议 `nvidia-smi` 看一眼。

**用 `--reference-cv` 让 `full` 不必实跑（省 2 折 × 200 epoch）**

本机已完成的五折结果在 `E:\Code\NeuroEvo\NeuroEvo_AD\复现版实验数据\Fold1..5\best_metrics.json`
（每个约 200 字节，共 5 个）。把它传到云上（保持 `Fold1..5/` 结构，建议目录名用英文 `ref_cv`）：

```bash
python g12_run.py --reference-cv /home/featurize/work/ref_cv --seeds 43,44 ...
```

给了就有 `full` 行与 Δ vs full，且 **G1 少跑 2 个 run**（省 4–8 GPU h）。
⚠️ 显式指定即显式：**目录不对会直接 FAIL 中止**，不会悄悄回退到别的目录。
不传这个参数也不影响正确性——只是表格没有 Δ vs full（预检查会 WARN）。

**G2 / G1 重跑即覆盖（不需要你手动删目录）：**

```bash
python g2_multi_seed.py --seeds 43,44          # 默认：重跑即覆盖，路径不变
python g2_multi_seed.py --seeds 43 --overwrite-dry-run   # 先看"会删什么"，不删不训
python g2_multi_seed.py --seeds 43 --no-overwrite        # 遇到已有产物就报错（不删）

python g1_ablation.py --variants no_sam --folds 1        # 同样：重跑即覆盖
python g1_ablation.py --variants no_sam --folds 1 --overwrite-dry-run
python g1_ablation.py --variants no_sam --folds 1 --no-overwrite
```

- G2 覆盖时清掉：上一次的 `dmsa_final_train_seedN_epoch126/`（模型/checkpoint/历史）、
  `log_seedN.txt`、`progress_seedN.log`；`g2_multiseed_summary.json` / `_table.csv` 直接重写。
- G1 覆盖时清掉：本次要跑的 `<variant>/fold_N/`（`metrics.csv`、`best_metrics.json`）；
  汇总文件直接重写。**上次跑过、这次没列进 `--variants` 的变体目录不会被删**，
  只会打印一行 `⚠️ 残留目录`，避免误删历史结果。
- 删除前都过 `safe_rmtree()` 的硬校验：**只删 out_dir 之下的产物**，
  数据集目录 / out_dir 本身 / 盘符根 / 符号链接 一律拒绝。

**进度与 ETA（屏幕和日志同一份内容）：**

```bash
tail -f /home/featurize/work/AIC_train_again/g2/progress_seed43.log   # G2：只有进度行
tail -f /home/featurize/work/AIC_train_again/g2/log_seed43.txt        # G2：完整原始输出（实时）
tail -f /home/featurize/work/AIC_train_again/g1/g1_progress.log       # G1：进度 + 关键行
```

- G2：总 step = `126 epoch × 640 iter`（10,240 张 / batch 16）。
- G1：总 step = `变体数×折数 × 200 epoch × 512 iter`（8,192 张 / batch 16）；
  百分比与 ETA 是**跨 变体×折 的全局口径**（不是每个 run 从 0% 重新数）。
- 两者的 step 级进度都来自基线 `train_one_epoch()` 里**本来就有的 tqdm**，
  本包**只解析不改造**，训练逻辑与结果含义 100% 不变。
  G2 从子进程输出流解析；G1 是进程内训练，用替身 stdout 接住 tqdm 帧，
  **即使一帧都没解析到，每 epoch 首尾仍会各强制输出一行**。

### 3.1 三个一起编排（另一个入口）

```bash
python g0_run_all.py --device cuda:0
```

会自动按 **g3 → g2 → g1** 顺序跑，最后产出 `G0_运行报告.md`。
任一步失败即停止，不会带着错误继续。

> 与 `g12_run.py` 的区别：`g0` 一次编排 g3+g2+g1（G3 也在里面）；
> `g12_run.py` 只管 g2+g1，且**带预检查**（显存实测 / 路径撞车 / G1 的 full 基准）。
> 想"先单独跑 G3、再一键跑 G2+G1"，用 `g12_run.py` 更可控。

要调参数就加：

```bash
python g0_run_all.py --device cuda:0 \
    --seeds 43,44 \
    --variants no_sam,no_mdc,no_ilf \
    --folds 1,5
```

只跑其中一步：

```bash
python g0_run_all.py --only g3              # 只跑最便宜的
python g0_run_all.py --skip g1              # 跳过最贵的
```

### 3.2 分开跑（**仅在确认没有任何 `g12_run.py` 在跑时**）

> 🔴 **先跑 `python check_running.py` 确认是"没有发现任务进程"，再往下走。**
> 如果已经有 `g12_run.py` 在跑（它会自动跑完 G2 再跑 G1），
> **绝对不要**再手动起 `g2_multi_seed.py` / `g1_ablation.py`——
> 它们会写同一个 `AIC_train_again/{g2,g1}/` 目录，后启动的那个还会先做覆盖清理，
> 把前者已经跑了几小时、已落盘的产物删掉。

```bash
# ---- 第一步：效率（几分钟，单独跑，先腾出卡）----
python g3_efficiency.py --device cuda:0 --batch-sizes 1,16

# ---- 第二步 + 第三步：多种子 + 消融（一键串行，见 §3.0，推荐）----
python g12_run.py --seeds 43,44 --variants no_sam,no_mdc,no_ilf --folds 1,5

# ---- 只有在"没有 g12 在跑"时，才这样分开跑 ----
python g2_multi_seed.py --device cuda:0 --seeds 43,44              # 多种子（一晚）
python g1_ablation.py --device cuda:0 --variants no_sam --folds 1 --epochs 200
# 确认无误后再全量：
python g1_ablation.py --device cuda:0 \
    --variants no_sam,no_mdc,no_ilf --folds 1,5
```

**断点续跑 / 重跑**：G2 与 G1 **默认都是"重跑即覆盖"**（路径不变），
中断后直接重跑同一条命令即可，不需要手工删目录；
想先看会删什么加 `--overwrite-dry-run`，想禁止覆盖加 `--no-overwrite`。
⚠️ 覆盖 ≠ 断点续跑：G2 会从 epoch 1 重新训（不是接着上次的 checkpoint），
若要从中断处继续，请改副本里的 `RESUME_FROM` 后直接跑副本（见 §10）。

### 3.3 产物落在哪里（重要）

**输出已定死为 `/home/featurize/work/AIC_train_again/`**，不需要传 `--out-root`。
每个脚本会把自己的**最终绝对路径**打印在 `out dir :` / `out root :` 那一行，
唯一真源就是这行日志——别猜。

- `--out-root` / `--out-dir` 仍然可用，传了就覆盖默认值；
- 在本机 Windows 上跑自检时，默认退回脚本目录下的 `_out_local/`
  （避免在 `C:\` 根目录真建出 `\home\featurize\...` 这种垃圾目录）。

跑完是这样（以 `g0_run_all.py --device cuda:0` 为例）：

```
/home/featurize/work/AIC_train_again/
├── G0_运行报告.md                    ← 先看这个
├── g0_plan.json  g0_results.json
├── g3/
│   ├── g3_efficiency_table.md        ← 可直接粘贴
│   ├── g3_efficiency.json
│   └── g3_latency_table.csv
├── g2/
│   ├── g2_multiseed_summary.json     ← 跨种子均值/标准差
│   ├── g2_multiseed_table.csv
│   ├── log_seed43.txt  log_seed44.txt
│   ├── _variants/
│   │   ├── final_train_seed43.py
│   │   ├── final_train_seed44.py
│   │   └── pcv_v2_folds_seed42.csv   ← 副本旁边必须有它，见 §10 修复记录 3
│   └── dmsa_final_train_seed43_epoch126/
│       ├── final_model_epoch126.pth      ~228 MB
│       ├── resume_checkpoint.pth         ~683 MB
│       ├── train_history.csv             ← 逐 epoch train_loss
│       ├── FINAL_MODEL_READY.json
│       ├── final_training_protocol.json
│       ├── model_info.json  class_to_idx.json  environment.json
│       ├── full_train_snapshot.csv  pcv_v2_folds_used.csv
│       └── final_train_code_used.py
│   └── dmsa_final_train_seed44_epoch126/   （同上）
└── g1/
    ├── g1_ablation_table.md          ← 可直接粘贴
    ├── g1_ablation_summary.json
    ├── g1_ablation_table.csv  g1_ablation_per_fold.csv
    ├── g1_run_config.json
    └── no_sam/fold_1/{metrics.csv, best_metrics.json}
        no_mdc/fold_1/...  no_mdc/fold_5/...
        no_ilf/fold_1/...  no_ilf/fold_5/...
```

> 注意：改写后的副本默认只放在 `g2/_variants/` 下。若加了 `--keep-code`，
> 会额外在 `g2/` 顶层复制一份留档；不加则不复制。

**磁盘占用**：

| 实验 | 占用 | 说明 |
|---|---|---|
| g3 | < 1 MB | 只有 JSON/CSV |
| g2 | **~1.8 GB** | 每个种子 ~912 MB（final model 228 MB + resume checkpoint 683 MB） |
| g1 | < 5 MB | **不保存模型权重**，只有逐 epoch 指标 |

> 小技巧：G2 跑完后 `resume_checkpoint.pth` 就没用了（它是容灾用的），
> 确认 `FINAL_MODEL_READY.json` 生成了就可以删，能省掉 683 MB/种子。
> `final_train.py` 里的 `RESUME_SAVE_EVERY = 5` 控制它的保存频率。

---

## 4. 共享前提：需要 `arch_probe.py`（G1、g4 与 g5 都要）

G1、**g4 的 B 阶段**、**g5 的 ABL 阶段**都依赖 `arch_probe.py` 里的 `make_switch_patch()` 做消融开关注入。
按 §1 的平铺布局，它与脚本**同一目录**即可：

```
arch_probe.py
lib_provenance.py      ← arch_probe 会 import 它（两者必须同目录）
```

查找顺序：**先搜 `g1_ablation.py` 自身所在目录**，再往上找 `DMSA_v24/` 之类的子目录。
仍找不到时才需要 `--arch-probe` 显式指定：

```bash
python g1_ablation.py --arch-probe ~/somewhere/arch_probe.py ...
```

> `g1 --dry-run` 会打印一行 `arch_probe : <实际解析到的路径>`。**跑正式任务前请核对这一行**
> —— 它是「G1 到底用了哪份注入机制」的唯一真源。

**硬门**：`full` 变体的 architecture fingerprint 必须等于
`8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975`。
不等则脚本会拒绝继续——这是防止注入机制悄悄失效的核心保护。

---

## 5. 第二段实验（g4）· 正式运行

> **一句话**：`g4_run.py` 一个文件、一条命令、同一终端，跑完 **40 个 run**，
> 一次性补齐第一段漏掉的三件事 —— **★★★ 对照五折** + **★★ 消融补齐（4 变体 × 5 折）** + **★★ 跨种子 val/acc**。
> 全部 run **同脚本 / 同卡 / 同 `data_root` / 同 seed 口径**，所以表里的 **Δ 可归因**。

### 5.1 它到底回答什么问题

| 阶段 | 内容 | run 数 | 回答的问题 |
|---|---|---:|---|
| **A1** | 分组臂 `full` × 5 折 | 5 | ① 基线协议的五折结果；② **同时充当 B 阶段消融的 `full` 参照**（这是关键） |
| **B** | `no_sam` / `no_mdc` / `no_ilf` / `no_dual_path` × **5 折** | **20** | **每个模块到底贡献多少**（Δ vs A1 的 `full`，同折对齐） |
| **A2** | 图像级随机划分（忽略 `group_id`）`full` × 5 折 | 5 | **P-CV v2 分组协议到底压掉了多少乐观偏差**（Δ = A2 − A1） |
| **A3 / A4** | 分组臂 `full` × seed 43 / 44 × 5 折 | 10 | **换个种子会不会崩**（跨种子 val 准确率极差） |

**与第一段 G1 的关键差别**：G1 的 Δ 是「参考性对照」（参照来自另一个脚本 / 另一张卡 / 另一个 `data_root`）；
本段的参照是 **A1 自己跑出来的 `full`**，**只剩「容量变化」这一个残留混淆** ⇒ 比 G1 强一个量级，**可以撤销 L3**。

### 5.2 上传（相比第一段只多两个文件）

```
g4_run.py
pcv_v2_folds_imgsplit_seed42_FROZEN.csv
```

其余（`gpu_lib.py` / `arch_probe.py` / `smoke_test.py` / `pcv_v2_folds_seed42.csv`）**云端已有，不用重传**。
`g4_run.py` **不 import** `g1/g2/g3/g12/g0`，也**不引用 `final_train.py`**。

上传后**回读确认**（不要相信「我保存过了」）：

```bash
ls -l g4_run.py pcv_v2_folds_imgsplit_seed42_FROZEN.csv
sha256sum pcv_v2_folds_imgsplit_seed42_FROZEN.csv
# 期望： d823021990e317fa997fa660ac36ada1abbb3ed2b29cd01712f06e3707c168b8
grep -n "^PKG_VERSION" gpu_lib.py        # 期望：2026-09-24.4
```

### 5.3 正式步骤（三步，全在同一个终端）

> ⛔ **开跑前先确认卡是空闲的**：`python check_running.py`（期望 `✅ 没有发现任务进程`）。
> `g4` 要独占整卡约 105 h，**不要与第一段的 `g12_run.py` 同时跑** ——
> 输出目录虽不重叠（`g4/` vs `g2/`、`g1/`，**不会互相覆盖**），但会**抢同一张卡**、互相拖慢。
> 另：`g4` 是**独立脚本**，不被 `g12_run.py` / `g0_run_all.py` 调度，**直接前台运行即可**。

```bash
# ① 预检 + 报总时长（不训练，几秒钟）—— 先看这一步的结论再决定跑不跑
python g4_run.py --plan

# ② CPU 接线自检（不做训练；验证 5 个变体的参数量与指纹）
python g4_run.py --dry-run --device cpu

# ③ 正式开跑（前台运行，进度就刷在**这个终端**里，不需要另开终端）
python g4_run.py
```

**①`--plan` 期望输出**（8 项预检，全 `OK` 才允许开跑）：

```
任务        : 40 个 run
   A1  阶段1 · 基线臂（分组协议 · full）                    5 run
   B   阶段2 · 消融矩阵（4 变体 × 5 折）                   20 run
   A2  阶段3 · 对照臂（图像级随机 · full）                   5 run
   A3  阶段4a · 跨种子（分组 · full · seed 43）           5 run
   A4  阶段4b · 跨种子（分组 · full · seed 44）           5 run
  ✅ [OK  ] 基线 manifest    pcv_v2_folds_seed42.csv SHA 与冻结常量一致
  ✅ [OK  ] 对照 manifest    pcv_v2_folds_imgsplit_seed42_FROZEN.csv SHA 与冻结常量一致
  ✅ [OK  ] 基线清单守卫         10,240 行 / 每折 8192+2048 / 同源跨折=0 / 每类每折 512 val
  ✅ [OK  ] 对照清单守卫         结构不变量 PASS；group 跨折=262、sha 跨折=140、被同源污染的 val=774
  ✅ [OK  ] 变体名守卫          5 个变体均有文档参数量可校验
  ✅ [OK  ] 数据目录           .../Combined_Dataset/train 四类目录齐全
  ✅ [OK  ] 磁盘空间           可用 ≥ 8 GB
  ✅ [OK  ] GPU 显存           <卡名>：空闲 / 总（单 run 训练峰值约 5.1 GB）
  结论：预检通过 ｜ 40 个 run × 2.63 h ≈ 105.0 h（按 G1 实测单 run 外推，非估算）
```

任何一项 `FAIL` → **exit 2，不做任何训练**（不会带着错误往前跑）。

**②`--dry-run` 期望结尾**：`✅ dry-run 通过：5/5 个变体参数与指纹均符合文档值。`

> 🔁 **不想一直挂着终端？** 改成后台跑（本机关机 / 断网 / SSH 断开都不影响）—— 见 **§5.9**，
> 一条 `nohup python -u g4_run.py --quiet > g4_console.log 2>&1 &` 即可。
> 105 小时 ≈ 4.4 天，**强烈建议用 §5.9 的后台方式**。

### 5.4 实时进度怎么看（**不需要另开终端**）

前台运行 `python g4_run.py`，屏幕上会有**三层**信息，全部逐行 `flush`，即时可见：

| 层 | 长相 | 含义 |
|---|---|---|
| ① 基线 tqdm 进度条（原样转发） | `train epoch[37/200]:  45%\|████▌  \| 230/512 [01:23<01:45, 3.34it/s]` | 逐 batch 进度；`g4` **只解析不改写**基线 tqdm，训练逻辑 100% 不变 |
| ② epoch 汇总行 | `ep  40/200 train_loss=… val_loss=… acc=… best=…` | 每 10 个 epoch（及最后一个）各一行 |
| ③ 全局进度行 | `[进度]  G4  epoch 40/200  本轮 step 128/512  step 224896/4096000  5.5%  用时 05:47:24  剩余≈99:15:00  loss=… val_acc=… run 3/40` | **跨全部 40 个 run** 的百分比与 ETA（不是每个 run 从 0% 重新数） |

> 全局分母口径：**40 run × 200 epoch × 512 step = 4,096,000 step**（val 前向不计入）。
> 每个 run 开始时脚本会打印一行 `本 run 口径   : 200 epoch × 512 step = 102,400 step
> （val 前向不计入）；全局总 step ≈ 4,096,000`，可直接对账。

同时会把同一份内容 **tee 到 `g4_progress.log`**（断线后想回看用）。若你想另开窗口 tail 也可以，但**不是必需的**：

```bash
tail -f /home/featurize/work/AIC_train_again/g4/g4_progress.log
```

### 5.5 产物落在哪里

输出目录 = **`/home/featurize/work/AIC_train_again/g4/`**（已定死，无需传参；脚本启动会打印 `out dir :` 那一行，以那行为唯一真源）。

```
/home/featurize/work/AIC_train_again/g4/
├── g4_precheck.json            ← 预检明细（8 项 + 结论 + 估计时长）
├── g4_progress.log             ← 全程进度 + 关键行（tee）
├── g4_per_run.csv              ← **逐 run 一行**（40 行）：params/fingerprint/best_epoch/acc/…/skipped
├── g4_summary.json             ← 4 张表的机器可读版（protocol_arms / protocol_delta / ablation / seed_dispersion）
├── g4_report.md                ← **可直接粘贴进方案的 Markdown 报告**（含 5 节诚实说明）
└── runs/
    ├── grouped__full__seed42__fold1/{metrics.csv, best_metrics.json}
    ├── grouped__no_sam__seed42__fold1/…      （4 变体 × 5 折 = 20 个）
    ├── control__full__seed42__fold1/…        （5 个）
    ├── grouped__full__seed43__fold1/…        （5 个）
    └── grouped__full__seed44__fold1/…        （5 个）
```

- `metrics.csv` = **逐 epoch** 的 `train_loss/val_loss/accuracy/precision/recall/f1/kappa`。
- `best_metrics.json` = 折内最优（含 `params` / `fingerprint` / `params_matches_doc` / `elapsed_seconds`）。
- **磁盘占用 < 10 MB**（g4 **不保存模型权重**，只落 CSV/JSON）；**数据集目录只读，不写入**。

### 5.6 断线 / 中断怎么办：`--resume`

```bash
python g4_run.py --resume
```

已有 `best_metrics.json` 的 run 会**自动跳过**（打印 `[跳过 i/40] <label>`），进度基数也跟着走，不会倒退。
⚠️ **`--resume` 是「跳过已完成的 run」，不是「接着上次的 epoch」**——被打断的那个 run 会**从头重跑**。

### 5.7 跑完怎么验收

```bash
G4=/home/featurize/work/AIC_train_again/g4

ls -l "$G4"                    # 期望：5 个文件 + runs/ 目录
ls "$G4/runs" | wc -l          # 期望：40
wc -l "$G4/g4_per_run.csv"     # 期望：41（表头 + 40 run）

python - <<'PY'
import json, pathlib
g4 = pathlib.Path("/home/featurize/work/AIC_train_again/g4")
d = json.load(open(g4 / "g4_summary.json", encoding="utf-8"))
print("n_runs                =", d["n_runs"])                       # 40
print("grouped  n_folds      =", d["protocol_arms"]["grouped"]["n_folds"])   # 5
print("control  n_folds      =", d["protocol_arms"]["control"]["n_folds"])   # 5
print("protocol Δ (pp)       =", round(d["protocol_delta"]["mean_pp"], 3))
print("ablation variants     =", [a["variant"] for a in d["ablation"]])      # 4 个
print("ablation n_folds      =", [a["n_folds"] for a in d["ablation"]])      # 全 5
print("seed 最大极差 (pp)     =", round(d["seed_dispersion"]["max_range_pp"], 3))
print("sealed_test_untouched =", d["sealed_test_untouched"])                  # True
PY
```

判定标准（**数个数，不是查函数在不在**）：

| 判据 | 期望 |
|---|---|
| `n_runs` / `runs/` 子目录数 | **40 / 40** |
| 每个 run 的 `params_matches_doc` | 全 `True`（`g4_per_run.csv` 里该列全为 True） |
| `runs/grouped__full__*` 的 `fingerprint` | 全 = `8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975` |
| `protocol_arms` | `grouped` / `control` 各 **n_folds = 5** |
| `ablation` | **4** 个变体，各 **n_folds = 5**，`delta_vs_full_pp` 覆盖 1–5 折 |
| `seed_dispersion` | 5 折 × 3 个种子（42/43/44） |
| `sealed_test_untouched` | `True` |

> 任一判据不符 ⇒ **不要往下写方案**，先把 `g4_precheck.json` / `g4_per_run.csv` 贴出来排查。

### 5.8 本段的诚实限制（**必须随数字一起写**，不要藏）

1. **仍然没有 CI**：对照 5 折、消融 5 折、跨种子 3 个 —— 按项目铁律，报 `spread` 与极差，**不得声称显著性**。
2. **容量不匹配**：`no_mdc` −1,177 万、`no_dual_path` −1,006 万、`no_ilf` −132 万参数。
   **只有 `no_sam`（−459）是容量匹配的**，只有它的 Δ 能干净归因给 SAM 本身；
   其余只能表述为「移除 X 后的整体效果（含容量变化）」。
3. **`no_ilf` 在 G1 里是「训练不收敛」**（best_epoch=2、acc≈0.25）。本段若复现同样现象，
   **只能写「移除 ILF 后训练不收敛」**，不得主张「ILF 的门控/融合机制不可或缺」。
4. **不碰 sealed test**：全程只评折内 val；`g4_run.py` 代码里**根本不引用** `pcv_test_manifest_SEALED.csv`。
5. **两条边界**：
   - **Swish 无法消融** —— `make_switch_patch()` 只有 4 个开关（无 Swish），所以消融是 **4 个机制**而不是 5 个。
   - **`--folds` 会作用于所有阶段** —— 想退回消融 2 折加 `--folds 1,5`，但那会把 A* 也压到只跑 fold 1；
     **正式跑请不要加 `--folds`。**

### 5.9 后台跑（脱离终端）：本机关机 / 断网 / SSH 断了都不影响

**结论：可以，用 `nohup`。** 第一段的 G2+G1 就是这么跑的，有铁证：

- `cloud_results/20260925_G1/AIC_train_again/g12_run.log` **第 1 行**就是 `nohup: ignoring input`；
- 当时进程自证 `PPID=1`（被 init 收养）、`TTY=?`（无控制终端）。
  ⚠️ 注意这两项是**「终端已经关掉了」的结果**，不是「可以放心关掉」的判据 —— 判据见下文 `SigIgn`。

**命令**（`--quiet` 必加，原因见下方「日志大小」）：

```bash
cd /tmp/pycharm_project_28d63a37          # 你放脚本的目录（PyCharm SFTP 的默认落地处）
nohup python -u g4_run.py --quiet > g4_console.log 2>&1 &
echo $!                                    # ← 记下这个 PID，下面要用
```

**立刻自证「真的脱离了终端」** —— **唯一判据 = `SigIgn` 最低位**，直接用下面的一键命令。

> 🔴 **不要拿 `PPID` / `TTY` 当判据**（本节早期版本这么写过，**已更正** —— 2026-09-26 被邱总实测打脸）：
>
> - `nohup` **只做一件事**：把子进程的 SIGHUP 设为 `SIG_IGN`。
>   它**不 fork、不换控制终端、不脱离会话**。
> - 所以**刚启动时 `TTY` 仍然是 `pts/N`、`PPID` 仍然是你那个 shell** —— 这是**完全正常的**。
> - 只有等你**关掉那个终端之后**，它才会变成 `PPID=1`、`TTY=?`（被 init 收养 + tty 已销毁）。
>   ⇒ **`PPID=1`/`TTY=?` 是"终端已消失"的*结果*，不是"是否安全"的*判据*。**
> - 反过来，`TTY=pts/2` 也**不代表有风险**：本命令的 stdout/stderr 已重定向到文件、
>   stdin 被 nohup 接到 `/dev/null`，进程**根本不往终端写** ⇒ tty 消失后**不会**撞 `EIO`。

> ⚠️ 认进程：`pgrep -af 'g4_run'` 会把 DataLoader 的 6 个 fork worker **一起列出来**
> （它们不 exec，`/proc/<pid>/cmdline` 是父进程的副本，命令行完全相同）。
> **主进程 = PID 最小 / 启动最早的那个**；worker 的 `PPID` = 主进程 PID。
> 别看到 7 行就以为跑了 7 个任务。

**看进度**（`g4_progress.log` 是脚本自己维护、每行 append+flush 的，永远是最新的）：

```bash
tail -f /home/featurize/work/AIC_train_again/g4/g4_progress.log
```

**到底什么会中断、什么不会**（这一栏必须说实话）：

| 情况 | 会中断吗 | 依据 |
|---|:---:|---|
| 你本机关机 / 合盖休眠 | ❌ **不会** | 进程在云端，与你本机无关 |
| 你本机断网 / 拔网线 | ❌ **不会** | 同上；训练本身零网络依赖 |
| SSH / PyCharm 终端断开 | ❌ **不会** | **`SigIgn` 最低位 = 1** ⇒ shell 发的 SIGHUP 被内核丢弃（`nohup` 的唯一作用） |
| **Featurize 实例被关机 / 回收 / 欠费** | ✅ **会** | **这才是真正的风险**——进程就活在那个实例里 |
| 人为误杀 / OOM killer | ✅ **会** | 用 `--resume` 续跑 |

> ⚠️ 前三条**有实测原件**（第一段 G2+G1：`g12_run.log` 第 1 行 `nohup: ignoring input`，
> 且运行时 `PPID=1`）；「实例被回收后进程是否还活」**未实测 `NOT_VERIFIED`**，
> 不要当保证——**别把实例关掉**。
> ⚠️ 另外：`nohup` 保护的是 **SIGHUP**，它**挡不住** `kill -9` / OOM killer，也别指望它救人。

**日志大小（这是 `--quiet` 存在的原因）**：

- 不加 `--quiet` 时，基线 tqdm 的**每一帧**都会转发到 stdout。全量 40 run 的帧数
  = 40 × 200 × 512（train）+ 40 × 200 × 128（val）≈ **512 万帧**，
  重定向到文件后 ≈ **0.38 GB 的单行**（tqdm 用 `\r` 分隔、不换行，grep 会明显变慢）。
- `--quiet` 只关掉 tqdm 转发，**进度信息一条不少**。同口径实测（都是 40 run × 1 epoch 的微缩版）：

  | | train tqdm 帧 | val tqdm 帧 | `[进度]` 行 | `ep … best=` 行 |
  |---|---:|---:|---:|---:|
  | 默认 | 160 | 120 | 90 | 40 |
  | `--quiet` | **0** | **0** | 81 | 40 |

  （`[进度]` 90→81 是 5 秒节流的时序抖动，不是信息丢失；有 `--quiet` 后全量日志 ≈ **10 MB** 量级。）
- **前台跑（§5.3 ③）不要加 `--quiet`** —— 那时你要的就是那个逐 batch 进度条。

**中断后接着跑**（`--resume` 跳过已完成的 run）：

```bash
nohup python -u g4_run.py --quiet --resume > g4_console2.log 2>&1 &
```

**⚠️ 已经用前台方式跑起来了、才发现没脱离终端？** —— **建议直接重启成 `nohup`**
（本实验只跑了十几分钟，代价可忽略）。**不要**用「`Ctrl-Z` → `bg` → `disown -h`」这条捷径，原因见下。

**决定性判据：SIGHUP 是否被*进程自己*忽略**（`nohup` / `setsid` 靠的就是这个）：

```bash
grep -E '^SigIgn' /proc/<主进程PID>/status
```

`SigIgn` 的**最低位（bit 0）= SIGHUP**。对照表：

| `SigIgn` 输出 | 含义 |
|---|---|
| 形如 `...0001`（**最低位 = 1**） | ✅ 进程忽略了 SIGHUP（`nohup` 生效）⇒ 会话断掉也不会死 |
| 形如 `...0000`（**最低位 = 0**） | ⚠️ 没有这层保护；若 `TTY` 还不是 `?`，就是前台跑 |

> 🔴 **别被其它位骗了**：Python 启动时会**自己**忽略 `SIGPIPE`(bit 12) 与 `SIGXFSZ`(bit 24)，
> 所以 `SigIgn: 0000000001001000` 这种「有一位是 1」的掩码**并不代表走过 nohup**。
> **只有最低位（SIGHUP）才是判据**；其余位是解释器默认值。

**一键自证（推荐：自动取 PID、自动翻译，直接给 ✅/❌；zsh/bash 均可）**：

```bash
P=$(pgrep -f 'g4_run\.py' | sort -n | head -1)
if [ -z "$P" ]; then echo "❌ 没找到 g4 进程（已退出？）"; else
SI=$(grep '^SigIgn' /proc/$P/status | awk '{print $2}'); SI=${SI:-0}
TT=$(ps -o tty= -p $P | tr -d ' '); PP=$(ps -o ppid= -p $P | tr -d ' ')
echo "主进程 PID=$P  SigIgn=$SI  TTY=${TT:-?}  PPID=$PP"
if [ $(( 0x$SI & 1 )) -eq 1 ]; then
  echo "✅ 已脱离终端（SIGHUP 被忽略）—— 关机 / 断网 / SSH 断开都不影响"
else
  echo "❌ 尚未脱离（SIGHUP 未忽略）—— 按上文重启成 nohup"
fi
fi
```

> **判据只有 `0x$SI & 1` 这一项**；`TTY` / `PPID` 只打印出来供参考，
> **不参与判断**（原因见上面的红字）。**不要**再给它们加 `&&`。
> `sort -n | head -1` 取的是 **PID 最小 = 启动最早 = 主进程**（worker 由它 fork，PID 更大）。
> 已在本机验证位运算：`0x0000000001001000 & 1 = 0`（无保护）、`0x0000000001001001 & 1 = 1`（已保护）。

> ✅ **实测验收（2026-09-26 13:20，g4 正式跑第一轮）**：
> `主进程 PID=13498  SigIgn=0000000001001001  TTY=pts/2  PPID=11610` → 命令输出 `✅`。
> ⚠️ 注意此时 **`TTY` 仍是 `pts/2`、`PPID` 仍是启动它的那个 shell(11610)** ——
> **终端还开着时"正确的样子"就是长这样**，不要因此以为没脱离。
> 等你把终端关掉、下次重连再看，同一条命令应当变成 `TTY=?  PPID=1`，判据仍是 `✅`。

**终极自证（可选，端到端）：直接给进程发一个 SIGHUP，看它死不死**

```bash
P=$(pgrep -f 'g4_run\.py' | sort -n | head -1)
kill -HUP $P; sleep 5
echo "剩余进程数 = $(pgrep -f 'g4_run\.py' | wc -l)  （仍为 7 就是扛住了）"
```

> 🔴 **只有在上面打印 `✅` 之后才能跑这一条。**
> 因为 `SigIgn` 最低位 = 1 时，`kill -HUP` **在内核层就是 no-op**（信号被丢弃、不会转交），
> 所以这条测试**在当前状态下是安全的**，而且它是**因果成立**的证据 —— 你亲手复现了
> "终端消失时 shell 会对它做的事"。

**实时看进度**（`g4_progress.log` 每行带时间戳，脚本每 ~5 s append+flush，永远最新）：

```bash
tail -f /home/featurize/work/AIC_train_again/g4/g4_progress.log
```

- `Ctrl-C` 只停 `tail`，**不影响训练**。
- 想只看每轮汇总：`tail -f …/g4_progress.log | grep --line-buffered -E '进度|ep '`
- 想看脚本完整 stdout（含 `[i/40]` run 横幅）：`tail -f /tmp/pycharm_project_28d63a37/g4_console.log`
- 想数"已完成几个 run"（一次性，不阻塞）：`ls /home/featurize/work/AIC_train_again/g4/runs/*/best_metrics.json 2>/dev/null | wc -l`

**为什么不推荐「`Ctrl-Z` → `bg` → `disown -h`」这条捷径**（本节早期版本写了它，**是错的，已更正**）：

1. `disown -h` **不会**改变子进程的 `SigIgn` —— 它只是让 **shell 退出时不发 SIGHUP**。
   所以上面那个 `SigIgn` 判据**根本验证不了** `disown`。
2. 更关键：`bg` 之后进程的 **stdout 仍然指着那个终端**。SSH 一断、终端消失，
   进程再往它写就会拿到 `EIO`。而本程序**每几秒就往 stdout 写进度行、并转发 tqdm**，
   `gpu_lib.make_tee_sink` 里的 `print(line)` **没有 try/except 兜底**
   ⇒ **可能直接把进程写崩**（那就白跑了）。
3. 所以：**前台跑的就老老实实重启成 `nohup`** —— stdout 落到文件，
   顺带把上面第 2 条一起解决了。

**怎么在一堆 PID 里认出主进程**：`pgrep -af 'g4_run\.py'` 会列出 **1 个主进程 + 6 个 DataLoader worker**
（共 7 行，**不是一个任务变成了 7 个**）。主进程 = **PID 最小 / 启动最早**的那个；
worker 的 `PPID` 等于主进程 PID。

---

## 6. 第三段实验（g5）· 最省补完

> **一句话**：`g5_run.py` 一个文件、一条命令、**6 个 run（≈15.75 h ≈ ¥38）**，
> 把第一段（G1/G2）留下的两个小洞补到「**可用**」——
> **★★ 消融补最后 1 个变体（`no_dual_path` × 折 {1,5}）** + **★★ 跨种子 val/acc（seed 43/44 × 折 {1,5}）**。
> ⛔ **这是省钱档，不是「把 g4 做完」**：只跑 **2 折**，只补 g4 里最便宜的两块，
> **不覆盖 ★★★ 对照五折，也不把消融补到 5 折** —— 那两块留给 **g4（§5，≈105 h）**。

### 6.1 为什么是「这两个」而不是别的

| 目标 | 为什么 g5 这样补 | 依据 |
|---|---|---|
| 消融补 `no_dual_path` | `g1_ablation.py` 的 `DEFAULT_VARIANTS = ["no_sam","no_mdc","no_ilf"]` **本来就不含它**（而 `VARIANT_SWITCHES` 支持）⇒ 补齐只需显式指定，**不用改代码** | `g1_ablation.py` 源码 |
| 多种子拿 val/acc | `g2_multi_seed.py` 的 `--mode cv` 被**脚本主动封禁**（源码原话：「本脚本不代跑 cv 模式以免误用」），`--mode final` 是 126 epoch 全量训练、**没有 val/acc** ⇒ 拿不到「跨种子准确率」，必须走**折内 val** 路线 | `g2_multi_seed.py:450` |
| 为什么只用折 {1,5} | 为了与**已有的 seed42 折 {1,5} `full` 参照（99.170%）同折对齐**；折不同则 Δ 不可比 | 见 §6.5 |
| 为什么**不写 `g1/`** | `g1_ablation.py` 的 `summarize()` 只用「本次 results」重建表并**覆盖** `g1_ablation_summary.json` ⇒ 再跑一次会把原有 3 变体表冲掉。g5 把结果写到 **`g5/abl/runs/`**，另**只读** g1 来**合并** | g5 刻意设计 |

### 6.2 上传（相比第二段只多 1 个文件）

```
g5_run.py
```

其余（`g4_run.py` / `gpu_lib.py` / `arch_probe.py` / `lib_provenance.py` / `smoke_test.py` / `pcv_v2_folds_seed42.csv`）**云端已有，不用重传**。
`g5_run.py` **import 复用** `g4_run.train_one_run()` / `g4_run.variant_probe()` / `g4_run.load_arch_probe()`，
并**只读** g1 已有产物（`AIC_train_again/g1/g1_ablation_summary.json`）。

上传后**回读确认**（不要相信「我保存过了」）：

```bash
ls -l g5_run.py
head -1 g5_run.py                  # 期望含：g5_run.py · 第三段实验（最省补完）
grep -n "^PKG_VERSION" gpu_lib.py  # 期望：2026-09-24.4
```

### 6.3 正式步骤（三步，全在同一个终端）

> ⛔ **开跑前先确认卡是空闲的**：`python check_running.py`（期望 `✅ 没有发现任务进程`）。
> `g5` 与 `g4` / 第一段 `g12_run.py` **输出目录互不重叠**（`g5/` vs `g4/`、`g1/`、`g2/`，**不会互相覆盖**），
> 但会**抢同一张卡**、互相拖慢 ⇒ **不要同时跑**。`g5` 是**独立脚本**，不被 `g12_run.py` / `g0_run_all.py` 调度。

```bash
# ① 预检 + 报总时长/费用（不训练，几秒钟）—— 先看这一步的结论再决定跑不跑
python g5_run.py --plan

# ② CPU 接线自检（不做训练；验证 no_dual_path 构参 + full 指纹硬门）
python g5_run.py --dry-run --device cpu

# ③ 正式开跑（前台运行，进度刷在**这个终端**里）
python g5_run.py
```

**①`--plan` 期望输出**（6 run / ≈15.75 h / ≈¥38）：

```
任务         : 6 run（阶段 ABL,SEED；折 [1, 5]）
   - [ABL] grouped__no_dual_path__seed42__fold1  ← g1 当年只跑了 no_sam/no_mdc/no_ilf…
   - [ABL] grouped__no_dual_path__seed42__fold5  ← 同上
   - [SEED] grouped__full__seed43__fold1  ← g2 只有 126ep final 的 train loss，无 val/acc…
   …（共 6 行）
预计用时     : 15.75 h（单 run 2.625 h，按 G1 六个 run 实测均值）
预计费用     : ≈ ¥38（单价 ¥2.4/h 为**反推值**，不是平台报价）
full 参照    : 找到 2 折（g1_ablation_summary 的 full 参照行）   ← 若显示 ❌ 未找到，则消融表没有 Δ 列
g1 已有产物  : 在 .../AIC_train_again/g1                        ← 若显示 ❌ 不在，则合并表只有 1 个变体
```

**②`--dry-run` 期望结尾**：`✅ dry-run 通过：\`full\` 与基线同构，待跑变体均可构造并前向。`
期望参数量（对不上会**直接报错**，不会静默继续）：

| 变体 | params | Δ vs full |
|---|---:|---:|
| `full` | 56,812,813 | +0（= 基线指纹 `8e542321…d76975` ✅ 硬门） |
| `no_dual_path` | 46,753,037 | −10,059,776 |

> ℹ️ `--dry-run` 打印的 `full` / `no_dual_path` 两行里那个 **`Δ` 列 = 「实测 − 文档值」**
> （`variant_probe` 的正确性自检，显示 `+0` 即「与文档完全一致」），**不是**「vs full」。
> 上表的「Δ vs full」是本手册按语义**另算**的（56,812,813 − 46,753,037 = 10,059,776）。

> ⚠️ 本机（Windows）自检请加 `--num-workers 0`，原因与 g4 相同（见 §10.1）。云端 Linux 用 fork，不受影响（保持默认 6）。

> 🔁 **不想一直挂着终端？** 改成后台跑（本机关机 / 断网 / SSH 断开都不影响）—— 见 **§6.9**。

### 6.4 实时进度怎么看（**不需要另开终端**）

前台运行 `python g5_run.py`，屏幕上有**三层**信息，全部逐行 `flush`，即时可见：

| 层 | 长相 | 含义 |
|---|---|---|
| ① 基线 tqdm 进度条（原样转发） | `train epoch[37/200]:  45%\|████▌  \| 230/512 [01:23<01:45, 3.34it/s]` | 逐 batch 进度；g5 **只解析不改写**基线 tqdm |
| ② epoch 汇总行 | `ep  40/200 train_loss=… val_loss=… acc=… best=…` | 每 10 个 epoch（及最后一个）各一行 |
| ③ 全局进度行 | `[进度]  G5  epoch 40/200  本轮 step 128/512  step 224896/614400  36.6%  用时 … 剩余≈… run 3/6` | **跨全部 6 个 run** 的百分比与 ETA（不是每个 run 从 0% 重新数） |

> 全局分母口径：**6 run × 200 epoch × 512 step = 614,400 step**（val 前向不计入）。
> 每个 run 开始时脚本会打印一行 `本 run 口径   : 200 epoch × 512 step = 102,400 step（val 前向不计入）`，可直接对账。

同一份内容 **tee 到 `g5_progress.log`**：

```bash
tail -f /home/featurize/work/AIC_train_again/g5/g5_progress.log
```

### 6.5 产物落在哪里 / 跑完怎么验收

输出目录 = **`/home/featurize/work/AIC_train_again/g5/`**（脚本启动会打印 `out dir :` 那一行，以那行为唯一真源）。

```
/home/featurize/work/AIC_train_again/g5/
├── g5_precheck.json          ← 预检明细（mode/时长/费用/full 参照/g1 是否在位）
├── g5_progress.log           ← 全程进度 + 关键行（tee）
├── g5_per_run.csv            ← **逐 run 一行**（6 行）：params/fingerprint/best_epoch/acc/…
├── g5_summary.json           ← 机器可读汇总（含 ablation_merged / seed_dispersion）
├── g5_ablation_merged.json   ← **4 变体合并表**（3 个来自 g1 只读 + 1 个本脚本产出）
├── g5_report.md              ← **可直接粘贴进方案**的 Markdown（含 5 节 + 诚实限制）
├── abl/runs/grouped__no_dual_path__seed42__fold{1,5}/   （2 个）
└── seed/runs/grouped__full__seed{43,44}__fold{1,5}/     （4 个）
```

> **磁盘占用 < 5 MB**（g5 与 g4 一样**不保存模型权重**，只落 CSV/JSON）；**数据集目录只读，不写入**。

验收（**数个数，不是查函数在不在**）：

```bash
G5=/home/featurize/work/AIC_train_again/g5
ls -l "$G5"                        # 期望：6 个文件 + abl/ seed/ 两个目录
wc -l "$G5/g5_per_run.csv"         # 期望：7（表头 + 6 run）
ls "$G5/abl/runs" | wc -l          # 期望：2
ls "$G5/seed/runs" | wc -l         # 期望：4
```

> 🟢 **本机一条命令验收**（把 `g5/` 下载到本机后）：`python verify_g5.py <g5_dir>`
> —— 12 条可执行判据 + 退出码（**0=全 PASS / 1=有 FAIL / 2=找不到产物**），并打印消融合并表与多种子表。
> 默认自动找 `cloud_results/20260928_g5` → `_out_local/g5`；不传参也行。

| 判据 | 期望 |
|---|---|
| 新跑 run 数 | **6**（`g5_per_run.csv` = 7 行） |
| `no_dual_path` 参数量 | **46,753,037**（该行 `params_matches_doc=True`） |
| `full`（seed43/44）指纹 | 全 = `8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975` |
| `seed` 表 | **3 个种子**（42 参照 + 43/44 新跑）× 折 {1,5} |
| 合并消融表 | **4 个变体**（`no_sam`/`no_mdc`/`no_ilf` 来自 g1 + `no_dual_path` 本脚本） |
| `full` 参照 | 折 1,5 = **99.170%**（来自 `g1_ablation_summary.json` / 复现版五折） |

> ⚠️ 若合并表**缺 `no_mdc` 等变体**：说明云端 `g1/` 里没有对应的 `metrics.csv` —— g5 **只读** g1、
> **不会**替它重跑；缺谁就在 `g5_ablation_merged.json::missing_variants` 里列出谁。
> ⚠️ 若消融表**没有 Δ 列**：说明 `full` 参照没找到（`g1_ablation_summary.json` 的 full 行与
> `复现版实验数据/Fold{1,5}` 都缺席）；传一份上来或用 `--reference-cv` 指定即可。

### 6.6 断线 / 中断怎么办：`--resume`

```bash
python g5_run.py --resume
```

已有 `best_metrics.json` 的 run **自动跳过**（打印 `--resume 保护 N 个已完成 run，未删除`），进度基数也跟着走。
✅ **`--resume` 会保护已完成的 run 不被覆盖清理删掉**（该顺序 bug 已在本机自检中抓到并修复）。
⚠️ `--resume` 是「跳过已完成的 run」，不是「接着上次的 epoch」——被打断的那个 run 会**从头重跑**。

### 6.7 本段的诚实限制（**必须随数字一起写，不要藏**）

1. **折数只有 2**（折 1、5）⇒ n=2 **没有置信区间**，**不得**写「显著」；只能在 |Δ| > 折间跨度时谈「方向」。
2. **容量不匹配仍在**：`no_dual_path` −10,059,776、`no_mdc` −11,775,012 参数 ⇒ 精度差**不能归因给单模块**，
   只能表述为「移除 X 后的整体效果（含容量变化）」；`no_sam`（−459）是唯一可干净归因的消融。
   `no_ilf` 若复现 G1 的不收敛现象，**只能写「移除 ILF 后训练不收敛」**。
3. **多种子只有 2 个新种子**（43/44；真源建议 2–3 个）⇒ 极差/标准差统计效力弱。
4. **不碰 sealed test**：全程只评折内 val；`g5_run.py` 代码里**根本不引用** `pcv_test_manifest_SEALED.csv`。
5. **本段不改变 L1 / L2 / L5** 三条客观限制（与 g4 一致）。

> ⚠️ **本段与第二段（g4）不是替代关系**：g5 = 省钱档（2 折、只补两块）；
> 要「Δ 可归因 + 5 折 + 对照协议定量 + 撤销 L3/L6」**仍必须跑 g4（§5，≈105 h）**。

### 6.8 结果怎么用（对应方案修改清单）

| 产出 | 写进方案哪里 |
|---|---|
| `g5_ablation_merged.json`（补全的 4 变体表） | §三(二) 消融小节：把 G1 **缺 `no_dual_path`** 的洞补上（**仍为 2 折参考性对照，不可归因**） |
| `g5_report.md` §3（多种子表） | §三(二)1 五折结果：给出 **seed 42/43/44 的折内 val 极差**（**2 折口径，n=2 无 CI**） |

> ⚠️ 插表会改动 63 页方案的**分页**，重导 PDF 后须复核附录图表页码索引。
> ⚠️ `g5_report.md` 只是**素材**；真正的提分发生在写进正文并重导 PDF 那一步（**0 GPU 成本，但必须做**）。

### 6.9 后台跑（脱离终端）：本机关机 / 断网 / SSH 断了都不影响

**命令**（`--quiet` 必加，理由同 §5.9 的「日志大小」）：

```bash
cd /tmp/pycharm_project_28d63a37          # 你放脚本的目录（PyCharm SFTP 的默认落地处）
nohup python -u g5_run.py --quiet > g5_console.log 2>&1 &
echo $!                                    # ← 记下这个 PID
```

**唯一判据 = `SigIgn` 最低位（bit 0 = SIGHUP）**，与 g4 完全一致。
🔴 **不要拿 `PPID` / `TTY` 当判据**（原理与红字说明见 **§5.9**，此处不重复）。一键自证：

```bash
P=$(pgrep -f 'g5_run\.py' | sort -n | head -1)
if [ -z "$P" ]; then echo "❌ 没找到 g5 进程（已退出？）"; else
SI=$(grep '^SigIgn' /proc/$P/status | awk '{print $2}'); SI=${SI:-0}
TT=$(ps -o tty= -p $P | tr -d ' '); PP=$(ps -o ppid= -p $P | tr -d ' ')
echo "主进程 PID=$P  SigIgn=$SI  TTY=${TT:-?}  PPID=$PP"
if [ $(( 0x$SI & 1 )) -eq 1 ]; then
  echo "✅ 已脱离终端（SIGHUP 被忽略）—— 关机 / 断网 / SSH 断开都不影响"
else
  echo "❌ 尚未脱离（SIGHUP 未忽略）—— 按 §5.9 重启成 nohup"
fi
fi
```

> ⚠️ 认进程：`pgrep -af 'g5_run'` 会把 DataLoader 的 6 个 fork worker **一起列出**，
> **主进程 = PID 最小 / 启动最早**的那个（worker 的 `PPID` = 主进程 PID）。

**看进度**（`g5_progress.log` 是脚本自己维护、每行 append+flush 的，永远最新）：

```bash
tail -f /home/featurize/work/AIC_train_again/g5/g5_progress.log
```

**到底什么会中断、什么不会**：与 §5.9 那张表**完全相同** ——
本机关机 / 断网 / SSH 断开 **都不会**；**只有 Featurize 实例被关机 / 回收 / 欠费才会**（进程就活在那个实例里）。

**中断后接着跑**：

```bash
nohup python -u g5_run.py --quiet --resume > g5_console2.log 2>&1 &
```

---

## 7. 三条铁律（脚本已强制，请勿绕过）

1. **sealed test 全程不碰**
   三个实验都只用 manifest 里 `val_fold` 指定的**折内验证集**。
   脚本文档里已写明：若方案要「多种子的 test 指标」，
   **必须另找一批从未用过的测试数据**，不能在原 sealed test 上重复挑选。

2. **train manifest SHA 必须对得上**
   `51f5b251f9e7cce6c557798ae8fa8e263a6baf58c125524be4e1a1df4bd9edfd`
   对不上直接报错退出。

3. **不需要手算任何能测的量**
   参数量、FLOPs、ResNeXt 对照全部实测（曾因手算 ResNeXt50 参数量差了一千万）。

---

## 8. 结果怎么用（对应方案修改清单）

| 实验 | 产出的表 | 写进方案哪里 |
|---|---|---|
| G3 | `g3_efficiency_table.md` | §三(二) 新增效率小节；**撤回 L4「未实测效率指标」** |
| G2（第一段） | `g2_multiseed_table.csv` | 只有 `train_loss`、**无 val/acc** ⇒ **不足以撤 L6**，仅作辅助 |
| G1（第一段） | `g1_ablation_table.md` | Δ 是「参考性对照」（参照 3 处未对齐）⇒ **不可归因**，仅作辅助 |
| **g4 · A1+A2** | `g4_report.md` §1 | §三(二) **新增「防泄漏协议的对照实验」小节**；把协议价值从定性变定量 |
| **g4 · B** | `g4_report.md` §2 | §三(二) 消融小节（**替换 G1 的参考性对照**）；§二(二)2 创新点补因果证据；**撤回 L3** |
| **g4 · A3+A4** | `g4_report.md` §3 | §三(二)1 五折结果：用**跨种子极差**替换「折间标准差」口径；**撤回 L6** |
| **g4 · 全** | `g4_report.md` §4/§5 | 附「对照清单泄漏描述」+ 5 条诚实说明 |

> ⚠️ 插表会改变 57 页方案的**分页**，请重导 PDF 并复核附录图表页码索引。
> ⚠️ `g4_report.md` 只是**素材**。真正的「提分」发生在把它写进方案正文并重导 PDF 那一步 ——
> 那一步是 **0 GPU 成本，但必须做**（否则实验做了也不落地）。

---

## 9. 已知的诚实限制（不要藏）

**第一段（G1 / G2）：**

- **G1 容量不匹配**：`no_mdc` 少 1,177 万参数、`no_dual_path` 少 1,006 万。
  精度差**不能直接归因给单个模块**，只能写「移除 X 后的整体效果（含容量变化）」。
  严格拆出模块语义贡献需要**容量匹配对照**（把省下的参数用等量普通卷积补回）。
- **G1 参照未对齐（g4 已消除）**：G1 的 `full` 参照来自**另一个脚本（`train_5fold.py`）、
  另一张卡（RTX 3060）、另一个 `data_root`（空格路径）** ⇒ Δ 只能叫「参考性对照」。
  **g4 的参照 = A1 自跑的同脚本/同卡/同 data_root/同 seed `full`**，此限制不再成立。
- **G1 统计效力**：只跑 2 折时，差异可能被折间噪声淹没。
  报告里给了 `fold_spread` 字段，**只在 |Δ| > 折间跨度时才声称差异显著**。
- **G2 不评 test**：多种子只给 train loss 离散度，不给 test 指标（见 §7 铁律 1）。
  其 val/acc 那一维由 **g4 A3/A4** 补上。

**第二段（g4）：**

- **仍然没有 CI / 不给显著性**：对照 5 折、消融 5 折 —— n=5 依然是小样本。
  只能报 `fold_spread` / 极差，**不得**写「显著提升/下降」。项目铁律要求
  统计单位 = 源图 + cluster bootstrap，本实验两者都不满足。
- **容量不匹配依然在**：见 §5.8 第 2 条。**只有 `no_sam`（ΔParams −459）可干净归因**。
- **`no_ilf` 若复现 G1 的不收敛现象**，只能写「移除 ILF 后训练不收敛」（§5.8 第 3 条）。
- **Swish 无开关**：`make_switch_patch()` 只有 4 个开关 ⇒ 消融覆盖 **4 个机制**，不是 5 个。
- **GC 口径**：FLOPs 必须绑定「batch=1, 224²; 1 MAC = 2 FLOPs; 只算 Conv/Linear」。
  不绑口径的 FLOPs 数字不可比。

---

## 10. 本机自检记录（2026-09-23 起；2026-09-24 补数据根目录变更回归；2026-09-26 补第二段 g4；2026-09-27 补第三段 g5）

| 脚本 | 自检方式 | 结果 |
|---|---|---|
| 5 个脚本 | `ast.parse` 语法检查 | ✅ 全部 OK |
| 4 个脚本 | `--help`（会真实求值固定路径 f-string） | ✅ 导入无缺失、路径文案正确 |
| `gpu_lib.py` | 路径分支单测（POSIX / Windows / 显式覆盖） | ✅ 8/8 PASS |
| `gpu_lib.py` | `--smoke` | ✅ 5/5 PASS（fp `8e542321…`、56,812,813 params） |
| `g3_efficiency.py` | `--dry-run --iters 2 --warmup 1` | ✅ MACs 精确复现 7,492,310,331；落盘字段自洽 |
| `g2_multi_seed.py` | `--dry-run` | ✅ 种子改写通过，源文件 SHA 不变 |
| `g2_multi_seed.py` | **真实子进程启动副本** | ✅ 越过 manifest 检查（见下方修复 3/4） |
| `g1_ablation.py` | `--dry-run` | ✅ 硬门通过；4 变体均可前向 |
| `g1_ablation.py` | CPU 真实训练 2 epoch × 2 batch | ✅ 训练/选best/落盘全链路通 |
| `g0_run_all.py` | `--dry-run` | ✅ 三步编排 + 汇总报告生成 |
| `g2_multi_seed.py` | `make_seed_variant()` 直调（默认 / 显式 `--data-root` / 非法值 / 源文件 SHA） | ✅ 4 组全 PASS（见修复 7） |
| `g0_run_all.py` | `--out-root` / `--out-dir` / 二者冲突 | ✅ 全 PASS（见修复 8） |
| `accept.py` | **4 场景退出码测试**（完整已实测 / GPU 缺失 / 空目录 / 缺 full 行） | ✅ `0` / `1` / `1` / `1`，逐条诊断正确 |
| `g1_ablation.py` | `load_reference_cv()` 显式语义 5 例 | ✅ 全 PASS（见修复 9） |
| 5 个脚本 | **纯 flat 目录副本重跑**（项目外，无 `DMSA_v24/` 子目录） | ✅ 5/5 退出码 0，`arch_probe` 解析到同目录副本（见修复 11） |
| `g2_multi_seed.py` | `make_seed_variant` 默认 / 显式 `data_root` 两例 | ✅ 2/2；显式必改写且回读通过（见修复 12） |
| 全部脚本 | **数据根目录变更后全量回归**（`ast` ×6 / `--help` ×4 / `--smoke` / 4 个 `--dry-run`） | ✅ 全 rc=0；帮助文本与产物字段均已显示新路径 |
| `g2_multi_seed.py` | **失败退出码**（数据根指向不存在的路径 → 副本必然失败） | ✅ rc=1，且逐个打印失败种子的退出码与日志路径（见修复 14） |
| **`g4_run.py`** | `py_compile` | ✅ rc=0 |
| **`g4_run.py`** | `--plan --device cpu`（本机数据） | ✅ 预检 **8/8 OK**；40 run / 105.0 h；对照泄漏 group 跨折=262 / sha 跨折=140 / 污染 val=774 |
| **`g4_run.py`** | `--dry-run --device cpu` | ✅ **5/5 变体**参数量逐位对档（见 §2.2 表）；`full` fp = 基线 |
| **`g4_run.py`** | **40 run 真实端到端**（`--num-workers 0 --epochs 1 --max-batches 1`，4m30s） | ✅ **rc=0**；`runs/` 40 个目录；`grouped`/`control` n_folds=**5**；消融 4 变体 n_folds=**5**、Δ 覆盖 1–5 折；6 个产物齐全 |
| **`g4_run.py`** | `--resume`（全跳过） | ✅ rc=0；`8/8`→`[跳过 i/N]` 逐条打印，各表仍完整，`skipped=True` |
| **`g4_run.py`** | 实时进度解析（console log） | ✅ tqdm 帧原样转发；`[进度] … 0.0%→12.5% … 剩余≈… run 1/8`；epoch 汇总行齐全 |
| **`g4_run.py`** | `--quiet` 彻底性（同口径 40 run 微缩版） | ✅ train/val tqdm 帧 **160/120 → 0/0**；`[进度]` 90→81（节流抖动）、`ep … best=` 40→40 **不变** |
| **`g4_run.py`** | **40 run + `--quiet` 端到端**（4m11s） | ✅ **rc=0**；`runs/`=40、`per_run.csv`=41 行、`n_runs`=40、消融 4 变体 × 5 折、`grouped` 指纹=基线 —— 与不加 `--quiet` 时**完全一致** |
| **`g5_run.py`** | `py_compile` | ✅ rc=0 |
| **`g5_run.py`** | `--plan --device cpu`（本机数据） | ✅ 预检通过；**6 run / 15.75 h / ≈¥38**；`full` 参照找到 2 折、`g1` 已有产物在位 |
| **`g5_run.py`** | `--dry-run --device cpu` | ✅ **rc=0**；`full` fp = 基线硬门 PASS；`no_dual_path` params=46,753,037（Δ −10,059,776） |
| **`g5_run.py`** | **6 run 真实端到端**（`--num-workers 0 --epochs 1 --max-batches 1`） | ✅ **rc=0**；6 个产物齐全；`abl/runs`=2、`seed/runs`=4；合并消融表 4 变体、`seed` 表 3 种子 |
| **`g5_run.py`** | `--resume`（全跳过） | ✅ **rc=0**；**6/6 保护、全部跳过**（修复 23 后的回归判据） |
| **`g5_run.py`** | `--overwrite-dry-run` / `--no-overwrite` | ✅ 前者列出 6 个待删目录 rc=0；后者 rc=**2**（按设计拒绝） |
| **`g5_run.py`** | `--quiet` 彻底性 | ✅ train/val tqdm 帧 0/0；日志 125 行、`[进度]` 与 `ep … best=` 行齐全 |
| **`g5_run.py`** | `nohup … --quiet &` 脱离终端 | ✅ 进程启动、`g5_progress.log` 正常写入（判据同 §5.9） |
| **`verify_g5.py`** | 对**本机 1-epoch 垃圾产物**跑（`--device cpu --epochs 1 --max-batches 1`） | ✅ **rc=1**，正确报 `missing=['no_mdc']`、`seed43/44 acc=0.00%` ⇒ 验收器**真会拦**不合格产物（不是摆设） |
| **`verify_g5.py`** | 无产物场景 | ✅ **rc=2**，并打印"该下载到哪个目录"的提示 |

`g1 --dry-run` 的四个变体实测 Δ（与 `arch_probe.py` 完全对上）：

| 变体 | params | Δ vs full |
|---|---:|---:|
| `full` | 56,812,813 | +0（= 基线，硬门 ✅） |
| `no_sam` | 56,812,354 | −459 |
| `no_mdc` | 45,037,801 | −11,775,012 |
| `no_ilf` | 55,495,947 | −1,316,866 |

顺带在自检中抓到并修正了二十四个错误（第 1–19 个来自第一段，第 20–22 个来自第二段 `g4_run.py`，第 23–24 个来自第三段 `g5_run.py`）：

1. **参数归因键名误导**：`arch_probe.param_attribution()` 的 `edpb_fusion_conv`
   用子串匹配 `"fusion_conv" in name`，**同时命中两处模块**。
   实际拆分：块出口双路径融合 **20,149,760（35.47%）**，
   块内 MDC 三支融合 **1,185,792（2.09%）**。已在报告里改正。
2. **FLOPs 字段笔误**：`_arch_out/arch_probe.json` 里写的是
   `14,985,420,662`，不等于 `2 × MACs`（差 800,000）。
   正确值 **14,984,620,662**。写 `14.985 G` 不受影响。
3. **G2 副本缺 manifest（会导致 G2 直接崩）**：`final_train.py` 用
   `script_dir = Path(__file__).resolve().parent` 找 `pcv_v2_folds_seed42.csv`。
   副本被放在 `_variants/` 下，而 manifest 在基线目录里 →
   `FileNotFoundError`。**已修**：现在会把冻结 manifest 逐字节复制到 `_variants/`
   并复核 SHA-256。
4. **`re.sub` 替换串的反斜杠转义（Windows 上必炸）**：`OUTPUT_DIR = r"E:\Code\..."`
   里的 `\U` 会被 `re` 当成非法转义，抛 `re.PatternError: bad escape \U`。
   **已修**：替换串统一经 lambda 插入，并在传入前 `rstrip` 掉结尾分隔符。
   这个 bug 在 Linux 路径下不触发，所以此前的 dry-run 没暴露它。
5. **固定路径改动引入的 `NameError`（会直接崩）**：把输出改成
   `resolve_out_dir()` 后，`g0/g1/g2/g3` 都在用
   `resolve_out_dir / resolve_data_root / FIXED_OUT_ROOT / FIXED_DATA_ROOT`，
   但**这四个名字一个都没写进 `from gpu_lib import (...)`**。
   纯 `ast.parse` 查不出来（语法合法），只有真正 import 才炸。
   **已修**，并用「四个脚本各跑一次 `--help`」作为回归判据
   （`--help` 会真实求值路径 f-string，缺名必 `ImportError`/`NameError`）。
6. **`PUBLISHED_FLOPS` 常量残留旧笔误值**：`g3_efficiency.py` 里
   `PUBLISHED_FLOPS = 14_985_420_662`（就是修复 2 勘误掉的那个数），
   而这个常量会被写进 `g3_efficiency.json::flops.published_flops`，
   **等于把笔误重新带进云端产物**。**已修**为 `14_984_620_662`，
   并加了 `_assert_flops_convention()`：`FLOPs != 2 × MACs` 直接抛异常。

> 修复 5 是本轮最有价值的一条：它说明「语法检查 + dry-run 都过」并不能
> 保证「改完就没事」——**改动引入的新符号，必须有一次真实 import 才算验证过**。

7. **G2 的 `--data-root` 之前是空转的**：`final_train.py` 里的 `DATA_ROOT` 是
   **硬编码**的，而 `make_seed_variant()` 原本只改写 `SEED` / `OUTPUT_DIR` /
   `RUN_NAME`，**从不碰 `DATA_ROOT`**。后果：副本仍用硬编码路径，
   用户传 `--data-root` 却毫无效果（静默失效，最坏的一种 bug）。
   **已修**：显式给出 `--data-root` 时才改写副本里的 `DATA_ROOT`，并回读校验。
   ⚠️ **此行为已被修复 12 取代** —— 因为「`final_train.py` 自带的值就等于我们的固定值」
   这个前提后来被云端实测推翻，现在改为**总是**改写（见修复 12）。
   安全性已核实：`DATA_ROOT` **不在** `_check_final_protocol()` 的 6 项断言内
   （断言只有 FINAL_EPOCHS/BATCH_SIZE/LEARNING_RATE/LR_GAMMA/SEED/NUM_CLASSES），
   改写它不会触发硬门。另加非法值校验（空串、含引号/换行一律拒绝）。
8. **G0 的 `--out-dir` 之前被静默忽略**：`g0` 从 `base_parser()` 继承了
   `--out-dir`，但代码里只读 `--out-root`。后果：传 `--out-dir` 不报错、
   也不生效，产物悄悄落到默认位置。**已修**：`--out-dir` 在 g0 中等同
   `--out-root`；两者同时给出且不一致时**直接报错**（而不是随便挑一个）；
   并把 `--help` 里那句会误导人的 `/<tag>` 文案改掉。
9. **G1 的 `full` 基准是 Windows-only 硬编码路径（云上必残缺）**：
   `REFERENCE_CV` 原本写死 `E:/Code/NeuroEvo/NeuroEvo_AD/复现版实验数据`。
   云机器上没有 `E:` 盘 ⇒ 读不到 ⇒ `full` 行消失、`delta_vs_full` 全变 `None`，
   **但退出码仍是 0**（"看着跑完了其实残缺"）。**已修**：新增 `--reference-cv`，
   并扩大自动搜索范围（基线目录及其父目录、`~/work`）；
   **显式即显式**——指错目录或凑不齐 5 折一律 `FileNotFoundError` 退出，
   绝不静默回退；找不到参照时打印显式告警，且 `full` 只要在 `--variants` 里就实跑。
10. **`summarize()` 把 `epochs` 硬编码成 `BASE_EPOCHS`（200）**：传
    `--epochs 50` 时，`g1_ablation_summary.json::epochs` 与 Markdown 里的
    "训练轮数" 仍写 200 —— 溯源信息与实际不符。**已修**：`epochs` 参数透传。
11. **`load_arch_probe()` 不搜脚本自身目录（云上 flat 布局必失败）**：
    `g1_ablation.py` 原本只在 `DMSA_v24/` 之类的**子目录**里找 `arch_probe.py`，
    候选表里**没有脚本自己所在的目录**。本机之所以没暴露，是因为候选表末尾有一条
    `Path(r"E:/Code/.../DMSA_v24/arch_probe.py")` —— 在 Windows 上 `E:` 存在，它
    恰好命中，于是 `--dry-run` 通过；**但在云上（POSIX）这条路径不绝对
    （`is_absolute()` 为 False），完全失效**，而 `arch_probe.py` 按 §1.1 就是与
    本脚本**同目录**上传的 ⇒ 找不到 ⇒ G1 起不来。
    **已修**：把 `here.parent / "arch_probe.py"`（脚本自身目录）加为**首选**候选，
    子目录候选与 `E:` 回退保留在后面。
    **验证**：在项目之外另建一个**纯 flat 副本**（任何层级都没有 `DMSA_v24/` 子目录）
    重跑五个脚本（`gpu_lib --smoke` + `g1/g2/g3/g0 --dry-run`），
    ✅ 5/5 退出码 0，硬门通过，且记录到 `arch_probe` 解析到了同目录副本。
    ⚠️ 它与 5/7/8/9/10 同源：**"本机能过" 与 "云上能过" 是两件事**，
    凡是含平台回退路径的代码，都必须在一个**不含该回退条件的目录结构**下再跑一遍。
12. **`g2` 副本的 `DATA_ROOT` 改写前提失效（单跑 g2 会去读不存在的目录）**：
    `make_seed_variant()` 原本只在**显式给** `--data-root` 时才改写副本里的
    `DATA_ROOT`，理由是"`final_train.py` 自带的硬编码值就是正确值"。
    **该前提在云端实测定下来之后失效**：基线写的是
    `.../data/dmsa_pcv2/Combined Dataset`，而实测位置是
    `/home/featurize/work/Combined_Dataset`（见 §1 与 `gpu_lib.FIXED_DATA_ROOT`）。
    后果：绕过 g0 **单跑 `g2` 且不给 `--data-root`** 时，副本仍用基线那条不存在的
    路径 → `final_train.py` 自己的 `train/` 检查会报错退出（不是静默，但报错文案
    会把你引向"去改 `DATA_ROOT`"，而不是"你漏传了 `--data-root`"）。
    **已修**：`main()` 现在**总是**把本次解析出的数据根写进副本
    （`make_seed_variant(data_root=a.data_root)`），回读校验保留。
    **验证**：直调 `make_seed_variant` 两例 —— `data_root=None` 不改写；传新路径则
    副本含 `DATA_ROOT = r"/home/featurize/work/Combined_Dataset"`。2/2 PASS。
    ⚠️ 与 7/9 同源：**"某个字段/参数自带正确值"这类前提一旦被实测推翻，
    静默降级会立刻出现。凡是靠"默认值恰好对"来省略写入的地方，都必须复核。**
13. **`--smoke` 第 3 步在云端被静默跳过**：`cpu_smoke()` 的候选路径**只有本机
    Windows 那条**（`E:/Code/.../Combined Dataset/train`）。云上不存在 ⇒ 打印
    "跳过真实数据加载"后继续，**退出码仍是 0** —— 恰好让「在云上验证数据路径」
    这最关键的一步失效。**已修**：候选表**首选** `Path(FIXED_DATA_ROOT)/"train"`，
    本机路径退为次选；跳过时把**试过的路径逐条打印**。**验证**：本机仍 5/5。
14. **G2 在所有种子都失败时仍返回 0**：`main()` 末尾是**无条件** `return 0`，
    于是 `exit_code=1 / epochs=0 / final_train_loss=None` 的失败结果照样打印
    "G2 完成"，**退出码 0**。（走 `g0` 时能被拦住，因为 g0 会聚合子进程退出码；
    但**直接单跑 g2 就会被骗过去**。）
    **已修**：`failed = [r for r in per_seed if r["exit_code"] != 0 or
    r["n_epochs_logged"] == 0]`；非空则逐个打印失败种子的退出码与日志路径，
    **`return 1`**。**验证**：全失败 → rc=1；全成功 → rc=0。
    ⚠️ 与 7/8/9/10/12 同源：**「看起来跑完了」与「真的产出了可用结果」是两件事**。
15. **云端跑的是旧包（排查成本很高，故加版本标记）**：用户云上报错时，
    日志显示 `data root : .../data/dmsa_pcv2/Combined Dataset` —— 这个值在包里
    **早已改掉两轮**。因为包没有任何版本标识，只能靠肉眼比对路径才能发现。
    **已修**：`gpu_lib.PKG_VERSION = "2026-09-24.1"`，`g0/g1/g2/g3` 启动时各打印一行
    `包版本      : <版本>`；RUNBOOK §1 给出 `grep -n "^PKG_VERSION" gpu_lib.py`
    的一步核对法。
16. **版本行只覆盖了正式训练路径（`--dry-run` 看不到版本）**：`g1_ablation.py` 的
    `print(f"包版本 …")` 落在**正式训练分支**里，而 `--dry-run` 在它之前就 `return 0`
    ⇒ **用 `--dry-run` 核对版本时，g1 永远不打印版本行**，"任何脚本启动都能看到版本"
    的约定对 g1 不成立（g0/g2/g3 的版本行在通用头块里，不受影响）。
    **已修**：把版本行移进**始终执行**的头块（`baseline dir` 之前），删掉正式分支里的那处。
    **验证**：`g1 --dry-run` 头部第 4 行即 `包版本      : 2026-09-24.1`；
    `grep -c "PKG_VERSION"` == 2（import + 打印）、打印语句计数 == 1。
    ⚠️ 与 5/9/13 同源：**新增的"核对探针"本身也可能只在某条分支上生效** ——
    凡是加"给人看的核对信息"，必须放在**所有分支都会走到**的位置。
17. **G2 的子进程输出被全量缓存（日志 0 字节、中断即丢失）**：
    `run_variant()` 原先用 `subprocess.run(stdout=PIPE)`，子进程输出**全部缓存在内存**，
    等进程退出才一次性写盘 ⇒ 训练期间 `log_seedN.txt` 恒为 **0 字节**
    （用户只能以为"卡住了"），且实例被抢占 / OOM kill 时**整个训练日志丢失**。
    **已修**：新增 `gpu_lib.stream_run()`——`Popen` + `read1()` 增量读 + 逐帧 flush，
    并顺带实现了实时进度（训练进行中日志就有内容，实测 260 字节 vs 旧实现 0 字节）。
    ⚠️ 与 14 同源：**"跑完了"和"过程中可观测"是两件事**，长任务必须有心跳。
18. **覆盖清理时变量遮蔽 `info`（会让 G2 在训练结束后崩）**：
    新加的 `info = safe_rmtree(...)` 把上面 `make_seed_variant()` 返回的 `info`
    **覆盖掉了**，导致后面组装 `rec["variant_sha256"]` 时 `KeyError`——
    而且是**训练跑完之后**才崩，几小时白跑。**已修**：改名 `rm`。
    ⚠️ 这是"新增代码污染既有变量"的典型；**在大函数里加代码前先看同名变量是否已存在**。
19. **`safe_rmtree()` 的拒绝校验写在 `exists()` 之后**：
    原实现先判 `if not p.exists(): return`，于是"路径不存在"会**绕过全部安全校验**——
    一旦 `--out-dir` 拼错，等于什么都没检查。**已修**：改成
    **先做全部拒绝判定，再看是否存在**；并补上 `is_junction()` 判定
    （Windows 目录联接的 `is_symlink()` 可能是 False，rmtree 顺着删会删到目标内容）。
20. **（g4）报告头部把 epoch 数写死成 `BASE_EPOCHS`**：`summarize()` 里
    `"epochs": BASE_EPOCHS`（=200），于是用 `--epochs 1` 自检时，
    `g4_report.md` 头部仍写「训练 **200** epoch、batch=16」——
    **溯源信息与实际不符**。**已修**：`summarize(..., epochs=…, batch_size=…)`
    改由**实参 / 结果里的 `epochs_run`** 推导。**验证**：`--epochs 1` 时
    报告头部显示「训练 **1** epoch」。
    ⚠️ 与第一段修复 10 是**同一个 bug 的复发**（G1 里也犯过一次，当时只改了 G1）。
    **教训升级：这类"写死默认值"的溯源字段，改一个脚本时必须把同类脚本一起扫一遍。**
21. **（g4）消融段把折数写死成「2（fold 1,5）」**：`write_report()` 里
    `L.append("⚠️ 折数仅 2（fold 1,5），无 CI…")` 是**硬编码字符串**。
    本次消融已改为 5 折，若不修就是**报告说谎**。**已修**：改由
    `max(a["n_folds"] for a in ablation)` 动态生成，并分「完整五折 / 仅 N 折」两种措辞。
    **验证**：5 折时输出「折数 5（完整五折）」。另把 summary 里同一处的
    `"折数"` 兜底值一并改为动态。
    ⚠️ 与 20 同源：**「默认值恰好对」是最危险的一类正确**——正式跑恰好等于默认值时
    它永远不会暴露，只有用小参数自检才会现形。
22. **（g4）新加的 `--quiet` 只挡住一半 tqdm**：为后台跑新增 `--quiet` 时，我只把
    `train_one_epoch()` 包进了 `redirect_stdout(_TqdmTap(...))`，而
    **`validate_one_epoch()` 是在那个 with 块之外调用的** —— 它也吐 tqdm
    （`smoke_test.py:1061` `tqdm(loader, file=sys.stdout, desc="validation")`），
    于是 `--quiet` 下仍会漏出 **40 × 200 × 128 = 102 万帧**。
    **已修**：把 validation 一并包进同样的 tap。**验证（同口径 40 run 微缩版）**：
    train/val tqdm 帧 **160/120 → 0/0**，而 `[进度]` 行 90→81、`ep` 汇总行 40→40
    （前者是 5 秒节流的时序抖动，后者完全不变 ⇒ 信息没丢）。
    ⚠️ 与 16 同源：**"新增的功能/探针"本身也必须被验证**，不能只验证它没弄坏老路径。
23. **（g5）覆盖清理跑在 `--resume` 判断之前，导致 `--resume` 静默失效**：
    `g5_run.py` 默认"重跑即覆盖"（`overwrite=True` → `safe_rmtree` 删掉已有的 run 目录）。
    最初把这段清理放在 `--resume` 的"跳过已完成 run"检查**之前** ⇒ 清理先删了产物，
    resume 再回头看已经没有可跳过的 ⇒ **6 个 run 全部从头重跑**（`--resume` 形同虚设，
    且**退出码仍是 0**，不报错）。**已修**：删除前先判
    `if a.resume and (d / "best_metrics.json").is_file(): kept.append(...); continue`
    —— 要复用的产物**先保护、后清理**。
    **验证**：重建 6 个假产物后跑 `--resume` → `6/6 保护、全部跳过`、rc=0；
    `--overwrite-dry-run` 仍正常列出待删项、`--no-overwrite` 仍 rc=2。
    ⚠️ 与 14 / 17 同源：**"看起来成功（rc=0）"与"真的按预期行为"是两件事** ——
    `--resume` 这类功能的验收判据只能是**「跳过数 = 已完成数」**，不是"进程没报错"。
24. **（g5）`g5_precheck.json` 只在 `--plan` 分支写，但报告把它列为产物**：
    `write_report()` 的「§5 产物」清单里列了 `g5_precheck.json`，而该文件**只在
    `--plan` 分支**里 `write_json(...)` ⇒ **不带 `--plan` 的正式跑根本不产出它** ——
    报告在**列一个不存在的文件**（"报告说谎"这一类）。
    **已修**：把 precheck 字典提出为变量，**`--plan` 与正式跑都落盘**（`mode` 分别写 `plan`/`train`）。
    **验证**：重跑本机 6 run（`--device cpu --epochs 1 --max-batches 1`）→ `ls` 见到
    `g5_precheck.json`；`verify_g5.py` 的「核心 5 产物」判据 PASS。
    ⚠️ 兼容性：09-28 之前跑的正式产物**没有**这个文件（无害）；`verify_g5.py` 对它
    按**提示（非 FAIL）**处理。**已跑完的云上产物不必重跑**，只有将来重跑才需重传 `g5_run.py`。
    ⚠️ 与 20/21 同源：**"文档/报告承诺的产物"必须由同一段代码真的产出**，
    不能只靠某条分支（`--plan`）偶尔带上。

### 10.1 为什么本机自检必须加 `--num-workers 0`

- **现象**：Windows 上用默认 `--num-workers 6` 跑本机端到端小测 → 子进程抛
  `ModuleNotFoundError: No module named 'dmsa_baseline_smoke'`，父进程挂死。
- **根因**：DataLoader 在 **Windows 用 `spawn`**（子进程重新 `import`），而 `smoke_test.py`
  是用 `importlib` **动态加载**成 `dmsa_baseline_smoke` 的，**不在 `sys.path` 上**
  ⇒ 子进程 import 不到。**Linux 用 `fork`，继承父进程内存，不受影响** ⇒
  **云端一直是对的，只有本机自检会踩**。
- **处置**：`g4_run.py` 在检测到 `os.name == "nt" and num_workers > 0` 时**主动打印提示**，
  **不改默认值**（避免动云端行为）。**本机自检请显式加 `--num-workers 0`；云端保持默认 6。**
- 这条对第一段的 g1/g2/g3 同样适用。

