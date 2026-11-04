# NeuroEvo-AD

当前代码包含原有 MRI 推理后端骨架，以及新增的 AD/MCI/MRI 文献研究站。

## 0. 交付物与数据说明（先读这一节）

本仓库是参赛交付版本。**代码、文献数据库、模型权重**都在仓库里；只有 336 篇文献 PDF 全文因为体积原因放在发行版附件里。

### 仓库内已有的东西

| 内容 | 位置 | 规模 |
|---|---|---|
| 前后端全部源码 | `apps/` | — |
| 文献数据库 | `apps/api/data/neuroevo_papers.db` | 902 篇 AD/MCI 文献元数据，7.8MB |
| 模型权重 | `model_artifacts/final_model_epoch126.pth.part01 … part05` | 217.5MB，切成 5 块 |

### 第一步：还原模型权重

MRI 推理功能需要 `final_model_epoch126.pth`。Gitee 单文件上限 50MB，所以它在仓库里是 5 个分块，先跑一次还原脚本：

```bat
RESTORE_MODEL_CHECKPOINT.bat        :: Windows
```
```bash
sh RESTORE_MODEL_CHECKPOINT.sh      # macOS / Linux
```

脚本按顺序拼接分块并校验 SHA-256，应得到：

```text
sha256 = ba96575888e35764e3ff196674e52757d9a5f9c90742a38aa8f1d1f9a8457658
size   = 228051470 bytes
```

哈希对不上说明分块损坏，不要继续使用。

### 第二步：取文献 PDF 全文（可选，336 篇 / 943,050,248 字节）

`apps/api/data/pdfs/` 的 PDF 全文**不在 Git 里**——Gitee 个人版单仓库上限 500MB，装不下 943MB。它们压缩成 10 个 zip 放在仓库的 **「发行版 / Releases」** 页面：

<https://gitee.com/qrochelimit/best-doctor/releases>

**附件清单**（10 卷合计 336 篇，与源目录逐一致）：

| 附件 | 体积 | 内含 PDF |
|---|---|---|
| `NeuroEvo-AD-literature-pdfs-part01of10.zip` | 81.4 MiB | 47 |
| `NeuroEvo-AD-literature-pdfs-part02of10.zip` | 85.2 MiB | 37 |
| `NeuroEvo-AD-literature-pdfs-part03of10.zip` | 84.7 MiB | 44 |
| `NeuroEvo-AD-literature-pdfs-part04of10.zip` | 82.3 MiB | 37 |
| `NeuroEvo-AD-literature-pdfs-part05of10.zip` | 82.8 MiB | 36 |
| `NeuroEvo-AD-literature-pdfs-part06of10.zip` | 81.6 MiB | 37 |
| `NeuroEvo-AD-literature-pdfs-part07of10.zip` | 80.6 MiB | 30 |
| `NeuroEvo-AD-literature-pdfs-part08of10.zip` | 71.9 MiB | 30 |
| `NeuroEvo-AD-literature-pdfs-part09of10.zip` | 79.9 MiB | 29 |
| `NeuroEvo-AD-literature-pdfs-part10of10.zip` | 22.5 MiB | 9 |
| **合计** | **752.8 MiB**（789,374,977 字节） | **336** |

**还原步骤**

1. 下载全部 10 个附件。只下其中几卷也可以，文献阅读功能对缺失条目会优雅降级。
2. 全部解压到**项目根目录**——zip 内部已带 `apps/api/data/pdfs/` 路径，会自动落到正确位置：

```bash
unzip NeuroEvo-AD-literature-pdfs-part*.zip -d .
```

   Windows 上也可全选 10 个 zip → 右键「解压到当前文件夹」。

3. **校验**（两个数必须都对，否则说明有卷没下全或解压不全）：

```bash
ls  apps/api/data/pdfs/*.pdf | wc -l    # 应为 336
cat apps/api/data/pdfs/*.pdf | wc -c    # 应为 943050248
```

不想下载的话，也可以自己从开放获取源头抓取（**不保证抓全**：部分 OA 链接已失效，实测抽样 8 个里约三成不可达）：

```bash
cd apps/api
python scripts/localize_literature_pdfs.py --all --workers 4
```

### 没有放进仓库的东西

| 内容 | 原因 |
|---|---|
| `apps/api/data/pdfs/`（943MB） | 超 Gitee 单仓库 500MB 上限 → 见上方发行版附件 |
| `node_modules/`、`.venv/`、`dist/` | 可由 `npm install` / `pip install` / `npm run build` 重建 |

## 1. 后端安装

```bash
cd apps/api
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`。建议在 OpenAlex 注册免费 API Key，并设置：

```env
NEUROEVO_OPENALEX_API_KEY=你的Key
```

API Key 不是最小测试的强制要求，但批量同步建议配置。

## 2. 初始化文献数据库

```bash
cd apps/api
python scripts/literature_sync.py init
```

## 3. 先做小样本抓取

建议第一次不要直接抓几千篇：

```bash
python scripts/literature_sync.py crawl --from-year 2024 --to-year 2026 --max-per-query 10
python scripts/literature_sync.py stats
python scripts/literature_sync.py audit --sample-per-topic 3
```

确认结果质量后，再扩大到核心库，并补一小部分 2010–2017 的高相关经典文献：

```bash
python scripts/literature_sync.py crawl --from-year 2018 --to-year 2026 --max-per-query 200
python scripts/literature_sync.py crawl-classics --max-per-query 20
```

采集流程会按 OpenAlex ID 去重，并进行 AD/MCI + MRI/AI 相关性过滤和 NeuroEvo 11 Topic 多标签分类。

## 4. 增量刷新

免费 API 无法使用 OpenAlex 的 premium `from_updated_date` 同步过滤器，因此本项目默认采用“最近年份窗口重新查询 + 本地 upsert”的免费方案：

```bash
python scripts/literature_sync.py sync --lookback-years 2 --max-per-query 100
```

可以通过 Windows Task Scheduler 或 cron 定期执行该命令。

## 5. OA PDF（可选）

只下载 OpenAlex 已记录的开放获取 PDF URL，不绕过登录、验证码或付费墙：

```bash
python scripts/literature_sync.py download-pdfs --limit 100
```

PDF 默认保存到 `apps/api/data/pdfs/`，并把本地路径写回数据库。

## 6. 启动 FastAPI

```bash
cd apps/api
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

主要文献接口：

- `GET /api/v1/literature/papers`
- `GET /api/v1/literature/papers/{id}`
- `GET /api/v1/literature/topics`
- `GET /api/v1/literature/stats/years`
- `GET /api/v1/literature/stats/topics`

`/papers` 支持 `q`、`year`、`year_from`、`year_to`、`topic`、`oa`、`sort`、`page`、`page_size`。

## 7. 前端

```bash
cd apps/web
npm install
npm run dev
```

Vite 已把 `/api` 代理到 `http://127.0.0.1:8000`。

研究中心 `/reports` 当前实现：

- 关键词搜索
- 年份柱状图 hover 数量 / 点击筛选
- Topic 动态统计 / 点击筛选
- 搜索 + 年份 + Topic 联动
- 相关性 / 引用量 / 最新 / 最早排序
- 分页
- OA PDF / DOI / 来源跳转

## 8. 测试

```bash
cd apps/api
pytest -q
```

前端完整 typecheck/build 需要先执行 `npm install`，因为交付压缩包不包含 `node_modules`。
