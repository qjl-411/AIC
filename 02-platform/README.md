# 平台代码 · NeuroEvo-AD 脑影溯析平台

## 一、这个平台是什么（背景与用途）

阿尔茨海默病（AD）起病隐匿且不可逆，临床阅片存在判读一致性问题。本平台把参赛算法 **DMSA-Net** 落地成一套可直接使用的系统，让"影像 → 结论 → 证据"成为一条完整链路：

- **影像分析**：上传一张脑部 MRI，得到四期分类（无障碍 / 极轻 / 轻度 / 中度）结论与四类概率分布，而非单一标签；
- **可解释性**：同时给出空间注意力图与 Grad-CAM 热力图，供医生核对模型关注区域是否与临床判断一致；
- **报告导出**：一键导出结构化 PDF 检测报告（含受检者信息、结论与置信度、模型版本、训练轮次、分析耗时、输入尺寸），结论可带走、可追溯；
- **研究文献站**：内置 902 篇 AD/MCI 文献元数据（其中 **336 篇全文 PDF 已随包提供**），支持按关键词检索、年份/主题聚类、被引次数排序，并直接打开全文——把"这一张片子的结论"接到"这一类病例的证据"上。

**平台后端加载的模型权重，就是封存测试所用的同一份**（见《实验数据和模型产物》`08_Model_Checkpoints_and_Artifacts`），不存在"平台与实验两套模型"的问题。

---

## 二、目录结构

```
平台代码/
└── NeuroEvo_AD/
    ├── README.md / README.en.md      # 平台说明（含权重还原、PDF 还原）
    ├── SHA256SUMS.txt                # 平台自带的文件校验清单
    ├── RESTORE_MODEL_CHECKPOINT.bat / .sh   # 权重分块还原脚本（如需）
    ├── LOCALIZE_LITERATURE_PDFS.bat / .sh   # 文献 PDF 本地化脚本
    ├── apps/
    │   ├── web/                      # 前端：Vite + React 19 + TypeScript（含 three.js 3D 大脑）
    │   │   ├── src/                  # 页面与组件（首页大屏、影像工作台、研究中心、文献详情）
    │   │   ├── public/assets/models/ # 3D 模型等静态资源（brain.glb）
    │   │   ├── package.json          # 依赖与脚本（dev / build / preview / typecheck）
    │   │   └── vite.config.ts        # 已配置 /api 代理到后端 8000 端口
    │   └── api/                      # 后端：FastAPI
    │       ├── requirements.txt      # Python 依赖
    │       ├── app/
    │       │   ├── main.py           # 应用入口
    │       │   ├── config.py         # 配置项（均可用 NEUROEVO_ 前缀环境变量覆盖）
    │       │   ├── model_service.py  # 模型加载与推理服务
    │       │   ├── schemas.py
    │       │   ├── routes/           # health（健康检查）/ inference（推理）/ literature（文献）
    │       │   ├── ml/dmsa/          # DMSA-Net 网络结构实现
    │       │   ├── literature/       # 文献库检索与全文服务
    │       │   └── reporting/        # PDF 检测报告生成
    │       ├── data/
    │       │   ├── neuroevo_papers.db    # 文献数据库（902 篇元数据）
    │       │   └── pdfs/                 # 文献全文 PDF（336 篇，约 899 MB，已随包）
    │       ├── scripts/              # 数据维护脚本
    │       └── tests/                # 后端测试
    ├── deploy/                       # 部署脚本
    └── docs/                         # 开发文档与变更记录
```

> 已剔除（可由命令重建，不占体积）：`node_modules/`、`dist/`、`.venv/`、`__pycache__`、工具残留目录。
> **模型权重**放在兄弟目录《实验数据和模型产物》中，启动时用环境变量指过去即可（见下一节）。

---

## 三、运行环境（实测通过的配置）

| 组件 | 要求 | 实测值 |
|---|---|---|
| Node.js | ≥ 22.12（package.json `engines`） | v22.22.2 |
| Python | 3.10+ | 3.14.5 |
| 后端依赖 | 见 `apps/api/requirements.txt` | fastapi、uvicorn[standard]、python-multipart、pydantic-settings、httpx、pillow、torch（实测 2.14.0+cpu 可正常运行）、torchvision、numpy |
| GPU | 可选 | 无 GPU 时用 `NEUROEVO_MODEL_DEVICE=cpu`（CPU 推理已实测通过） |

文献全文 PDF（336 篇）已随包提供，无需联网下载。

---

## 四、如何运行（以下命令均已实测通过）

### 1) 启动后端

```bash
cd 平台代码/NeuroEvo_AD/apps/api

python -m venv .venv                 # Windows: .venv\Scripts\activate
source .venv/bin/activate            # Linux / macOS
pip install -r requirements.txt
```

指定模型权重（权重文件在《实验数据和模型产物》目录中）：

```bash
# Windows（cmd）
set NEUROEVO_MODEL_CHECKPOINT=..\..\..\实验数据和模型产物\08_Model_Checkpoints_and_Artifacts\final_model_epoch126.pth
set NEUROEVO_MODEL_DEVICE=cpu

# Linux / macOS
export NEUROEVO_MODEL_CHECKPOINT=../../实验数据和模型产物/08_Model_Checkpoints_and_Artifacts/final_model_epoch126.pth
export NEUROEVO_MODEL_DEVICE=cpu
```

> 也可以把该 `.pth` 复制到 `NeuroEvo_AD/model_artifacts/` 下，同样会被识别。

启动：

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 2) 启动前端

```bash
cd 平台代码/NeuroEvo_AD/apps/web
npm install
npm run dev
```

浏览器打开 **http://localhost:5173** 。Vite 已把 `/api` 代理到 `http://127.0.0.1:8000`，无需额外配置跨域。

生产构建（可选）：`npm run build`（会先做 TypeScript 类型检查）。

---

## 五、常用配置项（环境变量，前缀 `NEUROEVO_`）

| 变量 | 作用 | 默认 |
|---|---|---|
| `NEUROEVO_MODEL_CHECKPOINT` | 模型权重 `.pth` 路径 | 空（未配置时推理不可用） |
| `NEUROEVO_MODEL_DEVICE` | 推理设备（`cpu` / `cuda:0` / `auto`） | `auto` |
| `NEUROEVO_MODEL_METRICS` | 模型性能指标 json 路径 | 空 |
| `NEUROEVO_LITERATURE_DATABASE` | 文献数据库路径 | `data/neuroevo_papers.db` |
| `NEUROEVO_LITERATURE_PDF_DIRECTORY` | 文献全文目录 | `data/pdfs` |
| `NEUROEVO_CORS_ORIGINS` | 允许的前端来源 | `http://localhost:5173` |
| `NEUROEVO_API_PREFIX` | 接口前缀 | `/api/v1` |

---

## 六、功能自检清单（均为本包实测结果）

| 检查项 | 命令 / 操作 | 期望结果 |
|---|---|---|
| 服务存活 | 打开 `http://127.0.0.1:8000/api/v1/health` | `{"status":"ok",...}` |
| 模型已加载 | `http://127.0.0.1:8000/api/v1/inference/model` | `"ready":true`，`epoch:126`，架构指纹 `8e542321…` |
| 上传影像推理 | 工作台上传一张 MRI | 返回结论 + 四类概率 + 注意力图（base64 PNG） |
| 可解释性 | `POST /api/v1/inference/explain` | `state:"completed"`，含热力图 |
| 文献检索 | `http://127.0.0.1:8000/api/v1/literature/papers?query=AD` | 返回条目（含 DOI、标题、摘要） |
| 文献主题/年份聚类 | `.../literature/topics`、`.../literature/stats/years` | 返回主题与年度分布 |
| 文献全文 | `.../literature/papers/{id}/pdf` | `200`、`application/pdf`（实测 2.5 MB，文件头 `%PDF-1.4`）；缺失全文时返回友好提示并给出外部 OA 链接 |
| 前端页面 | 打开 `http://localhost:5173` | 首页加载，标题 NeuroEvo-AD |

---

## 七、常见问题

- **推理接口提示权重未部署**：检查 `NEUROEVO_MODEL_CHECKPOINT` 是否指向正确的 `.pth`（在《实验数据和模型产物》目录）。
- **没有 GPU**：设置 `NEUROEVO_MODEL_DEVICE=cpu`，CPU 推理已验证可用（单张耗时高于 GPU，属正常）。
- **`npm run dev` 报 Node 版本低**：需 Node ≥ 22.12。
- **文献全文打不开**：336 篇有本地全文，其余条目会提示外部开放获取链接（属预期行为，非故障）。
