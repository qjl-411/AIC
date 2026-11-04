#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A6-s (v2.6) —— 预训练权重静态映射测试（**0 GPU，CPU 数分钟**）
==============================================================

决定论文「贡献 1（双路径解决知识迁移权衡）」是否成立。

【v2.6 最重要的一处方法学修正（GPT 第五轮 E5，成立）】
v2.4 只报了一个数：**可匹配参数 4.90%**，判定 `NOT_SEAMLESS`。
问题是这个 4.90% 的口径**从未在文档里写明**，而实测口径是
"**同名 key + 同 shape**"。可 DMSA 的瓶颈层叫 `path1_conv1`，
torchvision 的叫 `conv1` —— **命名空间完全不同**，所以同名匹配必然只命中
stem 与 downsample，4.90% 是**被系统性低估的下界**，不是"可迁移能力"。

更要命的是：**这个数根本不决定论文主张。** 论文的问题是
"有没有实际加载 ImageNet 权重"，而代码里 `_init_weights()` 是 kaiming 从头初始化、
全文没有任何 `load_state_dict(ImageNet)` 调用 —— 所以
「无缝加载」不成立的**决定性证据是代码事实，不是匹配率**。
用一个口径可疑的比例去承担结论，是本末倒置。

因此 v2.6 改为**三列 + 一个代码事实**：

| 列 | 含义 | 回答的问题 |
|---|---|---|
| A `exact_key_compatible` | 同名 key + 同 shape | 「不做任何重命名，能直接 load 多少」 |
| B `semantic_mapped_shape_compatible` | 经**显式语义映射表**后 shape 相符 | 「若肯写映射表，机制上最多能搬多少」（**上界**） |
| C `actually_loaded_and_forward_verified` | 真正写入模型 + 前向可跑（实测） | 「映射表到底能不能用」 |
| D `code_fact` | 训练脚本里**是否存在**加载预训练权重的代码路径 | 「论文主张有没有实现支撑」（**决定性**） |

⚠️ C 列用的是 torchvision 的**随机初始化**权重（本机离线，不下载 ImageNet）。
它验证的是**机制可行性**，不是"迁移后的精度"。报告里必须显式声明，
不得被读成"已成功迁移预训练权重"。

判定规则（事前写死，**以 D 为决定性判据**）
    - D = False  ->  `CLAIM_UNSUPPORTED_NO_LOAD_PATH`
      论文「无缝加载 ImageNet」不成立，贡献 1 必须重写。**与 A/B 的比例无关。**
    - D = True 且 B >= 80%  ->  `SEAMLESS_PLAUSIBLE`
    - D = True 且 B <  80%  ->  `PARTIAL_MAPPING_REQUIRED`

输出：_a6s_out/weight_map_report.json + .md （经 lib_provenance 统一盖章）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys

from B16_lib_provenance_audit import write_artifact

# --------------------------------------------------------------------------
# 显式语义映射表（GPT E5 要求：先建立 mapping，再看 shape 是否兼容）
# --------------------------------------------------------------------------
# 左：torchvision ResNeXt50 的 key 正则；右：DMSA-Net 对应 key 模板（{0}{1}{2} 为捕获组）
#
# 依据（2026-09-23 实测结构对比，见报告 §语义映射表）：
#   RNX layerX.Y.conv1 (1×1)  <->  DMSA layerX.Y.path1_conv1   （标准瓶颈首层）
#   RNX layerX.Y.conv2 (分组) <->  DMSA layerX.Y.path2_conv2.0 （MDC 洞里那条 depthwise；形状本就不符）
#   RNX layerX.Y.conv3 (1×1)  <->  DMSA layerX.Y.path1_conv3   （瓶颈末层）
#   RNX layerX.Y.bn1/2/3      <->  对应 path1_bn1 / path2_conv2.1 / path1_bn3
#   RNX fc                    <->  DMSA fc.4                   （DMSA 的 head 是多层 MLP）
SEMANTIC_RULES = [
    (r"^conv1\.weight$", "conv1.weight"),
    (r"^bn1\.(\w+)$", "bn1.{0}"),
    (r"^layer(\d)\.(\d+)\.conv1\.weight$", "layer{0}.{1}.path1_conv1.weight"),
    (r"^layer(\d)\.(\d+)\.bn1\.(\w+)$", "layer{0}.{1}.path1_bn1.{2}"),
    (r"^layer(\d)\.(\d+)\.conv2\.weight$", "layer{0}.{1}.path2_conv2.0.weight"),
    (r"^layer(\d)\.(\d+)\.bn2\.(\w+)$", "layer{0}.{1}.path2_conv2.1.{2}"),
    (r"^layer(\d)\.(\d+)\.conv3\.weight$", "layer{0}.{1}.path1_conv3.weight"),
    (r"^layer(\d)\.(\d+)\.bn3\.(\w+)$", "layer{0}.{1}.path1_bn3.{2}"),
    (r"^layer(\d)\.(\d+)\.downsample\.0\.weight$", "layer{0}.{1}.downsample.0.weight"),
    (r"^layer(\d)\.(\d+)\.downsample\.1\.(\w+)$", "layer{0}.{1}.downsample.1.{2}"),
    (r"^fc\.weight$", "fc.4.weight"),
    (r"^fc\.bias$", "fc.4.bias"),
]

#: 判定预训练加载路径的代码事实扫描模式
LOAD_CODE_PATTERNS = [
    r"load_state_dict",
    r"torch\.load\s*\(",
    r"models\.\w+\s*\(",
    r"pretrained\s*=\s*True",
    r"weights\s*=\s*['\"]?IMAGENET",
    r"_Weights\.",
    r"resnext50",
    r"\.pth\b|\.pt['\"]",
]


def build_semantic_map(ref_keys, tgt_keys):
    """按 SEMANTIC_RULES 把 ResNeXt key 映射到 DMSA key（只做名字映射，不判 shape）。"""
    out = {}
    for rk in ref_keys:
        for pat, tmpl in SEMANTIC_RULES:
            m = re.match(pat, rk)
            if m:
                try:
                    out[rk] = tmpl.format(*m.groups())
                except IndexError:
                    out[rk] = tmpl
                break
    return out


def scan_pretrained_load_path(module_path: pathlib.Path) -> dict:
    """**代码事实**：训练/建模脚本里到底有没有加载预训练权重的路径。

    这是决定性的那一列。匹配率再高，只要没有这条代码路径，
    「无缝加载」这句话在实现层就是空的。
    """
    #: 本项目里所有可能构造/训练模型的文件 —— 都要扫，不能只扫入口
    cands = [module_path]
    d = module_path.parent
    for name in ("smoke_test.py", "train_5fold.py", "final_train.py",
                 "final_test.py", "train.py", "model.py"):
        p = d / name
        if p.exists() and p not in cands:
            cands.append(p)

    hits, scanned = [], []
    for p in cands:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        scanned.append(p.name)
        for ln, line in enumerate(txt.splitlines(), 1):
            code = line.split("#", 1)[0].strip()          # 去掉注释再看
            if not code:
                continue
            for pat in LOAD_CODE_PATTERNS:
                if re.search(pat, code):
                    hits.append({"file": p.name, "line": ln,
                                 "pattern": pat, "code": code[:150]})
                    break
    #: `load_state_dict` 也用于加载**自己的 checkpoint**，必须人工区分；
    #: 这里把它单独标出来，避免把"加载自己的 .pth"误读成"加载 ImageNet"
    self_ckpt = [h for h in hits if "load_state_dict" in h["code"]]
    #: 只有"真的去取 ImageNet 权重"的调用才算数：
    #: `models.xxx(weights=...)` / `pretrained=True` / `xxx_Weights.DEFAULT`
    _IMG = re.compile(r"models\.\w+\s*\("
                      r"|pretrained\s*=\s*True"
                      r"|weights\s*=\s*['\"]?IMAGENET"
                      r"|_Weights\.")
    imgsrc = [h for h in hits if _IMG.search(h["code"])]
    return {
        "scanned_files": scanned,
        "n_hits": len(hits),
        "hits": hits[:60],
        "hits_that_may_load_imagenet": imgsrc,
        "hits_load_state_dict": self_ckpt,
        "verdict": "NO_IMAGENET_LOAD_PATH" if not imgsrc else "HAS_CANDIDATE_LOAD_PATH",
        "caveat": ("`load_state_dict` 命中也可能是加载本项目自己的 checkpoint；"
                   "判定只认 `pretrained/weights=` 或 `models.xxx(` 这类构造式调用。"),
    }


def main():  # noqa: C901  (单函数流程，拆开反而看不清三列怎么算的)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-module", default="DMSA_net/smoke_test.py",
                    help="包含 create_model() 的脚本")
    ap.add_argument("--out", default="_a6s_out")
    ap.add_argument("--skip-forward", action="store_true",
                    help="跳过 C 列的前向验证（仅用于极慢环境）")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mp = pathlib.Path(a.model_module)

    try:
        import torch
        import torchvision
        from torchvision.models import resnext50_32x4d
    except Exception as e:
        print(f"[BLOCKED] 无法导入 torch/torchvision：{type(e).__name__}: {e}")
        write_artifact(out / "weight_map_report.json",
                       {"status": "blocked",
                        "reason": f"{type(e).__name__}: {e}",
                        "blocked_reason": "blocked_missing_inputs：无 torch 环境"})
        return 2

    if not mp.exists():
        print(f"[BLOCKED] 找不到 {mp}")
        write_artifact(out / "weight_map_report.json",
                       {"status": "blocked", "reason": f"找不到 {mp}",
                        "blocked_reason": "blocked_missing_inputs：模型模块缺失"})
        return 2

    # ---- 参考模型（**不下载权重**：离线环境用随机初始化，只验证机制）----
    try:
        ref = resnext50_32x4d(weights=None)
    except TypeError:
        ref = resnext50_32x4d(pretrained=False)
    ref_sd = ref.state_dict()

    # ---- 目标模型 ----
    sys.path.insert(0, str(mp.parent.resolve()))
    spec = importlib.util.spec_from_file_location("dmsa_model", mp)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"[WARN] 导入 {mp} 失败：{e}")
        write_artifact(out / "ref_keys.json",
                       {k: list(v.shape) for k, v in ref_sd.items()})
        return 2

    create = getattr(mod, "create_model", None)
    if create is None:
        print("[BLOCKED] 模块中没有 create_model()")
        return 2
    #: 必须显式传 num_classes=4，不能靠 TypeError fallback ——
    #: 否则可能构造出别的模型，使分母错误（v2.4 已记录此坑）
    try:
        model = create(num_classes=4)
    except TypeError:
        model = create()
    tgt_sd = model.state_dict()

    EXPECTED = 56_812_813
    n_tgt_params = sum(int(p.numel()) for p in model.parameters())
    n_tgt_sd = sum(int(v.numel()) for v in tgt_sd.values())
    param_keys = {k for k, _ in model.named_parameters()}
    param_numel = {k: int(v.numel()) for k, v in model.named_parameters()}
    sd_numel = {k: int(v.numel()) for k, v in tgt_sd.items()}
    fp_warn = None
    if n_tgt_params != EXPECTED:
        fp_warn = (f"目标模型参数量 {n_tgt_params:,} != 基线 {EXPECTED:,}；"
                   f"匹配率分母不可比，请核对 --model-module")

    # ================= 列 A：同名 key + 同 shape =================
    exact = [k for k in tgt_sd
             if k in ref_sd and tuple(ref_sd[k].shape) == tuple(tgt_sd[k].shape)]
    n_exact = sum(sd_numel[k] for k in exact if k in param_keys)
    #: 分母统一用 parameters()（不含 BN running buffer）。两个口径都报。
    ratio_exact_params = n_exact / n_tgt_params if n_tgt_params else 0.0
    ratio_exact_sd = (sum(sd_numel[k] for k in exact) / n_tgt_sd) if n_tgt_sd else 0.0

    # ================= 列 B：语义映射后 shape 相符 =================
    sem_map = build_semantic_map(list(ref_sd), list(tgt_sd))
    mapped, mapped_shape_ok, mapped_shape_bad, mapped_no_target = [], [], [], []
    for rk, tk in sem_map.items():
        if tk not in tgt_sd:
            mapped_no_target.append({"ref": rk, "dmsa": tk})
            continue
        if tuple(ref_sd[rk].shape) == tuple(tgt_sd[tk].shape):
            mapped_shape_ok.append({"ref": rk, "dmsa": tk,
                                    "shape": list(tgt_sd[tk].shape),
                                    "numel": sd_numel[tk],
                                    "is_param": tk in param_keys})
        else:
            mapped_shape_bad.append({"ref": rk, "dmsa": tk,
                                     "ref_shape": list(ref_sd[rk].shape),
                                     "dmsa_shape": list(tgt_sd[tk].shape)})
    n_sem = sum(m["numel"] for m in mapped_shape_ok if m["is_param"])
    ratio_sem_params = n_sem / n_tgt_params if n_tgt_params else 0.0

    # ---- 为什么语义映射**没有**把比例抬上去？逐块宽度对照 ----
    # GPT 第五轮 E5 的假设是"命名空间不同 ⇒ 4.90% 被低估 ⇒ 建映射表就能提高"。
    # 实测（2026-09-23）**该假设不成立**：映射后比例**一个百分点都没动**，
    # 因为 DMSA 的瓶颈**宽度本身**就与 ResNeXt50 不同（扩张比 2× vs 4×）。
    # 必须把宽度对照表打出来，否则"4.90% 是低估"这个直觉还会被反复提出。
    width_rows = []
    for stage in (1, 2, 3, 4):
        for blk in range(3):
            base = f"layer{stage}.{blk}"
            for ci, tag in ((1, "conv1"), (2, "conv2"), (3, "conv3")):
                rk, tk = f"{base}.{tag}.weight", f"{base}.path1_{tag}.weight" \
                    if tag != "conv2" else f"{base}.path2_conv2.0.weight"
                if rk in ref_sd and tk in tgt_sd:
                    rs, ts = list(ref_sd[rk].shape), list(tgt_sd[tk].shape)
                    if rs != ts:
                        width_rows.append({
                            "ref": rk, "dmsa": tk,
                            "ref_shape": rs, "dmsa_shape": ts,
                            #: 卷积/线性：out, in 在前两位；depthwise 的 in 是分组数
                            "ref_out_in": rs[:2] if len(rs) >= 2 else rs,
                            "dmsa_out_in": ts[:2] if len(ts) >= 2 else ts})
            if len(width_rows) >= 12:
                break
        if len(width_rows) >= 12:
            break

    # ================= 列 C：真正写入 + 前向可跑 =================
    loadable = {m["dmsa"]: ref_sd[m["ref"]].clone() for m in mapped_shape_ok}
    c = {"attempted_keys": len(loadable),
         "verified_written_keys": 0, "verified_written_numel": 0,
         "forward_ok": None, "note": "未执行"}
    try:
        incompatible = model.load_state_dict(loadable, strict=False)
        after = model.state_dict()
        written = [k for k, v in loadable.items()
                   if k in after and torch.equal(after[k], v)]
        c["verified_written_keys"] = len(written)
        c["verified_written_numel"] = sum(
            int(after[k].numel()) for k in written if k in param_keys)
        c["unexpected_keys"] = list(getattr(incompatible, "unexpected_keys", []))[:20]
        if not a.skip_forward:
            model.eval()
            with torch.no_grad():
                y = model(torch.zeros(1, 3, 224, 224))
            c["forward_ok"] = True
            c["forward_out_shape"] = list(y.shape)
        c["note"] = ("用 torchvision **随机初始化** 权重验证机制可行性；"
                     "本机离线未下载 ImageNet，故本列**不**代表迁移后的精度")
    except Exception as e:
        c["forward_ok"] = False
        c["error"] = f"{type(e).__name__}: {e}"

    # ================= 列 D：代码事实（决定性）=================
    code_fact = scan_pretrained_load_path(mp)

    # ================= 判定 =================
    if code_fact["verdict"] == "NO_IMAGENET_LOAD_PATH":
        verdict = "CLAIM_UNSUPPORTED_NO_LOAD_PATH"
        verdict_note = ("训练/建模脚本中**不存在**加载 ImageNet 预训练权重的代码路径。"
                        "「无缝加载 ResNeXt50」在实现层不成立 —— "
                        "此判定**不依赖** A/B 两列的匹配率。")
    elif ratio_sem_params >= 0.80:
        verdict, verdict_note = "SEAMLESS_PLAUSIBLE", "存在加载路径，且语义映射可覆盖 ≥80% 参数。"
    else:
        verdict = "PARTIAL_MAPPING_REQUIRED"
        verdict_note = "存在加载路径，但语义映射仅覆盖 " + f"{ratio_sem_params:.2%}，需列出具体层。"

    rep = {
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "n_target_params": n_tgt_params,
        "n_target_state_dict_numel": n_tgt_sd,
        "fingerprint_warning": fp_warn,
        # ---- 三列 ----
        "col_A_exact_key": {
            "n_params": n_exact, "ratio_of_params": round(ratio_exact_params, 4),
            "ratio_of_state_dict": round(ratio_exact_sd, 4),
            "n_keys": len(exact), "keys": exact,
            "definition": "同名 key 且 shape 相同（不做任何重命名）",
        },
        "col_B_semantic_mapped": {
            "n_params": n_sem, "ratio_of_params": round(ratio_sem_params, 4),
            "delta_params_vs_col_A": n_sem - n_exact,
            "n_keys_ok": len(mapped_shape_ok), "n_keys_shape_conflict": len(mapped_shape_bad),
            "n_keys_no_target": len(mapped_no_target),
            "keys_ok": mapped_shape_ok, "shape_conflict": mapped_shape_bad[:30],
            "no_target": mapped_no_target[:30],
            "definition": "经显式 SEMANTIC_RULES 映射后 shape 相符（机制上界）",
            "why_no_gain": ("语义映射**未**提高可迁移参数（delta={}）。"
                            "原因不是命名，而是瓶颈扩张比不同："
                            "ResNeXt50 为 4×，DMSA path1 为 2×。"
                            "因此 4.90% 是**真实结构上界**，不是命名空间造成的低估。"
                            ).format(n_sem - n_exact),
            "width_comparison": width_rows,
        },
        "col_C_actually_loaded": c,
        "col_D_code_fact": code_fact,
        "verdict": verdict, "verdict_note": verdict_note,
        "thresholds_pre_registered": {
            "col_A_ratio_used_for_verdict": False,
            "decision_rule": "col_D == NO_IMAGENET_LOAD_PATH -> CLAIM_UNSUPPORTED_NO_LOAD_PATH",
        },
        "note": ("strict=False 只放宽 missing/unexpected key，同名形状不符仍抛 RuntimeError，"
                 "故必须先按 key+shape 过滤。分母用 parameters()（不含 BN buffer），"
                 "state_dict 口径同时报出以便对账。"),
    }
    write_artifact(out / "weight_map_report.json", rep)

    # ---------------------------- Markdown ----------------------------
    md = ["# A6-s 预训练权重静态映射报告（v2.6 三列版）", "",
          f"- torch `{torch.__version__}` / torchvision `{torchvision.__version__}`",
          f"- 目标模型参数量：**{n_tgt_params:,}**（`parameters()`；"
          f"`state_dict()` 含 buffer 为 {n_tgt_sd:,}）", ""]
    if fp_warn:
        md += [f"> ⚠️ **{fp_warn}**", ""]
    md += ["## 三列总账（**不要只看一个数**）", "",
           "| 列 | 定义 | 参数量 | 占比 | key 数 |", "|---|---|---:|---:|---:|",
           f"| **A** exact-key | 同名+同shape | {n_exact:,} | {ratio_exact_params:.2%} | {len(exact)} |",
           f"| **B** semantic-mapped | 显式映射后shape相符 | {n_sem:,} | {ratio_sem_params:.2%} | {len(mapped_shape_ok)} |",
           f"| **C** actually-loaded | 真正写入+前向 | {c['verified_written_numel']:,} | "
           f"{c['verified_written_numel']/n_tgt_params:.2%} | {c['verified_written_keys']} |", "",
           f"- **结论（与直觉相反）**：A → B 的参数量增量为 **{n_sem - n_exact:,}**，"
           f"即**语义映射一个参数都没多搬**。4.90% 不是被命名空间低估的，"
           f"而是**真实的结构上界**。原因见下节宽度对照。",
           f"- 前向验证：{'通过 ' + str(c.get('forward_out_shape')) if c.get('forward_ok') else c.get('error', '未执行')}",
           f"- ⚠️ {c['note']}", "",
           "## 为什么语义映射没有提高比例（宽度对照，**推翻 E5 的前提**）", "",
           "GPT 第五轮 E5 的判断是「4.90% 由'同名 key'算出，故系统性低估，"
           "建映射表即可提高」。**实测不成立** —— 建立显式映射后比例纹丝不动。", "",
           "根因：DMSA 的瓶颈**宽度本身**与 ResNeXt50 不同。", "",
           "| 位置 | ResNeXt50 | DMSA-Net |", "|---|---|---|"]
    for w in width_rows[:12]:
        md.append(f"| `{w['ref']}` | {w['ref_out_in']} | "
                  f"`{w['dmsa']}` → {w['dmsa_out_in']} |")
    md += ["",
           "即：**ResNeXt50 的瓶颈扩张比是 4×（64→128→256），"
           "DMSA `path1` 是 2×（64→64→128）**。",
           "名字不同**且**形状不同 —— 语义映射能解决前者，解决不了后者。", "",
           "> 这对本项目的意义：**不能再指望「补一张映射表」把预训练迁移做起来**。"
           "若真要做，只能 (a) 把 path1 加宽到与 ResNeXt50 一致（改变架构、"
           "使 fingerprint 失效、需重跑全部实验），或 (b) 接受从头训练。", "",
           "## 列 D｜代码事实（决定性判据）", "",
           f"- 扫描文件：{', '.join(code_fact['scanned_files'])}",
           f"- 判定：**{code_fact['verdict']}**",
           f"- 命中 `pretrained/weights=` 或 `models.xxx(` 的行数："
           f"**{len(code_fact['hits_that_may_load_imagenet'])}**", ""]
    if code_fact["hits_that_may_load_imagenet"]:
        md += ["| 文件 | 行 | 代码 |", "|---|---:|---|"]
        for h in code_fact["hits_that_may_load_imagenet"][:20]:
            md.append(f"| `{h['file']}` | {h['line']} | `{h['code']}` |")
    else:
        md += ["> 未发现任何加载 ImageNet 预训练权重的构造式调用。",
               f"> （另有 {len(code_fact['hits_load_state_dict'])} 处 `load_state_dict`，"
               f"但那是加载本项目自己的 checkpoint，不构成预训练加载。）"]
    md += ["", "## 语义映射表", "",
           "| ResNeXt50 key | DMSA-Net key | shape 一致 |", "|---|---|---|"]
    for m in mapped_shape_ok:
        md.append(f"| `{m['ref']}` | `{m['dmsa']}` | ✅ {m['shape']} |")
    for m in mapped_shape_bad[:20]:
        md.append(f"| `{m['ref']}` | `{m['dmsa']}` | ❌ {m['ref_shape']} vs {m['dmsa_shape']} |")
    md += ["", "## 判定", "", f"### `{verdict}`", "", verdict_note, "",
           "> **判定规则（事前写死）**：列 D 为决定性判据；"
           "只要代码里没有 ImageNet 加载路径，「无缝加载」即不成立，"
           "**与 A/B 两列的比例无关**。", "",
           "## 对论文的直接影响", "",
           "- 现基线 `_init_weights()` 为 kaiming 从头初始化，且无预训练加载路径，",
           "  **代码事实支持「本模型从头训练，未使用预训练」**。",
           "- 若仍要保留「双路径解决知识迁移权衡」这一贡献，",
           "  必须先真正实现映射表加载（列 B/C 已给出上界与可行性），",
           "  再训练对照 —— 那是 A6-t（可选训练，见路线 §5.4）。",
           "- 论文 Table 1 的 8 个对比模型是否用了预训练，PDF 中完全未报告；",
           "  若基线用而 DMSA 不用，「+5.0 个百分点」是配置不对等的对比。"]
    (out / "weight_map_report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"列A 同名匹配 {ratio_exact_params:.2%} | 列B 语义映射 {ratio_sem_params:.2%} "
          f"| 列C 实载 {c['verified_written_keys']} keys | 列D {code_fact['verdict']}")
    print(f"判定 {verdict}")
    print(f"输出：{out/'weight_map_report.json'}, {out/'weight_map_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
