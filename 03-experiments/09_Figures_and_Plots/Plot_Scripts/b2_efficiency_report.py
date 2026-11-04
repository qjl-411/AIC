# -*- coding: utf-8 -*-
"""
b2_efficiency_report.py -- 把已有的架构探针实测值 + 本机解析计算，汇总成
比赛方案可用的「效率指标」证据表。

证据来源（全部已存在于磁盘，本脚本不重新造数据）:
    1. DMSA_v24/_arch_out/arch_probe.json   —— DMSA-Net 的 params / FLOPs / 参数归因 / 消融容量
    2. 本脚本解析计算 + torchvision 实测     —— ResNeXt50-32x4d 对照参数量
    3. DMSA_v24/_arch_out/arch_probe.json    —— CPU 单张前向耗时（相对参考）

输出:
    out/b2_efficiency_dmsa.json
    out/b2_efficiency_baseline.json
    out/b2_efficiency_table.md      —— 可直接粘贴进方案文档的 Markdown 表
    out/b2_efficiency_attribution.md

纪律（必须遵守）:
  - **本机 torch 为 CPU-only（torch 2.13.0+cpu, cuda_available=False）**，
    因此 GPU 延迟/吞吐/显存 **无法在本机测量**。凡涉及 GPU 的行一律标注
    `NOT_MEASURED_ON_THIS_MACHINE`，不得用 CPU 数字冒充 GPU 数字。
  - FLOPs 必须绑定计数约定（见 arch_probe.json 的 convention 字段），
    否则同一模型会出现多个"FLOPs"值，跨论文比较不成立。
  - ResNeXt50 参数量用 torchvision 实测（而非手算），并记录 torchvision 版本。
  - DMSA-Net 的参数量只在 fp 与基线一致时才可引用（arch_probe 已断言）。
"""

import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

ARCH_PROBE = os.path.join(HERE, "..", "DMSA_v24", "_arch_out", "arch_probe.json")

#: 本机测量能力（由调用方在运行前确认；见下方 probe_local_capability）
LOCAL_CAPS = {}


def probe_local_capability():
    """探测本机是否能测 GPU 指标。测不到就如实标 NOT_MEASURED。"""
    caps = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": None,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_mem_gb": None,
    }
    try:
        import torch
        caps["torch"] = torch.__version__
        caps["cuda_available"] = bool(torch.cuda.is_available())
        if caps["cuda_available"]:
            caps["gpu_name"] = torch.cuda.get_device_name(0)
            caps["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    except Exception as e:
        caps["torch"] = "unavailable: %s" % e
    return caps


def load_probe():
    with open(ARCH_PROBE, "r", encoding="utf-8") as f:
        return json.load(f)


def resnext50_reference():
    """torchvision 实测 ResNeXt50-32x4d 参数量（weights=None，只看结构）。"""
    try:
        import torch
        import torchvision
        from torchvision.models import resnext50_32x4d
        m = resnext50_32x4d(weights=None)
        n = sum(p.numel() for p in m.parameters())
        return {"params": n, "torchvision": torchvision.__version__,
                "torch": torch.__version__, "source": "torchvision 实测"}
    except Exception as e:
        return {"params": None, "error": "%s: %s" % (type(e).__name__, e)}


def main():
    global LOCAL_CAPS
    LOCAL_CAPS = probe_local_capability()
    probe = load_probe()
    ref = resnext50_reference()

    dmsa_params = probe["params"]["total"]
    fp = probe["params"]["fp"]
    fp_ok = probe["params"]["fp_matches_baseline"]
    flops = probe["flops"]

    # ---- ResNeXt50 对照 ----
    base = {
        "model": "ResNeXt50-32x4d (torchvision, weights=None)",
        "params": ref.get("params"),
        "params_source": ref.get("source"),
        "torchvision": ref.get("torchvision"),
        "note": ("仅作参数量量级对照；本方案**未在同协议下重训该基线**，"
                 "因此不得用于性能归因或互为对照的精度比较。"),
    }
    if ref.get("params"):
        base["param_ratio_vs_dmsa"] = round(dmsa_params / ref["params"], 4)
        base["param_delta_vs_dmsa"] = dmsa_params - ref["params"]

    with open(os.path.join(OUT, "b2_efficiency_baseline.json"), "w",
              encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)

    # ---- DMSA-Net 效率记录 ----
    dmsa = {
        "model": "DMSA-Net (EnhancedResNeXt50, full 变体)",
        "params_total": dmsa_params,
        "params_trainable": probe["params"]["trainable"],
        "architecture_fingerprint": fp,
        "fingerprint_matches_baseline": fp_ok,
        "flops_gflops": round(flops["flops"] / 1e9, 3),
        "macs_g": round(flops["macs"] / 1e9, 3),
        "flops_convention": flops["convention"],
        "cpu_ms_per_image": {
            "value": round(flops["cpu_ms_per_image"], 3),
            "scope": "CPU 单张前向（相对参考）",
            "machine": LOCAL_CAPS.get("platform"),
            "torch": LOCAL_CAPS.get("torch"),
        },
        "gpu_latency_ms": "NOT_MEASURED_ON_THIS_MACHINE",
        "gpu_throughput_img_s": "NOT_MEASURED_ON_THIS_MACHINE",
        "gpu_peak_memory_mb": "NOT_MEASURED_ON_THIS_MACHINE",
        "gpu_not_measured_reason": (
            "本机 torch 为 CPU-only 构建（cuda_available=%s），无法测 GPU 指标。"
            % LOCAL_CAPS.get("cuda_available")
        ),
        "param_attribution": probe["param_attribution"],
    }
    with open(os.path.join(OUT, "b2_efficiency_dmsa.json"), "w",
              encoding="utf-8") as f:
        json.dump(dmsa, f, ensure_ascii=False, indent=2)

    # ---- 参数归因表 ----
    attr = probe["param_attribution"]
    tot = max(sum(attr.values()), 1)
    rows = sorted(attr.items(), key=lambda kv: -kv[1])
    lines = [
        "| 模块 | 参数量 | 占比 |",
        "|---|---:|---:|",
    ]
    for k, v in rows:
        lines.append("| `%s` | %s | %.2f%% |" % (k, format(v, ","), v / tot * 100))
    fusion = attr.get("edpb_fusion_conv", 0) + attr.get("edpb_fusion_bn", 0)
    lines += [
        "",
        "> **EDPB `fusion_conv` + `fusion_bn` 合计 %s（%.2f%%）** —— 是参数量占比最大的单项，"
        % (format(fusion, ","), fusion / tot * 100),
        "> 且初版论文正文与 Fig.4 均未单独列出。这是「论文写的参数量对不上」的主因。",
        "> 空间注意力模块（SAM）仅 **%s** 个参数（%.4f%%），在报告中必须如实说明其边际性。"
        % (format(attr.get("spatial_attention", 0), ","),
           attr.get("spatial_attention", 0) / tot * 100),
    ]
    with open(os.path.join(OUT, "b2_efficiency_attribution.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ---- 主表 ----
    variants = {v["variant"]: v for v in probe["variants"]}
    m = [
        "# DMSA-Net 效率指标证据表",
        "",
        "> 全部数字来源：`DMSA_v24/_arch_out/arch_probe.json`（本项目自实现 torch forward hook）。",
        "> 本表**不新增任何未经测量的数字**。",
        "",
        "## 1. 复杂度指标（实测）",
        "",
        "| 指标 | DMSA-Net | ResNeXt50-32x4d | 说明 |",
        "|---|---:|---:|---|",
        "| 参数量 | **%s** | %s | ResNeXt50 为 torchvision 结构实测 |"
        % (format(dmsa_params, ","),
           format(base["params"], ",") if base.get("params") else "—"),
        "| 参数量比值 | — | — | DMSA-Net 为 ResNeXt50 的 **%.2fx** |"
        % (base.get("param_ratio_vs_dmsa") or 0),
        "| FLOPs (batch=1, 224²) | **%.3f G** | 未测 | 1 MAC = 2 FLOPs |"
        % dmsa["flops_gflops"],
        "| MACs (batch=1, 224²) | **%.3f G** | 未测 | — |" % dmsa["macs_g"],
        "| CPU 单张前向 | %.1f ms | 未测 | **仅相对参考**，非部署指标 |"
        % dmsa["cpu_ms_per_image"]["value"],
        "| GPU 延迟 / 吞吐 / 峰值显存 | **未测** | 未测 | 本机 torch 为 CPU-only 构建 |",
        "",
        "## 2. FLOPs 计数约定（跨论文比较必须绑定）",
        "",
        "| 项 | 取值 |",
        "|---|---|",
    ]
    for k, v in flops["convention"].items():
        m.append("| %s | %s |" % (k, v))
    m += [
        "",
        "> ⚠️ **FLOPs 没有绝对唯一数字。** 不绑定上表约定的 FLOPs 数字不具备可比性。",
        "",
        "## 3. 结构消融的容量变化（已实测，尚未训练）",
        "",
        "| 变体 | 参数量 | Δ vs full | fingerprint=基线 | 前向可跑 |",
        "|---|---:|---:|:---:|:---:|",
    ]
    for name in ["full", "no_sam", "no_mdc", "no_ilf", "no_dual_path"]:
        v = variants.get(name)
        if not v:
            continue
        m.append("| `%s` | %s | %s | %s | %s |" % (
            name, format(v["params"], ","),
            format(v["params_delta"], "+,") if v["params_delta"] else "0",
            "✅" if v["fp_equals_baseline"] else "—",
            "✅" if v.get("forward_ok") else "❌",
        ))
    m += [
        "",
        "> 上表证明**开关注入机制可用且 `full` 与基线结构完全一致**（硬约束）。",
        "> 但**这 4 个变体都还没有训练过** —— 没有精度数字，因此**不能**据此写任何精度结论。",
        "",
        "## 4. 本机测量能力声明",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| 平台 | %s |" % LOCAL_CAPS.get("platform"),
        "| Python | %s |" % LOCAL_CAPS.get("python"),
        "| torch | %s |" % LOCAL_CAPS.get("torch"),
        "| CUDA 可用 | %s |" % LOCAL_CAPS.get("cuda_available"),
        "| GPU | %s |" % (LOCAL_CAPS.get("gpu_name") or "无"),
        "",
        "> 结论：**本机不具备 GPU 效率测量能力**。方案中的 GPU 延迟/吞吐/显存数字",
        "> 必须标注「待云环境实测」，不得由 CPU 数据外推。",
        "",
    ]
    with open(os.path.join(OUT, "b2_efficiency_table.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(m))

    print("[B2] 产物 -> %s" % OUT)
    print("     DMSA params      = %s" % format(dmsa_params, ","))
    print("     ResNeXt50 params = %s" % (format(base["params"], ",") if base.get("params") else "N/A"))
    print("     FLOPs            = %.3f G" % dmsa["flops_gflops"])
    print("     GPU 指标         = NOT_MEASURED_ON_THIS_MACHINE")


if __name__ == "__main__":
    main()
