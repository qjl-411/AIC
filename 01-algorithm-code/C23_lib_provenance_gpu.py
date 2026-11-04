#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DMSA-Net v2.6 —— 产物溯源公共库（**纯标准库，不依赖 numpy / torch**）
=====================================================================

【为什么单独一个模块，而不是塞进 lib_dedup.py】
`a0_inventory.py`（阶段 0.0 输入盘点）是**整条流水线的第一步**，必须能在
"环境还没搭好、numpy 还没装"的状态下直接跑 —— 它本身也确实只用了标准库。
如果把溯源函数放进 `lib_dedup.py`，`a0` 就被迫背上 numpy 依赖，
等于把"第一步"变成"最后一步才能跑"。
溯源是元信息，不应该有科学计算依赖。

【这个模块要解决什么（N6，v2.6 自查新增）】
v2.4 建了 `code_fingerprint()`，并在总结里把它当作"可复现性已解决"的证据。
实测（2026-09-23）打脸：

| 产物 | 指纹 |
|---|---|
| `_a1cal_out/suggested_thresholds.json` | 有，但**已过期**（产物 08:50，代码 08:52 又改过） |
| `_a0_out/inventory.json` | **无** |
| `_a6s_out/weight_map_report.json` | **无** |
| `_arch_out/arch_probe.json` | **无** |
| `_audit_out/leakage_audit.json` | **无** |
| `_s2_out/stage2_report.json` | **无** |

即：**覆盖率 1/6，且那 1 个是坏的。**
一个只覆盖 1/6、且没有任何回读校验的指纹机制，比没有更危险 ——
它会制造"已经溯源了"的错觉。因此本模块同时提供：

1. `code_fingerprint()` —— 算指纹（**排除本文件自身？不，包含全部 .py**，
   保持"任何一行代码改动都会改变指纹"这一强性质）；
2. `stamp()` / `write_artifact()` —— 落盘时**统一盖章**；
3. `assert_fresh()` —— 回读时**硬校验**，不一致直接抛异常，不许只打 WARN。

【纪律】
- 所有产出脚本**只许**通过 `write_artifact()` 落盘 JSON 产物；
- `test_acceptance.py::T13` 会遍历产物目录，任何一个缺指纹即验收失败；
- `test_acceptance.py::T11.6` 会验证"改一行代码 → 旧产物立刻被判 STALE"。
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
from typing import Sequence

#: 本模块所在目录 —— 所有指纹以此为锚，避免受调用方 cwd 影响
_HERE = pathlib.Path(__file__).resolve().parent

#: 参与代码指纹的文件后缀。只认 .py：数据/产物/配置不进来（它们各有各的指纹）
_CODE_SUFFIX = ".py"

PRODUCER_VERSION = "v2.7"


def code_fingerprint(files: Sequence[pathlib.Path] | None = None) -> str:
    """代码指纹：对本目录下所有 `.py` 按文件名排序后取内容 SHA-256。

    用途（GPT T7 要求）：把"这一版结果是用哪一版代码跑出来的"钉死。

    ⚠️ 不要用 `lib_version="2.4.0"` 这类**手写常量**代替 —— 那是可以被遗忘更新
    的声明，不是可验证的指纹。本函数从文件内容直接算。

    ⚠️ 指纹**包含 `test_acceptance.py` 本身**。这是有意的：验收脚本的改动
    同样会改变"这一版结论"的含义。
    """
    if files is None:
        files = sorted(_HERE.glob(f"*{_CODE_SUFFIX}"))
    h = hashlib.sha256()
    for p in sorted(files, key=lambda x: pathlib.Path(x).name):
        p = pathlib.Path(p)
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def config_hash(cfg: dict) -> str:
    """配置指纹：对配置字典做 canonical JSON（键序无关）后取 SHA-256。"""
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, ensure_ascii=False,
                   default=str).encode("utf-8")).hexdigest()


class StaleArtifactError(RuntimeError):
    """产物指纹与当前代码/配置不一致 —— 该产物**不得**被当作本版结论引用。"""


def stamp(payload: dict, cfg: dict | None = None) -> dict:
    """给任何要落盘的产物字典**统一盖章**：代码指纹 + 配置指纹 + 时间戳。

    不修改入参（返回新 dict）—— 避免调用方在写盘后还持有被污染的引用。
    """
    out = dict(payload)
    out["code_fingerprint"] = code_fingerprint()
    if cfg is not None:
        out["config_fingerprint"] = config_hash(cfg)
        out["config"] = cfg
    out["stamped_at_utc"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")
    out["producer_version"] = PRODUCER_VERSION
    return out


def write_artifact(path, payload: dict, cfg: dict | None = None) -> dict:
    """**落盘一个已盖章的 JSON 产物**（唯一允许的产物写入口）。"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = stamp(payload, cfg)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    return out


def assert_fresh(artifact, strict: bool = True) -> dict:
    """回读产物时校验指纹是否仍与当前代码（及配置）一致。

    返回 `{"fresh": bool, "artifact_fp": ..., "current_fp": ..., "reasons": [...]}`。
    `strict=True` 且不一致时抛 `StaleArtifactError`。

    设计原则：**过期产物必须硬失败，不能只是打印警告。**
    "打印了警告但继续用"正是本项目反复出现的失败模式 ——
    文档声明已修、代码没修、产物过期，三件事都能被一句 `[WARN]` 掩盖过去。
    """
    if isinstance(artifact, (str, pathlib.Path)):
        artifact = json.loads(pathlib.Path(artifact).read_text(encoding="utf-8"))
    cur = code_fingerprint()
    rep = {"fresh": True, "artifact_fp": artifact.get("code_fingerprint"),
           "current_fp": cur, "reasons": []}

    if rep["artifact_fp"] is None:
        rep["fresh"] = False
        rep["reasons"].append("产物缺少 code_fingerprint（N6：未盖章）")
    elif rep["artifact_fp"] != cur:
        rep["fresh"] = False
        rep["reasons"].append(
            f"code_fingerprint 不一致：产物 {rep['artifact_fp'][:16]}… ≠ 当前 {cur[:16]}…")

    cfg = artifact.get("config")
    if cfg is not None and artifact.get("config_fingerprint") is not None:
        if artifact["config_fingerprint"] != config_hash(cfg):
            rep["fresh"] = False
            rep["reasons"].append("config_fingerprint 与 config 内容不自洽")

    if not rep["fresh"] and strict:
        raise StaleArtifactError("；".join(rep["reasons"]))
    return rep


def scan_freshness(root) -> dict:
    """扫描一个产物目录（或若干目录）下所有 .json，汇总新鲜度。

    给 `run_all.py` 与 `test_acceptance.py` 共用：**任何一个 STALE 都算硬失败**。
    """
    root = pathlib.Path(root)
    dirs = [root] if root.is_dir() and root.name.endswith("_out") else \
           sorted(p for p in root.glob("*_out") if p.is_dir())
    rows, n_stale, n_nofp = [], 0, 0
    for d in dirs:
        for f in sorted(d.glob("*.json")):
            try:
                rep = assert_fresh(f, strict=False)
            except Exception as e:                      # 解析失败也要记账
                rows.append({"file": str(f.relative_to(root)),
                             "fresh": False, "reason": f"{type(e).__name__}: {e}"})
                n_stale += 1
                continue
            if not rep["fresh"]:
                n_stale += 1
                if rep["artifact_fp"] is None:
                    n_nofp += 1
            rows.append({"file": str(f.relative_to(root)),
                         "fresh": rep["fresh"],
                         "reason": "；".join(rep["reasons"]) or "OK"})
    return {"n_json": len(rows), "n_stale": n_stale, "n_missing_fp": n_nofp,
            "rows": rows}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="DMSA_v24 产物新鲜度检查")
    ap.add_argument("root", nargs="?", default=str(_HERE),
                    help="DMSA_v24 目录（默认本目录）")
    a = ap.parse_args()
    res = scan_freshness(a.root)
    for r in res["rows"]:
        flag = "OK   " if r["fresh"] else "STALE"
        print(f"[{flag}] {r['file']:44s} {r['reason']}")
    print(f"\n共 {res['n_json']} 个 JSON 产物；过期/缺指纹 {res['n_stale']} "
          f"（其中完全无指纹 {res['n_missing_fp']}）")
    raise SystemExit(0 if res["n_stale"] == 0 else 1)
