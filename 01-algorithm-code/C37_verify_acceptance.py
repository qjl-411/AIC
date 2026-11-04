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
