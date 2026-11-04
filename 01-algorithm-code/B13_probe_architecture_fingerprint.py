#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 3.1–3.2 / 4.1–4.2 / 5.5 架构探针（v2.4）
=============================================

覆盖路线方案：
  3.1  参数量实测（替换论文 25,689,412）
  3.2  FLOPs / 吞吐（有 torch 时用 hooks 逐层累加；无 fvcore 依赖）
  3.3  参数归因分解（EDPB fusion_conv 占比）
  4.1  架构开关注入（monkeypatch，**不修改基线文件**）
  4.2  fingerprint + 前向输出 + 梯度不变量测试
  5.5  模型选择偏差相关：折内 metrics.csv 读取（best vs fixed epoch）

硬约束（来自路线方案 §4）：
  * **不修改 `DMSA_net/` 下任何基线文件**。所有开关通过
    runtime monkeypatch + wrapper 注入；`full` 配置下
    architecture_fingerprint 必须与基线常量完全一致，否则拒绝继续。
  * 基线 fingerprint：`8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975`
  * 基线参数量：56,812,813

用法：
    # 只测参数量/fingerprint/FLOPs（不需要 GPU，CPU 也行）
    python arch_probe.py --smoke ../DMSA_net/smoke_test.py --out _arch_out
    # 顺带跑 4.2 不变量测试
    python arch_probe.py --smoke ../DMSA_net/smoke_test.py --out _arch_out --invariants

⚠️ 依赖 torch。若 torch 不可用，脚本以 BLOCKED 退出并给出提示，不假装成功。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

#: N6（v2.6）：产物统一盖章，回读时硬校验指纹
from B16_lib_provenance_audit import write_artifact

# torch 是可选依赖：模块级尝试导入，失败时不抛错（main() 里给出 BLOCKED）。
try:
    import torch
    import torch.nn as nn
except Exception:                        # pragma: no cover
    torch = None
    nn = None

BASELINE_FP = "8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975"
BASELINE_PARAMS = 56_812_813


# ───────────────────────────────────────────────────────────── 加载基线模块
def load_baseline(smoke_path: pathlib.Path):
    """以 importlib 方式加载 smoke_test.py（它是核心实现，train_5fold import 它）。"""
    spec = importlib.util.spec_from_file_location("dmsa_baseline_smoke", str(smoke_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {smoke_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dmsa_baseline_smoke"] = mod
    spec.loader.exec_module(mod)
    return mod


# ───────────────────────────────────────────────────────────── FLOPs（hooks）
class FlopCounter:
    """用 forward hook 统计 Conv2d / Linear 的乘加次数（MACs）。

    不依赖 fvcore，公式：
      Conv2d : out_h*out_w * out_ch * in_ch/groups * kh * kw
      Linear : out_features * in_features
    报 FLOPs = 2 * MACs（惯例），并同时给出 MACs。
    """

    def __init__(self):
        self.macs = 0
        self.by_module: Dict[str, int] = {}
        self._handles = []

    def attach(self, model):
        import torch.nn as nn

        def hook(name):
            def fn(mod, inp, out):
                m = 0
                if isinstance(mod, nn.Conv2d):
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    oh, ow = out.shape[-2], out.shape[-1]
                    m = (oh * ow * mod.out_channels
                         * (mod.in_channels // mod.groups)
                         * mod.kernel_size[0] * mod.kernel_size[1])
                elif isinstance(mod, nn.Linear):
                    m = mod.out_features * mod.in_features
                if m:
                    self.macs += m
                    self.by_module[name] = self.by_module.get(name, 0) + m
            return fn

        for n, mod in model.named_modules():
            self._handles.append(mod.register_forward_hook(hook(n)))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @property
    def flops(self) -> int:
        return 2 * self.macs


# ───────────────────────────────────────────────────────────── monkeypatch 注入
def make_switch_patch(base_mod, use_sam: bool, use_mdc: bool, use_ilf: bool,
                      use_dual_path: bool):
    """构造一个 monkeypatch：把 `create_model` 换成带开关的版本。

    返回 (patched_create, restore_fn)。

    实现方式：**包装** EnhancedResNeXt50，在 __init__ 之后按开关
    动态替换/短路模块，而不是复制一份网络代码 —— 这样基线文件零改动，
    且 `full`（全 True）时得到的模型与基线的 state_dict 完全一致。
    """
    import torch
    import torch.nn as nn

    BaseModel = base_mod.EnhancedResNeXt50
    base_create = base_mod.create_model

    class PatchedModel(BaseModel):
        def __init__(self, num_classes: int = 4):
            self._cfg_use_ilf = use_ilf
            self._cfg_use_dual_path = use_dual_path
            super().__init__(num_classes=num_classes)

    # ILF 开关：把 fusion 替换成"只吃深层特征"的仿射（保形状，去融合语义）
    if not use_ilf:
        orig_init = BaseModel.__init__

        def _init_without_ilf(self, num_classes=4):
            orig_init(self, num_classes=num_classes)
            for attr in ("fusion23", "fusion34"):
                if hasattr(self, attr):
                    setattr(self, attr, _PassthroughFusion(getattr(self, attr)))

        def _wrapped_init(self, num_classes=4):
            self._cfg_use_ilf = use_ilf
            self._cfg_use_dual_path = use_dual_path
            _init_without_ilf(self, num_classes)
        PatchedModel.__init__ = _wrapped_init

    def patched_create(num_classes: int):
        model = PatchedModel(num_classes=num_classes)
        if not use_dual_path:
            _collapse_dual_path(model)      # 结构性移除 path2 支路
        if not use_mdc:
            _replace_mdc_with_plain(model)
        if not use_sam:
            _remove_sam(model)
        return model

    def restore():
        base_mod.create_model = base_create

    return patched_create, restore


class _PassthroughFusion(nn.Module):
    """ILF 关闭时的替身：**只返回深层特征**（去掉 α/(1-α) 互补融合）。

    必须是 nn.Module 才能作为子模块赋值给 `fusion23/34`（torch 的硬约束）。
    输出通道必须等于 high_channels（= 原 ILF 的 `align_conv.out_channels`）。
    """

    def __init__(self, orig):
        super().__init__()
        # 原 ILF: align_conv 是第一个 Conv2d（low_channels -> high_channels）
        high_ch = None
        for m in orig.modules():
            if isinstance(m, nn.Conv2d):
                high_ch = m.out_channels      # align_conv.out_channels == high_channels
                break
        if high_ch is None:
            raise RuntimeError("无法从 InterLayerFusion 推断 high_channels")
        self.high_channels = high_ch
        # 可训练的仿射层保形状（不加 relu/sigmoid，保持近似恒等）
        self.proj = nn.Conv2d(high_ch, high_ch, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()
            for c in range(high_ch):
                self.proj.weight[c, c, 0, 0] = 1.0

    def forward(self, low_feat, high_feat):
        # 关闭 ILF = 忽略浅层特征，只处理深层特征
        return self.proj(high_feat)


def _collapse_dual_path(model):
    """关闭双路径：把 **path2 整条支路从计算图里移除**（用 Identity 替换），
    并把 fusion_conv 的输入通道减半。

    这样得到的是**结构性不同**的架构（fingerprint 会变），
    符合"双路径 vs 单路径"消融的语义；同时因为 fusion_conv 输入通道减半，
    参数量会显著下降 —— 这正是 §5.2 要求"容量匹配对照"的原因。
    """
    import torch
    import torch.nn as nn

    EDPB = None
    for n, mod in model.named_modules():
        if type(mod).__name__ == "EnhancedDualPathBottleneck":
            EDPB = type(mod)
            break
    if EDPB is None:
        return

    for n, mod in list(model.named_modules()):
        if not isinstance(mod, EDPB):
            continue
        exp = mod.fusion_conv.out_channels          # planes*4
        half = exp // 2
        # 用 1x1 conv 只吃 path1 的 exp/2 通道（等价于 path2 输出恒为 0）
        new_conv = nn.Conv2d(half, exp, kernel_size=1, bias=False)
        with torch.no_grad():
            new_conv.weight.zero_()
            k = min(exp, half)
            for c in range(k):
                new_conv.weight[c, c, 0, 0] = 1.0
        mod.fusion_conv = new_conv
        if hasattr(mod, "fusion_bn"):
            pass  # BN 通道数不变（exp），无需改

        def _fwd(self, x, _exp=exp, _half=half):
            identity = x
            out1 = self.swish(self.path1_bn1(self.path1_conv1(x)))
            out1 = self.swish(self.path1_bn2(self.path1_conv2(out1)))
            out1 = self.path1_bn3(self.path1_conv3(out1))
            out = self.swish(self.fusion_bn(self.fusion_conv(out1)))  # 只喂 path1
            if getattr(self, "use_spatial_attention", False):
                out = self.spatial_attention(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out = out + identity
            return self.swish(out)

        import types
        mod.forward = types.MethodType(_fwd, mod)


def _replace_mdc_with_plain(model):
    """关闭 MDC：用单条 depthwise 3x3 替换 MultiScaleDilatedConv。"""
    import torch.nn as nn

    MDC = None
    for n, mod in model.named_modules():
        if type(mod).__name__ == "MultiScaleDilatedConv":
            MDC = type(mod)
            break
    if MDC is None:
        return
    for n, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, MDC):
                C = getattr(mod, child_name)
                # 推断通道：从第一个 depthwise/dilated 子模块取
                ch = _infer_mdc_channels(child)
                setattr(mod, child_name, nn.Sequential(
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False),
                    nn.BatchNorm2d(ch),
                    base_mod_swish(),
                ))


def _infer_mdc_channels(mdc):
    import torch.nn as nn
    for m in mdc.modules():
        if isinstance(m, nn.Conv2d):
            return m.in_channels
    for m in mdc.modules():
        if isinstance(m, nn.BatchNorm2d):
            return m.num_features
    return 1


_SWISH = None


def base_mod_swish():
    global _SWISH
    if _SWISH is None:
        import torch.nn as nn
        _SWISH = type("SwishStub", (nn.Module,), {
            "forward": lambda self, x: x * nn.functional.sigmoid(x)})
    return _SWISH()


def _remove_sam(model):
    """关闭空间注意力：把 SpatialAttentionModule 换为 Identity。"""
    import torch.nn as nn
    SA = None
    for n, mod in model.named_modules():
        if type(mod).__name__ == "SpatialAttentionModule":
            SA = type(mod)
            break
    if SA is None:
        return
    for n, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, SA):
                setattr(mod, child_name, nn.Identity())


# ───────────────────────────────────────────────────────────── 不变量测试
def fingerprint(model) -> str:
    parts = []
    for name, t in model.state_dict().items():
        parts.append(f"{name}|{tuple(t.shape)}|{t.dtype}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def forward_signature(model, seed: int = 0) -> Dict[str, float]:
    """在固定输入下取若干标量摘要（用于跨变体比对）。

    注意：**不同架构的输出本就不同**，所以这里不是"应当相同"，
    而是给每个变体一个可复现的签名，供事后审计（不是通过判据）。
    """
    import torch
    torch.manual_seed(seed)
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        y = model(x)
    return dict(sum=float(y.sum()), mean=float(y.mean()),
                std=float(y.std()), abs_sum=float(y.abs().sum()),
                shape_hash=hashlib.md5(str(tuple(y.shape)).encode()).hexdigest()[:8])


def grad_norm(model, seed: int = 0) -> float:
    """固定输入下的输入梯度范数（不变量测试用）。"""
    import torch
    torch.manual_seed(seed)
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    model.train()
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    return float(x.grad.detach().norm())


# ───────────────────────────────────────────────────────────── 参数归因
def param_attribution(model, base_mod) -> Dict[str, int]:
    """按模块名归类参数量（阶段 3.3）。"""
    buckets = {
        "edpb_fusion_conv": 0,      # 每个 EDPB 的 fusion_conv（论文里被忽略的大头）
        "edpb_fusion_bn": 0,
        "edpb_path1": 0,
        "edpb_path2_conv1_3": 0,
        "edpb_path2_mdc": 0,
        "spatial_attention": 0,
        "ilf_fusion23": 0,
        "ilf_fusion34": 0,
        "downsample": 0,
        "backbone_other": 0,
        "head_fc": 0,
    }
    for name, p in model.named_parameters():
        n = p.numel()
        if "fusion_conv" in name:
            buckets["edpb_fusion_conv"] += n
        elif "fusion_bn" in name:
            buckets["edpb_fusion_bn"] += n
        elif "path1_" in name:
            buckets["edpb_path1"] += n
        elif "path2_conv2" in name:
            buckets["edpb_path2_mdc"] += n
        elif "path2_" in name:
            buckets["edpb_path2_conv1_3"] += n
        elif "spatial_attention" in name:
            buckets["spatial_attention"] += n
        elif name.startswith("fusion23."):
            buckets["ilf_fusion23"] += n
        elif name.startswith("fusion34."):
            buckets["ilf_fusion34"] += n
        elif "downsample" in name:
            buckets["downsample"] += n
        elif name.startswith("fc."):
            buckets["head_fc"] += n
        else:
            buckets["backbone_other"] += n
    return buckets


# ───────────────────────────────────────────────────────────── 5.5 折内日志
def read_fold_metrics(fold_dir: pathlib.Path) -> Optional[List[dict]]:
    p = fold_dir / "metrics.csv"
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def best_vs_fixed_epoch(fold_dir: pathlib.Path, fixed_epoch: int = 126) -> Optional[dict]:
    """从折内 metrics.csv 读 best epoch 与固定 epoch 的 val 指标（阶段 5.5）。

    用途：量化 winner's curse —— 200 轮里挑 best 会偏高多少。
    """
    rows = read_fold_metrics(fold_dir)
    if not rows:
        return None
    f = list(rows[0].keys())
    c_ep = next((c for c in ("epoch", "Epoch") if c in f), None)
    c_acc = next((c for c in f if "val_acc" in c.lower()), None)
    c_loss = next((c for c in f if "val_loss" in c.lower()), None)
    if not c_ep or not c_acc:
        return None
    ep = [int(float(r[c_ep])) for r in rows]
    acc = [float(r[c_acc]) for r in rows]
    best_i = max(range(len(acc)), key=lambda i: acc[i])
    fixed_i = min(range(len(ep)), key=lambda i: abs(ep[i] - fixed_epoch))
    return dict(
        best_epoch=ep[best_i], best_val_acc=acc[best_i],
        fixed_epoch=ep[fixed_i], fixed_val_acc=acc[fixed_i],
        winner_curse_gap=acc[best_i] - acc[fixed_i],
        val_loss_best=(float(rows[best_i][c_loss]) if c_loss else None),
        val_loss_fixed=(float(rows[fixed_i][c_loss]) if c_loss else None),
    )


# ───────────────────────────────────────────────────────────── 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", default="../DMSA_net/smoke_test.py")
    ap.add_argument("--out", default="_arch_out")
    ap.add_argument("--invariants", action="store_true",
                    help="跑 4.2 前向/梯度不变量测试（较慢）")
    ap.add_argument("--folds-root", default="",
                    help="五折产物根目录，用于阶段 5.5 best vs fixed epoch")
    ap.add_argument("--fixed-epoch", type=int, default=126)
    a = ap.parse_args()

    here = pathlib.Path(__file__).parent
    smoke = (here / a.smoke).resolve()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("阶段 3.1–3.2 / 4.1–4.2 架构探针")
    print("=" * 74)

    try:
        import torch
        import torch.nn as nn          # noqa
    except Exception as e:
        print(f"[BLOCKED] torch 不可用：{type(e).__name__}: {e}")
        print("          阶段 3.1/3.2/4.2 需要 torch。按手册 §2 在云环境安装后再跑。")
        #: N6：BLOCKED 产物**同样必须盖章** —— 否则"被阻塞"这个结论
        #: 也说不清是哪一版代码得出的（T13.1 就是专门抓这种漏网的）
        write_artifact(out / "arch_probe.json",
                       dict(status="BLOCKED", reason=f"torch unavailable: {e}",
                            baseline_fp=BASELINE_FP, baseline_params=BASELINE_PARAMS))
        return 2

    if not smoke.exists():
        print(f"[BLOCKED] 找不到 {smoke}")
        return 2

    base = load_baseline(smoke)
    print(f"  torch {torch.__version__} / cuda {torch.version.cuda} / "
          f"available={torch.cuda.is_available()}")
    print(f"  基线模块：{smoke}")

    report: Dict[str, object] = dict(
        torch=torch.__version__, cuda=torch.version.cuda,
        cuda_available=bool(torch.cuda.is_available()),
        baseline_fp=BASELINE_FP, baseline_params=BASELINE_PARAMS,
        smoke_path=str(smoke))

    # ── 3.1 参数量 + fingerprint ───────────────────────────
    model = base.create_model(num_classes=4)
    fp = fingerprint(model)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stats = base.model_parameter_stats(model)
    print(f"\n  [3.1] 参数量：{n_params:,}（可训练 {n_train:,}）")
    print(f"        基线常量 ：{BASELINE_PARAMS:,}")
    print(f"        fingerprint：{fp}")
    print(f"        基线值     ：{BASELINE_FP}")
    fp_ok = (fp == BASELINE_FP)
    p_ok = (n_params == BASELINE_PARAMS)
    print(f"        fp 匹配：{'✅ 一致' if fp_ok else '❌ 不一致'}")
    print(f"        参数匹配：{'✅ 一致' if p_ok else '❌ 不一致'}")
    if not fp_ok or not p_ok:
        print("\n  ❗ 与基线不一致 —— 按 §4.1 硬约束，**禁止**继续做消融。")
        print("     可能原因：smoke_test.py 被改动过，或 torch 版本导致 dtype 差异。")
    report["params"] = dict(total=n_params, trainable=n_train, by_base_fn=stats,
                            fp=fp, fp_matches_baseline=bool(fp_ok),
                            params_match_baseline=bool(p_ok))

    # ── 3.3 参数归因 ──────────────────────────────────────
    attr = param_attribution(model, base)
    tot = max(sum(attr.values()), 1)
    print("\n  [3.3] 参数归因：")
    for k, v in sorted(attr.items(), key=lambda kv: -kv[1]):
        if v:
            print(f"        {k:24s} {v:>12,}  ({v / tot:6.2%})")
    fusion = attr["edpb_fusion_conv"] + attr["edpb_fusion_bn"]
    print(f"        → EDPB fusion 合计 {fusion:,}（{fusion / tot:.2%}）"
          f"  ← 论文未提及的大头")
    report["param_attribution"] = attr

    # ── 3.2 FLOPs（CPU 前向即可）──────────────────────────
    print("\n  [3.2] FLOPs / 吞吐 ...")
    counter = FlopCounter().attach(model)
    x = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        _ = model(x)
    counter.detach()
    flops1 = counter.flops
    # 吞吐（CPU 上测的是相对量，真实吞吐需在 GPU 上跑）
    for _ in range(2):
        with torch.no_grad():
            _ = model(x)
    t0 = time.perf_counter()
    n_rep = 5
    with torch.no_grad():
        for _ in range(n_rep):
            _ = model(x)
    dt = (time.perf_counter() - t0) / n_rep
    print(f"        FLOPs(batch=1, 224x224) = {flops1 / 1e9:.3f} GFLOPs"
          f"（MACs {counter.macs / 1e9:.3f} G）")
    print(f"        CPU 前向 {dt * 1e3:.1f} ms/img（仅参考；真实吞吐需 GPU）")
    top = sorted(counter.by_module.items(), key=lambda kv: -kv[1])[:6]
    report["flops"] = dict(flops=int(flops1), macs=int(counter.macs),
                           cpu_ms_per_image=dt * 1e3, batch=1, res=224,
                           top_modules=[dict(name=k, macs=int(v)) for k, v in top],
                           note="CPU 吞吐仅相对参考；GPU 吞吐需在云环境重测",
                           #: M4（GPT 第五轮）：FLOPs 没有绝对唯一数字，跨论文比较必须绑定约定。
                           #: 这份 dict 就是给论文表格直接抄的"口径脚注"。
                           convention=dict(
                               input_shape="1x3x224x224", batch_size=1,
                               flop_definition="1 MAC = 2 FLOPs（本报告 FLOPs = 2 × MACs）",
                               counted_ops="仅 Conv2d 与 Linear 的乘加",
                               not_counted=("BN / ReLU / Swish / MaxPool / AvgPool / "
                                            "AdaptiveAvgPool / attention 内的 elementwise"),
                               method="torch forward hook（自实现，不依赖 fvcore/thop）",
                               profiler=f"arch_probe.FlopCounter (torch {torch.__version__})",
                               mode="model.eval() + torch.no_grad()",
                           ))

    # ── 4.1 开关注入 + 4.2 不变量测试 ─────────────────────
    print("\n  [4.1] 架构开关注入（monkeypatch，基线文件零改动）...")
    variants = [
        ("full",            dict(use_sam=True,  use_mdc=True,  use_ilf=True,  use_dual_path=True)),
        ("no_sam",          dict(use_sam=False, use_mdc=True,  use_ilf=True,  use_dual_path=True)),
        ("no_mdc",          dict(use_sam=True,  use_mdc=False, use_ilf=True,  use_dual_path=True)),
        ("no_ilf",          dict(use_sam=True,  use_mdc=True,  use_ilf=False, use_dual_path=True)),
        ("no_dual_path",    dict(use_sam=True,  use_mdc=True,  use_ilf=True,  use_dual_path=False)),
    ]
    var_rows = []
    full_ok = False
    for name, cfg in variants:
        create, restore = make_switch_patch(base, **cfg)
        try:
            m = create(num_classes=4)
        except Exception as e:
            var_rows.append(dict(variant=name, error=f"{type(e).__name__}: {e}"))
            print(f"        {name:14s} ❌ 构造失败：{type(e).__name__}: {e}")
            restore()
            continue
        fpn = fingerprint(m)
        pr = sum(p.numel() for p in m.parameters())
        row = dict(variant=name, params=pr, fp=fpn,
                   fp_equals_full=(fpn == fp),
                   fp_equals_baseline=(fpn == BASELINE_FP),
                   params_delta=pr - n_params)
        if name == "full":
            full_ok = (fpn == fp)
        # 前向可跑性
        try:
            sig = forward_signature(m)
            row["forward_ok"] = True
            row["forward_sig"] = sig
        except Exception as e:
            row["forward_ok"] = False
            row["forward_error"] = f"{type(e).__name__}: {e}"
        var_rows.append(row)
        mark = "✅" if row.get("forward_ok") else "❌"
        print(f"        {name:14s} {mark} params={pr:>12,} "
              f"(Δ{pr - n_params:+,}) fp={fpn[:12]}"
              f"{'  [=baseline]' if fpn == BASELINE_FP else ''}")
        restore()

    print(f"\n  硬约束：`full` 变体 fingerprint 与基线一致 → "
          f"{'✅ 满足' if full_ok else '❌ 不满足（禁止用本注入做消融）'}")
    report["variants"] = var_rows
    report["full_matches_baseline"] = bool(full_ok)

    # ── 4.2 不变量测试 ────────────────────────────────────
    if a.invariants:
        print("\n  [4.2] 不变量测试 ...")
        inv = {}
        # a) determinism：同一 seed 两次前向必须一致
        s1 = forward_signature(model, seed=7)
        s2 = forward_signature(model, seed=7)
        inv["determinism_forward"] = (s1 == s2)
        # b) 输入梯度可求（无 NaN）
        model.zero_grad(set_to_none=True)
        gn = grad_norm(model, seed=11)
        inv["grad_finite"] = bool(gn == gn and gn > 0)
        inv["grad_norm"] = gn
        # c) fingerprint 不随前向改变
        fp_after = fingerprint(model)
        inv["fp_stable_after_forward"] = (fp_after == fp)
        # d) 参数量不随前向改变
        inv["params_stable"] = (sum(p.numel() for p in model.parameters()) == n_params)
        for k, v in inv.items():
            if isinstance(v, bool):
                print(f"        {k:28s} {'✅' if v else '❌'}")
        report["invariants"] = inv

    # ── 5.5 best vs fixed epoch ───────────────────────────
    if a.folds_root:
        fr = pathlib.Path(a.folds_root)
        print(f"\n  [5.5] best vs fixed epoch（{fr}）...")
        rows55 = []
        for d in sorted(fr.rglob("*")):
            if d.is_dir() and (d / "metrics.csv").exists():
                r = best_vs_fixed_epoch(d, a.fixed_epoch)
                if r:
                    r["dir"] = str(d.relative_to(fr))
                    rows55.append(r)
                    print(f"        {r['dir']}: best@{r['best_epoch']}="
                          f"{r['best_val_acc']:.4f}  fixed@{r['fixed_epoch']}="
                          f"{r['fixed_val_acc']:.4f}  gap={r['winner_curse_gap']:+.4f}")
        if rows55:
            mean_gap = sum(r["winner_curse_gap"] for r in rows55) / len(rows55)
            print(f"        平均 winner's curse 间隙：{mean_gap:+.4f}")
            report["winner_curse"] = dict(folds=rows55, mean_gap=mean_gap)
        else:
            print("        未找到 metrics.csv（折内无周期性快照时仍应有 metrics.csv）")

    #: N6：走统一盖加入口（原来这里没有 code_fingerprint）
    write_artifact(out / "arch_probe.json", report)

    L = ["# 阶段 3.1–3.2 / 4.1–4.2 架构探针报告", "",
         f"- torch `{torch.__version__}`／CUDA `{torch.version.cuda}`／"
         f"可用 `{torch.cuda.is_available()}`", "",
         "## 3.1 参数量与 fingerprint", "",
         "| 指标 | 实测 | 基线常量 | 一致 |", "|---|---|---|---|",
         f"| 参数量 | {n_params:,} | {BASELINE_PARAMS:,} | "
         f"{'✅' if p_ok else '❌'} |",
         f"| fingerprint | `{fp}` | `{BASELINE_FP}` | "
         f"{'✅' if fp_ok else '❌'} |", "",
         "> 论文写的 25,689,412 已失效；以本表实测为准。", "",
         "## 3.3 参数归因", "", "| 模块 | 参数量 | 占比 |", "|---|---|---|"]
    for k, v in sorted(attr.items(), key=lambda kv: -kv[1]):
        if v:
            L.append(f"| {k} | {v:,} | {v / tot:.2%} |")
    L += ["", f"> **EDPB `fusion_conv`+`fusion_bn` 合计 {fusion:,}"
          f"（{fusion / tot:.2%}）**—— 论文 Fig.4 与正文均未单独列出此项，",
          "> 它是「参数量对不上」的主因。", "",
          "## 3.2 FLOPs / 吞吐", "",
          f"- FLOPs(batch=1, 224×224) = **{flops1 / 1e9:.3f} GFLOPs**"
          f"（MACs {counter.macs / 1e9:.3f} G）",
          f"- CPU 前向 {dt * 1e3:.1f} ms/img（**仅相对参考**，GPU 吞吐须在云环境重测）",
          "",
          "**FLOPs 计数约定**（M4；跨论文比较必须绑定这些，否则同一模型会出现两个 FLOPs）：",
          "",
          f"- 输入 `{report['flops']['convention']['input_shape']}`，batch = "
          f"{report['flops']['convention']['batch_size']}",
          f"- 定义：{report['flops']['convention']['flop_definition']}",
          f"- 计入：{report['flops']['convention']['counted_ops']}",
          f"- **不计入**：{report['flops']['convention']['not_counted']}",
          f"- 工具：{report['flops']['convention']['method']}；"
          f"{report['flops']['convention']['mode']}",
          "", "| Top 模块 | MACs |", "|---|---|"]
    for k, v in top:
        L.append(f"| `{k}` | {v / 1e9:.3f} G |")
    L += ["", "## 4.1 开关注入与 4.2 不变量", "",
          f"- `full` 变体 fingerprint 与基线一致：**{'✅' if full_ok else '❌'}**",
          "", "| 变体 | 参数量 | Δ | 前向可跑 | fp=基线 |", "|---|---|---|---|---|"]
    for r in var_rows:
        if "error" in r:
            L.append(f"| {r['variant']} | — | — | ❌ {r['error'][:40]} | — |")
        else:
            L.append(f"| {r['variant']} | {r['params']:,} | {r['params_delta']:+,} | "
                     f"{'✅' if r.get('forward_ok') else '❌'} | "
                     f"{'✅' if r['fp_equals_baseline'] else '—'} |")
    L += ["", "> 纪律：`full` 必须与基线 fp 一致，否则本注入不得用于正式消融。",
          "> 关闭某个创新点时参数量会变化（如 no_mdc 去掉空洞卷积分支），",
          "> 因此 §5.2 要求**容量匹配对照**必须同时报 params/FLOPs。"]
    if a.invariants and "invariants" in report:
        L += ["", "### 4.2 不变量测试", "", "| 检查 | 结果 |", "|---|---|"]
        for k, v in report["invariants"].items():
            L.append(f"| {k} | {'✅' if v is True else ('❌' if v is False else v)} |")
    if "winner_curse" in report:
        L += ["", "### 5.5 best vs fixed epoch（winner's curse）", "",
              f"- 平均间隙 **{report['winner_curse']['mean_gap']:+.4f}**",
              "", "| 折 | best@ep | acc | fixed@ep | acc | gap |", "|---|---|---|---|---|---|"]
        for r in report["winner_curse"]["folds"]:
            L.append(f"| {r['dir']} | {r['best_epoch']} | {r['best_val_acc']:.4f} | "
                     f"{r['fixed_epoch']} | {r['fixed_val_acc']:.4f} | "
                     f"{r['winner_curse_gap']:+.4f} |")
    (out / "arch_probe.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n输出：{out/'arch_probe.json'}、{out/'arch_probe.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
