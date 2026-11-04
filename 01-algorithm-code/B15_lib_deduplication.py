#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DMSA-Net v2.4 —— 去重 / 谱系重建公共库
=====================================

【v2.4 相对 v2.3 修订的硬缺陷】
  P0-1  汉明距离：v2.3 对 8 个字节**逐字节取最小**（np.minimum），
        导致真实距离 56 的一对被算成 0。正确做法：
        先在同一变换配对内**逐字节求和**（0..64），再在变换配对间取最小。
        两级归约顺序不可颠倒。
  P0-1b 单侧化简 `min(min_T H(A,T·B), min_T H(T·A,B))` **不是代数恒等式**，
        它只是全配对 `min_{i,j}` 的**上界**（GPT 给的 4-bit 反例：0 vs 1）。
        本库默认用**全配对精确解**；`mode="one-sided"` 仅作快速预筛，
        且必须配 `--slack` 回检。
  P0-2a 形状错误：v2.3 的 `one(HA, HB[:, :1])` 让 A 有 nT*8 字节、B 只有 8 字节 → IndexError。
        本库统一 reshape 成 (n, nT, 8) 后再做配对归约。
  P0-2b 缓存：v2.3 用 json.dumps 存 ndarray → TypeError。本库用 .npy + meta.json。
  P0-2c 对角线：v2.3 把 D 对角线填 255 后算组件直径 → 所有非单例组件被误判 chain_risk。
        本库保持 D[i,i]=0（真实自距离），取边只用严格上三角，算直径时排除对角线。

依赖：numpy, pillow（纯 CPU，不需要 GPU / torch / imagehash）
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

__version__ = "2.4.0"

# ---------------------------------------------------------------- 基础常量
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_HASH_SIZE = 8
_HIGHFREQ = 4
_PHASH_INPUT = _HASH_SIZE * _HIGHFREQ  # 32


# ---------------------------------------------------------------- pHash
def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)
    c = np.cos(np.pi * k[None, :] * (2 * k[:, None] + 1) / (2 * n))
    c[0, :] *= np.sqrt(1.0 / n)
    c[1:, :] *= np.sqrt(2.0 / n)
    return c


_DCT32 = _dct_matrix(_PHASH_INPUT).astype(np.float64)


def phash_u64(img: Image.Image) -> int:
    """64-bit 感知哈希。与 imagehash.phash 同算法（8x8 DCT 低频，中位数二值化）。"""
    g = np.asarray(img.convert("L").resize((_PHASH_INPUT, _PHASH_INPUT), Image.BILINEAR),
                   dtype=np.float64)
    d = _DCT32 @ g @ _DCT32.T
    low = d[:_HASH_SIZE, :_HASH_SIZE]
    med = np.median(low[1:, 1:]) if False else np.median(low)
    bits = (low > med).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return int(v)


# ---------------------------------------------------------------- 变换集
@dataclass
class TransformSet:
    """变换集 G。

    默认 **45° 步旋转 × {恒等, 水平翻转}** → |G| = 16。
    实测（v2.3 §1.1.3）：D4（|G|=8）对 15/30/45° 的距离为 16/14/12（全漏检），
    45° 步网格同为 |G|=8（不含镜像）时降到 2/4/0。
    ⚠️ 一旦加上镜像，|G| 变为 16（或 15° 步时为 48），**不是零额外成本**——
    v2.3 文档里"与 D4 同成本"只在不含镜像时成立，此处如实记录。
    """
    angle_step: int = 45
    use_flip: bool = True

    def __post_init__(self):
        if 360 % self.angle_step != 0:
            raise ValueError("angle_step 必须整除 360")
        self.angles: List[int] = list(range(0, 360, self.angle_step))
        self.flips: List[bool] = [False, True] if self.use_flip else [False]

    @property
    def members(self) -> List[Tuple[int, bool]]:
        return [(a, f) for a in self.angles for f in self.flips]

    def __len__(self) -> int:
        return len(self.angles) * len(self.flips)

    def apply(self, im: Image.Image, deg: int, flip: bool) -> Image.Image:
        r = im.rotate(deg, expand=False, resample=Image.BICUBIC)
        return r.transpose(Image.FLIP_LEFT_RIGHT) if flip else r

    def hashes(self, im: Image.Image) -> np.ndarray:
        """返回 (nT,) uint64。保留全部变换，**绝不取整数最小值**。"""
        return np.array([phash_u64(self.apply(im, a, f)) for a, f in self.members],
                        dtype=np.uint64)


# ---------------------------------------------------------------- 汉明距离（核心修正）
def _as_bytes(H: np.ndarray) -> np.ndarray:
    """(n, nT) uint64 -> (n, nT, 8) uint8"""
    H = np.ascontiguousarray(H, dtype=np.uint64)
    return H.view(np.uint8).reshape(H.shape[0], H.shape[1], 8)


def hamming_single(HA: np.ndarray, HB: np.ndarray) -> np.ndarray:
    """(nA,1) x (nB,1) uint64 -> (nA,nB) uint16。字节内求和（这才是 Hamming）。"""
    Ab = _as_bytes(HA)[:, 0, :]
    Bb = _as_bytes(HB)[:, 0, :]
    d = np.zeros((Ab.shape[0], Bb.shape[0]), dtype=np.uint16)
    for k in range(8):
        d += _POP[Ab[:, k][:, None] ^ Bb[None, :, k]].astype(np.uint16)
    return d


def hamming_min_over_T(HA: np.ndarray,
                       HB: np.ndarray,
                       mode: str = "two-sided",
                       block: int = 512,
                       slack: int = 0) -> np.ndarray:
    """变换集上的最小汉明距离。

    正确定义   d(A,B) = min_{i,j} H(h(T_i·A), h(T_j·B))
    归约顺序   先在同一 (i,j) 配对内对 8 个字节**求和**，再在 (i,j) 上取**最小**。

    mode="two-sided"  : 精确全配对，|G|² 次 (nA x nB) 归约。默认。
    mode="one-sided"  : 2·|G| 次归约，返回**上界**；须配合 slack 回检
                        （见 `hamming_two_stage`）。
    """
    if HA.ndim != 2 or HB.ndim != 2:
        raise ValueError("HA/HB 必须是 (n, nT) 二维")
    nA, nT = HA.shape
    nB, nT2 = HB.shape
    if nT != nT2:
        raise ValueError(f"变换集维度不一致: {nT} vs {nT2}")

    Ab = _as_bytes(HA)
    Bb = _as_bytes(HB)
    out = np.full((nA, nB), 255, dtype=np.uint8)

    if mode == "one-sided":
        # 方向 1：A 的恒等变换 vs B 的全部变换
        for j in range(nT):
            np.minimum(out, hamming_single(HA[:, :1], HB[:, j:j + 1]).astype(np.uint8), out=out)
        # 方向 2：A 的全部变换 vs B 的恒等变换
        for i in range(nT):
            np.minimum(out, hamming_single(HA[:, i:i + 1], HB[:, :1]).astype(np.uint8), out=out)
        return out

    if mode != "two-sided":
        raise ValueError("mode 必须是 'two-sided' 或 'one-sided'")

    for s in range(0, nA, block):
        e = min(s + block, nA)
        acc = np.full((e - s, nB), 255, dtype=np.uint8)
        for i in range(nT):
            Ai = Ab[s:e, i, :]                       # (b, 8)
            for j in range(nT):
                Bj = Bb[:, j, :]                     # (nB, 8)
                d = np.zeros((e - s, nB), dtype=np.uint16)
                for k in range(8):
                    d += _POP[Ai[:, k][:, None] ^ Bj[None, :, k]].astype(np.uint16)
                np.minimum(acc, d.astype(np.uint8), out=acc)
        out[s:e] = acc
    return out


def hamming_two_stage(HA: np.ndarray, HB: np.ndarray, threshold: int,
                      slack: int = 2, block: int = 512) -> np.ndarray:
    """两级筛选：先算便宜的单侧上界，只对 `<= threshold + slack` 的候选做精确全配对。

    单侧结果是上界 ⇒ 用它做预筛**不会产生假阴**（真值 ≤ 上界）。
    slack 控制边界安全裕度；实际回检比例会打印出来供审计。
    """
    d1 = hamming_min_over_T(HA, HB, mode="one-sided")
    cand = d1 <= (threshold + slack)
    out = d1
    idx = np.nonzero(cand)
    if len(idx[0]):
        sub = hamming_min_over_T(HA[idx[0]], HB[idx[1]], mode="two-sided", block=block)
        out[idx[0], idx[1]] = np.minimum(sub, 255)
    return out


def hamming_reference(HA: np.ndarray, HB: np.ndarray) -> np.ndarray:
    """朴素双层循环参考实现（仅用于单元测试，慢）。"""
    nA, nT = HA.shape
    nB = HB.shape[0]
    out = np.zeros((nA, nB), dtype=np.uint16)
    for a in range(nA):
        for b in range(nB):
            out[a, b] = min((int(HA[a, i]) ^ int(HB[b, j])).bit_count()
                            for i in range(nT) for j in range(nT))
    return out.astype(np.uint8)


# ---------------------------------------------------------------- 并查集
class UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


# ---------------------------------------------------------------- 缓存（.npy，不是 json）
class HashCache:
    """v2.3 用 json.dumps 存 ndarray 会 TypeError；这里用 .npy + meta.json。

    meta 记录算法版本、变换集、文件 sha256 与大小，防止缓存过期被静默复用。
    """

    def __init__(self, root: pathlib.Path, ts: TransformSet):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ts = ts
        self.tag = hashlib.md5(
            json.dumps([__version__, ts.angle_step, ts.use_flip,
                        _HASH_SIZE, _HIGHFREQ], sort_keys=True).encode()
        ).hexdigest()[:12]

    def _path(self, key: str) -> pathlib.Path:
        h = hashlib.md5(key.encode()).hexdigest()
        return self.root / f"{self.tag}_{h}.npy"

    def get(self, key: str, file_sha: str):
        p = self._path(key)
        m = p.with_suffix(".meta.json")
        if p.exists() and m.exists():
            meta = json.loads(m.read_text(encoding="utf-8"))
            if meta.get("file_sha") == file_sha and meta.get("tag") == self.tag:
                return np.load(p)
        return None

    def put(self, key: str, file_sha: str, value: np.ndarray) -> None:
        p = self._path(key)
        np.save(p, value)
        p.with_suffix(".meta.json").write_text(
            json.dumps({"tag": self.tag, "file_sha": file_sha,
                        "algo": f"phash{_HASH_SIZE}x{_HIGHFREQ}"}, ensure_ascii=False),
            encoding="utf-8")


# ---------------------------------------------------------------- 文件工具
def sha256_file(p: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_config(path: pathlib.Path) -> dict:
    """读取 dedup_config.json。**强校验**：缺关键字段直接失败，不静默 fallback。"""
    cfg = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    required = ["angle_step", "use_flip", "primary_th", "ssim_th",
                "consistency_slack", "distance_mode"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"dedup_config.json 缺少必需字段 {missing}。"
                       f"v2.3 的 A2 用 CFG.get('ssim_th', 0.98) 静默回退，"
                       f"等于把'必须标定'的阈值偷偷写死，已被禁止。")
    if cfg["ssim_th"] is None:
        raise ValueError("ssim_th 未标定（应为 A1-cal 输出），禁止运行。")
    return cfg


# ---------------------------------------------------------------- 可复现性指纹（GPT T7）
# --------------------------------------------------------------------------
# 产物溯源 —— 实现已迁到 lib_provenance.py（纯标准库），此处**转发**以保持兼容
# --------------------------------------------------------------------------
#: 迁移理由：`a0_inventory.py` 是流水线第一步、只依赖标准库，不应被迫装 numpy。
#: 这里保留同名转发，使 `from lib_dedup import code_fingerprint` 仍然可用，
#: 但**唯一的实现**在 lib_provenance 里 —— 绝不允许两处各写一份。
from B16_lib_provenance_audit import (          # noqa: E402  (放在中间是为了紧邻原位置)
    PRODUCER_VERSION, StaleArtifactError,
    code_fingerprint, config_hash, stamp, write_artifact, assert_fresh,
    scan_freshness,
)


class TestAccessLog:
    """`test_access_log`（GPT P1-3 / T7 要求）。

    sealed test 一旦被用于任何调参/选择，独立性就不再成立，
    且**必须留下不可否认的访问记录**。这里把每次访问落成 append-only 的
    JSONL，字段包括时间、用途、操作者、代码/配置指纹与是否"正式评价"。

    纪律：`purpose="tuning"` 出现任意一次，就不得再把该 test 称作 independent test。
    """

    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, purpose: str, *, formal: bool,
            protocol_id: str = "unknown", operator: str = "",
            n_items: int = 0, note: str = "") -> dict:
        import datetime as _dt
        rec = dict(
            utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            purpose=purpose,            # "audit" | "tuning" | "formal_eval" | "sanity"
            formal=bool(formal),        # 是否计为一次正式评价
            protocol_id=protocol_id,
            operator=operator,
            n_items=int(n_items),
            code_fingerprint=code_fingerprint(),
            note=note,
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def is_sealed_intact(self) -> bool:
        """只要出现过 tuning 或 formal_eval，就判定 sealed 已破。"""
        if not self.path.exists():
            return True
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("purpose") in ("tuning", "formal_eval") or r.get("formal"):
                return False
        return True

    def summary(self) -> dict:
        rows = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        return dict(n_accesses=len(rows),
                    sealed_intact=self.is_sealed_intact(),
                    purposes=sorted({r.get("purpose") for r in rows}),
                    n_formal=sum(1 for r in rows if r.get("formal")),
                    log_path=str(self.path))


# ---------------------------------------------------------------- 全局唯一 ID
def global_id(split: str, cls: str, filename: str) -> str:
    """跨 split / 跨类的全局唯一样本 ID。

    v2.3 的 `component=root` 在每个 (split, class) 内从零计数，
    会造成不同类、不同 split 的 ID 碰撞。
    """
    return f"{split}|{cls}|{filename}"


def cluster_id(members: Sequence[str]) -> str:
    """簇 ID = 字典序最小的成员 global_id（确定性，且跨 split/class 不碰撞）。"""
    return "C:" + min(members)


# ---------------------------------------------------------------- 直径（排除对角线）
def component_diameter(D: np.ndarray, members: Sequence[int]) -> int:
    """组件直径。**排除对角线**（v2.3 把对角线填 255 后算 max，导致所有非单例
    组件都被误判 chain_risk —— 连字节级完全相同的一对也逃不掉）。"""
    if len(members) < 2:
        return 0
    sub = D[np.ix_(members, members)].astype(np.uint16).copy()
    np.fill_diagonal(sub, 0)
    return int(sub.max())


# ---------------------------------------------------------------- 证据分级
S_PROVEN = "S1"          # 可证明同源（发布者 lineage / 字节级相同）
S_INVARIANT = "S2"       # 变换不变 + 组件内部一致
S_BOUNDARY = "S3"        # 阈值边界
S_UNRESOLVED = "unresolved"
S_SINGLETON = "singleton"  # 未匹配到任何同源候选 → **自身即一个源**（纳入主统计）

#: 进入主统计的置信等级。**singleton 必须包含**——
#: v2.3 写"主统计只用 S1∪S2"会把 9,328/10,240 张单例整体丢掉
#: （No Impairment 全部 2,560 张都不在已检测重复组里），使四类主指标无法计算。
MAIN_STAT_LEVELS = (S_PROVEN, S_INVARIANT, S_SINGLETON)


# ============================================================ 极简表格层（零 pandas 依赖）
# 原因：受限 PyPI 镜像下 pandas 不可得；同时让用户云环境只需 numpy+pillow(+scikit-image)。
class Mask(list):
    """布尔掩码。必须支持 `&` / `|` / `~`：
    `list` 本身不定义这些运算符，而 pandas 风格的
    `df[(df["a"] >= 1) & (df["b"] > 0)]` 会直接 TypeError。
    （v2.4 实测：a1cal_calibrate.py 曾因此完全无法运行。）"""

    def any(self) -> bool:
        return any(self)

    def all(self) -> bool:
        return all(self)

    def tolist(self) -> list:
        return [bool(x) for x in self]

    @staticmethod
    def _coerce(other):
        if isinstance(other, Mask):
            return list(other)
        if isinstance(other, (list, tuple)):
            return list(other)
        return [bool(other)]      # 标量广播

    def __and__(self, other):
        o = self._coerce(other)
        if len(o) == 1:
            o = o * len(self)
        return Mask([bool(a) and bool(b) for a, b in zip(self, o)])

    __rand__ = __and__

    def __or__(self, other):
        o = self._coerce(other)
        if len(o) == 1:
            o = o * len(self)
        return Mask([bool(a) or bool(b) for a, b in zip(self, o)])

    __ror__ = __or__

    def __invert__(self):
        return Mask([not bool(a) for a in self])


class Column:
    """最小列对象，只实现本项目用到的操作。"""

    def __init__(self, values):
        self.v = list(values)

    def __len__(self):
        return len(self.v)

    def __iter__(self):
        return iter(self.v)

    def __getitem__(self, i):
        return self.v[i]

    def tolist(self):
        return list(self.v)

    def notna(self) -> bool:
        return all(x is not None and x == x for x in self.v)

    def nunique(self) -> int:
        return len(set(self.v))

    @property
    def is_unique(self) -> bool:
        return len(set(self.v)) == len(self.v)

    def map(self, fn):
        if isinstance(fn, dict):                       # 兼容 pandas 的 dict 映射语义
            return Column([fn.get(x) for x in self.v])
        return Column([fn(x) if x is not None else None for x in self.v])

    def fillna(self, other):
        """填充缺失值。**支持标量**（v2.4 实测：原实现只支持 iterable，
        `fillna(0)` 会 `TypeError: 'int' object is not iterable`）。"""
        if isinstance(other, (list, tuple)):
            o = list(other)
            return Column([(o[i] if (x is None) else x) for i, x in enumerate(self.v)])
        return Column([(other if (x is None or x != x) else x) for x in self.v])

    def max(self):
        vals = [x for x in self.v if x is not None]
        return max(vals) if vals else None

    def min(self):
        vals = [x for x in self.v if x is not None]
        return min(vals) if vals else None

    def sum(self):
        return sum(x for x in self.v if x is not None)

    def astype(self, typ):
        return Column([(None if x is None else typ(x)) for x in self.v])

    def _cmp(self, other, op):
        try:
            return Mask([(x is not None) and op(x, other) for x in self.v])
        except TypeError:
            return Mask([False] * len(self.v))

    def __gt__(self, o):
        return self._cmp(o, lambda a, b: a > b)

    def __ge__(self, o):
        return self._cmp(o, lambda a, b: a >= b)

    def __lt__(self, o):
        return self._cmp(o, lambda a, b: a < b)

    def __le__(self, o):
        return self._cmp(o, lambda a, b: a <= b)

    def __eq__(self, o):
        return self._cmp(o, lambda a, b: a == b)

    def __ne__(self, o):
        return self._cmp(o, lambda a, b: a != b)

    __hash__ = None


class Table:
    """最小表对象：dict 列表 + 列选择 + 布尔掩码 + CSV 落盘。"""

    def __init__(self, rows):
        self.rows = [dict(r) for r in rows]

    def __len__(self):
        return len(self.rows)

    @property
    def empty(self) -> bool:
        return not self.rows

    @property
    def columns(self):
        return list(self.rows[0].keys()) if self.rows else []

    def __getitem__(self, key):
        if isinstance(key, str):
            return Column([r.get(key) for r in self.rows])
        if isinstance(key, Mask):                      # 必须排在 list 之前：Mask 是 list 子类
            return Table([r for r, m in zip(self.rows, key) if m])
        if isinstance(key, (list, tuple)):
            return Table([{k: r.get(k) for k in key} for r in self.rows])
        return Table([r for r, m in zip(self.rows, list(key)) if m])

    def __setitem__(self, key: str, values):
        vals = list(values)
        if len(vals) != len(self.rows):
            raise ValueError("赋值长度与行数不一致")
        for r, v in zip(self.rows, vals):
            r[key] = v

    def drop_duplicates(self, subset, keep: str = "last"):
        seen, out = {}, []
        for i, r in enumerate(self.rows):
            k = tuple(r.get(s) for s in subset)
            seen.setdefault(k, []).append(i)
        keep_idx = set()
        for k, idxs in seen.items():
            keep_idx.add(idxs[-1] if keep == "last" else idxs[0])
        for i, r in enumerate(self.rows):
            if i in keep_idx:
                out.append(r)
        return Table(out)

    def reset_index(self, drop: bool = True):
        return Table(self.rows)

    def to_string(self, index: bool = False) -> str:
        """极简文本渲染（v2.4 实测：原实现缺此方法，a1cal 的 print 会崩）。"""
        cols = self.columns
        if not cols:
            return "(empty table)"
        cells = [[("" if r.get(c) is None else str(r.get(c))) for c in cols]
                 for r in self.rows]
        w = [max(len(c), *(len(row[i]) for row in cells)) if cells else len(c)
             for i, c in enumerate(cols)]
        line = "  ".join(c.ljust(w[i]) for i, c in enumerate(cols))
        sep = "  ".join("-" * w[i] for i in range(len(cols)))
        body = ["  ".join(row[i].ljust(w[i]) for i in range(len(cols))) for row in cells]
        return "\n".join([line, sep] + body)

    def to_csv(self, path, index: bool = False, encoding: str = "utf-8-sig"):
        import csv
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding=encoding) as f:
            w = csv.DictWriter(f, fieldnames=self.columns)
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k) for k in self.columns})
        return path
