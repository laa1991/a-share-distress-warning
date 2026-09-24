"""ST 标签下的负对照：打乱标签 5 个种子，看 AUC 分布落在哪。

为什么要单独跑：`run_experiments_st.py` 那两个种子给出 0.42 —— **低于 0.5**，与「打乱标签应该回到 0.5」的直觉不符。
样本只有 40–100 个正例，先量分布再说话；量完如果还是偏，就照实写「未解释」，不硬圆。

用法: python diag_st_negative_control.py [--seeds 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import CAT, FEATURES, PANEL, fit_predict, metrics, make_samples  # noqa: E402
from st_label import build as build_st  # noqa: E402

TEST_YEARS = [2022, 2023, 2024, 2025]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    panel = pd.read_pickle(PANEL)
    s = make_samples(panel)
    s = s[s["label"].notna() & s["t_decision"].notna() & s["t_label"].notna()].copy()
    _, first = build_st()
    s = s.merge(first, on="code", how="left")
    d = (s["st_first"] - pd.to_datetime(s["t_label"], errors="coerce")).dt.days
    for col, (lo, hi) in (("label_st_90", (0, 90)), ("label_st_y1", (0, 365))):
        s[col] = ((d >= lo) & (d <= hi)).astype(float)

    out = {}
    for ycol in ("label", "label_st_90", "label_st_y1"):
        sub = s[s["year"].isin(TEST_YEARS)]
        aucs = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            rows = []
            for Y in TEST_YEARS:
                tr, va, te = s[s["year"] <= Y - 2].copy(), s[s["year"] == Y - 1], s[s["year"] == Y]
                if min(len(tr), len(va), len(te)) == 0:
                    continue
                tr["label"] = rng.permutation(tr[ycol].values)     # ★ 打乱的是训练标签
                p, _ = fit_predict(tr, va, te, FEATURES + CAT)
                rows.append((te["year"].values, te[ycol].values, p))
            y = np.concatenate([r[1] for r in rows])
            p = np.concatenate([r[2] for r in rows])
            per_year = [metrics(r[1], r[2])["auc"] for r in rows if len(np.unique(r[1])) > 1]
            aucs.append({"pooled": metrics(y, p)["auc"], "macro": float(np.mean(per_year))})
        pooled = [a["pooled"] for a in aucs]
        macro = [a["macro"] for a in aucs]
        out[ycol] = {"pooled": [round(a, 4) for a in pooled],
                     "macro": [round(a, 4) for a in macro],
                     "pooled_mean": round(float(np.mean(pooled)), 4), "pooled_sd": round(float(np.std(pooled)), 4),
                     "macro_mean": round(float(np.mean(macro)), 4), "macro_sd": round(float(np.std(macro)), 4),
                     "positives": int(sub[ycol].sum()), "n": int(len(sub))}
        print(f"{ycol:14s} 正例 {out[ycol]['positives']:5d} / {out[ycol]['n']}："
              f"pooled {out[ycol]['pooled_mean']:.4f} ± {out[ycol]['pooled_sd']:.4f} · "
              f"**macro {out[ycol]['macro_mean']:.4f} ± {out[ycol]['macro_sd']:.4f}**")
        print(f"                macro per-seed {out[ycol]['macro']}")

    (ROOT / "data" / "st_negative_control.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n读数 -> data/st_negative_control.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
