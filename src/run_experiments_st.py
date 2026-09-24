"""换标签：把「当年年报亏损」换成「被实施风险警示（*ST / ST）」，同一条流水线重跑一遍。

只跑三格（够回答问题）：**A 诚实** · **基线（上年是否亏损）** · **D 负对照（标签打乱，2 个种子）**。
两套标签并排报，外加业务读数（前 10% 名单里有多少真会戴帽、相对基准率几倍）。

用法: python run_experiments_st.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import (CAT, FEATURES, PANEL, fit_predict, metrics,  # noqa: E402
                             pooled_and_macro, run_walkforward)
from run_experiments import make_samples  # noqa: E402
from st_label import build as build_st  # noqa: E402

TEST_YEARS = [2022, 2023, 2024, 2025]


def attach_label(s: pd.DataFrame, first: pd.DataFrame) -> pd.DataFrame:
    s = s.merge(first, on="code", how="left")
    t_label = pd.to_datetime(s["t_label"], errors="coerce")
    d = (s["st_first"] - t_label).dt.days
    s["label_st_90"] = ((d >= 0) & (d <= 90)).astype(float)
    s["label_st_y1"] = ((d >= 0) & (d <= 365)).astype(float)
    return s


def walkforward(s: pd.DataFrame, feats: list[str], ycol: str) -> dict:
    s = s.copy()
    s["label"] = s[ycol]
    rows = []
    for Y in TEST_YEARS:
        tr, va, te = s[s["year"] <= Y - 2], s[s["year"] == Y - 1], s[s["year"] == Y]
        if min(len(tr), len(va), len(te)) == 0 or tr["label"].nunique() < 2:
            continue
        p, _ = fit_predict(tr, va, te, feats)
        rows.append((te["year"].values, te["label"].values, p))
        m = metrics(te["label"].values, p)
        per_y = {"auc": m["auc"], "n": len(te), "pos": int(te["label"].sum()),
                 "%": round(float(te["label"].mean()), 4)}
        print(f"    {Y}: n={per_y['n']:5d} 正例 {per_y['pos']:4d}（{per_y['%']:.2%}） AUC {per_y['auc']:.4f}")
    return pooled_and_macro(rows) | {"rows": rows}


def biz_top10(rows) -> dict:
    out = []
    for year, y, p in rows:
        k = max(int(round(0.10 * len(y))), 1)
        idx = np.argsort(-p)[:k]
        out.append({"year": int(year[0]), "base": float(y.mean()),
                    "top10_precision": float(y[idx].mean()),
                    "lift": float(y[idx].mean() / max(y.mean(), 1e-9)),
                    "recall": float(y[idx].sum() / max(y.sum(), 1))})
    d = pd.DataFrame(out)
    return {"per_year": out, "precision_mean": round(float(d["top10_precision"].mean()), 4),
            "lift_mean": round(float(d["lift"].mean()), 3), "recall_mean": round(float(d["recall"].mean()), 4)}


def main() -> int:
    panel = pd.read_pickle(PANEL)
    s = make_samples(panel)
    s = s[s["label"].notna() & s["t_decision"].notna() & s["t_label"].notna()]
    _, first = build_st()
    s = attach_label(s, first)
    t = s[s["year"].isin(TEST_YEARS)]
    print(f"样本 {len(s)}（测试年 {len(t)}）· 损失标签正例率 {t['label'].mean():.2%} · "
          f"ST(90天) {t['label_st_90'].mean():.2%} · ST(一年) {t['label_st_y1'].mean():.2%}")

    res = {}
    for ycol, name in (("label", "① 当年年报亏损"), ("label_st_90", "② 90 天内被实施风险警示"),
                       ("label_st_y1", "③ 一年内被实施风险警示")):
        print(f"\n=== {name} ===")
        wf = walkforward(s, FEATURES + CAT, ycol)
        b = biz_top10(wf["rows"])
        # 基线：上年是否亏损（直接当分数）
        base_rows = []
        for Y in TEST_YEARS:
            te = s[s["year"] == Y]
            base_rows.append((te["year"].values, te[ycol].values, te["prev_loss"].values))
        base = pooled_and_macro(base_rows)
        # 负对照：标签打乱（2 个种子）
        ds = []
        for seed in (7, 77):
            r = []
            for Y in TEST_YEARS:
                tr, va, te = s[s["year"] <= Y - 2], s[s["year"] == Y - 1], s[s["year"] == Y]
                if min(len(tr), len(va), len(te)) == 0 or tr[ycol].nunique() < 2:
                    continue
                tr = tr.copy()
                tr["label"] = tr[ycol]
                p, _ = fit_predict(tr, va, te, FEATURES + CAT, shuffle=True)
                r.append((te["year"].values, te[ycol].values, p))
            ds.append(pooled_and_macro(r)["pooled"]["auc"])
        res[name] = {"A": {"pooled": wf["pooled"], "macro_auc": wf["macro_auc"], "biz": b},
                     "baseline_prev_loss": {"pooled": base["pooled"], "macro_auc": base["macro_auc"]},
                     "neg_control_auc": ds}
        print(f"  A 诚实：pooled AUC {wf['pooled']['auc']} · macro {wf['macro_auc']} · "
              f"top10 精确率 {b['precision_mean']:.2%}（基准率 {np.mean([x['base'] for x in b['per_year']]):.2%}）· "
              f"lift {b['lift_mean']:.2f}× · 召回 {b['recall_mean']:.2%}")
        print(f"  基线（上年是否亏损）：pooled {base['pooled']['auc']} · macro {base['macro_auc']}")
        print(f"  负对照（打乱标签）：{ds}")

    (ROOT / "data" / "results_st_label.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                                                         encoding="utf-8")
    print("\n读数 -> data/results_st_label.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
