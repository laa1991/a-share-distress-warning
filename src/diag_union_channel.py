"""把诉讼当**第二条通道**：模型名单 ∪ 诉讼名单，覆盖率与精确率各变成多少？

上一格的结论是「事件特征加进模型几乎不加分（ΔAUC +0.0014）」。但那是**把两种信息混在一张表里**。
产品上还有另一种用法：**两条名单并列**——模型名单（前 10%）管"分数高的"，诉讼名单管"有官司的"，
谁进了哪条、由谁负责跟进，是业务自己决定。这一格要量的是：

- 并集的**覆盖率**：亏损公司里有多少至少进了一条名单（模型抓不到的那一半，诉讼能补多少）
- 并集的**精确率**：名单里有多少是真亏的（多抓的代价）
- **增量那一块**（只在诉讼名单、不在模型名单里）：它自己的精确率是多少 ⇒ 值不值得多派人复核

口径：同批样本、同一套折；名单规模按**同年内**前 10%（与产品视角那页一致）。

用法: python diag_union_channel.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_events_ablation import run  # noqa: E402
from events_features import build as build_events  # noqa: E402
from run_experiments import CAT, FEATURES  # noqa: E402
from run_leadtime_curve import base_label, samples_at_quarter  # noqa: E402


def summarize(g: pd.DataFrame, mask_model, mask_sue, tag_model="模型前10%", tag_sue="有诉讼") -> dict:
    n, npos = len(g), int(g["label"].sum())
    m, s = mask_model, mask_sue
    u = m | s
    only_s = s & ~m
    return {
        "n": n, "n_pos": npos,
        "model": {"size": int(m.sum()), "share": round(float(m.mean()), 4),
                  "precision": round(float(g.loc[m, "label"].mean()), 4) if m.any() else None,
                  "coverage": round(float(g.loc[m, "label"].sum() / npos), 4)},
        "sue": {"size": int(s.sum()), "share": round(float(s.mean()), 4),
                "precision": round(float(g.loc[s, "label"].mean()), 4) if s.any() else None,
                "coverage": round(float(g.loc[s, "label"].sum() / npos), 4)},
        "union": {"size": int(u.sum()), "share": round(float(u.mean()), 4),
                  "precision": round(float(g.loc[u, "label"].mean()), 4) if u.any() else None,
                  "coverage": round(float(g.loc[u, "label"].sum() / npos), 4)},
        "only_sue": {"size": int(only_s.sum()),
                     "precision": round(float(g.loc[only_s, "label"].mean()), 4) if only_s.any() else None,
                     "adds_true": int(g.loc[only_s, "label"].sum()),
                     "adds_false": int(only_s.sum() - g.loc[only_s, "label"].sum())},
    }


def main() -> int:
    panel = pd.read_pickle(ROOT / "data" / "panel.pkl")
    ann = base_label(panel)
    s = samples_at_quarter(panel, 1, ann)
    ev = build_events()
    s = s.merge(ev, on=["code", "year"], how="left")
    for c in ("sue_prev", "sue_q1", "sue_any", "gua_heavy"):
        s[c] = s[c].fillna(0.0)

    _, pred = run(s, FEATURES + CAT)                    # A 臂：只用财报
    pred = pred.merge(ev[["code", "year", "sue_any", "gua_heavy"]], on=["code", "year"], how="left")
    for c in ("sue_any", "gua_heavy"):
        pred[c] = pred[c].fillna(0.0)

    rows, per_year = [], {}
    for Y, g in pred.groupby("year"):
        g = g.copy()
        k = max(int(round(0.10 * len(g))), 1)
        threshold = g["p"].nlargest(k).min()
        mask_model = g["p"] >= threshold
        mask_sue = g["sue_any"] > 0
        st = summarize(g, mask_model, mask_sue)
        per_year[int(Y)] = st
        rows.append({"年": int(Y), **{f"模型{kk}": vv for kk, vv in st["model"].items()},
                     **{f"诉讼{kk}": vv for kk, vv in st["sue"].items()},
                     **{f"并集{kk}": vv for kk, vv in st["union"].items()},
                     "增量块": st["only_sue"]["size"], "增量块精确率": st["only_sue"]["precision"],
                     "多抓真亏": st["only_sue"]["adds_true"], "多带误报": st["only_sue"]["adds_false"]})

    tab = pd.DataFrame(rows)
    print("逐年（测试年 2022–2025，① 一季报那一格）：")
    print(tab[["年", "模型size", "模型precision", "模型coverage", "诉讼size", "诉讼precision",
               "并集size", "并集precision", "并集coverage", "增量块", "增量块精确率", "多抓真亏", "多带误报"]]
          .to_string(index=False))

    agg = tab[["模型coverage", "诉讼coverage", "并集coverage", "模型precision", "诉讼precision", "并集precision",
               "多抓真亏", "多带误报"]].mean()
    print("\n四年的平均：")
    print(f"  模型前 10%：覆盖率 {agg['模型coverage']:.2%} · 精确率 {agg['模型precision']:.2%}")
    print(f"  诉讼名单　：覆盖率 {agg['诉讼coverage']:.2%} · 精确率 {agg['诉讼precision']:.2%}")
    print(f"  并集　　　：覆盖率 {agg['并集coverage']:.2%}（{agg['并集coverage'] - agg['模型coverage']:+.2%}）"
          f" · 精确率 {agg['并集precision']:.2%}（{agg['并集precision'] - agg['模型precision']:+.2%}）")
    print(f"  增量块（只在诉讼名单里）：平均每年多抓 {agg['多抓真亏']:.0f} 家真亏、多带 "
          f"{agg['多带误报']:.0f} 家误报 ⇒ 一块换一块")

    (ROOT / "data" / "union_channel.json").write_text(
        json.dumps({"per_year": per_year, "mean": {k: round(float(v), 4) for k, v in agg.items()}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n读数 -> data/union_channel.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
