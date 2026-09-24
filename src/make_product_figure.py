"""产品视角的一张图：**愿意提前多久 → 你能拿到什么样的名单**（中/英各出一张）。

同一份逐样本分数（`data/leadtime_scores.csv`）与提前量曲线（`data/leadtime_curve.json`），换一个问法：
- 名单质量：榜单前 X% 里真会亏的比例（精确率）
- 覆盖面：所有会亏的公司里有多少进了榜单（召回率）
- 误报：没亏的公司里有多少被误抓

口径（与其它装置一致，都是踩过才知道的）：
- **分年算、再平均（macro）**：每个阶段是各自一个模型，跨年分数不在同一把尺上 ⇒ 名单只能按「**同年内**排前 X%」切。
- 名单规模扫 1%–50%，不只报 10% 一个点 —— 取舍整条曲线才是产品，一个点是结论。
- ④ 业绩预告那格是**子集**（当年发了预告的公司，正例率 43.8%），图上标注，不与前三格混读。

用法: python make_product_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, FIGS = ROOT / "data", ROOT / "docs" / "figs"

STAGES = ["① 一季报披露", "② 半年报披露", "③ 三季报披露", "④ 业绩预告公布"]
COLORS = {"① 一季报披露": "#1f77b4", "② 半年报披露": "#2ca02c",
          "③ 三季报披露": "#ff7f0e", "④ 业绩预告公布": "#d62728"}
SIZES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]

L = {
    "zh": {
        "font": ["Microsoft YaHei", "SimHei"],
        "short": {"① 一季报披露": "一季报\n4 月 · 361 天", "② 半年报披露": "半年报\n8 月 · 239 天",
                  "③ 三季报披露": "三季报\n10 月 · 177 天", "④ 业绩预告公布": "业绩预告\n次年 1 月 · 86 天"},
        "leg_short": {"① 一季报披露": "一季报（361 天）", "② 半年报披露": "半年报（239 天）",
                      "③ 三季报披露": "三季报（177 天）", "④ 业绩预告公布": "预告（86 天，子集）"},
        "t1": "愿意等到哪一步，能拿到什么（名单取前 10%）",
        "t2": "名单放大能多抓多少：取舍前沿",
        "l_auc": "AUC（排序能力）", "l_prec": "前 10% 名单的精确率",
        "l_cov": "前 10% 名单的覆盖率（抓住了多少亏损公司）", "l_fa": "误报率（没亏却被抓）",
        "x2": "名单规模（占全体样本的比例）", "y2": "覆盖率（抓住了多少亏损公司）",
        "cheat": "偷看一期能拿到的 AUC {:.3f}", "out": "fig_product.png",
    },
    "en": {
        "font": ["Segoe UI", "DejaVu Sans"],
        "short": {"① 一季报披露": "Q1 report\nApr · 361d", "② 半年报披露": "H1 report\nAug · 239d",
                  "③ 三季报披露": "Q3 report\nOct · 177d", "④ 业绩预告公布": "Pre-ann.\nJan · 86d"},
        "leg_short": {"① 一季报披露": "Q1 (361 d)", "② 半年报披露": "H1 (239 d)",
                      "③ 三季报披露": "Q3 (177 d)", "④ 业绩预告公布": "Pre-ann. (86 d, subset)"},
        "t1": "How late are you willing to decide? (list = top 10%)",
        "t2": "Growing the list buys more coverage: the trade-off frontier",
        "l_auc": "AUC (ranking power)", "l_prec": "Precision of the top-10% list",
        "l_cov": "Coverage of the top-10% list (share of losers caught)",
        "l_fa": "False alarms (non-losers caught)",
        "x2": "List size (share of all samples)", "y2": "Coverage (share of losers caught)",
        "cheat": "AUC reachable by peeking one step ahead: {:.3f}", "out": "fig_product_en.png",
    },
}


def stage_curve(sc: pd.DataFrame, stage: str, sizes=SIZES) -> dict:
    sub = sc[sc["stage"] == stage]
    rows = {s: {"precision": [], "coverage": [], "false_alarm": []} for s in sizes}
    for _y, g in sub.groupby("year"):
        y = g["label"].values
        order = np.argsort(-g["score"].values)
        pos_total = y.sum()
        n_neg = max(len(y) - pos_total, 1)
        for s in sizes:
            k = max(int(round(s * len(y))), 1)
            tp = float(y[order[:k]].sum())
            rows[s]["precision"].append(tp / k)
            rows[s]["coverage"].append(tp / max(pos_total, 1))
            # 误报率 = 被抓进来的负例 / 全部负例（**不是** tp/n_neg —— 那是我第一版写错的式子）
            rows[s]["false_alarm"].append((k - tp) / n_neg)
    return {"stage": stage, "n": int(len(sub)), "pos_rate": float(sub["label"].mean()),
            "by_size": {f"{s:.2f}": {k: round(float(np.mean(v)), 4) for k, v in d.items()}
                        for s, d in rows.items()}}


def draw(lang: str, curves: dict, lead: dict, auc: dict, cheat: dict) -> None:
    t = L[lang]
    plt.rcParams["font.sans-serif"] = t["font"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    ax = axes[0]
    xs = np.arange(len(STAGES))
    ax.plot(xs, [auc[s] for s in STAGES], "o-", color="#111", lw=2, label=t["l_auc"])
    ax.plot(xs, [curves[s]["by_size"]["0.10"]["precision"] for s in STAGES], "s-",
            color="#7f7f7f", lw=1.6, label=t["l_prec"])
    ax.plot(xs, [curves[s]["by_size"]["0.10"]["coverage"] for s in STAGES], "^-",
            color="#d62728", lw=2, label=t["l_cov"])
    ax.plot(xs, [curves[s]["by_size"]["0.10"]["false_alarm"] for s in STAGES], "v--",
            color="#1f77b4", lw=1.4, label=t["l_fa"])
    ax.axhline(cheat["auc"], color="#8c564b", ls=":", lw=1.3)
    ax.annotate(t["cheat"].format(cheat["auc"]), xy=(0.02, cheat["auc"]), xytext=(0.42, 0.10),
                textcoords="axes fraction", fontsize=8.5, color="#8c564b")
    for i, s in enumerate(STAGES):
        ax.annotate(f"{auc[s]:.3f}", (i, auc[s]), textcoords="offset points", xytext=(0, -13),
                    ha="center", fontsize=8.5)
        ax.annotate(f"{curves[s]['by_size']['0.10']['coverage']:.1%}",
                    (i, curves[s]["by_size"]["0.10"]["coverage"]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8.5, color="#d62728")
    ax.set_xticks(xs); ax.set_xticklabels([t["short"][s] for s in STAGES], fontsize=8.5)
    ax.set_ylim(0, 1.06); ax.set_ylabel("0–1")
    ax.set_title(t["t1"], fontsize=10.5)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="center right")

    ax = axes[1]
    for s in STAGES:
        cov = [curves[s]["by_size"][f"{x:.2f}"]["coverage"] for x in SIZES]
        ax.plot([x * 100 for x in SIZES], cov, "o-", color=COLORS[s], lw=1.8, ms=3.5,
                label=t["leg_short"][s])
    ax.set_xlabel(t["x2"]); ax.set_ylabel(t["y2"]); ax.set_title(t["t2"], fontsize=10.5)
    ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="lower right")

    fig.tight_layout(); fig.savefig(FIGS / t["out"], dpi=150); plt.close(fig)


def main() -> int:
    sc = pd.read_csv(DATA / "leadtime_scores.csv", dtype={"code": str})
    lt = json.loads((DATA / "leadtime_curve.json").read_text(encoding="utf-8"))
    res = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
    lead = {s: lt[s]["lead_days_median"] for s in STAGES}
    auc = {s: lt[s]["pooled"]["auc"] for s in STAGES}
    curves = {s: stage_curve(sc, s) for s in STAGES}
    cheat = {"auc": res["B_lookahead_next_report"]["pooled"]["auc"],
             "top10_precision": res["B_lookahead_next_report"]["pooled"]["top10_rate"]}
    (DATA / "product_curve.json").write_text(
        json.dumps({"lead_days": lead, "auc": auc, "curves": curves,
                    "cheat_reference": {"arm": "B 偷看未来一期（4 月决策）", **cheat}},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'阶段':16s} {'提前':>6s} {'AUC':>6s} {'前10%精确':>9s} {'前10%覆盖':>9s} "
          f"{'前10%误报':>9s} {'前20%覆盖':>9s} {'前50%覆盖':>9s}")
    for s in STAGES:
        b = curves[s]["by_size"]
        print(f"{s:16s} {lead[s]:6.0f} {auc[s]:6.4f} {b['0.10']['precision']:9.4f} "
              f"{b['0.10']['coverage']:9.4f} {b['0.10']['false_alarm']:9.4f} "
              f"{b['0.20']['coverage']:9.4f} {b['0.50']['coverage']:9.4f}")

    for lang in ("zh", "en"):
        draw(lang, curves, lead, auc, cheat)
    print("\n图 -> docs/figs/fig_product.png · fig_product_en.png · 读数 -> data/product_curve.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
