"""英文版的两张图（给英文一页纸用）—— 不动已有的中文出图脚本，单独出 `*_en.png`。

画的是同两份读数：提前量曲线（`data/leadtime_curve.json`）与信号时间线（逐样本分数 `data/leadtime_scores.csv`）。
口径与中文版完全一致（阶段内分位、公共样本内排名、第 90 百分位阈值）—— 换的只有字。

用法: python make_figures_en.py
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
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

S1, S2, S3, S4 = "① 一季报披露", "② 半年报披露", "③ 三季报披露", "④ 业绩预告公布"
EN_SHORT = {S1: "Q1\nApr", S2: "H1\nAug", S3: "Q3\nOct", S4: "Pre-ann.\nJan"}


def fig_leadtime() -> None:
    lt = json.loads((DATA / "leadtime_curve.json").read_text(encoding="utf-8"))
    res = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
    order = ["① 一季报披露", "② 半年报披露", "③ 三季报披露", "④ 业绩预告公布", "⑥ 年报披露（=答案）"]
    names = {order[0]: "Q1 report", order[1]: "H1 report", order[2]: "Q3 report",
             order[3]: "Pre-ann.", order[4]: "Annual (= answer)"}
    x = [lt[k]["lead_days_median"] for k in order]
    a = [lt[k]["pooled"]["auc"] for k in order]
    b_arm = res["B_lookahead_next_report"]["pooled"]["auc"]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(x, a, "o-", color="#1f77b4", lw=2)
    for xi, yi, k in zip(x, a, order):
        high = yi > 0.98            # 顶上的两个点把标签放到点下方，免得撞标题
        ax.annotate(f"{yi:.3f}\n{names[k]}", (xi, yi), textcoords="offset points",
                    xytext=(0, -18 if high else 9), ha="center",
                    va="top" if high else "bottom", fontsize=9)
    ax.axhline(b_arm, color="#d62728", ls="--", lw=1.2,
               label=f"B peeking one step ahead (Apr decision) AUC={b_arm:.3f}")
    ax.set_ylim(min(a) - 0.012, 1.035)
    ax.annotate("peeking ≈ waiting until August\n(what is stolen is lead time, not a model)",
                xy=(0.35, 0.30), xycoords="axes fraction", fontsize=9, color="#d62728")
    ax.set_xlabel("Days before the annual report is filed (smaller = later, more information)")
    ax.set_ylabel("AUC")
    ax.set_title("Lead time → discrimination: same target, only the decision date moves")
    ax.invert_xaxis(); ax.grid(alpha=0.25); ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "fig_leadtime_en.png", dpi=150); plt.close(fig)


def fig_timeline() -> None:
    sc = pd.read_csv(DATA / "leadtime_scores.csv", dtype={"code": str})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), sharey=True)
    for ax, (stages, tag) in zip(axes, (([S1, S2, S3], "Full sample (Q1/H1/Q3)"),
                                        ([S1, S2, S3, S4], "Subset with a pre-announcement"))):
        sub = sc[sc["stage"].isin(stages)]
        common = sub.pivot_table(index=["code", "year", "label"], columns="stage", values="score",
                                 aggfunc="last").dropna()
        inc = sub.set_index(["code", "year", "label"]).loc[common.index].reset_index()
        inc["pct"] = inc.groupby(["stage", "year"])["score"].rank(pct=True)
        thr = inc.groupby(["stage", "year"])["score"].quantile(0.90).rename("thr")
        inc = inc.merge(thr, on=["stage", "year"], how="left")
        inc["lit"] = inc["score"] >= inc["thr"]
        pos, neg = inc[inc["label"] == 1], inc[inc["label"] == 0]
        xs = np.arange(len(stages))
        ax.plot(xs, [pos[pos["stage"] == s]["lit"].mean() for s in stages], "o-", color="#d62728",
                label="Firms that did lose: lit share")
        ax.plot(xs, [neg[neg["stage"] == s]["lit"].mean() for s in stages], "s--", color="#1f77b4",
                label="Firms that did not lose: false alarms")
        ax.set_xticks(xs); ax.set_xticklabels([EN_SHORT[s] for s in stages], fontsize=8)
        ax.set_title(f"{tag} — common samples {len(common)}", fontsize=10)
        ax.grid(alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel("Lit share (score ≥ 90th percentile of that stage)")
            ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("When does the signal light up: one target, walking down the disclosure chain", fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / "fig_signal_timeline_en.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_leadtime()
    fig_timeline()
    print("已出：", "fig_leadtime_en.png", "fig_signal_timeline_en.png")
