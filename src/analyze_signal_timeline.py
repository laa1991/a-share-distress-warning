"""风险信号是「什么时候亮起来的」——用同一批公司在披露链各阶段的**阶段内分位**画轨迹。

三个口径上的讲究（都是踩过才知道的）：
1. **分数不在同一把尺上**：每个阶段是各自一个模型 ⇒ 只能用**阶段内分位**，不能用原始分数比。
2. **分母也要一致**：先取「各阶段都打过分」的公共样本，**再在这批样本内部**排名；
   否则每个阶段的分位来自不同总体（预告那格的正例率 43.8%，另一个总体）。
3. **并列会把"进前 10%"判没**：年报那格是 0/1 的退化分数，按名次取前 10% 会一个都取不到 ⇒
   一律用「**分数 ≥ 该阶段该年的第 90 百分位**」当亮灯阈值，而不是按名次数人头。

年报那一格（AUC=1.0）是**答案本身**，不放进轨迹。

用法: python analyze_signal_timeline.py
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
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

S1, S2, S3, S4 = "① 一季报披露", "② 半年报披露", "③ 三季报披露", "④ 业绩预告公布"
SHORT = {S1: "一季报\n4月", S2: "半年报\n8月", S3: "三季报\n10月", S4: "预告\n次年1月"}


def prep(sc: pd.DataFrame, stages: list[str]) -> tuple[pd.DataFrame, dict]:
    sub = sc[sc["stage"].isin(stages)]
    common = sub.pivot_table(index=["code", "year", "label"], columns="stage", values="score",
                             aggfunc="last").dropna()
    incommon = sub.set_index(["code", "year", "label"]).loc[common.index].reset_index()
    incommon["pct"] = incommon.groupby(["stage", "year"])["score"].rank(pct=True)
    thr = incommon.groupby(["stage", "year"])["score"].quantile(0.90).rename("thr")
    incommon = incommon.merge(thr, on=["stage", "year"], how="left")
    incommon["lit"] = incommon["score"] >= incommon["thr"]
    wide = incommon.pivot_table(index=["code", "year", "label"], columns="stage", values="pct", aggfunc="last")
    return incommon, {"n": int(len(common)), "wide": wide}


def summarize(incommon: pd.DataFrame, stages: list[str]) -> pd.DataFrame:
    pos = incommon[incommon["label"] == 1]
    neg = incommon[incommon["label"] == 0]
    rows = []
    for s in stages:
        p, n = pos[pos["stage"] == s], neg[neg["stage"] == s]
        rows.append({"阶段": s, "会亏的（中位分位）": round(float(p["pct"].median()), 4),
                     "不会亏的（中位分位）": round(float(n["pct"].median()), 4),
                     "会亏的·已亮灯": round(float(p["lit"].mean()), 4),
                     "不会亏的·误亮灯": round(float(n["lit"].mean()), 4)})
    return pd.DataFrame(rows)


def first_light(incommon: pd.DataFrame, stages: list[str]) -> dict:
    pos = incommon[incommon["label"] == 1]
    first = {}
    for s in stages:
        hit = pos[(pos["stage"] == s) & pos["lit"]]
        for idx in hit.set_index(["code", "year"]).index:
            first.setdefault(idx, s)
    cnt = {s: sum(1 for v in first.values() if v == s) for s in stages}
    n = pos[["code", "year"]].drop_duplicates().shape[0]
    return {"n_pos": int(n), "share": {k: round(v / n, 4) for k, v in cnt.items()},
            "never": round(1 - len(first) / n, 4)}


def main() -> int:
    sc = pd.read_csv(DATA / "leadtime_scores.csv", dtype={"code": str})
    print("阶段 × 样本数（全量）:")
    print(sc.groupby("stage").agg(n=("score", "size"), 正例率=("label", "mean")).round(4).to_string())

    out = {}
    for tag, stages in (("全样本（①②③）", [S1, S2, S3]), ("有预告的子集（①②③④）", [S1, S2, S3, S4])):
        inc, meta = prep(sc, stages)
        tab = summarize(inc, stages)
        fl = first_light(inc, stages)
        print(f"\n=== {tag} ===  公共样本 {meta['n']}")
        print(tab.to_string(index=False))
        print("正例「第一次亮灯」落在哪一阶段:", fl["share"], "| 从未亮灯:", fl["never"])
        out[tag] = {"n_common": meta["n"], "table": tab.to_dict("records"), "first_light": fl}

    # 示例轨迹（用全样本那一组的公共集）
    inc, meta = prep(sc, [S1, S2, S3])
    wide = meta["wide"]
    pos = wide[wide.index.get_level_values("label") == 1]
    ex = {}
    early = pos[(pos[S1] >= 0.9) & (pos[S3] >= 0.9)]
    if len(early):
        ex["一路在名单上（4 月就亮、10 月仍亮）"] = early.index[0]
    late = pos[(pos[S1] < 0.5) & (pos[S3] >= 0.9)]
    if len(late):
        ex["越查越坏（4 月不起眼、10 月进前排）"] = late.index[0]
    false_alarm = wide[(wide.index.get_level_values("label") == 0) & (wide[S1] >= 0.9) & (wide[S3] >= 0.9)]
    if len(false_alarm):
        ex["误报（一路高分，最后没亏）"] = false_alarm.index[0]
    examples = {k: {"code_year": f"{v[0]}-{v[1]}", **{s: round(float(wide.loc[v, s]), 3) for s in [S1, S2, S3]}}
                for k, v in ex.items()}
    print("\n示例轨迹（阶段内分位）:", json.dumps(examples, ensure_ascii=False, indent=2))
    out["examples"] = examples

    (DATA / "signal_timeline.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 图：两栏，分别对应两组公共样本 ----
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), sharey=True)
    for ax, (tag, stages) in zip(axes, (("全样本（①②③）", [S1, S2, S3]), ("有预告的子集（①②③④）", [S1, S2, S3, S4]))):
        inc, meta = prep(sc, stages)
        tab = summarize(inc, stages)
        x = np.arange(len(stages))
        ax.plot(x, tab["会亏的·已亮灯"], "o-", color="#d62728", label="当年真亏的公司：已亮灯比例")
        ax.plot(x, tab["不会亏的·误亮灯"], "s--", color="#1f77b4", label="当年没亏的公司：误亮灯比例")
        ax.set_xticks(x); ax.set_xticklabels([SHORT[s] for s in stages], fontsize=8)
        ax.set_title(f"{tag}　公共样本 {meta['n']}", fontsize=10)
        ax.grid(alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel("亮灯比例（分数 ≥ 该阶段第 90 百分位）")
            ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("风险信号什么时候亮：同一个目标，沿披露链往后走", fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / "fig_signal_timeline.png", dpi=150); plt.close(fig)
    print("\n图 ->", FIGS / "fig_signal_timeline.png")
    print("读数 ->", DATA / "signal_timeline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
