"""两段式标签：「当年亏损」→「次年戴帽」——一套特征能不能撑起**两级名单**？

业务上的两级名单长这样：
- **一级**（当年 4 月出）：哪些公司当年会亏 → 现在的 A 臂。
- **二级**（在一级名单内部再排）：哪些公司**下一年会被实施风险警示** → 风控真正要拦的那批。

这一格量三件事：
1. **两段之间的条件概率**：P(次年戴帽 | 当年亏损) vs P(次年戴帽 | 当年没亏)；
2. **分数能不能跨目标用**：把"亏损模型"的分直接拿去排"戴帽"这个目标，AUC 是多少？反过来呢？
   两个分**合起来**（秩平均）会不会更好？
3. **两级名单长什么样**：在一级名单（前 10%）内部按二级分再排，前几名的戴帽精确率是多少；
   跟"直接拿二级分在全体里排同样大小"比，谁更好。

用法: python two_stage_labels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import CAT, FEATURES, PANEL, fit_predict, make_samples, metrics  # noqa: E402
from st_label import build as build_st  # noqa: E402

TEST_YEARS = [2022, 2023, 2024, 2025]


def walkforward_scores(s: pd.DataFrame, ycol: str) -> pd.DataFrame:
    out = []
    for Y in TEST_YEARS:
        tr, va, te = s[s["year"] <= Y - 2].copy(), s[s["year"] == Y - 1], s[s["year"] == Y].copy()
        if min(len(tr), len(va), len(te)) == 0 or tr[ycol].nunique() < 2:
            continue
        tr["label"] = tr[ycol]
        p, _ = fit_predict(tr, va, te, FEATURES + CAT)
        te["p"] = p
        out.append(te[["code", "year", ycol, "p"]])
    return pd.concat(out, ignore_index=True)


def main() -> int:
    panel = pd.read_pickle(PANEL)
    s = make_samples(panel)
    s = s[s["label"].notna() & s["t_decision"].notna() & s["t_label"].notna()].copy()
    _, first = build_st()
    s = s.merge(first, on="code", how="left")
    d = (s["st_first"] - pd.to_datetime(s["t_label"], errors="coerce")).dt.days
    s["L1"] = s["label"]                                     # 当年年报亏损
    s["L2"] = ((d >= 0) & (d <= 365)).astype(float)          # 年报披露后一年内被戴帽
    t = s[s["year"].isin(TEST_YEARS)].copy()
    print(f"测试年样本 {len(t)} · L1（当年亏损）{t['L1'].mean():.2%} · L2（次年戴帽）{t['L2'].mean():.2%}")

    print("\n【1. 两段之间的条件概率】")
    ct = pd.crosstab(t["L1"], t["L2"], margins=True)
    print(ct.to_string())
    p_l2_given_l1 = t[t["L1"] == 1]["L2"].mean()
    p_l2_given_l0 = t[t["L1"] == 0]["L2"].mean()
    print(f"  P(次年戴帽 | 当年亏损) = {p_l2_given_l1:.2%}（n={int((t['L1']==1).sum())}）")
    print(f"  P(次年戴帽 | 当年没亏) = {p_l2_given_l0:.2%}（n={int((t['L1']==0).sum())}）"
          f" ⇒ 相差 {p_l2_given_l1 / max(p_l2_given_l0, 1e-9):.1f} 倍")
    print("  逐年：")
    print(t.groupby("year").agg(n=("L1", "size"), L1率=("L1", "mean"), L2率=("L2", "mean")).round(4).to_string())
    for Y, g in t.groupby("year"):
        print(f"    {int(Y)}: P(戴帽|亏损) {g[g['L1'] == 1]['L2'].mean():.2%} · "
              f"P(戴帽|没亏) {g[g['L1'] == 0]['L2'].mean():.2%}")

    print("\n【2. 两个分数能不能跨目标用】")
    m1 = walkforward_scores(s, "L1").rename(columns={"p": "s1"})[["code", "year", "s1"]]
    m2 = walkforward_scores(s, "L2").rename(columns={"p": "s2"})[["code", "year", "s2"]]
    j = t[["code", "year", "L1", "L2"]].merge(m1, on=["code", "year"]).merge(m2, on=["code", "year"])
    j["s_avg"] = j.groupby("year")[["s1", "s2"]].rank(pct=True)[["s1", "s2"]].mean(axis=1)
    rows = []
    for name, col in (("s1（亏损模型）", "s1"), ("s2（戴帽模型）", "s2"), ("秩平均（s1+s2）", "s_avg")):
        for target in ("L1", "L2"):
            per, ys, ps = [], [], []
            for Y, g in j.groupby("year"):
                if g[target].nunique() < 2:
                    continue
                per.append(metrics(g[target].values, g[col].values)["auc"])
                ys.append(g[target].values)
                ps.append(g[col].values)
            pooled = metrics(np.concatenate(ys), np.concatenate(ps))["auc"]
            rows.append({"分数": name, "目标": target, "pooled": round(pooled, 4),
                         "macro": round(float(np.mean(per)), 4)})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    print("  ⇒ 读法：**同一列分数在 L1 / L2 两个目标上的 AUC**；以及 s1+s2 在 L2 上有没有超过 s2 单独。")

    print("\n【3. 两级名单（一级名单内部按二级分再排）】")
    out = {}
    for Y, g in j.groupby("year"):
        k1 = max(int(round(0.10 * len(g))), 1)
        lvl1 = g.nlargest(k1, "s1")
        n_st = int(lvl1["L2"].sum())
        base = float(g["L2"].mean())
        # 一级名单内部的二级排序
        for frac in (0.1, 0.25, 0.5, 1.0):
            kk = max(int(round(frac * len(lvl1))), 1)
            sub = lvl1.nlargest(kk, "s2")
            # 同样大小的"全体里直接按 s2 排"
            sub2 = g.nlargest(kk, "s2")
            sub3 = g.nlargest(kk, "s1")          # ★ 对照：同样大小，直接拿一级分排（= 一级名单的头几名）
            out.setdefault(frac, []).append({
                "year": int(Y), "k": kk,
                "一级内_戴帽精确率": float(sub["L2"].mean()),
                "全体直接按s2_戴帽精确率": float(sub2["L2"].mean()),
                "直接按s1(同一级分)_戴帽精确率": float(sub3["L2"].mean()),
                "一级内_亏损精确率": float(sub["L1"].mean()),
                "基准率": base})
    for frac, rows_ in out.items():
        d = pd.DataFrame(rows_)
        print(f"  一级名单（前 10%）内部取前 {frac:.0%}（约 {d['k'].mean():.0f} 家/年）：")
        print(f"      按二级分 s2 排：戴帽精确率 {d['一级内_戴帽精确率'].mean():.2%}")
        print(f"      按一级分 s1 排（同样大小）：{d['直接按s1(同一级分)_戴帽精确率'].mean():.2%}")
        print(f"      全体里直接按 s2 排（同样大小）：{d['全体直接按s2_戴帽精确率'].mean():.2%}")
        print(f"      基准率 {d['基准率'].mean():.2%} · 这批的亏损率 {d['一级内_亏损精确率'].mean():.2%}")

    (ROOT / "data" / "two_stage.json").write_text(
        json.dumps({"crosstab": ct.to_dict(), "p_l2_given_l1": round(float(p_l2_given_l1), 4),
                    "p_l2_given_l0": round(float(p_l2_given_l0), 4),
                    "score_transfer": tab.to_dict("records"),
                    "two_level_lists": {str(k): v for k, v in out.items()}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n读数 -> data/two_stage.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
