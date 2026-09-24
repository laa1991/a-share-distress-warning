"""一家真实公司，沿四个决策时刻走一遍 —— 教具（`docs/教学-一家真实公司走一遍.md` 的判据）。

它做四件事：
1. 把这家公司在**四个决策时刻**分别能看到的关键财务数（季度累计口径）摊开；
2. 给出**当时那个模型**在每一格给它的分与**阶段内分位**（分位 ≥0.9 = 进前 10% 名单）；
3. 用**同一套折模型**重训一次，说明这一格的分是怎么来的（按贡献排序的前若干特征），
   并检查重训出来的分与当时存下的分是否一致（不一致就说明教具没接到真模型上）；
4. 摊开真相：当年年报的净利润与它的实际披露日、距决策时刻多少天。

用法: python tutorial_one_company.py --code 600872 --year 2022
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import CAT, FEATURES, PANEL, fit_predict  # noqa: E402
from run_leadtime_curve import base_label, load_yjyg, samples_at_quarter  # noqa: E402

STAGES = {1: "① 一季报披露", 2: "② 半年报披露", 3: "③ 三季报披露"}
SHOW = [("rev", "营业收入"), ("np", "净利润"), ("np_yoy", "净利润同比%"), ("roe", "ROE%"),
        ("gross_margin", "毛利率%"), ("debt_ratio", "资产负债率%"), ("ar_ratio", "应收/总资产%"),
        ("ocf_to_assets", "经营现金流/总资产%")]


def fmt(v, kind="num"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if kind == "pct":
        return f"{v:.1f}"
    if abs(v) >= 1e8:
        return f"{v/1e8:.2f}亿"
    return f"{v:,.0f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    panel = pd.read_pickle(PANEL)
    ann = base_label(panel)
    code, year = args.code, args.year

    a = ann[(ann["code"] == code) & (ann["year"] == year)]
    if not len(a):
        print(f"{code} {year}：标注表里没有这条，换一年")
        return 1
    a = a.iloc[0]
    truth = "亏" if a["label"] == 1 else "不亏"
    print(f"=== {code} · {year} 年 ===  真相：年报净利润 {a['np_annual']/1e8:+.2f} 亿 ⇒ **{truth}**"
          f"（年报实际披露日 {str(a['t_annual'])[:10]}）")

    scores = pd.read_csv(ROOT / "data" / "leadtime_scores.csv", dtype={"code": str})
    scores["pct"] = scores.groupby(["stage", "year"])["score"].rank(pct=True)

    rows, snapshots = [], {}
    for q, stage in STAGES.items():
        s = samples_at_quarter(panel, q, ann)
        r = s[(s["code"] == code) & (s["year"] == year)]
        if not len(r):
            rows.append({"决策时刻": stage, "决策日": "—", "分数": None, "分位": None, "进前10%": "—"})
            continue
        r = r.iloc[0]
        snapshots[q] = r
        sc = scores[(scores["code"] == code) & (scores["year"] == year) & (scores["stage"] == stage)]
        pct = float(sc["pct"].iloc[0]) if len(sc) else None
        rows.append({"决策时刻": stage, "决策日": str(r["t_decision"])[:10],
                     "分数": round(float(sc["score"].iloc[0]), 4) if len(sc) else None,
                     "分位": round(pct, 3) if pct is not None else None,
                     "进前10%": "✅" if (pct or 0) >= 0.9 else "—",
                     "距年报": int((a["t_annual"] - r["t_decision"]).days)})

    yjyg = load_yjyg()
    y = yjyg[(yjyg["code"] == code) & (yjyg["year"] == year)]
    if len(y):
        y = y.iloc[0]
        rows.append({"决策时刻": "④ 业绩预告公布", "决策日": str(y["yjyg_date"])[:10],
                     "分数": None, "分位": None,
                     "进前10%": "（预告里直接给了预计净利润 {:.2f} 亿，类型「{}」）".format(
                         y["yjyg_pred"] / 1e8, y["yjyg_type"]),
                     "距年报": int((a["t_annual"] - y["yjyg_date"]).days)})
        # 预告原文里"为什么"的那一句（有就打印，没有就跳过）
        raw = pd.read_csv(ROOT / "data" / "raw" / "yjyg" / f"{year}1231.csv", dtype={"股票代码": str})
        r = raw[raw["股票代码"].str.zfill(6) == code]
        for col in ("业绩变动原因", "预告类型", "预测数值"):
            if col in r.columns and len(r):
                print(f"  （预告原文 · {col}）{str(r[col].iloc[0])[:220]}")

    print("\n--- 四个决策时刻（分数与分位来自当时那个模型） ---")
    print(pd.DataFrame(rows).to_string(index=False, na_rep="—"))

    print("\n--- 它在每一格能看到的数（季度累计口径） ---")
    tab = pd.DataFrame({name: [fmt(snapshots[q][col], "pct" if "yoy" in col or "ratio" in col or "margin" in col or col in ("roe",) else "num")
                               for q in (1, 2, 3)] for col, name in SHOW},
                       index=[STAGES[q] for q in (1, 2, 3)])
    print(tab.to_string())

    # --- 这一格的分是怎么来的：用同一套折模型重训一次，看贡献 ---
    s1 = samples_at_quarter(panel, 1, ann)
    tr, va, te = s1[s1["year"] <= year - 2], s1[s1["year"] == year - 1], s1[s1["year"] == year]
    if min(len(tr), len(va), len(te)) and tr["label"].nunique() > 1:
        preds, booster = fit_predict(tr, va, te, FEATURES + CAT)
        te2 = te.reset_index(drop=True)
        i = te2.index[(te2["code"] == code)].tolist()
        if i:
            i = i[0]
            p_again = float(preds[i])
            saved = scores[(scores["code"] == code) & (scores["year"] == year) & (scores["stage"] == STAGES[1])]
            saved_p = float(saved["score"].iloc[0]) if len(saved) else float("nan")
            print(f"\n--- ① 那一格的分是怎么来的（同一套折模型重训；重训分 {p_again:.6f} vs 存档分 {saved_p:.6f}，"
                  f"差 {abs(p_again - saved_p):.2e}）---")
            contrib = booster.predict(te2[FEATURES + CAT].iloc[[i]], pred_contrib=True)[0]
            names = FEATURES + CAT + ["（基准分）"]
            order = np.argsort(-np.abs(contrib))
            row_vals = te2[FEATURES + CAT].iloc[i]
            print("  特征贡献（正=推高亏损分，负=压低）：")
            for k in order[: args.topk]:
                val = "（所有样本的起点）" if k == len(FEATURES) + len(CAT) else f"该样本取值 {row_vals[names[k]]}"
                print(f"    {names[k]:22s} {contrib[k]:+9.4f}   （{val}）")
            print(f"    {'合计 raw':22s} {contrib.sum():+9.4f} ⇒ 概率 {1/(1+np.exp(-contrib.sum())):.4f}")

    print("\n结论句（每个阶段的位置）：", " → ".join(
        f"{r['决策时刻'][:4]}分位{r['分位']}" if r["分位"] is not None else f"{r['决策时刻'][:4]}—" for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
