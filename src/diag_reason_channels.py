"""把「戴帽原因」接回产品面：如果风控只能盯一件事，该盯哪一类？

上一格把 211 次戴帽按原因分成了三类（`src/classify_st_reasons.py`）：
**合规/治理类**（内控被否 · 行政处罚 · 冻结 · 资金占用 · 违规担保 · 规范运作）·
**财务类**（收入/净资产指标 · 净利润为负）· **重整/破产**。
这一格问的是：**同一张名单，对这三类的"抓得住"程度差多少** —— 即"该盯哪一类"。

读数口径：
- 分数用**两条**（生产上会是哪条就用哪条）：`s1` = 亏损模型（现在的 A 臂名单）、`s2` = 戴帽模型；
- 每个类别单独当一个二值目标：该类事件 = 1，其余 = 0（含其它类别的事件）；
- 报 **AUC（macro，四年平均）** 与**名单规模扫描**（前 5/10/20/30% 的召回与名单内该类占比）。

用法: python diag_reason_channels.py
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
COMPLIANCE = {"内控被否 / 重大缺陷", "信息披露违法 / 立案 / 处罚", "账户 / 资产被冻结",
              "资金占用（非经营性）", "违规担保", "规范运作类其他"}
FINANCIAL = {"财务类：收入或净资产指标", "财务类：净利润为负 / 连亏"}


def bucket(cat: str) -> str:
    if cat in COMPLIANCE:
        return "合规/治理类"
    if cat in FINANCIAL:
        return "财务类"
    if cat == "重整 / 破产":
        return "重整/破产"
    return "未归类"


def scores(s: pd.DataFrame, ycol: str, tag: str) -> pd.DataFrame:
    out = []
    for Y in TEST_YEARS:
        tr, va, te = s[s["year"] <= Y - 2].copy(), s[s["year"] == Y - 1], s[s["year"] == Y].copy()
        if min(len(tr), len(va), len(te)) == 0 or tr[ycol].nunique() < 2:
            continue
        tr["label"] = tr[ycol]
        p, _ = fit_predict(tr, va, te, FEATURES + CAT)
        te["p"] = p
        out.append(te[["code", "year", "p"]].rename(columns={"p": tag}))
    return pd.concat(out, ignore_index=True)


def main() -> int:
    panel = pd.read_pickle(PANEL)
    s = make_samples(panel)
    s = s[s["label"].notna() & s["t_decision"].notna() & s["t_label"].notna()].copy()
    _, first = build_st()
    s = s.merge(first, on="code", how="left")
    d = (s["st_first"] - pd.to_datetime(s["t_label"], errors="coerce")).dt.days
    s["L1"] = s["label"]
    s["L2"] = ((d >= 0) & (d <= 365)).astype(float)

    # 事件 → 类别
    reasons = pd.read_csv(ROOT / "data" / "st_reasons.csv", dtype={"code": str})
    reasons["code"] = reasons["code"].str.zfill(6)
    reasons["组"] = reasons["类别"].map(bucket)
    ev = reasons[["code", "date", "组"]].copy()
    ev["date"] = pd.to_datetime(ev["date"])

    t = s[s["year"].isin(TEST_YEARS)].copy()
    t["st_date"] = pd.to_datetime(t["st_first"])
    t = t.merge(ev, left_on=["code", "st_date"], right_on=["code", "date"], how="left")
    t.loc[t["L2"] == 0, "组"] = np.nan                     # 只给真事件贴类别

    m1 = scores(s, "L1", "s1")
    m2 = scores(s, "L2", "s2")
    t = t.merge(m1, on=["code", "year"]).merge(m2, on=["code", "year"])
    for c in ("s1", "s2"):
        t[f"{c}_pct"] = t.groupby("year")[c].rank(pct=True)

    print(f"测试年样本 {len(t)} · 期内戴帽事件 {int(t['L2'].sum())} 次")
    print("事件按类别：" + " · ".join(f"{k} {v}" for k, v in t[t['L2'] == 1]['组'].value_counts().items()))
    warn = t[t['L2'] == 1]['组'].isna().sum()
    if warn:
        print(f"⚠️ 有 {warn} 次事件没匹配到原因（正文没抓到或未归类）")

    rows, sweep = [], {}
    for cat in ("合规/治理类", "财务类", "重整/破产"):
        sub = t[t["组"] == cat]
        n = len(sub)
        tgt = (t["组"] == cat).astype(float)
        per1 = [metrics((g["组"] == cat).astype(float).values, g["s1"].values)["auc"]
                for _, g in t.groupby("year") if (g["组"] == cat).sum()]
        per2 = [metrics((g["组"] == cat).astype(float).values, g["s2"].values)["auc"]
                for _, g in t.groupby("year") if (g["组"] == cat).sum()]
        # ★ 交叉：在①组（没亏但戴帽）与③组（亏了也戴帽）里的分布
        g1 = int((sub["L1"] == 0).sum())
        g3 = int((sub["L1"] == 1).sum())
        rows.append({"类别": cat, "事件数": n, "其中没亏也戴帽": g1, "其中亏了才戴帽": g3,
                     "s1(亏损分) macro AUC": round(float(np.mean(per1)), 4) if per1 else None,
                     "s2(戴帽分) macro AUC": round(float(np.mean(per2)), 4) if per2 else None})

        # 名单规模扫描（用 s2 排，因为生产上要抓戴帽）
        sw = []
        for frac in (0.05, 0.10, 0.20, 0.30):
            hit = tot = 0
            hit_off = tot_off = 0
            for Y, g in t.groupby("year"):
                k = max(int(round(frac * len(g))), 1)
                sel = g.nlargest(k, "s2")
                hit += int((sel["组"] == cat).sum())
                tot += int((g["组"] == cat).sum())
                # ★ 「没亏也戴帽」那半（这才是一级名单管不到的部分）
                hit_off += int(((sel["组"] == cat) & (sel["L1"] == 0)).sum())
                tot_off += int(((g["组"] == cat) & (g["L1"] == 0)).sum())
            sw.append({"名单规模": f"{frac:.0%}",
                       "召回": round(hit / max(tot, 1), 4),
                       "其中没亏那半的召回": round(hit_off / max(tot_off, 1), 4) if tot_off else None,
                       "该类别在名单里占比": round(hit / max(k, 1), 4)})
        sweep[cat] = sw
    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n名单规模扫描（按戴帽分 s2 排；召回 = 抓到的该类事件 / 该类全部事件）：")
    for cat, sw in sweep.items():
        print(f"  {cat}：" + " · ".join(f"{x['名单规模']}→{x['召回']:.1%}" for x in sw))
        off = [x for x in sw if x["其中没亏那半的召回"] is not None]
        if off:
            print(f"     └ 其中「没亏也戴帽」那半：" +
                  " · ".join(f"{x['名单规模']}→{x['其中没亏那半的召回']:.1%}" for x in off))

    print("\n【判据】同一张名单对三类的召回差多少 —— 差得开的那个才是「该盯的那一件」。")
    (ROOT / "data" / "reason_channels.json").write_text(
        json.dumps({"rows": rows, "sweep": sweep}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("读数 -> data/reason_channels.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
