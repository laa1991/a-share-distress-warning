"""那 49 个「当年没亏、次年却被戴帽」的样本 —— 一级名单该不该为它们开口子？

上一格的漏：一级名单以「会亏」当门票，漏掉了 268 个戴帽样本里的 49 个（18%）。这一格问三件事：

1. **它们长什么样**：与「亏了但没戴帽」（4,437 个）、「干净样本」（14,768 个）比，哪些特征的**分位**明显不同
   —— 分位一律**在同一年内**算，跨年才可比。
2. **模型能不能看见它们**：亏损分 s1 与戴帽分 s2 各自把它们排在哪（中位分位、进前 10/20/30% 的比例）。
3. **要不要为它们开口子**：如果 s2 能把它们排到前面 ⇒ 「直接按 s2 排全市场」就够，不必改一级名单；
   如果 s2 也排不出来 ⇒ 它们与 600872 那种「一次性事件」同类，任何读财报的模型都看不见。

用法: python diag_st_outside_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import CAT, FEATURES, PANEL, fit_predict, make_samples  # noqa: E402
from st_label import build as build_st  # noqa: E402

TEST_YEARS = [2022, 2023, 2024, 2025]
FOCUS = ["debt_ratio", "debt_ratio_lag1", "ar_ratio", "inv_ratio", "ocf_to_assets", "ocf_to_assets_lag1",
         "roe", "roe_lag1", "np_margin", "cash_ratio", "equity_ratio", "log_assets", "log_rev",
         "total_assets_yoy", "rev_yoy", "np_yoy", "gross_margin", "ap_ratio", "sell_exp_ratio",
         "fin_exp_ratio", "d_roe", "d_debt_ratio", "d_np_margin", "prev_loss"]


def walkforward(s: pd.DataFrame, ycol: str) -> pd.DataFrame:
    out = []
    for Y in TEST_YEARS:
        tr, va, te = s[s["year"] <= Y - 2].copy(), s[s["year"] == Y - 1], s[s["year"] == Y].copy()
        if min(len(tr), len(va), len(te)) == 0:
            continue
        tr["label"] = tr[ycol]
        p, _ = fit_predict(tr, va, te, FEATURES + CAT)
        te["p"] = p
        out.append(te[["code", "year", "p"]])
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
    t = s[s["year"].isin(TEST_YEARS)].copy()

    t["grp"] = np.select(
        [(t["L1"] == 0) & (t["L2"] == 1), (t["L1"] == 1) & (t["L2"] == 0),
         (t["L1"] == 1) & (t["L2"] == 1)],
        ["① 没亏但戴帽", "② 亏了没戴帽", "③ 亏了也戴帽"], default="④ 干净")
    print(t["grp"].value_counts().to_string())

    # 特征：同年内分位，再按组取中位数
    pct_cols = {}
    for c in FOCUS:
        if c not in t.columns:
            continue
        pct_cols[c] = t.groupby("year")[c].rank(pct=True)
    P = pd.DataFrame(pct_cols)
    P["grp"] = t["grp"].values
    med = P.groupby("grp").median().T
    med["差(①−②)"] = med.get("① 没亏但戴帽", 0) - med.get("② 亏了没戴帽", 0)
    med["差(①−④)"] = med.get("① 没亏但戴帽", 0) - med.get("④ 干净", 0)
    med = med.reindex(med["差(①−④)"].abs().sort_values(ascending=False).index)
    print("\n各组的特征中位分位（同年内分位；只看最有区分度的 12 行）：")
    print(med.head(12).round(3).to_string())

    # 模型把这三组排在哪
    m1 = walkforward(s, "L1").rename(columns={"p": "s1"})
    m2 = walkforward(s, "L2").rename(columns={"p": "s2"})
    t = t.merge(m1, on=["code", "year"]).merge(m2, on=["code", "year"])
    for col in ("s1", "s2"):
        t[f"{col}_pct"] = t.groupby("year")[col].rank(pct=True)

    print("\n模型把它们排在哪（同年内分位）：")
    rows = []
    for g, d_ in t.groupby("grp"):
        rows.append({"组": g, "n": len(d_),
                     "s1 中位分位": round(float(d_["s1_pct"].median()), 3),
                     "s1 进前10%": f"{(d_['s1_pct'] >= 0.9).mean():.1%}",
                     "s1 进前20%": f"{(d_['s1_pct'] >= 0.8).mean():.1%}",
                     "s2 中位分位": round(float(d_["s2_pct"].median()), 3),
                     "s2 进前10%": f"{(d_['s2_pct'] >= 0.9).mean():.1%}",
                     "s2 进前20%": f"{(d_['s2_pct'] >= 0.8).mean():.1%}",
                     "s2 进前30%": f"{(d_['s2_pct'] >= 0.7).mean():.1%}"})
    print(pd.DataFrame(rows).to_string(index=False))

    # 具体名字（给人工看）
    g1 = t[t["grp"] == "① 没亏但戴帽"].nsmallest(12, "s1_pct")[["code", "year", "s1_pct", "s2_pct", "st_first"]]
    print("\n① 组里 s1 分位最低的 12 个（模型最看不见的）：")
    print(g1.round(3).to_string(index=False))

    print("\n【判据】口径：都看 s2（戴帽分）在**全体**里排名的分位。")
    g1_, g3_ = t[t["grp"] == "① 没亏但戴帽"], t[t["grp"] == "③ 亏了也戴帽"]
    for cut in (0.9, 0.8, 0.7):
        print(f"  分位 ≥{cut:.1f}：① 没亏但戴帽 {(g1_['s2_pct'] >= cut).mean():6.1%} · "
              f"③ 亏了也戴帽 {(g3_['s2_pct'] >= cut).mean():6.1%}")

    # ★ 名单要放多大才能把 ① 组捞进来（这是"开不开口子"的真正代价）
    print("\n名单规模 → 能捞到多少（四年合计，按 s2 排；分母是各组的全部样本）：")
    rows2 = []
    for frac in (0.05, 0.10, 0.20, 0.30, 0.50):
        hit1 = hit3 = hitall = tot1 = tot3 = totall = 0
        prec_num = prec_den = 0
        for Y, g in t.groupby("year"):
            k = max(int(round(frac * len(g))), 1)
            sel = g.nlargest(k, "s2")
            hit1 += int(sel["grp"].eq("① 没亏但戴帽").sum())
            hit3 += int(sel["grp"].eq("③ 亏了也戴帽").sum())
            hitall += int(sel["L2"].sum())
            prec_num += int(sel["L2"].sum())
            prec_den += len(sel)
            tot1 += int(g["grp"].eq("① 没亏但戴帽").sum())
            tot3 += int(g["grp"].eq("③ 亏了也戴帽").sum())
            totall += int(g["L2"].sum())
        rows2.append({"名单规模": f"{frac:.0%}", "名单条数/年": k,
                      "①组召回": f"{hit1/max(tot1,1):.1%}", "③组召回": f"{hit3/max(tot3,1):.1%}",
                      "全部戴帽召回": f"{hitall/max(totall,1):.1%}",
                      "名单里戴帽比例": f"{prec_num/max(prec_den,1):.2%}"})
    print(pd.DataFrame(rows2).to_string(index=False))

    (ROOT / "data" / "st_outside_gate.json").write_text(json.dumps({
        "counts": t["grp"].value_counts().to_dict(),
        "feature_medians": med.round(4).to_dict(),
        "model_position": rows,
        "examples": g1.assign(st_first=g1["st_first"].astype(str)).to_dict("records"),
        "list_size_sweep": rows2,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ★ 为什么会戴帽？——**标题答不了**（照实记，不硬归类）
    from st_label import clean, load_events
    ev = load_events()
    keys = set(zip(g1["code"], pd.to_datetime(g1["st_first"]).dt.normalize()))
    def _match(c, dt) -> bool:
        try:
            return (c, pd.Timestamp(dt).normalize()) in keys
        except Exception:  # noqa: BLE001
            return False
    sub = ev[[_match(c, dt) for c, dt in zip(ev["code"], ev["date"])]]
    MECH = ["冻结", "占用", "违规", "担保", "审计", "内控", "诉讼", "重整", "破产", "立案", "调查",
            "亏损", "营业收入", "净资产", "追溯", "信息披露", "否定", "无法表示"]
    print(f"\n① 组匹配到 {len(sub)} 条戴帽公告；**标题里能读出机制的只有 "
          f"{int(sub['标题'].map(lambda x: any(k in clean(x) for k in MECH)).sum())} 条**")
    for k in MECH:
        n = int(sub["标题"].map(lambda x: k in clean(x)).sum())
        if n:
            print(f"   含「{k}」: {n}")
    print("  ⇒ 其余都是模板化的「关于（股票）被实施…风险警示暨停牌的公告」——**原因在正文里，标题没有**；")
    print("     巨潮详情页是 JS 壳（8.6 KB HTML、正文不在其中），要答「为什么戴帽」得走公告 PDF，**本条未做**。")
    st_reasons = {"matched": int(len(sub)),
                  "with_mechanism_in_title": int(sub["标题"].map(lambda x: any(k in clean(x) for k in MECH)).sum())}

    (ROOT / "data" / "st_outside_gate.json").write_text(json.dumps({
        "counts": t["grp"].value_counts().to_dict(),
        "feature_medians": med.round(4).to_dict(),
        "model_position": rows,
        "examples": g1.assign(st_first=g1["st_first"].astype(str)).to_dict("records"),
        "list_size_sweep": rows2,
        "st_reasons": st_reasons,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n读数 -> data/st_outside_gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
