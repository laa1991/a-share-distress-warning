"""提前量 → 判别力，以及「风险信号什么时候亮起来」。

同一个预测目标（Y 年年报是否亏损），沿着**披露链**一路往后走，每一步都只用那一刻真能看见的东西：
  ① Y 年一季报披露        （中位 4/23）—— 信息 = Q1
  ② Y 年半年报披露        （中位 8/14）—— 信息 = Q1–Q2
  ③ Y 年三季报披露        （中位 10/23）—— 信息 = Q1–Q3
  ④ Y 年**业绩预告**公布   （次年 1 月）—— Q1–Q3 + 预告（区间/类型）
  ⑤ Y 年**业绩快报**公布   （次年 2–4 月）—— Q1–Q3 + 快报（实际数字）
  ⑥ Y 年年报披露          （次年 3–4 月）—— = 答案，用来标定上界
每一步都用同一套走查（训 ≤ Y−2 / 验 Y−1 / 测 Y）、同一个 LightGBM、同一个测试年集合。
逐样本分数落 `data/leadtime_scores.csv`，供「信号亮灯时间」分析用。

用法: python run_leadtime_curve.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import CAT, FEATURES, fit_predict, metrics, PANEL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
YJYG = ROOT / "data" / "raw" / "yjyg"
YJKB = ROOT / "data" / "raw" / "yjkb"
TEST_YEARS = [2022, 2023, 2024, 2025]


def base_label(panel: pd.DataFrame) -> pd.DataFrame:
    ann = panel[panel["period"] % 10000 == 1231][["code", "period", "np", "disc_actual"]].rename(
        columns={"np": "np_annual", "disc_actual": "t_annual"})
    ann["year"] = ann["period"] // 10000
    ann["label"] = (ann["np_annual"] < 0).astype(float)
    prev = ann[["code", "year", "np_annual"]].rename(columns={"np_annual": "np_prev"})
    prev["year"] += 1
    ann = ann.merge(prev, on=["code", "year"], how="left")
    ann["prev_loss"] = (ann["np_prev"] < 0).astype(float)
    return ann[["code", "year", "label", "prev_loss", "t_annual", "np_annual"]]


def samples_at_quarter(panel: pd.DataFrame, q: int, ann: pd.DataFrame) -> pd.DataFrame:
    row = panel[panel["q"] == q].copy()
    row["year"] = row["period"] // 10000
    s = row[["code", "year", "disc_actual"] + FEATURES + CAT].rename(columns={"disc_actual": "t_decision"})
    s = s.merge(ann, on=["code", "year"], how="inner")
    s["industry"] = s["industry"].astype("category")
    return s[s["label"].notna() & s["t_decision"].notna()]


def load_yjyg() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(YJYG / "*.csv"))):
        period = int(Path(f).stem)
        if period % 10000 != 1231:
            continue
        d = pd.read_csv(f, dtype={"股票代码": str})
        d["code"] = d["股票代码"].str.zfill(6)
        d["year"] = period // 10000
        d["ann_date"] = pd.to_datetime(d["公告日期"], errors="coerce")
        d = d[d["预测指标"].astype(str).str.contains("归属于上市公司股东的净利润")]
        if d.empty:
            continue
        frames.append(d.groupby(["code", "year"]).agg(
            yjyg_pred=("预测数值", "last"), yjyg_change=("业绩变动幅度", "last"),
            yjyg_prev=("上年同期值", "last"), yjyg_type=("预告类型", "last"),
            yjyg_date=("ann_date", "min"), yjyg_date_max=("ann_date", "max"),
        ).reset_index())
    return pd.concat(frames, ignore_index=True)


def load_yjkb() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(YJKB / "*.csv"))):
        period = int(Path(f).stem)
        if period % 10000 != 1231:
            continue
        d = pd.read_csv(f, dtype={"股票代码": str})
        d["code"] = d["股票代码"].str.zfill(6)
        d["year"] = period // 10000
        d["kb_date"] = pd.to_datetime(d["公告日期"], errors="coerce")
        frames.append(d.groupby(["code", "year"]).agg(
            yjkb_eps=("每股收益", "last"), yjkb_rev=("营业收入-营业收入", "last"),
            yjkb_rev_yoy=("营业收入-同比增长", "last"), yjkb_np=("净利润-净利润", "last"),
            yjkb_np_yoy=("净利润-同比增长", "last"), yjkb_bps=("每股净资产", "last"),
            yjkb_roe=("净资产收益率", "last"), yjkb_date=("kb_date", "min"),
        ).reset_index())
    return pd.concat(frames, ignore_index=True)


def walkforward(s: pd.DataFrame, feats: list[str], tag: str, date_col: str = "t_decision"):
    ys, ps, per, detail = [], [], {}, []
    for Y in TEST_YEARS:
        tr, va, te = s[s["year"] <= Y - 2], s[s["year"] == Y - 1], s[s["year"] == Y]
        if min(len(tr), len(va), len(te)) == 0 or tr["label"].nunique() < 2:
            continue
        p, _ = fit_predict(tr, va, te, feats)
        ys.append(te["label"].values); ps.append(p); per[Y] = metrics(te["label"].values, p)
        detail.append(pd.DataFrame({"code": te["code"].values, "year": te["year"].values,
                                    "label": te["label"].values, "score": p,
                                    "stage_date": te[date_col].values}))
    y, p = np.concatenate(ys), np.concatenate(ps)
    res = {"tag": tag, "pooled": metrics(y, p),
           "macro_auc": round(float(np.mean([m["auc"] for m in per.values()])), 4), "per_year": per,
           "y": y, "p": p, "n_features": len(feats)}
    return res, pd.concat(detail, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-yjkb", action="store_true",
                    help="把业绩快报那一格也报出来（⚠️ 它的公告日期字段被'下一年再披露'污染，见代码里的注释）")
    args = ap.parse_args()
    include_yjkb = args.include_yjkb
    panel = pd.read_pickle(PANEL)
    ann = base_label(panel)
    yjyg, yjkb = load_yjyg(), load_yjkb()
    print(f"业绩预告 {len(yjyg)} 组 · 公告日中位 {yjyg['yjyg_date'].median().date()} | "
          f"业绩快报 {len(yjkb)} 组 · 公告日中位 {yjkb['yjkb_date'].median().date()}")

    out, scores = {}, []
    for q, tag in ((1, "① 一季报披露"), (2, "② 半年报披露"), (3, "③ 三季报披露")):
        s = samples_at_quarter(panel, q, ann)
        r, det = walkforward(s, FEATURES + CAT, tag)
        det["stage"] = tag
        scores.append(det)
        r["decision_date_median"] = str(s[s["year"].isin(TEST_YEARS)]["t_decision"].median().date())
        r["lead_days_median"] = float((s[s["year"].isin(TEST_YEARS)]["t_annual"] -
                                       s[s["year"].isin(TEST_YEARS)]["t_decision"]).dt.days.median())
        out[tag] = r
        print(f"{tag}: n={r['pooled']['n']} AUC={r['pooled']['auc']} macro={r['macro_auc']} "
              f"决策中位 {r['decision_date_median']} 距年报 {r['lead_days_median']:.0f} 天", flush=True)

    # ④ 业绩预告
    s3 = samples_at_quarter(panel, 3, ann).merge(yjyg, on=["code", "year"], how="inner")
    s3 = s3[s3["yjyg_date"].notna()].copy()
    s3["yjyg_type"] = s3["yjyg_type"].astype("category")
    r4, d4 = walkforward(s3, FEATURES + CAT + ["yjyg_pred", "yjyg_change", "yjyg_prev", "yjyg_type"],
                         "④ 业绩预告公布", "yjyg_date")
    d4["stage"] = "④ 业绩预告公布"; scores.append(d4)
    r4["decision_date_median"] = str(s3[s3["year"].isin(TEST_YEARS)]["yjyg_date"].median().date())
    r4["lead_days_median"] = float((s3[s3["year"].isin(TEST_YEARS)]["t_annual"] -
                                    s3[s3["year"].isin(TEST_YEARS)]["yjyg_date"]).dt.days.median())
    out["④ 业绩预告公布"] = r4
    r4b, _ = walkforward(s3, FEATURES + CAT, "④b 同样本·不看预告")
    out["④b 同样本·不看预告"] = r4b
    print(f"④ 业绩预告: n={r4['pooled']['n']} AUC={r4['pooled']['auc']} | 消融 ④b={r4b['pooled']['auc']} | "
          f"公告日中位 {r4['decision_date_median']} 距年报 {r4['lead_days_median']:.0f} 天", flush=True)

    # ⑤ 业绩快报 —— **默认不纳入**：它的「公告日期」字段被同一个机制污染
    #    （带「上年同期」栏的数字会被下一年的快报再披露一次，日期随之刷新）。
    #    实测（`src/diag_yjkb_dates.py` 那条诊断）：20221231 那份文件的公告日**中位数落在 2024-02**，
    #    而真实的快报窗口是 2023-01..04；把快报日与年报实际披露日相减，**67.8% 的行为负**（即"快报晚于年报"）。
    #    ⇒ 装不上 PIT 锚点，这一格**不报**（要看得显式加 --include-yjkb，并且读数带 ⚠️）。
    if include_yjkb and len(yjkb):
        s3k = samples_at_quarter(panel, 3, ann).merge(yjkb, on=["code", "year"], how="inner")
        s3k = s3k[s3k["yjkb_date"].notna()].copy()
        feats_k = FEATURES + CAT + ["yjkb_eps", "yjkb_rev", "yjkb_rev_yoy", "yjkb_np",
                                    "yjkb_np_yoy", "yjkb_bps", "yjkb_roe"]
        r5, d5 = walkforward(s3k, feats_k, "⑤ 业绩快报公布（⚠️ 日期未验）", "yjkb_date")
        d5["stage"] = "⑤ 业绩快报公布（⚠️ 日期未验）"; scores.append(d5)
        r5["decision_date_median"] = str(s3k[s3k["year"].isin(TEST_YEARS)]["yjkb_date"].median().date())
        r5["lead_days_median"] = float((s3k[s3k["year"].isin(TEST_YEARS)]["t_annual"] -
                                        s3k[s3k["year"].isin(TEST_YEARS)]["yjkb_date"]).dt.days.median())
        out["⑤ 业绩快报公布（⚠️ 日期未验）"] = r5
        r5b, _ = walkforward(s3k, FEATURES + CAT, "⑤b 同样本·不看快报")
        out["⑤b 同样本·不看快报"] = r5b
        print(f"⑤ 业绩快报（⚠️ 日期未验）: n={r5['pooled']['n']} AUC={r5['pooled']['auc']} | "
              f"消融 ⑤b={r5b['pooled']['auc']} | 公告日中位 {r5['decision_date_median']} "
              f"距年报 {r5['lead_days_median']:.0f} 天", flush=True)

    # ⑥ 年报（上界）
    s4 = samples_at_quarter(panel, 4, ann)
    r6, d6 = walkforward(s4, FEATURES + CAT, "⑥ 年报披露（=答案）")
    d6["stage"] = "⑥ 年报披露"; scores.append(d6)
    r6["decision_date_median"] = str(s4[s4["year"].isin(TEST_YEARS)]["t_decision"].median().date())
    r6["lead_days_median"] = 0.0
    out["⑥ 年报披露（=答案）"] = r6

    print("\n================ 披露链上的判别力 ================")
    rows = []
    for k, v in out.items():
        rows.append({"决策时刻": k, "决策日中位": v.get("decision_date_median"),
                     "距年报(天)": v.get("lead_days_median"), "n": v["pooled"]["n"],
                     "AUC": v["pooled"]["auc"], "macro AUC": v["macro_auc"],
                     "KS": v["pooled"]["ks"], "top10%": v["pooled"]["top10_rate"],
                     "正例率": v["pooled"]["pos_rate"]})
    print(pd.DataFrame(rows).to_string(index=False))

    keep = {k: {kk: vv for kk, vv in v.items() if kk not in ("y", "p")} for k, v in out.items()}
    (ROOT / "data" / "leadtime_curve.json").write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")
    sc = pd.concat(scores, ignore_index=True)
    sc.to_csv(ROOT / "data" / "leadtime_scores.csv", index=False, encoding="utf-8-sig")
    print(f"逐样本分数 -> {ROOT / 'data' / 'leadtime_scores.csv'}  ({len(sc)} 行 · {sc['stage'].nunique()} 个阶段)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
