"""提前量 → 判别力的曲线：同一个任务，决策时刻往后挪一格，分数涨多少。

同一个预测目标（Y 年年报是否亏损），五个决策时刻：
  ① Y 年一季报披露      （中位 4/27）—— 信息 = Q1
  ② Y 年半年报披露      （中位 8/26）—— 信息 = Q1–Q2
  ③ Y 年三季报披露      （中位 10/28）—— 信息 = Q1–Q3
  ④ Y 年**业绩预告**公布（中位次年 1–2 月）—— 信息 = Q1–Q3 + 预告本身（★ 合法，且它就是提前版的答案）
  ⑤ Y 年年报披露        （次年 4 月）—— 信息 = 全年（= 答案，用来标定上界）
每一步都用同一套走查（训 ≤ Y−2 / 验 Y−1 / 测 Y）、同一个 LightGBM、同一个测试年集合。

用法: python run_leadtime_curve.py
"""
from __future__ import annotations

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
TEST_YEARS = [2022, 2023, 2024, 2025]
ANNUAL = FEATURES      # 年报那一格：特征里就有答案本身


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
        g = d.groupby(["code", "year"]).agg(
            yjyg_pred=("预测数值", "last"),
            yjyg_change=("业绩变动幅度", "last"),
            yjyg_prev=("上年同期值", "last"),
            yjyg_type=("预告类型", "last"),
            yjyg_date=("ann_date", "min"),
            yjyg_date_max=("ann_date", "max"),
            yjyg_rows=("ann_date", "size"),
        ).reset_index()
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def walkforward(s: pd.DataFrame, feats: list[str], tag: str) -> dict:
    ys, ps, per = [], [], {}
    for Y in TEST_YEARS:
        tr, va, te = s[s["year"] <= Y - 2], s[s["year"] == Y - 1], s[s["year"] == Y]
        if min(len(tr), len(va), len(te)) == 0 or tr["label"].nunique() < 2:
            continue
        p, _ = fit_predict(tr, va, te, feats)
        ys.append(te["label"].values); ps.append(p); per[Y] = metrics(te["label"].values, p)
    y, p = np.concatenate(ys), np.concatenate(ps)
    return {"tag": tag, "pooled": metrics(y, p),
            "macro_auc": round(float(np.mean([m["auc"] for m in per.values()])), 4), "per_year": per,
            "y": y, "p": p, "n_features": len(feats)}


def main() -> int:
    panel = pd.read_pickle(PANEL)
    ann = base_label(panel)
    yjyg = load_yjyg()
    print(f"业绩预告：{len(yjyg)} 个 (代码, 年份) 组合 | 公告日中位 {yjyg['yjyg_date'].median().date()} "
          f"| 首次与末次同日占比 {(yjyg['yjyg_date'] == yjyg['yjyg_date_max']).mean():.1%}")

    out = {}
    for q, tag in ((1, "① 一季报披露"), (2, "② 半年报披露"), (3, "③ 三季报披露")):
        s = samples_at_quarter(panel, q, ann)
        r = walkforward(s, FEATURES + CAT, tag)
        r["decision_date_median"] = str(s[s["year"].isin(TEST_YEARS)]["t_decision"].median().date())
        r["lead_days_median"] = float((s[s["year"].isin(TEST_YEARS)]["t_annual"] - s[s["year"].isin(TEST_YEARS)]["t_decision"]).dt.days.median())
        out[tag] = r
        print(f"{tag}: n={r['pooled']['n']} AUC={r['pooled']['auc']} macro={r['macro_auc']} "
              f"决策中位 {r['decision_date_median']} 距年报 {r['lead_days_median']:.0f} 天", flush=True)

    # ④ 业绩预告：信息 = Q3 报表 + 预告
    s3 = samples_at_quarter(panel, 3, ann).merge(yjyg, on=["code", "year"], how="inner")
    s3 = s3[s3["yjyg_date"].notna()].copy()
    s3["yjyg_type"] = s3["yjyg_type"].astype("category")
    feats_e = FEATURES + CAT + ["yjyg_pred", "yjyg_change", "yjyg_prev", "yjyg_type"]
    print(f"\n④ 业绩预告：有预告的样本 {len(s3)} | 预告公告日中位 {s3['yjyg_date'].median().date()}"
          f" | 距年报披露 {(s3['t_annual'] - s3['yjyg_date']).dt.days.median():.0f} 天")
    r4 = walkforward(s3, feats_e, "④ 业绩预告公布")
    r4["decision_date_median"] = str(s3[s3["year"].isin(TEST_YEARS)]["yjyg_date"].median().date())
    r4["lead_days_median"] = float((s3[s3["year"].isin(TEST_YEARS)]["t_annual"] - s3[s3["year"].isin(TEST_YEARS)]["yjyg_date"]).dt.days.median())
    out["④ 业绩预告公布"] = r4
    # 同一样本上、不给预告特征的对照（消融）
    r4b = walkforward(s3, FEATURES + CAT, "④b 同样本·不看预告")
    out["④b 同样本·不看预告"] = r4b
    print(f"④ 业绩预告: n={r4['pooled']['n']} AUC={r4['pooled']['auc']} macro={r4['macro_auc']} | "
          f"消融（同样本不给预告特征）AUC={r4b['pooled']['auc']}", flush=True)

    # ⑤ 年报（上界标定）
    s4 = samples_at_quarter(panel, 4, ann)
    r5 = walkforward(s4, ANNUAL + CAT, "⑤ 年报披露（=答案）")
    r5["decision_date_median"] = str(s4[s4["year"].isin(TEST_YEARS)]["t_decision"].median().date())
    r5["lead_days_median"] = 0.0
    out["⑤ 年报披露（=答案）"] = r5

    print("\n================ 提前量 → 判别力 ================")
    rows = []
    for k, v in out.items():
        rows.append({"决策时刻": k, "决策日中位": v.get("decision_date_median"), "距年报(天)": v.get("lead_days_median"),
                     "n": v["pooled"]["n"], "AUC": v["pooled"]["auc"], "macro AUC": v["macro_auc"],
                     "KS": v["pooled"]["ks"], "top10%": v["pooled"]["top10_rate"]})
    print(pd.DataFrame(rows).to_string(index=False))

    keep = {k: {kk: vv for kk, vv in v.items() if kk not in ("y", "p")} for k, v in out.items()}
    (ROOT / "data" / "leadtime_curve.json").write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(ROOT / "data" / "leadtime_preds.npz",
                        **{f"{k}__y": v["y"] for k, v in out.items()}, **{f"{k}__p": v["p"] for k, v in out.items()})
    print("->", ROOT / "data" / "leadtime_curve.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
