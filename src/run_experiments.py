"""LightGBM 落地实验：上市公司财务风险预警 —— 以及「数据用对没有」值多少分。

样本：`(股票代码, 年份 Y)`，特征取 **Y 年一季报**（决策时刻 = 一季报实际披露日，通常 4 月），
标签 = **Y 年年报净利润为负**。

四格对照（同一份数据 · 同一套超参 · 只改「数据用对没有」）：
  A 诚实（PIT + 走查）：特征 = 一季报；切分按披露日逐年游走（训 ≤ Y-2 / 验 Y-1 / 测 Y）
  B 偷看未来一期   ：特征 = 一季报 + **半年报**（8 月才披露，4 月不存在）
  C 随机切分        ：同一份特征，训练集是全体样本里随机 70%（含未来年份）
  D 负对照          ：A 的结构，训练标签打乱 ⇒ AUC 应回到 0.5 附近
基线：会计持续性（拿「上年是否亏损」当分数）。

口径提醒：同一年的样本来自同一折模型，**不同折的分数不在同一把尺上** ⇒ 报两种口径：
  · pooled：把各折测试样本拼起来算一次（跨年会混进「折与折的刻度差」）
  · macro ：逐年算 AUC 再取平均（只比同年内的排序）
两者都报，别只报好看的那个。

用法: python run_experiments.py [--test-start 2022] [--test-end 2025]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "panel.pkl"

FEATURES = [
    "eps", "rev", "rev_yoy", "rev_qoq", "np", "np_yoy", "np_qoq", "bps", "roe", "ocfps", "gross_margin",
    "cash", "ar", "inventory", "total_assets", "total_assets_yoy", "ap", "advance_receipts",
    "total_liab", "total_liab_yoy", "debt_ratio", "equity",
    "np_l", "np_yoy_l", "rev_l", "rev_yoy_l", "op_expense", "sell_exp", "admin_exp", "fin_exp",
    "op_expense_total", "op_profit", "total_profit",
    "net_cf", "net_cf_yoy", "ocf", "ocf_ratio", "icf", "icf_ratio", "fcf", "fcf_ratio",
    "np_margin", "ocf_to_assets", "ocf_to_rev", "ar_ratio", "inv_ratio", "cash_ratio",
    "equity_ratio", "ap_ratio", "sell_exp_ratio", "admin_exp_ratio", "fin_exp_ratio",
    "op_profit_margin", "log_assets", "log_rev",
    "roe_lag1", "debt_ratio_lag1", "np_margin_lag1", "ocf_to_assets_lag1", "rev_yoy_lag1",
    "ar_ratio_lag1", "inv_ratio_lag1",
    "d_roe", "d_debt_ratio", "d_np_margin",
]
CAT = ["industry"]

PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, num_threads=4, seed=20260924,
)
N_ROUNDS, EARLY_STOP = 2000, 100


def make_samples(panel: pd.DataFrame) -> pd.DataFrame:
    q1 = panel[panel["q"] == 1].copy()
    q1 = q1.rename(columns={"disc_actual": "t_decision", "period": "p_q1"})
    s = q1[["code", "p_q1", "t_decision"] + FEATURES + CAT].copy()
    s["year"] = s["p_q1"] // 10000

    ann = panel[panel["period"] % 10000 == 1231][["code", "period", "np", "disc_actual"]].rename(
        columns={"np": "np_annual", "disc_actual": "t_label"})
    ann["year"] = ann["period"] // 10000
    s = s.merge(ann[["code", "year", "np_annual", "t_label"]], on=["code", "year"], how="left")

    ann_prev = ann[["code", "year", "np_annual"]].rename(columns={"np_annual": "np_prev"})
    ann_prev["year"] += 1
    s = s.merge(ann_prev, on=["code", "year"], how="left")
    s["prev_loss"] = (s["np_prev"] < 0).astype("Int64")

    q2 = panel[panel["q"] == 2][["code", "period", "disc_actual"] + FEATURES].copy()
    q2["year"] = q2["period"] // 10000
    q2 = q2.rename(columns={c: f"fut_{c}" for c in FEATURES + ["disc_actual"]})
    s = s.merge(q2[["code", "year"] + [c for c in q2.columns if c.startswith("fut_")]], on=["code", "year"], how="left")

    s["label"] = np.where(s["np_annual"].isna(), np.nan, (s["np_annual"] < 0).astype(float))
    s["industry"] = s["industry"].astype("category")
    return s


def metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    if len(np.unique(y)) < 2:
        return {}
    fpr, tpr, _ = roc_curve(y, p)
    k = max(int(0.10 * len(y)), 1)
    top = np.argsort(-p)[:k]
    return {"n": int(len(y)), "pos": int(y.sum()), "pos_rate": round(float(y.mean()), 4),
            "auc": round(float(roc_auc_score(y, p)), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "ks": round(float(np.max(tpr - fpr)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "top10_rate": round(float(y[top].mean()), 4),
            "top10_lift": round(float(y[top].mean() / max(y.mean(), 1e-9)), 2)}


def fit_predict(tr, va, te, feats, shuffle=False):
    y = tr["label"].values
    if shuffle:
        y = np.random.default_rng(7).permutation(y)
    ds_tr = lgb.Dataset(tr[feats], label=y, categorical_feature=[c for c in CAT if c in feats])
    ds_va = lgb.Dataset(va[feats], label=va["label"].values,
                        categorical_feature=[c for c in CAT if c in feats], reference=ds_tr)
    booster = lgb.train(PARAMS, ds_tr, num_boost_round=N_ROUNDS, valid_sets=[ds_va],
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return booster.predict(te[feats]), booster


def pooled_and_macro(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    """rows = [(year_arr, y_arr, p_arr), ...]；同时给 pooled 与 macro 两种口径"""
    if not rows:
        return {}
    y_all = np.concatenate([r[1] for r in rows])
    p_all = np.concatenate([r[2] for r in rows])
    per_year = {int(r[0][0]): metrics(r[1], r[2]) for r in rows}
    aucs = [m["auc"] for m in per_year.values() if m]
    return {"pooled": metrics(y_all, p_all), "macro_auc": round(float(np.mean(aucs)), 4) if aucs else None,
            "per_year": per_year}


def run_walkforward(s: pd.DataFrame, feats: list[str], test_years: list[int]) -> tuple[dict, list, object]:
    rows, boosters = [], []
    for Y in test_years:
        tr, va, te = s[s["year"] <= Y - 2], s[s["year"] == Y - 1], s[s["year"] == Y]
        if min(len(tr), len(va), len(te)) == 0 or tr["label"].nunique() < 2:
            continue
        p, b = fit_predict(tr, va, te, feats)
        rows.append((te["year"].values, te["label"].values, p))
        boosters.append(b)
    return pooled_and_macro(rows), rows, (boosters[-1] if boosters else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--test-start", type=int, default=2022)
    ap.add_argument("--test-end", type=int, default=2025)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_pickle(PANEL)
    s = make_samples(panel)
    print(f"样本行数（一季报）: {len(s)} | 有标签: {int(s['label'].notna().sum())} "
          f"| 正例率: {float(s['label'].mean()):.4f}")
    s = s[s["label"].notna() & s["t_decision"].notna() & s["t_label"].notna()]
    print(f"可用样本: {len(s)} | 年份 {int(s['year'].min())}-{int(s['year'].max())} "
          f"| 决策时刻 {s['t_decision'].min().date()} ~ {s['t_decision'].max().date()}")

    # 「未来信息」要等多久：B 臂那期数据（半年报）距决策时刻多少天
    gap = (s["fut_disc_actual"] - s["t_decision"]).dt.days
    print(f"B 臂那期数据（半年报）距决策时刻: 中位 {gap.median():.0f} 天 "
          f"(p10={gap.quantile(.1):.0f}, p90={gap.quantile(.9):.0f}) —— 4 月做决策时它还不存在")

    test_years = list(range(args.test_start, args.test_end + 1))
    feats_a = FEATURES + CAT
    feats_b = feats_a + [f"fut_{c}" for c in FEATURES]
    out, preds = {}, {}

    for tag, feats, label in (("A_pit_walkforward", feats_a, "A 诚实（PIT + 走查）"),
                              ("B_lookahead_next_report", feats_b, "B 偷看未来一期（+半年报）")):
        print(f"[{label}] ...", flush=True)
        res, rows, booster = run_walkforward(s, feats, test_years)
        out[tag] = {"arm": tag, "n_features": len(feats), **res}
        preds[tag] = rows
        if tag == "A_pit_walkforward":
            imp = pd.DataFrame({"feature": booster.feature_name(),
                                "gain": booster.feature_importance("gain")}).sort_values("gain", ascending=False)
            out["importance_A"] = imp.head(25).to_dict("records")
            print("A 臂 top8 特征(gain):", ", ".join(f"{r.feature}" for r in imp.head(8).itertuples()))

    print("[D 负对照（训练标签打乱）] ...", flush=True)
    rows_d = []
    for Y in test_years:
        tr, va, te = s[s["year"] <= Y - 2], s[s["year"] == Y - 1], s[s["year"] == Y]
        if min(len(tr), len(va), len(te)) == 0:
            continue
        p, _ = fit_predict(tr, va, te, feats_a, shuffle=True)
        rows_d.append((te["year"].values, te["label"].values, p))
    out["D_shuffled_label"] = {"arm": "D_shuffled_label", **pooled_and_macro(rows_d)}
    preds["D_shuffled_label"] = rows_d

    print("[C 随机切分（含未来年份）] ...", flush=True)
    rng = np.random.default_rng(20260924)
    m = rng.random(len(s)) < 0.70
    tr, te = s[m], s[~m]
    p_c, _ = fit_predict(tr, tr.sample(frac=0.2, random_state=1), te, feats_a)
    te = te.assign(pred=p_c)
    out["C_random_split"] = {
        "arm": "C_random_split", "note": "训练集含未来年份样本（随机 70%）",
        "on_random_test": metrics(te["label"].values, te["pred"].values),
        "on_same_test_years": metrics(te.loc[te["year"].isin(test_years), "label"].values,
                                      te.loc[te["year"].isin(test_years), "pred"].values),
    }
    sub = te[te["year"].isin(test_years)]
    preds["C_random_split"] = [(sub["year"].values, sub["label"].values, sub["pred"].values)]

    mask = s["prev_loss"].notna() & s["year"].isin(test_years)
    yb = s.loc[mask, "label"].values
    pb = s.loc[mask, "prev_loss"].astype(float).values
    pb = pb + np.random.default_rng(3).random(len(pb)) * 1e-6      # 打破并列，便于算 AUC
    out["baseline_prev_loss"] = {"arm": "baseline_prev_loss", "note": "分数 = 上年是否亏损",
                                 "overall": metrics(yb, pb)}
    preds["baseline_prev_loss"] = [(s.loc[mask, "year"].values, yb, pb)]

    # ---- 汇总 ----
    print("\n================ 结果（同一个测试集 / 同一个标签） ================")
    tbl = []
    for k in ("A_pit_walkforward", "B_lookahead_next_report", "C_random_split", "D_shuffled_label",
              "baseline_prev_loss"):
        r = out[k]
        p = r.get("pooled", r.get("on_same_test_years") if k == "C_random_split" else r.get("overall", {}))
        tbl.append({"臂": k, "口径": "同一批测试年", "n": p.get("n"), "正例率": p.get("pos_rate"),
                    "AUC": p.get("auc"), "PR-AUC": p.get("pr_auc"), "KS": p.get("ks"),
                    "top10%风险率": p.get("top10_rate"), "lift": p.get("top10_lift"),
                    "macro AUC": r.get("macro_auc")})
    print(pd.DataFrame(tbl).to_string(index=False))

    print("\n分年 AUC（同一批测试样本）:")
    per = {}
    for k in ("A_pit_walkforward", "B_lookahead_next_report", "D_shuffled_label"):
        per[k] = {y: m.get("auc") for y, m in out[k]["per_year"].items()}
    per["C_random_split"] = {y: m.get("auc") for y, m in
                             pooled_and_macro(preds["C_random_split"])["per_year"].items()}
    print(pd.DataFrame(per).to_string())

    (outdir / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(outdir / "preds.npz",
                        **{f"{k}__{i}__y": r[1] for k, rows in preds.items() for i, r in enumerate(rows)},
                        **{f"{k}__{i}__p": r[2] for k, rows in preds.items() for i, r in enumerate(rows)},
                        **{f"{k}__{i}__yr": r[0] for k, rows in preds.items() for i, r in enumerate(rows)})
    print(f"\n结果 -> {outdir / 'results.json'} · 预测 -> {outdir / 'preds.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
