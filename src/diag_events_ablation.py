"""消融实验：把报表外事件（诉讼 / 担保）加进 ① 一季报那一格，分到底涨不涨？

**同批样本、同一套折、只差这几列特征** —— 与四格对照同一条纪律。
除了 AUC，还报三个业务读数：前 10% 名单的**精确率**、**覆盖率**（抓住了多少亏损公司）、
以及**只有模型看不见的那一半**（① 分位 < 0.9 的样本）里覆盖率有没有变化。

用法: python diag_events_ablation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from events_features import EVENT_FEATURES, build as build_events  # noqa: E402
from run_experiments import CAT, FEATURES, PARAMS, fit_predict, metrics  # noqa: E402
from run_leadtime_curve import base_label, samples_at_quarter  # noqa: E402

TEST_YEARS = [2022, 2023, 2024, 2025]


def run(s: pd.DataFrame, feats: list[str]) -> tuple[dict, pd.DataFrame]:
    rows, preds = [], []
    for Y in TEST_YEARS:
        tr, va, te = s[s["year"] <= Y - 2], s[s["year"] == Y - 1], s[s["year"] == Y]
        if min(len(tr), len(va), len(te)) == 0 or tr["label"].nunique() < 2:
            continue
        p, _ = fit_predict(tr, va, te, feats)
        te = te.copy()
        te["p"] = np.clip(p, 0, 1)
        preds.append(te[["code", "year", "label", "p"]])
        rows.append((np.array([Y] * len(te)), te["label"].values, te["p"].values))
    y = np.concatenate([r[1] for r in rows])
    p = np.concatenate([r[2] for r in rows])
    per_year = {int(r[0][0]): metrics(r[1], r[2]) for r in rows}
    return {"pooled": metrics(y, p),
            "macro_auc": round(float(np.mean([m["auc"] for m in per_year.values()])), 4),
            "per_year": per_year}, pd.concat(preds, ignore_index=True)


def biz(pred: pd.DataFrame, tag: str) -> dict:
    """前 10% 名单的三个业务读数"""
    out = {}
    for Y, g in pred.groupby("year"):
        k = max(int(round(0.10 * len(g))), 1)
        sel = g.nlargest(k, "p")
        out[int(Y)] = {"precision": round(float(sel["label"].mean()), 4),
                       "coverage": round(float(sel["label"].sum() / max(g["label"].sum(), 1)), 4),
                       "n": int(len(g)), "n_pos": int(g["label"].sum())}
    agg = pd.DataFrame(out).T
    return {"per_year": out,
            "precision_mean": round(float(agg["precision"].mean()), 4),
            "coverage_mean": round(float(agg["coverage"].mean()), 4),
            "coverage_2022": out.get(2022, {}).get("coverage"),
            "tag": tag}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1, help="跑几个种子（判断 +0.00x 是不是噪声）")
    args = ap.parse_args()

    panel = pd.read_pickle(ROOT / "data" / "panel.pkl")
    ann = base_label(panel)
    s = samples_at_quarter(panel, 1, ann)
    ev = build_events()
    s = s.merge(ev, on=["code", "year"], how="left")
    for c in EVENT_FEATURES:
        s[c] = s[c].fillna(0.0)
    print(f"样本 {len(s)}（测试年 {TEST_YEARS}）· 事件覆盖：诉讼 {s['sue_any'].mean():.2%} · "
          f"担保≥净资产50% {s['gua_heavy'].mean():.2%}")

    base_seed = PARAMS["seed"]
    deltas = []
    for k in range(args.seeds):
        PARAMS["seed"] = base_seed + k * 101
        a_m, a_p = run(s, FEATURES + CAT)
        b_m, b_p = run(s, FEATURES + CAT + EVENT_FEATURES)
        d = {"seed": PARAMS["seed"], "auc_a": a_m["pooled"]["auc"], "auc_b": b_m["pooled"]["auc"],
             "delta_auc": round(b_m["pooled"]["auc"] - a_m["pooled"]["auc"], 5),
             "delta_prec": round(b_m["pooled"]["top10_rate"] - a_m["pooled"]["top10_rate"], 5),
             "cov_a": biz(a_p, "a")["coverage_mean"], "cov_b": biz(b_p, "b")["coverage_mean"]}
        d["delta_cov"] = round(d["cov_b"] - d["cov_a"], 5)
        deltas.append(d)
        print(f"  种子 {PARAMS['seed']}: AUC {d['auc_a']:.4f} → {d['auc_b']:.4f}（{d['delta_auc']:+.5f}）· "
              f"top10 精确率 {d['delta_prec']:+.5f} · 覆盖率 {d['delta_cov']:+.5f}", flush=True)
    if args.seeds > 1:
        dd = pd.DataFrame(deltas)
        print(f"\n★ {args.seeds} 个种子的 ΔAUC：{dd['delta_auc'].mean():+.5f} ± {dd['delta_auc'].std():.5f}"
              f"（范围 {dd['delta_auc'].min():+.5f} – {dd['delta_auc'].max():+.5f}）")
        print(f"★ 同一个种子下 A 臂自身：AUC 极差 {dd['auc_a'].max() - dd['auc_a'].min():.5f}")
        (ROOT / "data" / "events_ablation_seeds.json").write_text(
            json.dumps(deltas, ensure_ascii=False, indent=2), encoding="utf-8")
        print("读数 -> data/events_ablation_seeds.json")
    PARAMS["seed"] = base_seed
    return 0

    res = {}
    for tag, feats in (("A 只用财报", FEATURES + CAT),
                       ("A+事件 加入报表外特征", FEATURES + CAT + EVENT_FEATURES),
                       ("仅事件（看它自己有多强）", EVENT_FEATURES)):
        m, pred = run(s, feats)
        b = biz(pred, tag)
        res[tag] = {"model": m, "biz": b}
        print(f"\n{tag}")
        print(f"  pooled AUC {m['pooled']['auc']} · macro {m['macro_auc']} · KS {m['pooled']['ks']} · "
              f"top10 精确率 {m['pooled']['top10_rate']}")
        print(f"  前 10% 名单：精确率 {b['precision_mean']:.2%} · 覆盖率 {b['coverage_mean']:.2%}"
              f"（逐年 {[b['per_year'][y]['coverage'] for y in sorted(b['per_year'])]}）")

    a, b = res["A 只用财报"]["model"]["pooled"], res["A+事件 加入报表外特征"]["model"]["pooled"]
    print(f"\n★ 差（同批样本、同一套折）：AUC {b['auc'] - a['auc']:+.4f} · KS {b['ks'] - a['ks']:+.4f} · "
          f"top10 精确率 {b['top10_rate'] - a['top10_rate']:+.4f}")
    cov_a, cov_b = res["A 只用财报"]["biz"]["coverage_mean"], res["A+事件 加入报表外特征"]["biz"]["coverage_mean"]
    print(f"★ 覆盖率（前 10% 名单能抓住多少亏损公司）：{cov_a:.2%} → {cov_b:.2%}（{cov_b - cov_a:+.2%}）")

    (ROOT / "data" / "events_ablation.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                                                        encoding="utf-8")
    print("\n读数 -> data/events_ablation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
