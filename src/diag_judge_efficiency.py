"""「判断器」的效率账：**我们这一边**的实测数（延迟 / 吞吐 / 成本 / 复算性 / **校准**）。

要对齐的口径（外部那两个数的来源见 `docs/探索-判断器的效率（GBDT vs Jev vs LLM）.md`）：
- 延迟：单条 p50 / p95（ms）
- 成本：每千次预测（GBDT 的边际成本≈电费，报 0 并说明）
- **复算性**：同输入重跑是否逐位一致（GBDT 应当是 100%）
- **校准**：预测概率与真实频率对不对得上（分箱 reliability + Brier + **落在不确定带 [0.2,0.8] 的比例**）
  ⚠️ 最后这一格是关键：第三方对 Gemini 的实测显示它的概率**退化成 0.0/1.0**，
  于是"阈值 0.8 转人工"这类流程根本跑不起来 —— 我们要知道自己这边是什么样。

用法: python diag_judge_efficiency.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import CAT, FEATURES, PANEL, fit_predict, make_samples  # noqa: E402

TEST_YEARS = [2022, 2023, 2024, 2025]


def main() -> int:
    panel = pd.read_pickle(PANEL)
    s = make_samples(panel)
    s = s[s["label"].notna() & s["t_decision"].notna() & s["t_label"].notna()].copy()

    preds, train_sec = [], 0.0
    for Y in TEST_YEARS:
        tr, va, te = s[s["year"] <= Y - 2].copy(), s[s["year"] == Y - 1], s[s["year"] == Y].copy()
        if min(len(tr), len(va), len(te)) == 0:
            continue
        tr["label"] = tr["label"]
        t0 = time.perf_counter()
        p, booster = fit_predict(tr, va, te, FEATURES + CAT)
        train_sec += time.perf_counter() - t0
        d = te[["code", "year", "label"]].copy()
        d["p"] = np.asarray(p)
        preds.append(d)

    t = pd.concat(preds, ignore_index=True)
    print(f"测试集 {len(t)} 行 · 四个折训练+预测总耗时 {train_sec:.1f}s（含早停）")

    # ① 批量打分延迟
    X = s[s["year"].isin(TEST_YEARS)][FEATURES + CAT]
    import lightgbm as lgb
    booster = lgb.Booster(model_file=None) if False else None  # 占位，避免误用
    # 用最后一折的模型做延迟实测（重新训练一次，保持独立）
    tr, va, te = s[s["year"] <= 2023].copy(), s[s["year"] == 2024], s[s["year"] == 2025].copy()
    tr["label"] = tr["label"]
    p, booster = fit_predict(tr, va, te, FEATURES + CAT)
    Xt = te[FEATURES + CAT]
    t0 = time.perf_counter()
    _ = booster.predict(Xt)
    batch_sec = time.perf_counter() - t0
    print(f"\n【① 批量】{len(Xt)} 行一次打分 {batch_sec*1000:.1f} ms ⇒ {len(Xt)/batch_sec:,.0f} 行/秒")

    # ② 单条延迟（batch=1）
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xt), size=min(300, len(Xt)), replace=False)
    lat = []
    for i in idx:
        one = Xt.iloc[[i]]
        t0 = time.perf_counter()
        booster.predict(one)
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    print(f"【② 单条】n={len(lat)} 条：p50 {np.percentile(lat,50):.3f} ms · "
          f"p95 {np.percentile(lat,95):.3f} ms · 均值 {lat.mean():.3f} ms")

    # ③ 复算性：同输入重跑是否逐位一致
    a = booster.predict(Xt.iloc[[0, 1, 2]]) 
    b = booster.predict(Xt.iloc[[0, 1, 2]])
    same = bool(np.array_equal(a, b))
    print(f"【③ 复算性】同输入重跑逐位一致：{'✅ 是' if same else '❌ 否'}")

    # ④ 校准
    p_all = booster.predict(Xt)
    y_all = te["label"].values
    bins = np.linspace(0, 1, 11)
    idxb = np.clip(np.digitize(p_all, bins) - 1, 0, 9)
    rows = []
    for k in range(10):
        m = idxb == k
        if m.sum():
            rows.append({"箱": f"{bins[k]:.1f}-{bins[k+1]:.1f}", "n": int(m.sum()),
                         "预测均值": round(float(p_all[m].mean()), 4),
                         "实际频率": round(float(y_all[m].mean()), 4)})
    rel = pd.DataFrame(rows)
    brier = float(np.mean((p_all - y_all) ** 2))
    mid = float(((p_all >= 0.2) & (p_all <= 0.8)).mean())
    print(f"\n【④ 校准】Brier {brier:.4f} · **落在不确定带 [0.2,0.8] 的比例 {mid:.2%}**"
          f"（第三方对 Gemini 的同类实测是 0%：它的概率全是 0.0/1.0）")
    print(rel.to_string(index=False))
    lo = rel[rel["箱"].isin(["0.0-0.1", "0.9-1.0"])]
    print(f"  两端箱占比：{lo['n'].sum()/len(p_all):.1%}（GBDT 会把分数摊开，不会全挤在两端）")

    # ⑤ 校准能不能补上：在**验证折**上拟合 isotonic，再看测试折（一次验证折的代价）
    from sklearn.isotonic import IsotonicRegression
    pv = booster.predict(va[FEATURES + CAT])
    iso = IsotonicRegression(out_of_bounds="clip").fit(pv, va["label"].values)
    pc = iso.predict(p_all)
    brier_c = float(np.mean((pc - y_all) ** 2))
    rows_c = []
    for k in range(10):
        m = idxb == k
        if m.sum():
            rows_c.append({"箱": f"{bins[k]:.1f}-{bins[k+1]:.1f}", "n": int(m.sum()),
                           "校准前_预测均值": round(float(p_all[m].mean()), 4),
                           "校准后_预测均值": round(float(pc[m].mean()), 4),
                           "实际频率": round(float(y_all[m].mean()), 4)})
    print(f"\n【⑤ 补一次校准（isotonic，拟合在验证折上）】Brier {brier:.4f} → **{brier_c:.4f}**")
    print(pd.DataFrame(rows_c).to_string(index=False))

    out = {"n": int(len(t)),
           "train_sec_4folds": round(train_sec, 2),
           "batch_rows": int(len(Xt)), "batch_ms": round(batch_sec * 1000, 2),
           "rows_per_sec": round(len(Xt) / batch_sec, 1),
           "single_p50_ms": round(float(np.percentile(lat, 50)), 3),
           "single_p95_ms": round(float(np.percentile(lat, 95)), 3),
           "reproducible_bitwise": same,
           "brier": round(brier, 4),
           "uncertain_band_share": round(mid, 4),
           "reliability": rows}
    (ROOT / "data" / "judge_efficiency.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n读数 -> data/judge_efficiency.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
