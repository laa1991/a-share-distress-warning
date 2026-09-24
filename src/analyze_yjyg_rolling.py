"""「业绩预告 → 滚动更新链」的取数底账：它到底能覆盖多少、提前多少天、内容可信到什么程度。

设计一份"能上线"的滚动链之前，先量四件事（都在测试年 2022–2025 上）：
1. **覆盖**：多少比例的样本会发业绩预告？（分子分母都要给，别只给比例）
2. **时点**：预告公告日相对**年报披露日**在哪（中位提前多少天）；相对**三季报**（上一个定期报告）呢？
3. **频次**：同一家公司同一年会发几次预告/修正（`yjyg_date` vs `yjyg_date_max`）？
4. **内容的分量**：预告说的"首亏/增亏/略减"等等，与最终"年报是否亏损"有多一致（这是那一格 0.9987 的来源）。

用法: python analyze_yjyg_rolling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import PANEL, make_samples  # noqa: E402
from run_leadtime_curve import base_label, load_yjyg  # noqa: E402

TEST_YEARS = [2022, 2023, 2024, 2025]

panel = pd.read_pickle(PANEL)
ann = base_label(panel)
s = make_samples(panel)
s = s[s["label"].notna() & s["t_decision"].notna() & s["t_label"].notna()].copy()
yj = load_yjyg()
t = s[s["year"].isin(TEST_YEARS)].merge(yj, on=["code", "year"], how="left")

t["t_ann"] = pd.to_datetime(t["t_label"], errors="coerce")
t["t_yjyg"] = pd.to_datetime(t["yjyg_date"], errors="coerce")
t["t_yjyg_max"] = pd.to_datetime(t["yjyg_date_max"], errors="coerce")
t["t_decision"] = pd.to_datetime(t["t_decision"], errors="coerce")
t["有预告"] = t["t_yjyg"].notna()
t["迟于年报"] = t["t_yjyg"] > t["t_ann"]
t["days_before_ann"] = (t["t_ann"] - t["t_yjyg"]).dt.days
t["days_after_q3"] = (t["t_yjyg"] - t["t_decision"]).dt.days
t["修正次数"] = ((t["t_yjyg_max"] - t["t_yjyg"]).dt.days > 0).astype(int)

print(f"测试年样本 {len(t)} · 有业绩预告 {int(t['有预告'].sum())} 条（{t['有预告'].mean():.2%}）")
print("\n【1. 覆盖：正例 vs 负例两边的比例都要给】")
cov = t.groupby("label")["有预告"].agg(["mean", "size"])
for lab, row in cov.iterrows():
    print(f"   {'真会亏' if lab == 1 else '不亏':6s} 有预告 {row['mean']:.2%}（n={int(row['size'])}）")

print("\n【2. 时点：预告公告日相对年报披露日 / 相对三季报】")
d = t[t["有预告"] & ~t["迟于年报"]]
print(f"  （{int(t['迟于年报'].sum())} 条预告的日期**晚于**年报，剔除后 {len(d)} 条）")
q = d["days_before_ann"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(0).astype(int)
print(f"  距年报天数：P5 {q.iloc[0]} · P25 {q.iloc[1]} · **中位 {q.iloc[2]}** · P75 {q.iloc[3]} · P95 {q.iloc[4]}")
q2 = d["days_after_q3"].quantile([0.05, 0.5, 0.95]).round(0).astype(int)
print(f"  距三季报（上一个定期报告）天数：P5 {q2.iloc[0]} · **中位 {q2.iloc[1]}** · P95 {q2.iloc[2]}")
print(f"  修正/多次预告的比例：{d['修正次数'].mean():.2%}")

print("\n【3. 内容的分量：预告类型 × 最终是否亏损】")
y = t[t["有预告"] & ~t["迟于年报"]].copy()
y["亏损预测"] = (pd.to_numeric(y["yjyg_pred"], errors="coerce") < 0).astype("Int64")
ct = pd.crosstab(y["yjyg_type"], y["label"], margins=True)
print(ct.to_string())
print(f"\n  预告里『预计亏损（预测数值<0）』的样本 {int((y['亏损预测'] == 1).sum())} 条，"
      f"其中最终真亏 {y[y['亏损预测'] == 1]['label'].mean():.2%}")
print(f"  预告里『预计不亏』的样本 {int((y['亏损预测'] == 0).sum())} 条，"
      f"其中最终真亏 {y[y['亏损预测'] == 0]['label'].mean():.2%}")

print("\n【4. 这一格的上限（把预告内容当输入）】")
from run_experiments import CAT, FEATURES, fit_predict, metrics  # noqa: E402
# ⚠️ 不要把 datetime 列丢给 LightGBM（会炸 dtype）——时点信息写成数值：距三季报多少天
# ⚠️ 用**全历史**建 s2（不能先按测试年过滤）：否则训练折/验证折会空掉，
#    只有 2024/2025 两个折能跑 —— 那样量到的数就与其它读数不在同一套折上，不可比。
s2 = s.merge(yj, on=["code", "year"], how="left")
s2["yjyg_lag"] = (pd.to_datetime(s2["yjyg_date"], errors="coerce")
                  - pd.to_datetime(s2["t_decision"], errors="coerce")).dt.days.astype(float)
s2["yjyg_type_code"] = s2["yjyg_type"].astype("category").cat.codes.astype(float)   # 字符串列不能直接喂 LightGBM
s2 = s2[s2["t_decision"].notna() & s2["t_label"].notna()]
F2 = FEATURES + CAT + ["yjyg_pred", "yjyg_change", "yjyg_type_code", "yjyg_lag"]
rows = []
for Y in TEST_YEARS:
    tr, va, te = s2[s2["year"] <= Y - 2], s2[s2["year"] == Y - 1], s2[s2["year"] == Y]
    if min(len(tr), len(va), len(te)) == 0:
        continue
    p, _ = fit_predict(tr, va, te, F2)
    rows.append({"Y": Y, "y": te["label"].values, "p": np.asarray(p),
                 "has": te["yjyg_date"].notna().values})      # ★ 预测与"这一格有没有"必须同源
print(f"  （可评年 {[r['Y'] for r in rows]}）")
sub_y, sub_p = [], []
for r in rows:
    m = r["has"]
    if m.sum() and len(np.unique(r["y"][m])) > 1:
        sub_y.append(r["y"][m]); sub_p.append(r["p"][m])
if sub_y:
    yy = np.concatenate(sub_y); pp = np.concatenate(sub_p)
    mm = metrics(yy, pp)
    print(f"  有预告子集（n={len(yy)}，正例率 {yy.mean():.2%}）AUC {mm['auc']:.4f}"
          f" · top-10% 命中 {mm['top10_rate']:.2%}")
    auc_sub = mm["auc"]
else:
    auc_sub = None
print(f"  对照：全体样本（n={len(t)}，正例率 {t['label'].mean():.2%}）A 臂 AUC 0.9103")

out = {
    "coverage": {str(int(k)): {"mean": round(float(v["mean"]), 4), "n": int(v["size"])} for k, v in cov.iterrows()},
    "days_before_ann": {str(k): int(v) for k, v in q.items()},
    "days_after_q3": {str(k): int(v) for k, v in q2.items()},
    "revision_rate": round(float(d["修正次数"].mean()), 4),
    "late_vs_ann": int(t["迟于年报"].sum()),
    "type_x_label": ct.to_dict(),
}
(ROOT / "data" / "yjyg_rolling.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n读数 -> data/yjyg_rolling.json")
