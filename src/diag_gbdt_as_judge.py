"""GBDT 能不能当"判断器"用？—— 在同一批 120 道题、同一份裁决真值上跑一遍。

**任务**：看一份公告该被归到 A/B/C/E 哪一类（"为什么戴帽"）。
**GBDT 手里只有什么**：那只股票在**该公告日之前已披露**的财务列（69 个数值列 + 行业）——**没有公告正文**。
**PIT 纪律**：每道题只取 `disc_actual <= 案件日期` 里**最近一期**的面板行（用巨潮实际披露日，不用公告源日期）。

预注册（写下来是为了被推翻）：我猜 **GBDT 在这 120 道上落在多数类基线附近（50–60%）**，
远低于两个判断器的 95.8–98.2%；而它的**不确定带会明显大于 0**（这是它与 LLM 那两家相反的强项）。
"""
import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = Path(r"C:\dev\finlab-ml")
A = ROOT / "data" / "judge_arena"

panel = pickle.loads((ROOT / "data" / "panel.pkl").read_bytes())
panel["disc_actual"] = pd.to_datetime(panel["disc_actual"], errors="coerce")
panel["period"] = panel["period"].astype(str)

gold = pd.read_csv(A / "st_reason_gold_labels.v6-120.csv", dtype=str).fillna("")
cases = pd.read_csv(A / "st_reason_cases.csv", dtype=str).fillna("")[["id"]]
g = gold[["id", "裁定"]]

rows, miss = [], 0
for _, r in g.iterrows():
    code, date = r["id"].split("_")
    d = pd.Timestamp(date)
    cand = panel[(panel["code"] == code) & (panel["disc_actual"].notna()) & (panel["disc_actual"] <= d)]
    if cand.empty:
        miss += 1
        continue
    row = cand.sort_values("period").iloc[-1]
    feat = {c: row[c] for c in panel.select_dtypes("number").columns
            if c not in ("code",)}
    feat.update({"industry": row["industry"], "code": code, "y": r["裁定"], "asof": row["disc_actual"], "period": row["period"]})
    rows.append(feat)

df = pd.DataFrame(rows)
print(f"拼出 {len(df)} 行（缺面板 {miss} 条）· 特征 {df.shape[1] - 5} 个 · 用到的最近一期分布：")
print("  ", df["period"].value_counts().sort_index().tail(6).to_dict())
print(f"  题日 − 用到的披露日：中位 {int((pd.to_datetime(df['id'].str.split('_').str[1]) - df['asof']).dt.days.median())} 天"
      if "id" in df else "")

drop = ["y", "asof", "period", "code"]
X = df.drop(columns=drop)
y = df["y"]
for c in X.columns:
    if X[c].dtype == object and c != "industry":
        X[c] = pd.to_numeric(X[c], errors="coerce")
X["industry"] = X["industry"].astype("category")   # LightGBM 要求 category dtype
cat = ["industry"]

print(f"\n标签分布：{y.value_counts().to_dict()} · **多数类基线（全猜 {y.value_counts().idxmax()}）= {y.value_counts().max()/len(y):.4f}**")

accs, f1s, bands, confs = [], [], [], []
for seed in (0, 1, 2):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=object)
    proba = np.zeros((len(y), len(y.unique())))
    for tr, te in skf.split(X, y):
        clf = lgb.LGBMClassifier(objective="multiclass", num_class=len(y.unique()),
                                 n_estimators=300, learning_rate=0.05, num_leaves=15,
                                 min_child_samples=5, random_state=seed, verbose=-1)
        clf.fit(X.iloc[tr], y.iloc[tr], categorical_feature=cat)
        p = clf.predict_proba(X.iloc[te])
        proba[te] = p
        pred[te] = clf.classes_[p.argmax(axis=1)]
    acc = float((pred == y.values).mean())
    accs.append(acc)
    from sklearn.metrics import f1_score
    f1s.append(float(f1_score(y, pred, average="macro")))
    mx = proba.max(axis=1)
    bands.append(float(((mx >= 0.2) & (mx <= 0.8)).mean()))
    confs.append(float(np.median(mx)))
    if seed == 0:
        conf0 = pd.crosstab(y, pred).to_string()

print(f"\n【5 折 × 3 种子】准确率 {np.mean(accs):.4f} ± {np.std(accs):.4f}（{['%.4f' % a for a in accs]}）")
print(f"  macro-F1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"  最大概率中位 {np.mean(confs):.3f} · **落在不确定带 [0.2,0.8] 的占比 {np.mean(bands):.2%}**")
print(f"\n种子 0 的混淆（行=真值，列=预测）：\n{conf0}")

out = {"n": int(len(y)), "majority_baseline": round(float(y.value_counts().max() / len(y)), 4),
       "acc_mean": round(float(np.mean(accs)), 4), "acc_sd": round(float(np.std(accs)), 4),
       "macro_f1_mean": round(float(np.mean(f1s)), 4),
       "max_prob_median": round(float(np.mean(confs)), 3),
       "uncertain_band_share": round(float(np.mean(bands)), 4),
       "feature_count": int(X.shape[1]), "seeds": [0, 1, 2]}
(A / "gbdt_as_judge.json").write_text(__import__("json").dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n读数 -> {A / 'gbdt_as_judge.json'}")
