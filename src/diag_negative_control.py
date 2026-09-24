"""负对照的分布：换 5 个打乱种子，看同年 AUC 的散布 —— 判断 0.53 是「结构性偏置」还是「折噪声」。

判据：若 5 个种子的分布跨过 0.5 两侧、且包含 0.5，则实测值落在这个分布里 = 折噪声；
      若系统性落在 0.5 上侧，则管道里有我不知道的结构（要写进结论）。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.insert(0, r"C:\dev\finlab-ml\src")
from run_experiments import FEATURES, CAT, PARAMS, make_samples, fit_predict, PANEL

s = make_samples(pd.read_pickle(PANEL))
s = s[s["label"].notna() & s["t_decision"].notna() & s["t_label"].notna()]
feats = FEATURES + CAT

rows = []
for seed in (7, 11, 23, 42, 99):
    for Y in (2023, 2025):
        tr, va, te = s[s["year"] <= Y - 2], s[s["year"] == Y - 1], s[s["year"] == Y]
        y = np.random.default_rng(seed).permutation(tr["label"].values)
        ds_tr = lgb.Dataset(tr[feats], label=y, categorical_feature=["industry"])
        ds_va = lgb.Dataset(va[feats], label=va["label"].values, categorical_feature=["industry"], reference=ds_tr)
        b = lgb.train(PARAMS, ds_tr, num_boost_round=2000, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        p = b.predict(te[feats])
        rows.append({"seed": seed, "year": Y, "auc": round(roc_auc_score(te["label"].values, p), 4),
                     "best_iter": b.best_iteration})
        print(rows[-1], flush=True)

df = pd.DataFrame(rows)
print("\n按年汇总:")
print(df.groupby("year")["auc"].agg(["mean", "std", "min", "max"]).round(4).to_string())
print("\n跨年合并: 均值 %.4f 标准差 %.4f 最小 %.4f 最大 %.4f" %
      (df["auc"].mean(), df["auc"].std(), df["auc"].min(), df["auc"].max()))
