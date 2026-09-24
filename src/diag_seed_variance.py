"""真臂的种子方差：A / B 两臂各换 3 个种子，量「+3.4 分」是真差还是种子噪声。

用法: python diag_seed_variance.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"C:\dev\finlab-ml\src")
from run_experiments import FEATURES, CAT, PARAMS, make_samples, PANEL
import lightgbm as lgb

s = make_samples(pd.read_pickle(PANEL))
s = s[s["label"].notna() & s["t_decision"].notna() & s["t_label"].notna()]
feats_a = FEATURES + CAT
feats_b = feats_a + [f"fut_{c}" for c in FEATURES]
test_years = [2022, 2023, 2024, 2025]


def run(feats, seed):
    yt, yp, per = [], [], {}
    for Y in test_years:
        tr, va, te = s[s["year"] <= Y - 2], s[s["year"] == Y - 1], s[s["year"] == Y]
        prm = dict(PARAMS, seed=seed)
        ds_tr = lgb.Dataset(tr[feats], label=tr["label"].values, categorical_feature=["industry"])
        ds_va = lgb.Dataset(va[feats], label=va["label"].values, categorical_feature=["industry"], reference=ds_tr)
        b = lgb.train(prm, ds_tr, num_boost_round=2000, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        p = b.predict(te[feats])
        yt.append(te["label"].values); yp.append(p)
        per[Y] = round(float(roc_auc_score(te["label"].values, p)), 4)
    return round(float(roc_auc_score(np.concatenate(yt), np.concatenate(yp))), 4), per


out = {}
for tag, feats in (("A", feats_a), ("B", feats_b)):
    out[tag] = {}
    for seed in (20260924, 1, 777):
        auc, per = run(feats, seed)
        out[tag][seed] = {"pooled_auc": auc, "per_year": per}
        print(tag, seed, auc, per, flush=True)

for tag in ("A", "B"):
    aucs = [v["pooled_auc"] for v in out[tag].values()]
    print(f"{tag} 臂 pooled AUC: 均值 {np.mean(aucs):.4f} 极差 {np.ptp(aucs):.4f} 明细 {aucs}")
print("A→B 提升（三种子配对）:",
      [round(out['B'][k]['pooled_auc'] - out['A'][k]['pooled_auc'], 4) for k in out['A']])
Path(r"C:\dev\finlab-ml\data\seed_variance.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                            encoding="utf-8")
