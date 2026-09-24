"""C 臂的对照口径复核：0.9246 vs 0.9103 是不是「同一批样本」比出来的？

问题：C 臂的测试集是随机 30% 子样本（n=5,908），A 臂的测试集是全部 2022–2025 样本（n=19,473）。
两个 n 不同 ⇒ 直接比 AUC 不干净。做法：把 A 臂的预测限制到**C 臂那批子样本**上再算一次 AUC。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, r"C:\dev\finlab-ml\src")
from run_experiments import FEATURES, CAT, make_samples, fit_predict, PANEL

s = make_samples(pd.read_pickle(PANEL))
s = s[s["label"].notna() & s["t_decision"].notna() & s["t_label"].notna()].reset_index(drop=True)
rng = np.random.default_rng(20260924)
m = rng.random(len(s)) < 0.70            # 与 run_experiments.py 完全同一把种子、同一个顺序
c_test_idx = np.where(~m)[0]
test_years = [2022, 2023, 2024, 2025]

feats = FEATURES + CAT
idx_all, y_all, p_all = [], [], []
for Y in test_years:
    tr, va, te = s[s["year"] <= Y - 2], s[s["year"] == Y - 1], s[s["year"] == Y]
    p, _ = fit_predict(tr, va, te, feats)
    idx_all.append(te.index.values); y_all.append(te["label"].values); p_all.append(p)
idx = np.concatenate(idx_all); y = np.concatenate(y_all); p = np.concatenate(p_all)

pos = {v: i for i, v in enumerate(idx)}
sel = np.array([pos[i] for i in c_test_idx if i in pos])
print("A 臂测试样本:", len(y), "| 其中落在 C 臂测试子样本里的:", len(sel))
res = {
    "A_on_full_test": round(float(roc_auc_score(y, p)), 4),
    "A_on_C_subset": round(float(roc_auc_score(y[sel], p[sel])), 4),
    "n_full": int(len(y)), "n_subset": int(len(sel)),
    "subset_pos_rate": round(float(y[sel].mean()), 4),
}
old = json.loads(Path(r"C:\dev\finlab-ml\data\results.json").read_text(encoding="utf-8"))
res["C_on_its_own_subset"] = old["C_random_split"]["on_same_test_years"]["auc"]
res["C_auc_minus_A_on_same_subset"] = round(res["C_on_its_own_subset"] - res["A_on_C_subset"], 4)
print(json.dumps(res, ensure_ascii=False, indent=2))
Path(r"C:\dev\finlab-ml\data\diag_c_arm.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
