"""用一张八行的表把 GBDT/LightGBM 走一遍 —— 每一步的中间结果都打印出来（教具，不进实验流程）。

八家公司，两个特征（ROA%、应收账款占收入%），标签 = 当年是否亏损。
超参故意开到最小：3 棵树、每棵只有一次分裂（max_depth=1）、学习率 0.5 ⇒ 每一步都手算得出来。

用法: python tutorial_gbdt_8rows.py
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent

df = pd.DataFrame({
    "公司": list("ABCDEFGH"),
    "ROA%": [8.0, 6.5, 5.0, 3.0, 2.5, 1.0, -1.0, -2.0],
    "应收占比%": [10, 12, 20, 25, 45, 50, 30, 60],
    "真值y": [0, 0, 0, 0, 1, 1, 1, 1],
})
X, y = df[["ROA%", "应收占比%"]], df["真值y"]

print("=" * 78)
print("第 0 步 · 数据（8 行 / 2 个特征 / 1 个标签）")
print(df.to_string(index=False))

model = lgb.LGBMClassifier(
    n_estimators=3, learning_rate=0.5, max_depth=1, min_child_samples=1,
    min_split_gain=0.0, reg_lambda=0.0, verbose=-1, random_state=0,
)
model.fit(X, y)

print("\n" + "=" * 78)
print("每一轮之后的分数与概率（raw = 树的加总；p = sigmoid(raw)）")
rows = []
for k in range(0, 4):
    raw = model.predict(X, raw_score=True, num_iteration=k) if k else np.zeros(len(X))
    p = 1 / (1 + np.exp(-raw))
    rows.append(pd.DataFrame({"公司": df["公司"], "y": y, f"raw{k}": raw.round(3), f"p{k}": p.round(3)}))
table = rows[0][["公司", "y", "raw0", "p0"]]
for k in (1, 2, 3):
    table[f"raw{k}"] = rows[k][f"raw{k}"]
    table[f"p{k}"] = rows[k][f"p{k}"]
    table[f"AUC{k}"] = round(roc_auc_score(y, rows[k][f"p{k}"]), 4)
print(table.to_string(index=False))

print("\n" + "=" * 78)
print("三棵树长什么样（threshold 是切分点，leaf 是落进这一侧要加多少分）")
dump = model.booster_.dump_model()
for i, tree in enumerate(dump["tree_info"], 1):
    node = tree["tree_structure"]
    if "split_feature" not in node:
        print(f"  第 {i} 棵：只有叶子（没分成）")
        continue
    feat = X.columns[node["split_feature"]]
    print(f"  第 {i} 棵：{feat} ≤ {node['threshold']:.3f} ?")
    for side, name in (("left_child", "是"), ("right_child", "否")):
        leaf = node[side]
        print(f"       {name} → 加 {leaf['leaf_value']:+.4f} 分（{leaf['leaf_count']} 条样本）")

print("\n" + "=" * 78)
print("最后一行的加总（看 Python 里的加法对得上对不上）")
raw_last = model.predict(X, raw_score=True)
print("  raw  =", np.round(raw_last, 4))
print("  p    = 1/(1+e^-raw) =", np.round(1 / (1 + np.exp(-raw_last)), 4))
print(f"  AUC（3 轮之后）= {roc_auc_score(y, raw_last):.4f}")

(ROOT / "data" / "tutorial_gbdt.json").write_text(
    json.dumps({"table": table.to_dict("records"),
                "trees": [t["tree_structure"] for t in dump["tree_info"]]}, ensure_ascii=False, indent=2,
               default=str), encoding="utf-8")
print("\n读数 -> data/tutorial_gbdt.json")
