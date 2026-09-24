"""把 rev_dir 的 30 条人工裁定**按 id** 重新贴回新的 120 行标签表（行序变了，索引贴法会错位）。

形状：上一版 `fill_rev_gold.py` 是按**行号**写标签的；题量从 60 扩到 120 之后模板行序变了
⇒ 再跑一次就会把标签贴到别人身上。改成 id 键，一次修掉。
"""
from pathlib import Path

import pandas as pd

A = Path(r"C:\dev\finlab-ml\data\judge_arena")
old = pd.read_csv(A / "rev_dir_gold_labels.csv", dtype=str).fillna("")
filled = old[old["人工裁定"].astype(str).str.strip() != ""]
print(f"旧表 {len(old)} 行 · 其中已填 {len(filled)} 条")
cols = [c for c in ("人工裁定", "刻度", "裁定理由") if c in old.columns]

tpl = pd.read_csv(A / "rev_dir_gold_template.csv", dtype=str).fillna("")
out = tpl[["id", "truth_rule"]].copy()
for c in cols:
    out[c] = ""
m = filled.set_index("id")
hit = 0
for i, r in out.iterrows():
    if r["id"] in m.index:
        for c in cols:
            out.at[i, c] = m.loc[r["id"], c]
        hit += 1
out.to_csv(A / "rev_dir_gold_labels.csv", index=False, encoding="utf-8-sig")
print(f"按 id 贴回 {hit} 条（应为 {len(filled)}）· 新表 {len(out)} 行")
print("抽查：", out[out["人工裁定"] != ""].head(3)[["id", "人工裁定", "刻度"]].to_dict("records"))
