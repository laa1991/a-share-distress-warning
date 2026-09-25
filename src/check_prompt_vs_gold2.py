"""重做（上一版取错了块：它把 rev_dir 的题面当成了 st_reason 的）。

正确取法：从 `"st_reason": {` 起，取到该块结束（`"truth_map"` 之后那一行）。
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\dev\finlab-ml")
t = (ROOT / "src" / "judge_arena.py").read_text(encoding="utf-8")
i = t.find('"st_reason": {')
assert i > 0, "找不到 st_reason 块"
j = t.find('"truth_map"', i)
block = t[i:j]
q = "".join(re.findall(r'"([^"]*)"', block))
# 只留 question 字段内容：把所有相邻字符串拼接后，截在 "schema" 之前
q = q.split("primitive")[-1].split("schema")[0]
print("=== st_reason 题面（原文，拼接后）===")
print(q[:900])
print("\n=== 关键词自查 ===")
for k in ("审计意见", "无法表示", "否定意见", "内控", "财务报表", "较早出现", "净利润", "净资产"):
    print(f"  含「{k}」：{k in q}")

print("\n=== gold 读者用的 B 定义（从 5 条理由与 v5 报告口径看）===")
d = pd.read_csv(ROOT / "data/judge_arena/st_reason_gold_labels.v6-120.csv", dtype=str).fillna("")
sub = d[d["理由"].astype(str).str.contains("无法表示|否定意见|扣非", regex=True)]
print(f"  理由里带『无法表示/否定意见/扣非』的：{len(sub)} 条 · 其中裁定 B 的 {int((sub['裁定'] == 'B').sum())} 条")
print("  例：", sub.head(5)[["id", "裁定", "理由"]].to_dict("records"))
