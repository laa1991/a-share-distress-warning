"""核口径：我在 arena 题面里给 A/B 的定义 vs gold 读者用的定义（这决定了"谁错"）。"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\dev\finlab-ml")
t = (ROOT / "src" / "judge_arena.py").read_text(encoding="utf-8")
i = t.find('"st_reason"')
seg = t[i:i + 1200]
q = re.search(r'"question":\s*(.*?)"schema"', seg, re.S).group(1)
q = " ".join(x.strip().strip('"') for x in q.splitlines() if x.strip().startswith('"'))
print("=== 我在题面里给模型的定义 ===")
print(q[:800])

print("\n=== gold 那 5 条（两家都判错）的裁定与读者理由 ===")
d = pd.read_csv(ROOT / "data/judge_arena/st_reason_gold_labels.v6-120.csv", dtype=str).fillna("")
for i in ["002528_2025-04-30", "300125_2024-08-19", "300225_2025-07-02", "600745_2026-04-30", "603557_2024-04-27"]:
    r = d[d["id"] == i]
    if len(r):
        print(f"  {i} · 裁定={r.iloc[0]['裁定']} · 理由：{r.iloc[0]['理由']}")

print("\n=== 判准：题面里有没有'审计意见'这半句 ===")
for k in ("审计意见", "无法表示", "否定意见", "内控被否", "财务报表"):
    print(f"  题面含「{k}」：{k in q}")

print("\n=== 300125 那条：题面规则 vs 读者口径 ===")
print("  题面写的是：", "取正文中较早出现的那个" if "较早出现" in q else "（找不到那句）")
