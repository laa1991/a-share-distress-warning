"""把"空答"分成三种：**真撞顶（http 200 且 completion≥5900）** / **配额或限流（429 等）** / **其他调用失败**。

列名口径：这些表用的是 `raw`（模型原文）+ `err`（错误原文），**没有 `content` 列**。
这一格决定 §11.10 里 glm 覆盖率那行该怎么写：是"模型想不完"，还是"上游不给"。
"""
import re
from pathlib import Path

import pandas as pd

A = Path(r"C:\dev\finlab-ml\data\judge_arena")
FILES = ["glm-5.3", "glm-5.3-prefixfix120", "glm-5.3-prefixfix120r2", "glm-5.3-refill",
         "kimi-k3", "kimi-k3-prefixfix120"]

for tag in FILES:
    f = A / f"st_reason_judge_opencode-{tag}.csv"
    if not f.exists():
        print(f"[{tag}] （无文件）")
        continue
    d = pd.read_csv(f, dtype=str).fillna("")
    has = d["raw"].astype(str).str.contains(r'"choice"', regex=True, na=False)
    blank = d[~has]
    http = pd.to_numeric(d["http"], errors="coerce")
    ctok = pd.to_numeric(d["ctok"], errors="coerce")
    b = blank
    trunc = b[(pd.to_numeric(b["http"], errors="coerce") == 200) & (pd.to_numeric(b["ctok"], errors="coerce") >= 5900)]
    quota = b[b["err"].astype(str).str.contains("429|UsageLimit", regex=True)]
    other = b[~b.index.isin(trunc.index) & ~b.index.isin(quota.index)]
    print(f"\n[{tag}] n={len(d)} · 有正文 {int(has.sum())} · 空 {len(b)}")
    print(f"   撞顶(200+≥5900) {len(trunc)} · 配额/限流(429) {len(quota)} · 其他失败 {len(other)}")
    if len(other):
        kinds = other["err"].astype(str).str.extract(r"^([A-Za-z]+)")[0].value_counts().to_dict()
        print("   其他失败的类型：", kinds)
        print("   其他失败的 http：", pd.to_numeric(other["http"], errors="coerce").value_counts(dropna=False).to_dict())
    if len(quota):
        print("   429 的原文样例：", quota.iloc[0]["err"][:90])
