"""把那 23 条非 A-E 的**原始输出**逐条打出来（不做 `.str[:1]` 截断）——因为截断会造出假的"答对了"。

⚠️ 这一版就是为此写的：audit 里我用 `.str[:1]` 把非字母的输出强行取首字符 ⇒ 20 条变成 100 条"可判"、
并给出 100% 一致的漂亮读数。**那是尺子造的，不是模型答的。**
"""
from pathlib import Path

import pandas as pd

A = Path(r"C:\dev\finlab-ml\data\judge_arena")
d = pd.read_csv(A / "st_reason_judge_opencode-glm-5.3.csv", dtype=str).fillna("")
g = pd.read_csv(A / "st_reason_gold_labels.v6-120.csv", dtype=str).fillna("")
gold = dict(zip(g["id"], g["裁定"]))

bad = d[~d["choice"].isin(list("ABCDE"))]
print(f"非 A-E 的 {len(bad)} 条，逐条原文（gold / choice / ms / raw）：\n")
for _, r in bad.iterrows():
    raw = str(r["raw"]).strip()
    kind = "空" if not raw else ("选项回显" if ("|" in raw or "选项" in raw) else "有字但非单字母")
    print(f"  {r['id']}  真值={gold.get(r['id'],'?')}  ms={r['ms']:>7}  [{kind}]  choice={r['choice']!r}")
    if raw:
        print(f"        raw: {raw[:160]!r}")
