"""更正 §11.10：glm「新题面覆盖率掉到 80.8%」是**配额**，不是模型、也不是题面。

并把这一批跑掉的量算出来（prompt/completion tokens 合计），配额是会咬人的东西。
"""
from pathlib import Path

import pandas as pd

A = Path(r"C:\dev\finlab-ml\data\judge_arena")
FILES = ["glm-5.3", "glm-5.3-refill", "glm-5.3-prefixfix", "glm-5.3-prefixfix120", "glm-5.3-prefixfix120r2",
         "kimi-k3", "kimi-k3-prefixfix", "kimi-k3-prefixfix120"]

tot = {"ptok": 0, "ctok": 0, "calls": 0, "quota": 0}
for tag in FILES:
    f = A / f"st_reason_judge_opencode-{tag}.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f, dtype=str).fillna("")
    pt = pd.to_numeric(d["ptok"], errors="coerce").sum()
    ct = pd.to_numeric(d["ctok"], errors="coerce").sum()
    q = int(d["err"].astype(str).str.contains("429|UsageLimit", regex=True).sum())
    tot["ptok"] += pt or 0
    tot["ctok"] += ct or 0
    tot["calls"] += len(d)
    tot["quota"] += q
    print(f"{tag:26s} 调用 {len(d):4d} · prompt {int(pt or 0):8,d} · completion {int(ct or 0):8,d} · 429 {q}")

print(f"\n今天这一格合计：调用 {tot['calls']} 次 · prompt {int(tot['ptok']):,} · completion {int(tot['ctok']):,} tokens · "
      f"其中 429 {tot['quota']} 次")

BLOCK = """

#### 11.10.1 **更正（11:2x，跑完更正数据后 15 分钟内）**：那个"覆盖率掉 10 点"是**配额**，不是模型也不是题面

我把每一次空答按原因拆开了（`src/split_blank_causes.py`）：

| 跑次 | 有正文 | 空 | 空的原因 |
|---|---|---|---|
| `glm-5.3` 旧题面（主跑 + 补跑） | 109 | 11 | **撞 token 顶 7** + 网络失败 4 |
| `glm-5.3` **新题面** | 97 | 23 | **HTTP 429 `GoUsageLimitError` 22** + 撞顶 **1** |
| `glm-5.3` 新题面·重复（对照） | **0** | 120 | **429 × 120（这一跑整体作废）** |

⇒ 两条更正：
1. **新题面下模型自己撞顶的只有 1 条**（旧题面是 7 条）⇒ **修好题面之后它是"更少想不完、覆盖更好"**，
   20 点的差里 **19 点来自 429**（OpenCode Go 套餐这一次跑用了 22 次额度，重复那一跑 120 次全 429）。
2. **那个"同题面重复"的对照作废**（它不是读数）⇒ §11.10 里"题面变长 vs 波动"这个问题**因此换了个答案**：
   **两种都不是** —— 是**第三方额度**。这也再次说明：**把"调用失败"和"模型答不出"分开数，是这一格的前提**；
   我原先那张表没有分，差点把一个配额问题写成模型属性。

**这一格今天烧掉的量**（同一个 Go 套餐）：`python src/split_blank_causes.py` 会现算并打印；
它是这一格第一次遇到**上游额度**这种失败原因（此前只有撞顶、网络、503）。
"""

P = Path(r"C:\dev\finlab-ml\docs\探索-判断器的效率（GBDT vs Jev vs LLM）.md")
t = P.read_text(encoding="utf-8")
assert "#### 11.10.1" not in t
P.write_text(t.rstrip() + BLOCK, encoding="utf-8")
print("\n追加 §11.10.1 ✓ · 文档", len(P.read_text(encoding="utf-8")), "字符")
