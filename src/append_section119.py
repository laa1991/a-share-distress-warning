"""追加 §11.9：两条路径的第一张切片 + 我自己的一个 own-goal（题面口径 vs 真值口径）。"""
from pathlib import Path

P = Path(r"C:\dev\finlab-ml\docs\探索-判断器的效率（GBDT vs Jev vs LLM）.md")
t = P.read_text(encoding="utf-8")
assert "### 11.9" not in t

SECTION = """

### 11.9 「它怎么走到答案的」—— 两条路径的第一张切片，以及**我自己的一个 own-goal**

Ann 问的是：**AI 是怎么得到这个数的？两个模型的路径一样吗？** 判据 `src/probe_reasoning_paths.py`（10 条样本 × 2 家，
同题面、temperature 0）· `src/read_decision_tails.py` · 明细 `data/judge_arena/reasoning_paths.json`。

**第一层答案：两家连"路径可不可读"都不一样。**

| | `glm-5.3` | `kimi-k3` |
|---|---|---|
| 暴露思考过程（`reasoning_content`） | ✅ 10/10 条都有，**481 – 24,130 字** | ❌ **10/10 条全是 0 字** |
| 最终输出 | 31–32 字的 JSON | 31–32 字的 JSON |
| 观察得到的路径 | 读全文 → 列候选原因 → 引条款号 → 应用题面规则 → 出 JSON | **只有最后那 31 个字** |
| 失败形态 | 想不完（19k–24k 字仍在想，正文空） | 想得出、但可能想错 |

⇒ **一家把过程写在纸上，一家是黑箱** —— 这不是"谁更透明"的道德评价，是**能不能被审计**的工程差别：
出错时，glm 的 trace 能让人定位到"它引了哪一句、用了哪条规则"；kimi 只能靠**改题面再跑一次**做行为推断。

**第二层答案（我更想让你看的那一层）：glm 的 trace 直接照出了我题面的两个洞。**

我把 10 条样本里"两家都判错"的那 5 条拿出来读它的决策段，于是看到：

| 案子 | gold | glm 答 | glm 的 trace 逐字说了什么 | 我的题面写了什么 |
|---|---|---|---|---|
| `600745` | **B**（财务会计报告被出具无法表示意见） | A | "the financial report disclaimer **doesn't fit neatly into B (which is about financial indicators like net profit, net assets, revenue)** … It's more of an audit/governance issue" | B 的定义里**只有**「净利润/净资产/收入指标」—— **漏了"财报被出具非标意见"这一条** |
| `300225` | **B**（财报无法表示意见 + 另:A 内控否定） | A | "**this question's B option specifically says '净利润/净资产/收入指标' which excludes audit opinions**" | 同上 |
| `002528` | **B**（三年扣非孰低为负） | A | "The background trigger: 内控无法表示 → A. The current trigger: 三年扣非负 → B. **Per the instruction, take the earlier one → A**" | 我写的规则是「取**正文中较早出现的**那个」；而 gold 读者用的是「**本次实施事件**的触发」 |
| `300125` | A | （空） | 想 19,085 字仍在推演条款编号，没写完 | —— 这一条**不是口径问题**，是想不完 |
| `603557` | B | （空） | 想 24,130 字，在追"2021 年那次的否定意见还算不算" | 同上 |

⇒ **三条是"照我说的规则答的"**：它们不是答错，是**按我写的错题面答的**。
⇒ 于是必须更正我之前那条结论：我曾写「**三种独立方法都错在同一条缝上 ⇒ 那是任务本身的难点**」——
   **那半句要撤回**：两个 LLM 是**被同一份有洞的题面喂出来的**（GBDT 没读题面，它的 36/41 是"手里没那条信息"，那半条仍然成立）。
   **"同一条缝"至少有三种来源**：① 题面定义漏项 ② 规则有歧义 ③ 任务真的难 —— 我之前只想到第三种。

**我已经做的两件事**：
1. **修题面**（`src/judge_arena.py`）：B 的定义补上「**对财务报表出具的无法表示意见或否定意见**」；
   规则改成「取**本次实施事件（本次叠加/本次被实施）的触发**；写明"前期已实施/已被实施"的旧原因不取；指不出才退回"取较早出现的"」。
2. **重跑受影响样本**（41 条 = 5 条两家都错的 + 22 条 gold=B 且理由含"非标意见/扣非"的，两家各跑一遍）——
   用来量**"修完题面之后，那条缝还剩多少"**。读数落在 `st_reason_judge_opencode-*-prefixfix.csv`。

⚠️ **边界**：① `reasoning_content` 是模型**自己讲的故事**，不是内部计算的忠实记录 —— 它能证明"它用了哪条规则/哪句证据"，
不能证明"它内部怎么算的"；② 10 条样本、非随机（刻意混了 5 条难例）⇒ 这一节是**形状**，不是比例；
③ kimi 没有 trace ⇒ 关于它的一切都是**行为推断**。
"""

P.write_text(t.rstrip() + SECTION, encoding="utf-8")
print("追加 §11.9 ✓ · 文档", len(P.read_text(encoding="utf-8")), "字符")
