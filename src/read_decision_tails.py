"""把 glm 在"两家都判错"那 5 条上的**决策段**（思考的最后 700 字）读出来 —— 它到底怎么定主因的。

kimi 完全没有 trace（10 条全 0 字），所以这一节只能观察一家；这本身就是"两条路径一样吗"的一半答案。
"""
import json
from pathlib import Path

A = Path(r"C:\dev\finlab-ml\data\judge_arena")
rows = json.loads((A / "reasoning_paths.json").read_text(encoding="utf-8"))
SHARED_WRONG = {"002528_2025-04-30", "300125_2024-08-19", "300225_2025-07-02",
                "600745_2026-04-30", "603557_2024-04-27"}

print("=== kimi 的 trace 长度（全部 10 条）===")
print("  ", {r["id"][:6]: r.get("kimi-k3", {}).get("think_len") for r in rows})
print("=== 两家 content 长度（答.json 那部分）===")
print("  glm:", {r["id"][:6]: r.get("glm-5.3", {}).get("content_len") for r in rows})
print("  kimi:", {r["id"][:6]: r.get("kimi-k3", {}).get("content_len") for r in rows})

for r in rows:
    if r["id"] not in SHARED_WRONG:
        continue
    g = r.get("glm-5.3", {})
    print(f"\n{'='*80}\n[{r['id']}] gold={r['gold']} · glm 答={g.get('choice')!r} · 思考 {g.get('think_len')} 字")
    t = g.get("think_norm", "")
    print("  ……决策段（最后 700 字）：")
    print("  " + (t[-700:] if t else "（无 trace）"))
