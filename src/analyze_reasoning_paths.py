"""分析两条推理路径：它们各自"引用了正文里的哪一句"。

做法：把公告正文切成句子（按 。；！ 切），去掉空白与标点做归一，再看**每一句是否出现在模型的 reasoning 里**
（至少 12 个归一字符才算引用，避免"公司""风险警示"这类词误命中）。
⇒ 得到的是"它叙述里用了哪条证据"，不是"它内部怎么算的"（那条边界写在文档里）。
"""
import json
import re
from pathlib import Path

import pandas as pd

A = Path(r"C:\dev\finlab-ml\data\judge_arena")
rows = json.loads((A / "reasoning_paths.json").read_text(encoding="utf-8"))
cases = pd.read_csv(A / "st_reason_cases.csv", dtype=str).fillna("")
text = dict(zip(cases["id"], cases["text"]))
PUNCT = re.compile(r"[\s，。；、：（）()《》\-—“”\"'’‘·\.]")


def norm(s: str) -> str:
    return PUNCT.sub("", s or "")


def sentences(t: str) -> list[str]:
    return [s for s in re.split(r"[。；！\n]", t) if len(norm(s)) >= 12]


print(f"{'id':22s} {'gold':4s} | {'glm: 答/思考字数/引用句数':28s} | {'kimi: 答/思考字数/引用句数':28s} | 引用同一句？")
same_cnt = 0
detail = []
for r in rows:
    cid, gold = r["id"], r["gold"]
    doc = [norm(s) for s in sentences(text.get(cid, ""))]
    per = {}
    for model in ("glm-5.3", "kimi-k3"):
        d = r.get(model, {})
        think = d.get("think_norm", "")
        cited = [i for i, s in enumerate(doc) if s in think]
        per[model] = {"choice": d.get("choice"), "think": d.get("think_len", 0), "cited": cited,
                      "ctok": d.get("ctok")}
    gl, km = per["glm-5.3"], per["kimi-k3"]
    inter = set(gl["cited"]) & set(km["cited"])
    same = bool(inter)
    same_cnt += same
    print(f"{cid:22s} {gold:4s} | {str(gl['choice']):4s} / {gl['think']:5d} / {len(gl['cited']):2d} 句"
          f"{'':18s} | {str(km['choice']):4s} / {km['think']:5d} / {len(km['cited']):2d} 句"
          f"{'':18s} | {'✅ 有共同引用' if same else '❌ 没有共同引用'}")
    detail.append({"id": cid, "gold": gold, "glm": gl, "kimi": km, "common": sorted(inter),
                   "sentences": text.get(cid, "")[:0]})

print(f"\n有共同引用句的题：{same_cnt}/{len(rows)}")
print(f"思考长度中位：glm {pd.Series([d['glm']['think'] for d in detail]).median():.0f} 字 · "
      f"kimi {pd.Series([d['kimi']['think'] for d in detail]).median():.0f} 字")
print(f"completion tokens 合计：glm {sum(d['glm']['ctok'] or 0 for d in detail)} · kimi {sum(d['kimi']['ctok'] or 0 for d in detail)}")

print("\n=== 逐题：两家各自引用到的正文句子（前 2 句）===")
for d in detail:
    sents = sentences(text.get(d["id"], ""))
    print(f"\n[{d['id']}] gold={d['gold']} · glm={d['glm']['choice']} kimi={d['kimi']['choice']}")
    for model in ("glm", "kimi"):
        idxs = d[model]["cited"][:2]
        print(f"  {model:4s}: " + (" ｜ ".join(sents[i][:60] for i in idxs) if idxs else "（没引用到任何正文句）"))
