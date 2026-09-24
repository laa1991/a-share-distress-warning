"""两个判断器并排：glm-5.3 vs kimi-k3（同一套 120 道题、同一份裁决真值）。

回答一件事：**换一个模型家族，分数 / 覆盖率 / 延迟 / 校准会不会变**。
（glm 那侧要把补跑合进来；kimi 是一次跑完的 120 条。）
"""
import re
from pathlib import Path

import pandas as pd

A = Path(r"C:\dev\finlab-ml\data\judge_arena")
gold = pd.read_csv(A / "st_reason_gold_labels.v6-120.csv", dtype=str).fillna("")
goldmap = dict(zip(gold["id"], gold["裁定"]))


def load(tag: str, refill: bool = False) -> pd.DataFrame:
    f = A / f"st_reason_judge_opencode-{tag}.csv"
    d = pd.read_csv(f, dtype=str).fillna("")
    d["来自"] = "主跑"
    if refill:
        rf = A / f"st_reason_judge_opencode-{tag}-refill.csv"
        if rf.exists():
            r = pd.read_csv(rf, dtype=str).fillna("")
            r["来自"] = "补跑"
            d = pd.concat([d[~d["id"].isin(r["id"])], r], ignore_index=True)
    def pri(v: str) -> str:
        m = re.match(r"^\s*([ABCDE])", str(v or "").strip().upper())
        return m.group(1) if m else ""
    d["primary"] = d["choice"].apply(pri)
    d["gold"] = d["id"].map(goldmap)
    d["ok"] = d["primary"].eq(d["gold"]) & d["primary"].ne("")
    d["_multi"] = d["choice"].astype(str).str.contains("另")
    return d


g = load("glm-5.3", refill=True)
k = load("kimi-k3")
rows = []
for name, d in (("glm-5.3", g), ("kimi-k3", k)):
    answered = d[d["primary"].ne("")]
    ms = pd.to_numeric(d["ms"], errors="coerce")
    conf = pd.to_numeric(answered["conf"], errors="coerce").dropna()
    rows.append({
        "模型": name,
        "覆盖率": f"{len(answered)}/{len(d)}（{len(answered)/len(d):.1%}）",
        "一致率(答出的)": f"{answered['ok'].mean():.4f}（{int(answered['ok'].sum())}/{len(answered)}）",
        "一致率(空算错)": f"{d['ok'].mean():.4f}",
        "延迟中位": f"{ms.median():.0f} ms",
        "延迟p95": f"{ms.quantile(.95):.0f} ms",
        "置信度中位": f"{conf.median():.3f}" if len(conf) else "-",
        "不确定带[0.2,0.8]": f"{((conf >= .2) & (conf <= .8)).mean():.2%}" if len(conf) else "-",
        "多值标注": f"{d['_multi'].mean():.1%}",
        "completion tokens": int(pd.to_numeric(d["ctok"], errors="coerce").sum()),
    })
print(pd.DataFrame(rows).to_string(index=False))

mg = g.set_index("id")
mk = k.set_index("id")
common = mg.index.intersection(mk.index)
same = (mg.loc[common, "primary"] == mk.loc[common, "primary"]).mean()
print(f"\n两个模型彼此一致：{same:.4f}（{int((mg.loc[common,'primary'] == mk.loc[common,'primary']).sum())}/{len(common)}）")

wg = set(g[~g["ok"]]["id"])
wk = set(k[~k["ok"]]["id"])
print(f"\nglm 错 {len(wg)} 条 · kimi 错 {len(wk)} 条 · **两者重合 {len(wg & wk)} 条**")
print("  重合的（更像任务难点）：", sorted(wg & wk))
print("  只有 glm 错的：", sorted(wg - wk))
print("  只有 kimi 错的：", sorted(wk - wg))
print("\nkimi 的 5 条错，逐条（gold / kimi / glm / 两者 conf）：")
for i in sorted(wk):
    print(f"  {i} gold={mg.loc[i,'gold']} kimi={mk.loc[i,'primary']}(conf {mk.loc[i,'conf']}) "
          f"glm={mg.loc[i,'primary']}(conf {mg.loc[i,'conf']}) raw_kimi={str(mk.loc[i,'raw'])[:70]!r}")
