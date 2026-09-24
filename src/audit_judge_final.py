"""把最终明细里的两类"要说清楚的东西"逐条列出来：① 仍答不出来的 11 条；② 判错的 2 条。
顺带比一次**多值标注**：模型在 choice 里标 `+另` 的比例 vs 读者在 `理由` 里标 `+另` 的比例。
"""
from pathlib import Path

import pandas as pd

A = Path(r"C:\dev\finlab-ml\data\judge_arena")
s = pd.read_csv(A / "st_reason_judge_opencode-glm-5.3.MERGED.csv", dtype=str).fillna("")
g = pd.read_csv(A / "st_reason_gold_labels.v6-120.csv", dtype=str).fillna("")
g["多值"] = g["理由"].astype(str).str.contains("另")
gm = dict(zip(g["id"], g["裁定"]))

blank = s[s["primary"].eq("")]
wrong = s[s["primary"].ne("") & s["primary"].ne(s["gold"])]
print(f"① 仍答不出 {len(blank)} 条（gold 分布 {blank['gold'].value_counts().to_dict()}）")
def fnum(v):
    try:
        return f"{float(v):.0f}"
    except (TypeError, ValueError):
        return f"<{v!r}>"


for _, r in blank.iterrows():
    print(f"   {r['id']} 真值={r['gold']} ms={fnum(r['ms'])} http={r['http']} ctok={fnum(r['ctok'])} raw={str(r['raw'])[:60]!r}")
print(f"\n② 判错 {len(wrong)} 条")
for _, r in wrong.iterrows():
    print(f"   {r['id']} 真值={r['gold']} 模型={r['primary']} conf={r['conf']} raw={str(r['raw'])[:90]!r}")
s["_multi"] = s["choice"].astype(str).str.contains("另")
print(f"\n③ 多值标注：模型 {int(s['_multi'].sum())} 条（{s['_multi'].mean():.1%}）· 读者在理由里标「另」的 {int(g['多值'].sum())} 条")
print("   两边都标了的 id：", sorted(set(s[s['_multi']]['id']) & set(g[g['多值']]['id'])))
print("\n⑤ 那 4 条 ms/http 为空的，err 列写了什么：")
for _, r in s[s["ms"].eq("")].iterrows():
    print(f"   {r['id']} 来自={r['来自']} err={str(r['err'])[:110]!r}")
print("\n④ 覆盖率按真值类别：")
print(s.assign(答出=s["primary"].ne("")).groupby("gold")["答出"].agg(["size", "sum"]).to_string())
