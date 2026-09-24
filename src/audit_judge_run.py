"""独立复核 glm-5.3 那一跑：**"accuracy 1.0" 这种读数先当 bug 查，不当战绩**。

要答四问：① 120 条里对上了多少条真值（为什么只报 100）；② 没对上的 20 条是什么形状（模型没答出来？id 对不上？）；
③ 1.0 是不是"只在答得出来的时候才答"造成的选择效应；④ 逐类混淆长什么样。
"""
from pathlib import Path

import pandas as pd

A = Path(r"C:\dev\finlab-ml\data\judge_arena")
res = pd.read_csv(A / "st_reason_judge_opencode-glm-5.3.csv", dtype=str).fillna("")
gold = pd.read_csv(A / "st_reason_gold_labels.v6-120.csv", dtype=str).fillna("")
print(f"judge 表 {len(res)} 行 · gold 表 {len(gold)} 行 · 两个 id 集合交集 {len(set(res['id']) & set(gold['id']))}")

m = res.merge(gold[["id", "裁定", "来源"]], on="id", how="left")
m["choice"] = m["choice"].astype(str).str.strip().str.upper().str[:1]
print(f"\n合并后 {len(m)} 行 · 有裁决真值 {m['裁定'].ne('').sum()} 行 · choice 是 A-E 的 {m['choice'].isin(list('ABCDE')).sum()} 行")
print("choice 取值分布：", m["choice"].value_counts().to_dict())
print("没答成字母的那些 choice 原文：", m.loc[~m["choice"].isin(list("ABCDE")), "choice"].value_counts().head(8).to_dict())

dec = m[m["choice"].isin(list("ABCDE")) & m["裁定"].ne("")]
print(f"\n【可判且有真值】{len(dec)} 条 · 一致 {(dec['choice'] == dec['裁定']).mean():.4f}")
print("逐类（真值 → 模型）：")
print(pd.crosstab(dec["裁定"], dec["choice"]).to_string())

print("\n【若把『没答成字母』一律算错】")
allg = m[m["裁定"].ne("")].copy()
allg["_ok"] = allg["choice"].isin(list("ABCDE")) & (allg["choice"] == allg["裁定"])
print(f"  分母 {len(allg)} · 正确 {int(allg['_ok'].sum())} ⇒ {allg['_ok'].mean():.4f}")

print("\n【那 20 条没答成字母的，有没有共同形状】")
bad = m[~m["choice"].isin(list("ABCDE"))]
if len(bad):
    print("  真值分布：", bad["裁定"].value_counts().to_dict())
    print("  读取自哪一半：", bad["来源"].value_counts().to_dict())
    print("  样本（前 3 条的 raw 前 120 字）：")
    for _, r in bad.head(3).iterrows():
        print(f"    {r['id']} 真值={r['裁定']} raw={str(r['raw'])[:120]!r}")
print("\n【前 3 条答对的，看它到底写了什么】")
for _, r in dec.head(3).iterrows():
    print(f"  {r['id']} 真值={r['裁定']} choice={r['choice']} conf={r['conf']} raw={str(r['raw'])[:110]!r}")
