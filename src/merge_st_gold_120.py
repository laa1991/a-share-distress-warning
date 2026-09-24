"""把 60（v4）+ 60（v5-new60）合并成 120 条定版真值，并顺手核两件数据面的事：

① 合并后的分布与「与规则标签的一致率」；
② **样本污染**：独立读者点名 `300527` 那条**不是公司公告、是评级机构的关注公告**，
   而 `002047` 是「**可能**被实施」的预告修正（触发条件尚未成立）⇒ 查这两条在不在我发布用的
   211 份样本里（在的话，"原因分布"那一节的分母里混了非公司文件）。
"""
from pathlib import Path

import pandas as pd

A = Path(r"C:\dev\finlab-ml\data\judge_arena")
ROOT = Path(r"C:\dev\finlab-ml")

v4 = pd.read_csv(A / "st_reason_gold_labels.v4.csv", dtype=str).fillna("")
v5 = pd.read_csv(A / "st_reason_gold_labels.v5-new60.csv", dtype=str).fillna("")
v4["来源"] = "读者A（60条那半）"
v5["来源"] = "读者B（新增60条那半）"
both = pd.concat([v4, v5], ignore_index=True)
assert len(both) == 120 and both["id"].nunique() == 120, "id 有重或有缺"
both.to_csv(A / "st_reason_gold_labels.v6-120.csv", index=False, encoding="utf-8-sig")
print(f"合并 -> st_reason_gold_labels.v6-120.csv · {len(both)} 条")
print("分布：", both["裁定"].value_counts().to_dict())
print("每半各自：")
for src, g in both.groupby("来源"):
    print(f"   {src}: {g['裁定'].value_counts().to_dict()}")

MAP = {"合规/治理类": "A", "财务类": "B", "重整/破产": "C", "未归类": "D", "生产经营类": "E"}
both["_r"] = both["truth_rule"].map(MAP)
ok = both[both["_r"].notna()]
a = (ok["裁定"] == ok["_r"])
print(f"**规则 vs 裁决（n={len(ok)}，另有 {len(both) - len(ok)} 条规则无标签）= {a.mean():.1%}**（{int(a.sum())}/{len(ok)}）")
mism = ok[~a]
print("不一致形状：规则=未归类 D", int((mism['_r'] == 'D').sum()), "条 · 规则给了 A/B/C", int((mism['_r'] != 'D').sum()), "条")

print("\n② 样本污染排查（发布用的 211 份那一批）")
s = pd.read_csv(ROOT / "data" / "st_reasons.csv", dtype=str).fillna("")
print("   st_reasons.csv 行数", len(s), "· 含 300527：", (s["code"] == "300527").any(), "· 含 002047：", (s["code"] == "002047").any())
# 找"非公司公告"的形状：评级机构 / 关注公告 / 受托管理
for pat in ("东方金诚", "评级", "关注公告", "受托管理", "信用评级"):
    hit = s[s.astype(str).apply(lambda r: r.str.contains(pat, na=False).any(), axis=1)]
    if len(hit):
        print(f"   含「{pat}」的样本 {len(hit)} 条：{list(hit['code'])[:8]}")
txtdir = ROOT / "data" / "raw" / "st_pdf"
n_rating = 0
samples = []
for f in txtdir.glob("*.txt"):
    t = f.read_text(encoding="utf-8", errors="replace")[:600]
    if any(k in t for k in ("信用评级", "评级机构", "关注公告", "东方金诚", "联合资信", "中证鹏元", "中诚信")):
        n_rating += 1
        samples.append(f.stem)
print(f"   正文前 600 字里出现评级机构字样的 PDF：{n_rating} / {len(list(txtdir.glob('*.txt')))} 份 · 例：{samples[:5]}")
