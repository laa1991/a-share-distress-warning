"""核「原因分布」那一节的分母里混进了什么：**非公司公告 / 尚未实施的公告**，并给出它们的边际影响。

独立读者在 60 条新样本里点名了两条：`300527`（东方金诚评级机构的**关注公告**，不是公司公告）、
`002047`（《业绩预告修正暨**可能**被实施》——触发条件尚未成立）。查它们在不在发布用的 211 份里，
并算出"剔掉它们之后那一节的三格会怎么变"。
"""
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\dev\finlab-ml")
s = pd.read_csv(ROOT / "data" / "st_reasons.csv", dtype=str).fillna("")
log = pd.read_csv(ROOT / "data" / "fetch_st_pdf_log.csv", dtype=str).fillna("")
title = {(r["code"], r["date"]): r.get("title", "") for _, r in log.iterrows()}
s["title"] = [title.get((r["code"], r["date"]), "") for _, r in s.iterrows()]

NONCO = r"评级|关注公告|受托管理|信用评级|资信评估"
# ⚠️ 第一版把「将被实施」也当可疑 —— 那是**正常实施公告的标准标题**，一网打上 21 条假阳性。
#    真正可疑的只有**条件式**：可能被实施 / 拟被实施（触发条件尚未成立）。
NOTYET = r"可能被实施|拟被实施"
s["非公司件"] = s["title"].str.contains(NONCO, regex=True)
# 「可能被实施」还要再分一次：标题里**另有**肯定式「被实施」的（如「被实施其他风险警示、可能被实施退市风险警示」）
# 本身就是实施公告，只是顺带警告 —— 只有**通篇条件式**的才是"尚未实施"。
import re as _re  # noqa: E402
s["未实施件"] = s["title"].str.contains(NOTYET, regex=True) & ~s["title"].str.contains(r"(?<!可能)被实施", regex=True)
print(f"211 份里：非公司件 {int(s['非公司件'].sum())} 份 · 尚未实施件 {int(s['未实施件'].sum())} 份")
for _, r in s[s["非公司件"] | s["未实施件"]].iterrows():
    print(f"   {r['code']} {r['date']} · {r['title'][:56]}")

MAP = {"内控被否 / 重大缺陷": "合规/治理类", "信息披露违法 / 立案 / 处罚": "合规/治理类",
       "资金占用（非经营性）": "合规/治理类", "违规担保": "合规/治理类",
       "规范运作类其他": "合规/治理类", "账户 / 资产被冻结": "合规/治理类",
       "财务类：收入或净资产指标": "财务类", "财务类：净利润为负 / 连亏": "财务类",
       "审计意见：无法表示 / 否定": "财务类", "重整 / 破产": "重整/破产"}
s["粗类"] = s["类别"].map(MAP).fillna("未归类")


def table(d: pd.DataFrame, tag: str) -> None:
    print(f"\n[{tag}] n={len(d)}")
    for prefix, name in (("①", "没亏但戴帽"), ("③", "亏了也戴帽")):
        g = d[d["grp"].str.startswith(prefix, na=False)]
        if not len(g):
            continue
        vc = g["粗类"].value_counts()
        pct = {k: f"{v / len(g):.0%}" for k, v in vc.items()}
        print(f"   {prefix} {name}（n={len(g)}）：{pct}")


table(s, "全部 211（现值）")
table(s[~(s["非公司件"] | s["未实施件"])], "剔掉非公司件/未实施件")
s.to_csv(ROOT / "data" / "st_reasons_flagged.csv", index=False, encoding="utf-8-sig")
print("\n已落 data/st_reasons_flagged.csv（多了 title / 非公司件 / 未实施件 三列）")
