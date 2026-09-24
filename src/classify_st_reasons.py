"""从戴帽公告**正文**里归类「为什么戴帽」，并对比两组（① 没亏但戴帽 / ③ 亏了也戴帽）。

⚠️ 第一版栽过一个坑：**在全文里搜关键词，命中的常常是"规则条文引用"而不是本案原因**
（例：正文里会照抄「（三）财务会计报告被出具保留意见、无法表示意见或者否定意见的审计报告」——那是规则清单）。
修法：先在正文里**定位"原因段"**（`…的原因` / `触及《…》` 这类锚点），只在该窗口内分类；
窗口找不到就退回全文，但**把"窗口找到没找到"作为覆盖率报出来**（一把尺要先自报量到多少比例）。

分类是**关键词规则**，因此每条都留一句证据原文；分不清的进「未归类」，**不硬凑**。

用法: python classify_st_reasons.py [--examples 2]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PDFTXT = ROOT / "data" / "raw" / "st_pdf"
TARGETS = ROOT / "data" / "st_event_targets.csv"

ANCHORS = ["实施其他风险警示的原因", "实施退市风险警示的原因", "被实施其他风险警示的原因",
           "被实施退市风险警示的原因", "的主要原因", "原因如下", "具体原因", "触及《"]

RULES: list[tuple[str, list[str]]] = [
    ("资金占用（非经营性）", ["非经营性占用", "资金占用"]),
    ("违规担保", ["违规担保", "违规对外担保"]),
    ("内控被否 / 重大缺陷", ["内部控制审计报告", "重大缺陷"]),
    ("审计意见：无法表示 / 否定", ["出具了无法表示意见", "出具了否定意见", "无法表示意见的审计报告",
                                   "否定意见的审计报告"]),
    ("财务类：净利润为负 / 连亏", ["净利润为负值", "连续亏损", "净利润为负"]),
    ("财务类：收入或净资产指标", ["营业收入低于", "净资产为负值", "扣除后的营业收入"]),
    ("重整 / 破产", ["重整", "破产"]),
    ("账户 / 资产被冻结", ["冻结"]),
    ("信息披露违法 / 立案 / 处罚", ["信息披露违法", "立案调查", "立案告知书", "行政处罚"]),
    ("规范运作类其他", ["规范运作", "未按规定", "公司治理", "信息披露"]),
]


def window(text: str, span: int = 700) -> tuple[str, str | None]:
    t = re.sub(r"\s+", "", text)
    for a in ANCHORS:
        i = t.find(a)
        if i >= 0:
            return t[i: i + span], a
    return t[:900], None


def classify(win: str) -> tuple[str, str]:
    for name, kws in RULES:
        for kw in kws:
            i = win.find(kw)
            if i >= 0:
                return name, win[max(0, i - 26): i + 62]
    return "未归类", win[:80]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=2)
    args = ap.parse_args()

    tg = pd.read_csv(TARGETS, dtype={"code": str})
    tg["code"] = tg["code"].str.zfill(6)
    rows = []
    for _, r in tg.iterrows():
        f = PDFTXT / f"{r['code']}_{r['date']}.txt"
        if not f.exists():
            rows.append({"code": r["code"], "date": r["date"], "grp": r["grp"], "chars": 0,
                         "锚点": None, "类别": "（正文没抓到）", "证据": ""})
            continue
        text = f.read_text(encoding="utf-8")
        win, anchor = window(text)
        cat, ev = classify(win)
        rows.append({"code": r["code"], "date": r["date"], "grp": r["grp"], "chars": len(text),
                     "锚点": anchor, "类别": cat, "证据": ev})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "data" / "st_reasons.csv", index=False, encoding="utf-8-sig")

    ok = d[d["chars"] > 0]
    print(f"样本 {len(d)} 条（① {int(d['grp'].str.startswith('①').sum())} · "
          f"③ {int(d['grp'].str.startswith('③').sum())}）· 正文中位 {d['chars'].median():.0f} 字")
    print(f"**锚点覆盖率**：{int(ok['锚点'].notna().sum())}/{len(ok)} = {ok['锚点'].notna().mean():.1%}"
          f"（找不到锚点的按全文分类）\n")

    for g in sorted(d["grp"].unique()):
        sub = d[d["grp"] == g]
        print(f"=== {g}（n={len(sub)}）")
        for k, v in sub["类别"].value_counts().items():
            print(f"   {k:26s} {v:4d}  {v/len(sub):6.1%}")
        print()

    ct = pd.crosstab(d["类别"], d["grp"], normalize="columns") * 100
    print("两类占比对照（列归一，%）：")
    print(ct.round(1).to_string())

    print(f"\n证据样例（每类 {args.examples} 条）：")
    for cat in d["类别"].unique():
        for _, r in d[d["类别"] == cat].head(args.examples).iterrows():
            print(f"   [{cat}] {r['code']} {r['date']}（{r['grp'][:1]}）…{str(r['证据'])[:72]}…")

    (ROOT / "data" / "st_reasons.json").write_text(json.dumps({
        "coverage_anchor": float(ok["锚点"].notna().mean()),
        "counts": {f"{g}|{c}": int(v) for (g, c), v in d.groupby(["grp", "类别"]).size().items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n读数 -> data/st_reasons.csv / data/st_reasons.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
