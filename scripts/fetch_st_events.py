"""抓「被实施风险警示」的公告（巨潮公告检索，keyword=风险警示），按年切 —— 用来做 **as-of ST 标签**。

为什么走这条路：`stock_info_change_name`（新浪曾用名）**只有名字、没有日期**，
无法还原「某年某日是不是 ST」；而这个检索接口返回的每条公告都带 **公告时间 + 当时简称**，是可时点化的。

判据处理（标题里的文字游戏很多，必须逐类分开）：
- **实施**：标题同时含「实施」与「风险警示」，且**不含「撤销」** ⇒ 记为一次**戴帽**事件；
- 标题含「撤销」的（含"申请撤销…暨继续实施…"这种混合句）⇒ **不算戴帽**（那是在说继续/撤销，不是新戴帽）；
- 单独的「终止上市风险提示」不算（那是退市流程的提示，不是实施风险警示）。

用法: python fetch_st_events.py [--start 2015] [--end 2026]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "st_events"
LOG = ROOT / "data" / "fetch_st_log.csv"


def clean(s: str) -> str:
    return re.sub(r"</?em>", "", str(s))


def classify(title: str) -> str:
    t = clean(title)
    if "风险警示" not in t:
        return "无关"
    if "撤销" in t:
        return "撤销/混合"
    if "实施" in t:
        return "实施"
    return "其他风险警示类"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2026)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for y in range(args.start, args.end + 1):
        out = OUT / f"{y}.csv"
        if out.exists() and out.stat().st_size > 50:
            d = pd.read_csv(out, dtype={"代码": str})
            status = "cached"
        else:
            status = "ok"
            try:
                d = ak.stock_zh_a_disclosure_report_cninfo(symbol="", market="沪深京", keyword="风险警示",
                                                           start_date=f"{y}0101", end_date=f"{y}1231")
                d.to_csv(out, index=False, encoding="utf-8-sig")
            except Exception as e:  # noqa: BLE001
                status, d = f"failed:{type(e).__name__}", pd.DataFrame()
        n = len(d)
        cls = d["公告标题"].map(classify).value_counts().to_dict() if n else {}
        rows.append({"year": y, "rows": n, "status": status,
                     "实施": cls.get("实施", 0), "撤销/混合": cls.get("撤销/混合", 0),
                     "其他": cls.get("其他风险警示类", 0)})
        print(f"  {y}: {n:5d} 条 → 实施 {cls.get('实施', 0):4d} · 撤销/混合 {cls.get('撤销/混合', 0):4d} [{status}]", flush=True)

    log = pd.DataFrame(rows)
    log.to_csv(LOG, index=False, encoding="utf-8-sig")
    print(f"\n合计 {log['rows'].sum()} 条 · 实施 {log['实施'].sum()} · 撤销/混合 {log['撤销/混合'].sum()}")
    print("账 ->", LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
