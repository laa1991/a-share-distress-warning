"""列出「要取正文」的戴帽公告：测试年样本里被戴帽的那批（按组分类），去重后写 CSV。

组：① 没亏但戴帽（当年 L1=0）· ③ 亏了也戴帽（L1=1）· 其余（不在测试年）不取。
用法: python st_sample_events.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from run_experiments import PANEL, make_samples  # noqa: E402
from st_label import build as build_st  # noqa: E402

TEST_YEARS = [2022, 2023, 2024, 2025]

panel = pd.read_pickle(PANEL)
s = make_samples(panel)
s = s[s["label"].notna() & s["t_decision"].notna() & s["t_label"].notna()].copy()
_, first = build_st()
s = s.merge(first, on="code", how="left")
d = (s["st_first"] - pd.to_datetime(s["t_label"], errors="coerce")).dt.days
s["L1"] = s["label"]
s["L2"] = ((d >= 0) & (d <= 365)).astype(float)
t = s[s["year"].isin(TEST_YEARS) & (s["L2"] == 1)].copy()
t["grp"] = ["① 没亏但戴帽" if x == 0 else "③ 亏了也戴帽" for x in t["L1"]]

targets = (t.assign(date=pd.to_datetime(t["st_first"]).dt.strftime("%Y-%m-%d"))[["code", "date", "grp"]]
           .drop_duplicates(subset=["code", "date"]).sort_values(["grp", "code"]))
out = ROOT / "data" / "st_event_targets.csv"
targets.to_csv(out, index=False, encoding="utf-8-sig")
print(t["grp"].value_counts().to_string())
print(f"\n去重后要取正文的公告 {len(targets)} 条（① {int((targets['grp'].str.startswith('①')).sum())} · "
      f"③ {int((targets['grp'].str.startswith('③')).sum())}）")
print("->", out)
