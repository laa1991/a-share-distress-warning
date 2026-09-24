"""量「预告修订」：同一家公司同一年度会不会发多次预告、两次之间差多少、内容会不会翻面。

为什么要单列：`load_yjyg()` 把同一 (code, year) 聚合成**一行**（last），所以**修订在聚合层被抹掉了**。
原始文件里每家公司有**多行**（不同「预测指标」），但那是同一天披露的多个指标 —— 得先分清：
**「多行」≠「多版本」**。判据：按公司看**不同的公告日期个数**。

用法: python diag_yjyg_versions.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "yjyg"
KEY_METRIC = "归属于上市公司股东的净利润"      # 主口径：归母净利润


def load_annual() -> pd.DataFrame:
    frames = []
    for f in sorted(RAW.glob("*1231.csv")):
        d = pd.read_csv(f, dtype=str)
        d["报告期"] = f.stem
        d["year"] = int(f.stem[:4])
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["公告日期"] = pd.to_datetime(d["公告日期"], errors="coerce")
    d["预测数值"] = pd.to_numeric(d["预测数值"], errors="coerce")
    d = d[d["股票代码"].astype(str).str.len() == 6].copy()
    return d


def main() -> int:
    d = load_annual()
    if d.empty:
        print("没找到年度预告文件")
        return 1
    print(f"年度预告原始行 {len(d):,} · 公司·年 {d.groupby(['股票代码','year']).ngroups:,} · "
          f"覆盖 {d['year'].min()}–{d['year'].max()}")

    # ① 「多行」≠「多版本」：先数一天里的指标数
    per_cd = d.groupby(["股票代码", "year"]).agg(
        rows=("预测指标", "size"),
        dates=("公告日期", "nunique"),
        d_first=("公告日期", "min"),
        d_last=("公告日期", "max"),
    ).reset_index()
    print(f"\n【① 一次披露 vs 多次披露】")
    print(f"  每家每年平均行数（多指标） {per_cd['rows'].mean():.2f}")
    print(f"  只有 1 个公告日期的：{(per_cd['dates'] == 1).mean():.2%}")
    print(f"  **≥2 个公告日期的（真·多版本）：{(per_cd['dates'] >= 2).mean():.2%}"
          f"（{int((per_cd['dates'] >= 2).sum()):,} 个公司·年）**")
    if (per_cd["dates"] >= 2).any():
        v = per_cd[per_cd["dates"] >= 2].copy()
        v["gap"] = (v["d_last"] - v["d_first"]).dt.days
        q = v["gap"].quantile([0.25, 0.5, 0.75]).astype(int)
        print(f"  两次之间天数：P25 {q.iloc[0]} · 中位 {q.iloc[1]} · P75 {q.iloc[2]}")

    # ② 内容会不会翻面 —— ⚠️ 必须按**公司**级别比，不能按单个指标比：
    #    实测多版本里常常是**部分指标被修订**（归母只发了一次、营业收入发了两次），
    #    按"归母净利润这一行的公告日期数"筛会把它们全漏掉（第一次跑就是这么漏的）。
    d2 = d.copy()
    d2["公告日期"] = pd.to_datetime(d2["公告日期"], errors="coerce")
    d2["预测数值"] = pd.to_numeric(d2["预测数值"], errors="coerce")
    k = d2[d2["预测指标"] == KEY_METRIC].copy()
    ks = k.sort_values(["股票代码", "year", "公告日期"])
    # 主口径的首次与末次（不管中间有几版）
    kf = ks.groupby(["股票代码", "year"]).first().reset_index()[["股票代码", "year", "预测数值", "预告类型", "公告日期"]]
    kl = ks.groupby(["股票代码", "year"]).last().reset_index()[["股票代码", "year", "预测数值", "预告类型", "公告日期"]]
    m = (per_cd[["股票代码", "year", "dates"]]
         .merge(kf, on=["股票代码", "year"], how="left")
         .merge(kl, on=["股票代码", "year"], how="left", suffixes=("_首次", "_末次")))
    m["首亏"] = m["预测数值_首次"] < 0
    m["末亏"] = m["预测数值_末次"] < 0
    m["翻面"] = m["首亏"] != m["末亏"]
    m["类型变了"] = m["预告类型_首次"] != m["预告类型_末次"]
    m["值变了"] = m["预测数值_首次"] != m["预测数值_末次"]
    multi = m[(m["dates"] >= 2) & m["预测数值_首次"].notna() & m["预测数值_末次"].notna()]
    print(f"\n【② 多版本里内容变不变的（按公司比；主口径 = 归母净利润）】")
    print(f"  多版本公司·年 {int((m['dates'] >= 2).sum()):,} · 其中主口径两端都有的 {len(multi):,}")
    if len(multi):
        print(f"  预测数值变了的 {multi['值变了'].mean():.2%} · 预告类型变了的 {multi['类型变了'].mean():.2%} · "
              f"**盈亏方向翻面的 {int(multi['翻面'].sum())} 个（{multi['翻面'].mean():.2%}）**")
        flips = multi[multi["翻面"]]
        if len(flips):
            print("  翻面的样本（首次→末次）：")
            for _, r in flips.head(8).iterrows():
                print(f"    {r['股票代码']} {int(r['year'])}  {r['公告日期_首次'].date()} {r['预告类型_首次']}"
                      f"({r['预测数值_首次']/1e4:.0f}万) → {r['公告日期_末次'].date()} {r['预告类型_末次']}"
                      f"({r['预测数值_末次']/1e4:.0f}万)")
    else:
        print("  （多版本里主口径两端都有的样本为 0 —— 说明修订发生在别的指标上）")

    # ①b 多日期到底差在哪 —— 抽两个例子把**全部行**打出来（别猜）
    ex = per_cd[per_cd["dates"] >= 2].head(2)
    print(f"\n【①b 多日期差在哪（抽 {len(ex)} 例，全部行照打）】")
    for _, r in ex.iterrows():
        sub = d[(d["股票代码"] == r["股票代码"]) & (d["year"] == r["year"])]
        print(f"  --- {r['股票代码']} {int(r['year'])} ---")
        for _, s in sub.sort_values(["公告日期", "预测指标"]).iterrows():
            vv = s["预测数值"]
            vv = "NaN" if pd.isna(vv) else f"{vv/1e4:.0f}万"
            print(f"      {s['公告日期'].date()}  {str(s['预测指标'])[:16]:18s} {str(s['预告类型']):5s} {vv}")

    # ③ 这些"末次"落在哪 —— 决定滚动链要不要追修订
    allv = m[m["dates"] >= 1]
    lead = (pd.to_datetime(allv["公告日期_末次"]).dt.dayofyear)
    print(f"\n【③ 末次预告落在一年中的哪一天（滚动链的触发窗口）】")
    print(f"  中位 {int(lead.median())} 日（≈ {pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(lead.median()) - 1):%m-%d}）· "
          f"P10 {int(lead.quantile(0.1))} · P90 {int(lead.quantile(0.9))}")

    out = {
        "rows": int(len(d)),
        "co_years": int(per_cd.shape[0]),
        "multi_version_rate": round(float((per_cd["dates"] >= 2).mean()), 4),
        "multi_version_n": int((per_cd["dates"] >= 2).sum()),
        "gap_median_days": int(v["gap"].median()) if (per_cd["dates"] >= 2).any() else None,
        "flip_n": int(multi["翻面"].sum()) if len(multi) else 0,
        "flip_rate": round(float(multi["翻面"].mean()), 4) if len(multi) else None,
        "type_change_rate": round(float(multi["类型变了"].mean()), 4) if len(multi) else None,
        "value_change_rate": round(float(multi["值变了"].mean()), 4) if len(multi) else None,
    }
    (ROOT / "data" / "yjyg_versions.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n读数 -> data/yjyg_versions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
