"""把「业绩预告修正」公告清单变成事件表，并量两件事：
① **修订发生日 → 年报披露日** 的提前量分布（滚动链触发口的依据）；
② 修订落在**谁**身上（与我们的面板能不能对上）、以及**报告期类型**（年度 / 半年度 / 季度）。

⚠️ 方向（好→坏 / 坏→好）要从**公告正文**读，不在这一步 —— 标题不含数字（见 `src/classify_yjyg_rev_dir.py`）。

用法: python diag_yjyg_revisions.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "yjyg_rev"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import PANEL  # noqa: E402
from run_leadtime_curve import base_label  # noqa: E402

EM = re.compile(r"</?em>")
YEAR_IN_TITLE = re.compile(r"(20\d{2})\s*年度")
SEMI = re.compile(r"半年度|半年报|中期")
QUART = re.compile(r"一季度|三季度|第[一三]季度|季度")


def load() -> pd.DataFrame:
    frames = []
    for f in sorted(RAW.glob("*.csv")):
        d = pd.read_csv(f, dtype=str)
        d["_file_year"] = int(f.stem)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["标题"] = d["公告标题"].fillna("").map(lambda s: EM.sub("", s))
    d["日期"] = pd.to_datetime(d["公告时间"], errors="coerce")
    d["代码"] = d["代码"].astype(str).str.zfill(6)

    def period(t: str) -> str:
        if SEMI.search(t):
            return "半年度"
        if QUART.search(t):
            return "季度"
        if YEAR_IN_TITLE.search(t):
            return "年度"
        return "未标"

    d["报告期类型"] = d["标题"].map(period)
    # 年度修正公告一般在下一年 1–4 月发；标题带年份的优先，否则按公告年 - 1 推
    def fy(r) -> float:
        m = YEAR_IN_TITLE.search(r["标题"])
        if m:
            return float(m.group(1))
        return float(r["_file_year"] - 1) if r["报告期类型"] == "年度" else float("nan")

    d["会计年度"] = d.apply(fy, axis=1)
    return d


def main() -> int:
    d = load()
    print(f"修正公告合计 {len(d):,} 条（2015–2026）· 唯一公司 {d['代码'].nunique():,}")
    print("\n【报告期类型】")
    print(d["报告期类型"].value_counts().to_string())
    ann = d[d["报告期类型"] == "年度"].copy()
    print(f"\n年度修正公告 {len(ann):,} 条 —— 逐年：")
    print(ann.groupby("_file_year").size().to_string())

    # 与时点对齐：这批修正发生在年报披露之前还是之后？
    from run_experiments import make_samples  # noqa: E402
    panel = pd.read_pickle(PANEL)
    lab = make_samples(panel)[["code", "year", "t_label"]].copy()   # t_label 在 make_samples 里，不在 base_label
    lab["code"] = lab["code"].astype(str).str.zfill(6)
    ann["会计年度"] = ann["会计年度"].astype("Int64")
    m = ann.merge(lab, left_on=["代码", "会计年度"], right_on=["code", "year"], how="left")
    m["年报日"] = pd.to_datetime(m["t_label"], errors="coerce")
    m["距年报天数"] = (m["年报日"] - m["日期"]).dt.days
    ok = m[m["年报日"].notna() & m["距年报天数"].notna()]
    print(f"\n【对上面板】{len(ok):,}/{len(ann):,} 条找到对应年报日（{len(ok)/len(ann):.1%}）")
    if len(ok):
        q = ok["距年报天数"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).astype(int)
        print(f"  修正日 → 年报日：P5 {q.iloc[0]} · P25 {q.iloc[1]} · **中位 {q.iloc[2]}** · P75 {q.iloc[3]} · P95 {q.iloc[4]} 天")
        before = (ok["距年报天数"] > 0).mean()
        print(f"  **发生在年报披露之前**的占 {before:.2%}（其余 {1-before:.2%} 是年报当天/之后）")
        print(f"  集中在几月：")
        print(ok["日期"].dt.month.value_counts().sort_index().to_string())
    print(f"\n按年（会计年度）看修订条数：")
    print(ann.groupby("会计年度").size().tail(10).to_string())

    out = {
        "total": int(len(d)),
        "annual": int(len(ann)),
        "matched": int(len(ok)),
        "lead_days": {str(k): int(v) for k, v in q.items()} if len(ok) else {},
        "before_annual_rate": round(float(before), 4) if len(ok) else None,
        "by_fy": {str(k): int(v) for k, v in ann.groupby("会计年度").size().items()},
    }
    (ROOT / "data" / "yjyg_revisions.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n读数 -> data/yjyg_revisions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
