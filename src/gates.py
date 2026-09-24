"""两道**确定性**的闸（把"我记得要看"变成"每次自动跑"）。

来历：这两道闸都是踩出来的 ——
① **PIT 闸门**：回放第一版从事件最早那天就应用，而名单要等该年一季报披露后才算得出来
   ⇒ **在自己的回放里犯了这个项目要治的错**；
② **右端截断**：年报都披露完了，之后发生的公告不该再改这份名单（漏了它时，日更版与一次性回放差 1 条）。

它们**不该由判断器（Jev/LLM）来担**：要的是每次都一样的确定性，不是概率。
（见 `docs/探索-判断器的效率（GBDT vs Jev vs LLM）.md` §8 的结论 1。）

用法: python gates.py --selftest        # 复现实测的四个数（2024 丢 228 / 2023 丢 268；右端 2024 丢 0 / 2023 丢 1）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def pit_gate(events: pd.DataFrame, base: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """① 一条事件只有在**该公司的骨架分数已存在**之后才允许应用。返回 (可用事件, 丢掉条数)。"""
    avail = base.set_index("code")["stage_date"].to_dict()
    ev = events.copy()
    ev["可用日"] = ev["code"].map(avail)
    keep = ev[ev["可用日"].notna() & (ev["date"] >= ev["可用日"])].copy()
    return keep, int(len(events) - len(keep))


def right_edge(events: pd.DataFrame, ann_end: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    """② 右端截断：只保留**年报最晚披露日之前**的事件。"""
    keep = events[events["date"] <= ann_end].copy()
    return keep, int(len(events) - len(keep))


def selftest() -> int:
    from rolling_replay import load_base, load_events  # noqa: E402
    sc = pd.read_csv(ROOT / "data" / "leadtime_scores.csv", usecols=["year", "stage_date", "stage"])
    expect = {2023: (268, 1), 2024: (228, 0)}
    ok_all = True
    for year in (2023, 2024):
        base = load_base(year)
        ev = load_events(year)
        gated, d1 = pit_gate(ev, base)
        ann_end = pd.to_datetime(sc[(sc["stage"].str.contains("⑥")) & (sc["year"] == year)]["stage_date"]).max()
        gated2, d2 = right_edge(gated, ann_end)
        exp = expect[year]
        ok = (d1, d2) == exp
        ok_all &= ok
        print(f"  {year}：PIT 闸丢 {d1}（期望 {exp[0]}）· 右端截断丢 {d2}（期望 {exp[1]}）"
              f" · 剩 {len(gated2)} 条  {'✅' if ok else '❌'}")
    print("自检：", "✅ 两道闸复现实测值" if ok_all else "❌ 与实测值不符（先查输入或闸的定义）")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
