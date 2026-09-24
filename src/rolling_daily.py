"""滚动链 · **日更循环**（从"能回放"到"能日更"）：按日推进，读当天新公告 → 更新状态 → 输出当日名单 diff + 变更台账。

与 `rolling_replay.py` 的关系：那个是**一次性走完**（给窗口、出总账），这个是**一天一天走**（给某一天、出当日 diff）。
两者用的是同一套骨架、同一套 PIT 闸门、同一套事件口径 ⇒ **总数必须对得上**（这就是自检）。

产物：
- `data/rolling_daily_<year>_ledger.csv`  变更台账：**谁 / 哪一天 / 因为哪条公告 / 从什么状态到什么状态 / 在不在名单里**
- `data/rolling_daily_<year>_days.csv`    逐日汇总：当天变更数、名单内变更数、名单里"已明说会亏"的累计数与"未确认"的累计数
- `data/rolling_daily_<year>.json`        总账 + 与一次性回放的对账

用法: python rolling_daily.py --year 2024
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rolling_replay import load_base, load_events, state_of  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--top", type=float, default=0.10)
    args = ap.parse_args()

    base = load_base(args.year)
    base["在名单"] = base["score"] >= base["score"].quantile(1 - args.top)
    in_list = set(base.loc[base["在名单"], "code"])
    score_of = base.set_index("code")["score"].to_dict()
    date_of = base.set_index("code")["stage_date"].to_dict()

    ev = load_events(args.year)
    if ev.empty:
        print(f"{args.year} 年没有事件")
        return 1
    # ① PIT 闸门：事件必须晚于该公司骨架分数的可用日
    ev["可用日"] = ev["code"].map(date_of)
    ev = ev[ev["可用日"].notna() & (ev["date"] >= ev["可用日"])].copy()
    # ② 跨源日期陷阱：有过修订的公司，其预告事件的日期＝最后一次披露日 ⇒ 不参与建立"修订前"状态
    rev_keys = set(ev.loc[ev["kind"] == "修正", "code"])
    ev = ev[~((ev["kind"] == "预告") & (ev["code"].isin(rev_keys)))].copy()
    ev = ev.sort_values(["date", "code"]).reset_index(drop=True)
    # ③ 窗口右端：年报都披露完了，再发生的公告不该改这份名单（与一次性回放同一口径）
    sc = pd.read_csv(ROOT / "data" / "leadtime_scores.csv", usecols=["code", "year", "stage_date", "stage"])
    a6 = sc[(sc["stage"].str.contains("⑥")) & (sc["year"] == args.year)]
    ann_end = pd.to_datetime(a6["stage_date"]).max()
    late = int((ev["date"] > ann_end).sum())
    ev = ev[ev["date"] <= ann_end].copy()

    state: dict[str, str] = {}
    ledger, days = [], []
    for day, grp in ev.groupby("date"):
        today = []
        for _, r in grp.iterrows():
            key = r["code"]
            new = state_of(r["kind"], r["type"], r["pred"], r["dir"])
            old = state.get(key, "预测中")
            if new == old:
                continue
            state[key] = new
            rec = {"date": day, "code": key, "from": old, "to": new, "kind": r["kind"],
                   "title": r["title"], "公告日": r["date"], "在名单": key in in_list,
                   "score": round(float(score_of.get(key, float("nan"))), 6),
                   "首次确认": old == "预测中"}
            today.append(rec)
            ledger.append(rec)
        if today:
            inl = [x for x in today if x["在名单"]]
            stated_now = sum(1 for c in in_list if state.get(c, "").startswith(("已明说", "已修订")))
            will = sum(1 for c in in_list if state.get(c) == "已明说会亏")
            days.append({"date": day, "变更": len(today), "名单内变更": len(inl),
                         "名单内·明说会亏累计": will,
                         "名单内·已确认累计": stated_now,
                         "名单内·未确认累计": len(in_list) - stated_now})

    led = pd.DataFrame(ledger)
    dys = pd.DataFrame(days)
    led.to_csv(ROOT / "data" / f"rolling_daily_{args.year}_ledger.csv", index=False, encoding="utf-8-sig")
    dys.to_csv(ROOT / "data" / f"rolling_daily_{args.year}_days.csv", index=False, encoding="utf-8-sig")

    n_list = len(in_list)
    print(f"=== 日更循环：会计年度 {args.year} ===")
    print(f"名单 {n_list} 只 · 事件（过两道闸后）{len(ev)} 条 · 有变更的天数 {len(dys)} 天")
    print(f"（右端截到年报披露最晚日 {ann_end:%Y-%m-%d}，丢掉 {late} 条" + ("）" if late == 0 else " —— 年报都出来了，再发的公告不该改这份名单）"))
    print(f"**变更总账 {len(led)} 条**（首次确认 {int(led['首次确认'].sum())} · 后续迁移 {int((~led['首次确认']).sum())}）· "
          f"名单内 {int(led['在名单'].sum())} 条")
    if len(dys):
        print(f"每天要人看的量：变更中位 **{int(dys['变更'].median())} 条/天** · 峰值 {int(dys['变更'].max())} 条"
              f"（{dys.loc[dys['变更'].idxmax(), 'date']:%Y-%m-%d}）· 名单内变更 ≤{int(dys['名单内变更'].max())} 条/天")
        first = dys.iloc[0]
        last = dys.iloc[-1]
        print(f"名单状态推进：明说会亏 {int(first['名单内·明说会亏累计'])} → **{int(last['名单内·明说会亏累计'])}** 只 · "
              f"未确认 {int(first['名单内·未确认累计'])} → **{int(last['名单内·未确认累计'])}** 只")
        print("\n台账样例（名单内，前后各 3 条）：")
        il = led[led["在名单"]]
        for _, r in pd.concat([il.head(3), il.tail(3)]).iterrows():
            print(f"  {r['date']:%Y-%m-%d}  {r['code']}  {r['from']} → {r['to']}   [{r['kind']}] {str(r['title'])[:40]}")

    # 与一次性回放对账（同一套闸门 ⇒ 总数必须一致）
    ref = ROOT / "data" / f"rolling_replay_{args.year}.json"
    check = None
    if ref.exists():
        r = json.loads(ref.read_text(encoding="utf-8"))
        check = {"replay_changes": r["changes"], "daily_changes": int(len(led)),
                 "match": int(r["changes"]) == int(len(led))}
        print(f"\n【自检】与一次性回放对账：replay {r['changes']} / daily {len(led)} ⇒ "
              f"{'✅ 一致' if check['match'] else '❌ 不一致（两套闸门没对齐）'}")

    out = {"year": args.year, "list_size": n_list, "events": int(len(ev)), "changes": int(len(led)),
           "in_list_changes": int(led["在名单"].sum()), "days_with_changes": int(len(dys)),
           "peak_per_day": int(dys["变更"].max()) if len(dys) else 0,
           "cross_check": check}
    (ROOT / "data" / f"rolling_daily_{args.year}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n台账 -> data/rolling_daily_{args.year}_ledger.csv · 逐日 -> data/rolling_daily_{args.year}_days.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
