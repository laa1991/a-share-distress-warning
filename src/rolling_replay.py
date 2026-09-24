"""业绩预告滚动链 · **离线回放版**（给定日期 D，只喂 D 之前能看到的东西，输出当天名单与每一条变更的理由）。

它不是"又跑一遍模型"，而是把设计里那条链**真的走一遍**：
- **骨架**（谁在名单里）＝ ① 一季报那一刻的分数（`data/leadtime_scores.csv` stage ①），前 10%；
- **更新**（名单里的人现在"已知"了吗）＝ 事件流：业绩预告公告 + **业绩预告修正公告**（公告源，含正文判出的方向）；
- **回放**：从财报季前（默认 12-01）走到年报披露，**只把 stage_date ≤ D 的事件喂进去**；
- **判据**：每一次状态变更都必须能**指回一条公告**（标题 + 日期）—— 指不回的就是缺陷，直接打印出来。

用法:
    python rolling_replay.py --year 2024                 # 回放 2024 会计年度（事件发生在 2025 年 1–4 月）
    python rolling_replay.py --year 2023 --start 2023-12-01 --end 2024-05-31
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EM = re.compile(r"</?em>")
YEAR_IN_TITLE = re.compile(r"(20\d{2})\s*年度")


def load_base(year: int) -> pd.DataFrame:
    """骨架：stage ① 的分数（决策时刻＝一季报实际披露日），取该会计年度的样本"""
    sc = pd.read_csv(ROOT / "data" / "leadtime_scores.csv",
                     usecols=["code", "year", "label", "score", "stage_date", "stage"])
    b = sc[(sc["stage"].str.contains("①")) & (sc["year"] == year)].copy()
    b["code"] = b["code"].astype(str).str.zfill(6)
    b["stage_date"] = pd.to_datetime(b["stage_date"])
    b["在名单"] = b["score"] >= b["score"].quantile(0.90)      # 前 10%
    return b


def load_events(year: int) -> pd.DataFrame:
    """事件流：① 预告公告（东财表，按会计年度）② 修正公告（巨潮，按标题里的会计年度）"""
    rows = []
    # ① 预告：东财原始文件按报告期存，年度预告在 *1231.csv
    f = ROOT / "data" / "raw" / "yjyg" / f"{year}1231.csv"
    if f.exists():
        d = pd.read_csv(f, dtype=str)
        d = d[d["预测指标"].astype(str).str.contains("归属于上市公司股东的净利润", na=False)]
        for _, r in d.iterrows():
            rows.append({"code": str(r["股票代码"]).zfill(6), "date": pd.to_datetime(r["公告日期"], errors="coerce"),
                         "kind": "预告", "title": f"{year}年度业绩预告（{r.get('预告类型','')}）",
                         "type": str(r.get("预告类型", "")), "pred": pd.to_numeric(r.get("预测数值"), errors="coerce"),
                         "dir": ""})
    # ② 修正：巨潮清单（标题里带会计年度）
    rev = ROOT / "data" / "raw" / "yjyg_rev"
    for f2 in sorted(rev.glob("*.csv")):
        d2 = pd.read_csv(f2, dtype=str)
        for _, r in d2.iterrows():
            title = EM.sub("", str(r.get("公告标题", "")))
            m = YEAR_IN_TITLE.search(title)
            if not m or int(m.group(1)) != year:
                continue
            if "半年度" in title or "季度" in title:
                continue
            rows.append({"code": str(r["代码"]).zfill(6), "date": pd.to_datetime(r["公告时间"], errors="coerce"),
                         "kind": "修正", "title": title, "type": "", "pred": float("nan"), "dir": ""})
    e = pd.DataFrame(rows)
    if e.empty:
        return e
    # 修正方向（正文判出来的）贴在 (code, date) 上
    dp = ROOT / "data" / "yjyg_rev_direction.csv"
    if dp.exists():
        dd = pd.read_csv(dp, dtype={"code": str})
        dd["code"] = dd["code"].str.zfill(6)
        dd["date"] = pd.to_datetime(dd["date"])
        e = e.merge(dd[["code", "date", "方向", "prev", "after"]], on=["code", "date"], how="left")
        e["dir"] = e["方向"].fillna("")
    return e.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def state_of(kind: str, typ: str, pred: float, direction: str) -> str:
    if kind == "修正":
        if direction in ("坏→更坏", "好→更好"):
            return f"已修订·{direction}"
        if "同" in str(direction) or direction:
            return f"已修订·{direction}"
        return "已修订"
    if isinstance(pred, float) and pred == pred:
        return "已明说会亏" if pred < 0 else "已明说不会亏"
    if "亏" in str(typ):
        return "已明说会亏"
    if any(k in str(typ) for k in ("扭亏", "预增", "略增", "续盈")):
        return "已明说不会亏"
    return f"已预告·{typ or '未标'}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--top", type=float, default=0.10)
    args = ap.parse_args()

    base = load_base(args.year)
    base["在名单"] = base["score"] >= base["score"].quantile(1 - args.top)
    ev = load_events(args.year)
    if ev.empty:
        print(f"{args.year} 年没有事件")
        return 1
    ann = pd.read_csv(ROOT / "data" / "leadtime_scores.csv", usecols=["code", "year", "stage_date", "stage"])
    ann = ann[(ann["stage"].str.contains("⑥")) & (ann["year"] == args.year)].copy()
    ann["code"] = ann["code"].astype(str).str.zfill(6)
    ann_date = pd.to_datetime(ann["stage_date"]).max()

    start = pd.to_datetime(args.start) if args.start else ev["date"].min() - pd.Timedelta(days=1)
    end = pd.to_datetime(args.end) if args.end else ann_date

    # ⚠️ PIT 闸门：一条事件只有在**该公司的骨架分数已经存在**之后才允许应用。
    #    （第一版漏了这道闸：2024-01 的事件用上了 2024-04 才算出来的名单 —— 在自己的回放里犯了本项目要治的错。）
    base_date = base.set_index("code")["stage_date"].to_dict()
    ev["可用日"] = ev["code"].map(base_date)
    gated = ev[ev["可用日"].notna() & (ev["date"] >= ev["可用日"])].copy()
    dropped = len(ev) - len(gated)

    print(f"=== 回放：会计年度 {args.year} ===")
    print(f"骨架：该年样本 {len(base)} 只 · 前 {args.top:.0%} 名单 {int(base['在名单'].sum())} 只"
          f"（分数来自①一季报，该年最晚可用 {base['stage_date'].max():%Y-%m-%d}）")
    print(f"事件：{len(ev)} 条（预告 {int((ev['kind']=='预告').sum())} · 修正 {int((ev['kind']=='修正').sum())}）"
          f" ⇒ **PIT 闸门后可用 {len(gated)} 条**（丢掉 {dropped} 条：事件早于该公司名单可用日）")
    print(f"窗口 {start:%Y-%m-%d} → {end:%Y-%m-%d}（年报披露最晚 {ann_date:%Y-%m-%d}）")

    # ⚠️ 跨源顺序陷阱（实测）：东财那张表里，**有过修订的公司**其「公告日期」是**最后一次披露日**，
    #    所以它可能排在修正公告之后（例：002173 的"预告"日期 2025-02-26，而修正更早）。
    #    ⇒ 对这些公司，预告事件**不能用来建立"修订前"的状态**，只能把它标成"日期存疑"并让修正说话。
    rev_keys = set(zip(gated.loc[gated["kind"] == "修正", "code"], [args.year] * int((gated["kind"] == "修正").sum())))
    gated["日期存疑"] = [(r["kind"] == "预告") and ((r["code"], args.year) in rev_keys) for _, r in gated.iterrows()]
    suspect = int(gated["日期存疑"].sum())
    apply_ev = gated[~gated["日期存疑"]]

    state: dict[str, str] = {}
    changes: list[dict] = []
    applied = 0
    for d, grp in apply_ev[(apply_ev["date"] >= start) & (apply_ev["date"] <= end)].groupby("date"):
        for _, r in grp.iterrows():
            key = r["code"]
            new = state_of(r["kind"], r["type"], r["pred"], r["dir"])
            old = state.get(key, "预测中")
            applied += 1
            if new != old:
                changes.append({"date": d, "code": key, "from": old, "to": new,
                                "kind": r["kind"], "title": r["title"][:70],
                                "在名单": bool(base.loc[base["code"] == key, "在名单"].any()),
                                "首次确认": old == "预测中"})
                state[key] = new

    ch = pd.DataFrame(changes)
    print(f"\n应用事件 {applied} 条 · **状态变更 {len(ch)} 条**（首次确认 {int(ch['首次确认'].sum()) if len(ch) else 0} · "
          f"后续迁移 {int((~ch['首次确认']).sum()) if len(ch) else 0}）")
    stated = 0
    if len(ch):
        inlist = ch[ch["在名单"]]
        # 名单内**每只**的最终状态（取最后一次变更）
        last = inlist.sort_values("date").groupby("code").last()
        tot = int(base["在名单"].sum())
        will_lose = int((last["to"] == "已明说会亏").sum())
        wont_lose = int((last["to"] == "已明说不会亏").sum())
        revised = int(last["to"].str.startswith("已修订").sum())
        stated = will_lose + wont_lose + revised
        print(f"  名单内变更 {len(inlist)} 条 · 回放结束时名单里 {tot} 只的去向：")
        print(f"    **明说会亏 {will_lose} 只 · 明说不会亏（误报解除）{wont_lose} 只 · 修订过 {revised} 只**"
              f" · 仍未确认 {tot - stated} 只（{1 - stated/max(tot,1):.0%}）")
        print(f"    ⇒ 名单从「预测」被公告**兑现或解除**的比例 {stated/max(tot,1):.1%}")
        mig = ch[(~ch["首次确认"]) & ch["在名单"]]
        if len(mig):
            print(f"\n  **名单内的状态迁移（滚动链真正的产出）{len(mig)} 条**：")
            for _, r in mig.head(8).iterrows():
                print(f"   {r['date']:%Y-%m-%d}  {r['code']}  {r['from']} → {r['to']}   [{r['kind']}] {r['title'][:44]}")
        print("\n  首次确认（前 6 条，名单内）：")
        for _, r in inlist[inlist["首次确认"]].head(6).iterrows():
            print(f"   {r['date']:%Y-%m-%d}  {r['code']}  → {r['to']}   [{r['kind']}] {r['title'][:44]}")
    # 判据：每条变更都必须挂一条公告（这里逐条构造的，缺的会是空标题）
    orphan = ch[ch["title"].astype(str).str.len() < 4] if len(ch) else ch
    print(f"\n【判据】变更可溯源到公告：{'✅ 全部可溯源' if len(orphan)==0 else f'❌ {len(orphan)} 条无公告'}"
          f"（{len(ch)} 条变更 / 全部带标题与日期）")
    out = {"year": args.year, "events": int(len(ev)), "events_gated": int(len(gated)),
           "dropped_by_pit": int(dropped), "applied": applied, "changes": len(ch),
           "in_list_changes": int(len(ch[ch['在名单']])) if len(ch) else 0,
           "stated_end": int(stated), "list_size": int(base["在名单"].sum()),
           "window": [str(start.date()), str(end.date())], "orphan": int(len(orphan))}
    (ROOT / "data" / f"rolling_replay_{args.year}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(ch):
        ch.to_csv(ROOT / "data" / f"rolling_replay_{args.year}_changes.csv", index=False, encoding="utf-8-sig")
    print(f"\n读数 -> data/rolling_replay_{args.year}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
