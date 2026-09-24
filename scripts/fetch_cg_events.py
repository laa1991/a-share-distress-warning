"""抓「报表外事件」的两个专题统计（巨潮）：公司诉讼、对外担保 —— 按**季度**切窗口。

为什么按季度而不是整年：接口的 `公告统计区间` 是它自己的窗口，整年一个数只能等年末才知道，
拿它去预测当年就是偷看；按季度取，才能让「4 月决策时只看 ≤3 月的事件」成立。

用法: python fetch_cg_events.py [--start 2016] [--end 2026] [--workers 4]
"""
from __future__ import annotations

import argparse
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw"
LOG = ROOT / "data" / "fetch_cg_log.csv"
_lock = threading.Lock()
_rows: list[dict] = []

KINDS = {"lawsuit": ("stock_cg_lawsuit_cninfo", "诉讼"), "guarantee": ("stock_cg_guarantee_cninfo", "担保")}


def quarters(start: int, end: int) -> list[tuple[str, str, str]]:
    out = []
    for y in range(start, end + 1):
        for q, (m0, m1, last) in enumerate(
                [(1, 3, 31), (4, 6, 30), (7, 9, 30), (10, 12, 31)], start=1):
            out.append((f"{y}{q}", f"{y}{m0:02d}01", f"{y}{m1:02d}{last:02d}"))
    return out


def fetch(kind: str, tag: str, s: str, e: str, tries: int = 3) -> None:
    d = OUT / f"cg_{kind}"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{tag}.csv"
    if out.exists() and out.stat().st_size > 50:
        status, n = "cached", sum(1 for _ in out.open(encoding="utf-8-sig")) - 1
    else:
        status, n = "failed", 0
        fn = getattr(ak, KINDS[kind][0])
        for i in range(tries):
            try:
                df = fn(symbol="全部", start_date=s, end_date=e)
                tmp = d / f"{tag}.csv.tmp"
                df.to_csv(tmp, index=False, encoding="utf-8-sig")
                tmp.replace(out)
                status, n = "ok", len(df)
                break
            except Exception as exc:  # noqa: BLE001
                if i == tries - 1:
                    status = f"failed:{type(exc).__name__}"
                else:
                    time.sleep(0.6 * (i + 1) + random.random())
    with _lock:
        _rows.append({"kind": kind, "period": tag, "start": s, "end": e, "rows": n, "status": status})
        print(f"  {KINDS[kind][1]} {tag} {s}–{e}: {n} 行 [{status}]", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = [(kind, tag, s, e) for kind in KINDS for (tag, s, e) in quarters(args.start, args.end)]
    print(f"共 {len(jobs)} 个窗口（{len(KINDS)} 张表 × {len(jobs)//len(KINDS)} 个季度）")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda j: fetch(*j), jobs))

    log = pd.DataFrame(_rows).sort_values(["kind", "period"])
    log.to_csv(LOG, index=False, encoding="utf-8-sig")
    print(f"\n合计 {len(log)} 次调用 · 失败 {int(log['status'].str.startswith('failed').sum())} · "
          f"空表 {int((log['rows'] == 0).sum())} 个")
    print(log.groupby("kind")["rows"].agg(["count", "sum"]).to_string())
    print("账 ->", LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
