"""拉取 A 股定期报告面板（东财「业绩报表 + 三大报表」批量接口）—— 并发版。

设计要点
- 每张表每个报告期一个 CSV，落盘即缓存（可断点续跑，重跑只补缺的）。
- 每次调用**立即**追加一行到 fetch_log.csv：表 / 报告期 / 行数 / 秒数 / 状态。
- 线程池并发（默认 4），任务粒度 = (表, 报告期)，彼此独立。
- 每张表都自带「公告日期」列 —— 这是做 as-of（时点纪律）的锚点。

用法: python fetch_panel.py [--start 2016-03-31] [--end 2026-06-30] [--workers 4]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

os.environ.setdefault("TQDM_DISABLE", "1")

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
LOG = ROOT / "data" / "fetch_log.csv"

TABLES = {
    "yjbb": ak.stock_yjbb_em,   # 业绩报表：每股收益/营收/净利/ROE/毛利率/所处行业/最新公告日期
    "zcfz": ak.stock_zcfz_em,   # 资产负债表
    "lrb": ak.stock_lrb_em,     # 利润表
    "xjll": ak.stock_xjll_em,   # 现金流量表
}

_lock = Lock()
_t0 = time.time()


def periods(start: str, end: str) -> list[str]:
    ps = pd.period_range(start=start, end=end, freq="Q")
    return [f"{p.year}{p.quarter * 3:02d}{'31' if p.quarter in (1, 4) else '30'}" for p in ps]


def log_row(**kw) -> None:
    with _lock:
        header = not LOG.exists()
        with LOG.open("a", encoding="utf-8-sig", newline="") as f:
            if header:
                f.write("table,period,status,rows,seconds,err,at\n")
            f.write(
                "{table},{period},{status},{rows},{seconds},{err},{at}\n".format(
                    err=str(kw.get("err", "")).replace(",", ";"),
                    at=time.strftime("%H:%M:%S"),
                    **{k: v for k, v in kw.items() if k != "err"},
                )
            )


def fetch_one(name: str, date: str) -> dict:
    fn = TABLES[name]
    out = RAW / name / f"{date}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 200:
        n = sum(1 for _ in out.open(encoding="utf-8-sig")) - 1
        return {"table": name, "period": date, "status": "cached", "rows": n, "seconds": 0.0, "err": ""}
    t0 = time.time()
    last = ""
    for _attempt in (1, 2):
        try:
            df = fn(date=date)
            if df is None or len(df) == 0:
                last = "empty"
                time.sleep(2)
                continue
            tmp = out.with_suffix(".csv.tmp")
            df.to_csv(tmp, index=False, encoding="utf-8-sig")
            tmp.replace(out)          # 原子落盘：别的读者永远看不到半个文件
            return {"table": name, "period": date, "status": "ok", "rows": len(df),
                    "seconds": round(time.time() - t0, 2), "err": ""}
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(3)
    return {"table": name, "period": date, "status": "FAIL", "rows": 0,
            "seconds": round(time.time() - t0, 2), "err": last}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-03-31")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    ps = periods(args.start, args.end)
    tasks = [(t, p) for p in ps for t in TABLES]
    print(f"报告期 {len(ps)} 个 × 表 {len(TABLES)} 张 = {len(tasks)} 次调用 · workers={args.workers}", flush=True)

    done = fails = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, t, p): (t, p) for t, p in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            log_row(**r)
            done += 1
            if r["status"] == "FAIL":
                fails += 1
            if r["status"] != "cached":
                flag = "OK " if r["status"] == "ok" else "!! "
                print(f"{flag}[{done:3d}/{len(tasks)}] {r['table']:5s} {r['period']} rows={r['rows']:6d} "
                      f"{r['seconds']:6.1f}s {r['err']}  (t+{time.time() - _t0:.0f}s)", flush=True)
    print(f"\n完成：{done} 次调用，失败 {fails}，用时 {time.time() - _t0:.0f}s；日志 -> {LOG}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
