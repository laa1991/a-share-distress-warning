"""拉取巨潮「定期报告预约/实际披露时间表」—— 干净的 PIT 锚点（第二个源）。

为什么单列：东财那四张批量表里的「公告日期」**不可直接当披露日**（见 docs/口径.md §源对账），
而巨潮这张表逐期给 `实际披露` —— 这是「这份报表哪天真的公开了」的直接读数，且与东财**不是一个源**。

用法: python fetch_disclosure.py [--start 2016] [--end 2026]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "disclosure"
LOG = ROOT / "data" / "fetch_disclosure_log.csv"

SUFFIX = [(1, "一季"), (2, "半年报"), (3, "三季"), (4, "年报")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--end", type=int, default=2026)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for year in range(args.start, args.end + 1):
        for q, suf in SUFFIX:
            period = f"{year}{suf}"
            f = OUT / f"{year}{q:02d}.csv"
            if f.exists() and f.stat().st_size > 200:
                rows.append({"period": period, "status": "cached", "rows": 0, "seconds": 0.0, "err": ""})
                continue
            t0 = time.time()
            try:
                d = ak.stock_report_disclosure(market="沪深京", period=period)
                tmp = f.with_suffix(".csv.tmp")
                d.to_csv(tmp, index=False, encoding="utf-8-sig")
                tmp.replace(f)
                print(f"OK  {period} rows={len(d)} {time.time() - t0:.1f}s", flush=True)
                rows.append({"period": period, "status": "ok", "rows": len(d), "seconds": round(time.time() - t0, 1), "err": ""})
            except Exception as e:  # noqa: BLE001
                print(f"!!  {period} {type(e).__name__}: {str(e)[:120]}", flush=True)
                rows.append({"period": period, "status": "FAIL", "rows": 0, "seconds": round(time.time() - t0, 1),
                             "err": f"{type(e).__name__}: {str(e)[:120]}"})
    pd.DataFrame(rows).to_csv(LOG, index=False, encoding="utf-8-sig")
    print("日志 ->", LOG)
    return 1 if any(r["status"] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
