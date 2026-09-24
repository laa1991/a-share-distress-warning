"""拉取东财「业绩预告」（stock_yjyg_em）—— 年度报告期的预告。

为什么单列：预告是**合法可得**的"提前版本答案"（1–2 月发，正式年报 4 月才出）。
它和「偷看下一期报表」构成一对：同样是提前知道，一个合规、一个不可上线。

用法: python fetch_yjyg.py [--start 2015] [--end 2026]
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
OUT = ROOT / "data" / "raw" / "yjyg"
LOG = ROOT / "data" / "fetch_yjyg_log.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2026)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for year in range(args.start, args.end + 1):
        for date, tag in ((f"{year}0331", "q1"), (f"{year}0630", "h1"), (f"{year}0930", "q3"), (f"{year}1231", "annual")):
            f = OUT / f"{date}.csv"
            if f.exists() and f.stat().st_size > 200:
                rows.append({"period": date, "status": "cached", "rows": 0, "seconds": 0.0})
                continue
            t0 = time.time()
            try:
                d = ak.stock_yjyg_em(date=date)
                tmp = f.with_suffix(".csv.tmp")
                d.to_csv(tmp, index=False, encoding="utf-8-sig")
                tmp.replace(f)
                print(f"OK  {date} ({tag}) rows={len(d)} {time.time() - t0:.1f}s", flush=True)
                rows.append({"period": date, "status": "ok", "rows": len(d), "seconds": round(time.time() - t0, 1)})
            except Exception as e:  # noqa: BLE001
                print(f"!!  {date} ({tag}) {type(e).__name__}: {str(e)[:110]}", flush=True)
                rows.append({"period": date, "status": "FAIL", "rows": 0, "seconds": round(time.time() - t0, 1)})
    pd.DataFrame(rows).to_csv(LOG, index=False, encoding="utf-8-sig")
    print("日志 ->", LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
