"""拉取东财「业绩快报」（stock_yjkb_em）—— 滚动链路上的第二站。

链路：三季报（10 月）→ **业绩预告**（次年 1 月，区间/类型）→ **业绩快报**（2–4 月，**实际数字**）→ 年报（4 月）。
快报比预告硬：预告给的是区间与类型，快报给的是营收/净利/每股收益的实数。

用法: python fetch_yjkb.py [--start 2015] [--end 2026]
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
OUT = ROOT / "data" / "raw" / "yjkb"
LOG = ROOT / "data" / "fetch_yjkb_log.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2026)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for year in range(args.start, args.end + 1):
        for date in (f"{year}0331", f"{year}0630", f"{year}0930", f"{year}1231"):
            f = OUT / f"{date}.csv"
            if f.exists() and f.stat().st_size > 200:
                rows.append({"period": date, "status": "cached", "rows": 0, "seconds": 0.0})
                continue
            t0 = time.time()
            last = ""
            for _attempt in (1, 2):
                try:
                    d = ak.stock_yjkb_em(date=date)
                    if d is None or len(d) == 0:
                        last = "empty"
                        time.sleep(2)
                        continue
                    tmp = f.with_suffix(".csv.tmp")
                    d.to_csv(tmp, index=False, encoding="utf-8-sig")
                    tmp.replace(f)
                    print(f"OK  {date} rows={len(d)} {time.time() - t0:.1f}s", flush=True)
                    rows.append({"period": date, "status": "ok", "rows": len(d), "seconds": round(time.time() - t0, 1)})
                    break
                except Exception as e:  # noqa: BLE001
                    last = f"{type(e).__name__}: {str(e)[:110]}"
                    time.sleep(3)
            else:
                print(f"!!  {date} {last}", flush=True)
                rows.append({"period": date, "status": "FAIL", "rows": 0, "seconds": round(time.time() - t0, 1)})
    pd.DataFrame(rows).to_csv(LOG, index=False, encoding="utf-8-sig")
    print("日志 ->", LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
