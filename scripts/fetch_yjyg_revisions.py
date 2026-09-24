"""抓「业绩预告修正」公告清单（巨潮公告检索，keyword=业绩预告修正），按年切。

为什么单列：东财的业绩预告表**只返回一条、且是修订后那一版**（见 `docs/设计-业绩预告滚动链.md` §7），
所以"有没有修订、什么时候修订"必须从**公告源**拿。这是滚动链触发口的第一手依据。

用法: python fetch_yjyg_revisions.py [--start 2015] [--end 2026]
产出: data/raw/yjyg_rev/<year>.csv + data/fetch_yjyg_rev_log.csv
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
OUT = ROOT / "data" / "raw" / "yjyg_rev"
LOG = ROOT / "data" / "fetch_yjyg_rev_log.csv"
KEYWORD = "业绩预告修正"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2026)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for y in range(args.start, args.end + 1):
        f = OUT / f"{y}.csv"
        if f.exists() and f.stat().st_size > 60:
            n = len(pd.read_csv(f))
            rows.append({"year": y, "status": "cached", "rows": n, "seconds": 0.0})
            print(f"  {y}: {n:4d} 条 [cached]", flush=True)
            continue
        t0 = time.time()
        try:
            d = ak.stock_zh_a_disclosure_report_cninfo(symbol="", market="沪深京", keyword=KEYWORD,
                                                       start_date=f"{y}0101", end_date=f"{y}1231")
            # ⚠️ 原子改名：中途挂掉不会留下半个文件被当成缓存
            tmp = f.with_suffix(".csv.tmp")
            d.to_csv(tmp, index=False, encoding="utf-8-sig")
            tmp.replace(f)
            print(f"  {y}: {len(d):4d} 条 {time.time() - t0:.1f}s", flush=True)
            rows.append({"year": y, "status": "ok", "rows": len(d), "seconds": round(time.time() - t0, 1)})
        except Exception as e:  # noqa: BLE001
            print(f"  !! {y} {type(e).__name__}: {str(e)[:110]}", flush=True)
            rows.append({"year": y, "status": f"FAIL:{type(e).__name__}", "rows": 0, "seconds": round(time.time() - t0, 1)})
    log = pd.DataFrame(rows)
    log.to_csv(LOG, index=False, encoding="utf-8-sig")
    print(f"\n合计 {log['rows'].sum()} 条 · 账 -> {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
