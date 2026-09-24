"""双源对账 · 第二批科目：资产负债表与现金流量表（东财 vs 新浪）。

第一批只对了营业收入与净利润（`diag_value_crosscheck.py`）。这一批把关键科目铺开：

| 面板列 | 东财 | 新浪（资产负债表 / 现金流量表） |
|---|---|---|
| cash / ar / inventory / total_assets / total_liab / ap | 资产-货币资金 / 资产-应收账款 / 资产-存货 / 资产-总资产 / 负债-总负债 / 负债-应付账款 | 货币资金 / 应收账款 / 存货 / 资产总计 / 负债合计 / 应付账款 |
| equity | 股东权益合计 | **两个候选**：`所有者权益(或股东权益)合计` 与 `归属于母公司股东权益合计` —— 哪个算得准要**比出来**（第一批判名歧义就是用这个办法抓到的） |
| ocf / icf / fcf / net_cf | 经营性/投资性/融资性现金流净额、净现金流 | 经营活动/投资活动/筹资活动产生的现金流量净额、现金及现金等价物净增加额 |

⚠️ 面板的 `fcf` 名字有误导：它来自东财「**融资性**现金流」（financing），不是 free cash flow。

用法: python diag_value_crosscheck2.py [--n 60] [--workers 8]   # 不带 --n 即全量
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = {"bs": DATA / "raw" / "sina_bs", "cf": DATA / "raw" / "sina_cf"}
PANEL = DATA / "panel.pkl"

BS_MAP = {
    "cash": ["货币资金"],
    "ar": ["应收账款"],
    "inventory": ["存货"],
    "total_assets": ["资产总计"],
    "total_liab": ["负债合计"],
    "ap": ["应付账款"],
    "equity": ["所有者权益(或股东权益)合计", "归属于母公司股东权益合计"],
}
CF_MAP = {
    "ocf": ["经营活动产生的现金流量净额"],
    "icf": ["投资活动产生的现金流量净额"],
    "fcf": ["筹资活动产生的现金流量净额"],
    "net_cf": ["现金及现金等价物净增加额"],
}
SINA_NAME = {"bs": "资产负债表", "cf": "现金流量表"}

_lock = threading.Lock()
_progress = {"done": 0, "fetched": 0, "cached": 0, "failed": 0}


def sina_symbol(code: str) -> str:
    return ("sh" if code[0] in "659" else "sz") + code


def fetch_one(kind: str, code: str, tries: int = 3) -> str:
    out = CACHE[kind] / f"{code}.csv"
    if out.exists() and out.stat().st_size > 100:
        status = "cached"
    else:
        status = "failed"
        for i in range(tries):
            try:
                df = ak.stock_financial_report_sina(stock=sina_symbol(code), symbol=SINA_NAME[kind])
                if df is None or len(df) == 0:
                    raise RuntimeError("empty")
                tmp = CACHE[kind] / f"{code}.csv.tmp"
                df.to_csv(tmp, index=False, encoding="utf-8-sig")
                tmp.replace(out)          # 原子改名：半截文件不会冒充成品
                status = "fetched"
                break
            except Exception as exc:  # noqa: BLE001
                if i == tries - 1:
                    status = f"failed:{type(exc).__name__}"
                else:
                    time.sleep(0.5 * (i + 1) + random.random() * 0.5)
    with _lock:
        _progress["done"] += 1
        _progress[status if status in _progress else "failed"] += 1
        if _progress["done"] % 500 == 0:
            print(f"  … {_progress['done']} 已完成（新抓 {_progress['fetched']} · 缓存 {_progress['cached']} "
                  f"· 失败 {_progress['failed']}）", flush=True)
    return status


def load(kind: str, code: str, want: list[str]) -> pd.DataFrame | None:
    f = CACHE[kind] / f"{code}.csv"
    if not f.exists():
        return None
    try:
        head = pd.read_csv(f, nrows=0)
    except Exception:
        return None
    cols = [c for c in want if c in head.columns]
    if "报告日" not in head.columns or not cols:
        return None
    d = pd.read_csv(f, dtype={"报告日": str}, usecols=["报告日"] + cols)
    out = pd.DataFrame({"code": code, "period": pd.to_numeric(d["报告日"], errors="coerce")})
    for c in cols:
        out[f"sina::{c}"] = pd.to_numeric(d[c], errors="coerce")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    panel = pd.read_pickle(PANEL)
    panel = panel[["code", "period", "cash", "ar", "inventory", "total_assets", "total_liab",
                   "ap", "equity", "ocf", "icf", "fcf", "net_cf"]].copy()
    codes = sorted(panel["code"].unique())
    step = max(len(codes) // args.n, 1)
    sample = codes[::step][: args.n]
    print(f"抽样 {len(sample)} 只（每 {step} 只取一只）：{sample[:6]} …", flush=True)

    for kind in ("bs", "cf"):
        CACHE[kind].mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda c: fetch_one(kind, c), sample))
        print(f"{SINA_NAME[kind]}：累计抓 {_progress['fetched']} · 缓存 {_progress['cached']} · 失败 {_progress['failed']}",
              flush=True)

    want_bs = sorted({c for v in BS_MAP.values() for c in v})
    want_cf = sorted({c for v in CF_MAP.values() for c in v})
    bs = [d for d in (load("bs", c, want_bs) for c in sample) if d is not None]
    cf = [d for d in (load("cf", c, want_cf) for c in sample) if d is not None]
    print(f"新浪覆盖：资产负债表 {len(bs)} 只 · 现金流量表 {len(cf)} 只")
    if not bs or not cf:
        print("没抓到 —— 先查函数名/网络，别下结论")
        return 1
    bs, cf = pd.concat(bs, ignore_index=True), pd.concat(cf, ignore_index=True)

    out: dict = {"n_stocks": len(sample)}
    for tag, other, mapping in (("bs", bs, BS_MAP), ("cf", cf, CF_MAP)):
        m = panel[panel["code"].isin(other["code"].unique())].merge(other, on=["code", "period"], how="inner")
        print(f"\n########## {SINA_NAME[tag]}（可比对 {len(m)} 对）##########")
        for col, candidates in mapping.items():
            for cand in candidates:
                sk = f"sina::{cand}"
                if sk not in m.columns:
                    print(f"  {col:14s} ↔ {cand}：新浪那列不存在，跳过")
                    continue
                sub = m[m[col].notna() & m[sk].notna()].copy()
                if not len(sub):
                    continue
                rel = (sub[col] - sub[sk]).abs() / sub[sk].abs().clip(lower=1)
                big = rel >= 1e-2
                key = f"{col} ↔ {cand}"
                out[key] = {
                    "n": int(len(sub)),
                    "exact": round(float((sub[col] == sub[sk]).mean()), 6),
                    "lt_0_1pct": round(float((rel < 1e-3).mean()), 6),
                    "ge_1pct": round(float(big.mean()), 6),
                    "n_ge_1pct": int(big.sum()),
                    "n_codes_ge_1pct": int(sub.loc[big, "code"].nunique()),
                }
                print(f"  {col:14s} ↔ {cand:26s} n={len(sub):6d} 精确 {out[key]['exact']:8.2%} "
                      f"<0.1% {out[key]['lt_0_1pct']:8.2%} ≥1% {out[key]['ge_1pct']:7.2%} "
                      f"({out[key]['n_ge_1pct']} 条 / {out[key]['n_codes_ge_1pct']} 只)")

    (DATA / "value_crosscheck_bs_cf.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n读数 ->", DATA / "value_crosscheck_bs_cf.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
