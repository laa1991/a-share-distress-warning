"""数值层的双源对账：**东财（面板） vs 新浪（利润表）** 逐期比营业收入与净利润。

为什么要有这一格：口径文档 §7 第 1 条一直挂着「只有日期做了双源对账，面板的数值没有」——
日期错了会让 PIT 锚点漂移（那已经量过：365 天），但**数值错了会直接污染标签和特征**，而且不报警。

对齐规则（错了就没法比）：
- 概念对齐：面板 `rev_l` / `np_l` 来自东财**利润表**（营业总收入 / 净利润）⇒ 对新浪的 `营业总收入` / `净利润`。
  （面板 `rev` / `np` 来自东财**业绩报表**，是另一个概念的口径，不用来对账，只做旁证。）
- 期间对齐：两边都按**报告期**（`报告日` == `period`）比，且都用**年初至今累计**口径。
- 单位：都是元，不做换算。
- 判据：相对差 |a-b| / max(|a|,1) —— 分档统计 **精确相等 / <0.01% / <0.1% / <1% / ≥1%**，并把 ≥1% 的逐条列出来人工看。

用法: python diag_value_crosscheck.py [--n 40] [--workers 6]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "raw" / "sina"
PANEL = DATA / "panel.pkl"


def sina_symbol(code: str) -> str:
    return ("sh" if code[0] in "659" else "sz") + code


def fetch_one(code: str) -> str:
    out = CACHE / f"{code}.csv"
    if out.exists() and out.stat().st_size > 100:
        return "cached"
    df = ak.stock_financial_report_sina(stock=sina_symbol(code), symbol="利润表")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return "fetched"


def load_sina(code: str) -> pd.DataFrame | None:
    f = CACHE / f"{code}.csv"
    if not f.exists():
        return None
    try:
        d = pd.read_csv(f, dtype={"报告日": str})
    except Exception:
        return None
    if "报告日" not in d.columns:
        return None
    # 报表版式随行业不同：银行没有「营业总收入」（只有营业收入/净利息收入…）⇒ 退到「营业收入」
    rev_col = "营业总收入" if "营业总收入" in d.columns else ("营业收入" if "营业收入" in d.columns else None)
    if rev_col is None or "净利润" not in d.columns:
        return None
    out = pd.DataFrame({
        "code": code,
        "period": pd.to_numeric(d["报告日"], errors="coerce"),
        "sina_rev": pd.to_numeric(d[rev_col], errors="coerce"),
        "sina_np": pd.to_numeric(d["净利润"], errors="coerce"),
    })
    # ★ 关键：新浪的「净利润」是**含少数股东权益**的净利润；东财的「净利润」是**归母**。
    #   两边不同名对不上不是数据错，是**概念不同** —— 必须把两个都带上，比出来才看得见。
    if "归属于母公司所有者的净利润" in d.columns:
        out["sina_np_parent"] = pd.to_numeric(d["归属于母公司所有者的净利润"], errors="coerce")
    out["rev_col"] = rev_col
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="抽多少只股票")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    panel = pd.read_pickle(PANEL)
    panel = panel[["code", "period", "rev_l", "np_l", "rev", "np"]].copy()
    codes = sorted(panel["code"].unique())
    step = max(len(codes) // args.n, 1)
    sample = codes[::step][: args.n]
    print(f"抽样 {len(sample)} 只（每 {step} 只取一只，确定性抽样，不用随机种子）：{sample[:6]} …")

    CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        stats = list(pool.map(fetch_one, sample))
    print(f"新浪利润表：新抓 {stats.count('fetched')} · 命中缓存 {stats.count('cached')} · "
          f"失败 {len([s for s in stats if s == 'failed'])}")

    got = [d for d in (load_sina(c) for c in sample) if d is not None and len(d)]
    if not got:
        print("没抓到任何新浪数据 —— 先查网络/函数名，别下结论")
        return 1
    sina = pd.concat(got, ignore_index=True)
    print(f"新浪覆盖 {sina['code'].nunique()} 只 / {len(sina)} 个报告期")

    m = panel[panel["code"].isin(sina["code"].unique())].merge(
        sina, on=["code", "period"], how="inner")
    print(f"可比 (股票, 报告期) 对：{len(m)}；其中两列都有值的："
          f"{(m['rev_l'].notna() & m['sina_rev'].notna()).sum()} / "
          f"{(m['np_l'].notna() & m['sina_np'].notna()).sum()}")
    fallback = sorted(sina.loc[sina["rev_col"] == "营业收入", "code"].unique())
    if fallback:
        print(f"  用「营业收入」代替「营业总收入」的（银行类版式）：{fallback}")

    out: dict = {"n_stocks": int(sina["code"].nunique()), "n_pairs": int(len(m)),
                 "rev_fallback_codes": fallback}
    pairs = [
        ("rev_l", "sina_rev", "营业收入（东财 营业总收入 ↔ 新浪 营业总收入）"),
        ("np_l", "sina_np", "净利润 · **同名错配**：东财「净利润」 ↔ 新浪「净利润」（含少数股东权益）"),
        ("np_l", "sina_np_parent", "净利润 · **概念对齐**：东财「净利润」 ↔ 新浪「归属母公司净利润」"),
        ("np", "sina_np_parent", "旁证：东财**业绩报表**的净利润 ↔ 新浪 归母净利润"),
    ]
    for a, b, name in pairs:
        if b not in m.columns:
            print(f"\n=== {name}：新浪那列不存在，跳过 ===")
            continue
        sub = m[m[a].notna() & m[b].notna()].copy()
        sub = m[m[a].notna() & m[b].notna()].copy()
        rel = (sub[a] - sub[b]).abs() / sub[b].abs().clip(lower=1)
        sub["rel"] = rel
        buckets = {
            "精确相等": float((sub[a] == sub[b]).mean()),
            "<0.01%": float((rel < 1e-4).mean()),
            "<0.1%": float((rel < 1e-3).mean()),
            "<1%": float((rel < 1e-2).mean()),
            "≥1%（要人工看）": float((rel >= 1e-2).mean()),
        }
        out[name] = {"n": int(len(sub)), "buckets": {k: round(v, 4) for k, v in buckets.items()}}
        print(f"\n=== {name}（东财 vs 新浪，n={len(sub)}）===")
        for k, v in buckets.items():
            print(f"  {k:16s} {v:7.2%}")
        worst = sub.nlargest(5, "rel")[["code", "period", a, b, "rel"]]
        print("  差异最大的 5 条：")
        print(worst.to_string(index=False, float_format=lambda x: f"{x:,.0f}" if abs(x) > 100 else f"{x:.6f}"))
        out[name]["worst"] = worst.assign(period=worst["period"].astype(str)).to_dict("records")

    (DATA / "value_crosscheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n读数 ->", DATA / "value_crosscheck.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
