"""把 data/raw 里的四张东财批量表 + 巨潮披露表拼成一张「(股票代码, 报告期)」面板。

口径要点（完整版见 docs/口径.md）：
- 键 = (code, period)，period 形如 20200331。
- **PIT 锚点 = 巨潮 `实际披露`**（`disc_actual`）；东财四表的 `公告日期` 只作**源对账**用 ——
  实测它们对有「上年同期」栏的报表（利润表/现金流量表/业绩报表）系统性晚一年（见 audit 输出）。
- 板块过滤：只留 0/3/6 开头（深主板/创业板/沪主板+科创板）。
- 衍生特征：比率（与规模无关）+ 上一期滞后 + 差分。

用法:
  python build_panel.py --audit     # 只打印体检，不落盘
  python build_panel.py             # 拼装并落盘 data/panel.pkl
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "panel.pkl"

RENAME = {
    "yjbb": {
        "股票代码": "code", "每股收益": "eps", "营业总收入-营业总收入": "rev",
        "营业总收入-同比增长": "rev_yoy", "营业总收入-季度环比增长": "rev_qoq",
        "净利润-净利润": "np", "净利润-同比增长": "np_yoy", "净利润-季度环比增长": "np_qoq",
        "每股净资产": "bps", "净资产收益率": "roe", "每股经营现金流量": "ocfps",
        "销售毛利率": "gross_margin", "所处行业": "industry", "最新公告日期": "ann",
    },
    "zcfz": {
        "股票代码": "code", "资产-货币资金": "cash", "资产-应收账款": "ar", "资产-存货": "inventory",
        "资产-总资产": "total_assets", "资产-总资产同比": "total_assets_yoy",
        "负债-应付账款": "ap", "负债-预收账款": "advance_receipts", "负债-总负债": "total_liab",
        "负债-总负债同比": "total_liab_yoy", "资产负债率": "debt_ratio", "股东权益合计": "equity",
        "公告日期": "ann",
    },
    "lrb": {
        "股票代码": "code", "净利润": "np_l", "净利润同比": "np_yoy_l",
        "营业总收入": "rev_l", "营业总收入同比": "rev_yoy_l",
        "营业总支出-营业支出": "op_expense", "营业总支出-销售费用": "sell_exp",
        "营业总支出-管理费用": "admin_exp", "营业总支出-财务费用": "fin_exp",
        "营业总支出-营业总支出": "op_expense_total", "营业利润": "op_profit",
        "利润总额": "total_profit", "公告日期": "ann",
    },
    "xjll": {
        "股票代码": "code", "净现金流-净现金流": "net_cf", "净现金流-同比增长": "net_cf_yoy",
        "经营性现金流-现金流量净额": "ocf", "经营性现金流-净现金流占比": "ocf_ratio",
        "投资性现金流-现金流量净额": "icf", "投资性现金流-净现金流占比": "icf_ratio",
        "融资性现金流-现金流量净额": "fcf", "融资性现金流-净现金流占比": "fcf_ratio",
        "公告日期": "ann",
    },
}
TABLES = tuple(RENAME)
KEEP_PREFIX = ("0", "3", "6")
Q_END = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}


def load_table(name: str) -> tuple[pd.DataFrame, dict]:
    frames, stats = [], {"dup_rows": 0, "periods": 0}
    for f in sorted(glob.glob(str(RAW / name / "*.csv"))):
        d = pd.read_csv(f, dtype={"股票代码": str})
        d["period"] = int(Path(f).stem)
        d = d.rename(columns=RENAME[name])
        d = d[[c for c in RENAME[name].values() if c in d.columns] + ["period"]]
        d["code"] = d["code"].str.zfill(6)
        d = d[d["code"].str[0].isin(KEEP_PREFIX)]
        d["ann"] = pd.to_datetime(d["ann"], errors="coerce")
        dup = int(d["code"].duplicated().sum())
        if dup:
            stats["dup_rows"] += dup
            d = d.sort_values("ann").drop_duplicates("code", keep="last")
        frames.append(d)
        stats["periods"] += 1
    return pd.concat(frames, ignore_index=True), stats


def load_disclosure() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(RAW / "disclosure" / "*.csv"))):
        stem = Path(f).stem                       # 201601 / 202312 ...
        year, q = int(stem[:4]), int(stem[4:])
        d = pd.read_csv(f, dtype={"股票代码": str})
        d["code"] = d["股票代码"].str.zfill(6)
        d["period"] = int(f"{year}{Q_END[q]}")
        d["disc_first"] = pd.to_datetime(d["首次预约"], errors="coerce")
        d["disc_actual"] = pd.to_datetime(d["实际披露"], errors="coerce")
        keep = d[d["code"].str[0].isin(KEEP_PREFIX)]
        frames.append(keep[["code", "period", "disc_first", "disc_actual"]])
    return pd.concat(frames, ignore_index=True)


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.where(b.abs() > 1e-9)
    return out.replace([np.inf, -np.inf], np.nan)


def build(audit: bool) -> int:
    panel = None
    for name in TABLES:
        d, stats = load_table(name)
        print(f"[{name}] 期数={stats['periods']} 行数={len(d)} 同键多行={stats['dup_rows']}")
        d = d.rename(columns={"ann": f"ann_{name}"})
        panel = d if panel is None else panel.merge(d, on=["code", "period"], how="outer")

    disc = load_disclosure()
    print(f"[disclosure] 期数={disc['period'].nunique()} 行数={len(disc)} "
          f"实际披露非空={disc['disc_actual'].notna().mean():.1%}")
    panel = panel.merge(disc, on=["code", "period"], how="left")

    ann_cols = [c for c in panel.columns if c.startswith("ann_")]
    panel["ann_max"] = panel[ann_cols].max(axis=1)
    panel["n_tables"] = panel[ann_cols].notna().sum(axis=1)

    print("\n=== 合并后 ===")
    print("行数", len(panel), "| 股票数", panel["code"].nunique(), "| 期数", panel["period"].nunique())
    print("四表齐全的行:", int((panel["n_tables"] == 4).sum()), f"({(panel['n_tables'] == 4).mean():.1%})")
    print("有巨潮实际披露日的行:", int(panel["disc_actual"].notna().sum()),
          f"({panel['disc_actual'].notna().mean():.1%})")

    # ---- 源对账：东财四表的「公告日期」 vs 巨潮「实际披露」 ----
    print("\n=== 源对账（东财 公告日期 − 巨潮 实际披露，天） ===")
    print(f"{'表':8s} {'中位差':>8s} {'|差|≤5天占比':>14s} {'中位差(仅Q4)':>14s}")
    for c in ann_cols:
        delta = (panel[c] - panel["disc_actual"]).dt.days
        q4 = panel["period"] % 10000 == 1231
        print(f"{c:8s} {delta.median():8.0f} {(delta.abs() <= 5).mean():13.1%} {delta[q4].median():14.0f}")

    # ---- 自检：利润表营收是年初至今累计 ⇒ 同年内必须单调不减 ----
    yr = panel["period"] // 10000
    piv = panel.assign(year=yr).pivot_table(index=["code", "year"], columns="period", values="rev_l")
    seq = piv.reindex(columns=sorted(piv.columns))
    ok = 0
    bad = 0
    for _, row in seq.iterrows():
        v = row.dropna().values
        if len(v) >= 2:
            if np.all(np.diff(v) >= -1e-6):
                ok += 1
            else:
                bad += 1
    print(f"\n=== 自检：同年营收(YTD)单调不减 ===\n通过 {ok} / 违反 {bad} ({bad / max(ok + bad, 1):.2%})")

    # ---- 衍生特征 ----
    for col in ("np", "rev", "total_assets", "total_liab", "equity", "ocf", "ar", "inventory", "cash", "ap"):
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel["np_margin"] = safe_div(panel["np"], panel["rev"])
    panel["ocf_to_assets"] = safe_div(panel["ocf"], panel["total_assets"])
    panel["ocf_to_rev"] = safe_div(panel["ocf"], panel["rev"])
    panel["ar_ratio"] = safe_div(panel["ar"], panel["total_assets"])
    panel["inv_ratio"] = safe_div(panel["inventory"], panel["total_assets"])
    panel["cash_ratio"] = safe_div(panel["cash"], panel["total_assets"])
    panel["equity_ratio"] = safe_div(panel["equity"], panel["total_assets"])
    panel["ap_ratio"] = safe_div(panel["ap"], panel["total_assets"])
    panel["sell_exp_ratio"] = safe_div(pd.to_numeric(panel["sell_exp"], errors="coerce"), panel["rev"])
    panel["admin_exp_ratio"] = safe_div(pd.to_numeric(panel["admin_exp"], errors="coerce"), panel["rev"])
    panel["fin_exp_ratio"] = safe_div(pd.to_numeric(panel["fin_exp"], errors="coerce"), panel["rev"])
    panel["op_profit_margin"] = safe_div(pd.to_numeric(panel["op_profit"], errors="coerce"), panel["rev"])
    panel["log_assets"] = np.log1p(panel["total_assets"].clip(lower=0))
    panel["log_rev"] = np.log1p(panel["rev"].clip(lower=0))

    panel["p_idx"] = (panel["period"] // 10000) * 4 + (panel["period"] % 10000) // 300
    panel = panel.sort_values(["code", "p_idx"]).reset_index(drop=True)
    for col in ("roe", "debt_ratio", "np_margin", "ocf_to_assets", "rev_yoy", "ar_ratio", "inv_ratio"):
        panel[f"{col}_lag1"] = panel.groupby("code")[col].shift(1)
    panel["d_roe"] = panel["roe"] - panel["roe_lag1"]
    panel["d_debt_ratio"] = panel["debt_ratio"] - panel["debt_ratio_lag1"]
    panel["d_np_margin"] = panel["np_margin"] - panel["np_margin_lag1"]
    panel["q"] = (panel["period"] % 10000) // 300

    print("\n面板:", panel.shape)
    key = ["eps", "rev", "np", "roe", "debt_ratio", "ocf", "total_assets", "np_margin", "industry"]
    print("关键列缺失率:", {k: round(float(panel[k].isna().mean()), 3) for k in key})
    if audit:
        return 0
    panel.to_pickle(OUT)
    print(f"\n已落盘 -> {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    raise SystemExit(build(ap.parse_args().audit))
