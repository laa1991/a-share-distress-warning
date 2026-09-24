"""报表外事件（诉讼 / 担保）→ 特征。**唯一一份口径**，探针与消融都从这里取。

PIT 纪律：只看**上一整年** + **当年 Q1**（公告统计区间是日历窗口，所以这两段在 4 月决策时都已公开）。
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

NUM = {"诉讼次数": "sue_n", "诉讼金额": "sue_amt", "担保笔数": "gua_n", "担保金额": "gua_amt",
       "归属于母公司所有者权益": "gua_eq", "担保金融占净资产比例": "gua_ratio"}


def _load(kind: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(RAW / f"cg_{kind}" / "*.csv"))):
        tag = Path(f).stem                      # 形如 20241 = 2024Q1
        d = pd.read_csv(f, dtype={"证券代码": str})
        if not len(d):
            continue
        d["code"] = d["证券代码"].str.zfill(6)
        d["year"] = int(tag[:4])
        d["q"] = int(tag[4:])
        frames.append(d.rename(columns={k: v for k, v in NUM.items() if k in d.columns}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build() -> pd.DataFrame:
    """返回 (code, year) → 事件特征（该年是**决策年**，特征只用上年 + 当年 Q1）。"""
    sue, gua = _load("lawsuit"), _load("guarantee")
    if not len(sue) or not len(gua):
        raise FileNotFoundError("先跑 scripts/fetch_cg_events.py")

    sue["sue_amt"] = pd.to_numeric(sue.get("sue_amt"), errors="coerce")
    gua["gua_ratio"] = pd.to_numeric(gua.get("gua_ratio"), errors="coerce")

    sy = sue.groupby(["code", "year"], as_index=False).agg(sue_n_y=("sue_n", "sum"))
    sq = sue[sue["q"] == 1].groupby(["code", "year"], as_index=False).agg(sue_n_q1=("sue_n", "sum"))
    gy = gua.groupby(["code", "year"], as_index=False).agg(gua_n_y=("gua_n", "sum"), gua_ratio_y=("gua_ratio", "max"))

    base = pd.concat([sy[["code", "year"]], gy[["code", "year"]], sq[["code", "year"]]]).drop_duplicates()
    f = base.merge(sy, on=["code", "year"], how="left").merge(sq, on=["code", "year"], how="left") \
            .merge(gy, on=["code", "year"], how="left")
    f = f.sort_values(["code", "year"])
    g = f.groupby("code")
    # 决策年 Y 用「Y−1 一整年」
    f["sue_prev"] = g["sue_n_y"].shift(1)
    f["gua_n_prev"] = g["gua_n_y"].shift(1)
    f["gua_ratio_prev"] = g["gua_ratio_y"].shift(1)
    f["sue_q1"] = f["sue_n_q1"]
    for c in ("sue_prev", "gua_n_prev", "sue_q1"):
        f[c] = f[c].fillna(0.0)
    f["sue_any"] = ((f["sue_prev"] > 0) | (f["sue_q1"] > 0)).astype(float)
    f["gua_heavy"] = (f["gua_ratio_prev"].fillna(-1) >= 50).astype(float)
    return f[["code", "year", "sue_prev", "sue_q1", "sue_any", "gua_n_prev", "gua_ratio_prev", "gua_heavy"]]


EVENT_FEATURES = ["sue_prev", "sue_q1", "sue_any", "gua_n_prev", "gua_ratio_prev", "gua_heavy"]

if __name__ == "__main__":
    d = build()
    print(f"{len(d)} 行 · 列 {list(d.columns)}")
    print(d.head().to_string(index=False))
