"""报表外事件（诉讼 / 担保）→ 特征。**唯一一份口径**，探针、消融、第二条通道都从这里取。

PIT 纪律：只看**上一整年** + **当年 Q1**（公告统计区间是日历窗口，所以这两段在 4 月决策时都已公开）。
另给一套**更干净**的版本「近两季」= 上年 Q4 + 当年 Q1（离决策时刻更近，噪声更少）。
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

NUM = {"诉讼次数": "sue_n", "诉讼金额": "sue_amt", "担保笔数": "gua_n", "担保金额": "gua_amt",
       "归属于母公司所有者权益": "gua_eq", "担保金融占净资产比例": "gua_ratio"}

COLS = ["code", "year", "sue_prev", "sue_q1", "sue_any", "sue_prev_q4", "sue_recent", "sue_any_recent",
        "gua_n_prev", "gua_ratio_prev", "gua_heavy", "gua_ratio_recent", "gua_heavy_recent"]


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
    """返回 (code, year) → 事件特征（year = **决策年**：特征只用 Y−1 年与 Y 年 Q1）。"""
    sue, gua = _load("lawsuit"), _load("guarantee")
    if not len(sue) or not len(gua):
        raise FileNotFoundError("先跑 scripts/fetch_cg_events.py")
    sue["sue_amt"] = pd.to_numeric(sue.get("sue_amt"), errors="coerce")
    gua["gua_ratio"] = pd.to_numeric(gua.get("gua_ratio"), errors="coerce")

    # 季度面板：诉讼次数 / 担保占比
    sq = sue.pivot_table(index=["code", "year"], columns="q", values="sue_n", aggfunc="sum")
    gq = gua.pivot_table(index=["code", "year"], columns="q", values="gua_ratio", aggfunc="max")
    for c in (1, 2, 3, 4):
        if c not in sq.columns:
            sq[c] = 0.0
        if c not in gq.columns:
            gq[c] = float("nan")
    idx = pd.concat([sq[[]], gq[[]]]).index.drop_duplicates()
    f = pd.DataFrame(index=idx).sort_index()
    for c in (1, 2, 3, 4):
        f[f"sue_q{c}"] = sq[c].reindex(f.index).fillna(0.0)
        f[f"gua_q{c}"] = gq[c].reindex(f.index)
    f = f.reset_index().sort_values(["code", "year"])

    # 上一整年 / 上年 Q4（按公司往后挪一年）
    f["sue_prev"] = f.groupby("code")["sue_q1"].shift(1).fillna(0) + f.groupby("code")["sue_q2"].shift(1).fillna(0) \
        + f.groupby("code")["sue_q3"].shift(1).fillna(0) + f.groupby("code")["sue_q4"].shift(1).fillna(0)
    f["gua_ratio_prev"] = f.groupby("code")[["gua_q1", "gua_q2", "gua_q3", "gua_q4"]].shift(1).max(axis=1)
    f["sue_prev_q4"] = f.groupby("code")["sue_q4"].shift(1).fillna(0)
    f["gua_prev_q4"] = f.groupby("code")["gua_q4"].shift(1)

    f["sue_q1"] = f["sue_q1"].astype(float)
    f["sue_any"] = ((f["sue_prev"] > 0) | (f["sue_q1"] > 0)).astype(float)
    f["sue_recent"] = f["sue_prev_q4"] + f["sue_q1"]
    f["sue_any_recent"] = (f["sue_recent"] > 0).astype(float)
    f["gua_heavy"] = (f["gua_ratio_prev"].fillna(-1) >= 50).astype(float)
    f["gua_ratio_recent"] = f[["gua_prev_q4", "gua_q1"]].max(axis=1)
    f["gua_heavy_recent"] = (f["gua_ratio_recent"].fillna(-1) >= 50).astype(float)

    # 担保笔数（上一整年）
    gua["gua_n"] = pd.to_numeric(gua["gua_n"], errors="coerce").fillna(0.0)
    gn = gua.groupby(["code", "year"], as_index=False)["gua_n"].sum().sort_values(["code", "year"])
    gn["gua_n_prev"] = gn.groupby("code")["gua_n"].shift(1).fillna(0.0)
    f = f.merge(gn[["code", "year", "gua_n_prev"]], on=["code", "year"], how="left")
    f["gua_n_prev"] = f["gua_n_prev"].fillna(0.0)
    return f[COLS]


EVENT_FEATURES = ["sue_prev", "sue_q1", "sue_any", "gua_n_prev", "gua_ratio_prev", "gua_heavy"]

CHANNELS = {
    "诉讼（上年+Y Q1）": "sue_any",
    "担保≥净资产50%": "gua_heavy",
    "并集 = 诉讼∨担保": None,           # 特殊处理
    "并集·近两季": None,
}

if __name__ == "__main__":
    d = build()
    print(f"{len(d)} 行 · 列 {[c for c in d.columns if c not in ('code', 'year')]}")
    print(d[d["sue_any"] == 1].head().to_string(index=False))
