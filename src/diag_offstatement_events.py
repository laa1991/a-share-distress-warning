"""探针：报表外事件（诉讼 / 担保）值不值得开一格？

只回答三个问题，不下结论：
1. **覆盖**：我们的样本里，有多少能在决策时刻之前拿到这类事件？（上一整年 + 当年 Q1）
2. **先看粗关系**：有事件的公司，当年亏损率是不是更高？
3. **★ 关键格**：**模型没给高分的那一批里**（① 一季报分位 < 0.9），有事件 vs 没事件，亏损率差多少？
   —— 只有这一格差得开，事件信号才有"增量"，否则它只是在重复模型已经看见的东西。

口径：
- 事件窗口按**公告统计区间**（季度）对齐 ⇒ 「4 月决策时用上一整年 + 当年 Q1」是干净的，不偷看。
- 「有事件」= 那段时间里有 ≥1 条诉讼仲裁公告（或担保笔数 ≥1）。
- 担保另外看一个连续量：担保金额 / 归属母公司净资产。

用法: python diag_offstatement_events.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"


def load_kind(kind: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(RAW / f"cg_{kind}" / "*.csv"))):
        tag = Path(f).stem                      # 形如 20241
        d = pd.read_csv(f, dtype={"证券代码": str})
        if not len(d):
            continue
        d["code"] = d["证券代码"].str.zfill(6)
        d["year"] = int(tag[:4])
        d["q"] = int(tag[4:])
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    panel = pd.read_pickle(DATA / "panel.pkl")
    scores = pd.read_csv(DATA / "leadtime_scores.csv", dtype={"code": str})
    s1 = scores[scores["stage"] == "① 一季报披露"].copy()
    s1["pct"] = s1.groupby("year")["score"].rank(pct=True)

    sue = load_kind("lawsuit")
    gua = load_kind("guarantee")
    print(f"事件表：诉讼 {len(sue)} 行 / 担保 {len(gua)} 行（季度窗口 "
          f"{sue['year'].min()}Q{sue['q'].min()}–{sue['year'].max()}Q{sue['q'].max()}）")

    num = {"诉讼次数": "sue_n", "诉讼金额": "sue_amt", "担保笔数": "gua_n", "担保金额": "gua_amt",
           "归属于母公司所有者权益": "gua_eq", "担保金融占净资产比例": "gua_ratio"}
    sue = sue.rename(columns={k: v for k, v in num.items() if k in sue.columns})
    gua = gua.rename(columns={k: v for k, v in num.items() if k in gua.columns})
    sue["sue_amt"] = pd.to_numeric(sue.get("sue_amt"), errors="coerce")
    gua["gua_ratio"] = pd.to_numeric(gua.get("gua_ratio"), errors="coerce")

    # 上一整年（Y−1）与当年 Q1（Y Q1）
    sue_y = sue.groupby(["code", "year"], as_index=False).agg(sue_n=("sue_n", "sum"), sue_amt=("sue_amt", "sum"))
    gua_y = gua.groupby(["code", "year"], as_index=False).agg(gua_n=("gua_n", "sum"), gua_ratio=("gua_ratio", "max"))
    sue_q1 = sue[sue["q"] == 1].groupby(["code", "year"], as_index=False).agg(sue_n_q1=("sue_n", "sum"))
    gua_q1 = gua[gua["q"] == 1].groupby(["code", "year"], as_index=False).agg(gua_ratio_q1=("gua_ratio", "max"))

    base = panel[["code", "period", "q"]].copy()
    base["year"] = base["period"] // 10000
    lab = panel[panel["period"] % 10000 == 1231][["code", "period", "np"]].copy()
    lab["year"] = lab["period"] // 10000
    lab["label"] = (lab["np"] < 0).astype(float)

    s = s1[["code", "year", "score", "pct", "label"]]
    s = s.merge(sue_y.rename(columns={"year": "y"}), left_on=["code", "year"], right_on=["code", "y"], how="left") \
         .drop(columns=["y"], errors="ignore")
    s["sue_prev"] = s["sue_n"]
    s = s.merge(gua_y.rename(columns={"year": "y", "gua_n": "gua_n_prev", "gua_ratio": "gua_ratio_prev"}),
                left_on=["code", "year"], right_on=["code", "y"], how="left").drop(columns=["y"], errors="ignore")
    # 上一整年 = 把上面那张表往前挪一年
    for col in ("sue_prev", "gua_n_prev", "gua_ratio_prev"):
        s[col] = s.groupby("code")[col].shift(1)
    s = s.merge(sue_q1, on=["code", "year"], how="left").merge(gua_q1, on=["code", "year"], how="left")
    s[["sue_prev", "sue_n_q1", "gua_n_prev"]] = s[["sue_prev", "sue_n_q1", "gua_n_prev"]].fillna(0)
    s["has_sue"] = ((s["sue_prev"] > 0) | (s["sue_n_q1"] > 0)).astype(int)
    s["has_gua"] = ((s["gua_n_prev"] > 0) | (s["gua_ratio_q1"].fillna(0) > 0)).astype(int)
    s["gua_heavy"] = (s["gua_ratio_prev"].fillna(0) >= 50).astype(int)

    print(f"\n样本 {len(s)}（测试年 2022–2025）· 亏损率 {s['label'].mean():.2%}")
    print("\n【1. 覆盖】")
    for col, name in (("has_sue", "≥1 条诉讼公告（上年或当年Q1）"), ("has_gua", "≥1 笔对外担保"),
                      ("gua_heavy", "担保占净资产 ≥50%（上年）")):
        print(f"  {name:28s} 覆盖 {s[col].mean():6.2%}（{int(s[col].sum())} 条）")

    print("\n【2. 粗关系：有事件 vs 没事件，当年亏损率】")
    for col, name in (("has_sue", "诉讼"), ("has_gua", "担保"), ("gua_heavy", "担保≥净资产50%")):
        g = s.groupby(col)["label"].agg(["mean", "size"])
        a = g.loc[1, "mean"] if 1 in g.index else float("nan")
        b = g.loc[0, "mean"] if 0 in g.index else float("nan")
        print(f"  {name:16s} 有 {a:7.2%}（n={int(g.loc[1,'size']) if 1 in g.index else 0}） · "
              f"无 {b:7.2%}（n={int(g.loc[0,'size']) if 0 in g.index else 0}） · 差 {a-b:+.2%}")

    print("\n【3. ★ 关键格：模型**没**给高分的那一批（① 一季报分位 < 0.9）里，事件还分得开吗？】")
    quiet = s[s["pct"] < 0.9]
    print(f"  这一批 {len(quiet)} 条，亏损率 {quiet['label'].mean():.2%}"
          f"（全部样本的亏损率 {s['label'].mean():.2%}）")
    for col, name in (("has_sue", "诉讼"), ("has_gua", "担保"), ("gua_heavy", "担保≥净资产50%")):
        g = quiet.groupby(col)["label"].agg(["mean", "size"])
        if 1 in g.index and 0 in g.index:
            a, b = g.loc[1, "mean"], g.loc[0, "mean"]
            print(f"  {name:16s} 有 {a:7.2%}（n={int(g.loc[1,'size'])}） · 无 {b:7.2%}（n={int(g.loc[0,'size'])}）"
                  f" · **差 {a-b:+.2%}** ·  lift {a/b:.2f}×")

    print("\n【4. 同一格，再看模型自己给的分：事件是「模型已经看见」还是「额外的」】")
    for col, name in (("has_sue", "诉讼"), ("gua_heavy", "担保≥净资产50%")):
        g = quiet.groupby(col)["score"].agg(["mean", "median", "size"])
        print(f"  {name:16s} 模型平均分 有事件 {g.loc[1,'mean']:.4f} / 无 {g.loc[0,'mean']:.4f}"
              f"（差 {g.loc[1,'mean']-g.loc[0,'mean']:+.4f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
