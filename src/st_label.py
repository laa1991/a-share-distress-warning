"""as-of ST 标签：把「被实施风险警示」的公告变成时间轴上的事件，再对齐到我们的样本。

为什么值得做：现在的标签是「当年年报亏损」，它是个**会计结果**；而业务真正关心的是
**被实施风险警示**（*ST / ST）—— 一个**监管动作**，也是风控名单真正要拦的东西。

判据（标题里的文字游戏很多，逐类分开，别用一个大正则糊过去）：
- 含「风险警示」+「实施」，且**不含**「撤销 / 进展 / 继续 / 提示 / 措施」⇒ **一次新的戴帽事件**；
- 含「撤销」⇒ 不算（那是摘帽或"继续实施"的混合句）；
- 「终止上市风险提示」不在其中（那是退市流程提示）。

标签定义（两个版本，都从决策时刻往后看）：
- `label_st_90`：**年报披露日起 90 天内**被实施风险警示（与「年报亏损」同口径、同一时点解出）
- `label_st_y1`：**决策之后到次年年底**之间被实施（更宽，接近"未来一年内出事"）
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "st_events"

EXCLUDE = ("撤销", "进展", "继续", "提示", "措施", "更正")


def clean(s: str) -> str:
    return re.sub(r"</?em>", "", str(s))


def is_entry(title: str) -> bool:
    t = clean(title)
    if "风险警示" not in t or "实施" not in t:
        return False
    return not any(k in t for k in EXCLUDE)


def load_events() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RAW / "*.csv"))):
        d = pd.read_csv(f, dtype={"代码": str})
        if not len(d):
            continue
        d["标题"] = d["公告标题"].map(clean)
        d["is_entry"] = d["标题"].map(is_entry)
        d["date"] = pd.to_datetime(d["公告时间"].astype(str).str[:10], errors="coerce")
        rows.append(d[["代码", "标题", "date", "is_entry"]])
    e = pd.concat(rows, ignore_index=True)
    e = e[e["is_entry"] & e["date"].notna()].copy()
    e["code"] = e["代码"].str.zfill(6)
    # 同一家公司同一天可能有多条 ⇒ 去重；再按时间排序
    return e.drop_duplicates(subset=["code", "date"]).sort_values(["code", "date"])


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (事件明细, 每家公司最早一次戴帽)"""
    e = load_events()
    first = e.groupby("code", as_index=False)["date"].min().rename(columns={"date": "st_first"})
    return e, first


def main() -> int:
    e, first = build()
    e["year"] = e["date"].dt.year
    print(f"新的戴帽事件 {len(e)} 条 · 涉及 {e['code'].nunique()} 家公司 · "
          f"年份 {e['year'].min()}–{e['year'].max()}")
    print("\n逐年新增戴帽（去重后）：")
    print(e.groupby("year").size().to_string())

    panel = pd.read_pickle(ROOT / "data" / "panel.pkl")
    ann = panel[panel["period"] % 10000 == 1231][["code", "period", "np", "disc_actual"]].copy()
    ann["year"] = ann["period"] // 10000
    ann["label_loss"] = (ann["np"] < 0).astype(int)
    ann = ann.merge(first, on="code", how="left")
    ann["days_to_st"] = (ann["st_first"] - ann["disc_actual"]).dt.days
    ann["label_st_90"] = ((ann["days_to_st"] >= 0) & (ann["days_to_st"] <= 90)).astype(int)
    ann["label_st_y1"] = ((ann["days_to_st"] >= 0) &
                          (ann["st_first"] <= ann["disc_actual"] + pd.Timedelta(days=365))).astype(int)

    t = ann[ann["year"].between(2022, 2025)]
    print(f"\n测试年样本 {len(t)}：")
    print(f"  当年亏损率 {t['label_loss'].mean():.2%}")
    print(f"  年报披露后 90 天内被戴帽 {t['label_st_90'].mean():.2%}（{int(t['label_st_90'].sum())} 条）")
    print(f"  年报披露后一年内被戴帽 {t['label_st_y1'].mean():.2%}（{int(t['label_st_y1'].sum())} 条）")
    both = pd.crosstab(t["label_loss"], t["label_st_90"], margins=True)
    print("\n两套标签的交叉表（列 = 90 天内被戴帽）：")
    print(both.to_string())
    print("\n「真会戴帽」却没被 ST 标签抓到（label_st_90=0）的亏损样本：",
          int(((t["label_loss"] == 1) & (t["label_st_90"] == 0)).sum()))

    (ROOT / "data" / "st_label.json").write_text(
        t[["code", "year", "label_loss", "label_st_90", "label_st_y1", "days_to_st"]]
        .assign(days_to_st=lambda d: d["days_to_st"].astype("Int64"), disc_actual=None)
        .to_json(orient="records", force_ascii=False), encoding="utf-8")
    print("\n读数 -> data/st_label.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
