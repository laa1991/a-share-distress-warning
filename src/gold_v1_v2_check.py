"""v1 vs v2 裁定对账 + 核「truth_rule 到底是哪一列」（同名不同物的嫌疑）。

要回答两件事：
① **复核册质量**：同一批 60 条，摘录改好之后裁定变了多少（v1 vs v2 交叉一致率）；
② **arena 的真值列是不是取错了**：`st_reasons.csv` 的「类别」如果来自**标题关键词**规则，
   而发布用的分布来自**正文锚点**规则，那这两列不是同一个口径 —— 拿它对账会得出假的"不一致"。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "data" / "judge_arena"


def main() -> int:
    v1 = pd.read_csv(A / "st_reason_gold_labels.v1-reviewed.csv", dtype=str).fillna("")
    v2 = pd.read_csv(A / "st_reason_gold_labels.v2.csv", dtype=str).fillna("")
    m = v1.merge(v2, on="id", suffixes=("_v1", "_v2"))
    print(f"① 两版裁定对账（n={len(m)}）")
    print(f"   **交叉一致率 {(m['裁定_v1'] == m['裁定_v2']).mean():.1%}**"
          f"（{int((m['裁定_v1'] == m['裁定_v2']).sum())}/{len(m)}）")
    r1 = m["裁定_v1"].isin(list("ABCD"))
    r2 = m["裁定_v2"].isin(list("ABCD"))
    print(f"   两版都可判 {int((r1 & r2).sum())} 条 · v1 判不了 {int((~r1).sum())} · v2 判不了 {int((~r2).sum())}")
    both = m[r1 & r2]
    print(f"   只看两版都可判的：一致 {(both['裁定_v1'] == both['裁定_v2']).mean():.1%}"
          f"（{int((both['裁定_v1'] == both['裁定_v2']).sum())}/{len(both)}）")
    ch = m[(m["裁定_v1"] != m["裁定_v2"])]
    print(f"\n   变化的 {len(ch)} 条：")
    for _, r in ch.iterrows():
        print(f"     {r['id']}  {r['裁定_v1'] or '?'} → {r['裁定_v2'] or '?'}   v2理由：{str(r.get('理由', ''))[:26]}")

    print("\n② 核 st_reasons.csv 的列与来源")
    s = pd.read_csv(ROOT / "data" / "st_reasons.csv", dtype=str).fillna("")
    print("   列：", list(s.columns))
    if "类别" in s.columns:
        print("   类别取值：", s["类别"].value_counts().to_dict())
    # 与 arena 那 60 条比：truth_rule 是不是就是这一列
    truth = pd.read_csv(A / "st_reason_gold_labels.csv", dtype=str).fillna("")
    key = truth["id"].str.split("_").str[0]
    sub = s.drop_duplicates("code").set_index("code")
    same = sum(1 for k, t in zip(key, truth["truth_rule"]) if k in sub.index and sub.loc[k, "类别"] == t)
    print(f"   arena 的 truth_rule 与 st_reasons.csv 的「类别」逐条相同：{same}/{len(truth)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
