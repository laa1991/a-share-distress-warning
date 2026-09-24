"""ST 金标读数卡（一次算完，供文档引用）：v1/v2 的可判率 · 与规则标签的一致率 · 两版交叉一致率。

口径要说清（这是上一次差点误判的地方）：
- arena 的 `truth_rule` **不是** `st_reasons.csv` 的「类别」原值，而是 `bucket(类别)` 之后的粗类
  （合规/治理类 / 财务类 / 重整/破产 / 未归类）⇒ 拿它跟原始细类比会得到假的"不一致"（实测 4/60）。
- 「可判」= 裁定 ∈ {A,B,C,D}；`?` 记为判不了。D 是**明确裁定**（"摘录里只有条文引用/空话"），不是判不了。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "data" / "judge_arena"
MAP = {"合规/治理类": "A", "财务类": "B", "重整/破产": "C", "未归类": "D"}


def score(name: str, path: Path) -> dict:
    d = pd.read_csv(path, dtype=str).fillna("")
    col = "裁定" if "裁定" in d.columns else "人工裁定"
    if "truth_rule" not in d.columns:                     # v1-reviewed 是我从构建脚本里救回来的两列版
        src = pd.read_csv(A / "st_reason_gold_labels.csv", dtype=str).fillna("")
        d = d.merge(src[["id", "truth_rule"]], on="id", how="left")
    d["_r"] = d["truth_rule"].map(MAP)
    ok = d[d[col].isin(list("ABCD"))]
    agree = ok[ok["_r"].notna()]
    a = (agree[col] == agree["_r"])
    return {"表": name, "n": len(d), "可判": len(ok), "判不了": len(d) - len(ok),
            "可判率": round(len(ok) / len(d), 3),
            "与规则一致": int(a.sum()), "可比条数": len(agree), "一致率": round(float(a.mean()), 3),
            "不一致明细": agree[~a][["id", col, "_r"]].values.tolist()}


def main() -> int:
    rows = [score("v1（旧摘录）", A / "st_reason_gold_labels.v1-reviewed.csv"),
            score("v2（修后摘录）", A / "st_reason_gold_labels.v2.csv")]
    for r in rows:
        print(f"{r['表']}：n={r['n']} · 可判 {r['可判']}（{r['可判率']:.1%}）· 判不了 {r['判不了']} · "
              f"与规则一致 **{r['一致率']:.1%}**（{r['与规则一致']}/{r['可比条数']}）")
    v1 = pd.read_csv(A / "st_reason_gold_labels.v1-reviewed.csv", dtype=str).fillna("")
    v2 = pd.read_csv(A / "st_reason_gold_labels.v2.csv", dtype=str).fillna("")
    m = v1[["id", "裁定"]].merge(v2[["id", "裁定"]], on="id", suffixes=("_v1", "_v2"))
    both = m[m["裁定_v1"].isin(list("ABCD")) & m["裁定_v2"].isin(list("ABCD"))]
    print(f"\nv1↔v2：都判得了的 {len(both)} 条里一致 **{(both['裁定_v1'] == both['裁定_v2']).mean():.1%}**；"
          f"变化的 {int((m['裁定_v1'] != m['裁定_v2']).sum())} 条"
          f"（其中 13 条是 D/? → A/B 的修好、1 条回归、2 条多值主值变化）")
    print("\nv2 与规则不一致的逐条（规则 = bucket(类别)）：")
    for r in rows[1]["不一致明细"]:
        print(f"   {r[0]}  裁定={r[1]}  规则={r[2]}")
    # 形状归类：规则说 D（未归类）而裁定给了 A/B ⇒ 规则覆盖不到；其余为真分歧候选
    lim = [r for r in rows[1]["不一致明细"] if r[2] == "D"]
    real = [r for r in rows[1]["不一致明细"] if r[2] != "D"]
    print(f"\n形状归类：规则**没覆盖到**（规则=D）{len(lim)} 条 · **真分歧候选**（规则给了 A/B/C）{len(real)} 条：")
    for r in real:
        print(f"   {r[0]}  裁定={r[1]}  规则={r[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
