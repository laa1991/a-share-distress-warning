"""金标对账：把人工裁定与**修正后**的规则标签比一次（并单列"判不了"）。

用法: python gold_vs_rule.py [task]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARENA = ROOT / "data" / "judge_arena"


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "rev_dir"
    lab = pd.read_csv(ARENA / f"{task}_gold_labels.csv", dtype={"id": str})
    # 刷新规则标签（分类器可能刚修过）
    fresh = pd.read_csv(ROOT / "data" / "yjyg_rev_direction.csv", dtype={"code": str})
    fresh["id"] = fresh["code"].str.zfill(6) + "_" + fresh["date"].astype(str)
    m = fresh.set_index("id")["方向"].to_dict()
    lab["truth_rule_v2"] = lab["id"].map(m)
    lab.to_csv(ARENA / f"{task}_gold_labels.csv", index=False, encoding="utf-8-sig")

    done = lab[lab["人工裁定"].notna() & (lab["人工裁定"] != "") & (lab["人工裁定"] != "?")]
    undec = lab[lab["人工裁定"] == "?"]
    both = done[done["truth_rule_v2"].notna()]
    print(f"{task}：已裁定 {len(lab[lab['人工裁定'].notna()])} 条 · 其中**判不了** {len(undec)} 条 · "
          f"规则这边仍抽到 {len(both)} 条")
    if len(both):
        a = (both["人工裁定"] == both["truth_rule_v2"])
        print(f"**一致率 {a.mean():.1%}**（{int(a.sum())}/{len(both)}）")
        bad = both[~a]
        for _, r in bad.iterrows():
            print(f"   不一致 {r['id']}：人工={r['人工裁定']} 规则={r['truth_rule_v2']}")
        print("\n混淆（人工 → 规则）：")
        print(pd.crosstab(both["人工裁定"], both["truth_rule_v2"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
