"""把主跑与补跑合成一张最终明细，并打出**四轴读数卡**（判断器那一格）。

主跑 = `st_reason_judge_opencode-glm-5.3.csv`（max_tokens=2000，20 条被截断）；
补跑 = `...-refill.csv`（同一题面，只把 max_tokens 提到 6000 ⇒ **不是改题，是改预算**）。
判据：同一 id 以**补跑**为准（预算更大）；只补到的那部分单独标出，便于对账。
"""
from pathlib import Path

import pandas as pd

A = Path(r"C:\dev\finlab-ml\data\judge_arena")
main = pd.read_csv(A / "st_reason_judge_opencode-glm-5.3.csv", dtype=str).fillna("")
refill_p = A / "st_reason_judge_opencode-glm-5.3-refill.csv"
gold = pd.read_csv(A / "st_reason_gold_labels.v6-120.csv", dtype=str).fillna("")
goldmap = dict(zip(gold["id"], gold["裁定"]))

s = main.copy()
s["来自"] = "主跑"
if refill_p.exists():
    r = pd.read_csv(refill_p, dtype=str).fillna("")
    r["来自"] = "补跑(6000)"
    s = pd.concat([s[~s["id"].isin(r["id"])], r], ignore_index=True)
    print(f"补跑覆盖 {len(r)} 条 · 合成后 {len(s)} 条")
else:
    print("⚠️ 还没有补跑结果，先按主跑统计")


def prim(v) -> str:
    import re
    m = re.match(r"^\s*([ABCDE])", str(v or "").strip().upper())
    return m.group(1) if m else ""


s["primary"] = s["choice"].apply(prim)
s["gold"] = s["id"].map(goldmap)
s["multi"] = s["choice"].astype(str).str.contains("另|\\+")
s["ok"] = s["primary"].eq(s["gold"]) & s["primary"].ne("")
s.to_csv(A / "st_reason_judge_opencode-glm-5.3.MERGED.csv", index=False, encoding="utf-8-sig")

answered = s[s["primary"].ne("")]
blank = s[s["primary"].eq("")]
print(f"\n【覆盖率】答出 {len(answered)}/{len(s)}（{len(answered)/len(s):.1%}）· 空 {len(blank)}")
print(f"【一致率·只算答出的】{answered['ok'].mean():.4f}（{int(answered['ok'].sum())}/{len(answered)}）")
print(f"【一致率·空当作错】{s['ok'].mean():.4f}（{int(s['ok'].sum())}/{len(s)}）")
print(f"【多值标注】{int(answered['multi'].sum())} 条（占答出的 {answered['multi'].mean():.1%}）· 真值里多值条数（`+另`）"
      f"{int(gold['裁定'].astype(str).str.contains('另').sum())}")
ms = pd.to_numeric(s["ms"], errors="coerce")
print(f"【延迟】中位 {ms.median():.0f} ms · p95 {ms.quantile(.95):.0f} ms · 最慢 {ms.max():.0f} ms（答出组中位 "
      f"{pd.to_numeric(answered['ms'], errors='coerce').median():.0f} · 空组中位 {pd.to_numeric(blank['ms'], errors='coerce').median():.0f}）")
print(f"【吞吐/成本】prompt {int(pd.to_numeric(s['ptok'], errors='coerce').sum())} · completion "
      f"{int(pd.to_numeric(s['ctok'], errors='coerce').sum())} tokens（Go 套餐内，无逐次计费字段）")
conf = pd.to_numeric(answered["conf"], errors="coerce").dropna()
if len(conf):
    print(f"【校准】置信度中位 {conf.median():.3f} · 落在不确定带 [0.2,0.8] 的 {((conf >= .2) & (conf <= .8)).mean():.2%}")
print("\n【真值 × 模型（只算答出的）】")
print(pd.crosstab(answered["gold"], answered["primary"]).to_string())
