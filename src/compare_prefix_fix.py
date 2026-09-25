"""修题面前后对照：那条 A↔B 的缝剩下多少。

前 = 旧题面下的读数（glm 主跑+补跑 / kimi 主跑）；后 = `*-prefixfix.csv`（题面修好之后，只跑受影响的 41 条）。
两边的真值都是 v6-120。**只比同一批 41 条**，不跨批比。
"""
import re
from pathlib import Path

import pandas as pd

A = Path(r"C:\dev\finlab-ml\data\judge_arena")
gold = pd.read_csv(A / "st_reason_gold_labels.v6-120.csv", dtype=str).fillna("")
gm = dict(zip(gold["id"], gold["裁定"]))
IDS = [x.strip() for x in (A / "st_reason_prefix_fix_ids.txt").read_text(encoding="utf-8").split() if x.strip()]
SHARED5 = ["002528_2025-04-30", "300125_2024-08-19", "300225_2025-07-02", "600745_2026-04-30", "603557_2024-04-27"]


def prim(v: str) -> str:
    m = re.match(r"^\s*([ABCDE])", str(v or "").strip().upper())
    return m.group(1) if m else ""


def load_pre(model: str) -> pd.DataFrame:
    d = pd.read_csv(A / f"st_reason_judge_opencode-{model}.csv", dtype=str).fillna("")
    rf = A / f"st_reason_judge_opencode-{model}-refill.csv"
    if rf.exists():
        r = pd.read_csv(rf, dtype=str).fillna("")
        d = pd.concat([d[~d["id"].isin(r["id"])], r], ignore_index=True)
    return d


print(f"{'模型':9s} {'批':22s} {'答出':>6s} {'对':>4s} {'错(答了)':>8s} {'空':>4s}")
out = {}
for model in ("glm-5.3", "kimi-k3"):
    pre = load_pre(model)
    pf = A / f"st_reason_judge_opencode-{model}-prefixfix.csv"
    if not pf.exists():
        print(f"{model:9s} （改题面后的读数还没落，跳过）")
        continue
    post = pd.read_csv(pf, dtype=str).fillna("")
    for tag, d in (("改题面前", pre), ("改题面后", post)):
        s = d[d["id"].isin(IDS)].copy()
        s["p"] = s["choice"].apply(prim)
        s["g"] = s["id"].map(gm)
        ans = s[s["p"].ne("")]
        wrong = int((ans["p"] != ans["g"]).sum())
        print(f"{model:9s} {tag:22s} {len(ans):6d} {len(ans) - wrong:4d} {wrong:8d} {len(s) - len(ans):4d}")
        out[(model, tag)] = s.set_index("id")

print("\n=== 那 5 条『两家都错』：改题面前后逐条 ===")
for cid in SHARED5:
    line = f"  {cid} gold={gm[cid]}"
    for model in ("glm-5.3", "kimi-k3"):
        a = out.get((model, "改题面前"))
        b = out.get((model, "改题面后"))
        av = prim(a.loc[cid]["choice"]) if a is not None and cid in a.index else "?"
        bv = prim(b.loc[cid]["choice"]) if b is not None and cid in b.index else "?"
        line += f" | {model[:4]}: {av or '空'}→{bv or '空'}"
    print(line)

print("\n=== 改题面后仍然错/空的，逐条列出（这就是残余）===")
for model in ("glm-5.3", "kimi-k3"):
    if (model, "改题面后") not in out:
        continue
    s = out[(model, "改题面后")]
    bad = s[(s["p"].eq("")) | (s["p"] != s["g"])]
    print(f"\n[{model}] 残余 {len(bad)} 条：")
    for _, r in bad.head(12).iterrows():
        print(f"   {r.name} gold={r['g']} 答={r['p'] or '空'} conf={r.get('conf')}")
