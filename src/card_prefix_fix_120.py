"""修好题面之后的四轴卡（120 道全量）：与旧题面那版并排，用来替换 §11.7 里的旧数字。

旧题面 = `st_reason_judge_opencode-<model>.csv`（+ glm 的补跑）
新题面 = `st_reason_judge_opencode-<model>-prefixfix120.csv`
真值都是 v6-120。
"""
import re
from pathlib import Path

import pandas as pd

A = Path(r"C:\dev\finlab-ml\data\judge_arena")
gold = pd.read_csv(A / "st_reason_gold_labels.v6-120.csv", dtype=str).fillna("")
gm = dict(zip(gold["id"], gold["裁定"]))


def prim(v):
    m = re.match(r"^\s*([ABCDE])", str(v or "").strip().upper())
    return m.group(1) if m else ""


def load(model, suffix=""):
    f = A / f"st_reason_judge_opencode-{model}{suffix}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, dtype=str).fillna("")
    if suffix == "":
        rf = A / f"st_reason_judge_opencode-{model}-refill.csv"
        if rf.exists():
            r = pd.read_csv(rf, dtype=str).fillna("")
            d = pd.concat([d[~d["id"].isin(r["id"])], r], ignore_index=True)
    d["p"] = d["choice"].apply(prim)
    d["g"] = d["id"].map(gm)
    d["ok"] = d["p"].eq(d["g"]) & d["p"].ne("")
    return d


def card(name, d):
    ans = d[d["p"].ne("")]
    ms = pd.to_numeric(d["ms"], errors="coerce")
    conf = pd.to_numeric(ans["conf"], errors="coerce").dropna()
    return {
        "批": name, "n": len(d),
        "覆盖": f"{len(ans)}/{len(d)}（{len(ans)/len(d):.1%}）",
        "一致率(答出的)": f"{ans['ok'].mean():.4f}（{int(ans['ok'].sum())}/{len(ans)}）",
        "一致率(空算错)": f"{d['ok'].mean():.4f}",
        "延迟中位": f"{ms.median():.0f} ms" if ms.notna().any() else "-",
        "置信度中位": f"{conf.median():.3f}" if len(conf) else "-",
        "不确定带": f"{(((conf >= .2) & (conf <= .8)).mean()):.2%}" if len(conf) else "-",
    }


rows = []
for model in ("glm-5.3", "kimi-k3"):
    for label, suf in (("旧题面", ""), ("**新题面**", "-prefixfix120")):
        d = load(model, suf)
        if d is None:
            print(f"{model} {label}:（还没落）")
            continue
        c = card(f"{model} · {label}", d)
        c["模型"] = model
        rows.append(c)
if rows:
    print(pd.DataFrame(rows)[["模型", "批", "覆盖", "一致率(答出的)", "一致率(空算错)", "延迟中位", "置信度中位", "不确定带"]].to_string(index=False))

print("\n=== 新题面下仍错的，逐条（看是不是又撞在真值歧义上）===")
for model in ("glm-5.3", "kimi-k3"):
    d = load(model, "-prefixfix120")
    if d is None:
        continue
    bad = d[d["p"].eq("") | (d["p"] != d["g"])]
    print(f"\n[{model}] {len(bad)} 条：")
    for _, r in bad.iterrows():
        g = gold[gold["id"] == r["id"]]
        why = g.iloc[0]["理由"] if len(g) else ""
        print(f"   {r['id']} gold={r['g']} 答={r['p'] or '空'} conf={r.get('conf')} · 真值理由：{why[:54]}")
