"""第二条通道：把「事件名单」并到模型名单上，到底值不值？

关键对照（上一版缺了这一格）：**并集把名单撑大了，那就得跟「同一份分数把门槛放低到同样大」比一比** ——
如果单纯放宽分数就能拿到同样的覆盖率和更好的精确率，那"第二条通道"就是幻觉。

判据：同样大小的名单下，`并集` 的覆盖率/精确率 − `模型放低门槛` 的覆盖率/精确率。
用法: python diag_union_channel.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_events_ablation import run  # noqa: E402
from events_features import build as build_events  # noqa: E402
from run_experiments import CAT, FEATURES  # noqa: E402
from run_leadtime_curve import base_label, samples_at_quarter  # noqa: E402

CHANNELS = {
    "诉讼（上年+Y Q1）": "sue_any",
    "诉讼·近两季": "sue_any_recent",
    "担保≥净资产50%": "gua_heavy",
    "担保·近两季": "gua_heavy_recent",
}


def ev_mask(g: pd.DataFrame, name: str) -> pd.Series:
    if name == "并集（诉讼∨担保）":
        return (g["sue_any"] > 0) | (g["gua_heavy"] > 0)
    if name == "并集·近两季":
        return (g["sue_any_recent"] > 0) | (g["gua_heavy_recent"] > 0)
    return g[CHANNELS[name]] > 0


def eval_year(g: pd.DataFrame, name: str) -> dict:
    g = g.copy()
    npos = int(g["label"].sum())
    k = max(int(round(0.10 * len(g))), 1)
    thr = g["p"].nlargest(k).min()
    m = g["p"] >= thr
    e = ev_mask(g, name)
    u = m | e
    # ★ 对照：不放事件，直接把分数门槛放低到并集那么大
    ku = int(u.sum())
    thr_u = g["p"].nlargest(ku).min()
    m2 = g["p"] >= thr_u
    only_e = e & ~m
    return {
        "n": len(g), "n_pos": npos,
        "model": {"size": int(m.sum()), "cov": float(g.loc[m, "label"].sum() / npos),
                  "prec": float(g.loc[m, "label"].mean())},
        "ev": {"size": int(e.sum()), "cov": float(g.loc[e, "label"].sum() / npos),
               "prec": float(g.loc[e, "label"].mean()) if e.any() else None},
        "union": {"size": ku, "cov": float(g.loc[u, "label"].sum() / npos),
                  "prec": float(g.loc[u, "label"].mean())},
        "model_widened": {"size": int(m2.sum()), "cov": float(g.loc[m2, "label"].sum() / npos),
                          "prec": float(g.loc[m2, "label"].mean())},
        "block": {"size": int(only_e.sum()),
                  "prec": float(g.loc[only_e, "label"].mean()) if only_e.any() else None,
                  "adds_true": int(g.loc[only_e, "label"].sum()),
                  "adds_false": int(only_e.sum() - g.loc[only_e, "label"].sum())},
    }


def main() -> int:
    panel = pd.read_pickle(ROOT / "data" / "panel.pkl")
    ann = base_label(panel)
    s = samples_at_quarter(panel, 1, ann)
    ev = build_events()
    s = s.merge(ev, on=["code", "year"], how="left")
    for c in ("sue_any", "sue_any_recent", "gua_heavy", "gua_heavy_recent"):
        s[c] = s[c].fillna(0.0)

    _, pred = run(s, FEATURES + CAT)          # A 臂：只用财报
    pred = pred.merge(ev, on=["code", "year"], how="left")
    for c in ("sue_any", "sue_any_recent", "gua_heavy", "gua_heavy_recent"):
        pred[c] = pred[c].fillna(0.0)

    names = list(CHANNELS) + ["并集（诉讼∨担保）", "并集·近两季"]
    out = {}
    rows = []
    for name in names:
        per = {int(Y): eval_year(g, name) for Y, g in pred.groupby("year")}
        out[name] = per
        mean = lambda path: sum(p[path[0]][path[1]] for p in per.values()) / len(per)  # noqa: E731
        rows.append({
            "通道": name,
            "事件覆盖": mean(("ev", "size")) / mean(("n", "n")) if False else per[2025]["ev"]["size"] / per[2025]["n"],
            "并集规模": mean(("union", "size")),
            "并集覆盖率": mean(("union", "cov")),
            "并集精确率": mean(("union", "prec")),
            "同样规模·模型放低门槛": mean(("model_widened", "cov")),
            "同样规模·模型精确率": mean(("model_widened", "prec")),
            "Δ覆盖率": mean(("union", "cov")) - mean(("model_widened", "cov")),
            "Δ精确率": mean(("union", "prec")) - mean(("model_widened", "prec")),
            "增量块精确率": sum((p["block"]["prec"] or 0) for p in per.values()) / len(per),
            "多抓真亏": sum(p["block"]["adds_true"] for p in per.values()) / len(per),
            "多带误报": sum(p["block"]["adds_false"] for p in per.values()) / len(per),
        })

    tab = pd.DataFrame(rows)
    print(f"基准：模型前 10% 覆盖率 {sum(p['model']['cov'] for p in out[names[0]].values())/4:.2%} · "
          f"精确率 {sum(p['model']['prec'] for p in out[names[0]].values())/4:.2%}（四年平均）\n")
    show = tab.copy()
    for c in show.columns:
        if c != "通道":
            show[c] = show[c].round(4)
    print(show.to_string(index=False))

    print("\n★ 判据（同样规模的名单）：Δ覆盖率 > 0 ⇒ 事件拼出来的确实比单纯放宽分数更会挑；")
    print("   Δ精确率 < 0 ⇒ 代价是名单更脏。两个都看，不看一个。")

    (ROOT / "data" / "union_channel.json").write_text(
        json.dumps({"benchmark_model": {k: v for k, v in out[names[0]][2025]["model"].items()},
                    "per_channel": {n: {str(k): v for k, v in p.items()} for n, p in out.items()},
                    "summary": tab.round(5).to_dict("records")}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("\n读数 -> data/union_channel.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
