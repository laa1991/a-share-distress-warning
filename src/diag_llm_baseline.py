"""LLM 基线（**我们自己量的**，不是转述别人）：同一批修正公告，让一个真实 LLM 判方向，量延迟/成本/一致率/概率形态。

为什么单列：`docs/探索-判断器的效率（GBDT vs Jev vs LLM）.md` 里 LLM 那格现在是**第三方**的数字。
把第三方换成我们自己的，才有资格和 Jev（若拿到 key）和我们的规则/GBDT 放进同一张表。

设计：
- 样本：`data/yjyg_rev_direction.csv` 里有方向标签的正文，**分层抽 20/20/20**（坏→更坏 / 好→更好 / 不变）
- 提问：只给正文（截 3,000 字），要一个 JSON：{"direction": "worse|better|same", "confidence": 0..1}
- 读数：**逐条延迟 · tokens（响应自带）· 与规则标签的一致率 · 置信度的分布（校准要看这个）**
- ⚠️ 标签本身来自我的规则抽取（覆盖率 48.4%、与勾选框一致 98.7%）⇒ **报的是"与规则标签的一致率"，不是绝对正确率**。

用法: python diag_llm_baseline.py [--n-per-class 20] [--model deepseek-chat]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
TXT = ROOT / "data" / "raw" / "yjyg_rev_pdf"
CRED = Path(os.path.expanduser("~")) / ".dsh" / ".credentials.yaml"
OUT = ROOT / "data" / "llm_baseline.json"

PROMPT = """你是财务公告分析员。下面是一份 A 股「业绩预告修正公告」的正文（可能被截断）。
请判断：**相比前次预告，这次修正把业绩预期改好了还是改坏了**。

只回一个 JSON，不要任何解释：
{"direction": "better|worse|same", "confidence": 0.0~1.0}

判定口径：
- better = 修正后的预计净利润高于前次（含由亏转盈、亏损收窄、盈利增加）
- worse  = 修正后的预计净利润低于前次（含由盈转亏、亏损扩大、盈利减少）
- same   = 两者实质相同
confidence 是你对自己这个判断的把握程度（0.5=几乎在猜，1.0=正文里写得非常明确）。

正文：
---
{body}
---"""


def read_key() -> str:
    text = CRED.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^DEEPSEEK_API_KEY:\s*(.+)$", text, re.M)
    if not m:
        raise SystemExit("凭据库里没有 DEEPSEEK_API_KEY")
    val = m.group(1).strip().strip('"').strip("'")
    if len(val) < 20:                                  # ⚠️ YAML 折行兜底
        nxt = text[m.end():].splitlines()
        if nxt and nxt[0].startswith((" ", "\t")):
            val += nxt[0].strip().strip('"')
    return val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=20)
    ap.add_argument("--model", default="deepseek-chat")
    args = ap.parse_args()

    d = pd.read_csv(ROOT / "data" / "yjyg_rev_direction.csv", dtype={"code": str})
    d = d[d["方向"].notna() & d["prev"].notna() & d["after"].notna()]
    d["code"] = d["code"].str.zfill(6)
    d["date"] = pd.to_datetime(d["date"])
    parts = [g.sample(min(len(g), args.n_per_class), random_state=0) for _, g in d.groupby("方向")]
    s = pd.concat(parts)
    print(f"抽样 {len(s)} 份（分层：{s['方向'].value_counts().to_dict()}）")

    key = read_key()
    url = "https://api.deepseek.com/chat/completions"
    rows = []
    for i, r in enumerate(s.itertuples(index=False), 1):
        f = TXT / f"{r.code}_{r.date:%Y-%m-%d}.txt"
        if not f.exists():
            continue
        body = f.read_text(encoding="utf-8", errors="replace")[:3000]
        payload = {"model": args.model, "temperature": 0,
                   "messages": [{"role": "user", "content": PROMPT.replace("{body}", body)}],
                   "response_format": {"type": "json_object"}, "max_tokens": 100}
        t0 = time.perf_counter()
        try:
            resp = requests.post(url, headers={"Authorization": f"Bearer {key}",
                                               "Content-Type": "application/json"},
                                 json=payload, timeout=90)
            dt = time.perf_counter() - t0
            j = resp.json()
            if resp.status_code != 200:
                rows.append({"code": r.code, "truth": r.方向, "pred": None, "conf": None,
                             "ms": round(dt * 1000, 1), "err": str(j)[:90]})
                print(f"  {i}: HTTP {resp.status_code} {str(j)[:70]}", flush=True)
                continue
            txt = j["choices"][0]["message"]["content"]
            got = json.loads(txt)
            usage = j.get("usage", {})
            rows.append({"code": r.code, "truth": r.方向, "pred": got.get("direction"),
                         "conf": got.get("confidence"), "ms": round(dt * 1000, 1),
                         "in_tok": usage.get("prompt_tokens"), "out_tok": usage.get("completion_tokens"),
                         "err": ""})
            if i % 10 == 0 or i == len(s):
                ok = sum(1 for x in rows if x.get("pred"))
                print(f"  {i}/{len(s)} · 成功 {ok} · 累计 {sum(x['ms'] for x in rows)/1000:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            rows.append({"code": r.code, "truth": r.方向, "pred": None, "conf": None,
                         "ms": round((time.perf_counter() - t0) * 1000, 1), "err": f"{type(e).__name__}: {str(e)[:70]}"})
            print(f"  {i}: {type(e).__name__} {str(e)[:70]}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "llm_baseline.csv", index=False, encoding="utf-8-sig")
    ok = res[res["pred"].notna()].copy()
    map_ = {"坏→更坏": "worse", "好→更好": "better", "不变": "same"}
    ok["期望"] = ok["truth"].map(map_)
    acc = float((ok["pred"] == ok["期望"]).mean()) if len(ok) else None
    lat = ok["ms"]
    conf = pd.to_numeric(ok["conf"], errors="coerce")
    print(f"\n成功 {len(ok)}/{len(res)} · **与规则标签的一致率 {acc:.1%}**")
    print(f"延迟：p50 {lat.quantile(0.5):.0f} ms · p95 {lat.quantile(0.95):.0f} ms · 均值 {lat.mean():.0f} ms")
    ti = ok["in_tok"].sum() if "in_tok" in ok else 0
    to = ok["out_tok"].sum() if "out_tok" in ok else 0
    print(f"tokens：输入 {ti}（{ti/max(len(ok),1):.0f}/条）· 输出 {to}（{to/max(len(ok),1):.1f}/条）")
    print(f"置信度分布：min {conf.min():.2f} · 中位 {conf.median():.2f} · max {conf.max():.2f} · "
          f"落在 [0.2,0.8] 的比例 {float(((conf>=0.2)&(conf<=0.8)).mean()):.1%}")
    wrong = ok[ok["pred"] != ok["期望"]]
    if len(wrong):
        print(f"\n判错的 {len(wrong)} 条的置信度：" + " · ".join(f"{float(c):.2f}" for c in wrong["conf"].head(10)))
    out = {"n": int(len(res)), "ok": int(len(ok)), "agreement": round(acc, 4) if acc is not None else None,
           "p50_ms": round(float(lat.quantile(0.5)), 1) if len(ok) else None,
           "p95_ms": round(float(lat.quantile(0.95)), 1) if len(ok) else None,
           "in_tok": int(ti), "out_tok": int(to),
           "conf_median": round(float(conf.median()), 3) if len(ok) else None,
           "conf_uncertain_share": round(float(((conf >= 0.2) & (conf <= 0.8)).mean()), 4) if len(ok) else None,
           "model": args.model, "n_per_class": args.n_per_class}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n读数 -> {OUT} · 明细 -> data/llm_baseline.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
