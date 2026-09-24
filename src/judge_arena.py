"""判断器对照台（Judge Arena）—— 把 §8 里「需要判断器」的用法做成**可跑的考卷**，只留一个 call hook。

设计要点（为什么是这个形状）：
- **只收"语义"任务**：修订方向（Score）· 戴帽原因（Choice）。凡能用算术判的（口径归属 / 两列关系）
  都**不进这张台** —— 它们是代码闸的事（见文档 §8 结论 1：把确定性的事交给概率装置 = 自造不确定源）。
- **每条都带基线**：规则基线（本机可跑）+ LLM 基线（已实测，落 `data/llm_baseline.json`）。
- **一个 call hook**：`--judge openrouter` 才联网；缺 key 时**大声停住**，不做静默降级。
- **金标抽样**：`--gold N` 导出人工裁定表 —— 因为这两条任务的"真值"目前都是**我的规则标签**，
  不补一层人工金标，所有比较都只是"与我的规则有多一致"。

用法:
    python judge_arena.py --dry                 # 建题 + 跑规则基线 + 打印一例 Jev 请求体（不联网）
    python judge_arena.py --gold 60             # 导出人工裁定表（真值升级用）
    python judge_arena.py --judge openrouter    # 有 key 时：把 Jev 那一格插进去（缺 key 会明确报错）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BODY_DIRS = {"rev_dir": ROOT / "data" / "raw" / "yjyg_rev_pdf", "st_reason": ROOT / "data" / "raw" / "st_pdf"}
OUT = ROOT / "data" / "judge_arena"

# 两个原语的题面（与 TypeSafe 的 Choice / Score 一一对应；换判断器只改 call_judge）
PROMPTS = {
    "rev_dir": {
        "primitive": "score",
        "question": "相比前次预告，这份修正公告把全年业绩预期改好了还是改坏了？请在 ordered scale 上给分："
                    "0=明显改坏 1=略改坏 2=基本不变 3=略改好 4=明显改好。",
        "schema": {"score": "int 0..4", "confidence": "0..1"},
        "truth_map": {"坏→更坏": 0, "不变": 2, "好→更好": 4},          # 粗三档；略改好/略改坏留给打分
    },
    "st_reason": {
        "primitive": "choice",
        "question": "这份「被实施风险警示」公告里，触发戴帽的原因属于哪一类？选项："
                    "A=治理/合规类（内控被否·处罚·账户冻结·资金占用·违规担保）"
                    "B=纯财务类（净利润/净资产/收入指标）"
                    "C=重整/破产 "
                    "E=生产经营类（生产经营活动受到严重影响且预计 3 个月内不能恢复 · 主要业务停产）"
                    "D=摘录里只有规则条文引用或空话，看不出触发原因 "
                    "?=摘录不足以判断",
        # ⚠️ E 不是我编的档：它是交易所规则里**与 A、B 并列的第三条触发**（「生产经营活动受到严重影响…」），
        #    实测 600165（子公司临时停产）· 688089（生产经营受严重影响）两条原本被硬塞在 A/B/C 之外。
        "schema": {"choice": "A|B|C|D|E", "probability": {"A": "0..1", "B": "0..1", "C": "0..1",
                                                          "D": "0..1", "E": "0..1"},
                   "confidence": "0..1"},
        "truth_map": {"合规/治理类": "A", "财务类": "B", "重整/破产": "C", "未归类": "D", "生产经营类": "E"},
    },
}


def load_rev_cases(n: int) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "yjyg_rev_direction.csv", dtype={"code": str})
    d = d[d["方向"].notna()]
    d["code"] = d["code"].str.zfill(6)
    d["date"] = pd.to_datetime(d["date"])
    rows = []
    for r in d.itertuples(index=False):
        f = BODY_DIRS["rev_dir"] / f"{r.code}_{r.date:%Y-%m-%d}.txt"
        if not f.exists():
            continue
        rows.append({"id": f"{r.code}_{r.date:%Y-%m-%d}", "text": f.read_text(encoding="utf-8", errors="replace")[:3000],
                     "truth_rule": r.方向})
        if len(rows) >= n:
            break
    return pd.DataFrame(rows)


def load_st_cases(n: int) -> pd.DataFrame:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from diag_reason_channels import bucket as bucket_of   # 复用同一套粗分类，别在这里另立一套口径
    d = pd.read_csv(ROOT / "data" / "st_reasons.csv", dtype={"code": str})
    d["code"] = d["code"].str.zfill(6)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.sample(frac=1.0, random_state=0)                 # 打散：别只拿代码靠前的公司
    rows = []
    for r in d.itertuples(index=False):
        want = getattr(r, "类别", None)
        if not isinstance(want, str):
            continue
        f = BODY_DIRS["st_reason"] / f"{r.code}_{r.date:%Y-%m-%d}.txt"
        if not f.exists():
            continue
        rows.append({"id": f"{r.code}_{r.date:%Y-%m-%d}", "text": f.read_text(encoding="utf-8", errors="replace")[:3000],
                     "truth_rule": bucket_of(want)})
        if len(rows) >= n:
            break
    return pd.DataFrame(rows)


def build_cases(task: str, n: int) -> pd.DataFrame:
    return load_rev_cases(n) if task == "rev_dir" else load_st_cases(n)


def rule_baseline(task: str, cases: pd.DataFrame) -> dict:
    """规则基线：这条任务上，"我的规则"就是标签来源本身 ⇒ 一致率 100%、覆盖 <100%（抽不到的没进题）。
    所以这张台真正要量的是**判断器相对规则多抓到了什么**，以及它自己的校准。"""
    m = PROMPTS[task]["truth_map"]
    cov = cases["truth_rule"].isin(m).mean()
    return {"cases": int(len(cases)), "rule_label_coverage": round(float(cov), 4),
            "rule_agreement": 1.0 if cov == 1 else None,
            "note": "规则基线的参差在**覆盖率**上（48.4% / 67.5%），不在一致率上；见文档 §2.5 与 §8"}


def jev_payload(task: str, one: pd.Series) -> dict:
    """Jev 的请求体形状（Decisions API：state + 类型化问题）。**没 key 也能打印出来核对形状。**"""
    p = PROMPTS[task]
    return {
        "model": "typesafe/jev-1.13",
        "state": {"document": one["text"]},
        "questions": [{"id": task, "primitive": p["primitive"], "question": p["question"]}],
        "response_format": p["schema"],
    }


def call_judge(cases: pd.DataFrame, task: str, provider: str) -> pd.DataFrame:
    if provider == "none":
        print("\n[judge] 未指定判断器（--judge none）：只跑规则基线。要接 Jev 请设 OPENROUTER_API_KEY 后 --judge openrouter。")
        return pd.DataFrame()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("[judge] ❌ 缺 OPENROUTER_API_KEY —— 本台**不静默降级**：要么设 key，要么用 --judge none。"
                         "\n        （key 也不在凭据库里：本机只有 DEEPSEEK / MINIMAX / OPENCODE / GITHUB。）")
    import requests  # noqa: E402
    rows = []
    for r in cases.itertuples(index=False):
        payload = jev_payload(task, pd.Series({"text": r.text}))
        t0 = pd.Timestamp.now()
        try:
            resp = requests.post("https://openrouter.ai/api/alpha/decisions",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                 json=payload, timeout=60)
            ms = (pd.Timestamp.now() - t0).total_seconds() * 1000
            j = resp.json()
            rows.append({"id": r.id, "truth_rule": r.truth_rule, "pred": json.dumps(j.get("answers", j), ensure_ascii=False),
                         "ms": round(ms, 1), "cost_usd": (j.get("usage") or {}).get("cost"), "err": ""})
        except Exception as e:  # noqa: BLE001
            rows.append({"id": r.id, "truth_rule": r.truth_rule, "pred": None, "ms": None, "cost_usd": None,
                         "err": f"{type(e).__name__}: {str(e)[:80]}"})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--gold", type=int, default=0)
    ap.add_argument("--judge", default="none", choices=["none", "openrouter"])
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    summary = {}
    for task in ("rev_dir", "st_reason"):
        cases = build_cases(task, args.n)
        base = rule_baseline(task, cases)
        print(f"\n=== {task}（{PROMPTS[task]['primitive']}）· 题 {len(cases)} 道")
        print(f"    规则基线：{base}")
        print(f"    题面：{PROMPTS[task]['question'][:78]}…")
        payload = jev_payload(task, cases.iloc[0])
        print(f"    Jev 请求体形状（前 200 字符）：{json.dumps(payload, ensure_ascii=False)[:200]}…")
        cases.to_csv(OUT / f"{task}_cases.csv", index=False, encoding="utf-8-sig")
        summary[task] = {"cases": len(cases), "baseline": base,
                         "llm_baseline_recorded": "data/llm_baseline.json" if task == "rev_dir" else None}
        if args.gold:
            g = cases.sample(min(args.gold, len(cases)), random_state=0).copy()
            g["人工裁定"] = ""
            g["备注"] = ""
            g[["id", "truth_rule", "人工裁定", "备注", "text"]].to_csv(
                OUT / f"{task}_gold_template.csv", index=False, encoding="utf-8-sig")
            print(f"    金标模板 -> {OUT / f'{task}_gold_template.csv'}（{len(g)} 行，填「人工裁定」列）")

    if args.judge != "none":
        for task in ("rev_dir", "st_reason"):
            cases = pd.read_csv(OUT / f"{task}_cases.csv").head(20)
            res = call_judge(cases, task, args.judge)
            if len(res):
                agree = (res["pred"].notna()).mean()
                print(f"  {task}: 成功 {agree:.0%} · 中位延迟 {res['ms'].median():.0f} ms · "
                      f"合计成本 ${pd.to_numeric(res['cost_usd'], errors='coerce').sum():.6f}")

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n考卷与汇总 -> {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
