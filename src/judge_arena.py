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
import time
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
                    "B=纯财务类（净利润/净资产/收入指标 · 「连续三年扣非孰低为负 + 持续经营存在不确定性」这一族也算）"
                    "C=重整/破产 "
                    "E=生产经营类（生产经营活动受到严重影响且预计 3 个月内不能恢复 · 主要业务停产）"
                    "D=正文里只有规则条文引用或空话，看不出触发原因 "
                    "?=读完全文仍判不了。"
                    "⚠️ 若公告同时写了背景触发与本次叠加的触发，取**正文中较早出现的那个**（多值在理由里标 `+另:X`）。",
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
    cases = load_rev_cases(n) if task == "rev_dir" else load_st_cases(n)
    return attach_gold(task, cases)


def attach_gold(task: str, cases: pd.DataFrame) -> pd.DataFrame:
    """把**裁决过的真值**（人工/独立读者）贴到题上 —— 这是 arena 该用的裁判，不是我的规则标签。

    规则标签只留作对照（它自己就有 25% 量级的噪声，见 `docs/为什么戴帽（原因分布）.md`）。
    真值文件按 id 键（多轮分别产出的都收）：`<task>_gold_labels*.csv` 里带 `裁定` 列的那些。
    """
    gold: dict[str, str] = {}
    for f in sorted(OUT.glob(f"{task}_gold_labels*.csv")):
        try:
            d = pd.read_csv(f, dtype=str).fillna("")
        except Exception:  # noqa: BLE001
            continue
        col = next((c for c in ("裁定", "人工裁定") if c in d.columns), None)
        if not col:
            continue
        for _, r in d.iterrows():
            v = str(r[col]).strip()
            if v and v != "?" and r.get("id"):
                gold[str(r["id"])] = v
    cases = cases.copy()
    cases["truth_gold"] = cases["id"].map(gold)
    hit = int(cases["truth_gold"].notna().sum())
    print(f"    [gold] {task}：{len(cases)} 道题里 **{hit}** 道有裁决真值"
          + ("" if hit == len(cases) else "（其余只跑规则基线）"))
    return cases


def rule_baseline(task: str, cases: pd.DataFrame) -> dict:
    """规则基线：这条任务上，"我的规则"就是标签来源本身 ⇒ 一致率 100%、覆盖 <100%（抽不到的没进题）。
    所以这张台真正要量的是**判断器相对规则多抓到了什么**，以及它自己的校准。"""
    m = PROMPTS[task]["truth_map"]
    cov = cases["truth_rule"].isin(m).mean()
    out = {"cases": int(len(cases)), "rule_label_coverage": round(float(cov), 4),
           "rule_agreement": 1.0 if cov == 1 else None,
           "note": "规则基线的参差在**覆盖率**上（48.4% / 67.5%），不在一致率上；见文档 §2.5 与 §8"}
    if "truth_gold" in cases.columns:
        g = cases[cases["truth_gold"].notna()].copy()
        # ⚠️ 两边都要过同一张映射表再比：rev_dir 的 map 把「坏→更坏」映成 0/2/4（数字刻度），
        #    而金标表里存的是三分类**字符串** ⇒ 只映一侧会得出 0% 一致的假读数（我撞过一次）。
        norm = lambda v: m.get(v, v)  # noqa: E731
        g["_rule"] = g["truth_rule"].map(norm)
        g["_gold"] = g["truth_gold"].map(norm)
        ok = g[g["_rule"].notna()]
        if len(ok):
            a = (ok["_gold"] == ok["_rule"])
            out["rule_vs_gold_n"] = int(len(ok))
            out["rule_vs_gold_agreement"] = round(float(a.mean()), 4)
            out["rule_vs_gold_disagree"] = ok[~a][["id", "_rule", "_gold"]].values.tolist()
    return out


def jev_payload(task: str, one: pd.Series) -> dict:
    """Jev 的请求体形状（Decisions API：state + 类型化问题）。**没 key 也能打印出来核对形状。**"""
    p = PROMPTS[task]
    return {
        "model": "typesafe/jev-1.13",
        "state": {"document": one["text"]},
        "questions": [{"id": task, "primitive": p["primitive"], "question": p["question"]}],
        "response_format": p["schema"],
    }


def _opencode_key() -> str:
    """OpenCode 的 key：先环境变量，再凭据库（⚠️ YAML 会对长值折行，要吃掉续行）。"""
    k = os.environ.get("OPENCODE_API_KEY", "")
    if k:
        return k
    p = Path(os.path.expanduser("~")) / ".dsh" / ".credentials.yaml"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = re.search(r"^OPENCODE_API_KEY:\s*(.+)$", text, re.M)
    if not m:
        return ""
    v = m.group(1).strip().strip('"').strip("'")
    if len(v) < 20:                                  # 折行兜底：下一行若有缩进就是续行
        nxt = text[m.end():].splitlines()
        if nxt and nxt[0].startswith((" ", "\t")):
            v += nxt[0].strip().strip('"')
    return v


def _parse_json_answer(s: str) -> dict:
    """从模型的自由文本里抠出第一个 JSON 对象（判断器不许我们改它的作文格式，只能宽收）。"""
    if not s:
        return {}
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            return json.loads(m.group(0).replace("'", '"'))
        except json.JSONDecodeError:
            return {}


OPENCODE_BASE = "https://opencode.ai/zen/go/v1/chat/completions"
OPENCODE_ZEN = "https://opencode.ai/zen/v1/chat/completions"
# ⚠️ 两个端点不是一回事，实测（2026-09-25 凌晨，同一把 key）：
#   · `/zen/go/v1`（Go 订阅那侧）：**必须带 `x-opencode-session`**（缺了 400 MissingSessionID）；**不含 jev**
#   · `/zen/v1`（按量付费那侧）：**jev-1.13 只在这儿**；且**部分上游不可用**时会回 503 `Endpoint is unavailable`
#     （实测同一时刻：deepseek-v4-pro / glm-5.3 → 200，而 claude-sonnet-5 / gpt-5.5 / jev-1.13 → 503）
#   ⇒ 路由规则按模型名走：`jev*` 走 Zen 按量付费，其余走 Go 订阅。


def call_judge(cases: pd.DataFrame, task: str, provider: str, model: str = "") -> pd.DataFrame:
    """只留一个 call hook：接哪个判断器改这里。**缺 key / 缺依赖一律大声停住，不静默降级。**"""
    if provider == "none":
        print("\n[judge] 未指定判断器（--judge none）：只跑规则基线。")
        return pd.DataFrame()

    import uuid

    import requests  # noqa: E402

    if provider == "opencode":
        key = _opencode_key()
        if not key:
            raise SystemExit("[judge] ❌ 缺 OPENCODE_API_KEY（环境变量与凭据库都没有）—— 本台**不静默降级**。")
        model = model or "glm-5.3"
        url = OPENCODE_ZEN if model.lower().startswith("jev") else OPENCODE_BASE
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                   "x-opencode-session": str(uuid.uuid4()), "x-opencode-client": "dsh-judge-arena/1.0"}
    elif provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise SystemExit("[judge] ❌ 缺 OPENROUTER_API_KEY —— 本台**不静默降级**：要么设 key，要么用 --judge none。")
        url = "https://openrouter.ai/api/alpha/decisions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    else:
        raise SystemExit(f"[judge] 未知 provider: {provider}")

    ask = PROMPTS[task]["question"]
    rows = []
    for r in cases.itertuples(index=False):
        if provider == "opencode":
            payload = {"model": model, "temperature": 0, "max_tokens": 6000,
                       "messages": [{"role": "user",
                                     "content": f"{ask}\n\n---\n{r.text}\n---\n"
                                                "只回一个 JSON：{\"choice\":\"…\",\"confidence\":0..1}"}]}
        else:
            payload = jev_payload(task, pd.Series({"text": r.text}))
        t0 = time.perf_counter()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=180)
            ms = (time.perf_counter() - t0) * 1000
            j = resp.json()
            if provider == "opencode":
                msg = (j.get("choices") or [{}])[0].get("message", {}) or {}
                content = msg.get("content") or ""
                ans = _parse_json_answer(content)
                # ⚠️ 这些模型会先写 `reasoning_content`：若正文为空或没抠出 JSON，去思考通道里再抠一次
                #    （实测 max_tokens=900 时 1/3 的样本 JSON 全落在思考通道里被截断）
                if not ans:
                    think = msg.get("reasoning_content") or msg.get("reasoning") or ""
                    ans = _parse_json_answer(think)
                    if ans:
                        content = f"[从 reasoning 抠出] {think[-200:]}"
                usage = j.get("usage") or {}
                rows.append({"id": r.id, "truth_rule": r.truth_rule, "http": resp.status_code,
                             "choice": ans.get("choice"), "conf": ans.get("confidence"),
                             "raw": content[:200], "ms": round(ms, 1),
                             "ptok": usage.get("prompt_tokens"), "ctok": usage.get("completion_tokens"),
                             "cost_usd": usage.get("cost"), "err": "" if resp.status_code == 200 else resp.text[:120]})
            else:
                rows.append({"id": r.id, "truth_rule": r.truth_rule, "http": resp.status_code,
                             "choice": json.dumps(j.get("answers", j), ensure_ascii=False), "conf": None,
                             "raw": "", "ms": round(ms, 1), "ptok": (j.get("usage") or {}).get("prompt_tokens"),
                             "ctok": (j.get("usage") or {}).get("completion_tokens"),
                             "cost_usd": (j.get("usage") or {}).get("cost"), "err": ""})
        except Exception as e:  # noqa: BLE001
            rows.append({"id": r.id, "truth_rule": r.truth_rule, "http": None, "choice": None, "conf": None,
                         "raw": "", "ms": None, "ptok": None, "ctok": None, "cost_usd": None,
                         "err": f"{type(e).__name__}: {str(e)[:80]}"})
    return pd.DataFrame(rows)


def _primary(choice) -> tuple[str, str]:
    """把「A+另:B」这种**主值 + 多值标注**拆开：主值 = 首个字母；剩下的是标注。

    ⚠️ 不许用 `.str[:1]` 图省事：它会把**回显/说明文字**的首字符当成答案，
      实测（2026-09-25）它把 3 条「A+另:B」当成「答对了 A」、还让读数看着更漂亮 —— **尺子造的，不是模型答的**。
    """
    s = str(choice or "").strip().upper()
    m = re.match(r"^\s*([ABCDE])", s)
    if not m:
        return "", s
    return m.group(1), s[len(m.group(1)):]


def score_judge(res: pd.DataFrame, task: str, cases: pd.DataFrame) -> dict:
    """四轴里能算的三轴 + 与裁决真值的一致率（裁判 = v6-120）。"""
    out: dict = {"n": int(len(res)), "ok": int(res["err"].eq("").sum())}
    if "ms" in res and res["ms"].notna().any():
        out["ms_p50"] = round(float(res["ms"].median()), 1)
        out["ms_p95"] = round(float(res["ms"].quantile(0.95)), 1)
    for c in ("ptok", "ctok"):
        if c in res and res[c].notna().any():
            out[c + "_sum"] = int(res[c].fillna(0).sum())
    if res.get("cost_usd") is not None and res["cost_usd"].notna().any():
        out["cost_usd_sum"] = float(pd.to_numeric(res["cost_usd"], errors="coerce").fillna(0).sum())

    gold = dict(zip(cases["id"], cases.get("truth_gold", pd.Series(dtype=str))))
    s = res.copy()
    s["gold"] = s["id"].map(gold)
    s[["_primary", "_extra"]] = s["choice"].apply(lambda v: pd.Series(_primary(v)))
    s["_multi"] = s["_extra"].str.contains("另|\\+", regex=True).fillna(False)

    allg = s[s["gold"].notna()]
    dec = allg[allg["_primary"].isin(list("ABCDE"))]
    out["answered"] = int(len(dec))
    out["answered_share"] = round(float(len(dec) / max(len(allg), 1)), 4)
    out["multi_value_share"] = round(float(dec["_multi"].mean()), 4) if len(dec) else None
    if len(dec):
        ok = (dec["_primary"] == dec["gold"])
        out["gold_n"] = int(len(dec))
        out["accuracy_on_answered"] = round(float(ok.mean()), 4)
        out["accuracy_all_counting_blank_as_wrong"] = round(float(ok.sum() / max(len(allg), 1)), 4)
    # 校准：confidence 是"我选的那个的概率" ⇒ 不确定带 [0.2,0.8]（与 GBDT 那格同口径）
    conf = pd.to_numeric(dec["conf"] if len(dec) else s["conf"], errors="coerce")
    if conf.notna().any():
        c = conf.dropna()
        out["conf_median"] = round(float(c.median()), 3)
        out["uncertain_band_share"] = round(float(((c >= 0.2) & (c <= 0.8)).mean()), 4)
        if len(dec):
            cc = pd.to_numeric(dec["conf"], errors="coerce")
            m = cc.notna()
            if m.any():
                out["brier_conf_vs_correct"] = round(
                    float(((cc[m] - (dec["_primary"][m] == dec["gold"][m]).astype(float)) ** 2).mean()), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--gold", type=int, default=0)
    ap.add_argument("--judge", default="none", choices=["none", "openrouter", "opencode"])
    ap.add_argument("--model", default="", help="opencode 通道用哪个模型（如 glm-5.3 / kimi-k3 / grok-4.7）")
    ap.add_argument("--task", default="", help="只跑一个任务（st_reason / rev_dir）；空 = 两个都跑")
    ap.add_argument("--limit", type=int, default=0, help="判断器只跑前 N 道（0 = 全跑）")
    ap.add_argument("--ids-file", default="", help="只跑这个文件里列出的 id（补跑弃答/失败用）")
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tasks = [args.task] if args.task else ["rev_dir", "st_reason"]
    summary = {}
    for task in tasks:
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
            run = cases
            if args.ids_file:
                want = [x.strip() for x in Path(args.ids_file).read_text(encoding="utf-8").split() if x.strip()]
                run = cases[cases["id"].isin(want)].copy()
                print(f"[judge] 只跑 ids-file 里的 {len(run)}/{len(want)} 道（{args.ids_file}）")
            elif args.limit:
                run = cases.head(args.limit)
            print(f"\n[judge] {args.judge}{' · ' + args.model if args.model else ''} · 跑 {len(run)} 道…", flush=True)
            res = call_judge(run, task, args.judge, args.model)
            tag = f"{args.judge}-{args.model or 'default'}".replace("/", "_")
            out_csv = OUT / f"{task}_judge_{tag}{'-refill' if args.ids_file else ''}.csv"
            res.to_csv(out_csv, index=False, encoding="utf-8-sig")
            sc = score_judge(res, task, cases)
            print(f"[judge] {task} 读数：{sc}")
            print(f"[judge] 明细 → {out_csv}")
            summary[task]["judge"] = {tag: sc}

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n考卷与汇总 -> {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
