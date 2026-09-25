"""两个模型"是怎么走到答案的"—— 把 reasoning 通道完整留档，做第一张对照切片。

⚠️ 边界先说清：`reasoning_content` 是**模型自己讲的故事**，不是它内部计算的忠实记录。
它能回答的是"**它在叙述里用了哪条证据**"（比如引的是内控那句还是财报那句），不能回答"它内部怎么算的"。

样本（10 条）：5 条**两家都判错**的（任务难点候选）+ 5 条两家都对的（覆盖 A/B/C/E 四档）。
同题面、temperature 0、max_tokens 6000；两家都用 Go 通道。
"""
import json
import os
import re
import time
import uuid
from pathlib import Path

import requests

A = Path(r"C:\dev\finlab-ml\data\judge_arena")
OUT = A / "reasoning_paths.json"
CRED = Path(os.path.expanduser("~")) / ".dsh" / ".credentials.yaml"
s = CRED.read_text(encoding="utf-8", errors="replace")
m = re.search(r"^OPENCODE_API_KEY:\s*(.+)$", s, re.M)
key = m.group(1).strip().strip('"').strip("'")
if len(key) < 20:
    key += s[m.end():].splitlines()[0].strip().strip('"')

gold = pd.read_csv(A / "st_reason_gold_labels.v6-120.csv", dtype=str).fillna("") if False else None
import pandas as pd  # noqa: E402

gold = pd.read_csv(A / "st_reason_gold_labels.v6-120.csv", dtype=str).fillna("")
gm = dict(zip(gold["id"], gold["裁定"]))
cases = pd.read_csv(A / "st_reason_cases.csv", dtype=str).fillna("")
text = dict(zip(cases["id"], cases["text"]))

SHARED_WRONG = ["002528_2025-04-30", "300125_2024-08-19", "300225_2025-07-02", "600745_2026-04-30", "603557_2024-04-27"]
# 5 条两家都对、且覆盖四档
RIGHT = ["603023_2024-04-30", "002598_2026-05-09", "000615_2023-04-29", "002021_2023-11-25", "600165_2024-04-03"]
IDS = SHARED_WRONG + RIGHT

ASK = ("这份「被实施风险警示」公告里，触发戴帽的原因属于哪一类？选项："
       "A=治理/合规类（内控被否·处罚·账户冻结·资金占用·违规担保）"
       "B=纯财务类（净利润/净资产/收入指标）C=重整/破产 "
       "E=生产经营类（生产经营活动受到严重影响且预计 3 个月内不能恢复）"
       "D=正文里只有规则条文引用或空话，看不出触发原因。"
       "若公告同时写了背景触发与本次叠加的触发，取正文中较早出现的那个。"
       "只回一个 JSON：{\"choice\":\"…\",\"confidence\":0..1}")

ANCHORS = {
    "内控意见": r"内部控制.{0,12}(否定|无法表示|意见)",
    "财报意见": r"(财务报表|财务报告).{0,12}(无法表示|否定|保留)",
    "净资产为负": r"净资产.{0,6}为负",
    "净利为负": r"净利润.{0,6}(为负|负值)",
    "营收门槛": r"(营业收入|扣除后).{0,12}(低于|不足|亿)",
    "重整破产": r"(重整|破产|受理)",
    "账户冻结": r"(冻结|资金占用|违规担保)",
    "停产": r"(停产|生产经营活动受到严重影响)",
    "条款号": r"\d+\.\d+\.\d+",
}


def norm_reasoning(x: str) -> str:
    return re.sub(r"\s+", "", x or "")


rows = []
for cid in IDS:
    rec = {"id": cid, "gold": gm.get(cid), "text_len": len(text.get(cid, ""))}
    for model in ("glm-5.3", "kimi-k3"):
        payload = {"model": model, "temperature": 0, "max_tokens": 6000,
                   "messages": [{"role": "user", "content": f"{ASK}\n\n---\n{text.get(cid,'')}\n---"}]}
        h = {"Authorization": f"Bearer {key}", "x-opencode-session": str(uuid.uuid4()),
             "x-opencode-client": "dsh-reasoning-probe/1.0"}
        t0 = time.perf_counter()
        try:
            r = requests.post("https://opencode.ai/zen/go/v1/chat/completions", headers=h, json=payload, timeout=300)
            ms = (time.perf_counter() - t0) * 1000
            j = r.json()
            msg = (j.get("choices") or [{}])[0].get("message", {}) or {}
            content, think = msg.get("content") or "", msg.get("reasoning_content") or ""
            nr = norm_reasoning(think)
            rec[model] = {
                "http": r.status_code, "ms": round(ms),
                "choice": (re.search(r'"choice"\s*:\s*"([^"]+)"', content) or [None, ""])[1],
                "content_len": len(content), "think_len": len(think),
                "think_norm": nr[:6000],
                "anchors": [k for k, pat in ANCHORS.items() if re.search(pat, nr)],
                "ctok": (j.get("usage") or {}).get("completion_tokens"),
            }
        except Exception as e:  # noqa: BLE001
            rec[model] = {"http": None, "err": f"{type(e).__name__}: {str(e)[:80]}"}
        time.sleep(1)
    rows.append(rec)
    g, k = rec.get("glm-5.3", {}), rec.get("kimi-k3", {})
    print(f"{cid} gold={rec['gold']} | glm={g.get('choice')}(think {g.get('think_len')}字 {g.get('anchors')}) "
          f"| kimi={k.get('choice')}(think {k.get('think_len')}字 {k.get('anchors')})", flush=True)

OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n明细 -> {OUT}")
