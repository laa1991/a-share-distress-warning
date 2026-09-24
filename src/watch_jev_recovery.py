"""Jev 恢复观察器：每 4 分钟打一发 jev-1.13，通了就记一行并退出；打满 15 轮（约 1 小时）也退出。

产物：`C:\dev\finlab-ml\data\judge_arena\jev_recovery.log`（每次一行：时刻 + 状态码 + 原文前 80 字）。
它回答的是明早那个问题：**Jev 今晚恢复过没有**。
"""
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests

LOG = Path(r"C:\dev\finlab-ml\data\judge_arena\jev_recovery.log")
CRED = Path(os.path.expanduser("~")) / ".dsh" / ".credentials.yaml"
t = CRED.read_text(encoding="utf-8", errors="replace")
m = re.search(r"^OPENCODE_API_KEY:\s*(.+)$", t, re.M)
key = m.group(1).strip().strip('"').strip("'")
if len(key) < 20:
    nxt = t[m.end():].splitlines()
    if nxt and nxt[0].startswith((" ", "\t")):
        key += nxt[0].strip().strip('"')

URL = "https://opencode.ai/zen/v1/chat/completions"
PAY = {"model": "jev-1.13", "temperature": 0, "max_tokens": 200,
       "messages": [{"role": "user", "content": "只回一个 JSON：{\"choice\":\"A\",\"confidence\":0.9}"}]}


def log(line: str) -> None:
    with LOG.open("a", encoding="utf-8") as f:          # 只追加，不读旧内容
        f.write(line + "\n")
    print(line, flush=True)


log(f"\n=== 观察开始 {datetime.now():%Y-%m-%d %H:%M:%S}（每 4 分钟一发，最多 15 轮）")
for i in range(1, 16):
    try:
        r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=PAY, timeout=120)
        body = r.text[:80].replace("\n", " ")
        log(f"[{datetime.now():%H:%M:%S}] 第{i:2d}轮 → {r.status_code} · {body}")
        if r.status_code == 200:
            log("✅ Jev 已恢复（上面这一发是通的）—— 可以跑 120 道了")
            break
    except Exception as e:  # noqa: BLE001
        log(f"[{datetime.now():%H:%M:%S}] 第{i:2d}轮 → {type(e).__name__}: {str(e)[:70]}")
    if i < 15:
        time.sleep(240)
else:
    log("⛔ 15 轮仍未通 —— 明早再看")
