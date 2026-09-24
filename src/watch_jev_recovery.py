"""Jev 恢复观察器：打一发 jev-1.13 记一行；通了就退出；打满轮数也退出。

用法：`python src/watch_jev_recovery.py [间隔分钟] [轮数]`（默认 4 / 15 ⇒ 约 1 小时）。
产物：`data/judge_arena/jev_recovery.log`（**只追加**：时刻 + 状态码 + 原文前 80 字）。
它回答的是明早那个问题：**Jev 今晚恢复过没有**。
⚠️ 2026-09-25 01:48 那次重启把后台 job 全杀了（重启后 `job_list` 为空）⇒ 观察器要**重启后重挂**；
   日志是追加的 ⇒ 不丢已有记录，只会缺一段（所以每一轮都带时刻）。
"""
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

INTERVAL_MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
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


log(f"\n=== 观察开始 {datetime.now():%Y-%m-%d %H:%M:%S}（每 {INTERVAL_MIN} 分钟一发，最多 {ROUNDS} 轮）")
for i in range(1, ROUNDS + 1):
    try:
        r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=PAY, timeout=120)
        body = r.text[:80].replace("\n", " ")
        log(f"[{datetime.now():%H:%M:%S}] 第{i:2d}轮 → {r.status_code} · {body}")
        if r.status_code == 200:
            log("✅ Jev 已恢复（上面这一发是通的）—— 可以跑 120 道了")
            break
    except Exception as e:  # noqa: BLE001
        log(f"[{datetime.now():%H:%M:%S}] 第{i:2d}轮 → {type(e).__name__}: {str(e)[:70]}")
    if i < ROUNDS:
        time.sleep(INTERVAL_MIN * 60)
else:
    log(f"⛔ {ROUNDS} 轮仍未通 —— 明早再看")
# ⚠️ 2026-09-25 02:5x 自纠：上面两处（`sleep(240)` 与 `range(1, 16)`）**都写死过**，
#    而我当时以为"改了 argv 解析就等于改了行为"—— 于是"每 10 分钟 × 36 轮"实际跑成"每 4 分钟 × 15 轮"。
#    同一个形状我这一晚撞了两次（另一处是 sleep）。⇒ **改参数解析之后，必须回头 grep 那两个常量还在不在。**
