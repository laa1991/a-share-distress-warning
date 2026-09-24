"""把这五条「摘录里看不到修正后数字」的补出来（钉住 id，别通读全库）。

复核册的缺陷：当公告用「修正前 / 修正后」两列表格时，原摘录（从前次业绩预告起 380 字）
只截到修正前那一半 ⇒ 裁定不了。这个脚本只把**修正后那几行**捞出来补齐。
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "raw" / "yjyg_rev_pdf"
IDS = ["000668_2024-04-23", "000851_2025-04-19", "000613_2022-04-26",
       "000587_2022-04-19", "000737_2022-03-26"]


def squeeze(t: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n+", " ", t)).strip()


for i in IDS:
    f = D / f"{i}.txt"
    if not f.exists():
        print(f"[{i}] 缺正文")
        continue
    t = squeeze(f.read_text(encoding="utf-8", errors="replace"))
    # 找「修正后」出现的位置（可能多处：表头 + 数据），取最后一处往后再看 520 字
    hits = [m.start() for m in re.finditer(r"修正后", t)]
    seg = t[hits[-1]: hits[-1] + 520] if hits else "（正文里没有『修正后』三字）"
    print(f"\n[{i}] 命中 {len(hits)} 处 · 末处之后：\n  {seg}")
