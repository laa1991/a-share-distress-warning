"""从「业绩预告修正公告」正文里读出**修订方向**：好→坏 / 坏→好（v2：按行取数 + 双读数交叉验证）。

标题不带数字，只能读正文。正文的固定结构（实测）：
    前次业绩预告情况：… 预计…归属于上市公司股东的净利润…区间为 10,260万元—17,100万元。
    3.修正后的预计业绩  … 归属于上市公司股东的净利润 … 盈利：7,980 万元至14,820 万元
    （同一段还有勾选框：亏损 / 扭亏为盈 / 同向上升 / √同向下降 / 其他）

**v1 的教训**：在整段里"取所有数字求平均"会把**不同科目**（营业收入、扣非、每股收益）混进来，
抽出来的前后对比有假样本（例：`-3,900万 → 10,319万`，而两段文本都写着 20000）。
⇒ v2 改成：**只在含「归属…净利润」的那一行里取数**，且**只用区间**（两个数）；
并**用勾选框当第二个独立读数**，两读法对不上的样本单独列出来 —— 覆盖率与一致率都照实报。

用法: python classify_yjyg_rev_dir.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TXT = ROOT / "data" / "raw" / "yjyg_rev_pdf"
LINE_KEY = re.compile(r"归属|净利润")
BAD_KEY = re.compile(r"净资产|所有者权益|股东权益|营业收入|总资产|负债|每股|现金流")  # 邻近科目：退一步搜索时要排除
RANGE = re.compile(r"([\-－]?\d[\d,]*(?:\.\d+)?)\s*万元?\s*[—–~～至\-]\s*([\-－]?\d[\d,]*(?:\.\d+)?)\s*万元?")
NEG_HINT = re.compile(r"亏损|负值|下滑")
TICK = {"亏损": "亏损", "扭亏为盈": "扭亏为盈", "同向上升": "同向上升",
        "同向下降": "同向下降", "其他": "其他"}


def wan(s: str) -> float:
    return float(s.replace(",", "").replace("－", "-").replace("—", "-"))


def pick_range(seg: str) -> tuple[float | None, str]:
    m = RANGE.search(seg)
    if not m:
        return None, ""
    a, b = wan(m.group(1)), wan(m.group(2))
    val = (a + b) / 2
    if a < 0 or b < 0 or NEG_HINT.search(seg[:m.start()][-24:]):
        val = -abs(val)
    return val, m.group(0)


def ticked(text: str) -> str | None:
    for label in TICK:
        # 勾选在正文里表现为 √ 紧跟标签，或 □ 未勾；实测用 √/■/● 之一
        if re.search(r"[√✓✔■●]\s*" + label, text) or re.search(label + r"\s*[√✓✔■●]", text):
            return label
    return None


def main() -> int:
    files = sorted(TXT.glob("*.txt"))
    if not files:
        print("还没有正文（先跑 scripts/fetch_yjyg_rev_pdfs.py）")
        return 1
    rows = []
    for f in files:
        code, date = f.stem.split("_")[0], f.stem.split("_")[1]
        t = re.sub(r"[ \t]+", " ", f.read_text(encoding="utf-8", errors="replace"))
        lines = [l.strip() for l in t.splitlines() if l.strip()]
        # 找"修正后"的位置（按行），把行分成前/后两半
        cut = None
        for i, l in enumerate(lines):
            if "修正后的预计业绩" in l or "修正后预计业绩" in l or re.match(r"^3[\.、]", l):
                cut = i
                break
        if cut is None:
            # ⚠️ 2026-09-25 修：找不到"修正后"的分界时，两段会退化成**同一段**，
            #    于是 v1 恒等于 v0 ⇒ 必然读出「不变」。实测这是「不变」桶的主要来源（金标 30 条里 10 条）。
            #    现在改成**判不了**（None），让"抽不到"长得像"抽不到"，而不是像"不变"。
            prev_lines, after_lines = [], []
        else:
            prev_lines, after_lines = lines[:cut], lines[cut:]

        def grab(seg_lines: list[str]) -> tuple[float | None, str]:
            for l in seg_lines:                      # 优先含"归属/净利润"且带区间的行
                if LINE_KEY.search(l):
                    val, seg = pick_range(l)
                    if val is not None:
                        return val, seg
            for l in seg_lines:                      # 退一步：任意带区间的行（**排除**净资产/营收这类邻近科目）
                if BAD_KEY.search(l):
                    continue
                val, seg = pick_range(l)
                if val is not None:
                    return val, seg
            return None, ""

        v0, s0 = grab(prev_lines)
        v1, s1 = grab(after_lines)
        if s0 and s1 and s0 == s1:
            # ⚠️ 同一段区间被读了两遍 ⇒ 是"没取到修正后"，不是"没变"（2026-09-25 金标抓出来的）
            v0, v1 = None, None
        if v0 is not None and v1 is not None and abs(v0 - v1) < 1e-9:
            # ⚠️ 抽出来的中点数一模一样：在一份**修正公告**里，"数值完全没变"几乎总是**同一句被读了两遍**
            #    （换行/破折号差异会让上面那条字符串比较漏过）。真正的"不变"另有 35 条，
            #    它们靠勾选框与原因说明能认出来 ⇒ 这里宁可不判，也不报一个假的"不变"。
            v0, v1 = None, None
        tk = ticked(t)
        rows.append({"code": code, "date": date, "prev": v0, "after": v1,
                     "段_前": s0[:48], "段_后": s1[:48], "勾选": tk,
                     "方向": (None if (v0 is None or v1 is None) else
                              ("坏→更坏" if v1 < v0 else ("好→更好" if v1 > v0 else "不变"))),
                     "翻面": (None if (v0 is None or v1 is None) else int((v0 < 0) != (v1 < 0)))})
    d = pd.DataFrame(rows)
    ok = d[d["prev"].notna() & d["after"].notna()].copy()
    print(f"正文 {len(d)} 份 · **两段都抽到区间的 {len(ok)} 份（覆盖率 {len(ok)/len(d):.1%}）**")
    print("\n【读数① 数值对比】")
    print(ok["方向"].value_counts().to_string())
    flips = ok[ok["翻面"] == 1]
    print(f"  盈亏翻面 {len(flips)} 条（{len(flips)/max(len(ok),1):.1%}）")
    print("\n【读数② 勾选框（独立于数值）】")
    print(d["勾选"].value_counts(dropna=False).to_string())
    # 交叉：两读法一致吗
    bad = ok[(ok["翻面"] == 1) & (ok["勾选"].isin(["同向上升", "同向下降"]))]
    agree = ok[~((ok["翻面"] == 1) & (ok["勾选"].isin(["同向上升", "同向下降"])))]
    print(f"\n【两读法交叉】翻面但勾了'同向*'的 {len(bad)} 条（不一致，抽 3 条看）")
    for _, r in bad.head(3).iterrows():
        print(f"   {r['code']} {r['date']} {r['prev']:,.0f}→{r['after']:,.0f} 勾选={r['勾选']} [{r['段_前']}] / [{r['段_后']}]")
    print(f"  一致率 {(1 - len(bad)/max(len(ok),1)):.1%}")
    d.to_csv(ROOT / "data" / "yjyg_rev_direction.csv", index=False, encoding="utf-8-sig")
    out = {"n": int(len(d)), "coverage": round(float(len(ok) / len(d)), 4),
           "dir": {str(k): int(v) for k, v in ok["方向"].value_counts().items()},
           "flips": int(len(flips)), "flip_rate": round(float(len(flips) / max(len(ok), 1)), 4),
           "tick": {str(k): int(v) for k, v in d["勾选"].value_counts(dropna=False).items()},
           "cross_inconsistent": int(len(bad))}
    (ROOT / "data" / "yjyg_rev_direction.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n读数 -> data/yjyg_rev_direction.json · 明细 -> data/yjyg_rev_direction.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
