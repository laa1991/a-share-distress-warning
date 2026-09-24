"""为人工裁定做一份**精简复核册**：每条只留"裁定要看的那几句"（不整篇贴，省上下文）。

- `rev_dir`：从「前次业绩预告情况 / 一、预计的本期业绩情况」起取 ~380 字（含前次区间）+ 从「修正后的预计业绩」起取 ~260 字；
- `st_reason`：从戴帽原因锚点（「…的原因」「触及《…》」）起取 ~360 字。

⚠️ 这套摘录是给**人读**的，裁定在摘录上做 —— 与规则的读法（正则取区间 + 中点比较）**不是同一条路**，
但也不是全文通读。这一点必须写进读数里。

用法: python make_gold_review.py
产物: data/judge_arena/<task>_gold_review.md（编号条目，供人读）+ <task>_gold_labels.csv（空列等填）
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARENA = ROOT / "data" / "judge_arena"

REV_HEAD = re.compile(r"前次业绩预告情况|一、\s*预计的本期业绩情况")
REV_AFTER = re.compile(r"修正后的预计业绩|修正后预计业绩")
ST_ANCHOR = re.compile(r"的原因|触及《|被实施[^。]{0,12}警示的原因|占用|冻结|内控|处罚|意见")
# 样板段：这些词一出现，后面的内容对"触发原因"没有信息量（v1 的 7 条判不了全落在这里）
BOILER = re.compile(r"董事会关于争取撤销|争取撤销[^。]{0,8}警示的意见|风险提示|联系方式|特此公告|公司将持续关注")


def squeeze(t: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n+", " ", t)).strip()


def rev_excerpt(t: str) -> str:
    t = squeeze(t)
    m1 = REV_HEAD.search(t)
    a = t[m1.start(): m1.start() + 380] if m1 else t[:380]
    m2 = REV_AFTER.search(t)
    b = t[m2.start(): m2.start() + 260] if m2 else ""
    # 顺带把勾选框那一行带上（它是最直接的证据）
    tick = re.search(r"(亏损|扭亏为盈|同向上升|同向下降|其他)\s*(√|✓|■|●)|[√✓■●]\s*(亏损|扭亏为盈|同向上升|同向下降)", t)
    return (a + "  ‖  " + b + (f"  ‖勾选:{tick.group(0)}" if tick else "")).strip()


def st_excerpt(t: str) -> str:
    """v2（2026-09-25 修）：v1 从**第一个**命中词起取 360 字，结果 7/60 落在「拟采取的措施」这类样板段、
    另有 5 条把「…意见的审计报告」的主体切在切口另一侧（而「内控 vs 财务报表」正是 A/B 的分界）。

    修法两件：
    ① **先切掉样板段**（董事会措施 / 风险提示 / 联系方式 / 特此公告）；
    ② **锚点按信息量排序**（适用情形 > 触及《 > 根据《 > 实施…警示），并**向锚点前后各取一段**
       （决定性限定词常在锚点之前，v1 只往后取 ⇒ 主语被吃掉）。
    """
    t = squeeze(t)
    cuts = [m.start() for m in BOILER.finditer(t)]
    head = t[: min(cuts)] if cuts else t
    for pat in (r"适用情形", r"触及《", r"根据《", r"实施退市风险警示", r"实施其他风险警示", r"被实施"):
        m = re.search(pat, head)
        if m:
            a = m.start()
            return head[max(0, a - 280): a + 340]
    return head[:400]


def main() -> int:
    for task, fn in (("rev_dir", rev_excerpt), ("st_reason", st_excerpt)):
        f = ARENA / f"{task}_gold_template.csv"
        if not f.exists():
            print(f"缺 {f}（先跑 judge_arena.py --dry --gold 60）")
            continue
        d = pd.read_csv(f)
        lines = [f"# {task} 人工裁定复核册（{len(d)} 条）",
                 "",
                 "> 每条一行：`编号 · id`，下一行是摘录。裁定在摘录上做。",
                 ""]
        for i, r in d.iterrows():
            lines.append(f"**[{i+1}] {r['id']}**")
            lines.append("")
            lines.append("```")
            lines.append(fn(str(r["text"]))[:900])
            lines.append("```")
            lines.append("")
        out = ARENA / f"{task}_gold_review.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        labels = d[["id", "truth_rule"]].copy()
        labels["人工裁定"] = ""
        lab_path = ARENA / f"{task}_gold_labels.csv"
        # ⚠️ 2026-09-25 撞过的坑：这个脚本无条件下写标签表，把上一轮**已经填好的 60 条裁定清空了**
        #    （救回来的路径：子代理的构建脚本里按 id 写死了标签）。⇒ 有内容就**另存**，绝不覆盖。
        if lab_path.exists():
            old = pd.read_csv(lab_path, dtype=str).fillna("")
            filled = old["人工裁定"].astype(str).str.strip()
            if (filled != "").any():
                alt = ARENA / f"{task}_gold_labels.new.csv"
                labels.to_csv(alt, index=False, encoding="utf-8-sig")
                print(f"⚠️ {lab_path.name} 里已有 {int((filled != '').sum())} 条裁定 —— **不覆盖**，"
                      f"新表另存为 {alt.name}")
                continue
        labels.to_csv(lab_path, index=False, encoding="utf-8-sig")
        print(f"{task}: 复核册 {len(d)} 条 -> {out.name}（{out.stat().st_size/1024:.0f} KB）· 标签表 -> {task}_gold_labels.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
