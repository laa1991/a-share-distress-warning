"""出图：ROC 对照 / 分年 AUC / 特征重要性。图里用中文（Windows 自带 Microsoft YaHei）。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
DATA, FIGS = ROOT / "data", ROOT / "docs" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

res = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
z = np.load(DATA / "preds.npz")


def gather(prefix: str):
    ys, ps = [], []
    i = 0
    while f"{prefix}__{i}__y" in z:
        ys.append(z[f"{prefix}__{i}__y"]); ps.append(z[f"{prefix}__{i}__p"]); i += 1
    return np.concatenate(ys), np.concatenate(ps)


# ---- 图 1：ROC ----
fig, ax = plt.subplots(figsize=(6.2, 5.6))
for prefix, label, color in (("A_pit_walkforward", "A 诚实（PIT + 逐年游走）", "#1f77b4"),
                             ("B_lookahead_next_report", "B 偷看未来一期（+半年报）", "#d62728"),
                             ("C_random_split", "C 随机切分（含未来年份）", "#ff7f0e"),
                             ("D_shuffled_label", "D 负对照（标签打乱）", "#7f7f7f"),
                             ("baseline_prev_loss", "基线：上年是否亏损", "#2ca02c")):
    y, p = gather(prefix)
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{label}  AUC={roc_auc_score(y, p):.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("假阳性率"); ax.set_ylabel("真阳性率")
ax.set_title("同一测试集（2022–2025 年测试样本）· 只改「数据用对没有」")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(FIGS / "fig_roc.png", dpi=150); plt.close(fig)

# ---- 图 2：分年 AUC ----
per = {k: res[k]["per_year"] for k in ("A_pit_walkforward", "B_lookahead_next_report")}
years = sorted(int(y) for y in per["A_pit_walkforward"])
x = np.arange(len(years)); w = 0.36
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for off, key, label, color in ((-w / 2, "A_pit_walkforward", "A 诚实", "#1f77b4"),
                               (w / 2, "B_lookahead_next_report", "B 偷看未来一期", "#d62728")):
    ax.bar(x + off, [per[key][str(y)]["auc"] for y in years], w, label=label, color=color)
ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years])
ax.set_ylim(0.80, 1.0); ax.set_ylabel("AUC（同年内）")
ax.set_title("分年稳定性：诚实的模型逐年都站得住")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(FIGS / "fig_by_year.png", dpi=150); plt.close(fig)

# ---- 图 3：特征重要性 ----
imp = res["importance_A"][:15][::-1]
fig, ax = plt.subplots(figsize=(6.6, 5.2))
ax.barh([r["feature"] for r in imp], [r["gain"] for r in imp], color="#4c72b0")
ax.set_xlabel("gain"); ax.set_title("A 臂最后一个折：特征重要性（top 15）")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(FIGS / "fig_importance.png", dpi=150); plt.close(fig)

# ---- 图 4：提前量 → 判别力 ----
lt = json.loads((DATA / "leadtime_curve.json").read_text(encoding="utf-8"))
order = ["① 一季报披露", "② 半年报披露", "③ 三季报披露", "④ 业绩预告公布", "⑤ 年报披露（=答案）"]
x = [lt[k]["lead_days_median"] for k in order]
a = [lt[k]["pooled"]["auc"] for k in order]
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.plot(x, a, "o-", color="#1f77b4", lw=2)
for xi, yi, k in zip(x, a, order):
    ax.annotate(f"{yi:.3f}\n{k.split(' ')[1]}", (xi, yi), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9)
b_arm = res["B_lookahead_next_report"]["pooled"]["auc"]
b_lead = 361 - 121          # 偷看半年报 ≈ 把决策时刻提前 121 天，这里是示意
ax.axhline(b_arm, color="#d62728", ls="--", lw=1.2,
           label=f"B 偷看一期（4 月决策）AUC={b_arm:.3f}")
ax.annotate("偷看一期 ≈ 等到 8 月\n（偷的是提前量，不是模型）", xy=(b_lead / 1.0, b_arm),
            xytext=(0.35, 0.30), textcoords="axes fraction", fontsize=9, color="#d62728")
ax.set_xlabel("距年报披露的天数（越小＝越晚决策、信息越多）")
ax.set_ylabel("AUC")
ax.set_title("提前量 → 判别力：同一个目标，只改决策时刻")
ax.invert_xaxis(); ax.grid(alpha=0.25); ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(FIGS / "fig_leadtime.png", dpi=150); plt.close(fig)

print("图已出：", *(p.name for p in sorted(FIGS.glob("*.png"))))
