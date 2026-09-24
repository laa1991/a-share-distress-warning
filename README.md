# finlab-ml · 上市公司财务风险预警（LightGBM 落地）

> **一句话**：用公开定期报告做一个**真能上线**的风险预警模型 —— 并把「数据用对没有」值多少分量出来。
> 后者是这个作品区别于 demo 的那一格：**同一个模型、同一份数据，只改「谁在什么时刻能看见什么」，分数就变。**

- 任务：`(股票代码, 年份 Y)` 的一季报 → 预测 **Y 年年报是否亏损**（决策时刻＝一季报实际披露日，中位 4 月 27 日；真实前瞻 8–12 个月）
- 数据：A 股 2016Q1–2026Q2，**42 个报告期 · 195,707 行 · 5,252 只股票**（东财批量接口 + 巨潮披露时间表）
- 结果：诚实 AUC **0.910**（逐年 0.89/0.90/0.92/0.93）· top-10% 里 **85%** 当年转亏（lift 3.55）
- 对照：**偷看一期 +3.4 分** · **随机切分 +1.2 分** · 负对照落在 0.5

📄 **结论与红线** → [`docs/结论.md`](docs/结论.md) ｜ 📐 **口径（含数据源的坑）** → [`docs/口径.md`](docs/口径.md) ｜ 📊 图 → `docs/figs/`

---

## 复跑（四条命令，全部确定性、可断点续跑）

```powershell
cd C:\dev\finlab-ml
python scripts/fetch_panel.py --start 2016-03-31 --end 2026-06-30 --workers 4   # ① 四张财报表（约 5 分钟，已落盘则跳过）
python scripts/fetch_disclosure.py --start 2016 --end 2026                      # ② 巨潮实际披露日（约 1 分钟）
python src/build_panel.py                                                       # ③ 拼面板 + 打印源对账/自检
python src/run_experiments.py                                                   # ④ 四格对照 → data/results.json
python src/make_figures.py                                                      # ⑤ 出图（可选）
```

辅助诊断（都是"不解释先量"的产物）：

```powershell
python src/diag_negative_control.py   # 负对照换 5 个种子的分布
python src/diag_seed_variance.py      # A/B 两臂各 3 个种子，判断提升是不是噪声
python src/diag_c_arm.py              # C 臂的 +1.2 分：把 A 限制到同一批样本上重算
```

## 目录

```
scripts/fetch_panel.py          四张东财批量表（业绩报表/资产负债表/利润表/现金流量表）→ data/raw/<表>/<报告期>.csv
scripts/fetch_disclosure.py     巨潮「定期报告预约/实际披露时间表」→ data/raw/disclosure/
src/build_panel.py              拼成 (代码, 报告期) 面板 + 源对账 + 列间恒等式自检 → data/panel.pkl
src/run_experiments.py          四格对照实验（A 诚实 / B 偷看一期 / C 随机切分 / D 负对照）+ 持续性基线
src/make_figures.py             ROC · 分年 AUC · 特征重要性
docs/口径.md                    每个字段的口径、数据源的坑、已知缺陷
docs/结论.md                    读数 + 每条读数的判据 + 不能说的话
data/fetch_log.csv              每次取数调用的行数/耗时（取数层也留账）
```

## 环境

Python 3.12 · `lightgbm 4.7.0` · `scikit-learn 1.9.1` · `akshare 1.18.97` · `pandas 3.0.5` · `matplotlib 3.11.2`

```powershell
python -m pip install lightgbm scikit-learn matplotlib akshare
```
