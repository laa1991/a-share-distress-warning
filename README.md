# finlab-ml · 上市公司财务风险预警（LightGBM 落地）

> **一句话**：用公开定期报告做一个**真能上线**的风险预警模型 —— 并把「数据用对没有」值多少分量出来。
> 后者是这个作品区别于 demo 的那一格：**同一个模型、同一份数据，只改「谁在什么时刻能看见什么」，分数就变。**

- 任务：`(股票代码, 年份 Y)` 的一季报 → 预测 **Y 年年报是否亏损**（决策时刻＝一季报实际披露日，中位 4 月 27 日；真实前瞻 8–12 个月）
- 数据：A 股 2016Q1–2026Q2，**42 个报告期 · 195,707 行 · 5,252 只股票**（东财批量接口 + 巨潮披露时间表）
- 结果：诚实 AUC **0.910**（逐年 0.89/0.90/0.92/0.93）· top-10% 里 **85%** 当年转亏（lift 3.55）
- 对照：**偷看一期 +3.4 分** · **随机切分 +1.2 分** · 负对照落在 0.5
- 提前量曲线（同一个目标，只把决策时刻往后挪）：**4 月 0.910 → 8 月 0.943 → 10 月 0.963 → 次年 1 月（业绩预告）0.999**
  ⇒ **"偷看一期"拿到的 3.4 分 ≈ "等到 8 月"拿到的分**：偷的是提前量，不是模型强弱
- 信号什么时候亮（同一批公司的轨迹）：4 月 **35.6%** 的"真会亏"公司已进前 10%（误报 2.0%），10 月升到 **40.1%**（误报 0.6%）；
  **43.8% 的亏损公司从未亮过灯** ⇒ 预警的天花板不是模型，是基本面什么时候开始变难看

📄 **结论与红线** → [`docs/结论.md`](docs/结论.md) ｜ 📐 **口径（含数据源的坑）** → [`docs/口径.md`](docs/口径.md) ｜ 📊 图 → `docs/figs/`
🧾 **作品一页纸（A4 单页 · 可直接当附件）** → `docs/作品一页纸.pdf`（中文）· `docs/作品一页纸-EN.pdf`（English）
　（源 `docs/作品一页纸.html` / `-EN.html`；**Edge headless 出 PDF**，无 pandoc 依赖。两版都是 1 页、各含两张图 —— 页数是判据，别靠眼估。）

---

## 复跑（四条命令，全部确定性、可断点续跑）

```powershell
cd C:\dev\finlab-ml
python scripts/fetch_panel.py --start 2016-03-31 --end 2026-06-30 --workers 4   # ① 四张财报表（约 5 分钟，已落盘则跳过）
python scripts/fetch_disclosure.py --start 2016 --end 2026                      # ② 巨潮实际披露日（约 1 分钟）
python scripts/fetch_yjyg.py --start 2015 --end 2026                            # ③ 业绩预告（约 3 分钟）
python scripts/fetch_yjkb.py --start 2015 --end 2026                            # ③b 业绩快报（可选：默认不纳入模型，见下）
python src/build_panel.py                                                       # ④ 拼面板 + 打印源对账/自检
python src/run_experiments.py                                                   # ⑤ 四格对照 → data/results.json
python src/run_leadtime_curve.py                                                # ⑥ 提前量曲线 → data/leadtime_curve.json + 逐样本分数
python src/analyze_signal_timeline.py                                           # ⑦ 信号什么时候亮 → data/signal_timeline.json + 图
python src/make_figures.py                                                      # ⑧ 出图（可选）
```

⚠️ 别把同一个抓取脚本**同时**起两个进程（我在这条线上真的踩了：后台一个、前台又一个 ⇒ 两条进程抢同一个 `.csv.tmp`，
报 `PermissionError: WinError 32`，4 次调用白跑 —— 重跑会自动补，但那是白等）。

辅助诊断（都是"不解释先量"的产物）：

```powershell
python src/diag_negative_control.py   # 负对照换 5 个种子的分布
python src/diag_seed_variance.py      # A/B 两臂各 3 个种子，判断提升是不是噪声
python src/diag_c_arm.py              # C 臂的 +1.2 分：把 A 限制到同一批样本上重算
python src/diag_yjyg_integrity.py     # 预告文件完整性（那两个进程抢同一 .tmp 之后做的对账）
python src/diag_yjkb_dates.py         # 快报「公告日期」为什么不能用（67.8% 的行为负）
```

## 重出一页纸的 PDF（Edge headless，无需 pandoc）

```powershell
$edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
foreach ($n in '作品一页纸','作品一页纸-EN') {
  & $edge --headless=new --disable-gpu --no-pdf-header-footer `
      --print-to-pdf="docs\$n.pdf" ([System.Uri]::new("$PWD\docs\$n.html").AbsoluteUri)
}
```

⚠️ 两个坑（都踩过）：① 判「是不是一页」看 **PDF 里的 `/Type /Page` 个数**，别靠眼看截图；
② **量版面时变体文件必须放在 `docs/` 里** —— 放 `%TEMP%` 时图片的相对路径解析不到，图没加载 ⇒ 版面变短 ⇒ **量出来是假绿**（我因此白测了两轮）。
一页纸在 9.05pt 正文 + 两张图 180mm（中文）/ 172mm（英文）下刚好一页；动字号就要重新量页数。

> **仓库里没有 `data/`**（取数产物与面板共 ~270 MB，`gitignore` 掉了）。按上面五步跑一遍即可重建；
> 每次取数都在 `data/fetch_*.csv` 留一行账（表 / 报告期 / 行数 / 耗时 / 状态）。

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
