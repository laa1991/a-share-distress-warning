# 上市公司财务风险预警（LightGBM 落地）

> **一句话**：用公开定期报告做一个**真能上线**的风险预警模型 —— 并把「数据用对没有」值多少分量出来。
> 后者是这个作品区别于 demo 的那一格：**同一个模型、同一份数据，只改「谁在什么时刻能看见什么」，分数就变。**
>
> **这个仓库里有什么 / 没有什么**：代码 · 文档 · 图，以及**可复算的读数**（`data/*.json`、`data/*.csv`，含每步的取数账）；
> **不含原始抓取数据**（`data/raw/` 与中间缓存被 gitignore，约 270 MB）—— 按下面「复跑」一节逐条跑即可重建。
> 所有读数的**口径与判据**在 [`docs/口径.md`](docs/口径.md)，**结论、边界与"不能说的话"**在 [`docs/结论.md`](docs/结论.md) 末尾。
> 许可：MIT（见 `LICENSE`）。数据来源为公开接口（东方财富 / 巨潮资讯 / 新浪财经），仅用于方法论演示，不构成任何投资建议。

**从这里读下去（十四份文档 + 一页纸）**：
📄 **结论与红线** → [`docs/结论.md`](docs/结论.md) ｜ 📐 **口径（含数据源的坑）** → [`docs/口径.md`](docs/口径.md) ｜ 📊 图 → `docs/figs/`
🧾 **作品一页纸（A4 单页 · 可直接当附件）** → [`docs/作品一页纸.pdf`](docs/作品一页纸.pdf)（中文）· [`docs/作品一页纸-EN.pdf`](docs/作品一页纸-EN.pdf)（English）
🎯 **产品视角（愿意提前多久 → 拿什么样的名单）** → [`docs/产品视角.md`](docs/产品视角.md) ｜ 图 `docs/figs/fig_product.png`
⏱ **候选设计：业绩预告 → 滚动更新链路** → [`docs/设计-业绩预告滚动链.md`](docs/设计-业绩预告滚动链.md)：这一格的分不是模型本事（**输入里写着答案**：0.9985），值钱的是**提前量中位 86 天 + 会亏的公司 99.25% 都会发预告**
🧪 **探索：判断器的效率（GBDT vs Jev vs LLM）** → [`docs/探索-判断器的效率（GBDT vs Jev vs LLM）.md`](docs/探索-判断器的效率（GBDT%20vs%20Jev%20vs%20LLM）.md)：四条轴（延迟/成本/复算性/**校准**）+ 我们这边的实测（**单条 p50 2.0 ms · 29.97 万行/秒 · 逐位可复算**）与**自己量的 LLM 基线**（置信度中位 **1.00**、不确定带 0.0%）
🔧 **报表外事件（诉讼/担保）值得开一格吗** → [`docs/报表外事件-值得开一格吗.md`](docs/报表外事件-值得开一格吗.md)：单看差 17.6 个点；加进模型只值 +0.14 分 AUC ⇒ **不开**
🏷 **换标签：被实施风险警示（\*ST/ST）** → [`docs/换标签-被实施风险警示.md`](docs/换标签-被实施风险警示.md)：AUC 0.910 → **0.813**，但前 10% 名单**抓住 49% 的戴帽事件**、lift **4.89×**
🪜 **两级名单（会亏 → 会戴帽）** → [`docs/两级名单-亏损与戴帽.md`](docs/两级名单-亏损与戴帽.md) ｜ 🚪 **名单外那 49 个** → [`docs/名单外的戴帽样本（49个）.md`](docs/名单外的戴帽样本（49个）.md)
❓ **为什么戴帽（211 份公告正文归类）** → [`docs/为什么戴帽（原因分布）.md`](docs/为什么戴帽（原因分布）.md) ｜ 🎯 **只能盯一件事，盯哪一类** → [`docs/盯哪一类（戴帽原因×名单）.md`](docs/盯哪一类（戴帽原因×名单）.md)
🎓 **三个教具** → [`docs/教学-LightGBM（八行走一遍）.md`](docs/教学-LightGBM（八行走一遍）.md) · [`docs/教学-一家真实公司走一遍.md`](docs/教学-一家真实公司走一遍.md) · [`docs/教学-LightGBM 与 AI 的区别.md`](docs/教学-LightGBM%20与%20AI%20的区别.md)

## 关键读数（一屏看完）

- 任务：`(股票代码, 年份 Y)` 的一季报 → 预测 **Y 年年报是否亏损**（决策时刻＝一季报实际披露日，中位 4 月 27 日；真实前瞻 8–12 个月）
- 数据：A 股 2016Q1–2026Q2，**42 个报告期 · 195,707 行 · 5,252 只股票**（东财批量接口 + 巨潮披露时间表）
- 结果：诚实 AUC **0.910**（逐年 0.89/0.90/0.92/0.93）· top-10% 里 **85%** 当年转亏（lift 3.55）
- 对照：**偷看一期 +3.4 分** · **随机切分 +1.2 分** · 负对照落在 0.5
- 提前量曲线（同一个目标，只把决策时刻往后挪）：**4 月 0.910 → 8 月 0.943 → 10 月 0.963 → 次年 1 月（业绩预告）0.999**
  ⇒ **"偷看一期"拿到的 3.4 分 ≈ "等到 8 月"拿到的分**：偷的是提前量，不是模型强弱
- 信号什么时候亮（同一批公司的轨迹）：4 月 **35.6%** 的"真会亏"公司已进前 10%（误报 2.0%），10 月升到 **40.1%**（误报 0.6%）；
  **43.8% 的亏损公司从未亮过灯** ⇒ 预警的天花板不是模型，是基本面什么时候开始变难看
- 产品视角（愿意提前多久 → 拿到什么名单）：前 10% 名单精确率 84.8→95.4%、覆盖率 **35.9→41.4%**；
  **名单放到前 20% 覆盖率升到 61.9%** ⇒ 召回低是名单大小的决定，不是模型的毛病
- 数值层双源对账（东财 vs 新浪，**全量 5,247 只 / 19.4 万对**）：营业收入 **99.73%** 精确相等；
  净利润**同名只有 23.18%**、**概念对齐后 99.60%** ⇒ 东财的「净利润」是归母、新浪的是含少数股东；
  对齐后剩 **0.22% 是重述率**，且 2016 年 121 条 → 2024 年 2 条（老数据才会被追溯调整）
- 第二批科目（资产负债 + 现金流，全量）：货币资金 **99.85%** · 应收账款 99.63% · 存货 99.77% · 总资产 99.15% · 总负债 99.34% 精确相等；
  **「股东权益」对含少数股东的 99.42%、对归母只有 22.97%** ⇒ **歧义是逐科目的**（同一个东财，净利润取归母、权益取含少数股东）

---

## 复跑（主链 11 条命令 + 判据 17 条，全部确定性、可断点续跑）

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
python src/make_product_figure.py                                               # ⑧ 产品视角（提前量 × 分数 × 覆盖面）→ data/product_curve.json + 图
python src/make_figures.py                                                      # ⑨ 出图（可选）
python src/make_figures_en.py                                                   # ⑨b 英文版两张图（英文一页纸用）
```

⚠️ 别把同一个抓取脚本**同时**起两个进程（我在这条线上真的踩了：后台一个、前台又一个 ⇒ 两条进程抢同一个 `.csv.tmp`，
报 `PermissionError: WinError 32`，4 次调用白跑 —— 重跑会自动补，但那是白等）。

辅助诊断（判据 **17 条**，都是"不解释先量"的产物）：

```powershell
python src/diag_negative_control.py   # 负对照换 5 个种子的分布
python src/diag_seed_variance.py      # A/B 两臂各 3 个种子，判断提升是不是噪声
python src/diag_c_arm.py              # C 臂的 +1.2 分：把 A 限制到同一批样本上重算
python src/diag_yjyg_integrity.py     # 预告文件完整性（那两个进程抢同一 .tmp 之后做的对账）
python src/diag_yjkb_dates.py         # 快报「公告日期」为什么不能用（67.8% 的行为负）
python src/diag_value_crosscheck.py --n 40   # 数值层双源对账：东财 vs 新浪（净利润同名不同物：归母 vs 含少数股东）
python scripts/fetch_cg_events.py            # 报表外事件取数（巨潮 诉讼/担保，按季度窗口，44×2 次调用）
python src/diag_offstatement_events.py       # 探针：事件单看分得开吗（分得开 ≠ 加得进）
python src/diag_events_ablation.py --seeds 3 # 消融：真加进去值多少分（+0.0014 ± 0.0004）
python scripts/fetch_st_events.py            # 取「被实施风险警示」公告（巨潮检索 keyword=风险警示，按年 12 次调用）
python src/st_label.py                       # 做成可时点化的 ST 标签（952 次戴帽 / 606 家公司）
python src/run_experiments_st.py             # 换标签重跑（A / 基线 / 负对照）
python src/diag_st_negative_control.py --seeds 5   # ★ 负对照：pooled 能骗到 0.40–0.56，macro 才回到 0.5
python src/st_sample_events.py                     # 列出要取正文的戴帽公告（按组去重，211 条）
python scripts/fetch_st_pdfs.py --only data/st_event_targets.csv  # 取公告 PDF + 抽正文（可断点续跑）
python src/classify_st_reasons.py                  # 归类「为什么戴帽」+ 两组对照（每条留证据句）
python src/diag_reason_channels.py                 # ★ 戴帽原因 × 名单：三类各抓到多少（该盯哪一类）
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
一页纸在 **8.6pt 正文 + 两张图**（页边距 9mm/11mm）下刚好一页；**动字号/动页边距就要重新量页数**
（2026-09-24 晚：加了两条新读数后先撑到 2 页，收紧字号 + 页边距才回到 1 页 —— 收紧后**必须再验一次"内容没被裁掉"**：
抽 PDF 文字，确认末行与几个新数字还在，见 `src/` 之外的核对脚本或手工 PyMuPDF 一行）。

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
