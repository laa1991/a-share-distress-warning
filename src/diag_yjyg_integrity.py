"""对账：yjyg（业绩预告）目录里的文件是否都完整可解析。

为什么查：这一批取数**被两个进程同时跑过**（我把同一个脚本既起了后台又起了前台），
两边抢同一个 `.csv.tmp` ⇒ 各自报了一批 `PermissionError`。
"文件在"不等于"文件对"—— 逐个解析 + 列名核对 + 行数合理性三件都要过。
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

D = Path(r"C:\dev\finlab-ml\data\raw\yjyg")
EXPECT = {"序号", "股票代码", "股票简称", "预测指标", "业绩变动", "预测数值",
          "业绩变动幅度", "业绩变动原因", "预告类型", "上年同期值", "公告日期"}

bad, ok = [], []
for f in sorted(glob.glob(str(D / "*.csv"))):
    name = Path(f).name
    try:
        d = pd.read_csv(f, dtype={"股票代码": str})
    except Exception as e:  # noqa: BLE001
        bad.append((name, f"parse: {type(e).__name__}: {str(e)[:80]}"))
        continue
    missing = EXPECT - set(d.columns)
    if missing:
        bad.append((name, f"缺列: {sorted(missing)}"))
        continue
    ann = pd.to_datetime(d["公告日期"], errors="coerce")
    ok.append({"file": name, "rows": len(d), "cols": len(d.columns),
               "ann_min": str(ann.min())[:10], "ann_max": str(ann.max())[:10], "ann_null": int(ann.isna().sum())})

print(f"文件数 {len(ok) + len(bad)} | 通过 {len(ok)} | 有问题 {len(bad)}")
if bad:
    print("有问题的文件:")
    for n, why in bad:
        print("  ", n, "->", why)
t = pd.DataFrame(ok)
if len(t):
    print("\n行数分布:", t["rows"].describe()[["min", "25%", "50%", "75%", "max"]].round(1).to_dict())
    print("行数 < 200 的文件:", t[t["rows"] < 200][["file", "rows"]].to_dict("records"))
    print("公告日期为空的行数 > 0 的文件:", t[t["ann_null"] > 0][["file", "rows", "ann_null"]].to_dict("records"))
