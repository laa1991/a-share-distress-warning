"""抓「业绩预告修正公告」的 PDF 正文 —— 标题不带数字，**方向（好→坏 / 坏→好）只能从正文读**。

地址规律同 `fetch_st_pdfs.py`：`http://static.cninfo.com.cn/finalpage/<公告日>/<announcementId>.PDF`
（announcementId 就在我们已存下的 `公告链接` 里）。

产物：`data/raw/yjyg_rev_pdf/<code>_<date>.pdf` + 同名 `.txt`（PyMuPDF 抽的正文）
可断点续跑（两个文件都在就跳过）· 下载先 `.tmp` 再原子改名。
⚠️ 单进程跑（本机 py_mini_racer / 线程实测会崩）。

用法: python fetch_yjyg_rev_pdfs.py [--limit N] [--from-year 2016]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "yjyg_rev"
OUT = ROOT / "data" / "raw" / "yjyg_rev_pdf"
LOG = ROOT / "data" / "fetch_yjyg_rev_pdf_log.csv"
HEAD = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Referer": "http://www.cninfo.com.cn/"}
EM = re.compile(r"</?em>")
YEAR_IN_TITLE = re.compile(r"(20\d{2})\s*年度")


def collect(from_year: int) -> pd.DataFrame:
    rows = []
    for f in sorted(RAW.glob("*.csv")):
        d = pd.read_csv(f, dtype=str)
        for _, r in d.iterrows():
            title = EM.sub("", str(r.get("公告标题", "")))
            if "半年度" in title or "季度" in title or not YEAR_IN_TITLE.search(title):
                continue                                   # 只要年度报告期的修正
            link = str(r.get("公告链接", ""))
            m, m2 = re.search(r"announcementId=(\d+)", link), re.search(r"announcementTime=([\d-]+)", link)
            if not (m and m2):
                continue
            fy = int(YEAR_IN_TITLE.search(title).group(1))
            if fy < from_year:
                continue
            rows.append({"code": str(r["代码"]).zfill(6), "date": m2.group(1), "aid": m.group(1),
                         "title": title, "fy": fy})
    d = pd.DataFrame(rows).drop_duplicates(subset=["code", "date"])
    return d.sort_values(["date", "code"]).reset_index(drop=True)


def work(code: str, date: str, aid: str, tries: int = 3) -> dict:
    pdf, txt = OUT / f"{code}_{date}.pdf", OUT / f"{code}_{date}.txt"
    if pdf.exists() and txt.exists():
        return {"status": "cached", "chars": len(txt.read_text(encoding="utf-8", errors="replace"))}
    status, chars, err = "failed", 0, ""
    for i in range(tries):
        try:
            r = requests.get(f"http://static.cninfo.com.cn/finalpage/{date}/{aid}.PDF", headers=HEAD, timeout=40)
            if r.status_code != 200 or not r.content.startswith(b"%PDF"):
                raise ValueError(f"HTTP {r.status_code} magic={r.content[:5]!r}")
            tmp = pdf.with_suffix(".pdf.tmp")
            tmp.write_bytes(r.content)
            tmp.replace(pdf)
            import pymupdf as fitz
            with fitz.open(pdf) as doc:
                text = "\n".join(p.get_text() for p in doc)
            ttmp = txt.with_suffix(".txt.tmp")
            ttmp.write_text(text, encoding="utf-8")
            ttmp.replace(txt)
            status, chars = "ok", len(text)
            break
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:80]}"
            time.sleep(1 + i)
    return {"status": status, "chars": chars, "err": err}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--from-year", type=int, default=2016)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    t = collect(args.from_year)
    if args.limit:
        t = t.head(args.limit)
    print(f"要抓 {len(t)} 条年度修正公告（会计年度 ≥ {args.from_year}）", flush=True)
    rows, t0 = [], time.time()
    for i, r in enumerate(t.itertuples(index=False), 1):
        res = work(r.code, r.date, r.aid)
        rows.append({"code": r.code, "date": r.date, "fy": r.fy, "title": r.title[:60], **res})
        if i % 25 == 0 or i == len(t):
            ok = sum(1 for x in rows if x["status"] == "ok")
            ca = sum(1 for x in rows if x["status"] == "cached")
            print(f"  {i}/{len(t)} · ok {ok} · cached {ca} · 用时 {time.time()-t0:.0f}s", flush=True)
    log = pd.DataFrame(rows)
    log.to_csv(LOG, index=False, encoding="utf-8-sig")
    got = (log["status"] != "failed").sum()
    print(f"\n成功/缓存 {got}/{len(log)} · 正文中位 {int(log.loc[log['chars']>0,'chars'].median()) if got else 0} 字")
    print("账 ->", LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
