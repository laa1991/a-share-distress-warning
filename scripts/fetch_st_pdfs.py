"""抓「戴帽公告」的 PDF 正文 —— 用来回答「为什么戴帽」（标题答不了，正文才带原因）。

地址规律（实测）：`http://static.cninfo.com.cn/finalpage/<公告日>/<announcementId>.PDF`
（announcementId 就在我们已存下的 `公告链接` 里 ⇒ 不用再走检索 API）。

产物：`data/raw/st_pdf/<code>_<date>.pdf`（原始 PDF）+ `<code>_<date>.txt`（抽出的正文，PyMuPDF）
可断点续跑：两个文件都在就跳过；下载先写 `.tmp` 再原子改名。

用法: python fetch_st_pdfs.py [--workers 4] [--limit N]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "st_events"
OUT = ROOT / "data" / "raw" / "st_pdf"
sys.path.insert(0, str(ROOT / "src"))
from st_label import is_entry  # noqa: E402

HEAD = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Referer": "http://www.cninfo.com.cn/"}
LOG: list[dict] = []


def collect() -> pd.DataFrame:
    rows = []
    for f in sorted(RAW.glob("*.csv")):
        d = pd.read_csv(f, dtype={"代码": str})
        if not len(d):
            continue
        for _, r in d.iterrows():
            title = re.sub(r"</?em>", "", str(r.get("公告标题", "")))
            link = str(r.get("公告链接", ""))
            m, m2 = re.search(r"announcementId=(\d+)", link), re.search(r"announcementTime=([\d-]+)", link)
            if not (m and m2) or not is_entry(title):
                continue
            rows.append({"code": r["代码"].zfill(6), "date": m2.group(1), "aid": m.group(1), "title": title})
    d = pd.DataFrame(rows).drop_duplicates(subset=["code", "date"])
    return d.sort_values(["code", "date"]).reset_index(drop=True)


def work(code: str, date: str, aid: str, title: str, tries: int = 3) -> None:
    pdf, txt = OUT / f"{code}_{date}.pdf", OUT / f"{code}_{date}.txt"
    status, chars = "cached", len(txt.read_text(encoding="utf-8")) if txt.exists() else 0
    if not (pdf.exists() and txt.exists()):
        status = "failed"
        for i in range(tries):
            try:
                r = requests.get(f"http://static.cninfo.com.cn/finalpage/{date}/{aid}.PDF",
                                 headers=HEAD, timeout=40)
                if r.status_code != 200 or not r.content.startswith(b"%PDF"):
                    raise ValueError(f"HTTP {r.status_code} magic={r.content[:5]!r}")
                tmp = pdf.with_suffix(".pdf.tmp")
                tmp.write_bytes(r.content)
                tmp.replace(pdf)
                import pymupdf  # noqa: PLC0415
                doc = pymupdf.open(pdf)
                text = "\n".join(p.get_text() for p in doc)
                doc.close()
                txt.write_text(text, encoding="utf-8")
                status, chars = "ok", len(text)
                break
            except Exception as exc:  # noqa: BLE001
                if i == tries - 1:
                    status = f"failed:{type(exc).__name__}"
                else:
                    time.sleep(0.8 * (i + 1))
    LOG.append({"code": code, "date": date, "title": title[:60], "chars": chars, "status": status})
    print(f"  {code} {date} {chars:6d} 字 [{status}]", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="只取这个 CSV（列 code,date）里的公告 —— 默认取全部新戴帽事件")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ev = collect()
    if args.only:
        want = pd.read_csv(args.only, dtype={"code": str})
        want["code"] = want["code"].str.zfill(6)
        ev = ev.merge(want[["code", "date"]].drop_duplicates(), on=["code", "date"], how="inner")
    if args.limit:
        ev = ev.head(args.limit)
    print(f"新的戴帽公告 {len(ev)} 条（按 code+date 去重）")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda r: work(**r), ev.to_dict("records")))

    log = pd.DataFrame(LOG)
    log.to_csv(ROOT / "data" / "fetch_st_pdf_log.csv", index=False, encoding="utf-8-sig")
    bad = log[log["status"].str.startswith("failed")]
    print(f"\n合计 {len(log)} 条 · 成功 {(log['status'] == 'ok').sum()} · 缓存 {(log['status'] == 'cached').sum()} · "
          f"失败 {len(bad)} · 正文中位 {log[log['chars'] > 0]['chars'].median():.0f} 字")
    if len(bad):
        print(bad[["code", "date", "status"]].head(8).to_string(index=False))
    print("账 -> data/fetch_st_pdf_log.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
