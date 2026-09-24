"""从 OpenRouter 的公开模型清单里取 Jev 条目（**第一手注册表事实**，不是营销页抄来的）。

用法: python fetch_jev_registry.py
落: data/jev_registry.json
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "jev_registry.json"


def main() -> int:
    r = requests.get("https://openrouter.ai/api/v1/models", timeout=40)
    r.raise_for_status()
    models = r.json()["data"]
    hits = [m for m in models if "jev" in m.get("id", "").lower() or "jev" in str(m.get("name", "")).lower()
            or "typesafe" in str(m.get("name", "")).lower() or "typesafe" in m.get("id", "").lower()]
    print(f"OpenRouter 模型总数 {len(models)} · 命中 Jev/TypeSafe {len(hits)} 条")
    for m in hits:
        pr = m.get("pricing", {})
        print(f"\n  id           {m['id']}")
        print(f"  name         {m.get('name')}")
        print(f"  context      {m.get('context_length')}")
        print(f"  pricing      prompt {pr.get('prompt')} / completion {pr.get('completion')} "
              f"（USD per token；completion 为 0 即「输出免费」那一档）")
        print(f"  modality     {m.get('architecture', {}).get('modality')}")
        print(f"  description  {str(m.get('description'))[:220]}")
    OUT.write_text(json.dumps({"total_models": len(models), "hits": hits}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
