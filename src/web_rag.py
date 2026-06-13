"""창의성 평가 전용 외부 DB 검색.

연구개발단계별 검색 소스:
  기초연구 → arXiv
  응용연구 → arXiv + KIPRIS
  개발연구 → KIPRIS

캐시: .cache/web_rag_cache.json (쿼리별 결과 저장 → API 호출 절약)
"""
from __future__ import annotations
import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
from dotenv import load_dotenv

# .env → .env.example 순으로 로드
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.example")   # fallback

KIPRIS_API_KEY: str = os.getenv("KIPRIS_API_KEY", "")

_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "RnD-SelectionAgent/1.0"
_TIMEOUT = 8          # arXiv
_KIPRIS_TIMEOUT = 20  # KIPRIS 정부 API는 응답이 느림

# ── 캐시 ─────────────────────────────────────────────────────────
_CACHE_PATH = _ROOT / ".cache" / "web_rag_cache.json"
_CACHE_PATH.parent.mkdir(exist_ok=True)

def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_cache(cache: dict) -> None:
    _CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def _cache_key(source: str, query: str, limit: int) -> str:
    raw = f"{source}:{query}:{limit}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── 공통 결과 타입 ────────────────────────────────────────────────
@dataclass
class SearchResult:
    source: str
    title: str
    year: int | None
    summary: str
    extra: str
    url: str

    def to_prompt_line(self) -> str:
        yr = f" ({self.year})" if self.year else ""
        extra = f" [{self.extra}]" if self.extra else ""
        return (
            f"[{self.source}] {self.title}{yr}{extra}\n"
            f"  {self.summary[:250]}"
        )


# ── arXiv ─────────────────────────────────────────────────────────
_ARXIV_URL = "http://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _search_arxiv(query: str, limit: int, cache: dict) -> list[SearchResult]:
    key = _cache_key("arxiv", query, limit)
    if key in cache:
        return [SearchResult(**r) for r in cache[key]]

    try:
        resp = _SESSION.get(
            _ARXIV_URL,
            params={"search_query": f"all:{query}", "max_results": limit,
                    "sortBy": "relevance"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception as e:
        print(f"[arXiv] 검색 실패: {e}")
        return []

    results = []
    for entry in root.findall("atom:entry", _NS):
        title = (entry.findtext("atom:title", "", _NS) or "").strip().replace("\n", " ")
        abstract = (entry.findtext("atom:summary", "", _NS) or "").strip().replace("\n", " ")
        url = entry.findtext("atom:id", "", _NS) or ""
        published = entry.findtext("atom:published", "", _NS) or ""
        year = int(published[:4]) if len(published) >= 4 else None
        cats = [c.get("term", "") for c in entry.findall("atom:category", _NS)]
        results.append(SearchResult(
            source="arXiv", title=title, year=year,
            summary=abstract[:400], extra=", ".join(cats[:2]), url=url,
        ))
    time.sleep(0.3)  # arXiv rate-limit 준수

    cache[key] = [asdict(r) for r in results]
    return results


# ── KIPRIS ────────────────────────────────────────────────────────
_KIPRIS_URL = "http://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getWordSearch"


def _search_kipris(query: str, limit: int, cache: dict) -> list[SearchResult]:
    if not KIPRIS_API_KEY:
        print("[KIPRIS] KIPRIS_API_KEY 미설정 — 검색 스킵")
        return []

    key = _cache_key("kipris", query, limit)
    if key in cache:
        return [SearchResult(**r) for r in cache[key]]

    try:
        # ServiceKey에 +, /, = 포함 → params 딕셔너리 사용 시 이중 인코딩됨.
        # 다른 파라미터는 requests가 인코딩하고, ServiceKey만 직접 이어붙인다.
        from urllib.parse import urlencode
        other = urlencode({"word": query, "numOfRows": limit, "pageNo": 1, "year": 0})
        url = f"{_KIPRIS_URL}?{other}&ServiceKey={KIPRIS_API_KEY}"
        resp = _SESSION.get(url, timeout=_KIPRIS_TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception as e:
        print(f"[KIPRIS] 검색 실패: {e}")
        return []

    results = []
    for item in root.iter("item"):
        def g(tag: str) -> str:
            el = item.find(tag)
            return (el.text or "").strip() if el is not None else ""

        title = g("inventionTitle")
        app_date = g("applicationDate")
        year = int(app_date[:4]) if len(app_date) >= 4 else None
        ipc = g("ipcNumber")
        abstract = g("astrtCont") or g("claim")
        app_num = g("applicationNumber")
        url = f"https://doi.kipris.or.kr/patinfo/searchLogina.do?applno={app_num}" if app_num else ""
        results.append(SearchResult(
            source="KIPRIS", title=title, year=year,
            summary=abstract[:400], extra=f"IPC {ipc}" if ipc else "", url=url,
        ))

    cache[key] = [asdict(r) for r in results]
    return results


# ── 통합 검색 (캐시 자동 저장) ───────────────────────────────────
def search_for_creativity(keywords: str, stage: str, limit: int = 3) -> str:
    if not keywords.strip():
        return ""

    cache = _load_cache()
    results: list[SearchResult] = []
    cache_updated = False

    if stage in ("기초연구", "응용연구"):
        before = len(cache)
        results += _search_arxiv(keywords, limit, cache)
        if len(cache) > before:
            cache_updated = True

    if stage in ("응용연구", "개발연구"):
        before = len(cache)
        results += _search_kipris(keywords, limit, cache)
        if len(cache) > before:
            cache_updated = True

    if cache_updated:
        _save_cache(cache)

    if not results:
        return ""

    source_labels = {
        "기초연구":  "arXiv 선행논문",
        "응용연구":  "arXiv 선행논문 + KIPRIS 특허",
        "개발연구":  "KIPRIS 특허",
    }
    label = source_labels.get(stage, "외부 검색")

    lines = [
        f"[외부 선행연구·특허 검색 — {label}]",
        "※ 아래 결과를 참고해 제안된 연구의 차별성과 창의성을 판단하십시오. "
        "유사 선행연구·특허가 많을수록 차별성 입증이 더 요구됩니다.",
    ]
    for r in results:
        lines.append(r.to_prompt_line())

    return "\n".join(lines)
