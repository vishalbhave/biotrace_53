"""
biotrace_pdf_meta.py  —  BioTrace v5.3
────────────────────────────────────────────────────────────────────────────
PDF metadata retrieval and intelligent renaming pipeline.

Provider cascade (each attempted in order, stops on first success):
  1. Semantic Scholar Public API — 80 calls / 5 min budget; DOI or title search
  2. Crossref REST (habanero)    — polite pool, 1 req/sec; DOI or title fuzzy
  3. PDF DOI extraction          — parse DOI from PDF text (page 1-2)
  4. PDF title extraction        — heuristic: first large-text line on page 1
  5. LLM title extraction        — ask LLM if above fails (optional)

Output:
  PaperMeta(title, authors, year, doi, journal, abstract, source, citation_string)

PDF renaming:
  "author_year_title.pdf"  e.g. "Pillai_1985_Marine_fauna_Gulf_of_Kutch.pdf"
  • author  = first author surname, ASCII-safe
  • year    = publication year (4-digit)
  • title   = first 8 words of title, underscored, no punctuation

Citation string (injected into LLM prompt as [CURRENT_DOCUMENT_METADATA]):
  "Pillai RSK, 1985. Marine fauna of the Gulf of Kutch. Journal of
   the Marine Biological Association of India, 27(1–2): 1–42."

Rate limiting:
  Semantic Scholar: ≤ 80 calls per 5 minutes (16/min). Token-bucket enforced.
  Crossref:         polite pool 1 req/sec with email contact.

Usage:
    from biotrace_pdf_meta import PaperMetaFetcher
    fetcher = PaperMetaFetcher(email="researcher@example.com")
    meta    = fetcher.fetch(pdf_path="papers/scan.pdf")
    new_path = fetcher.rename_pdf(pdf_path, meta)
    print(meta.citation_string)
"""
from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("biotrace.pdf_meta")

# ─────────────────────────────────────────────────────────────────────────────
#  OPTIONAL DEPS
# ─────────────────────────────────────────────────────────────────────────────
# _HABANERO_OK = False
# try:
#     from habanero import Crossref as _Crossref
#     _HABANERO_OK = True
# except ImportError:
#     pass

_S2_OK = False
try:
    import semanticscholar as _s2_pkg
    _S2_OK = True
except ImportError:
    pass

_PYPDF_OK = False
try:
    from pypdf import PdfReader as _PdfReader
    _PYPDF_OK = True
except ImportError:
    try:
        import fitz as _fitz   # PyMuPDF fallback
        _PYPDF_OK = True
    except ImportError:
        pass

_REQUESTS_OK = False
try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PaperMeta:
    title:           str = ""
    authors:         list[str] = field(default_factory=list)
    year:            str = ""
    doi:             str = ""
    journal:         str = ""
    volume:          str = ""
    issue:           str = ""
    pages:           str = ""
    abstract:        str = ""
    source:          str = "unknown"   # semantic_scholar | crossref | pdf_doi | pdf_heuristic | llm

    @property
    def first_author_surname(self) -> str:
        if not self.authors:
            return "Unknown"
        raw = self.authors[0]
        # "Pillai RSK" or "Pillai, R.S.K." or "R.S.K. Pillai"
        parts = raw.replace(",", " ").split()
        # Longest token that starts with uppercase is likely surname
        surnames = [p for p in parts if p and p[0].isupper() and len(p) > 2
                    and not all(c.isupper() or c == "." for c in p)]
        return surnames[0] if surnames else parts[0]

    @property
    def citation_string(self) -> str:
        """Full citation string for LLM prompt injection."""
        authors_str = "; ".join(self.authors[:4])
        if len(self.authors) > 4:
            authors_str += " et al."
        parts = [authors_str]
        if self.year:
            parts.append(self.year + ".")
        if self.title:
            parts.append(self.title + ".")
        if self.journal:
            jpart = self.journal
            if self.volume:
                jpart += f", {self.volume}"
                if self.issue:
                    jpart += f"({self.issue})"
            if self.pages:
                jpart += f": {self.pages}"
            parts.append(jpart + ".")
        if self.doi:
            parts.append(f"DOI: {self.doi}")
        return " ".join(filter(None, parts))

    @property
    def safe_filename_stem(self) -> str:
        """
        Generate 'FirstAuthor_Year_First_Eight_Words_Of_Title' filename stem.
        All characters ASCII-safe, no punctuation, underscored.
        """
        def _ascii(s: str) -> str:
            return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

        author = _ascii(self.first_author_surname)
        author = re.sub(r"[^A-Za-z0-9]", "", author)[:20]

        year = re.sub(r"[^0-9]", "", self.year)[:4]

        title_words = re.sub(r"[^A-Za-z0-9\s]", " ", _ascii(self.title)).split()
        title_slug  = "_".join(title_words[:8])
        title_slug  = re.sub(r"_+", "_", title_slug).strip("_")

        parts = [p for p in [author, year, title_slug] if p]
        return "_".join(parts) or "unknown_paper"

    def is_complete(self) -> bool:
        return bool(self.title and self.authors and self.year)


# ─────────────────────────────────────────────────────────────────────────────
#  TOKEN BUCKET — Semantic Scholar rate limit (80 calls / 5 min)
# ─────────────────────────────────────────────────────────────────────────────
class _TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self._rate     = rate        # tokens added per second
        self._capacity = capacity
        self._tokens   = float(capacity)
        self._last_ts  = time.monotonic()

    def consume(self, n: int = 1) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_ts
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_ts = now
        if self._tokens >= n:
            self._tokens -= n
            return True
        return False

    def wait_and_consume(self, n: int = 1):
        while not self.consume(n):
            time.sleep(0.1)


# 80 calls / 5 min = 80/300 ≈ 0.267 calls/sec
_S2_BUCKET = _TokenBucket(rate=0.267, capacity=80)


# ─────────────────────────────────────────────────────────────────────────────
#  PROVIDER 1 — Semantic Scholar
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_s2_by_doi(doi: str, api_key: str = "") -> Optional[PaperMeta]:
    if not doi or not _REQUESTS_OK:
        return None
    _S2_BUCKET.wait_and_consume()
    headers = {"x-api-key": api_key} if api_key else {}
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi.strip()}"
        params = {"fields": "title,authors,year,externalIds,abstract,journal,venue"}
        r = _requests.get(url, params=params, headers=headers, timeout=12)
        if r.status_code == 200:
            return _parse_s2(r.json(), "semantic_scholar")
        if r.status_code == 429:
            logger.warning("[pdf_meta] S2 rate limit — sleeping 10s")
            time.sleep(10)
    except Exception as exc:
        logger.debug("[pdf_meta] S2 DOI: %s", exc)
    return None


def _fetch_s2_by_title(title: str, api_key: str = "") -> Optional[PaperMeta]:
    if not title or not _REQUESTS_OK:
        return None
    _S2_BUCKET.wait_and_consume()
    headers = {"x-api-key": api_key} if api_key else {}
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query":  title[:200],
            "fields": "title,authors,year,externalIds,abstract,journal,venue",
            "limit":  1,
        }
        r = _requests.get(url, params=params, headers=headers, timeout=12)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                return _parse_s2(data[0], "semantic_scholar")
        if r.status_code == 429:
            logger.warning("[pdf_meta] S2 rate limit — sleeping 10s")
            time.sleep(10)
    except Exception as exc:
        logger.debug("[pdf_meta] S2 title: %s", exc)
    return None


def _parse_s2(item: dict, source: str) -> PaperMeta:
    authors = [
        a.get("name", "") for a in item.get("authors", [])
    ]
    doi = (item.get("externalIds") or {}).get("DOI", "")
    j   = item.get("journal") or {}
    return PaperMeta(
        title    = item.get("title",""),
        authors  = authors,
        year     = str(item.get("year","") or ""),
        doi      = doi,
        journal  = j.get("name","") if isinstance(j,dict) else item.get("venue",""),
        volume   = j.get("volume","") if isinstance(j,dict) else "",
        pages    = j.get("pages","")  if isinstance(j,dict) else "",
        abstract = (item.get("abstract","") or "")[:800],
        source   = source,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PROVIDER 2 — Crossref (habanero)
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_crossref_by_doi(doi: str, email: str = "") -> Optional[PaperMeta]:
    if not doi or not _HABANERO_OK:
        return None
    try:
        cr = _Crossref(mailto=email or "biotrace@example.com")
        time.sleep(1.0)   # polite pool
        result = cr.works(ids=doi.strip())
        items  = result.get("message", {})
        return _parse_crossref(items, "crossref")
    except Exception as exc:
        logger.debug("[pdf_meta] Crossref DOI: %s", exc)
    return None


def _fetch_crossref_by_title(title: str, email: str = "") -> Optional[PaperMeta]:
    if not title or not _HABANERO_OK:
        return None
    try:
        cr = _Crossref(mailto=email or "biotrace@example.com")
        time.sleep(1.0)
        result = cr.works(query=title[:200], limit=1, sort="relevance")
        items  = (result.get("message",{}).get("items") or [])
        if items:
            return _parse_crossref(items[0], "crossref")
    except Exception as exc:
        logger.debug("[pdf_meta] Crossref title: %s", exc)
    return None


def _parse_crossref(item: dict, source: str) -> PaperMeta:
    def _name(a: dict) -> str:
        fn = a.get("given","")
        ln = a.get("family","")
        initials = "".join(p[0].upper() for p in fn.split() if p) if fn else ""
        return f"{ln} {initials}".strip() if ln else fn

    authors = [_name(a) for a in (item.get("author") or [])[:8]]
    title   = " ".join((item.get("title") or [""])[:1])
    year    = str((item.get("published","") or item.get("issued","") or {})
                  .get("date-parts",[[""]])[0][0])
    journal = " ".join(item.get("container-title") or [""])
    doi     = item.get("DOI","")
    volume  = item.get("volume","") or ""
    issue   = item.get("issue","")  or ""
    pages   = item.get("page","")   or ""
    return PaperMeta(
        title=title, authors=authors, year=year, doi=doi,
        journal=journal, volume=volume, issue=issue, pages=pages,
        source=source,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PROVIDER 3 — Extract DOI from PDF text
# ─────────────────────────────────────────────────────────────────────────────
_DOI_RE = re.compile(
    r"\b(?:doi[:\s/]*)?10\.\d{4,9}/[-._;()/:A-Za-z0-9]+",
    re.IGNORECASE,
)

def extract_doi_from_pdf(pdf_path: str) -> str:
    """Extract a DOI string from the first two pages of a PDF."""
    text = _pdf_first_pages(pdf_path, n=2)
    matches = _DOI_RE.findall(text)
    if matches:
        doi = matches[0].strip().rstrip(".")
        doi = re.sub(r"^doi[:\s/]*", "", doi, flags=re.IGNORECASE)
        logger.debug("[pdf_meta] DOI from PDF: %s", doi)
        return doi
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  PROVIDER 4 — Heuristic title extraction from PDF
# ─────────────────────────────────────────────────────────────────────────────
def extract_title_from_pdf(pdf_path: str, llm_fn: Optional[Callable] = None) -> str:
    """
    Extract paper title from PDF.
    Strategy:
      1. pypdf: first non-empty lines of page 1, longest run before 'Abstract'
      2. LLM fallback: ask LLM to identify title from first 500 chars
    """
    text = _pdf_first_pages(pdf_path, n=1)
    if not text.strip():
        return ""

    # Heuristic: title = text before 'Abstract', 'Introduction', or author lines
    # (all-caps author abbreviations often follow title)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Find where 'abstract' or 'introduction' starts
    stop_words = re.compile(r"^(abstract|introduction|keywords|received|accepted|doi)", re.I)
    title_lines: list[str] = []
    for line in lines:
        if stop_words.match(line):
            break
        if len(line) > 15:  # skip very short lines (page numbers, etc.)
            title_lines.append(line)
        if len(title_lines) >= 4:  # title rarely spans > 4 lines
            break

    candidate = " ".join(title_lines).strip()

    # LLM fallback
    if not candidate and llm_fn:
        try:
            excerpt = text[:500]
            resp = llm_fn(
                f"Extract ONLY the paper title from this PDF text. "
                f"Output the title verbatim, nothing else.\n\n{excerpt}"
            )
            candidate = resp.strip().strip('"').strip("'")
            logger.info("[pdf_meta] LLM title: %s", candidate[:60])
        except Exception as exc:
            logger.debug("[pdf_meta] LLM title: %s", exc)

    return candidate[:300]


def _pdf_first_pages(pdf_path: str, n: int = 2) -> str:
    """Extract raw text from first n pages of PDF."""
    try:
        if _PYPDF_OK:
            try:
                from pypdf import PdfReader
                reader = PdfReader(pdf_path)
                pages  = reader.pages[:n]
                return "\n".join(p.extract_text() or "" for p in pages)
            except Exception:
                pass
        # PyMuPDF fallback
        import fitz
        doc  = fitz.open(pdf_path)
        text = "\n".join(doc[i].get_text() for i in range(min(n, len(doc))))
        doc.close()
        return text
    except Exception as exc:
        logger.debug("[pdf_meta] PDF read: %s", exc)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN FETCHER CLASS
# ─────────────────────────────────────────────────────────────────────────────
class PaperMetaFetcher:
    """
    Cascade-based paper metadata fetcher for BioTrace.

    Attempts (in order):
      1. S2 by DOI extracted from PDF
      2. Crossref by DOI
      3. S2 by heuristic PDF title
      4. Crossref by heuristic PDF title
      5. Partial metadata from PDF heuristics only

    Respects Semantic Scholar rate limit: ≤ 80 calls / 5 minutes.
    """

    def __init__(
        self,
        email:       str = "",
        s2_api_key:  str = "",
        llm_fn:      Optional[Callable] = None,
    ):
        self.email      = email
        self.s2_api_key = s2_api_key
        self.llm_fn     = llm_fn

    def fetch(self, pdf_path: str = "", title_hint: str = "", doi_hint: str = "") -> PaperMeta:
        """
        Fetch metadata for a paper.
        pdf_path:   path to the PDF (used for DOI extraction + title heuristic)
        title_hint: known title string (skip PDF parsing)
        doi_hint:   known DOI string (skip PDF parsing)
        """
        # Step 0: extract DOI and heuristic title from PDF
        doi   = doi_hint   or (extract_doi_from_pdf(pdf_path)   if pdf_path else "")
        title = title_hint or (extract_title_from_pdf(pdf_path, self.llm_fn) if pdf_path else "")

        logger.info("[pdf_meta] doi=%r title=%r", doi[:40] if doi else "", title[:60] if title else "")

        # Step 1: S2 by DOI
        if doi:
            meta = _fetch_s2_by_doi(doi, self.s2_api_key)
            if meta and meta.is_complete():
                logger.info("[pdf_meta] S2 DOI hit: %s", meta.title[:60])
                return meta

        # Step 2: Crossref by DOI
        if doi:
            meta = _fetch_crossref_by_doi(doi, self.email)
            if meta and meta.is_complete():
                logger.info("[pdf_meta] Crossref DOI hit: %s", meta.title[:60])
                return meta

        # Step 3: S2 by title
        if title:
            meta = _fetch_s2_by_title(title, self.s2_api_key)
            if meta and meta.is_complete():
                logger.info("[pdf_meta] S2 title hit: %s", meta.title[:60])
                return meta

        # Step 4: Crossref by title
        if title:
            meta = _fetch_crossref_by_title(title, self.email)
            if meta and meta.is_complete():
                logger.info("[pdf_meta] Crossref title hit: %s", meta.title[:60])
                return meta

        # Step 5: Partial from PDF heuristics
        logger.warning("[pdf_meta] All providers failed — partial metadata from PDF only")
        return PaperMeta(
            title  = title,
            doi    = doi,
            source = "pdf_heuristic",
        )

    def rename_pdf(self, pdf_path: str, meta: PaperMeta, dest_dir: str = "") -> str:
        """
        Rename a PDF file to 'Author_Year_Title.pdf'.
        Returns the new file path. If dest_dir is given, moves there.
        Original file is kept if rename fails.
        """
        if not pdf_path or not os.path.exists(pdf_path):
            return pdf_path

        stem    = meta.safe_filename_stem
        new_name = f"{stem}.pdf"
        src_dir  = os.path.dirname(os.path.abspath(pdf_path))
        target   = os.path.join(dest_dir or src_dir, new_name)

        if os.path.abspath(pdf_path) == os.path.abspath(target):
            return pdf_path  # already correctly named

        try:
            os.rename(pdf_path, target)
            logger.info("[pdf_meta] Renamed: %s → %s", os.path.basename(pdf_path), new_name)
            return target
        except Exception as exc:
            logger.warning("[pdf_meta] Rename failed: %s", exc)
            return pdf_path

    def fetch_and_rename(
        self,
        pdf_path: str,
        dest_dir: str = "",
        title_hint: str = "",
        doi_hint: str = "",
    ) -> tuple[PaperMeta, str]:
        """
        Convenience: fetch metadata and rename the PDF.
        Returns (PaperMeta, new_pdf_path).
        """
        meta     = self.fetch(pdf_path=pdf_path, title_hint=title_hint, doi_hint=doi_hint)
        new_path = self.rename_pdf(pdf_path, meta, dest_dir=dest_dir)
        return meta, new_path


# ─────────────────────────────────────────────────────────────────────────────
#  AVAILABILITY REPORT
# ─────────────────────────────────────────────────────────────────────────────
def availability_report() -> dict[str, bool]:
    return {
        "semantic_scholar": _REQUESTS_OK,
        "crossref_habanero": _HABANERO_OK,
        "pdf_doi_extraction": _PYPDF_OK,
        "pdf_title_heuristic": _PYPDF_OK,
    }
