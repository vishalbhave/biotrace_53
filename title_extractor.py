"""
title_extractor.py — Multi-strategy PDF title extraction for BioTrace v3.

Strategy ladder (first non-trivial result wins):
  0. pdftitle   (metebalci/pdftitle — SciPlore Xtract font-size algorithm)
  1. PDF XMP / Document Info metadata
  2. fitz font-size heuristic (improved span-merging, noise filtering)
  3. First substantive line on page 1
  4. LLM inference from first 800 characters of document text
  5. Cleaned filename  (guaranteed fallback)

The LLM strategy uses TITLE_INFERENCE_PROMPT from prompts.py and is injected
at call time to avoid circular imports — pass llm_fn=<callable> to extract_title().
"""

from __future__ import annotations

import os
import re
import logging
from typing import Callable, Optional

import fitz  # PyMuPDF

logger = logging.getLogger("biotrace.title")

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

_MIN_TITLE_LEN = 12
_MAX_TITLE_LEN = 200

# Patterns that should never be the title
_NOISE_RE = re.compile(
    r"^("
    r"doi|http|www\.|©|volume|vol\.|issue|issn|isbn|"
    r"journal|proceedings|conference|elsevier|springer|wiley|taylor|"
    r"received|accepted|available online|published|abstract|keywords|"
    r"running head|short title|manuscript|preprint|"
    r"page\s*\d|^\d+$|^[ivxlcdm]+$"           # page numbers / roman numerals
    r")",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", "")).strip()


def _is_noise(text: str) -> bool:
    return bool(_NOISE_RE.match(text)) or len(text) < _MIN_TITLE_LEN


def _trim_to_sentence(text: str, max_len: int = _MAX_TITLE_LEN) -> str:
    """
    Trim title candidate at a natural boundary (punctuation) within max_len.
    Avoids cutting mid-word.
    """
    if len(text) <= max_len:
        return text
    # Try to cut at last full stop / colon / semicolon before max_len
    for sep in (". ", ": ", "; ", " — ", " - "):
        idx = text.rfind(sep, 0, max_len)
        if idx > _MIN_TITLE_LEN:
            return text[:idx].rstrip(". :;")
    # Cut at last space before max_len
    idx = text.rfind(" ", 0, max_len)
    return text[:idx] if idx > _MIN_TITLE_LEN else text[:max_len]


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 0 — pdftitle  (pip install pdftitle)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_via_pdftitle(pdf_path: str) -> str:
    try:
        import pdftitle
        result = pdftitle.get_title_from_file(pdf_path)
        if result:
            t = _clean(result)
            if not _is_noise(t):
                logger.debug(f"[title/pdftitle] '{t[:80]}'")
                return _trim_to_sentence(t)
    except ImportError:
        logger.debug("[title/pdftitle] package not installed — skipping")
    except Exception as exc:
        logger.debug(f"[title/pdftitle] failed: {exc}")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 1 — XMP / Document Info metadata
# ─────────────────────────────────────────────────────────────────────────────

# def _extract_via_metadata(pdf_bytes: bytes) -> str:
#     try:
#         doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#         # XMP first (more reliable for modern academic PDFs)
#         xmp = doc.get_xml_metadata()
#         if xmp:
#             m = re.search(r"<dc:title[^>]*>.*?<rdf:li[^>]*>(.*?)</rdf:li>", xmp, re.DOTALL)
#             if m:
#                 t = _clean(re.sub(r"<[^>]+>", "", m.group(1)))
#                 if not _is_noise(t):
#                     logger.debug(f"[title/xmp] '{t[:80]}'")
#                     return _trim_to_sentence(t)
#         # DocInfo fallback
#         meta_title = _clean((doc.metadata or {}).get("title", ""))
#         if meta_title and not _is_noise(meta_title) and not re.match(
#             r"^(untitled|microsoft word|word -)", meta_title, re.IGNORECASE
#         ):
#             logger.debug(f"[title/docinfo] '{meta_title[:80]}'")
#             return _trim_to_sentence(meta_title)
#     except Exception as exc:
#         logger.debug(f"[title/metadata] {exc}")
#     return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 2 — fitz font-size heuristic (span-merging, academic-noise filter)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_via_font_heuristic(pdf_bytes: bytes) -> str:
    """
    Collect all text spans on page 1. Group spans at the largest font size
    (±1.5 pt), merge adjacent lines that are vertically close, filter noise.
    Handles multi-line titles split across consecutive spans.
    """
    try:
        doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
        p2_texts: set[str] = set()
        if len(doc) > 1:
            for b in doc[1].get_text("dict")["blocks"]:
                for ln in b.get("lines", []):
                    for sp in ln.get("spans", []):
                        t = _clean(sp.get("text", ""))
                        if len(t) >= 8:
                            p2_texts.add(t.lower())
        page  = doc[0]
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        spans: list[dict] = []
        for b in blocks:
            if b.get("type") != 0:
                continue
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    text = _clean(span.get("text", ""))
                    size = round(span.get("size", 0), 1)
                    y    = span.get("origin", (0, 0))[1]
                    if len(text) < 3:
                        continue
                    spans.append({"text": text, "size": size, "y": y})

        if not spans:
            return ""

        max_size = max(
            (s["size"] for s in spans if not _is_noise(s["text"])),
            default=0,
        )
        if max_size < 10:
            return ""

        # Collect spans within 1.5 pt of max size (multi-line title tolerance)
        title_spans = [
            s for s in spans
            if abs(s["size"] - max_size) <= 1.5
            and not _is_noise(s["text"])
            and s["text"].lower() not in p2_texts   # exclude running headers
        ]
        title_spans.sort(key=lambda s: s["y"])

        # Merge consecutive lines closer than 45 pt (single title block)
        merged: list[str] = []
        prev_y: Optional[float] = None
        for s in title_spans:
            if prev_y is None or abs(s["y"] - prev_y) < 45:
                merged.append(s["text"])
            else:
                break
            prev_y = s["y"]

        candidate = _clean(" ".join(merged))
        if len(candidate.split()) <= 5 and candidate == candidate.upper():
            # Looks like a running header — try next font tier (max_size - 1.5 to - 4.0)
            fallback_spans = [
                s for s in spans
                if 1.5 < (max_size - s["size"]) <= 4.0
                and not _is_noise(s["text"])
                and s["text"].lower() not in p2_texts
            ]
            fallback_spans.sort(key=lambda s: s["y"])
            fb_merged, prev_y = [], None
            for s in fallback_spans:
                if prev_y is None or abs(s["y"] - prev_y) < 45:
                    fb_merged.append(s["text"])
                else:
                    break
                prev_y = s["y"]
            fb_candidate = _clean(" ".join(fb_merged))
            if fb_candidate and not _is_noise(fb_candidate) and len(fb_candidate) > len(candidate):
                candidate = fb_candidate
                
        if not _is_noise(candidate):
            logger.debug(f"[title/font] '{candidate[:80]}'")
            return _trim_to_sentence(candidate)
    except Exception as exc:
        logger.debug(f"[title/font] {exc}")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 3 — First substantive line
# ─────────────────────────────────────────────────────────────────────────────

def _extract_via_first_line(pdf_bytes: bytes) -> str:
    try:
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = doc[0].get_text()
        for line in text.split("\n"):
            t = _clean(line)
            if len(t) >= _MIN_TITLE_LEN and not _is_noise(t):
                logger.debug(f"[title/firstline] '{t[:80]}'")
                return _trim_to_sentence(t)
    except Exception as exc:
        logger.debug(f"[title/firstline] {exc}")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 4 — LLM inference  (injected via llm_fn parameter)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_via_llm(pdf_bytes: bytes, llm_fn: Callable[[str], str]) -> str:
    """
    Feed the first ~1 000 characters of page 1 text to an LLM and ask for a title.
    llm_fn should accept a formatted prompt string and return a plain-text title.
    """
    try:
        from .prompts import TITLE_INFERENCE_PROMPT   # relative import (scripts package)
    except ImportError:
        try:
            from prompts import TITLE_INFERENCE_PROMPT    # flat import
        except ImportError:
            return ""

    try:
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = doc[0].get_text()[:1200].strip()
        if len(text) < 100:
            return ""
        prompt = TITLE_INFERENCE_PROMPT.format(chunk=text)
        result = llm_fn(prompt)
        if result:
            t = _clean(result).strip('"\'')
            if not _is_noise(t):
                logger.info(f"[title/llm] '{t[:80]}'")
                return _trim_to_sentence(t)
    except Exception as exc:
        logger.debug(f"[title/llm] {exc}")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 5 — Filename (guaranteed fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_via_filename(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    return _clean(re.sub(r"[_\-]+", " ", base))[:_MAX_TITLE_LEN]


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_title(
    pdf_bytes:   bytes,
    pdf_path:    Optional[str],
    filename:    str,
    llm_fn:      Optional[Callable[[str], str]] = None,
) -> str:
    """
    Run the strategy ladder and return the best title found.

    Parameters
    ----------
    pdf_bytes : raw PDF bytes
    pdf_path  : path to saved PDF file (needed for pdftitle strategy)
    filename  : original upload filename (guaranteed fallback)
    llm_fn    : optional callable(prompt_str) → title_str.
                When provided and all heuristics fail, the LLM is called
                with TITLE_INFERENCE_PROMPT to generate a descriptive title.

    Returns
    -------
    Title string (never empty — filename is the guaranteed last resort).
    """
    strategies = []

    if pdf_path and os.path.exists(pdf_path):
        strategies.append(("pdftitle",      lambda: _extract_via_pdftitle(pdf_path)))

    strategies.append(("metadata",          lambda: _extract_via_metadata(pdf_bytes)))
    strategies.append(("font_heuristic",    lambda: _extract_via_font_heuristic(pdf_bytes)))
    strategies.append(("first_line",        lambda: _extract_via_first_line(pdf_bytes)))

    if llm_fn is not None:
        strategies.append(("llm_inference", lambda: _extract_via_llm(pdf_bytes, llm_fn)))

    strategies.append(("filename",          lambda: _extract_via_filename(filename)))

    for name, fn in strategies:
        try:
            result = fn()
            if result and len(result) >= _MIN_TITLE_LEN:
                logger.info(f"[title] strategy={name} → '{result[:80]}'")
                return result
        except Exception as exc:
            logger.debug(f"[title/{name}] unexpected error: {exc}")

    return _extract_via_filename(filename)
