"""
biotrace_ocr.py  —  BioTrace v5.2
────────────────────────────────────────────────────────────────────────────
OCR & Multimodal pipeline for image-based (scanned) PDFs.

Detection order:
  1. is_scanned_pdf()    — quick heuristic: text layer chars per page
  2. DocTR OCR           — best accuracy, GPU-optional, offline
  3. Pytesseract         — system Tesseract, always available
  4. Multimodal LLM OCR  — Gemma4 / LLaVA / Mistral-Pixtral via Ollama vision API
                           also used for figure captions and embedded tables

All paths produce clean Markdown text ready for DocumentChunker.

Usage:
    from biotrace_ocr import OCRPipeline, is_scanned_pdf
    pipe = OCRPipeline(prefer="doctr")
    md_text, method = pipe.pdf_to_text("scan.pdf")

    # Multimodal page-by-page (for figure-heavy documents)
    pipe2 = OCRPipeline(prefer="multimodal", ollama_model="gemma4")
    md_text, method = pipe2.pdf_to_text("thesis_with_figures.pdf")
"""
from __future__ import annotations

import base64
import io
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("biotrace.ocr")

# ─────────────────────────────────────────────────────────────────────────────
#  OPTIONAL DEPS
# ─────────────────────────────────────────────────────────────────────────────
_DOCTR_AVAILABLE = False
_doctr_model     = None   # lazy-loaded singleton

try:
    from doctr.io import DocumentFile as _DoctrDocFile
    from doctr.models import ocr_predictor as _ocr_predictor
    _DOCTR_AVAILABLE = True
    logger.info("[ocr] DocTR available")
except ImportError:
    pass

_PDF2IMAGE_AVAILABLE = False
try:
    from pdf2image import convert_from_path as _convert_from_path
    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    pass

_TESSERACT_AVAILABLE = False
try:
    import pytesseract as _pytesseract
    from PIL import Image as _PILImage
    _TESSERACT_AVAILABLE = True
except ImportError:
    pass

_PYMUPDF_AVAILABLE = False
try:
    import fitz as _fitz
    _PYMUPDF_AVAILABLE = True
except ImportError:
    pass

_OLLAMA_AVAILABLE = False
try:
    import ollama as _ollama_pkg
    _OLLAMA_AVAILABLE = True
except ImportError:
    pass

_REQUESTS_AVAILABLE = False
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  SCAN DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def is_scanned_pdf(pdf_path: str, sample_pages: int = 5, min_chars_per_page: int = 80) -> bool:
    """
    Heuristic: if the average extractable text per page is below
    min_chars_per_page, the PDF is likely scanned (image-only).

    Returns True  → needs OCR.
    Returns False → has a native text layer.
    """
    if not _PYMUPDF_AVAILABLE:
        return False
    try:
        doc   = _fitz.open(pdf_path)
        n     = min(len(doc), sample_pages)
        total = sum(len(doc[i].get_text().strip()) for i in range(n))
        doc.close()
        avg = total / max(n, 1)
        result = avg < min_chars_per_page
        logger.info(
            "[ocr] %s avg_chars_per_page=%.0f scanned=%s",
            Path(pdf_path).name, avg, result,
        )
        return result
    except Exception as exc:
        logger.debug("[ocr] scan detect: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _pdf_to_pil_pages(pdf_path: str, dpi: int = 200) -> list:
    """Rasterise PDF pages to PIL Images."""
    if _PDF2IMAGE_AVAILABLE:
        try:
            return _convert_from_path(pdf_path, dpi=dpi)
        except Exception as exc:
            logger.warning("[ocr] pdf2image: %s", exc)

    if _PYMUPDF_AVAILABLE:
        try:
            doc    = _fitz.open(pdf_path)
            pages  = []
            matrix = _fitz.Matrix(dpi / 72, dpi / 72)
            for page in doc:
                pix = page.get_pixmap(matrix=matrix)
                from PIL import Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pages.append(img)
            doc.close()
            return pages
        except Exception as exc:
            logger.warning("[ocr] pymupdf raster: %s", exc)

    return []


def _pil_to_base64(img, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64-encoded string for multimodal API."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _clean_ocr_text(text: str) -> str:
    """Post-process raw OCR output → cleaner Markdown."""
    # Remove form-feed chars
    text = text.replace("\f", "\n\n")
    # Collapse 3+ blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix hyphenation across line breaks  (e.g. "spe-\ncie" → "specie")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Strip trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  OCR BACKENDS
# ─────────────────────────────────────────────────────────────────────────────
def _ocr_doctr(pdf_path: str) -> str:
    """
    DocTR OCR — best offline accuracy.
    Lazy-loads the model singleton on first call.
    """
    global _doctr_model
    if not _DOCTR_AVAILABLE:
        return ""
    try:
        if _doctr_model is None:
            logger.info("[ocr] Loading DocTR model (first call)…")
            _doctr_model = _ocr_predictor(pretrained=True)
            logger.info("[ocr] DocTR model loaded")

        doc    = _DoctrDocFile.from_pdf(pdf_path)
        result = _doctr_model(doc)

        # Flatten pages → paragraphs → words
        pages_text = []
        for page in result.pages:
            page_words = []
            for block in page.blocks:
                for line in block.lines:
                    line_words = [w.value for w in line.words]
                    page_words.append(" ".join(line_words))
            pages_text.append("\n".join(page_words))

        return _clean_ocr_text("\n\n".join(pages_text))
    except Exception as exc:
        logger.warning("[ocr] DocTR: %s", exc)
        return ""


def _ocr_tesseract(pdf_path: str, dpi: int = 200, lang: str = "eng") -> str:
    """Tesseract OCR via pytesseract — system Tesseract required."""
    if not _TESSERACT_AVAILABLE:
        return ""
    try:
        pages = _pdf_to_pil_pages(pdf_path, dpi=dpi)
        if not pages:
            return ""
        texts = []
        for i, img in enumerate(pages):
            t = _pytesseract.image_to_string(img, lang=lang)
            texts.append(t)
            if i < len(pages) - 1:
                texts.append("\n\n---\n\n")   # page separator
        return _clean_ocr_text("".join(texts))
    except Exception as exc:
        logger.warning("[ocr] Tesseract: %s", exc)
        return ""


def _ocr_multimodal_ollama(
    pdf_path: str,
    model: str         = "gemma4",
    base_url: str      = "http://localhost:11434",
    dpi: int           = 150,
    max_pages: int     = 50,
    page_prompt: str   = "",
) -> str:
    """
    Multimodal OCR via Ollama vision API.
    Works with any Ollama model that supports images:
    gemma4, llava, llava:13b, mistralpixral, moondream, bakllava.

    For very long documents only the first max_pages pages are processed
    (configurable); text extraction continues with a text-only method.
    """
    if not _OLLAMA_AVAILABLE:
        return ""

    prompt = page_prompt or (
        "Extract ALL text from this PDF page image exactly as it appears. "
        "Preserve paragraph structure. Output scientific species names verbatim "
        "in italics (*Genus species*). Do NOT add commentary. "
        "Do NOT summarise. Transcribe every word."
    )

    pages = _pdf_to_pil_pages(pdf_path, dpi=dpi)
    if not pages:
        return ""

    texts: list[str] = []
    n = min(len(pages), max_pages)
    logger.info("[ocr] Multimodal: %d/%d pages via %s", n, len(pages), model)

    for i, img in enumerate(pages[:n]):
        b64 = _pil_to_base64(img, fmt="PNG")
        try:
            resp = _ollama_pkg.chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [b64],
                }],
                options={"num_predict": 2048, "temperature": 0.0},
            )
            page_text = resp.message.content if hasattr(resp, "message") else resp["message"]["content"]
            texts.append(page_text)
            if i < n - 1:
                texts.append(f"\n\n---\n\n")
        except Exception as exc:
            logger.warning("[ocr] Multimodal page %d: %s", i + 1, exc)
            texts.append(f"\n\n[OCR failed for page {i+1}]\n\n")

    if len(pages) > max_pages:
        texts.append(
            f"\n\n[Note: document has {len(pages)} pages; "
            f"multimodal OCR processed first {max_pages}]\n"
        )

    return _clean_ocr_text("".join(texts))


def _ocr_multimodal_openai_compat(
    pdf_path: str,
    model: str         = "gpt-4o",
    api_key: str       = "",
    base_url: str      = "https://api.openai.com/v1",
    dpi: int           = 150,
    max_pages: int     = 30,
) -> str:
    """OpenAI-compatible vision endpoint (OpenAI GPT-4o or Gemini vision)."""
    if not _REQUESTS_AVAILABLE or not api_key:
        return ""

    pages = _pdf_to_pil_pages(pdf_path, dpi=dpi)
    if not pages:
        return ""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    texts: list[str] = []
    n = min(len(pages), max_pages)

    for i, img in enumerate(pages[:n]):
        b64 = _pil_to_base64(img, fmt="PNG")
        payload = {
            "model": model,
            "max_tokens": 2048,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL text from this scientific paper page exactly as written. "
                            "Preserve scientific names. Output clean text only."
                        ),
                    },
                ],
            }],
        }
        try:
            r = _requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            page_text = r.json()["choices"][0]["message"]["content"]
            texts.append(page_text)
            if i < n - 1:
                texts.append("\n\n---\n\n")
        except Exception as exc:
            logger.warning("[ocr] OpenAI-compat page %d: %s", i + 1, exc)

    return _clean_ocr_text("".join(texts))


# ─────────────────────────────────────────────────────────────────────────────
#  OCR PIPELINE CLASS
# ─────────────────────────────────────────────────────────────────────────────
class OCRPipeline:
    """
    Unified OCR pipeline with automatic scanned-PDF detection.

    prefer order (configurable):
      "doctr"       → DocTR → Tesseract → Multimodal
      "multimodal"  → Multimodal → DocTR → Tesseract
      "tesseract"   → Tesseract → DocTR → Multimodal
      "auto"        → Multimodal if model available, else DocTR, else Tesseract
    """

    def __init__(
        self,
        prefer:        str  = "doctr",
        ollama_model:  str  = "gemma4",
        ollama_base_url: str= "http://localhost:11434",
        openai_api_key: str = "",
        openai_model:  str  = "gpt-4o",
        dpi:           int  = 200,
        max_pages:     int  = 50,
        lang:          str  = "eng",
    ):
        self.prefer            = prefer
        self.ollama_model      = ollama_model
        self.ollama_base_url   = ollama_base_url
        self.openai_api_key    = openai_api_key
        self.openai_model      = openai_model
        self.dpi               = dpi
        self.max_pages         = max_pages
        self.lang              = lang

    def availability(self) -> dict[str, bool]:
        return {
            "doctr":      _DOCTR_AVAILABLE,
            "tesseract":  _TESSERACT_AVAILABLE,
            "pdf2image":  _PDF2IMAGE_AVAILABLE,
            "multimodal_ollama": _OLLAMA_AVAILABLE,
            "multimodal_openai": bool(self.openai_api_key),
        }

    def pdf_to_text(self, pdf_path: str) -> tuple[str, str]:
        """
        Auto-detect if PDF is scanned; apply appropriate OCR.
        Returns (text, method_name).
        """
        path = Path(pdf_path)
        if not path.exists():
            return f"[File not found: {pdf_path}]", "error"

        scanned = is_scanned_pdf(str(pdf_path))
        logger.info("[ocr] %s scanned=%s prefer=%s", path.name, scanned, self.prefer)

        if not scanned:
            # Has text layer — use fast extraction, not OCR
            if _PYMUPDF_AVAILABLE:
                try:
                    import fitz
                    doc  = fitz.open(str(pdf_path))
                    text = "\n\f".join(p.get_text() for p in doc)
                    doc.close()
                    if text.strip():
                        return _clean_ocr_text(text), "fitz_native"
                except Exception:
                    pass

        # Needs OCR — try backends in preference order
        backends = self._backend_order()
        for name, fn in backends:
            logger.info("[ocr] Trying backend: %s", name)
            result = fn(str(pdf_path))
            if result.strip():
                logger.info("[ocr] ✅ %s produced %d chars", name, len(result))
                return result, name

        return f"[OCR failed for {path.name}]", "failed"

    def ocr_image(self, img_path: str) -> tuple[str, str]:
        """OCR a single image file (PNG/JPG)."""
        if _DOCTR_AVAILABLE:
            try:
                doc    = _DoctrDocFile.from_images([img_path])
                result = (_doctr_model or _ocr_predictor(pretrained=True))(doc)
                texts  = []
                for page in result.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            texts.append(" ".join(w.value for w in line.words))
                return _clean_ocr_text("\n".join(texts)), "doctr"
            except Exception as exc:
                logger.warning("[ocr] DocTR image: %s", exc)

        if _TESSERACT_AVAILABLE:
            try:
                img  = _PILImage.open(img_path)
                text = _pytesseract.image_to_string(img, lang=self.lang)
                return _clean_ocr_text(text), "tesseract"
            except Exception as exc:
                logger.warning("[ocr] Tesseract image: %s", exc)

        return "", "failed"

    def _backend_order(self) -> list[tuple[str, callable]]:
        """Return list of (name, callable) in preference order."""
        doctr_fn = lambda p: _ocr_doctr(p)
        tess_fn  = lambda p: _ocr_tesseract(p, dpi=self.dpi, lang=self.lang)
        mm_fn    = lambda p: _ocr_multimodal_ollama(
            p, model=self.ollama_model, base_url=self.ollama_base_url,
            dpi=self.dpi, max_pages=self.max_pages,
        )
        oa_fn    = lambda p: _ocr_multimodal_openai_compat(
            p, model=self.openai_model, api_key=self.openai_api_key,
            dpi=self.dpi, max_pages=self.max_pages,
        )

        orders = {
            "doctr":      [("doctr", doctr_fn),
                           ("tesseract", tess_fn),
                           ("multimodal_ollama", mm_fn)],
            "multimodal": [("multimodal_ollama", mm_fn),
                           ("doctr", doctr_fn),
                           ("tesseract", tess_fn)],
            "tesseract":  [("tesseract", tess_fn),
                           ("doctr", doctr_fn),
                           ("multimodal_ollama", mm_fn)],
            "auto":       ([("multimodal_ollama", mm_fn)] if _OLLAMA_AVAILABLE else [])
                           + ([("doctr", doctr_fn)] if _DOCTR_AVAILABLE else [])
                           + ([("tesseract", tess_fn)] if _TESSERACT_AVAILABLE else []),
        }
        base = orders.get(self.prefer, orders["doctr"])

        # Add OpenAI-compat if key is set (always last resort)
        if self.openai_api_key:
            base = base + [("multimodal_openai", oa_fn)]

        # Filter to actually available backends
        avail = self.availability()
        return [
            (name, fn) for name, fn in base
            if avail.get(name, True)   # True = no check needed (native)
        ]


def availability_report() -> dict[str, bool]:
    return {
        "doctr":      _DOCTR_AVAILABLE,
        "tesseract":  _TESSERACT_AVAILABLE,
        "pdf2image":  _PDF2IMAGE_AVAILABLE,
        "pymupdf":    _PYMUPDF_AVAILABLE,
        "ollama":     _OLLAMA_AVAILABLE,
    }
