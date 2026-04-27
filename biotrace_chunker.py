"""
biotrace_chunker.py  —  BioTrace v5.1
────────────────────────────────────────────────────────────────────────────
Section-aware document chunking for long academic PDFs (1 page → 1 000 pages).

Core problem solved:
  Naïve character-count chunking bisects taxonomic descriptions mid-sentence,
  causing species names to be split across chunk boundaries and dropped.

Strategy (Revised Data Extraction Protocol):
  1. Structural parsing   — Docling (ibm/granite-docling via Ollama) converts
                           large PDFs to fully-structured Markdown with headers.
  2. Section segmentation — chunk ON Markdown headers (#/##/###) so every
                           species description stays inside one logical block.
  3. Context overlap      — the last N chars of the previous section are
                           prepended to the next chunk to catch cross-boundary names.
  4. Adaptive batch size  — Gemma 4 / large-context models accept 50–100 page
                           batches; smaller models fall back to single sections.
  5. Vision-Parse fallback — figure-heavy / table-dense pages are rasterised and
                           OCR-read via vision_parse before text chunking.
  6. LangExtract bridge   — optional Google LangExtract pass for structured data
                           extraction from each section chunk.

Chunk metadata attached to every chunk:
  {
    "chunk_id":    int,
    "section":     "3.2 Results — Intertidal fauna",
    "page_range":  "45-48",          # estimated from Markdown heading annotations
    "char_start":  int,
    "char_end":    int,
    "text":        str,
    "source_file": str,
  }

Usage:
    from biotrace_chunker import DocumentChunker
    chunker = DocumentChunker(
        strategy      = "section",     # "section" | "fixed" | "page"
        chunk_chars   = 6000,          # max chars per section chunk
        overlap_chars = 400,           # context carry-over between chunks
        model_context = 32000,         # target model context window (chars)
        use_docling   = True,
        use_vision    = False,
    )
    chunks = chunker.chunk_file("thesis.pdf")
    # or from already-extracted markdown:
    chunks = chunker.chunk_markdown(md_text, source_label="thesis.pdf")
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("biotrace.chunker")

# ─────────────────────────────────────────────────────────────────────────────
#  OPTIONAL HEAVY DEPS — all gracefully degraded
# ─────────────────────────────────────────────────────────────────────────────


_DOCLING_AVAILABLE = False
try:
    from docling.document_converter import DocumentConverter as _DoclingConverter
    _DOCLING_AVAILABLE = True
    logger.info("[chunker] docling available")
except ImportError:
    pass

_VISIONPARSE_AVAILABLE = False
try:
    from vision_parse import VisionParser
    _VISIONPARSE_AVAILABLE = True
    logger.info("[chunker] vision_parse available")
except ImportError:
    pass

_LANGEXTRACT_AVAILABLE = False
try:
    import langextract as _langextract
    _LANGEXTRACT_AVAILABLE = True
    logger.info("[chunker] langextract available")
except ImportError:
    pass

_PYMUPDF_AVAILABLE = False
try:
    import fitz as _fitz
    _PYMUPDF_AVAILABLE = True
except ImportError:
    pass

_PYMUPDF4LLM_AVAILABLE = False
try:
    import pymupdf4llm as _pymupdf4llm
    _PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    pass

_MARKITDOWN_AVAILABLE = False
try:
    from markitdown import MarkItDown as _MarkItDown
    _MARKITDOWN_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Markdown heading pattern — detects H1/H2/H3 and setext-style headings
_HEADING_RE = re.compile(
    r"^(#{1,4})\s+(.+)|^([A-Z][^\n]{3,80})\n[=\-]{3,}",
    re.MULTILINE,
)

# Scientific name quick-detect (binomial heuristic) — UPDATED to allow abbreviations
# _SPECIES_RE = re.compile(
#     r"\b([A-Z][a-z]*\.?)\s+(?:"
#     r"(?:cf\.|aff\.|sp\.|spp\.|subsp\.|var\.)\s+)?[a-z]{2,}\b"
# )
_SPECIES_RE = re.compile(
    r"\b([A-Z][a-z]{3,})"           # Genus (≥4 chars total, e.g. Elysia)
    r"(?:"
    r"\s+(?:cf\.|aff\.|sp\.n\.|sp\.|spp\.|subsp\.|var\.|n\.\s*sp\.)"
    r"(?:\s+[a-z]{3,})?"            # optional epithet after qualifier
    r"|\s+[a-z]{3,}"                # standard species epithet (binomial)
    r")?"                           # the whole species part is now OPTIONAL
    r"\b",
    re.MULTILINE,
)

# Gemma 4 context window sizes
_MODEL_CONTEXTS: dict[str, int] = {
    "gemma4":        131072,   # 128K tokens ≈ ~490 K chars (3.75 chars/tok)
    "gemma4:12b":    131072,
    "gemma3":         32768,
    "gemma3:12b":     32768,
    "llama3.2":        8192,
    "llama3.3":       32768,
    "qwen2.5:7b":     32768,
    "mistral":         8192,
}

# How many chars per token (conservative estimate)
_CHARS_PER_TOKEN = 3.5
# Reserve fraction of context for system prompt + JSON output
_CONTEXT_RESERVE  = 0.55


# ─────────────────────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id:    int
    text:        str
    section:     str  = ""
    page_range:  str  = ""
    char_start:  int  = 0
    char_end:    int  = 0
    source_file: str  = ""
    has_species: bool = False   # quick pre-filter flag
    strategy:    str  = ""      # "section" | "fixed" | "page" | "batch"

    @property
    def char_count(self) -> int:
        return len(self.text)

    def preview(self, n: int = 120) -> str:
        return self.text[:n].replace("\n", " ")


@dataclass
class ChunkStats:
    total_chunks:   int = 0
    total_chars:    int = 0
    species_chunks: int = 0
    section_names:  list[str] = field(default_factory=list)
    strategy_used:  str = ""
    parse_method:   str = ""


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION SPLITTER
# ─────────────────────────────────────────────────────────────────────────────
def split_by_sections(
    md_text: str,
    max_chars: int    = 6000,
    overlap_chars: int = 400,
    source_file: str   = "",
) -> list[Chunk]:
    """
    Split Markdown on heading boundaries.  If a section exceeds max_chars it is
    further split at paragraph boundaries (never mid-sentence).  The last
    overlap_chars of each chunk are prepended to the next to preserve context
    across boundaries (critical for species names near section headers).
    """
    # Find all heading positions
    heading_positions: list[tuple[int, str]] = []
    for m in _HEADING_RE.finditer(md_text):
        level_str = m.group(1) or ""
        title     = (m.group(2) or m.group(3) or "").strip()
        heading_positions.append((m.start(), title))

    # If no headings found, fall back to paragraph-aware fixed split
    if not heading_positions:
        return split_by_paragraphs(
            md_text, max_chars=max_chars,
            overlap_chars=overlap_chars, source_file=source_file,
        )

    # Build raw sections: [(section_title, text), ...]
    raw_sections: list[tuple[str, str]] = []
    for i, (pos, title) in enumerate(heading_positions):
        next_pos = heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(md_text)
        section_text = md_text[pos:next_pos].strip()
        raw_sections.append((title, section_text))

    # Include any preamble before the first heading
    if heading_positions[0][0] > 0:
        preamble = md_text[: heading_positions[0][0]].strip()
        if preamble:
            raw_sections.insert(0, ("Preamble", preamble))

    # Flatten over-long sections using paragraph split
    chunks:   list[Chunk] = []
    prev_tail: str         = ""
    cid       = 0

    for sec_title, sec_text in raw_sections:
        # Add overlap from previous section
        effective_text = (prev_tail + "\n\n" + sec_text).strip() if prev_tail else sec_text

        if len(effective_text) <= max_chars:
            sub_chunks = [effective_text]
        else:
            sub_chunks = _split_text_paragraphs(effective_text, max_chars, overlap_chars)

        for sc_idx, sc_text in enumerate(sub_chunks):
            has_sp = bool(_SPECIES_RE.search(sc_text))
            page_est = _estimate_page(sc_text, md_text)
            chunks.append(Chunk(
                chunk_id    = cid,
                text        = sc_text,
                section     = sec_title + (f" [{sc_idx+1}]" if len(sub_chunks) > 1 else ""),
                page_range  = page_est,
                char_start  = md_text.find(sec_text[:40]) if sc_idx == 0 else 0,
                char_end    = 0,
                source_file = source_file,
                has_species = has_sp,
                strategy    = "section",
            ))
            cid += 1

        # Carry overlap tail into next section
        prev_tail = sec_text[-overlap_chars:] if len(sec_text) > overlap_chars else sec_text

    logger.info(
        "[chunker] section split → %d chunks (overlap=%d chars)",
        len(chunks), overlap_chars,
    )
    return chunks


def split_by_paragraphs(
    text: str,
    max_chars: int    = 6000,
    overlap_chars: int = 400,
    source_file: str   = "",
) -> list[Chunk]:
    """
    Paragraph-aware fixed split.  Splits at blank-line boundaries so taxonomic
    descriptions are never broken mid-sentence.
    """
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[Chunk] = []
    buf    = ""
    cid    = 0
    prev_tail = ""

    for para in paragraphs:
        candidate = (prev_tail + "\n\n" + buf + "\n\n" + para).strip() if prev_tail else (buf + "\n\n" + para).strip()
        if len(candidate) > max_chars and buf:
            has_sp = bool(_SPECIES_RE.search(buf))
            chunks.append(Chunk(
                chunk_id    = cid,
                text        = buf.strip(),
                section     = f"Paragraph block {cid+1}",
                source_file = source_file,
                has_species = has_sp,
                strategy    = "paragraph",
            ))
            prev_tail = buf[-overlap_chars:] if len(buf) > overlap_chars else buf
            buf = para
            cid += 1
        else:
            buf = (buf + "\n\n" + para).strip() if buf else para

    if buf.strip():
        chunks.append(Chunk(
            chunk_id    = cid,
            text        = (prev_tail + "\n\n" + buf).strip() if prev_tail else buf.strip(),
            section     = f"Paragraph block {cid+1}",
            source_file = source_file,
            has_species = bool(_SPECIES_RE.search(buf)),
            strategy    = "paragraph",
        ))

    logger.info("[chunker] paragraph split → %d chunks", len(chunks))
    return chunks


def _split_text_paragraphs(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Sub-split an oversized section at paragraph boundaries."""
    paragraphs = re.split(r"\n{2,}", text)
    parts:  list[str] = []
    buf     = ""
    tail    = ""
    for para in paragraphs:
        candidate = (tail + "\n\n" + buf + "\n\n" + para).strip()
        if len(candidate) > max_chars and buf:
            parts.append((tail + "\n\n" + buf).strip() if tail else buf.strip())
            tail = buf[-overlap_chars:] if len(buf) > overlap_chars else buf
            buf  = para
        else:
            buf = (buf + "\n\n" + para).strip() if buf else para
    if buf.strip():
        parts.append((tail + "\n\n" + buf).strip() if tail else buf.strip())
    return parts or [text]


def _estimate_page(chunk_text: str, full_text: str) -> str:
    """
    Heuristically estimate page range from docling-style page markers
    like '<!-- Page 12 -->' or '---Page 12---' in the text.
    """
    page_markers = re.findall(
        r"(?:<!--\s*[Pp]age\s*(\d+)\s*-->|---[Pp]age\s*(\d+)---|\f.*?(\d+)\s*$)",
        chunk_text,
        re.MULTILINE,
    )
    pages = [int(m[0] or m[1] or m[2]) for m in page_markers if any(m)]
    if pages:
        return f"{min(pages)}-{max(pages)}" if len(pages) > 1 else str(pages[0])
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  BATCH ASSEMBLER  (Gemma 4 large-context mode)
# ─────────────────────────────────────────────────────────────────────────────
def assemble_batches(
    chunks: list[Chunk],
    model_name: str = "gemma4",
    chars_per_token: float = _CHARS_PER_TOKEN,
) -> list[list[Chunk]]:
    """
    Group section chunks into model-context-sized batches.
    Gemma 4 (128K ctx) → batch up to ~350K chars of actual text.
    Falls back to single-section batches for small-context models.

    Returns list of chunk groups, each fitting in one LLM call.
    """
    # Determine usable context in chars
    ctx_tokens   = _MODEL_CONTEXTS.get(model_name.lower().split(":")[0].replace("-",""), 8192)
    usable_chars = int(ctx_tokens * chars_per_token * _CONTEXT_RESERVE)
    logger.info(
        "[chunker] batch mode: model=%s ctx=%d tokens usable_chars=%d",
        model_name, ctx_tokens, usable_chars,
    )

    batches: list[list[Chunk]] = []
    current_batch: list[Chunk] = []
    current_chars = 0

    for chunk in chunks:
        if current_chars + chunk.char_count > usable_chars and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append(chunk)
        current_chars += chunk.char_count

    if current_batch:
        batches.append(current_batch)

    logger.info(
        "[chunker] %d chunks → %d batches (usable_chars=%d)",
        len(chunks), len(batches), usable_chars,
    )
    return batches


def batch_to_text(batch: list[Chunk]) -> str:
    """
    Merge a batch of chunks into a single text block with section headers
    preserved so the LLM can distinguish locality/context per section.
    """
    parts = []
    for c in batch:
        header = f"\n\n## [{c.section}] (chunk {c.chunk_id})\n" if c.section else ""
        parts.append(header + c.text)
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  PDF → MARKDOWN  (Docling-first, fallback chain)
# ─────────────────────────────────────────────────────────────────────────────
def pdf_to_structured_markdown(
    pdf_path: str,
    use_docling:    bool = True,
    use_vision:     bool = False,
    vision_dpi:     int  = 150,
    ollama_base_url: str = "http://localhost:11434",
) -> tuple[str, str]:
    """
    Convert a PDF to Markdown.  Returns (markdown_text, parse_method_used).

    Priority:
      1. Docling   — best structure preservation, IBM granite-docling
      2. Vision-Parse — for figure-heavy pages (DPI-configurable)
      3. pymupdf4llm  — fast, pure-Python
      4. markitdown   — lightweight Microsoft fallback
      5. raw fitz     — last resort text extraction
    """
    path = Path(pdf_path)
    if not path.exists():
        return f"[File not found: {pdf_path}]", "error"

    n_pages = _count_pages(pdf_path)
    logger.info("[chunker] %s — %d pages, use_docling=%s use_vision=%s",
                path.name, n_pages, use_docling, use_vision)

    # ── 1. Docling ────────────────────────────────────────────────────────────
    if use_docling and _DOCLING_AVAILABLE:
        try:
            logger.info("[chunker] Docling converting %s…", path.name)
            conv   = _DoclingConverter()
            result = conv.convert(str(path))
            md     = result.document.export_to_markdown()
            if md.strip():
                logger.info("[chunker] Docling OK (%d chars)", len(md))
                return md, "docling"
        except Exception as exc:
            logger.warning("[chunker] Docling failed: %s", exc)

    # ── 2. Vision-Parse (figure/table-heavy pages) ────────────────────────────
    if use_vision and _VISIONPARSE_AVAILABLE and n_pages <= 200:
        try:
            logger.info("[chunker] vision_parse OCR…")
            vp  = VisionParser(dpi=vision_dpi)
            md  = vp.parse_pdf(str(path))
            if isinstance(md, str) and md.strip():
                return md, "vision_parse"
        except Exception as exc:
            logger.warning("[chunker] vision_parse failed: %s", exc)

    # ── 3. pymupdf4llm ────────────────────────────────────────────────────────
    if _PYMUPDF4LLM_AVAILABLE:
        try:
            md = _pymupdf4llm.to_markdown(str(path))
            if md.strip():
                return md, "pymupdf4llm"
        except Exception as exc:
            logger.warning("[chunker] pymupdf4llm failed: %s", exc)

    # ── 4. markitdown ─────────────────────────────────────────────────────────
    if _MARKITDOWN_AVAILABLE:
        try:
            mk = _MarkItDown()
            md = mk.convert(str(path)).text_content
            if md.strip():
                return md, "markitdown"
        except Exception as exc:
            logger.warning("[chunker] markitdown failed: %s", exc)

    # ── 5. raw fitz ───────────────────────────────────────────────────────────
    if _PYMUPDF_AVAILABLE:
        try:
            doc  = _fitz.open(str(path))
            text = "\n\f".join(page.get_text() for page in doc)
            doc.close()
            return text, "fitz_raw"
        except Exception as exc:
            logger.warning("[chunker] fitz raw failed: %s", exc)

    return f"[Could not extract text from {path.name}]", "failed"


def _count_pages(pdf_path: str) -> int:
    if _PYMUPDF_AVAILABLE:
        try:
            doc = _fitz.open(pdf_path)
            n   = len(doc)
            doc.close()
            return n
        except Exception:
            pass
    if _PYMUPDF4LLM_AVAILABLE:
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            n   = len(doc)
            doc.close()
            return n
        except Exception:
            pass
    return -1


# ─────────────────────────────────────────────────────────────────────────────
#  LANGEXTRACT BRIDGE
# ─────────────────────────────────────────────────────────────────────────────
def langextract_chunk(
    chunk_text: str,
    schema: dict | None = None,
) -> dict | None:
    """
    Run Google LangExtract structured extraction on a single chunk.
    Returns a dict of extracted fields or None if unavailable/failed.

    Falls back gracefully when langextract is not installed.
    """
    if not _LANGEXTRACT_AVAILABLE:
        return None
    default_schema = schema or {
        "species_names":   "list of scientific species names mentioned",
        "localities":      "list of geographic place names",
        "habitats":        "list of habitat types mentioned",
        "dates_mentioned": "list of collection or observation dates",
    }
    try:
        result = _langextract.extract(chunk_text, schema=default_schema)
        return result if isinstance(result, dict) else None
    except Exception as exc:
        logger.debug("[chunker] langextract: %s", exc)
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PUBLIC CLASS
# ─────────────────────────────────────────────────────────────────────────────
class DocumentChunker:
    """
    High-level interface: PDF/Markdown → list[Chunk] ready for LLM extraction.

    Handles:
      • 1-page handouts  → single chunk
      • 30-page papers   → section-aware chunks, 3–6 K chars each
      • 300-page theses  → docling parse → section chunks → Gemma-4 batches
      • 1000-page atlases→ docling + vision page-by-page, then section merge
    """

    def __init__(
        self,
        strategy:       str  = "section",   # "section" | "fixed" | "page" | "batch"
        chunk_chars:    int  = 6000,
        overlap_chars:  int  = 400,
        model_name:     str  = "gemma4",
        batch_mode:     bool = False,        # True → assemble large batches for big-context models
        use_docling:    bool = True,
        use_vision:     bool = False,
        vision_dpi:     int  = 150,
        ollama_base_url: str = "http://localhost:11434",
        
    ):
        self.strategy        = strategy
        self.chunk_chars     = chunk_chars
        self.overlap_chars   = overlap_chars
        self.model_name      = model_name
        self.batch_mode      = batch_mode
        self.use_docling     = use_docling
        self.use_vision      = use_vision
        self.vision_dpi      = vision_dpi
        self.ollama_base_url = ollama_base_url

        # Removed forced auto-batching. We now strictly respect the UI toggle (batch_mode).
        # FIX: Moved this block inside __init__ — it referenced 'self' at class body level,
        #      causing NameError: name 'self' is not defined on import.
        if self.batch_mode:
            logger.info(
                "[chunker] batch mode enabled for model '%s'", self.model_name
            )

    # ── PDF file entry point ──────────────────────────────────────────────────
    def chunk_file(
        self, pdf_path: str, source_label: str = ""
    ) -> tuple[list[Chunk], ChunkStats]:
        """
        Full pipeline:  PDF → markdown → section chunks → (optionally) batches.
        Returns (chunks, stats).
        """
        label = source_label or Path(pdf_path).name
        md_text, parse_method = pdf_to_structured_markdown(
            pdf_path,
            use_docling    = self.use_docling,
            use_vision     = self.use_vision,
            vision_dpi     = self.vision_dpi,
            ollama_base_url= self.ollama_base_url,
        )
        chunks, stats = self.chunk_markdown(md_text, source_label=label)
        stats.parse_method = parse_method
        return chunks, stats

    # ── Markdown entry point ──────────────────────────────────────────────────
    def chunk_markdown(
        self, md_text: str, source_label: str = ""
    ) -> tuple[list[Chunk], ChunkStats]:
        """
        Chunk pre-extracted Markdown text into logical segments.
        """
        if not md_text.strip():
            return [], ChunkStats(strategy_used=self.strategy)

        # Normalise Windows line endings
        md_text = md_text.replace("\r\n", "\n").replace("\r", "\n")

        if self.strategy in ("section", "auto"):
            chunks = split_by_sections(
                md_text,
                max_chars     = self.chunk_chars,
                overlap_chars = self.overlap_chars,
                source_file   = source_label,
            )
        elif self.strategy == "fixed":
            chunks = self._fixed_chunks(md_text, source_label)
        else:
            chunks = split_by_sections(md_text, self.chunk_chars, self.overlap_chars, source_label)

        # If large-context batch mode: re-group chunks into model-sized batches
        if self.batch_mode and len(chunks) > 1:
            batches = assemble_batches(chunks, model_name=self.model_name)
            batch_chunks: list[Chunk] = []
            for bi, batch in enumerate(batches):
                merged_text = batch_to_text(batch)
                sections    = [c.section for c in batch]
                pages       = [c.page_range for c in batch if c.page_range]
                batch_chunks.append(Chunk(
                    chunk_id    = bi,
                    text        = merged_text,
                    section     = f"Batch {bi+1} ({len(batch)} sections)",
                    page_range  = f"{pages[0]}-{pages[-1]}" if pages else "",
                    source_file = source_label,
                    has_species = any(c.has_species for c in batch),
                    strategy    = "batch",
                ))
            chunks = batch_chunks
            logger.info("[chunker] batch merge: %d chunks → %d batches", len(chunks), len(batches))

        stats = ChunkStats(
            total_chunks   = len(chunks),
            total_chars    = sum(c.char_count for c in chunks),
            species_chunks = sum(1 for c in chunks if c.has_species),
            section_names  = [c.section for c in chunks[:20]],
            strategy_used  = self.strategy + ("+batch" if self.batch_mode else ""),
        )

        logger.info(
            "[chunker] %d chunks | %d chars | %d species-flagged | strategy=%s",
            stats.total_chunks, stats.total_chars,
            stats.species_chunks, stats.strategy_used,
        )
        return chunks, stats

    def _fixed_chunks(self, text: str, source_label: str) -> list[Chunk]:
        """Simple fixed-size split — used only when explicitly requested."""
        chunks = []
        step   = max(self.chunk_chars - self.overlap_chars, 500)
        for i, start in enumerate(range(0, len(text), step)):
            end    = min(start + self.chunk_chars, len(text))
            chunk_text = text[start:end]
            chunks.append(Chunk(
                chunk_id    = i,
                text        = chunk_text,
                section     = f"Chunk {i+1}",
                char_start  = start,
                char_end    = end,
                source_file = source_label,
                has_species = bool(_SPECIES_RE.search(chunk_text)),
                strategy    = "fixed",
            ))
        return chunks

    # ── Convenience iterator ──────────────────────────────────────────────────
    def iter_chunks(
        self, pdf_path: str, source_label: str = ""
    ) -> Iterator[tuple[Chunk, ChunkStats]]:
        """Yield (chunk, stats) one at a time — memory-efficient for large docs."""
        chunks, stats = self.chunk_file(pdf_path, source_label)
        for chunk in chunks:
            yield chunk, stats


# ─────────────────────────────────────────────────────────────────────────────
#  AVAILABILITY REPORT
# ─────────────────────────────────────────────────────────────────────────────
def availability_report() -> dict[str, bool]:
    return {
        "docling":         _DOCLING_AVAILABLE,
        "vision_parse":    _VISIONPARSE_AVAILABLE,
        "langextract":     _LANGEXTRACT_AVAILABLE,
        "pymupdf4llm":     _PYMUPDF4LLM_AVAILABLE,
        "markitdown":      _MARKITDOWN_AVAILABLE,
        "fitz_raw":        _PYMUPDF_AVAILABLE,
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print("=== DocumentChunker availability ===")
    for k, v in availability_report().items():
        print(f"  {'✅' if v else '❌'}  {k}")

    if len(sys.argv) > 1:
        pdf = sys.argv[1]
        model = sys.argv[2] if len(sys.argv) > 2 else "gemma4"
        chunker = DocumentChunker(
            strategy="section", model_name=model,
            use_docling=True, use_vision=False,
        )
        chunks, stats = chunker.chunk_file(pdf)
        print(f"\n{stats.total_chunks} chunks | {stats.total_chars} chars "
              f"| {stats.species_chunks} flagged | {stats.parse_method}")
        for c in chunks[:5]:
            print(f"  [{c.chunk_id}] {c.section[:50]:<50}  {c.char_count:>6} chars  "
                  f"{'🐠' if c.has_species else '  '}")