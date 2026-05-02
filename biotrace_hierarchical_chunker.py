"""
biotrace_hierarchical_chunker.py  —  BioTrace v5.3
────────────────────────────────────────────────────────────────────────────
Hierarchical Late-Chunking to circumvent the classic RAG "Chunking Problem."

The Problem
───────────
Traditional fixed-size or section-based chunking severs the link between a
species name and its locality when they appear in different sentences, or
cuts a Methods table away from the Results paragraph that cites it.

The Solution: THREE-LEVEL HIERARCHY
────────────────────────────────────
Level 0  SECTION    — Markdown heading block (##, ###).  High-level context.
                      Used for: retrieving the "chapter" a name came from.
Level 1  PARAGRAPH  — Blank-line-separated block within a section.
                      Used for: associating locality sentences with species.
Level 2  SENTENCE   — Individual sentence within a paragraph.
                      Used for: precise "Raw Text Evidence" quotes.

"Late Chunking" principle (Wang et al. 2024):
  • Store all three levels in SQLite for each document.
  • At extraction time, for each Level-2 (sentence) hit, automatically
    attach its Level-0 (section) context and Level-1 (paragraph) context.
  • The LLM sees: SECTION header + PARAGRAPH context + TARGET SENTENCE.
    This gives it the locality/date/method context without overwhelming the
    context window with unrelated pages.

Species-Locality Proximity Linking
───────────────────────────────────
After chunking, a proximity pass scans consecutive sentence-level chunks
to link species names with locality strings within a sliding window of
±3 sentences. This pre-fills verbatimLocality before the LLM is called,
dramatically reducing hallucination of localities.

Database schema (SQLite, per document):
    chunks(
      id            — auto
      doc_hash      — SHA-1 of source file
      level         — 0=section, 1=paragraph, 2=sentence
      chunk_id      — sequential within level
      section       — nearest heading text
      parent_para   — parent paragraph text (for sentences)
      text          — chunk text
      char_start    — character offset in full document
      char_end
      page_est      — estimated page number
      has_species   — 1 if likely contains a scientific name
      has_locality  — 1 if likely contains a place name
    )

Usage:
    from biotrace_hierarchical_chunker import HierarchicalChunker

    chunker = HierarchicalChunker(db_path="biodiversity_data/chunks.db")
    doc_hash = chunker.ingest(markdown_text, source_label="Thesis Chapter 3")

    # Get enriched extraction batches (each batch = section + relevant paragraphs)
    for batch in chunker.extraction_batches(doc_hash, window_sentences=3):
        prompt_text = batch["context"]          # section + paragraphs + sentences
        section     = batch["section"]
        pre_locs    = batch["candidate_localities"]   # pre-linked localities
        # → call LLM with prompt_text

    # Retrieve by level for analysis
    sentences  = chunker.get_level(doc_hash, level=2)
    paragraphs = chunker.get_level(doc_hash, level=1)
    sections   = chunker.get_level(doc_hash, level=0)
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import re as _re

import sqlite3
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger("biotrace.hierarchical_chunker")

# ─────────────────────────────────────────────────────────────────────────────
#  REGEX PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
_HEADING_RE   = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_SENTENCE_RE  = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+(?=[A-Z])"
)
# _SPECIES_SIGNAL = re.compile(
#     r"\b[A-Z][a-z]{2,}\s+(?:cf\.|aff\.|sp\.|spp\.|var\.|)?[a-z]{3,}\b"
# )

_SPECIES_SIGNAL = re.compile(
    r"\b[A-Z][a-z]{3,}"             # Genus ≥4 chars
    r"(?:"
    r"\s+(?:cf\.|aff\.|sp\.n\.|sp\.|spp\.|subsp\.|var\.|n\.\s*sp\.)"
    r"(?:\s+[a-z]{3,})?"
    r"|\s+[a-z]{3,}"
    r")?\b",
    re.MULTILINE,
)

_LOCALITY_SIGNAL = re.compile(
    r"\b(?:at|from|in|near|off|along|around|collected at|recorded from|"
    r"observed at|station|site|reef|island|gulf|bay|coast|creek|river|lake)\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
#  DB SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_hash     TEXT NOT NULL,
    level        INTEGER NOT NULL,    -- 0=section 1=para 2=sentence
    chunk_id     INTEGER NOT NULL,
    section      TEXT DEFAULT '',
    parent_para  TEXT DEFAULT '',
    text         TEXT NOT NULL,
    char_start   INTEGER DEFAULT 0,
    char_end     INTEGER DEFAULT 0,
    page_est     TEXT DEFAULT '',
    has_species  INTEGER DEFAULT 0,
    has_locality INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_level ON chunks(doc_hash, level);
CREATE INDEX IF NOT EXISTS idx_chunks_species   ON chunks(doc_hash, has_species);

CREATE TABLE IF NOT EXISTS docs (
    doc_hash   TEXT PRIMARY KEY,
    label      TEXT,
    n_sections INTEGER DEFAULT 0,
    n_paras    INTEGER DEFAULT 0,
    n_sentences INTEGER DEFAULT 0,
    ingested_at TEXT DEFAULT (datetime('now'))
);
"""


# ─────────────────────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class HChunk:
    level:       int
    chunk_id:    int
    section:     str
    parent_para: str
    text:        str
    char_start:  int = 0
    char_end:    int = 0
    page_est:    str = ""
    has_species: bool = False
    has_locality:bool = False

    @property
    def preview(self) -> str:
        return self.text[:100].replace("\n", " ") + "…" if len(self.text) > 100 else self.text


@dataclass
class ExtractionBatch:
    """A single prompt-ready batch for LLM extraction."""
    doc_hash:            str
    section:             str
    section_text:        str
    paragraph_text:      str
    sentence_texts:      list[str]
    candidate_localities:list[str]  = field(default_factory=list)
    candidate_species:   list[str]  = field(default_factory=list)
    char_start:          int = 0
    char_end:            int = 0
    page_est:            str = ""

    @property
    def context(self) -> str:
        """
        Build a structured context string for the LLM prompt.
        Layout:
          [SECTION: <heading>]
          <section intro — first 400 chars>

          [PARAGRAPH CONTEXT:]
          <paragraph text>

          [TARGET SENTENCES:]
          <sentence 1>
          <sentence 2>
          ...
        """
        parts = []
        if self.section:
            parts.append(f"[SECTION: {self.section}]")
        if self.section_text.strip():
            intro = self.section_text.strip()[:400]
            parts.append(intro)
        if self.paragraph_text.strip():
            parts.append(f"\n[PARAGRAPH CONTEXT:]\n{self.paragraph_text.strip()}")
        if self.sentence_texts:
            parts.append(f"\n[TARGET SENTENCES:]\n" + "\n".join(self.sentence_texts))
        if self.candidate_localities:
            parts.append(
                f"\n[PRE-LINKED LOCALITIES (from Methods)]: "
                + "; ".join(self.candidate_localities[:5])
            )
        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  SPLITTERS
# ─────────────────────────────────────────────────────────────────────────────
def _split_sections(text: str) -> list[tuple[str, str, int, int]]:
    """
    Split on Markdown headings.
    Returns list of (heading_text, body_text, char_start, char_end).
    """
    boundaries: list[tuple[int, str]] = [(0, "Preamble")]
    for m in _HEADING_RE.finditer(text):
        boundaries.append((m.start(), m.group(2).strip()))

    sections: list[tuple[str, str, int, int]] = []
    for i, (start, heading) in enumerate(boundaries):
        end   = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        body  = text[start:end].strip()
        # Remove the heading line itself from body
        body  = _HEADING_RE.sub("", body, count=1).strip()
        if body:
            sections.append((heading, body, start, end))
    return sections


def _split_paragraphs(text: str, offset: int = 0) -> list[tuple[str, int, int]]:
    """Split on blank lines. Returns (para_text, abs_start, abs_end)."""
    paras: list[tuple[str, int, int]] = []
    for m in re.finditer(r"(.+?)(?:\n\s*\n|$)", text, re.DOTALL):
        para = m.group(1).strip()
        if len(para) > 20:
            paras.append((para, offset + m.start(), offset + m.end()))
    return paras


def _split_sentences(text: str, offset: int = 0) -> list[tuple[str, int, int]]:
    """Sentence split. Returns (sentence, abs_start, abs_end)."""
    # Simple but effective: split on punctuation + capital
    raw_sents = re.split(r"(?<=[.!?])\s+", text)
    sents: list[tuple[str, int, int]] = []
    pos = offset
    for s in raw_sents:
        s_clean = s.strip()
        if len(s_clean) > 10:
            sents.append((s_clean, pos, pos + len(s)))
        pos += len(s) + 1
    return sents


def _estimate_page(char_pos: int, doc_len: int, total_pages: int = 1) -> str:
    if doc_len == 0 or total_pages == 0:
        return ""
    fraction = char_pos / doc_len
    page     = max(1, round(fraction * total_pages))
    return str(page)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────
class HierarchicalChunker:
    """
    Three-level hierarchical chunker with SQLite persistence.

    Levels:
      0 = section  (Markdown heading block)
      1 = paragraph (blank-line-separated)
      2 = sentence  (punctuation-split)
    """

    def __init__(self, db_path: str = "biodiversity_data/chunks.db"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self._conn   = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_DDL)
        self._conn.commit()

    # ── Ingest ────────────────────────────────────────────────────────────────
    def ingest(
        self,
        text:           str,
        source_label:   str = "",
        total_pages:    int = 1,
        force_reingest: bool = False,
    ) -> str:
        """
        Parse and store a document at all three levels.
        Returns the doc_hash (used for retrieval).
        """
        doc_hash = hashlib.sha1(text.encode()).hexdigest()[:16]

        # Skip if already ingested
        existing = self._conn.execute(
            "SELECT n_sentences FROM docs WHERE doc_hash=?", (doc_hash,)
        ).fetchone()
        if existing and not force_reingest:
            logger.info("[hchunk] Already ingested: %s (%d sentences)", source_label, existing[0])
            return doc_hash

        # Remove stale data
        self._conn.execute("DELETE FROM chunks WHERE doc_hash=?", (doc_hash,))

        doc_len = len(text)
        rows: list[tuple] = []
        n_sec = n_para = n_sent = 0

        # Level 0: sections
        sections = _split_sections(text)
        if not sections:
            sections = [("Document", text, 0, len(text))]

        for sec_heading, sec_body, sec_start, sec_end in sections:
            page_s = _estimate_page(sec_start, doc_len, total_pages)
            has_sp = bool(_SPECIES_SIGNAL.search(sec_body))
            has_lo = bool(_LOCALITY_SIGNAL.search(sec_body))
            rows.append((
                doc_hash, 0, n_sec, sec_heading, "",
                sec_body[:2000],   # truncate section body stored at level 0
                sec_start, sec_end, page_s,
                int(has_sp), int(has_lo),
            ))
            n_sec += 1

            # Level 1: paragraphs within section
            paras = _split_paragraphs(sec_body, offset=sec_start)
            for para_text, pa_start, pa_end in paras:
                page_p = _estimate_page(pa_start, doc_len, total_pages)
                has_sp_p = bool(_SPECIES_SIGNAL.search(para_text))
                has_lo_p = bool(_LOCALITY_SIGNAL.search(para_text))
                rows.append((
                    doc_hash, 1, n_para, sec_heading, "",
                    para_text,
                    pa_start, pa_end, page_p,
                    int(has_sp_p), int(has_lo_p),
                ))
                para_id = n_para
                n_para += 1

                # Level 2: sentences within paragraph
                sents = _split_sentences(para_text, offset=pa_start)
                for sent_text, s_start, s_end in sents:
                    page_se = _estimate_page(s_start, doc_len, total_pages)
                    has_sp_s = bool(_SPECIES_SIGNAL.search(sent_text))
                    has_lo_s = bool(_LOCALITY_SIGNAL.search(sent_text))
                    rows.append((
                        doc_hash, 2, n_sent, sec_heading, para_text[:300],
                        sent_text,
                        s_start, s_end, page_se,
                        int(has_sp_s), int(has_lo_s),
                    ))
                    n_sent += 1

        # Batch insert
        self._conn.executemany(
            """INSERT INTO chunks
               (doc_hash,level,chunk_id,section,parent_para,text,
                char_start,char_end,page_est,has_species,has_locality)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self._conn.execute(
            """INSERT OR REPLACE INTO docs
               (doc_hash, label, n_sections, n_paras, n_sentences)
               VALUES (?,?,?,?,?)""",
            (doc_hash, source_label, n_sec, n_para, n_sent),
        )
        self._conn.commit()

        logger.info(
            "[hchunk] Ingested '%s': %d sections, %d paragraphs, %d sentences",
            source_label, n_sec, n_para, n_sent,
        )
        return doc_hash

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def get_level(self, doc_hash: str, level: int) -> list[HChunk]:
        rows = self._conn.execute(
            """SELECT level, chunk_id, section, parent_para, text,
                      char_start, char_end, page_est, has_species, has_locality
               FROM chunks
               WHERE doc_hash=? AND level=?
               ORDER BY chunk_id""",
            (doc_hash, level),
        ).fetchall()
        return [
            HChunk(
                level=r[0], chunk_id=r[1], section=r[2], parent_para=r[3],
                text=r[4], char_start=r[5], char_end=r[6], page_est=r[7],
                has_species=bool(r[8]), has_locality=bool(r[9]),
            )
            for r in rows
        ]

    def _nearby_localities(
        self,
        doc_hash: str,
        sentence_chunk_id: int,
        window: int = 5,
    ) -> list[str]:
        """Return locality candidate sentences within ±window of a given sentence."""
        rows = self._conn.execute(
            """SELECT text FROM chunks
               WHERE doc_hash=? AND level=2
                 AND has_locality=1
                 AND chunk_id BETWEEN ? AND ?
               ORDER BY chunk_id""",
            (doc_hash, max(0, sentence_chunk_id - window),
             sentence_chunk_id + window),
        ).fetchall()
        return [r[0] for r in rows]

    # ── Extraction batches ────────────────────────────────────────────────────
    def extraction_batches(
        self,
        doc_hash: str,
        window_sentences: int = 5,
        max_batch_chars:  int = 6000,
        species_only:     bool = False,
    ) -> Iterator[ExtractionBatch]:
        """
        Yield ExtractionBatch objects.  Each batch covers one PARAGRAPH and its
        sentences, enriched with section context and pre-linked localities.

        For very long paragraphs, splits into sub-batches of max_batch_chars.
        """
        sections = {
            r[2]: r[4]   # heading → first 400 chars of section body
            for r in self._conn.execute(
                "SELECT level,chunk_id,section,parent_para,text FROM chunks "
                "WHERE doc_hash=? AND level=0 ORDER BY chunk_id",
                (doc_hash,),
            ).fetchall()
        }

        paragraphs = self._conn.execute(
            """SELECT chunk_id, section, text, char_start, char_end, page_est,
                      has_species, has_locality
               FROM chunks
               WHERE doc_hash=? AND level=1
               ORDER BY chunk_id""",
            (doc_hash,),
        ).fetchall()

        # Pre-fetch all sentences for this document to avoid N+1 queries
        # sentence_row: (chunk_id, text, char_start, char_end, has_species, has_locality, parent_para)
        all_sentences = self._conn.execute(
            """SELECT chunk_id, text, char_start, char_end, has_species, has_locality, parent_para
               FROM chunks
               WHERE doc_hash=? AND level=2
               ORDER BY chunk_id""",
            (doc_hash,),
        ).fetchall()

        # Index sentences by parent_para (first 300 chars)
        sents_by_para: dict[str, list[tuple]] = {}
        # Also keep a flat list for range-based fallback and nearby localities
        # We need chunk_id to be the index for fast lookup if possible, but they might not be contiguous 0..N
        # Using a list and knowing they are ordered by chunk_id.
        for s_row in all_sentences:
            p_para = s_row[6]
            if p_para not in sents_by_para:
                sents_by_para[p_para] = []
            sents_by_para[p_para].append(s_row)

        for para_row in paragraphs:
            p_id, p_section, p_text, p_start, p_end, p_page, p_sp, p_lo = para_row

            if species_only and not p_sp:
                continue

            # Get sentences belonging to this paragraph
            # (match by parent_para truncated to 300 chars)
            sents = sents_by_para.get(p_text[:300], [])

            if not sents:
                # Fallback: get sentences within char range
                sents = [
                    s for s in all_sentences
                    if s[2] >= p_start and s[3] <= p_end
                ]

            if not sents:
                continue

            # Pre-link localities from nearby sentences
            # sents is list of (chunk_id, text, char_start, char_end, has_species, has_locality, parent_para)
            all_sent_ids = [s[0] for s in sents]
            mid_id = all_sent_ids[len(all_sent_ids) // 2]

            # Optimized nearby localities: filter from all_sentences in memory
            # chunk_id is sequential within level, so we can use it for range filtering
            pre_locs = [
                s[1] for s in all_sentences
                if s[5] == 1 and max(0, mid_id - window_sentences) <= s[0] <= mid_id + window_sentences
            ]

            # Pre-detect species candidates from paragraph
            sp_candidates = _SPECIES_SIGNAL.findall(p_text)

            # Split into sub-batches if paragraph is very long
            sent_texts = [s[1] for s in sents]
            batch_text = "\n".join(sent_texts)

            if len(batch_text) > max_batch_chars:
                # Sub-batch: sliding window of sentences
                step = max(1, len(sents) // max(1, len(batch_text) // max_batch_chars))
                for i in range(0, len(sents), step):
                    sub_sents = sents[i:i + step]
                    yield ExtractionBatch(
                        doc_hash            = doc_hash,
                        section             = p_section,
                        section_text        = sections.get(p_section, "")[:400],
                        paragraph_text      = p_text[:500],
                        sentence_texts      = [s[1] for s in sub_sents],
                        candidate_localities= pre_locs,
                        candidate_species   = sp_candidates,
                        char_start          = sub_sents[0][2] if sub_sents else p_start,
                        char_end            = sub_sents[-1][3] if sub_sents else p_end,
                        page_est            = p_page,
                    )
            else:
                yield ExtractionBatch(
                    doc_hash            = doc_hash,
                    section             = p_section,
                    section_text        = sections.get(p_section, "")[:400],
                    paragraph_text      = p_text[:500],
                    sentence_texts      = sent_texts,
                    candidate_localities= pre_locs,
                    candidate_species   = sp_candidates,
                    char_start          = p_start,
                    char_end            = p_end,
                    page_est            = p_page,
                )

    # ── Stats ─────────────────────────────────────────────────────────────────
    def doc_stats(self, doc_hash: str) -> dict:
        row = self._conn.execute(
            "SELECT label, n_sections, n_paras, n_sentences FROM docs WHERE doc_hash=?",
            (doc_hash,),
        ).fetchone()
        if not row:
            return {}
        n_sp = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_hash=? AND has_species=1 AND level=2",
            (doc_hash,),
        ).fetchone()[0]
        n_lo = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_hash=? AND has_locality=1 AND level=2",
            (doc_hash,),
        ).fetchone()[0]
        return {
            "label":      row[0],
            "sections":   row[1],
            "paragraphs": row[2],
            "sentences":  row[3],
            "species_sentences":   n_sp,
            "locality_sentences":  n_lo,
        }

    def list_documents(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT doc_hash, label, n_sections, n_paras, n_sentences, ingested_at FROM docs"
        ).fetchall()
        return [
            {"hash":r[0],"label":r[1],"sections":r[2],
             "paragraphs":r[3],"sentences":r[4],"ingested_at":r[5]}
            for r in rows
        ]
        
    _TABLE_ROW_RE = _re.compile(r"^\|.*\|", _re.MULTILINE)
    _TABLE_BLOCK_RE = _re.compile(
        r"(\|[^\n]+\|\n)(\|[-:| ]+\|\n)((?:\|[^\n]+\|\n)+)",
        _re.MULTILINE,
    )

    def _split_preserving_tables(self, text: str, target_chars: int) -> list[str]:
        """
        Split text into chunks of ~target_chars, but NEVER split inside a
        Markdown table block. If a table is larger than target_chars, it is
        returned as a single oversized chunk with a header prefix.
        """
        # Mark table spans
        table_spans = [(m.start(), m.end()) for m in _TABLE_BLOCK_RE.finditer(text)]

        chunks: list[str] = []
        pos = 0
        current: list[str] = []
        current_len = 0

        lines = text.splitlines(keepends=True)
        line_pos = 0

        for line in lines:
            in_table = any(s <= line_pos < e for s, e in table_spans)

            if current_len + len(line) > target_chars and not in_table and current:
                chunks.append("".join(current))
                current = []
                current_len = 0

            current.append(line)
            current_len += len(line)
            line_pos += len(line)

        if current:
            chunks.append("".join(current))

        return [c for c in chunks if c.strip()]

    def close(self):
        self._conn.close()
