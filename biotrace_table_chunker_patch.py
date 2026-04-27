"""
biotrace_table_chunker_patch.py  —  BioTrace v5.4
────────────────────────────────────────────────────────────────────────────
Table-aware chunking extension for HierarchicalChunker / ScientificPaperChunker.

PROBLEM
───────
Standard chunkers split on character count, which can bisect a Markdown table
mid-row. For a checklist paper like the Gulf of Kutch opisthobranch survey,
Table 1 lists all 33 species. If a chunk boundary falls inside the table, the
LLM only sees a partial species list and misses the remainder.

Additionally, when a table is split, the COLUMN HEADERS (which tell the LLM
what each checkmark means) are lost from the second half-chunk.

SOLUTION
────────
This module provides:

  1. TableAwareChunker  — standalone chunker that never cuts inside a table.
     Tables larger than the chunk budget are emitted as a single oversized
     chunk with the section heading prepended.

  2. inject_table_context()  — post-processor for existing chunk lists.
     Finds chunks that start mid-table (no header row) and prepends the
     nearest preceding table header.

  3. enrich_chunk_with_table_metadata()  — adds metadata flags to chunks
     so the LLM prompt can be adapted when a table-dominant chunk is detected.

INTEGRATION (biotrace_v5.py)
─────────────────────────────
In extract_occurrences(), after the Priority 1/2/3/4 chunking waterfall
(approximately line 1639), add:

    from biotrace_table_chunker_patch import inject_table_context, enrich_chunk_with_table_metadata
    all_chunks_raw = batches if batches else flat_chunks
    all_chunks = inject_table_context(all_chunks_raw, markdown_text)
    all_chunks = [enrich_chunk_with_table_metadata(c) for c in all_chunks]

Then in process_chunk() (~line 1413), BEFORE the prompt is assembled, add:

    if getattr(chunk, 'is_table_dominant', False):
        text = (
            f"[TABLE CHUNK — {getattr(chunk, 'table_row_count', '?')} rows detected]\n"
            f"Column headers: {getattr(chunk, 'table_header', 'unknown')}\n\n"
        ) + text
"""
from __future__ import annotations

import logging
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Any

logger = logging.getLogger("biotrace.table_chunker")

# ─────────────────────────────────────────────────────────────────────────────
#  TABLE DETECTION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

# Markdown table row: starts and ends with |
_TABLE_ROW_RE  = re.compile(r"^\|.+\|", re.MULTILINE)
# Separator row: | --- | :---: | etc.
_TABLE_SEP_RE  = re.compile(r"^\|[\s\-:|]+\|", re.MULTILINE)
# A full table block: header + separator + ≥1 data rows
_TABLE_BLOCK_RE = re.compile(
    r"(?m)^(\|[^\n]+\|\n)(\|[\s\-:|]+\|\n)((?:\|[^\n]+\|\n?)+)",
)

# Pipe density threshold above which a chunk is "table-dominant"
_TABLE_DENSITY_THRESHOLD = 0.07   # fraction of chars that are '|'

# Minimum rows to consider a block a real table (not a two-cell emphasis)
_MIN_TABLE_ROWS = 3


@dataclass
class TableBlock:
    """A detected Markdown table block within the full document text."""
    start:      int          # char offset in full document
    end:        int          # char offset in full document
    text:       str          # full table text
    header:     str          # first row (column labels)
    row_count:  int          # data rows (excluding header + separator)
    section:    str = ""     # nearest preceding section heading


def detect_tables(full_text: str) -> list[TableBlock]:
    """
    Detect all Markdown table blocks in the full document text.

    Returns list of TableBlock objects sorted by start position.
    Only tables with ≥ MIN_TABLE_ROWS data rows are returned.
    """
    tables: list[TableBlock] = []
    section_heading = "Unknown Section"

    for m in _TABLE_BLOCK_RE.finditer(full_text):
        header_row  = m.group(1).strip()
        data_block  = m.group(3)
        data_rows   = [r for r in data_block.splitlines() if r.strip().startswith("|")]
        row_count   = len(data_rows)

        if row_count < _MIN_TABLE_ROWS:
            continue

        # Find nearest preceding section heading
        preceding = full_text[max(0, m.start() - 500): m.start()]
        headings  = re.findall(r"^#{1,4}\s+(.+)$", preceding, re.MULTILINE)
        if headings:
            section_heading = headings[-1].strip()

        tables.append(TableBlock(
            start     = m.start(),
            end       = m.end(),
            text      = m.group(0),
            header    = header_row,
            row_count = row_count,
            section   = section_heading,
        ))
        logger.debug(
            "[table_chunker] Table detected: section='%s', rows=%d, chars=%d",
            section_heading, row_count, len(m.group(0)),
        )

    return tables


def _char_in_table(pos: int, tables: list[TableBlock]) -> Optional[TableBlock]:
    """Return the TableBlock if character position pos falls inside a table."""
    for t in tables:
        if t.start <= pos < t.end:
            return t
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  TABLE-AWARE SPLITTING
# ─────────────────────────────────────────────────────────────────────────────

def split_text_preserving_tables(
    text:        str,
    target_chars:int = 6000,
    overlap_chars:int = 400,
) -> list[str]:
    """
    Split text into chunks of ~target_chars without splitting inside tables.

    Tables larger than target_chars are emitted as a single oversized chunk
    (prefixed with a [TABLE OVERSIZED] note). The LLM receives the whole table
    so no species row is ever cut.

    Parameters
    ----------
    text          : full document markdown text
    target_chars  : target chunk size in characters
    overlap_chars : character overlap between consecutive chunks

    Returns list of chunk strings.
    """
    tables = detect_tables(text)
    table_ranges = [(t.start, t.end) for t in tables]

    def _in_any_table(pos: int) -> bool:
        return any(s <= pos < e for s, e in table_ranges)

    lines     = text.splitlines(keepends=True)
    chunks:   list[str]  = []
    current:  list[str]  = []
    cur_len   = 0
    line_pos  = 0

    for line in lines:
        line_end = line_pos + len(line)

        # If we're inside a table, never split — accumulate the whole table
        in_table = _in_any_table(line_pos) or _in_any_table(max(0, line_end - 1))

        if cur_len + len(line) > target_chars and not in_table and current:
            chunk_text = "".join(current)
            chunks.append(chunk_text)
            # Overlap: carry the tail of current chunk into the next
            if overlap_chars > 0 and len(chunk_text) > overlap_chars:
                overlap_text = chunk_text[-overlap_chars:]
                current = [overlap_text]
                cur_len = len(overlap_text)
            else:
                current = []
                cur_len = 0

        current.append(line)
        cur_len  += len(line)
        line_pos  = line_end

    if current:
        chunks.append("".join(current))

    # Post-process: label oversized table chunks
    result = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        pipe_density = chunk.count("|") / max(len(chunk), 1)
        if len(chunk) > target_chars * 1.5 and pipe_density > _TABLE_DENSITY_THRESHOLD:
            chunk = f"[TABLE OVERSIZED CHUNK — {len(chunk):,} chars]\n\n" + chunk
        result.append(chunk)

    logger.info(
        "[table_chunker] split_text_preserving_tables: %d chunks from %d chars",
        len(result), len(text),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  CONTEXT INJECTION FOR EXISTING CHUNK LISTS
# ─────────────────────────────────────────────────────────────────────────────

def inject_table_context(
    chunks:    list[Any],
    full_text: str,
) -> list[Any]:
    """
    Post-processor: prepend table header rows to chunks that start mid-table.

    Works with any chunk objects that have a `.text` or `.context` attribute
    (compatible with HierarchicalChunker, ScientificPaperChunker, and the
    naive _FC fallback class).

    For each chunk, if it contains table rows but NO separator row
    (indicating the chunk starts mid-table after the header was cut),
    the nearest preceding table header is prepended.

    Parameters
    ----------
    chunks    : list of chunk objects from any chunker
    full_text : full document text (for locating headers)

    Returns the same list with context prepended where needed.
    """
    tables = detect_tables(full_text)
    if not tables:
        return chunks   # nothing to inject

    patched_count = 0

    for chunk in chunks:
        text = getattr(chunk, "context", None) or getattr(chunk, "text", "") or ""
        if not text:
            continue

        # Detect if this chunk has table rows but no separator row
        has_table_rows = bool(_TABLE_ROW_RE.search(text))
        has_separator  = bool(_TABLE_SEP_RE.search(text))

        if has_table_rows and not has_separator:
            # Find the matching table (by content overlap)
            for tbl in tables:
                # Check if any row of this chunk matches the table body
                first_row = text.strip().splitlines()[0] if text.strip() else ""
                if first_row and first_row in tbl.text:
                    prefix = (
                        f"[TABLE HEADER — '{tbl.section}']\n"
                        f"{tbl.header}\n"
                        f"|{'---|' * (tbl.header.count('|') - 1)}\n\n"
                    )
                    if hasattr(chunk, "context") and chunk.context:
                        chunk.context = prefix + chunk.context
                    elif hasattr(chunk, "text"):
                        chunk.text = prefix + chunk.text
                    patched_count += 1
                    logger.debug(
                        "[table_chunker] Injected header for table '%s' into chunk",
                        tbl.section,
                    )
                    break

    if patched_count:
        logger.info(
            "[table_chunker] Injected table headers into %d mid-table chunks",
            patched_count,
        )
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  CHUNK METADATA ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def enrich_chunk_with_table_metadata(chunk: Any) -> Any:
    """
    Add table-related metadata attributes to a chunk object.

    Sets on chunk (in-place):
      .is_table_dominant  (bool)   — True if >7% of chars are pipes
      .table_row_count    (int)    — number of detected table rows
      .table_header       (str)    — first detected table header row
      .has_species_signal (bool)   — True if a binomial pattern found

    Compatible with any chunk object (uses setattr).
    """
    text = getattr(chunk, "context", None) or getattr(chunk, "text", "") or ""
    if not text:
        return chunk

    pipe_density = text.count("|") / max(len(text), 1)
    table_rows   = _TABLE_ROW_RE.findall(text)
    header       = table_rows[0].strip() if table_rows else ""

    # Binomial pattern: Capitalised genus + lowercase epithet
    has_binomial = bool(re.search(r"\b[A-Z][a-z]{2,}\s+[a-z]{3,}\b", text))

    setattr(chunk, "is_table_dominant", pipe_density > _TABLE_DENSITY_THRESHOLD)
    setattr(chunk, "table_row_count",   len(table_rows))
    setattr(chunk, "table_header",      header)
    # Don't override has_species if already set by hierarchical chunker
    if not getattr(chunk, "has_species", None):
        setattr(chunk, "has_species", has_binomial)

    return chunk


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE TABLE-AWARE CHUNKER
#  Drop-in replacement for the naive Priority 4 fixed-size slicer.
# ─────────────────────────────────────────────────────────────────────────────

class TableAwareChunker:
    """
    Standalone chunker that preserves table integrity.

    Produces chunk objects compatible with biotrace_v5.py's extraction loop
    (has .text, .section, .chunk_id, .has_species, .candidate_localities,
    .is_table_dominant, .table_header, .table_row_count attributes).

    Usage in biotrace_v5.py (Priority 3.5 — insert between DocumentChunker
    and naive fallback, or use as the sole chunker for checklist papers):

        from biotrace_table_chunker_patch import TableAwareChunker
        chunker = TableAwareChunker(chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        flat_chunks = chunker.chunk(markdown_text, source_label=doc_title)
        log_cb(f"[TableChunk] {len(flat_chunks)} table-safe chunks")
        use_flat = False
    """

    def __init__(
        self,
        chunk_chars:   int = 6000,
        overlap_chars: int = 400,
    ):
        self.chunk_chars   = chunk_chars
        self.overlap_chars = overlap_chars

    def chunk(self, text: str, source_label: str = "") -> list[Any]:
        """
        Chunk the text, preserving table boundaries.
        Returns a list of simple namespace objects with chunk metadata.
        """
        from types import SimpleNamespace

        raw_chunks = split_text_preserving_tables(
            text, self.chunk_chars, self.overlap_chars
        )

        result = []
        for i, chunk_text in enumerate(raw_chunks):
            # Detect section heading within this chunk
            headings = re.findall(r"^#{1,4}\s+(.+)$", chunk_text, re.MULTILINE)
            section  = headings[-1].strip() if headings else f"Chunk {i+1}"

            pipe_density = chunk_text.count("|") / max(len(chunk_text), 1)
            table_rows   = _TABLE_ROW_RE.findall(chunk_text)
            header       = table_rows[0].strip() if table_rows else ""
            has_binomial = bool(re.search(r"\b[A-Z][a-z]{2,}\s+[a-z]{3,}\b", chunk_text))

            c = SimpleNamespace(
                chunk_id             = i,
                text                 = chunk_text,
                section              = section,
                has_species          = has_binomial,
                candidate_localities = [],
                candidate_species    = [],
                is_table_dominant    = pipe_density > _TABLE_DENSITY_THRESHOLD,
                table_row_count      = len(table_rows),
                table_header         = header,
            )
            result.append(c)

        logger.info(
            "[TableAwareChunker] %d chunks | %d table-dominant | source='%s'",
            len(result),
            sum(1 for c in result if c.is_table_dominant),
            source_label[:60],
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  TABLE-CONTEXT PROMPT BUILDER
#  Call from process_chunk() to augment prompt when chunk is table-dominant.
# ─────────────────────────────────────────────────────────────────────────────

def build_table_context_prefix(chunk: Any, citation_str: str = "") -> str:
    """
    Build a prompt prefix for table-dominant chunks.

    Add to the start of the text passed to the LLM in process_chunk():

        from biotrace_table_chunker_patch import build_table_context_prefix
        if getattr(chunk, 'is_table_dominant', False):
            text = build_table_context_prefix(chunk, cite_str) + text

    This tells the LLM:
      - How to interpret checkmarks in checklist tables
      - What the column headers mean
      - That EVERY row must produce a JSON record
    """
    header    = getattr(chunk, "table_header", "")
    row_count = getattr(chunk, "table_row_count", 0)

    lines = [
        f"[TABLE CHUNK — {row_count} rows detected]",
        f"Source: {citation_str}" if citation_str else "",
        "",
        "CRITICAL INSTRUCTIONS FOR TABLE EXTRACTION:",
        "  • Every table row with a species name must produce EXACTLY ONE JSON record.",
        "  • A '√' (checkmark) in the 'Present Study' column means occurrenceType = 'Primary'.",
        "  • Rows WITHOUT '√' in 'Present Study' are Secondary records (cited from literature).",
        "  • '√' in 'New record to India' → flag as new_record_india = true.",
        "  • '√' in 'New record to Gujarat' → flag as new_record_gujarat = true.",
        "  • Do NOT skip any row, including rows at the top or bottom of the table.",
        "  • Use the broadest study area as verbatimLocality when no finer site is given.",
        "",
    ]
    if header:
        lines.insert(3, f"Column headers: {header}")

    return "\n".join(l for l in lines if l is not None) + "\n\n"


# ─────────────────────────────────────────────────────────────────────────────
#  SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAMPLE = """\
# Results

The species are listed in Table 1.

| Sr. No | Species | Present Study | New record to India | New record to Gujarat |
|--------|---------|:---:|:---:|:---:|
| 1. | *Hydatina zonata* | √ | - | - |
| 2. | *Bulla ampulla* | √ | - | - |
| 3. | *Haminoea ovalis* | √ | √ | √ |
| 4. | *Aplysia dactylomela* | √ | - | - |
| 5. | *Berthellina citrina* | √ | - | - |
| 6. | *Berthellina cf. citrina* (spotted form) | √ | - | - |
| 7. | *Berthella stellata* | √ | √ | √ |
| 8. | *Elysia tomentosa* | √ | - | √ |
| 9. | *Elysia thompsoni* | √ | √ | √ |
| 10. | *Elysia obtusa* | √ | √ | √ |

The total of 33 species were recorded from the Gulf of Kutch.
"""

    print("=== Table Detection ===")
    tables = detect_tables(SAMPLE)
    for t in tables:
        print(f"  Table: section='{t.section}', rows={t.row_count}")
        print(f"  Header: {t.header}")

    print("\n=== Table-Aware Chunking ===")
    chunker = TableAwareChunker(chunk_chars=300, overlap_chars=50)
    chunks  = chunker.chunk(SAMPLE, source_label="Test")
    for c in chunks:
        print(f"  Chunk {c.chunk_id}: section='{c.section}' | "
              f"table_dominant={c.is_table_dominant} | "
              f"rows={c.table_row_count} | "
              f"has_species={c.has_species}")

    print("\n=== Table Context Prefix ===")
    for c in chunks:
        if c.is_table_dominant:
            prefix = build_table_context_prefix(c, "Gulf of Kutch Checklist")
            print(prefix)
            break
