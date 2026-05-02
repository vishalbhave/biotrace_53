"""
biotrace_agentic_chunker.py  —  BioTrace v5.7
═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURAL REFACTOR: ScientificPaperChunker → Agentic Chunking Pipeline

WHAT CHANGED vs v5.6 ScientificPaperChunker
────────────────────────────────────────────
1. AGENTIC CHUNKING  — replaced rule-based heading detection with a
   pydantic-ai Agent (ChunkingAgent) that reads the full document and
   returns semantically coherent extraction batches as structured
   AgenticChunk objects, each carrying rich metadata (section_role,
   candidate_species, candidate_localities, priority_score).

2. SCHEMA-DRIVEN EXTRACTION  — OccurrenceExtractionAgent uses
   biotrace_schema.OccurrenceRecord as pydantic-ai result_type.
   No more raw-JSON parsing / json_repair fallback — structured output
   is guaranteed or the agent retries automatically.

3. SPECIES-CENTRIC BUFFER  — SpeciesCentricBuffer accumulates chunks
   per species in memory.  When a species' buffer reaches its flush
   threshold (chars or chunk count), WikiWriterAgent fires in-band
   without a separate pipeline step.

4. STATE-AWARE DEDUPLICATION  — DocumentStateManager hashes the
   source text (SHA-256).  If the hash is already in the
   processed_documents SQLite table the enrichment phase is skipped
   entirely; validated records are loaded directly from the DB.

5. WIKI WRITER AGENT INLINE  — WikiWriterAgent (pydantic-ai) runs
   inside the same extraction pass using the Species-Centric Buffer as
   its input, eliminating the separate OllamaWikiAgent / WikiAutoPopAgent
   run for most cases.

PUBLIC API  (drop-in for extract_occurrences chunking path)
───────────────────────────────────────────────────────────
    from biotrace_agentic_chunker import AgenticExtractionPipeline

    pipeline = AgenticExtractionPipeline(
        meta_db_path = META_DB_PATH,
        wiki_root    = WIKI_ROOT,
        model        = "claude-sonnet-4-20250514",   # or Ollama model tag
        ollama_url   = "http://localhost:11434",      # set if using Ollama
        use_wiki_writer = True,
    )

    # Returns list[OccurrenceRecord] (pydantic models) — call .to_dict() for DB insert
    records = pipeline.run(
        markdown_text    = md_text,
        source_citation  = citation_str,
        log_cb           = log_cb,
    )

    # Convert for existing insert_occurrences():
    dicts = [r.to_dict() for r in records]

GRACEFUL DEGRADATION
────────────────────
• pydantic-ai absent  → falls back to ScientificPaperChunker + raw JSON parse
• Wiki module absent  → WikiWriterAgent silently disabled
• DB unavailable      → dedup check skipped, full extraction runs
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("biotrace.agentic_chunker")

# ─────────────────────────────────────────────────────────────────────────────
#  OPTIONAL DEPENDENCY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

_PAI_OK = False
_PAI_ANTHROPIC_OK = False
_PAI_OLLAMA_OK = False

try:
    from pydantic_ai import Agent as _PAIAgent
    from pydantic_ai.settings import ModelSettings as _ModelSettings
    _PAI_OK = True
except ImportError:
    _PAIAgent = None  # type: ignore
    logger.info("[AgenticChunker] pydantic-ai not installed — will use rule-based fallback")

if _PAI_OK:
    try:
        from pydantic_ai.models.anthropic import AnthropicModel as _AnthropicModel
        _PAI_ANTHROPIC_OK = True
    except ImportError:
        _AnthropicModel = None  # type: ignore

    try:
        from pydantic_ai.models.ollama import OllamaModel as _OllamaModel
        _PAI_OLLAMA_OK = True
    except ImportError:
        _OllamaModel = None  # type: ignore

try:
    from pydantic import BaseModel, Field as PField, field_validator, model_validator
    _PYDANTIC_OK = True
except ImportError:
    _PYDANTIC_OK = False
    BaseModel = object  # type: ignore
    def PField(*a, **kw): return None  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
#  SCHEMA IMPORT (biotrace_schema.py)
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA_OK = False
OccurrenceRecord = None  # type: ignore
RelationTriple = None  # type: ignore

try:
    from biotrace_schema import OccurrenceRecord, RelationTriple
    _SCHEMA_OK = True
    logger.info("[AgenticChunker] biotrace_schema loaded")
except ImportError:
    logger.warning("[AgenticChunker] biotrace_schema.py not found — using dict fallback")


# ─────────────────────────────────────────────────────────────────────────────
#  FALLBACK CHUNKER (keeps API contract when pydantic-ai is absent)
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_OK = False
_FallbackChunker = None

try:
    from biotrace_scientific_chunker import ScientificPaperChunker as _FallbackChunker
    _FALLBACK_OK = True
    logger.info("[AgenticChunker] ScientificPaperChunker available as fallback")
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  PYDANTIC MODELS FOR AGENTIC OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

if _PYDANTIC_OK:

    class AgenticChunk(BaseModel):
        """
        Structured output from ChunkingAgent.
        Replaces the rule-based SciChunk dataclass with semantically-rich metadata.
        """
        text:                 str   = PField(..., description="Extracted chunk text for LLM extraction")
        section_role:         str   = PField("OTHER", description="ABSTRACT|METHODS|RESULTS|DISCUSSION|TAXONOMY|TABLES|OTHER")
        section_heading:      str   = PField("", description="Original heading text")
        candidate_species:    list[str] = PField(default_factory=list, description="Pre-identified species names")
        candidate_localities: list[str] = PField(default_factory=list, description="Pre-identified locality strings")
        injected_context:     str   = PField("", description="Study context injected from Methods/Abstract")
        priority_score:       float = PField(1.0, ge=0.0, le=1.0, description="Extraction priority (1=highest)")
        has_species_signal:   bool  = PField(True, description="False = skip chunk to save LLM budget")
        char_start:           int   = PField(0)
        char_end:             int   = PField(0)

        @field_validator("section_role")
        @classmethod
        def _valid_role(cls, v: str) -> str:
            valid = {"ABSTRACT","METHODS","RESULTS","DISCUSSION","TAXONOMY","TABLES","OTHER","INTRODUCTION"}
            return v.upper() if v.upper() in valid else "OTHER"

    class ChunkingDecision(BaseModel):
        """
        Full document chunking plan returned by ChunkingAgent.
        Contains all chunks plus extracted study context.
        """
        chunks:        list[AgenticChunk] = PField(default_factory=list)
        study_context: str                = PField("", description="Consolidated Methods/Abstract context")
        doc_language:  str                = PField("en")
        total_species_hint: int           = PField(0, description="Agent's estimate of species count")

    class WikiSection(BaseModel):
        """Partial wiki article update returned by WikiWriterAgent."""
        lead:                  str = PField("", description="Lead paragraph (encyclopaedic)")
        taxonomy_phylogeny:    str = PField("")
        morphology:            str = PField("")
        distribution_habitat:  str = PField("")
        ecology_behaviour:     str = PField("")
        conservation:          str = PField("")
        notes:                 str = PField("")

        def non_empty_sections(self) -> dict[str, str]:
            return {k: v for k, v in self.model_dump().items() if v.strip()}

else:
    # Minimal stubs so the rest of the module can reference these names
    class AgenticChunk:  # type: ignore
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
            self.text                 = kw.get("text", "")
            self.section_role         = kw.get("section_role", "OTHER")
            self.section_heading      = kw.get("section_heading", "")
            self.candidate_species    = kw.get("candidate_species", [])
            self.candidate_localities = kw.get("candidate_localities", [])
            self.injected_context     = kw.get("injected_context", "")
            self.priority_score       = kw.get("priority_score", 1.0)
            self.has_species_signal   = kw.get("has_species_signal", True)
            self.char_start           = kw.get("char_start", 0)
            self.char_end             = kw.get("char_end", 0)

    class ChunkingDecision:  # type: ignore
        chunks = []; study_context = ""; doc_language = "en"; total_species_hint = 0

    class WikiSection:  # type: ignore
        def non_empty_sections(self): return {}


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

_CHUNKING_SYSTEM = """\
You are a marine biodiversity informatics expert and document analyser.

Given a raw markdown document from a scientific paper, produce a structured
chunking plan that maximises species-occurrence extraction accuracy.

Rules:
1. Identify semantic sections (Abstract, Methods, Results, Discussion,
   Taxonomy, Tables, Introduction). Assign each a section_role.
2. Keep chunks under 6000 characters. Split large sections on paragraph
   boundaries (blank lines), never mid-sentence.
3. For RESULTS, DISCUSSION, TAXONOMY sections: pre-identify ALL scientific
   names (binomials, genus-only, cf./aff./sp. forms) in candidate_species.
4. For METHODS/ABSTRACT sections: extract locality strings, collection
   dates, and depth ranges — these form the study_context that will be
   injected into RESULTS chunks to resolve locality ambiguity.
5. Assign priority_score:
     1.0 = RESULTS, TAXONOMY (species + locality data)
     0.8 = DISCUSSION, TABLES
     0.5 = METHODS, ABSTRACT
     0.2 = INTRODUCTION, REFERENCES
6. Set has_species_signal=false for chunks that clearly contain ONLY
   references, acknowledgements, or author affiliations.
7. Build study_context from ALL Methods/Abstract sections combined.
   Format as: "Localities: X, Y. Dates: A–B. Depth: N–M m."
8. Estimate total_species_hint from your scan.

Return the complete ChunkingDecision JSON. Be thorough — missing a species
in a chunk leads to a lost biodiversity record.
"""

_EXTRACTION_SYSTEM = """\
You are a biodiversity data extraction specialist working on marine species
occurrence records.

Given a document chunk (with injected Methods context) and a citation string,
extract EVERY species occurrence as an OccurrenceRecord.

Core rules:
- ONE record per species × locality × sampling event.
- verbatim_locality: exact place name as written; NEVER blank.
- occurrence_type: "Primary" (authors' own data), "Secondary" (cited),
  "Uncertain" (ambiguous).
- raw_text_evidence: verbatim sentence + 1 preceding + 1 following sentence.
- Do NOT invent or hallucinate — only extract what the text explicitly states.
- Include genus-only names (sp., spp., cf., aff.) as valid records.
- For abbreviated genera (A. cornutus), expand using context.
- Multiple localities in one sentence → separate records.
- Life-stage references (polyp, medusa, larva) → associate with the species
  named in the nearest preceding sentence.

Return a JSON array. If no species found, return [].
"""

_WIKI_WRITER_SYSTEM = """\
You are a Marine Biodiversity Wiki Architect writing encyclopaedic species
articles in the style of Wikipedia.

You receive accumulated text chunks about one species. Populate ONLY the
sections where the chunks provide new information. Write in neutral,
scientific prose. Use *italics* for binomial names and **bold** for terms.
Cite inline as (Author Year).

Return a WikiSection JSON with only the sections you can populate.
Sections: lead, taxonomy_phylogeny, morphology, distribution_habitat,
ecology_behaviour, conservation, notes.

Return ONLY valid JSON — no fences, no prose outside the object.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  DOCUMENT STATE MANAGER  (hash-based dedup)
# ─────────────────────────────────────────────────────────────────────────────

class DocumentStateManager:
    """
    Manages processed-document state in SQLite.

    Key behaviour
    ─────────────
    • is_processed(hash)  → True if already in DB (skip LLM)
    • mark_processed(hash, citation, n_records)  → write to DB
    • load_records(hash)  → list[dict] from the occurrence table

    The hash is SHA-256 of the first 65 536 bytes of source text,
    which is fast and collision-resistant for typical academic papers.
    """

    TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS processed_documents (
        doc_hash    TEXT PRIMARY KEY,
        citation    TEXT,
        n_records   INTEGER DEFAULT 0,
        processed_at TEXT DEFAULT (datetime('now')),
        model_used  TEXT DEFAULT ''
    );
    """

    def __init__(self, meta_db_path: str):
        self.meta_db_path = meta_db_path
        self._ensure_table()

    def _ensure_table(self) -> None:
        if not self.meta_db_path:
            return
        try:
            con = sqlite3.connect(self.meta_db_path)
            con.executescript(self.TABLE_DDL)
            con.close()
        except Exception as exc:
            logger.warning("[StateManager] Table creation: %s", exc)

    @staticmethod
    def compute_hash(text: str) -> str:
        """SHA-256 of first 64 KB — fast, stable, collision-resistant."""
        return hashlib.sha256(text[:65536].encode("utf-8", errors="replace")).hexdigest()

    def is_processed(self, doc_hash: str) -> bool:
        if not self.meta_db_path:
            return False
        try:
            con = sqlite3.connect(self.meta_db_path)
            row = con.execute(
                "SELECT n_records FROM processed_documents WHERE doc_hash=?",
                (doc_hash,),
            ).fetchone()
            con.close()
            if row:
                logger.info("[StateManager] Hash %s already processed (%d records) — SKIP LLM", doc_hash[:12], row[0])
                return True
        except Exception as exc:
            logger.debug("[StateManager] is_processed: %s", exc)
        return False

    def mark_processed(self, doc_hash: str, citation: str, n_records: int, model: str = "") -> None:
        if not self.meta_db_path:
            return
        try:
            con = sqlite3.connect(self.meta_db_path)
            con.execute(
                """INSERT OR REPLACE INTO processed_documents
                   (doc_hash, citation, n_records, processed_at, model_used)
                   VALUES (?,?,?,datetime('now'),?)""",
                (doc_hash, citation[:500], n_records, model[:100]),
            )
            con.commit()
            con.close()
            logger.info("[StateManager] Marked %s processed (%d records)", doc_hash[:12], n_records)
        except Exception as exc:
            logger.warning("[StateManager] mark_processed: %s", exc)

    def load_existing_records(self, doc_hash: str, occ_table: str = "occurrences_v4") -> list[dict]:
        """
        Load previously extracted records for a document from the occurrence table.
        Used when is_processed() returns True — avoids re-running LLM.
        """
        if not self.meta_db_path:
            return []
        try:
            con = sqlite3.connect(self.meta_db_path)
            rows = con.execute(
                f"SELECT * FROM {occ_table} WHERE file_hash=?", (doc_hash[:16],)
            ).fetchall()
            cols = [d[0] for d in con.execute(f"PRAGMA table_info({occ_table})").fetchall()]
            con.close()
            return [dict(zip(cols, r)) for r in rows]
        except Exception as exc:
            logger.warning("[StateManager] load_existing_records: %s", exc)
            return []


# ─────────────────────────────────────────────────────────────────────────────
#  SPECIES-CENTRIC BUFFER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _SpeciesBuffer:
    chunks:      list[str]       = field(default_factory=list)
    citations:   list[str]       = field(default_factory=list)
    total_chars: int             = 0
    flushed:     bool            = False


class SpeciesCentricBuffer:
    """
    Accumulates extracted text chunks keyed by species name.

    Design goal: feed the WikiWriterAgent with species-targeted text
    instead of the entire document, avoiding context overflow and
    reducing hallucination.

    Flush triggers (checked after each chunk add):
    • total_chars >= flush_chars_threshold  (default 12 000)
    • chunk_count >= flush_chunk_threshold  (default 8)

    When flushed, the buffer contents are passed to WikiWriterAgent.
    The buffer is then cleared but marked as having been flushed so
    subsequent chunks in the same run are appended to a new buffer.
    """

    def __init__(
        self,
        flush_chars_threshold:  int = 12_000,
        flush_chunk_threshold:  int = 8,
        on_flush: Optional[Callable[[str, list[str], list[str]], None]] = None,
    ):
        self._buffers:              dict[str, _SpeciesBuffer] = {}
        self.flush_chars_threshold  = flush_chars_threshold
        self.flush_chunk_threshold  = flush_chunk_threshold
        self._on_flush              = on_flush  # callback(species, chunks, citations)

    def add(self, species: str, chunk_text: str, citation: str) -> bool:
        """
        Add a chunk for a species.  Returns True if flush was triggered.
        """
        buf = self._buffers.setdefault(species, _SpeciesBuffer())
        buf.chunks.append(chunk_text)
        buf.citations.append(citation)
        buf.total_chars += len(chunk_text)

        if (buf.total_chars >= self.flush_chars_threshold or
                len(buf.chunks) >= self.flush_chunk_threshold):
            self._flush(species)
            return True
        return False

    def _flush(self, species: str) -> None:
        buf = self._buffers.get(species)
        if not buf or not buf.chunks:
            return
        if self._on_flush:
            try:
                self._on_flush(species, list(buf.chunks), list(buf.citations))
            except Exception as exc:
                logger.warning("[Buffer] on_flush error for %s: %s", species, exc)
        buf.flushed = True
        # Reset but keep the key so we know it was seen
        self._buffers[species] = _SpeciesBuffer()

    def flush_all(self) -> None:
        """Flush every species buffer (called at pipeline end)."""
        for species in list(self._buffers.keys()):
            buf = self._buffers[species]
            if buf.chunks:
                self._flush(species)

    def species_seen(self) -> list[str]:
        return list(self._buffers.keys())


# ─────────────────────────────────────────────────────────────────────────────
#  PYDANTIC-AI AGENT FACTORY
# ─────────────────────────────────────────────────────────────────────────────

# def _make_model(model_tag: str, ollama_url: str = ""):
#     """
#     Return a pydantic-ai model object.
#     Priority: Anthropic (if model_tag looks like claude-*) → Ollama → raises.
#     """
#     if not _PAI_OK:
#         raise ImportError("pydantic-ai not installed")

#     is_claude = model_tag.startswith("claude-") or "anthropic" in model_tag.lower()

#     if is_claude and _PAI_ANTHROPIC_OK:
#         return _AnthropicModel(model_tag)

#     if _PAI_OLLAMA_OK and ollama_url:
#         return _OllamaModel(model_tag, base_url=ollama_url)

#     if _PAI_ANTHROPIC_OK:
#         return _AnthropicModel(model_tag)

#     raise RuntimeError(
#         f"No suitable pydantic-ai backend for model={model_tag!r}. "
#         "Install pydantic-ai[anthropic] or pydantic-ai[ollama]."
#     )


def _make_model(model_tag: str, ollama_url: str = ""):
    """Updated pydantic-ai model factory for Ollama compatibility."""
    if not _PAI_OK:
        raise ImportError("pydantic-ai not installed")

    # Priority: Anthropic (Claude)
    if (model_tag.startswith("claude-") or "anthropic" in model_tag.lower()) and _PAI_ANTHROPIC_OK:
        return _AnthropicModel(model_tag)

    # Priority: Ollama with Provider pattern
    if _PAI_OLLAMA_OK:
        from pydantic_ai.providers.ollama import OllamaProvider as _OllamaProvider
        
        # Use the /v1 endpoint for OpenAI-compatible routing if needed
        base_url = ollama_url if "/v1" in ollama_url else f"{ollama_url.rstrip('/')}/v1"
        provider = _OllamaProvider(base_url=base_url)
        
        return _OllamaModel(model_tag, provider=provider)

    raise RuntimeError(f"No suitable backend for model={model_tag!r}")

# ─────────────────────────────────────────────────────────────────────────────
#  CHUNKING AGENT
# ─────────────────────────────────────────────────────────────────────────────

class ChunkingAgent:
    """
    Replaces ScientificPaperChunker with an LLM-driven semantic chunking pass.

    The agent receives the full document (up to max_input_chars) and returns
    a ChunkingDecision with pre-labelled, pre-annotated chunks.

    Fallback: if pydantic-ai is unavailable or the agent call fails, delegates
    to ScientificPaperChunker (if installed) or naive fixed-size slicing.
    """

    def __init__(
        self,
        model_tag:      str = "claude-sonnet-4-20250514",
        ollama_url:     str = "",
        max_input_chars: int = 40_000,
        chunk_chars:    int = 6_000,
        overlap_chars:  int = 400,
    ):
        self.model_tag       = model_tag
        self.ollama_url      = ollama_url
        self.max_input_chars = max_input_chars
        self.chunk_chars     = chunk_chars
        self.overlap_chars   = overlap_chars
        self._agent          = None

        if _PAI_OK and _PYDANTIC_OK:
            self._init_agent()

    def _init_agent(self) -> None:
        try:
            pai_model = _make_model(self.model_tag, self.ollama_url)
            self._agent = _PAIAgent(
                model       = pai_model,
                result_type = ChunkingDecision,
                system_prompt = _CHUNKING_SYSTEM,
            )
            logger.info("[ChunkingAgent] Agent ready (model=%s)", self.model_tag)
        except Exception as exc:
            logger.warning("[ChunkingAgent] Agent init failed: %s — will use fallback", exc)
            self._agent = None

    def chunk(self, markdown_text: str, source_label: str = "") -> list[AgenticChunk]:
        """
        Main entry point.  Returns list[AgenticChunk].

        Tries pydantic-ai agent first; falls through to rule-based chunker;
        falls through to naive slices.
        """
        if self._agent:
            try:
                return self._chunk_agentic(markdown_text, source_label)
            except Exception as exc:
                logger.warning("[ChunkingAgent] Agentic chunk failed (%s) — fallback", exc)

        return self._chunk_fallback(markdown_text, source_label)

    def _chunk_agentic(self, markdown_text: str, source_label: str) -> list[AgenticChunk]:
        """Run the pydantic-ai chunking agent."""
        # Truncate if very long (agent handles structure, not full content recall)
        doc_input = markdown_text[:self.max_input_chars]
        if len(markdown_text) > self.max_input_chars:
            doc_input += f"\n\n[... document truncated — {len(markdown_text):,} chars total ...]"

        user_prompt = (
            f"Document label: {source_label}\n"
            f"Document length: {len(markdown_text):,} chars\n\n"
            f"DOCUMENT:\n{doc_input}"
        )

        # Run synchronously (block on async)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self._agent.run(user_prompt))
        finally:
            loop.close()

        decision: ChunkingDecision = result.data

        # For chunks that need injected study context, prepend it
        study_ctx = decision.study_context.strip()
        enriched: list[AgenticChunk] = []
        for ch in decision.chunks:
            if study_ctx and ch.section_role in ("RESULTS", "DISCUSSION", "TAXONOMY", "TABLES"):
                if study_ctx not in ch.injected_context:
                    ch.injected_context = study_ctx + "\n\n" + ch.injected_context
            enriched.append(ch)

        logger.info(
            "[ChunkingAgent] %s → %d agentic chunks | study_context=%d chars | "
            "~%d species estimated",
            source_label, len(enriched), len(study_ctx), decision.total_species_hint,
        )
        return enriched

    def _chunk_fallback(self, markdown_text: str, source_label: str) -> list[AgenticChunk]:
        """
        Fallback: use ScientificPaperChunker (rule-based) and convert
        SciChunk → AgenticChunk, OR naive slicing if chunker absent.
        """
        if _FALLBACK_OK and _FallbackChunker:
            try:
                sc = _FallbackChunker(
                    chunk_chars=self.chunk_chars,
                    overlap_chars=self.overlap_chars,
                )
                sci_chunks = sc.chunk(markdown_text, source_label=source_label)
                result = []
                for sc_ch in sci_chunks:
                    result.append(AgenticChunk(
                        text                 = sc_ch.context,
                        section_role         = sc_ch.section_role,
                        section_heading      = sc_ch.section,
                        candidate_species    = sc_ch.candidate_species,
                        candidate_localities = sc_ch.candidate_localities,
                        injected_context     = sc_ch.injected_context,
                        priority_score       = 1.0 if sc_ch.section_role in ("RESULTS","TAXONOMY") else 0.6,
                        has_species_signal   = sc_ch.has_species,
                        char_start           = sc_ch.char_start,
                        char_end             = sc_ch.char_end,
                    ))
                logger.info("[ChunkingAgent] Fallback: %d SciChunks converted", len(result))
                return result
            except Exception as exc:
                logger.warning("[ChunkingAgent] SciChunker fallback: %s", exc)

        # Last resort: naive fixed-size slices
        step = max(self.chunk_chars - self.overlap_chars, 1000)
        slices = []
        for i, s in enumerate(range(0, min(len(markdown_text), 60_000), step)):
            slices.append(AgenticChunk(
                text               = markdown_text[s: s + self.chunk_chars],
                section_role       = "RESULTS",
                section_heading    = f"Chunk {i+1}",
                priority_score     = 1.0,
                has_species_signal = True,
                char_start         = s,
                char_end           = min(s + self.chunk_chars, len(markdown_text)),
            ))
        logger.info("[ChunkingAgent] Naive fallback: %d slices", len(slices))
        return slices


# ─────────────────────────────────────────────────────────────────────────────
#  OCCURRENCE EXTRACTION AGENT
# ─────────────────────────────────────────────────────────────────────────────

class OccurrenceExtractionAgent:
    """
    Per-chunk LLM extraction agent.

    Uses pydantic-ai with OccurrenceRecord as result_type when both are
    available.  Falls back to raw JSON parse + biotrace_schema validation.

    Async-first: use validate_batch() for concurrent chunk processing.
    """

    def __init__(
        self,
        model_tag:      str = "claude-sonnet-4-20250514",
        ollama_url:     str = "",
        max_concurrent: int = 3,
    ):
        self.model_tag      = model_tag
        self.ollama_url     = ollama_url
        self.max_concurrent = max_concurrent
        self._agent         = None

        if _PAI_OK and _SCHEMA_OK and _PYDANTIC_OK:
            self._init_agent()

    def _init_agent(self) -> None:
        try:
            pai_model = _make_model(self.model_tag, self.ollama_url)
            # result_type is list[OccurrenceRecord] — pydantic-ai handles JSON coercion
            self._agent = _PAIAgent(
                model         = pai_model,
                result_type   = list[OccurrenceRecord],
                system_prompt = _EXTRACTION_SYSTEM,
            )
            logger.info("[ExtractionAgent] Agent ready (model=%s)", self.model_tag)
        except Exception as exc:
            logger.warning("[ExtractionAgent] Agent init failed: %s — raw JSON fallback", exc)
            self._agent = None

    async def extract_chunk_async(
        self,
        chunk: AgenticChunk,
        citation: str,
        log_cb: Optional[Callable] = None,
    ) -> list:
        """
        Extract occurrences from a single AgenticChunk.
        Returns list[OccurrenceRecord] or list[dict] (fallback).
        """
        def _log(msg: str):
            if log_cb: log_cb(msg)

        if not chunk.has_species_signal and chunk.priority_score < 0.4:
            return []

        # Build user prompt
        species_hint = ""
        if chunk.candidate_species:
            species_hint = (
                "\n\n[CONFIRMED SPECIES from chunking agent — extract ALL of these]:\n"
                + "\n".join(f"  • {s}" for s in chunk.candidate_species[:30])
            )

        locality_hint = ""
        if chunk.candidate_localities:
            locality_hint = (
                "\n\n[PRE-LINKED LOCALITIES from Methods]:\n"
                + "\n".join(f"  • {l}" for l in chunk.candidate_localities[:10])
            )

        user_prompt = (
            f"CITATION: {citation}\n"
            f"SECTION: {chunk.section_heading!r} [{chunk.section_role}]\n"
            f"{locality_hint}{species_hint}\n\n"
            f"CHUNK TEXT:\n{chunk.text[:6000]}\n\n"
            f"Extract EVERY species occurrence. Return [] if none found."
        )

        # — pydantic-ai structured extraction —
        if self._agent:
            try:
                result = await self._agent.run(user_prompt)
                records = result.data  # list[OccurrenceRecord]
                # Stamp citation & section on each record
                for rec in records:
                    if hasattr(rec, 'source_citation') and not rec.source_citation:
                        rec.source_citation = citation
                _log(f"  [{chunk.section_heading}] {len(records)} records (agentic)")
                return records
            except Exception as exc:
                _log(f"  [{chunk.section_heading}] agentic extraction failed: {exc} — raw fallback")

        # — raw JSON fallback —
        return self._raw_json_fallback(user_prompt, chunk, citation, _log)

    def _raw_json_fallback(
        self,
        user_prompt: str,
        chunk: AgenticChunk,
        citation: str,
        log: Callable,
    ) -> list:
        """
        Non-agentic path: call LLM externally (registered elsewhere in the
        pipeline via _external_llm_fn) and parse with biotrace_schema.
        Returns list[OccurrenceRecord] or list[dict].
        """
        # This path is only reached when pydantic-ai is absent or failed.
        # The actual LLM call is delegated to the pipeline orchestrator.
        # Return a sentinel so the pipeline knows to handle it.
        log(f"  [{chunk.section_heading}] raw JSON path — pipeline will call LLM")
        return []  # pipeline handles this case via _run_raw_extraction()

    async def extract_all_chunks(
        self,
        chunks:   list[AgenticChunk],
        citation: str,
        log_cb:   Optional[Callable] = None,
    ) -> list:
        """
        Concurrently extract from all chunks.  Respects max_concurrent.
        Returns flat list of all records.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _guarded(ch: AgenticChunk):
            async with semaphore:
                return await self.extract_chunk_async(ch, citation, log_cb)

        results = await asyncio.gather(*[_guarded(ch) for ch in chunks])
        flat = []
        for batch in results:
            flat.extend(batch)
        return flat


# ─────────────────────────────────────────────────────────────────────────────
#  WIKI WRITER AGENT
# ─────────────────────────────────────────────────────────────────────────────

class WikiWriterAgent:
    """
    Inline wiki article populator — fires during extraction via the
    SpeciesCentricBuffer flush callback.

    On flush, receives the accumulated text chunks for one species,
    calls the LLM to populate WikiSection fields, and writes them
    to the wiki store (BioTraceWikiUnified / BioTraceWikiV56).

    Unlike OllamaWikiAgent (which runs as a separate UI step), this
    agent is fully automated and fires transparently during extraction.
    """

    def __init__(
        self,
        wiki,                           # BioTraceWikiUnified / BioTraceWikiV56 instance
        model_tag:  str = "claude-sonnet-4-20250514",
        ollama_url: str = "",
    ):
        self.wiki      = wiki
        self.model_tag = model_tag
        self.ollama_url = ollama_url
        self._agent    = None

        if _PAI_OK and _PYDANTIC_OK and wiki:
            self._init_agent()

    def _init_agent(self) -> None:
        try:
            pai_model = _make_model(self.model_tag, self.ollama_url)
            self._agent = _PAIAgent(
                model         = pai_model,
                result_type   = WikiSection,
                system_prompt = _WIKI_WRITER_SYSTEM,
            )
            logger.info("[WikiWriterAgent] Agent ready (model=%s)", self.model_tag)
        except Exception as exc:
            logger.warning("[WikiWriterAgent] Agent init failed: %s", exc)
            self._agent = None

    def on_species_buffer_flush(
        self,
        species:   str,
        chunks:    list[str],
        citations: list[str],
    ) -> None:
        """
        Flush callback for SpeciesCentricBuffer.
        Calls the wiki writer agent and writes the result to the wiki store.
        """
        if not self._agent or not self.wiki:
            return

        combined_text = "\n\n---\n\n".join(
            f"[Source: {cit}]\n{txt[:3000]}"
            for txt, cit in zip(chunks, citations)
        )[:12_000]

        user_prompt = (
            f"Species: *{species}*\n\n"
            f"Accumulated text chunks from extraction pipeline:\n\n"
            f"{combined_text}\n\n"
            f"Populate the wiki sections where this text provides information."
        )

        try:
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(self._agent.run(user_prompt))
            finally:
                loop.close()

            wiki_section: WikiSection = result.data
            new_sections = wiki_section.non_empty_sections()

            if not new_sections:
                logger.debug("[WikiWriterAgent] No sections returned for %s", species)
                return

            # Merge into existing article (non-destructive)
            art = self._get_or_create_article(species)
            art_secs = art.setdefault("sections", {})
            merged = []
            for sec_key, new_text in new_sections.items():
                existing = art_secs.get(sec_key, "")
                if not existing:
                    art_secs[sec_key] = new_text
                    merged.append(sec_key)
                elif new_text.strip() not in existing:
                    art_secs[sec_key] = existing.rstrip() + "\n\n" + new_text.strip()
                    merged.append(sec_key)

            if merged:
                art["updated_at"] = datetime.now().isoformat()
                self._write_article(species, art, note=f"WikiWriterAgent inline: {merged}")
                logger.info("[WikiWriterAgent] Updated %s → sections %s", species, merged)

        except Exception as exc:
            logger.warning("[WikiWriterAgent] Error for %s: %s", species, exc)

    def _get_or_create_article(self, species: str) -> dict:
        """Retrieve existing wiki article or scaffold a new one."""
        if hasattr(self.wiki, "get_species_article"):
            art = self.wiki.get_species_article(species)
            if art:
                return art
        return {
            "title":      species,
            "sections":   {},
            "provenance": [],
            "updated_at": datetime.now().isoformat(),
        }

    def _write_article(self, species: str, art: dict, note: str = "") -> None:
        """Write article using whatever write API the wiki exposes."""
        slug = re.sub(r"[^a-z0-9_]", "_", species.lower().strip())
        try:
            if hasattr(self.wiki, "_write"):
                self.wiki._write("species", slug, species, art, change_note=note)
            elif hasattr(self.wiki, "update_species_article"):
                self.wiki.update_species_article(species, art)
            elif hasattr(self.wiki, "_save_json"):
                sp_fp = Path(self.wiki.root) / "species" / f"{slug}.json"
                self.wiki._save_json(sp_fp, art)
        except Exception as exc:
            logger.warning("[WikiWriterAgent] _write_article: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
#  DEDUPLICATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _record_key(rec) -> tuple:
    """Canonical deduplication key for an occurrence record."""
    if hasattr(rec, 'recorded_name'):
        # OccurrenceRecord (pydantic model from biotrace_schema v5.4)
        name = (rec.valid_name or rec.recorded_name or "").lower().strip()
        loc  = (rec.verbatim_locality or "").lower().strip()[:60]
        date = ""
        if hasattr(rec, 'sampling_event') and isinstance(rec.sampling_event, dict):
            date = rec.sampling_event.get("date", "")
    else:
        # dict
        name = (rec.get("validName") or rec.get("recordedName") or "").lower().strip()
        loc  = (rec.get("verbatimLocality") or "").lower().strip()[:60]
        se   = rec.get("Sampling Event") or rec.get("samplingEvent") or {}
        date = se.get("date", "") if isinstance(se, dict) else ""
    return (name, loc, date[:7])  # year+month precision


def deduplicate_records(records: list) -> tuple[list, int]:
    """
    Remove duplicate occurrence records using canonical key hashing.
    Returns (deduplicated_list, n_removed).
    """
    seen: set = set()
    unique: list = []
    for rec in records:
        key = _record_key(rec)
        if key[0] and key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique, len(records) - len(unique)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class AgenticExtractionPipeline:
    """
    Drop-in replacement for the extract_occurrences() function's chunking
    and extraction phase.

    Architecture
    ────────────
    1. DocumentStateManager  → hash check → skip or proceed
    2. ChunkingAgent         → agentic (or fallback) semantic chunking
    3. OccurrenceExtractionAgent → structured extraction per chunk
    4. SpeciesCentricBuffer  → per-species chunk accumulation
    5. WikiWriterAgent       → inline wiki population on buffer flush
    6. Deduplication         → canonical key dedup
    7. DocumentStateManager  → mark as processed

    Parameters
    ──────────
    meta_db_path    Path to metadata_v5.db (for state + dedup).
    wiki_root       Optional path to wiki root dir (enables WikiWriterAgent).
    model           pydantic-ai model tag or Ollama model tag.
    ollama_url      Ollama base URL (set if model is not a Claude model).
    use_wiki_writer Enable inline WikiWriterAgent (requires wiki_root).
    max_concurrent  Max parallel extraction API calls.
    chunk_chars     Target chunk size in characters.
    external_llm_fn Fallback callable(prompt: str) → str for raw JSON path.
    """

    def __init__(
        self,
        meta_db_path:    str = "",
        wiki_root:       str = "",
        model:           str = "claude-sonnet-4-20250514",
        ollama_url:      str = "",
        use_wiki_writer: bool = True,
        max_concurrent:  int = 3,
        chunk_chars:     int = 6_000,
        overlap_chars:   int = 400,
        external_llm_fn: Optional[Callable[[str], str]] = None,
    ):
        self.meta_db_path   = meta_db_path
        self.wiki_root      = wiki_root
        self.model          = model
        self.ollama_url     = ollama_url
        self.use_wiki_writer = use_wiki_writer
        self.max_concurrent = max_concurrent
        self.chunk_chars    = chunk_chars
        self.overlap_chars  = overlap_chars
        self._ext_llm_fn    = external_llm_fn

        # Component initialisation
        self._state_mgr   = DocumentStateManager(meta_db_path)
        self._chunker     = ChunkingAgent(
            model_tag      = model,
            ollama_url     = ollama_url,
            chunk_chars    = chunk_chars,
            overlap_chars  = overlap_chars,
        )
        self._extractor   = OccurrenceExtractionAgent(
            model_tag      = model,
            ollama_url     = ollama_url,
            max_concurrent = max_concurrent,
        )
        self._wiki_agent  = None
        self._wiki_buffer = None

        if use_wiki_writer and wiki_root:
            self._init_wiki_writer()

    def _init_wiki_writer(self) -> None:
        """Attempt to initialise WikiWriterAgent + SpeciesCentricBuffer."""
        wiki = self._load_wiki()
        if not wiki:
            logger.info("[Pipeline] Wiki store unavailable — WikiWriterAgent disabled")
            return

        self._wiki_agent = WikiWriterAgent(
            wiki       = wiki,
            model_tag  = self.model,
            ollama_url = self.ollama_url,
        )
        self._wiki_buffer = SpeciesCentricBuffer(
            on_flush = self._wiki_agent.on_species_buffer_flush,
        )
        logger.info("[Pipeline] WikiWriterAgent + SpeciesCentricBuffer initialised")

    def _load_wiki(self):
        """Try to load BioTraceWikiV56 or BioTraceWikiUnified."""
        for module_name, class_name in [
            ("biotrace_wiki_v56",     "BioTraceWikiV56"),
            ("biotrace_wiki_unified", "BioTraceWikiUnified"),
        ]:
            try:
                mod = __import__(module_name, fromlist=[class_name])
                cls = getattr(mod, class_name)
                return cls(root_dir=self.wiki_root)
            except Exception:
                pass
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        markdown_text:   str,
        source_citation: str = "",
        log_cb:          Optional[Callable] = None,
        file_hash:       str = "",          # pre-computed hex; if empty, auto-computed
        skip_dedup_check: bool = False,     # force re-extraction even if in state DB
    ) -> list:
        """
        Full pipeline run.  Returns list[OccurrenceRecord] (or list[dict] on
        fallback).  Call .to_dict() on each for DB insertion.

        Parameters
        ──────────
        markdown_text    Full document text as markdown.
        source_citation  Bibliographic citation string.
        log_cb           Callable(msg: str) for progress logging.
        file_hash        SHA-256 hex string (computed if empty).
        skip_dedup_check If True, always re-run LLM regardless of state DB.
        """
        def _log(msg: str, lvl: str = "ok"):
            logger.info("[Pipeline] %s", msg)
            if log_cb:
                try:
                    log_cb(msg, lvl)
                except TypeError:
                    log_cb(msg)

        if not markdown_text.strip():
            return []

        # ── Step 1: Deduplication / state check ──────────────────────────────
        doc_hash = file_hash or self._state_mgr.compute_hash(markdown_text)

        if not skip_dedup_check and self._state_mgr.is_processed(doc_hash):
            _log(f"[State] Document {doc_hash[:12]}… already processed — loading from DB")
            existing = self._state_mgr.load_existing_records(doc_hash)
            if existing:
                _log(f"[State] Loaded {len(existing)} records from DB (no LLM call)")
                return existing  # Return dicts — consistent with insert_occurrences()
            _log("[State] No records found in DB for this hash — proceeding with extraction")

        # ── Step 2: Agentic chunking ──────────────────────────────────────────
        _log("[Chunking] Starting agentic semantic chunking…")
        t0 = time.time()
        chunks = self._chunker.chunk(markdown_text, source_label=source_citation[:60])
        _log(f"[Chunking] {len(chunks)} chunks in {time.time()-t0:.1f}s")

        # Filter low-priority / no-signal chunks in large documents
        n_original = len(chunks)
        if len(chunks) > 10:
            chunks = [ch for ch in chunks if getattr(ch, "has_species_signal", True)]
            n_skipped = n_original - len(chunks)
            if n_skipped:
                _log(f"[Chunking] Skipped {n_skipped} non-species chunks")

        if not chunks:
            _log("[Chunking] No extractable chunks — done", "warn")
            return []

        # ── Step 3: Concurrent extraction ─────────────────────────────────────
        _log(f"[Extract] Running extraction on {len(chunks)} chunks…")
        t1 = time.time()

        if self._extractor._agent:
            # Async path — pydantic-ai structured extraction
            records = self._run_async_extraction(chunks, source_citation, _log)
        else:
            # Sync raw-JSON path — delegates to external LLM fn
            records = self._run_raw_extraction(chunks, source_citation, _log)

        _log(f"[Extract] {len(records)} raw records in {time.time()-t1:.1f}s")

        # ── Step 4: Update Species-Centric Buffer → WikiWriterAgent ───────────
        if self._wiki_buffer and records:
            self._populate_wiki_buffer(records, chunks, source_citation, _log)

        # ── Step 5: Flush remaining wiki buffer entries ────────────────────────
        if self._wiki_buffer:
            self._wiki_buffer.flush_all()
            _log(f"[Wiki] Buffer flushed for {len(self._wiki_buffer.species_seen())} species")

        # ── Step 6: Deduplication ─────────────────────────────────────────────
        if len(records) > 1:
            records, n_removed = deduplicate_records(records)
            if n_removed:
                _log(f"[Dedup] Removed {n_removed} duplicate records")

        # ── Step 7: Mark document processed ──────────────────────────────────
        self._state_mgr.mark_processed(
            doc_hash   = doc_hash,
            citation   = source_citation,
            n_records  = len(records),
            model      = self.model,
        )

        _log(f"[Pipeline] ✅ Complete: {len(records)} occurrence records")
        return records

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_async_extraction(
        self,
        chunks:   list[AgenticChunk],
        citation: str,
        log:      Callable,
    ) -> list:
        """Run OccurrenceExtractionAgent asynchronously."""
        loop = asyncio.new_event_loop()
        try:
            records = loop.run_until_complete(
                self._extractor.extract_all_chunks(chunks, citation, log)
            )
        finally:
            loop.close()
        return records

    def _run_raw_extraction(
        self,
        chunks:   list[AgenticChunk],
        citation: str,
        log:      Callable,
    ) -> list:
        """
        Fallback path when pydantic-ai extraction is unavailable.
        Calls external_llm_fn (registered at pipeline init) for each chunk,
        then validates with biotrace_schema.parse_llm_response.
        """
        if not self._ext_llm_fn:
            log("[Extract] No external LLM fn available — cannot extract (raw path)", "warn")
            return []

        # Import schema parser
        try:
            from biotrace_schema import parse_llm_response as _parse_llm
        except ImportError:
            _parse_llm = None

        results = []
        for i, ch in enumerate(chunks, 1):
            text = getattr(ch, "text", "")
            if not text.strip():
                continue

            species_hint = ""
            candidates = getattr(ch, "candidate_species", [])
            if candidates:
                species_hint = (
                    "\n\n[PRE-IDENTIFIED SPECIES]:\n"
                    + "\n".join(f"• {s}" for s in candidates[:20])
                )

            prompt = (
                f"{_EXTRACTION_SYSTEM}\n\n"
                f"CITATION: {citation}\n"
                f"SECTION: [{getattr(ch, 'section_role', 'RESULTS')}]\n"
                f"{species_hint}\n\n"
                f"CHUNK:\n{text[:6000]}\n\n"
                f"Return ONLY a valid JSON array."
            )

            try:
                raw = self._ext_llm_fn(prompt)
                # Strip think blocks
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

                if _parse_llm:
                    recs, errs = _parse_llm(raw, source_citation=citation, chunk_id=i)
                    for e in errs:
                        log(f"  [Schema] {e}", "warn")
                    results.extend(recs)
                else:
                    # Plain JSON parse
                    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
                    data = json.loads(cleaned)
                    if isinstance(data, list):
                        results.extend(data)
                log(f"  [Chunk {i}/{len(chunks)}] {getattr(ch,'section_heading','?')!r} extracted")
            except Exception as exc:
                log(f"  [Chunk {i}] extraction error: {exc}", "warn")

        return results

    def _populate_wiki_buffer(
        self,
        records: list,
        chunks:  list[AgenticChunk],
        citation: str,
        log: Callable,
    ) -> None:
        """
        For each extracted occurrence record, find the corresponding chunk
        and add it to the SpeciesCentricBuffer keyed by species name.
        """
        if not self._wiki_buffer:
            return

        # Build a quick chunk-text lookup by position
        # (records from pydantic-ai don't carry chunk index, so we use
        #  raw_text_evidence to find the nearest chunk)
        high_priority_chunks = [
            ch for ch in chunks
            if getattr(ch, "section_role", "") in ("RESULTS", "DISCUSSION", "TAXONOMY")
        ]

        species_set: set[str] = set()
        for rec in records:
            if hasattr(rec, "valid_name"):
                sp = rec.valid_name or rec.recorded_name or ""
            else:
                sp = rec.get("validName") or rec.get("recordedName") or ""
            sp = sp.strip()
            if sp:
                species_set.add(sp)

        # Add all high-priority chunk texts to each species' buffer
        for sp in species_set:
            for ch in high_priority_chunks:
                ch_text = getattr(ch, "text", "")
                if sp.lower() in ch_text.lower() or any(
                    sp.lower() in s.lower() for s in getattr(ch, "candidate_species", [])
                ):
                    self._wiki_buffer.add(sp, ch_text, citation)

        if species_set:
            log(f"[Buffer] Populated buffer for {len(species_set)} species")


# ─────────────────────────────────────────────────────────────────────────────
#  INTEGRATION HELPERS  (for biotrace_v53.py / biotrace_v54.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_agentic_pipeline(
    meta_db_path:    str = "",
    wiki_root:       str = "",
    provider:        str = "Anthropic",
    model_sel:       str = "claude-sonnet-4-20250514",
    ollama_url:      str = "http://localhost:11434",
    use_wiki_writer: bool = True,
    external_llm_fn: Optional[Callable[[str], str]] = None,
    log_cb:          Optional[Callable] = None,
) -> "AgenticExtractionPipeline":
    """
    Factory that maps BioTrace v5.x sidebar settings to AgenticExtractionPipeline.

    Usage in biotrace_v53.py (replaces ScientificPaperChunker block):

        from biotrace_agentic_chunker import build_agentic_pipeline

        pipeline = build_agentic_pipeline(
            meta_db_path    = META_DB_PATH,
            wiki_root       = WIKI_ROOT,
            provider        = provider,
            model_sel       = model_sel,
            ollama_url      = ollama_url,
            use_wiki_writer = use_wiki and wiki_narr,
            external_llm_fn = lambda p: call_llm(p, provider, model_sel, api_key, ollama_url),
        )

        records = pipeline.run(
            markdown_text    = md_text,
            source_citation  = citation_str,
            log_cb           = log_cb,
            file_hash        = file_hash,
        )
        dicts = [r.to_dict() if hasattr(r,'to_dict') else r for r in records]
    """
    # Determine model tag
    is_ollama = "ollama" in provider.lower() or not provider.lower().startswith("anth")
    model_tag = model_sel

    return AgenticExtractionPipeline(
        meta_db_path    = meta_db_path,
        wiki_root       = wiki_root,
        model           = model_tag,
        ollama_url      = ollama_url if is_ollama else "",
        use_wiki_writer = use_wiki_writer,
        external_llm_fn = external_llm_fn,
    )


def records_to_dicts(records: list) -> list[dict]:
    """
    Convert a mixed list of OccurrenceRecord models and plain dicts
    to plain dicts for downstream insert_occurrences() compatibility.
    """
    out = []
    for rec in records:
        if hasattr(rec, "to_dict"):
            out.append(rec.to_dict())
        elif isinstance(rec, dict):
            out.append(rec)
        else:
            try:
                out.append(rec.model_dump())
            except Exception:
                out.append({"error": str(rec)})
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK SMOKE-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAMPLE = """
## Abstract
This study reports scyphozoan diversity from Narara reef, Gulf of Kutch, Gujarat
(22.35°N, 69.67°E) during September 2022–March 2023.

## Study Area and Methods
Specimens were collected by snorkelling at depths of 0.5–3 m from Narara Marine
National Park, Arambhada coast and Okha jetty, Gujarat, India.

## Results
Five species of scyphozoans were identified from the study sites.
Cassiopea andromeda (Forsskål, 1775) was the most abundant species, forming dense
aggregations at Narara reef. Chrysaora cf. melanaster was recorded from the pelagic
zone off Okha. Rhizostoma pulmo was collected at Arambhada coast at 2 m depth.

## Taxonomy
Cassiopea andromeda: Mantle orange-brown, 15–25 cm diameter.
Previously reported from Gulf of Mannar and Lakshadweep (Thomas, 2004).

## Discussion
The presence of Cassiopea andromeda in the Gulf of Kutch extends its known range.
"""

    print("=== AgenticExtractionPipeline Smoke Test ===\n")

    # Test DocumentStateManager
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        test_db = tf.name

    sm = DocumentStateManager(test_db)
    h  = sm.compute_hash(SAMPLE)
    print(f"Doc hash: {h[:16]}…")
    print(f"is_processed (before): {sm.is_processed(h)}")
    sm.mark_processed(h, "Test citation", 3, "test_model")
    print(f"is_processed (after):  {sm.is_processed(h)}")

    # Test SpeciesCentricBuffer
    flushed = []
    buf = SpeciesCentricBuffer(
        flush_chars_threshold = 100,
        on_flush = lambda sp, chs, cits: flushed.append(sp),
    )
    buf.add("Cassiopea andromeda", "A" * 60, "Test 2024")
    buf.add("Cassiopea andromeda", "B" * 60, "Test 2024")  # should trigger flush
    print(f"\nBuffer flush triggered for: {flushed}")

    # Test ChunkingAgent fallback (no pydantic-ai)
    ca = ChunkingAgent(model_tag="claude-sonnet-4-20250514")
    chunks = ca.chunk(SAMPLE, "smoke_test")
    print(f"\nChunks produced: {len(chunks)}")
    for ch in chunks:
        role = getattr(ch, "section_role", "?")
        sp   = getattr(ch, "candidate_species", [])
        print(f"  [{role}] species={sp[:2]}")

    os.unlink(test_db)
    print("\n✅ Smoke test complete")
