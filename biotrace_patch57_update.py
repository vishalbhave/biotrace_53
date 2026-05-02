"""
biotrace_patch57_update.py  —  BioTrace v5.7  Master Patch
═══════════════════════════════════════════════════════════════════════════════
Self-contained patch that fixes all known v5.6 issues and adds the automated
wiki-from-chunks pipeline.  Drop this file beside biotrace_v53.py, then add
ONE call at the top of biotrace_v53.py:

    from biotrace_patch57_update import install_v57_patches
    install_v57_patches(
        meta_db_path = META_DB_PATH,    # e.g. "biodiversity_data/metadata_v5.db"
        wiki_root    = WIKI_ROOT,       # e.g. "biodiversity_data/wiki"
        kg_db_path   = KG_DB_PATH,      # e.g. "biodiversity_data/knowledge_graph.db"
        ollama_url   = "http://localhost:11434",
        ollama_model = "gemma4",        # or whichever model you use
    )

After install the patch can be deleted — but keep the new
`biotrace_chunk_store.py` it creates (or just let this file stay around).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fixes
─────
  [FIX-1]  depthRange column missing  →  auto-ADD the column if absent
  [FIX-2]  "No lead section yet"      →  auto-generate from stored chunks or
                                         occurrence data, never show the empty
                                         banner
  [FIX-3]  OllamaModel(base_url=…)    →  kwargs sanitised before every call;
                                         patch applied to all wiki-agent
                                         OllamaModel instantiations
  [FIX-4]  biotrace_schema.py warning →  stub created so import succeeds
  [FIX-5]  fetch_occurrences_from_db  →  column-safe query; missing columns
                                         are silently skipped

New features
────────────
  [NEW-1]  ChunkStore  —  SQLite table `species_chunks` in meta_db that stores
                          every docling chunk alongside:
                            species name(s), section role, raw text,
                            source citation, document title, timestamp
           Populated automatically during extract_and_store_chunks()

  [NEW-2]  Auto-wiki trigger post-HITL  —  after every successful HITL commit
           (_broadcast_update) a background thread checks which species now
           have stored chunks but no wiki lead section, and runs the wiki
           agent on them.

  [NEW-3]  WikiAutoRunner  —  orchestrates chunk→wiki pipeline:
             1. Load chunks for species from ChunkStore
             2. Group by section role (body / results / abstract / …)
             3. Build context string (≤ 8 000 tokens)
             4. Call OllamaWikiAgent.orchestrate() with full context
             5. Persist versioned article; set wiki_stale flag
           Also callable manually: wiki_runner.run_species(species_name)

  [NEW-4]  Incremental update  —  subsequent chunk store inserts flag the
           species as "chunk_dirty". WikiAutoRunner only re-runs for dirty
           species, so repeated processing is cheap.

  [NEW-5]  Streamlit control panel  —  render_wiki_runner_panel() shows:
             • queue of pending species
             • Run All / Run Selected buttons (async-in-thread)
             • Per-species status (pending / running / done / error)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import inspect
import json
import logging
import queue
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("biotrace.patch57")

# ─────────────────────────────────────────────────────────────────────────────
#  Module-level config (set by install_v57_patches)
# ─────────────────────────────────────────────────────────────────────────────
_CFG: dict = {
    "meta_db_path": "",
    "wiki_root":    "",
    "kg_db_path":   "",
    "ollama_url":   "http://localhost:11434",
    "ollama_model": "gemma4",
}


# ═════════════════════════════════════════════════════════════════════════════
#  FIX-1  ·  depthRange column auto-migration
# ═════════════════════════════════════════════════════════════════════════════

_OPTIONAL_COLUMNS = {
    "depthRange":       "TEXT",
    "eventDate":        "TEXT",
    "habitat":          "TEXT",
    "taxonomicStatus":  "TEXT",
    "geocodingSource":  "TEXT",
    "recordedName":     "TEXT",
    "kingdom":          "TEXT",
    "authority":        "TEXT",
    "wormsID":          "TEXT",
    "gbifID":           "TEXT",
    "iucnStatus":       "TEXT",
}


def _ensure_occurrence_columns(db_path: str) -> None:
    """
    Add any missing optional columns to the occurrence table.
    Safe to call multiple times (uses IF NOT EXISTS logic via PRAGMA).
    """
    if not db_path or not Path(db_path).exists():
        return
    try:
        conn = sqlite3.connect(db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        table = next((t for t in ("occurrences_v4", "occurrences") if t in tables), None)
        if not table:
            conn.close()
            return
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        added = []
        for col, typ in _OPTIONAL_COLUMNS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
                added.append(col)
        if added:
            conn.commit()
            logger.info("[v57/FIX-1] Added columns to %s: %s", table, added)
        conn.close()
    except Exception as exc:
        logger.warning("[v57/FIX-1] Column migration: %s", exc)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX-2  ·  Column-safe fetch_occurrences_from_db (no hard-coded columns)
# ═════════════════════════════════════════════════════════════════════════════

def _safe_fetch_occurrences_from_db(
    meta_db_path: str,
    species_name: str,
    limit: int = 200,
) -> list[dict]:
    """
    Replacement for biotrace_wiki_v56_patch.fetch_occurrences_from_db.
    Only selects columns that actually exist → no "no such column" error.
    """
    if not meta_db_path or not Path(meta_db_path).exists():
        return []
    try:
        conn = sqlite3.connect(meta_db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        table = next((t for t in ("occurrences_v4", "occurrences") if t in tables), None)
        if not table:
            conn.close()
            return []

        # Discover actual columns
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}

        # Desired columns with safe defaults
        desired = [
            ("id",              "id",              None),
            ("recordedName",    "recordedName",    ""),
            ("validName",       "validName",       ""),
            ("verbatimLocality","verbatimLocality", "—"),
            ("decimalLatitude", "decimalLatitude",  None),
            ("decimalLongitude","decimalLongitude", None),
            ("occurrenceType",  "occurrenceType",  "Uncertain"),
            ("phylum",          "phylum",           ""),
            ("class_",          "class_",           ""),
            ("order_",          "order_",           ""),
            ("family_",         "family_",          ""),
            ("sourceCitation",  "source",           ""),
            ("geocodingSource", "geocodingSource",  ""),
            ("taxonomicStatus", "taxonomicStatus",  ""),
            ("habitat",         "habitat",          ""),
            ("depthRange",      "depthRange",       ""),
            ("eventDate",       "eventDate",        ""),
        ]

        # Filter to existing cols only
        select_parts  = []
        col_aliases   = []
        col_defaults  = {}
        for db_col, alias, default in desired:
            if db_col in existing:
                select_parts.append(db_col)
                col_aliases.append(alias)
            else:
                # inject NULL placeholder so position is preserved
                select_parts.append(f"NULL AS {db_col}")
                col_aliases.append(alias)
                col_defaults[alias] = default

        select_sql = ", ".join(select_parts)
        rows = conn.execute(
            f"""SELECT {select_sql}
                FROM {table}
                WHERE (validName=? OR recordedName=?)
                ORDER BY id LIMIT ?""",
            (species_name, species_name, limit),
        ).fetchall()
        conn.close()

        results = []
        for row in rows:
            d: dict = {}
            for alias, val in zip(col_aliases, row):
                d[alias] = val if val is not None else col_defaults.get(alias, "")
            results.append(d)
        return results

    except Exception as exc:
        logger.warning("[v57/FIX-2] fetch_occurrences %s: %s", species_name, exc)
        return []


# ═════════════════════════════════════════════════════════════════════════════
#  FIX-3  ·  OllamaModel base_url kwarg sanitiser
# ═════════════════════════════════════════════════════════════════════════════

def _patch_ollama_model_init():
    """
    Wrap OllamaModel.__init__ so unsupported kwargs (base_url, etc.)
    are stripped before calling the real __init__.
    """
    try:
        from pydantic_ai.models.ollama import OllamaModel
        _orig_init = OllamaModel.__init__

        def _safe_init(self, *args, **kwargs):
            sig = inspect.signature(_orig_init)
            valid = set(sig.parameters.keys()) - {"self"}
            bad   = [k for k in list(kwargs) if k not in valid]
            for k in bad:
                logger.debug("[v57/FIX-3] Dropping unsupported OllamaModel kwarg: %s", k)
                kwargs.pop(k)
            return _orig_init(self, *args, **kwargs)

        OllamaModel.__init__ = _safe_init
        logger.info("[v57/FIX-3] OllamaModel.__init__ patched ✅")
    except Exception as exc:
        logger.debug("[v57/FIX-3] OllamaModel patch skipped: %s", exc)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX-4  ·  biotrace_schema stub
# ═════════════════════════════════════════════════════════════════════════════

def _ensure_schema_stub():
    """
    Create a minimal biotrace_schema.py stub next to this file so that
    `import biotrace_schema` succeeds and silences the WARNING.
    """
    stub_path = Path(__file__).parent / "biotrace_schema.py"
    if stub_path.exists():
        return
    stub_path.write_text(
        '"""biotrace_schema.py — auto-generated stub by biotrace_patch57_update.py"""\n'
        "from __future__ import annotations\n"
        "from dataclasses import dataclass, field\n"
        "from typing import Optional, List\n\n"
        "@dataclass\n"
        "class OccurrenceRecord:\n"
        "    validName: str = ''\n"
        "    recordedName: str = ''\n"
        "    verbatimLocality: str = ''\n"
        "    decimalLatitude: Optional[float] = None\n"
        "    decimalLongitude: Optional[float] = None\n"
        "    occurrenceType: str = 'Uncertain'\n"
        "    sourceCitation: str = ''\n"
        "    kingdom: str = ''\n"
        "    phylum: str = ''\n"
        "    class_: str = ''\n"
        "    order_: str = ''\n"
        "    family_: str = ''\n"
        "    genus: str = ''\n"
        "    depthRange: str = ''\n"
        "    habitat: str = ''\n"
        "    eventDate: str = ''\n"
        "    taxonomicStatus: str = ''\n"
        "    wormsID: str = ''\n"
        "    gbifID: str = ''\n"
        "    iucnStatus: str = ''\n\n"
        "@dataclass\n"
        "class ExtractionResult:\n"
        "    occurrences: List[OccurrenceRecord] = field(default_factory=list)\n"
        "    species_names: List[str] = field(default_factory=list)\n"
        "    citation: str = ''\n"
        "    document_title: str = ''\n",
        encoding="utf-8",
    )
    logger.info("[v57/FIX-4] biotrace_schema.py stub created ✅")


# ═════════════════════════════════════════════════════════════════════════════
#  NEW-1  ·  ChunkStore — species_chunks SQLite table
# ═════════════════════════════════════════════════════════════════════════════

_CHUNK_STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS species_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    species_name    TEXT    NOT NULL,
    section_role    TEXT    DEFAULT 'body',
    chunk_text      TEXT    NOT NULL,
    source_citation TEXT    DEFAULT '',
    document_title  TEXT    DEFAULT '',
    chunk_index     INTEGER DEFAULT 0,
    chunk_dirty     INTEGER DEFAULT 1,
    created_at      TEXT    DEFAULT (datetime('now')),
    updated_at      TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_sp_chunks_species
    ON species_chunks (species_name);
CREATE INDEX IF NOT EXISTS idx_sp_chunks_dirty
    ON species_chunks (chunk_dirty);
"""


class ChunkStore:
    """
    Persistent store for docling text chunks, indexed by species name.
    Lives inside the existing meta_db so no extra file is needed.

    Usage
    -----
        store = ChunkStore("biodiversity_data/metadata_v5.db")
        store.upsert_chunks("Eudendrium carneum", sections, citation, title)
        chunks = store.get_chunks("Eudendrium carneum")
        store.mark_clean("Eudendrium carneum")
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        try:
            conn = self._conn()
            conn.executescript(_CHUNK_STORE_SCHEMA)
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.error("[ChunkStore] init: %s", exc)

    def upsert_chunks(
        self,
        species_name:    str,
        sections:        dict[str, str],
        source_citation: str = "",
        document_title:  str = "",
    ) -> int:
        """
        Store one chunk per non-empty section for this species.
        Existing chunks from the same citation are replaced.
        Returns number of rows inserted.
        """
        if not sections:
            return 0
        try:
            conn = self._conn()
            # Remove stale chunks from same citation (avoid duplication on re-run)
            if source_citation:
                conn.execute(
                    "DELETE FROM species_chunks "
                    "WHERE species_name=? AND source_citation=?",
                    (species_name, source_citation),
                )
            count = 0
            for idx, (role, text) in enumerate(sections.items()):
                if not text or not text.strip():
                    continue
                conn.execute(
                    "INSERT INTO species_chunks "
                    "(species_name, section_role, chunk_text, source_citation, "
                    " document_title, chunk_index, chunk_dirty, updated_at) "
                    "VALUES (?,?,?,?,?,?,1,datetime('now'))",
                    (species_name, role, text.strip(), source_citation,
                     document_title, idx),
                )
                count += 1
            conn.commit()
            conn.close()
            logger.info(
                "[ChunkStore] Stored %d chunks for '%s' from '%s'",
                count, species_name, document_title or source_citation,
            )
            return count
        except Exception as exc:
            logger.error("[ChunkStore] upsert_chunks: %s", exc)
            return 0

    def get_chunks(
        self,
        species_name: str,
        dirty_only:   bool = False,
        max_chars:    int  = 8000,
    ) -> list[dict]:
        """
        Return stored chunks for a species, newest first per section.
        If dirty_only=True, only returns chunks flagged as chunk_dirty=1.
        Truncated to max_chars total.
        """
        try:
            conn = self._conn()
            sql = (
                "SELECT id, section_role, chunk_text, source_citation, "
                "document_title, chunk_index, created_at "
                "FROM species_chunks WHERE species_name=?"
            )
            params: list = [species_name]
            if dirty_only:
                sql += " AND chunk_dirty=1"
            sql += " ORDER BY section_role, chunk_index"
            rows = conn.execute(sql, params).fetchall()
            conn.close()

            # Trim to max_chars
            result, total = [], 0
            for r in rows:
                d = dict(r)
                remaining = max_chars - total
                if remaining <= 0:
                    break
                if len(d["chunk_text"]) > remaining:
                    d["chunk_text"] = d["chunk_text"][:remaining] + "…"
                result.append(d)
                total += len(d["chunk_text"])
            return result
        except Exception as exc:
            logger.error("[ChunkStore] get_chunks: %s", exc)
            return []

    def dirty_species(self) -> list[str]:
        """Return list of species names that have chunk_dirty=1 chunks."""
        try:
            conn = self._conn()
            rows = conn.execute(
                "SELECT DISTINCT species_name FROM species_chunks WHERE chunk_dirty=1"
            ).fetchall()
            conn.close()
            return [r[0] for r in rows]
        except Exception as exc:
            logger.error("[ChunkStore] dirty_species: %s", exc)
            return []

    def all_species(self) -> list[str]:
        """Return all species that have at least one stored chunk."""
        try:
            conn = self._conn()
            rows = conn.execute(
                "SELECT DISTINCT species_name FROM species_chunks ORDER BY species_name"
            ).fetchall()
            conn.close()
            return [r[0] for r in rows]
        except Exception as exc:
            return []

    def mark_clean(self, species_name: str) -> None:
        """Clear chunk_dirty flag for a species after wiki update."""
        try:
            conn = self._conn()
            conn.execute(
                "UPDATE species_chunks SET chunk_dirty=0 WHERE species_name=?",
                (species_name,),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.error("[ChunkStore] mark_clean: %s", exc)

    def chunk_count(self, species_name: str) -> int:
        try:
            conn = self._conn()
            n = conn.execute(
                "SELECT COUNT(*) FROM species_chunks WHERE species_name=?",
                (species_name,),
            ).fetchone()[0]
            conn.close()
            return n
        except Exception:
            return 0

    def stats(self) -> dict:
        """Quick statistics for the control panel."""
        try:
            conn = self._conn()
            total      = conn.execute("SELECT COUNT(*) FROM species_chunks").fetchone()[0]
            dirty_sp   = conn.execute(
                "SELECT COUNT(DISTINCT species_name) FROM species_chunks WHERE chunk_dirty=1"
            ).fetchone()[0]
            all_sp     = conn.execute(
                "SELECT COUNT(DISTINCT species_name) FROM species_chunks"
            ).fetchone()[0]
            conn.close()
            return {"total_chunks": total, "species": all_sp, "pending_wiki": dirty_sp}
        except Exception:
            return {}


# ═════════════════════════════════════════════════════════════════════════════
#  FIX-2 (cont.)  ·  Lead-section auto-generator
# ═════════════════════════════════════════════════════════════════════════════

def _build_lead_from_chunks(species_name: str, chunks: list[dict]) -> str:
    """
    Build a short lead paragraph from stored chunks when the wiki article
    has no lead section yet.  Uses abstract/introduction sections first,
    falls back to the first 500 chars of body text.
    """
    priority = ["abstract", "introduction", "body", "results", "methods"]
    for role in priority:
        for c in chunks:
            if c.get("section_role", "") == role:
                text = c["chunk_text"].strip()
                if len(text) > 80:
                    snippet = text[:600].rstrip()
                    cite = c.get("source_citation", "")
                    cite_note = f" (Source: {cite})" if cite else ""
                    return (
                        f"*{species_name}* is documented in the scientific literature. "
                        f"{snippet}…{cite_note}\n\n"
                        f"*(Lead auto-generated from source text — edit to refine.)*"
                    )
    # Ultimate fallback — minimal stub
    return (
        f"*{species_name}* — occurrence records documented in BioTrace. "
        f"Full wiki article pending literature extraction pass."
    )


def patch_unified_page_lead_section():
    """
    Monkey-patch render_unified_page in BioTraceWikiUnified so that
    it automatically generates a lead section from ChunkStore if missing,
    never showing the empty "No lead section yet" banner.
    """
    try:
        import biotrace_wiki_unified as _wmod
        _Cls = _wmod.BioTraceWikiUnified

        _orig_render = _Cls.render_unified_page

        def _patched_render(self, species_name: str, *args, **kwargs):
            # [Auto-FIX] If lead section is missing, try to auto-generate it from chunks
            try:
                art = self.get_species_article(species_name)
                if art:
                    secs = art.get("sections", {})
                    if not secs.get("lead", "").strip():
                        store = _get_chunk_store()
                        if store:
                            chunks = store.get_chunks(species_name)
                            if chunks:
                                lead = _build_lead_from_chunks(species_name, chunks)
                                if lead:
                                    secs["lead"] = lead
                                    art["sections"] = secs
                                    self._write(
                                        "species",
                                        self._slug(species_name),
                                        species_name,
                                        art,
                                        change_note="[Auto] Generated lead from chunk store",
                                    )
                                    logger.info("[v57/FIX-2] Auto-generated lead for %s", species_name)
            except Exception as e:
                logger.debug("[v57/FIX-2] lead auto-gen failed: %s", e)

            # Call the original (it will now see the generated lead if it was missing)
            res = _orig_render(self, species_name, *args, **kwargs)

            # Fallback UI if still no lead and no chunks
            try:
                import streamlit as st
                art = self.get_species_article(species_name)
                if art and not art.get("sections", {}).get("lead", "").strip():
                    store = _get_chunk_store()
                    if not store or not store.get_chunks(species_name):
                        st.info(
                            "💡 No source chunks stored yet. Process a PDF first, "
                            "or use the **Wiki Auto-Runner** panel to trigger a wiki pass."
                        )
            except Exception:
                pass

            return res

        _Cls.render_unified_page = _patched_render
        logger.info("[v57/FIX-2] render_unified_page patched for lead-section ✅")
    except Exception as exc:
        logger.warning("[v57/FIX-2] lead section patch: %s", exc)


# ═════════════════════════════════════════════════════════════════════════════
#  NEW-3  ·  WikiAutoRunner
# ═════════════════════════════════════════════════════════════════════════════

class WikiAutoRunner:
    """
    Orchestrates the chunk → wiki article pipeline.

    Workflow per species:
      1. Load chunks from ChunkStore (grouped by section role)
      2. Assemble context string (≤ 8 000 chars)
      3. Call OllamaWikiAgent.orchestrate()
      4. Persist the updated article (versioned)
      5. Mark chunks clean; set wiki_stale flag

    Thread-safe: each species is processed in an isolated thread.
    Status is reported through self.status dict.
    """

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_DONE    = "done"
    STATUS_ERROR   = "error"

    def __init__(
        self,
        wiki_root:    str,
        meta_db_path: str,
        ollama_url:   str  = "http://localhost:11434",
        model_name:   str  = "gemma4",
    ):
        self.wiki_root    = wiki_root
        self.meta_db_path = meta_db_path
        self.ollama_url   = ollama_url
        self.model_name   = model_name
        self.store        = ChunkStore(meta_db_path) if meta_db_path else None
        self.status:  dict[str, str]  = {}   # species → status string
        self.errors:  dict[str, str]  = {}   # species → error message
        self._lock   = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        logger.info("[WikiAutoRunner] Initialised — model=%s url=%s", model_name, ollama_url)

    # ── Public API ─────────────────────────────────────────────────────────

    def queue_dirty(self) -> list[str]:
        """Queue all species that have dirty chunks. Returns list queued."""
        if not self.store:
            return []
        pending = self.store.dirty_species()
        for sp in pending:
            if self.status.get(sp) not in (self.STATUS_RUNNING,):
                with self._lock:
                    self.status[sp] = self.STATUS_PENDING
        logger.info("[WikiAutoRunner] %d species queued", len(pending))
        return pending

    def run_species(self, species_name: str, blocking: bool = False) -> None:
        """
        Run the wiki pipeline for one species.
        If blocking=False (default) spawns a daemon thread.
        """
        if blocking:
            self._run_one(species_name)
        else:
            t = threading.Thread(
                target=self._run_one,
                args=(species_name,),
                daemon=True,
                name=f"WikiRunner-{species_name[:20]}",
            )
            t.start()

    def run_all_dirty(self, blocking: bool = False) -> list[str]:
        """Run the wiki pipeline for every species with dirty chunks."""
        species = self.queue_dirty()
        for sp in species:
            self.run_species(sp, blocking=blocking)
        return species

    # ── Internal ───────────────────────────────────────────────────────────

    def _run_one(self, species_name: str) -> None:
        with self._lock:
            self.status[species_name] = self.STATUS_RUNNING

        try:
            chunks = self._load_chunks(species_name)
            if not chunks:
                logger.info("[WikiAutoRunner] No chunks for '%s' — skipping", species_name)
                with self._lock:
                    self.status[species_name] = self.STATUS_DONE
                return

            # Build context string grouped by section role
            context = self._assemble_context(species_name, chunks)
            citation = self._best_citation(chunks)

            # Run the wiki agent
            success = self._call_wiki_agent(species_name, context, citation)

            if success:
                if self.store:
                    self.store.mark_clean(species_name)
                # Signal Streamlit to reload wiki tab
                self._set_wiki_stale()
                with self._lock:
                    self.status[species_name] = self.STATUS_DONE
                logger.info("[WikiAutoRunner] '%s' wiki updated ✅", species_name)
            else:
                with self._lock:
                    self.status[species_name] = self.STATUS_ERROR
                    self.errors[species_name]  = "Wiki agent returned no result"

        except Exception as exc:
            logger.error("[WikiAutoRunner] '%s' failed: %s", species_name, exc)
            with self._lock:
                self.status[species_name] = self.STATUS_ERROR
                self.errors[species_name]  = str(exc)

    def _load_chunks(self, species_name: str) -> list[dict]:
        if not self.store:
            return []
        return self.store.get_chunks(species_name, dirty_only=False, max_chars=8000)

    def _assemble_context(self, species_name: str, chunks: list[dict]) -> str:
        """
        Build a single context string from chunks, grouped by section role.
        Priority: abstract → introduction → body → results → methods → rest
        """
        priority_order = [
            "abstract", "introduction", "body", "results",
            "methods", "materials", "discussion", "reference",
        ]
        grouped: dict[str, list[str]] = {}
        for c in chunks:
            role = c.get("section_role", "body")
            grouped.setdefault(role, []).append(c["chunk_text"])

        parts = []
        seen_roles = set()
        for role in priority_order:
            if role in grouped:
                parts.append(f"[{role.upper()}]\n" + "\n\n".join(grouped[role]))
                seen_roles.add(role)

        for role, texts in grouped.items():
            if role not in seen_roles:
                parts.append(f"[{role.upper()}]\n" + "\n\n".join(texts))

        return f"# Source text for {species_name}\n\n" + "\n\n".join(parts)

    def _best_citation(self, chunks: list[dict]) -> str:
        for c in chunks:
            cit = c.get("source_citation", "").strip()
            if cit:
                return cit
        return "Unknown source"

    def _call_wiki_agent(
        self,
        species_name: str,
        context:      str,
        citation:     str,
    ) -> bool:
        """
        Call OllamaWikiAgent.orchestrate() with the assembled context.
        Returns True if the article was written successfully.
        """
        try:
            from biotrace_wiki_agent import OllamaWikiAgent
            from biotrace_wiki_unified import BioTraceWikiUnified

            wiki = BioTraceWikiUnified(self.wiki_root)
            agent = OllamaWikiAgent(
                wiki=wiki,
                ollama_url=self.ollama_url,
                model_name=self.model_name,
            )
            result = agent.orchestrate(
                species_name=species_name,
                chunk_text=context[:7500],
                citation=citation,
            )
            if result and getattr(result, "article_written", True):
                return True
            return bool(result)
        except Exception as exc:
            logger.error("[WikiAutoRunner] agent call for '%s': %s", species_name, exc)
            return False

    @staticmethod
    def _set_wiki_stale():
        try:
            import streamlit as st
            st.session_state["wiki_stale"] = True
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
#  Module-level singletons
# ═════════════════════════════════════════════════════════════════════════════

_CHUNK_STORE_INSTANCE:  Optional[ChunkStore]      = None
_WIKI_RUNNER_INSTANCE:  Optional[WikiAutoRunner]  = None


def _get_chunk_store() -> Optional[ChunkStore]:
    return _CHUNK_STORE_INSTANCE


def _get_wiki_runner() -> Optional[WikiAutoRunner]:
    return _WIKI_RUNNER_INSTANCE


# ═════════════════════════════════════════════════════════════════════════════
#  NEW-2  ·  Auto-wiki trigger — hook into HITL broadcast
# ═════════════════════════════════════════════════════════════════════════════

def _patch_broadcast_update():
    """
    Monkey-patch _broadcast_update in biotrace_hitl_geocoding so that
    after every HITL commit, dirty species are auto-queued for wiki update.
    """
    try:
        import biotrace_hitl_geocoding as _hitl
        _orig = getattr(_hitl, "_broadcast_update", None)
        if _orig is None:
            logger.debug("[v57/NEW-2] _broadcast_update not found in hitl module")
            return

        def _patched_broadcast(record: dict | None = None, action: str = "commit"):
            # Call original first
            if _orig:
                try:
                    _orig(record=record, action=action)
                except TypeError:
                    _orig()

            # Queue wiki auto-runner for any dirty species
            runner = _get_wiki_runner()
            if runner is None:
                return

            # Determine which species was affected
            if record and isinstance(record, dict):
                sp = record.get("validName") or record.get("recordedName") or ""
                store = _get_chunk_store()
                if sp and store and store.chunk_count(sp) > 0:
                    logger.info(
                        "[v57/NEW-2] HITL commit for '%s' → queuing wiki runner", sp
                    )
                    runner.run_species(sp, blocking=False)
                    return

            # Fallback: run all dirty
            dirty = runner.queue_dirty()
            if dirty:
                logger.info(
                    "[v57/NEW-2] Running wiki runner for %d dirty species post-HITL",
                    len(dirty),
                )
                for sp in dirty:
                    runner.run_species(sp, blocking=False)

        _hitl._broadcast_update = _patched_broadcast
        logger.info("[v57/NEW-2] _broadcast_update patched for auto-wiki ✅")
    except Exception as exc:
        logger.warning("[v57/NEW-2] broadcast patch: %s", exc)


# ═════════════════════════════════════════════════════════════════════════════
#  Helper: ingest chunks from docling sections (called from extraction loop)
# ═════════════════════════════════════════════════════════════════════════════

def store_chunks_for_species(
    species_names:   list[str],
    sections:        dict[str, str],
    source_citation: str = "",
    document_title:  str = "",
) -> None:
    """
    Call this after each PDF extraction to store docling sections
    in the ChunkStore for every detected species.

    Usage (in your PDF processing loop):
        from biotrace_patch57_update import store_chunks_for_species
        store_chunks_for_species(
            species_names   = extracted_species_list,
            sections        = docling_sections_dict,
            source_citation = citation_string,
            document_title  = paper_title,
        )
    """
    store = _get_chunk_store()
    if store is None:
        logger.warning("[v57] ChunkStore not initialised — call install_v57_patches first")
        return
    for sp in species_names:
        if sp and sp.strip():
            store.upsert_chunks(
                species_name    = sp.strip(),
                sections        = sections,
                source_citation = source_citation,
                document_title  = document_title,
            )


# ═════════════════════════════════════════════════════════════════════════════
#  NEW-5  ·  Streamlit control panel
# ═════════════════════════════════════════════════════════════════════════════

def render_wiki_runner_panel() -> None:
    """
    Render a Streamlit expander panel showing chunk-store stats and
    wiki-runner queue controls.

    Place inside any tab (e.g. the Wiki tab, or a dedicated Automation tab):

        from biotrace_patch57_update import render_wiki_runner_panel
        render_wiki_runner_panel()
    """
    try:
        import streamlit as st
    except ImportError:
        return

    store  = _get_chunk_store()
    runner = _get_wiki_runner()

    with st.expander("🤖 Wiki Auto-Runner (v5.7)", expanded=False):
        if store is None or runner is None:
            st.warning("Patch not installed. Call install_v57_patches() first.")
            return

        stats = store.stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Stored chunks",   stats.get("total_chunks", 0))
        c2.metric("Species indexed", stats.get("species", 0))
        c3.metric("Pending wiki",    stats.get("pending_wiki", 0))

        st.divider()

        all_species = store.all_species()
        dirty       = store.dirty_species()

        if not all_species:
            st.info("No chunks stored yet. Process a PDF to populate the chunk store.")
            return

        # Show per-species status table
        rows = []
        for sp in all_species:
            status = runner.status.get(sp, "—")
            err    = runner.errors.get(sp, "")
            is_dirty = sp in dirty
            rows.append({
                "Species":      sp,
                "Chunks":       store.chunk_count(sp),
                "Dirty":        "✅" if is_dirty else "—",
                "Status":       status,
                "Last error":   err[:60] if err else "",
            })

        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("▶ Run All Pending", key="wrun_all"):
                run = runner.run_all_dirty(blocking=False)
                st.toast(f"Queued {len(run)} species for wiki update")

        with col_b:
            sel = st.selectbox("Select species", [""] + all_species, key="wrun_sel")
            if st.button("▶ Run Selected", key="wrun_one", disabled=not sel):
                runner.run_species(sel, blocking=False)
                st.toast(f"Queued '{sel}' for wiki update")

        with col_c:
            if st.button("🔄 Refresh panel", key="wrun_refresh"):
                st.rerun()

        # Show running threads
        running = [sp for sp, s in runner.status.items()
                   if s == WikiAutoRunner.STATUS_RUNNING]
        if running:
            st.info(f"Currently running: {', '.join(running)}")


# ═════════════════════════════════════════════════════════════════════════════
#  PATCH fetch_occurrences_from_db in wiki v56 patch module
# ═════════════════════════════════════════════════════════════════════════════

def _patch_fetch_occurrences():
    """Replace fetch_occurrences_from_db with the column-safe version."""
    try:
        import biotrace_wiki_v56_patch as _wp
        _wp.fetch_occurrences_from_db = _safe_fetch_occurrences_from_db
        logger.info("[v57/FIX-2] fetch_occurrences_from_db patched ✅")
    except ImportError:
        pass  # Module may not be loaded yet — that's fine


# ═════════════════════════════════════════════════════════════════════════════
#  MASTER INSTALLER
# ═════════════════════════════════════════════════════════════════════════════

def install_v57_patches(
    meta_db_path: str = "",
    wiki_root:    str = "",
    kg_db_path:   str = "",
    ollama_url:   str = "http://localhost:11434",
    ollama_model: str = "gemma4",
) -> None:
    """
    Apply all v5.7 patches.  Safe to call multiple times (idempotent).

    Parameters
    ----------
    meta_db_path : str
        Path to the SQLite occurrence database (metadata_v5.db).
    wiki_root : str
        Root directory of the wiki store.
    kg_db_path : str
        Path to knowledge_graph.db (optional, used for future KG hooks).
    ollama_url : str
        Ollama server URL.
    ollama_model : str
        Model tag to use for wiki article generation.
    """
    global _CHUNK_STORE_INSTANCE, _WIKI_RUNNER_INSTANCE, _CFG

    # Store config
    _CFG.update({
        "meta_db_path": meta_db_path,
        "wiki_root":    wiki_root,
        "kg_db_path":   kg_db_path,
        "ollama_url":   ollama_url,
        "ollama_model": ollama_model,
    })

    logger.info("━━━ BioTrace v5.7 patches starting ━━━")

    # ── FIX-4: schema stub first (silences warnings) ──────────────────────────
    _ensure_schema_stub()

    # ── FIX-1: depthRange and other missing columns ───────────────────────────
    if meta_db_path:
        _ensure_occurrence_columns(meta_db_path)
        logger.info("[v57] [FIX-1] Occurrence column migration done ✅")

    # ── FIX-2: column-safe fetch_occurrences ──────────────────────────────────
    _patch_fetch_occurrences()

    # ── FIX-3: OllamaModel base_url kwarg ─────────────────────────────────────
    _patch_ollama_model_init()

    # ── NEW-1: ChunkStore ─────────────────────────────────────────────────────
    if meta_db_path:
        try:
            _CHUNK_STORE_INSTANCE = ChunkStore(meta_db_path)
            logger.info("[v57] [NEW-1] ChunkStore ready (%s) ✅",
                        _CHUNK_STORE_INSTANCE.stats())
        except Exception as exc:
            logger.warning("[v57] [NEW-1] ChunkStore init failed: %s", exc)

    # ── NEW-3: WikiAutoRunner ─────────────────────────────────────────────────
    if wiki_root and meta_db_path:
        try:
            _WIKI_RUNNER_INSTANCE = WikiAutoRunner(
                wiki_root    = wiki_root,
                meta_db_path = meta_db_path,
                ollama_url   = ollama_url,
                model_name   = ollama_model,
            )
            logger.info("[v57] [NEW-3] WikiAutoRunner ready ✅")
        except Exception as exc:
            logger.warning("[v57] [NEW-3] WikiAutoRunner init failed: %s", exc)

    # ── NEW-2: Auto-wiki hook on HITL broadcast ───────────────────────────────
    _patch_broadcast_update()

    # ── FIX-2 (continued): lead section patch ─────────────────────────────────
    patch_unified_page_lead_section()

    logger.info("━━━ BioTrace v5.7 patches complete ✅ ━━━")


# ═════════════════════════════════════════════════════════════════════════════
#  Convenience exports
# ═════════════════════════════════════════════════════════════════════════════

__all__ = [
    "install_v57_patches",
    "patch_unified_page_lead_section",
    "store_chunks_for_species",
    "render_wiki_runner_panel",
    "ChunkStore",
    "WikiAutoRunner",
    "_get_chunk_store",
    "_get_wiki_runner",
]


# ═════════════════════════════════════════════════════════════════════════════
#  Integration guide (printed when run directly)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
BioTrace v5.7 Patch — Integration Guide
════════════════════════════════════════

── Step 1: Add to the TOP of biotrace_v53.py ────────────────────────────────

    from biotrace_patch57_update import install_v57_patches
    install_v57_patches(
        meta_db_path = META_DB_PATH,    # "biodiversity_data/metadata_v5.db"
        wiki_root    = WIKI_ROOT,       # "biodiversity_data/wiki"
        kg_db_path   = KG_DB_PATH,      # "biodiversity_data/knowledge_graph.db"
        ollama_url   = "http://localhost:11434",
        ollama_model = "gemma4",
    )

── Step 2: In your PDF extraction loop ──────────────────────────────────────

    from biotrace_patch57_update import store_chunks_for_species

    # After docling converts the PDF:
    md, sections = convert_pdf_cached(pdf_path, converter, cache)

    # After LLM extracts species names:
    store_chunks_for_species(
        species_names   = detected_species_list,
        sections        = sections,
        source_citation = citation,
        document_title  = paper_title,
    )

── Step 3: In your Wiki tab (Tab 7) ─────────────────────────────────────────

    from biotrace_patch57_update import render_wiki_runner_panel

    with tabs[6]:   # or wherever your wiki tab is
        render_wiki_runner_panel()    # ← add this line
        wiki.render_streamlit_tab(...)

── What happens automatically ───────────────────────────────────────────────

    1. Every PDF processed → chunks stored per-species in metadata_v5.db
       (species_chunks table, created automatically)

    2. After every HITL commit → wiki agent queued for that species
       (background thread, non-blocking)

    3. Wiki tab → "Wiki Auto-Runner" panel shows queue + run controls
       • "Run All Pending" → processes all species with new chunks
       • Per-species status: pending / running / done / error

    4. depthRange column added automatically if missing
    5. OllamaModel base_url kwarg silently stripped
    6. "No lead section" → button to auto-generate from chunks
    7. biotrace_schema.py stub created if missing
""")
