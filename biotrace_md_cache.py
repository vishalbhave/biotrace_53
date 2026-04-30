"""
biotrace_md_cache.py  —  BioTrace v5.6  Markdown Cache Layer
═══════════════════════════════════════════════════════════════════════════════
Caches docling-converted documents as Markdown files to avoid re-processing.

Design
──────
• Hash-based: SHA-256 of the raw PDF bytes → unique cache key
• Stores converted Markdown at  <cache_dir>/<hash>.md
• Stores section-split JSON at  <cache_dir>/<hash>.sections.json
• Thread-safe (SQLite manifest for atomic cache state tracking)

Public API
──────────
    from biotrace_md_cache import DoclingMDCache

    cache = DoclingMDCache("biodiversity_data/md_cache")

    # Check before running docling
    md, sections = cache.get(pdf_path)
    if md is None:
        # Run docling (expensive)
        md, sections = run_docling(pdf_path)
        cache.put(pdf_path, md, sections)

    # Use cached result
    ...
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("biotrace.md_cache")

_MANIFEST_SCHEMA = """
CREATE TABLE IF NOT EXISTS md_cache_manifest (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash   TEXT NOT NULL UNIQUE,
    source_path TEXT NOT NULL,
    md_file     TEXT NOT NULL,
    sec_file    TEXT NOT NULL,
    page_count  INTEGER DEFAULT 0,
    word_count  INTEGER DEFAULT 0,
    created_at  TEXT DEFAULT (datetime('now')),
    last_hit    TEXT
);
CREATE INDEX IF NOT EXISTS idx_mc_hash ON md_cache_manifest(file_hash);
"""


def _file_hash(path: str | Path) -> str:
    """SHA-256 of the first 4 MB + file size (fast but collision-resistant enough)."""
    p = Path(path)
    h = hashlib.sha256()
    h.update(str(p.stat().st_size).encode())
    with p.open("rb") as fh:
        h.update(fh.read(4 * 1024 * 1024))
    return h.hexdigest()


class DoclingMDCache:
    """
    Persistent Markdown cache for docling-converted PDFs.

    Parameters
    ----------
    cache_dir : str
        Directory where .md and .sections.json files are stored.
    """

    def __init__(self, cache_dir: str = "biodiversity_data/md_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.cache_dir / "manifest.db")
        self._init_db()

    def _init_db(self):
        con = sqlite3.connect(self.db_path)
        con.executescript(_MANIFEST_SCHEMA)
        con.commit()
        con.close()

    # ── Public: get ───────────────────────────────────────────────────────────

    def get(
        self, pdf_path: str | Path
    ) -> tuple[Optional[str], Optional[dict]]:
        """
        Return (markdown_text, sections_dict) if cached, else (None, None).

        sections_dict maps docling section roles → text, e.g.:
            {"body": "...", "table": "...", "reference": "..."}
        """
        try:
            h = _file_hash(pdf_path)
            con = sqlite3.connect(self.db_path)
            row = con.execute(
                "SELECT md_file, sec_file FROM md_cache_manifest WHERE file_hash=?", (h,)
            ).fetchone()
            if row:
                md_path, sec_path = Path(row[0]), Path(row[1])
                if md_path.exists() and sec_path.exists():
                    md = md_path.read_text(encoding="utf-8")
                    sections = json.loads(sec_path.read_text(encoding="utf-8"))
                    con.execute(
                        "UPDATE md_cache_manifest SET last_hit=? WHERE file_hash=?",
                        (datetime.now().isoformat(), h),
                    )
                    con.commit()
                    con.close()
                    logger.info("[MDCache] HIT  %s → %s", Path(pdf_path).name, md_path.name)
                    return md, sections
            con.close()
        except Exception as exc:
            logger.warning("[MDCache] get error: %s", exc)
        return None, None

    # ── Public: put ───────────────────────────────────────────────────────────

    def put(
        self,
        pdf_path:  str | Path,
        markdown:  str,
        sections:  dict,
        page_count: int = 0,
    ) -> bool:
        """
        Persist markdown + sections dict to cache.

        Parameters
        ----------
        pdf_path  : original PDF path (used for hash + display)
        markdown  : full markdown text from docling
        sections  : dict mapping section roles → text blocks
        page_count: optional page count for manifest metadata
        """
        try:
            h = _file_hash(pdf_path)
            md_file  = self.cache_dir / f"{h}.md"
            sec_file = self.cache_dir / f"{h}.sections.json"

            md_file.write_text(markdown, encoding="utf-8")
            sec_file.write_text(
                json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            word_count = len(markdown.split())
            con = sqlite3.connect(self.db_path)
            con.execute(
                """INSERT OR REPLACE INTO md_cache_manifest
                   (file_hash, source_path, md_file, sec_file, page_count, word_count)
                   VALUES (?,?,?,?,?,?)""",
                (h, str(pdf_path), str(md_file), str(sec_file), page_count, word_count),
            )
            con.commit()
            con.close()
            logger.info(
                "[MDCache] STORE %s → %s (%d words, %d pages)",
                Path(pdf_path).name, h[:12], word_count, page_count,
            )
            return True
        except Exception as exc:
            logger.error("[MDCache] put error: %s", exc)
            return False

    # ── Public: stats / management ────────────────────────────────────────────

    def list_cached(self) -> list[dict]:
        """Return manifest rows as list of dicts for display."""
        try:
            con = sqlite3.connect(self.db_path)
            rows = con.execute(
                "SELECT file_hash, source_path, word_count, page_count, created_at, last_hit "
                "FROM md_cache_manifest ORDER BY created_at DESC"
            ).fetchall()
            con.close()
            return [
                {
                    "hash":        r[0][:12],
                    "source":      Path(r[1]).name,
                    "words":       r[2],
                    "pages":       r[3],
                    "cached":      r[4][:10],
                    "last_used":   (r[5] or "")[:10],
                }
                for r in rows
            ]
        except Exception:
            return []

    def clear(self, pdf_path: str | Path) -> bool:
        """Remove a specific file from the cache."""
        try:
            h = _file_hash(pdf_path)
            con = sqlite3.connect(self.db_path)
            row = con.execute(
                "SELECT md_file, sec_file FROM md_cache_manifest WHERE file_hash=?", (h,)
            ).fetchone()
            if row:
                for f in row:
                    Path(f).unlink(missing_ok=True)
                con.execute("DELETE FROM md_cache_manifest WHERE file_hash=?", (h,))
                con.commit()
            con.close()
            return True
        except Exception as exc:
            logger.error("[MDCache] clear error: %s", exc)
            return False

    def render_streamlit_panel(self):
        """Render a small Streamlit expander showing cache status."""
        try:
            import streamlit as st
            import pandas as pd

            with st.expander("📦 Docling MD Cache", expanded=False):
                rows = self.list_cached()
                if not rows:
                    st.info("Cache is empty — PDFs will be converted on first run.")
                    return
                st.success(f"✅ **{len(rows)} documents cached** — docling re-conversion skipped for these.")
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption(
                    "Cache location: `" + str(self.cache_dir) + "`  |  "
                    "Files: `<hash>.md` + `<hash>.sections.json`"
                )
        except ImportError:
            pass
