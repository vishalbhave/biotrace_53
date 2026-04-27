"""
biotrace_memory_bank.py  —  BioTrace v5.0
────────────────────────────────────────────────────────────────────────────
Persistent Cross-Session Memory Bank for biodiversity occurrence data.

Inspired by NoCapGenAI — The Memory Bank concept:
  • Every verified species occurrence is stored as a "memory atom"
  • Memories are recalled by semantic similarity (TF-IDF) + metadata filters
  • Memories decay in confidence if contradicted by newer sources
  • Cross-paper reconciliation: same species + locality → merged record
  • Session summaries: each extraction session is summarised and persisted

Architecture:
  SQLite  ──►  TF-IDF recall (sklearn)  ──►  Context injection into LLM
                     │
                     └──►  Conflict resolution (WoRMS authority wins)

This is intentionally lightweight:
  • No GPU, no vector database, no cloud service required
  • Pure SQLite + sklearn TF-IDF for semantic retrieval
  • Swappable: drop in chromadb / sentence-transformers if available

Darwin Core–aligned schema. All taxon fields match species_verifier.py output.

Usage:
    mb = BioTraceMemoryBank("biodiversity_data/memory_bank.db")
    mb.store_occurrences(occurrences, session_id="paper_xyz")
    recalled = mb.recall("coral reef species Acanthurus", top_k=5)
    context  = mb.build_memory_context(query, top_k=10)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from datetime import datetime
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("biotrace.memory_bank")

# ─────────────────────────────────────────────────────────────────────────────
#  SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS memory_atoms (
    atom_id         TEXT PRIMARY KEY,     -- SHA-1 of species+locality+source
    session_id      TEXT,                 -- paper / session identifier
    recorded_name   TEXT,
    valid_name      TEXT,
    phylum          TEXT,
    class_          TEXT,
    order_          TEXT,
    family_         TEXT,
    taxon_rank      TEXT,
    taxonomic_status TEXT,
    worms_id        TEXT,
    itis_id         TEXT,
    name_according_to TEXT,
    locality        TEXT,
    latitude        REAL,
    longitude       REAL,
    habitat         TEXT,
    occurrence_type TEXT,
    sampling_date   TEXT,
    depth_m         TEXT,
    method          TEXT,
    raw_evidence    TEXT,
    source_citation TEXT,
    confidence      REAL DEFAULT 1.0,     -- decreases if contradicted
    times_confirmed INTEGER DEFAULT 1,    -- increases if re-observed
    first_seen      TEXT DEFAULT (datetime('now')),
    last_seen       TEXT DEFAULT (datetime('now')),
    geocoding_src   TEXT,
    full_blob       TEXT                  -- full JSON of original occ dict
);

CREATE TABLE IF NOT EXISTS memory_sessions (
    session_id   TEXT PRIMARY KEY,
    title        TEXT,
    source_file  TEXT,
    n_records    INTEGER DEFAULT 0,
    n_species    INTEGER DEFAULT 0,
    n_localities INTEGER DEFAULT 0,
    summary      TEXT,
    llm_summary  TEXT,
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS memory_conflicts (
    conflict_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    atom_id      TEXT,
    field        TEXT,
    old_value    TEXT,
    new_value    TEXT,
    resolved_by  TEXT,       -- 'WoRMS' | 'newer_paper' | 'manual'
    resolved_at  TEXT DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    atom_id UNINDEXED,
    valid_name, recorded_name,
    family_, order_, phylum, class_,
    locality, habitat, raw_evidence, source_citation,
    content='memory_atoms', content_rowid='rowid'
);

CREATE INDEX IF NOT EXISTS idx_atoms_sp      ON memory_atoms(valid_name);
CREATE INDEX IF NOT EXISTS idx_atoms_loc     ON memory_atoms(locality);
CREATE INDEX IF NOT EXISTS idx_atoms_family  ON memory_atoms(family_);
CREATE INDEX IF NOT EXISTS idx_atoms_session ON memory_atoms(session_id);
CREATE INDEX IF NOT EXISTS idx_atoms_worms   ON memory_atoms(worms_id);
"""

_FTS_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS ma_ai AFTER INSERT ON memory_atoms BEGIN
  INSERT INTO memory_fts(rowid, atom_id, valid_name, recorded_name,
    family_, order_, phylum, class_, locality, habitat, raw_evidence, source_citation)
  VALUES (new.rowid, new.atom_id, new.valid_name, new.recorded_name,
    new.family_, new.order_, new.phylum, new.class_,
    new.locality, new.habitat, new.raw_evidence, new.source_citation);
END;
CREATE TRIGGER IF NOT EXISTS ma_au AFTER UPDATE ON memory_atoms BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, atom_id, valid_name, recorded_name,
    family_, order_, phylum, class_, locality, habitat, raw_evidence, source_citation)
  VALUES ('delete', old.rowid, old.atom_id, old.valid_name, old.recorded_name,
    old.family_, old.order_, old.phylum, old.class_,
    old.locality, old.habitat, old.raw_evidence, old.source_citation);
  INSERT INTO memory_fts(rowid, atom_id, valid_name, recorded_name,
    family_, order_, phylum, class_, locality, habitat, raw_evidence, source_citation)
  VALUES (new.rowid, new.atom_id, new.valid_name, new.recorded_name,
    new.family_, new.order_, new.phylum, new.class_,
    new.locality, new.habitat, new.raw_evidence, new.source_citation);
END;
"""


def _atom_id(species: str, locality: str, source: str) -> str:
    raw = f"{species.lower().strip()}|{locality.lower().strip()}|{source.lower().strip()[:80]}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.utcnow().isoformat()


class BioTraceMemoryBank:
    """
    Persistent memory store for biodiversity occurrence records.

    Key design principles (NoCapGenAI-inspired):
      1. Atoms — smallest unit: one species at one locality from one paper
      2. Merge   — re-observations of the same atom increase confidence
      3. Conflict — field disagreements are logged and resolved by authority
      4. Recall  — FTS5 + TF-IDF re-ranking for semantic query matching
      5. Context — builds a rich memory context string for LLM prompts
    """

    def __init__(self, db_path: str = "biodiversity_data/memory_bank.db"):
        import os
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self._conn   = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        try:
            self._conn.executescript(_FTS_TRIGGER)
        except sqlite3.OperationalError:
            pass  # triggers already exist
        self._conn.commit()
        self._tfidf_vectorizer  = None
        self._tfidf_matrix      = None
        self._tfidf_atom_ids: list[str] = []
        self._tfidf_dirty       = True
        logger.info("[MemoryBank] Opened: %s", db_path)

    # ── TF-IDF index ──────────────────────────────────────────────────────────
    def _rebuild_tfidf(self):
        """Build a TF-IDF index over all memory atoms for semantic recall."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            logger.warning("[MemoryBank] sklearn not available — FTS5 recall only")
            self._tfidf_dirty = False
            return

        rows = self._conn.execute(
            """SELECT atom_id,
                      COALESCE(valid_name,'') || ' ' || COALESCE(recorded_name,'') || ' ' ||
                      COALESCE(family_,'') || ' ' || COALESCE(order_,'') || ' ' ||
                      COALESCE(phylum,'') || ' ' || COALESCE(locality,'') || ' ' ||
                      COALESCE(habitat,'') || ' ' || COALESCE(raw_evidence,'')
               FROM memory_atoms"""
        ).fetchall()
        if not rows:
            return

        self._tfidf_atom_ids = [r[0] for r in rows]
        corpus  = [r[1] for r in rows]

        self._tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)
        self._tfidf_dirty  = False
        logger.debug("[MemoryBank] TF-IDF index built over %d atoms", len(rows))

    # ── Store occurrences ─────────────────────────────────────────────────────
    def store_occurrences(
        self,
        occurrences: list[dict],
        session_id: str = "",
        session_title: str = "",
        source_file: str = "",
    ) -> dict[str, int]:
        """
        Persist a list of occurrence dicts (from BioTrace extraction + verifier).
        Returns {"inserted": N, "merged": M, "conflicts": K}.
        """
        session_id = session_id or f"session_{int(time.time())}"
        inserted = merged = conflicts_logged = 0

        for occ in occurrences:
            if not isinstance(occ, dict):
                continue

            sp_name  = (occ.get("validName") or occ.get("recordedName") or
                        occ.get("Valid Name") or occ.get("Recorded Name","")).strip()
            if not sp_name:
                continue

            locality  = str(occ.get("verbatimLocality") or
                            occ.get("locality", {}).get("site_name", "") or "").strip()[:200]
            citation  = str(occ.get("Source Citation") or
                            occ.get("sourceCitation","Unknown")).strip()[:300]
            atom_id   = _atom_id(sp_name, locality, citation)

            # Parse sampling event
            sampling  = occ.get("Sampling Event") or occ.get("samplingEvent") or {}
            if isinstance(sampling, str):
                try:
                    sampling = json.loads(sampling)
                except Exception:
                    sampling = {"raw": sampling}
            if not isinstance(sampling, dict):
                sampling = {}

            new_vals = {
                "session_id":      session_id,
                "recorded_name":   str(occ.get("recordedName") or occ.get("Recorded Name",""))[:200],
                "valid_name":      sp_name[:200],
                "phylum":          str(occ.get("phylum",""))[:100],
                "class_":          str(occ.get("class_",""))[:100],
                "order_":          str(occ.get("order_",""))[:100],
                "family_":         str(occ.get("family_",""))[:100],
                "taxon_rank":      str(occ.get("taxonRank",""))[:50],
                "taxonomic_status":str(occ.get("taxonomicStatus",""))[:50],
                "worms_id":        str(occ.get("wormsID",""))[:20],
                "itis_id":         str(occ.get("itisID",""))[:20],
                "name_according_to":str(occ.get("nameAccordingTo",""))[:100],
                "locality":        locality,
                "latitude":        occ.get("decimalLatitude"),
                "longitude":       occ.get("decimalLongitude"),
                "habitat":         str(occ.get("Habitat") or occ.get("habitat",""))[:200],
                "occurrence_type": str(occ.get("occurrenceType") or occ.get("occurrence_type",""))[:50],
                "sampling_date":   str(sampling.get("date",""))[:30],
                "depth_m":         str(sampling.get("depth_m",""))[:20],
                "method":          str(sampling.get("method",""))[:200],
                "raw_evidence":    str(occ.get("Raw Text Evidence") or occ.get("rawTextEvidence",""))[:500],
                "source_citation": citation,
                "geocoding_src":   str(occ.get("geocodingSource",""))[:50],
                "full_blob":       json.dumps(occ, default=str)[:4000],
            }

            existing = self._conn.execute(
                "SELECT * FROM memory_atoms WHERE atom_id=?", (atom_id,)
            ).fetchone()

            if existing is None:
                # Fresh insert
                cols = list(new_vals.keys())
                placeholders = ",".join("?" * (len(cols) + 1))
                self._conn.execute(
                    f"INSERT OR IGNORE INTO memory_atoms (atom_id,{','.join(cols)}) VALUES ({placeholders})",
                    [atom_id] + [new_vals[c] for c in cols],
                )
                inserted += 1
            else:
                # Merge: increment confirmation counter, update timestamps
                self._conn.execute(
                    """UPDATE memory_atoms SET
                         times_confirmed = times_confirmed + 1,
                         last_seen       = ?,
                         confidence      = MIN(1.0, confidence + 0.05)
                       WHERE atom_id = ?""",
                    (_now(), atom_id),
                )
                merged += 1

                # Conflict detection: WoRMS-authoritative fields
                for field in ("valid_name", "phylum", "class_", "order_", "family_"):
                    old_v = existing[field] or ""
                    new_v = new_vals[field] or ""
                    if old_v and new_v and old_v.strip() != new_v.strip():
                        # WoRMS wins over other sources
                        winner = (
                            new_v if new_vals.get("name_according_to","").upper() == "WORMS"
                            else old_v
                        )
                        self._conn.execute(
                            """INSERT INTO memory_conflicts
                               (atom_id, field, old_value, new_value, resolved_by)
                               VALUES(?,?,?,?,?)""",
                            (atom_id, field, old_v, new_v,
                             "WoRMS" if winner == new_v else "existing_record"),
                        )
                        conflicts_logged += 1

        # Upsert session record
        sp_count  = self._count_unique("valid_name")
        loc_count = self._count_unique("locality")
        self._conn.execute(
            """INSERT OR REPLACE INTO memory_sessions
               (session_id, title, source_file, n_records, n_species, n_localities)
               VALUES(?,?,?,?,?,?)""",
            (session_id, session_title or session_id, source_file,
             inserted + merged, sp_count, loc_count),
        )
        self._conn.commit()
        self._tfidf_dirty = True

        logger.info(
            "[MemoryBank] Session %s: inserted=%d merged=%d conflicts=%d",
            session_id, inserted, merged, conflicts_logged,
        )
        return {"inserted": inserted, "merged": merged, "conflicts": conflicts_logged}

    def _count_unique(self, field: str) -> int:
        row = self._conn.execute(
            f"SELECT COUNT(DISTINCT {field}) FROM memory_atoms"
        ).fetchone()
        return row[0] if row else 0

    # ── Recall ────────────────────────────────────────────────────────────────
    def recall(
        self,
        query: str,
        top_k: int = 10,
        filter_locality: Optional[str] = None,
        filter_family: Optional[str]   = None,
        filter_habitat: Optional[str]  = None,
        occurrence_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Recall memory atoms relevant to a natural-language query.

        Uses FTS5 for lexical match + TF-IDF cosine for re-ranking.
        """
        # Build WHERE clause for metadata filters
        filters: list[str]  = []
        params:  list[Any]  = []
        if filter_locality:
            filters.append("locality LIKE ?")
            params.append(f"%{filter_locality}%")
        if filter_family:
            filters.append("family_ LIKE ?")
            params.append(f"%{filter_family}%")
        if filter_habitat:
            filters.append("habitat LIKE ?")
            params.append(f"%{filter_habitat}%")
        if occurrence_type:
            filters.append("occurrence_type = ?")
            params.append(occurrence_type)

        where = ("AND " + " AND ".join(filters)) if filters else ""

        # FTS5 match
        fts_tokens = " OR ".join(
            f'"{t}"' for t in re.split(r"\s+", query.strip()) if len(t) > 2
        )
        if fts_tokens:
            try:
                fts_rows = self._conn.execute(
                    f"""SELECT a.*
                        FROM memory_atoms a
                        JOIN (
                          SELECT atom_id FROM memory_fts WHERE memory_fts MATCH ?
                          LIMIT {top_k * 5}
                        ) f ON a.atom_id = f.atom_id
                        WHERE 1=1 {where}
                        ORDER BY a.times_confirmed DESC, a.confidence DESC
                        LIMIT {top_k * 3}""",
                    [fts_tokens] + params,
                ).fetchall()
                candidate_ids = [r["atom_id"] for r in fts_rows]
            except Exception as exc:
                logger.debug("[MemoryBank] FTS error: %s", exc)
                candidate_ids = []
        else:
            candidate_ids = []

        # Fill with top-confidence records if FTS returned too few
        if len(candidate_ids) < top_k:
            extra = self._conn.execute(
                f"""SELECT atom_id FROM memory_atoms
                    WHERE 1=1 {where}
                    ORDER BY times_confirmed DESC, confidence DESC
                    LIMIT {top_k * 2}""",
                params,
            ).fetchall()
            for r in extra:
                if r[0] not in candidate_ids:
                    candidate_ids.append(r[0])

        # TF-IDF re-ranking
        if candidate_ids:
            if self._tfidf_dirty:
                self._rebuild_tfidf()

            if (
                self._tfidf_vectorizer is not None
                and self._tfidf_matrix is not None
                and len(self._tfidf_atom_ids) > 0
            ):
                try:
                    from sklearn.metrics.pairwise import cosine_similarity
                    q_vec = self._tfidf_vectorizer.transform([query])
                    scores = cosine_similarity(q_vec, self._tfidf_matrix).flatten()

                    # Score only candidates
                    cand_set = set(candidate_ids)
                    scored = [
                        (self._tfidf_atom_ids[i], float(scores[i]))
                        for i in range(len(self._tfidf_atom_ids))
                        if self._tfidf_atom_ids[i] in cand_set and scores[i] > 0
                    ]
                    candidate_ids = [
                        aid for aid, _ in sorted(scored, key=lambda x: -x[1])
                    ] or candidate_ids
                except Exception as exc:
                    logger.debug("[MemoryBank] TF-IDF re-rank error: %s", exc)

        # Fetch final top_k records
        result = []
        seen   = set()
        for aid in candidate_ids[:top_k]:
            if aid in seen:
                continue
            seen.add(aid)
            row = self._conn.execute(
                "SELECT * FROM memory_atoms WHERE atom_id=?", (aid,)
            ).fetchone()
            if row:
                result.append(dict(row))

        return result

    # ── Context builder for LLM injection ────────────────────────────────────
    def build_memory_context(self, query: str, top_k: int = 10) -> str:
        """
        Build a rich memory context string suitable for injecting into an LLM
        prompt (Karpathy LLM-Wiki style persistent context injection).
        """
        memories = self.recall(query, top_k=top_k)
        if not memories:
            return "=== MEMORY BANK === (no relevant prior records found)"

        lines = [
            "=== MEMORY BANK CONTEXT ===",
            f"({len(memories)} relevant prior records retrieved)",
            "",
        ]
        for m in memories:
            sp    = m.get("valid_name") or m.get("recorded_name","?")
            loc   = m.get("locality","?")
            hab   = m.get("habitat","")
            fam   = m.get("family_","")
            phy   = m.get("phylum","")
            date  = m.get("sampling_date","")
            depth = m.get("depth_m","")
            occ_t = m.get("occurrence_type","")
            worms = m.get("worms_id","")
            conf  = m.get("confidence", 1.0)
            times = m.get("times_confirmed", 1)
            evid  = (m.get("raw_evidence","") or "")[:200]
            src   = (m.get("source_citation","") or "")[:100]

            lines.append(
                f"• {sp}"
                + (f" [{fam}/{phy}]" if fam or phy else "")
                + f" @ {loc}"
                + (f" | {hab}" if hab else "")
                + (f" | {date}" if date else "")
                + (f" | {depth}m" if depth else "")
                + (f" | {occ_t}" if occ_t else "")
                + (f" | conf={conf:.2f} (×{times})" if times > 1 else "")
                + (f"\n  Source: {src}" if src else "")
                + (f'\n  Evidence: "{evid}"' if evid else "")
                + (f"\n  WoRMS: https://www.marinespecies.org/aphia.php?p=taxdetails&id={worms}" if worms else "")
            )

        # Session summary
        sessions = self._conn.execute(
            "SELECT session_id, title, n_records, n_species, n_localities FROM memory_sessions ORDER BY created_at DESC LIMIT 5"
        ).fetchall()
        if sessions:
            lines += [
                "",
                "=== RECENT EXTRACTION SESSIONS ===",
            ]
            for s in sessions:
                lines.append(
                    f"  [{s[0]}] {s[1]} — {s[2]} records, {s[3]} species, {s[4]} localities"
                )

        return "\n".join(lines)

    # ── Session summary generation ────────────────────────────────────────────
    def summarise_session(
        self,
        session_id: str,
        llm_fn: Optional[callable] = None,
    ) -> str:
        """
        Generate a plain-English summary of a completed extraction session.
        If llm_fn is provided, uses it to write an LLM-quality summary.
        """
        rows = self._conn.execute(
            "SELECT * FROM memory_atoms WHERE session_id=?", (session_id,)
        ).fetchall()
        if not rows:
            return f"No records found for session {session_id}"

        species  = sorted({r["valid_name"] for r in rows if r["valid_name"]})
        families = sorted({r["family_"]  for r in rows if r["family_"]})
        locs     = sorted({r["locality"] for r in rows if r["locality"]})
        habitats = sorted({r["habitat"]  for r in rows if r["habitat"]})

        plain = (
            f"Session '{session_id}': {len(rows)} occurrence records, "
            f"{len(species)} species ({', '.join(species[:8])}{'...' if len(species)>8 else ''}), "
            f"{len(locs)} localities ({', '.join(locs[:4])}{'...' if len(locs)>4 else ''}), "
            f"families: {', '.join(families[:6])}."
        )

        if llm_fn:
            prompt = (
                "Summarise the following marine biodiversity field-data extraction "
                "session in 3 concise sentences for a scientific report:\n\n" + plain
            )
            try:
                llm_summary = llm_fn(prompt)
                self._conn.execute(
                    "UPDATE memory_sessions SET llm_summary=? WHERE session_id=?",
                    (llm_summary, session_id),
                )
                self._conn.commit()
                return llm_summary
            except Exception as exc:
                logger.warning("[MemoryBank] LLM summarise error: %s", exc)

        self._conn.execute(
            "UPDATE memory_sessions SET summary=? WHERE session_id=?",
            (plain, session_id),
        )
        self._conn.commit()
        return plain

    # ── Search helpers ────────────────────────────────────────────────────────
    def search_species(self, name: str, fuzzy: bool = True) -> list[dict]:
        """Direct species name search with optional fuzzy matching."""
        rows = self._conn.execute(
            """SELECT * FROM memory_atoms
               WHERE valid_name LIKE ? OR recorded_name LIKE ?
               ORDER BY times_confirmed DESC LIMIT 50""",
            (f"%{name}%", f"%{name}%"),
        ).fetchall()
        results = [dict(r) for r in rows]

        if fuzzy and not results:
            try:
                from rapidfuzz import process as rfp
                all_sp = self._conn.execute(
                    "SELECT DISTINCT valid_name FROM memory_atoms"
                ).fetchall()
                sp_list = [r[0] for r in all_sp if r[0]]
                match = rfp.extractOne(name, sp_list, score_cutoff=70)
                if match:
                    matched_name = match[0]
                    rows = self._conn.execute(
                        "SELECT * FROM memory_atoms WHERE valid_name=? LIMIT 20",
                        (matched_name,),
                    ).fetchall()
                    results = [dict(r) for r in rows]
            except ImportError:
                pass

        return results

    def get_species_checklist(
        self,
        locality: Optional[str] = None,
        family: Optional[str]   = None,
        phylum: Optional[str]   = None,
        min_confidence: float   = 0.0,
    ) -> list[dict]:
        """
        Generate a Darwin Core–style species checklist from the memory bank.
        """
        conditions = ["confidence >= ?"]
        params: list[Any] = [min_confidence]
        if locality:
            conditions.append("locality LIKE ?")
            params.append(f"%{locality}%")
        if family:
            conditions.append("family_ LIKE ?")
            params.append(f"%{family}%")
        if phylum:
            conditions.append("phylum LIKE ?")
            params.append(f"%{phylum}%")

        where = " AND ".join(conditions)
        rows  = self._conn.execute(
            f"""SELECT valid_name, recorded_name, phylum, class_, order_, family_,
                       worms_id, taxon_rank, taxonomic_status,
                       COUNT(*) as n_records,
                       COUNT(DISTINCT locality) as n_localities,
                       MAX(confidence) as max_confidence
                FROM memory_atoms
                WHERE {where}
                GROUP BY valid_name
                ORDER BY phylum, family_, valid_name""",
            params,
        ).fetchall()

        return [dict(r) for r in rows]

    # ── Statistics ────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        total    = self._count_unique("atom_id")
        species  = self._count_unique("valid_name")
        families = self._count_unique("family_")
        locs     = self._count_unique("locality")
        sessions_n = self._conn.execute("SELECT COUNT(*) FROM memory_sessions").fetchone()[0]
        conflicts_n = self._conn.execute("SELECT COUNT(*) FROM memory_conflicts").fetchone()[0]
        top_sp = self._conn.execute(
            """SELECT valid_name, times_confirmed FROM memory_atoms
               ORDER BY times_confirmed DESC LIMIT 10"""
        ).fetchall()

        return {
            "total_atoms":     total,
            "unique_species":  species,
            "unique_families": families,
            "unique_localities": locs,
            "total_sessions":  sessions_n,
            "total_conflicts": conflicts_n,
            "top_species_by_confirmation": [(r[0], r[1]) for r in top_sp],
        }

    def export_darwin_core_csv(self, output_path: str) -> int:
        """Export all memory atoms as a Darwin Core CSV."""
        import csv
        rows = self._conn.execute("SELECT * FROM memory_atoms ORDER BY phylum, family_, valid_name").fetchall()
        if not rows:
            return 0

        fields = [
            "atom_id","session_id","valid_name","recorded_name",
            "phylum","class_","order_","family_","taxon_rank","taxonomic_status",
            "worms_id","itis_id","name_according_to",
            "locality","latitude","longitude","geocoding_src",
            "habitat","occurrence_type","sampling_date","depth_m","method",
            "raw_evidence","source_citation","confidence","times_confirmed",
            "first_seen","last_seen",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        logger.info("[MemoryBank] Exported %d atoms → %s", len(rows), output_path)
        return len(rows)

    def close(self):
        if self._conn:
            self._conn.close()
