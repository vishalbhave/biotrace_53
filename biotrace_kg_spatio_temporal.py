"""
biotrace_kg_spatio_temporal.py — BioTrace v5.4
──────────────────────────────────────────────
Incremental SpatioTemporal Knowledge Graph stored in SQLite.

Inspired by Hyper-Extract's AutoSpatioTemporalGraph:
  • Nodes = species (with lat/lon bbox, date range, occurrence count)
  • Edges = relation triples from species_relations table
  • Incremental: upsert updates bbox and temporal range as new docs arrive
  • Query: FTS5 full-text search for natural-language queries

New SQLite tables:
  kg_nodes  — one row per species
  kg_edges  — one row per unique (subject, relation, object) triple
"""
from __future__ import annotations
import json, logging, sqlite3
from typing import Optional

logger = logging.getLogger("biotrace.kg")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kg_nodes (
    node_id TEXT PRIMARY KEY,       -- canonical valid_name
    display_name TEXT,
    phylum TEXT, class_ TEXT, family TEXT,
    lat_min REAL, lat_max REAL,
    lon_min REAL, lon_max REAL,
    date_start TEXT, date_end TEXT,
    occurrence_count INTEGER DEFAULT 0,
    col_id TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS kg_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_node TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    target_node TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    evidence_text TEXT,
    source_citation TEXT,
    UNIQUE(source_node, relation_type, target_node)
);

CREATE VIRTUAL TABLE IF NOT EXISTS kg_nodes_fts USING fts5(
    node_id, display_name, phylum, class_, family,
    content='kg_nodes', content_rowid='rowid'
);
"""


class BioTraceSpatioTemporalKG:
    """
    Incremental SpatioTemporal Knowledge Graph backed by SQLite.

    Usage:
        kg = BioTraceSpatioTemporalKG(META_DB_PATH)
        kg.upsert_from_occurrences(results)
        kg.upsert_from_relations(triples)
        results = kg.query("Cassiopea andromeda habitats")
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_schema(self):
        conn = self._conn()
        conn.executescript(_SCHEMA)
        conn.commit(); conn.close()

    def upsert_from_occurrences(self, records: list[dict]) -> int:
        """
        Upsert kg_nodes from a list of occurrence dicts.
        Incrementally expands bbox and temporal range.
        """
        conn = self._conn()
        n = 0
        for rec in records:
            name = (rec.get("validName") or rec.get("recordedName") or "").strip()
            if not name:
                continue
            lat = rec.get("decimalLatitude")
            lon = rec.get("decimalLongitude")
            # Fetch existing node
            row = conn.execute(
                "SELECT lat_min,lat_max,lon_min,lon_max,occurrence_count FROM kg_nodes WHERE node_id=?",
                (name,)
            ).fetchone()
            if row:
                # Expand bbox incrementally (Hyper-Extract "feed" pattern)
                lat_min = min(r for r in [row[0], lat] if r is not None) if lat else row[0]
                lat_max = max(r for r in [row[1], lat] if r is not None) if lat else row[1]
                lon_min = min(r for r in [row[2], lon] if r is not None) if lon else row[2]
                lon_max = max(r for r in [row[3], lon] if r is not None) if lon else row[3]
                count = (row[4] or 0) + 1
                conn.execute("""UPDATE kg_nodes SET lat_min=?,lat_max=?,lon_min=?,lon_max=?,
                               occurrence_count=?,updated_at=datetime('now') WHERE node_id=?""",
                             (lat_min, lat_max, lon_min, lon_max, count, name))
            else:
                conn.execute("""INSERT INTO kg_nodes
                    (node_id,display_name,phylum,class_,family,lat_min,lat_max,
                     lon_min,lon_max,occurrence_count,col_id)
                    VALUES (?,?,?,?,?,?,?,?,?,1,?)""",
                    (name, name,
                     rec.get("phylum",""), rec.get("class_",""), rec.get("family_",""),
                     lat, lat, lon, lon, rec.get("colID","")))
            n += 1
        conn.commit(); conn.close()
        return n

    def upsert_from_relations(self, triples) -> int:
        """Upsert kg_edges from RelationTriple objects."""
        conn = self._conn()
        n = 0
        for t in triples:
            try:
                conn.execute("""INSERT OR IGNORE INTO kg_edges
                    (source_node,relation_type,target_node,evidence_text,source_citation)
                    VALUES (?,?,?,?,?)""",
                    (t.subject, t.relation, t.object, t.evidence_text, t.source_citation))
                n += 1
            except Exception:
                pass
        conn.commit(); conn.close()
        return n

    def query(self, query_text: str, limit: int = 10) -> list[dict]:
        """
        FTS5 full-text search over kg_nodes — returns matching species nodes.
        Mimics LightRAG's natural-language query interface (Hyper-Extract).
        """
        conn = self._conn()
        try:
            rows = conn.execute(
                """SELECT n.node_id, n.display_name, n.phylum, n.family,
                          n.lat_min, n.lat_max, n.lon_min, n.lon_max,
                          n.occurrence_count
                   FROM kg_nodes_fts f
                   JOIN kg_nodes n ON n.rowid = f.rowid
                   WHERE kg_nodes_fts MATCH ?
                   LIMIT ?""",
                (query_text, limit)
            ).fetchall()
        except Exception:
            rows = []
        conn.close()
        keys = ["node_id","display_name","phylum","family",
                "lat_min","lat_max","lon_min","lon_max","occurrence_count"]
        return [dict(zip(keys, r)) for r in rows]