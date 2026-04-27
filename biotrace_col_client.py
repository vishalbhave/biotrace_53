"""
biotrace_col_client.py  —  BioTrace v5.4
────────────────────────────────────────────────────────────────────────────
Catalogue of Life REST API client.
Mirrors COL Agent's COLClient + Pydantic model pattern.

BUG FIX (19-04-2026):
  ERROR: [COL] cache write: Error binding parameter 3 — type 'dict' is not supported

  ROOT CAUSE:
    The COL API /nameusage/search endpoint returns the species name as a NESTED
    OBJECT, not a plain string:

      usage["name"] = {
          "scientificName": "Cassiopea andromeda",
          "authorship":     "(Forsskål, 1775)",
          "label":          "Cassiopea andromeda (Forsskål, 1775)",
          ...
      }

    The original code did `accepted.get("name", "")` which returns the whole
    dict, then tried to INSERT it into a TEXT column — causing the SQLite error.

  FIX (two layers):
    1. _parse_col_response() now extracts scientificName from the nested object.
    2. _cache_taxon() coerces every field to str() as a safety net.

COL API endpoint:
  GET https://api.catalogueoflife.org/nameusage/search?q={name}&limit=5

New SQLite table:
  col_taxonomy_cache  — see _ensure_col_table() below.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("biotrace.col_client")

_COL_API_BASE = "https://api.catalogueoflife.org"
_RATE_LIMIT_S = 1.0   # 1 req/sec — COL is rate-limited


# ─────────────────────────────────────────────────────────────────────────────
#  Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class COLTaxon:
    """Typed result from a COL nameusage lookup."""
    col_id:       str       = ""
    query_name:   str       = ""
    accepted_name:str       = ""
    status:       str       = ""   # "accepted" | "synonym" | "bare name"
    kingdom:      str       = ""
    phylum:       str       = ""
    class_:       str       = ""
    order_:       str       = ""
    family:       str       = ""
    genus:        str       = ""
    synonyms:     list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  Response parser  (BUG FIX HERE)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_scientific_name(name_field) -> str:
    """
    Safely extract a plain string scientific name from a COL name field.

    The COL API returns 'name' as either:
      • A dict:   {"scientificName": "Cassiopea andromeda", "authorship": "...", ...}
      • A string: "Cassiopea andromeda"  (rare, earlier API versions)
      • None / missing

    BUG FIX: the original code returned the entire dict, causing SQLite
    'type dict is not supported' errors on INSERT.
    """
    if name_field is None:
        return ""
    if isinstance(name_field, str):
        return name_field.strip()
    if isinstance(name_field, dict):
        # Prefer scientificName; fall back to label, then any string value
        for key in ("scientificName", "label", "uninomial", "name"):
            val = name_field.get(key, "")
            if isinstance(val, str) and val.strip():
                return val.strip()
    # Last resort: stringify
    return str(name_field).strip()


def _parse_col_response(data: dict, query_name: str) -> Optional[COLTaxon]:
    """
    Parse a COL /nameusage/search JSON response into a COLTaxon.
    Handles the nested name object structure correctly.
    """
    results = data.get("result", [])
    if not results:
        return None

    # Take first result (highest-ranked match)
    r0    = results[0]
    usage = r0.get("usage", r0)   # some endpoints nest under 'usage'

    if not isinstance(usage, dict):
        return None

    # ── Classification hierarchy ───────────────────────────────────────────
    classification = usage.get("classification", [])
    ranks: dict[str, str] = {}
    for c in classification:
        if isinstance(c, dict):
            rank = str(c.get("rank", "")).lower()
            name = str(c.get("name", ""))
            if rank and name:
                ranks[rank] = name

    # ── Accepted taxon ─────────────────────────────────────────────────────
    # COL returns an "accepted" sub-object when the matched name is a synonym.
    # If the match is itself accepted, "accepted" may be absent — fall back to usage.
    accepted = usage.get("accepted", usage)
    if not isinstance(accepted, dict):
        accepted = usage

    # ── BUG FIX: extract plain string from nested name object ──────────────
    name_field     = accepted.get("name", usage.get("name", ""))
    accepted_name  = _extract_scientific_name(name_field)

    # status: string in the COL API ("accepted", "synonym", etc.)
    status = usage.get("status", "")
    if isinstance(status, dict):
        # Edge case: some API versions return status as an object
        status = status.get("name", status.get("label", str(status)))
    status = str(status).strip()

    taxon = COLTaxon(
        col_id        = str(usage.get("id", r0.get("id", ""))).strip(),
        query_name    = query_name,
        accepted_name = accepted_name,
        status        = status,
        kingdom       = ranks.get("kingdom", ""),
        phylum        = ranks.get("phylum", ""),
        class_        = ranks.get("class", ""),
        order_        = ranks.get("order", ""),
        family        = ranks.get("family", ""),
        genus         = ranks.get("genus", ""),
    )

    # Collect synonyms from remaining results
    taxon.synonyms = []
    for r in results[1:6]:
        if not isinstance(r, dict):
            continue
        u = r.get("usage", r)
        if not isinstance(u, dict):
            continue
        syn_status = str(u.get("status", "")).lower()
        if "synonym" in syn_status:
            syn_name = _extract_scientific_name(u.get("name", ""))
            if syn_name:
                taxon.synonyms.append(syn_name)

    return taxon


# ─────────────────────────────────────────────────────────────────────────────
#  SQLite cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_col_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS col_taxonomy_cache (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            query_name   TEXT NOT NULL,
            col_id       TEXT,
            accepted_name TEXT,
            status       TEXT,
            kingdom      TEXT,
            phylum       TEXT,
            class_       TEXT,
            order_       TEXT,
            family       TEXT,
            genus        TEXT,
            synonyms_json TEXT,
            fetched_at   TEXT DEFAULT (datetime('now')),
            UNIQUE(query_name)
        )
    """)
    conn.commit()


def _cache_taxon(taxon: COLTaxon, db_path: str) -> None:
    """
    Write a COLTaxon to the local cache.

    BUG FIX: every field is coerced to str() before binding.
    Previously, accepted_name could be a dict (from the nested name object),
    causing 'type dict is not supported' SQLite errors.
    """
    try:
        conn = sqlite3.connect(db_path)
        _ensure_col_table(conn)
        conn.execute(
            """INSERT OR REPLACE INTO col_taxonomy_cache
               (query_name, col_id, accepted_name, status,
                kingdom, phylum, class_, order_, family, genus, synonyms_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(taxon.query_name   or ""),
                str(taxon.col_id       or ""),
                str(taxon.accepted_name or ""),  # BUG FIX: was a dict from nested name obj
                str(taxon.status       or ""),
                str(taxon.kingdom      or ""),
                str(taxon.phylum       or ""),
                str(taxon.class_       or ""),
                str(taxon.order_       or ""),
                str(taxon.family       or ""),
                str(taxon.genus        or ""),
                json.dumps(taxon.synonyms or []),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.error("[COL] cache write: %s", exc)


def _row_to_taxon(row: tuple, cols: list[str]) -> COLTaxon:
    """Reconstruct a COLTaxon from a sqlite3 row."""
    d = dict(zip(cols, row))
    return COLTaxon(
        col_id        = d.get("col_id",        ""),
        query_name    = d.get("query_name",    ""),
        accepted_name = d.get("accepted_name", ""),
        status        = d.get("status",        ""),
        kingdom       = d.get("kingdom",       ""),
        phylum        = d.get("phylum",        ""),
        class_        = d.get("class_",        ""),
        order_        = d.get("order_",        ""),
        family        = d.get("family",        ""),
        genus         = d.get("genus",         ""),
        synonyms      = json.loads(d.get("synonyms_json", "[]") or "[]"),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def lookup_col(name: str, db_path: str) -> Optional[COLTaxon]:
    """
    Look up a species name in the Catalogue of Life.

    Checks SQLite cache first; falls back to COL API if not cached.
    Rate-limited to 1 req/sec (COL policy).

    Parameters
    ----------
    name    : scientific name string (binomial preferred)
    db_path : path to the main SQLite DB (META_DB_PATH)

    Returns
    -------
    COLTaxon or None
    """
    name = name.strip()
    if not name:
        return None

    conn = sqlite3.connect(db_path)
    _ensure_col_table(conn)

    # ── Cache hit ──────────────────────────────────────────────────────────
    row = conn.execute(
        "SELECT * FROM col_taxonomy_cache WHERE query_name=?", (name,)
    ).fetchone()
    if row:
        cols = [d[0] for d in conn.execute("PRAGMA table_info(col_taxonomy_cache)").fetchall()]
        conn.close()
        return _row_to_taxon(row, cols)
    conn.close()

    # ── API call ───────────────────────────────────────────────────────────
    try:
        time.sleep(_RATE_LIMIT_S)
        url = (
            f"{_COL_API_BASE}/nameusage/search"
            f"?q={urllib.parse.quote(name)}&limit=5"
        )
        req = urllib.request.Request(
            url,
            headers={
                "Accept":     "application/json",
                "User-Agent": "BioTrace/5.4 (biotrace_col_client)",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        logger.warning("[COL] API error for '%s': %s", name, exc)
        return None

    taxon = _parse_col_response(data, name)
    if taxon:
        _cache_taxon(taxon, db_path)
    return taxon


def enrich_records_with_col(
    records:  list[dict],
    db_path:  str,
) -> list[dict]:
    """
    Enrich a list of occurrence dicts with COL taxonomy.

    Fills these fields if currently blank:
      validName, phylum, class_, order_, family_, colID

    Parameters
    ----------
    records : occurrence dicts (from extract_occurrences pipeline)
    db_path : path to META_DB_PATH SQLite database

    Returns
    -------
    The same list with fields filled in-place.
    """
    for rec in records:
        if not isinstance(rec, dict):
            continue

        name = (
            rec.get("validName")
            or rec.get("recordedName")
            or rec.get("Recorded Name", "")
        ).strip()

        if not name:
            continue

        taxon = lookup_col(name, db_path)
        if not taxon:
            continue

        # Fill accepted name
        if taxon.accepted_name and not rec.get("validName"):
            rec["validName"] = taxon.accepted_name

        # Fill taxonomy hierarchy (only if blank)
        for rec_key, taxon_val in [
            ("phylum",  taxon.phylum),
            ("class_",  taxon.class_),
            ("order_",  taxon.order_),
            ("family_", taxon.family),
        ]:
            if taxon_val and not rec.get(rec_key):
                rec[rec_key] = taxon_val

        # Store COL ID
        if taxon.col_id:
            rec["colID"] = taxon.col_id

    return records


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    # ── Unit test _parse_col_response with mock data ───────────────────────
    MOCK = {
        "result": [
            {
                "id": "TEST001",
                "usage": {
                    "id": "TEST001",
                    "name": {
                        "id": "N1",
                        "scientificName": "Cassiopea andromeda",
                        "authorship": "(Forsskål, 1775)",
                        "label": "Cassiopea andromeda (Forsskål, 1775)",
                    },
                    "status": "accepted",
                    "accepted": {
                        "id": "TEST001",
                        "name": {
                            "scientificName": "Cassiopea andromeda",
                        },
                    },
                    "classification": [
                        {"rank": "KINGDOM", "name": "Animalia"},
                        {"rank": "PHYLUM",  "name": "Cnidaria"},
                        {"rank": "CLASS",   "name": "Scyphozoa"},
                        {"rank": "ORDER",   "name": "Rhizostomeae"},
                        {"rank": "FAMILY",  "name": "Cassiopeidae"},
                    ],
                },
            }
        ]
    }

    taxon = _parse_col_response(MOCK, "Cassiopea andromeda")
    assert taxon is not None, "parse returned None"
    assert isinstance(taxon.accepted_name, str), \
        f"accepted_name must be str, got {type(taxon.accepted_name).__name__}: {taxon.accepted_name}"
    assert taxon.accepted_name == "Cassiopea andromeda", \
        f"Wrong accepted_name: {taxon.accepted_name}"
    assert taxon.phylum == "Cnidaria",  f"Wrong phylum: {taxon.phylum}"
    assert taxon.class_  == "Scyphozoa", f"Wrong class: {taxon.class_}"
    print("✅ _parse_col_response: accepted_name is a plain string — dict bug FIXED")

    # ── Unit test _cache_taxon with an in-memory DB ────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        tmp_db = tf.name

    try:
        _cache_taxon(taxon, tmp_db)
        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT accepted_name FROM col_taxonomy_cache WHERE query_name=?",
            ("Cassiopea andromeda",)
        ).fetchone()
        conn.close()
        assert row is not None, "Row not written to cache"
        assert row[0] == "Cassiopea andromeda", f"Cache row wrong: {row[0]}"
        print("✅ _cache_taxon: dict-in-TEXT-column bug FIXED — string written cleanly")
    finally:
        os.unlink(tmp_db)

    print("\n✅ All self-tests passed.")