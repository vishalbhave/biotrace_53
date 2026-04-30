"""
biotrace_hitl_geocoding.py  —  BioTrace v5.6  (Enriched-Geocoder + Auto-Broadcast edition)
═══════════════════════════════════════════════════════════════════════════════════════════
Human-in-the-Loop geocoding + full database sync with automatic cross-tab refresh.

CHANGES vs v5.5
────────────────
1. Enriched Nominatim geocoding
   • Replaced single _nominatim_lookup() with EnrichedGeocoder (biotrace_geocoder_enriched.py):
     – 6 waterfall strategies: exact → India suffix → state hint → unrestricted
       → Photon (no key) → OpenCage (optional key)
     – SQLite cache honoured across sessions; repeated HITL lookups are instant
     – Strict rate-limiting per backend (1.15 s / Nominatim, 0.6 s / Photon)
   • Enriched locality display: shows detected state, strategy used, confidence score.
   • Cache stats sidebar widget ("🗺️ Geocoder Cache Stats").

2. Cross-tab auto-propagation
   • After every HITL commit (accept / manual edit / delete) _broadcast_update()
     sets Streamlit session_state flags consumed by other tabs:
       – "occ_map_stale"     → Occurrence Map tab re-fetches from DB on next render
       – "data_table_stale"  → Data tab refreshes dataframe
       – "kg_dirty"          → KG tab shows rebuild banner
       – "wiki_stale"        → Wiki tab reloads index
   • No page reload required for siblings — flags are checked by each tab's
     render function; tabs that use st.cache_data get their cache cleared.

3. Wiki-relation filter nodes
   • When locality or species edits propagate to the KG the affected
     WikiRelation nodes are updated so the KG Filter sidebar in other tabs
     reflects the change without a manual rebuild.

Usage (unchanged from v5.5):
    from biotrace_hitl_geocoding import render_hitl_geocoding_tab
    with tabs[N]:
        render_hitl_geocoding_tab(
            META_DB_PATH, KG_DB_PATH, MB_DB_PATH, WIKI_ROOT,
            kg_st_db_path=KG_ST_DB_PATH,
        )
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from typing import Optional

import pandas as pd

logger = logging.getLogger("biotrace.hitl_geocoding")

# ─── Enriched geocoder ────────────────────────────────────────────────────────
try:
    from biotrace_geocoder_enriched import get_geocoder, _is_in_india
    _ENRICHED_GEOCODER_OK = True
except ImportError:
    _ENRICHED_GEOCODER_OK = False
    logger.warning("[HITL] biotrace_geocoder_enriched.py not found — using basic Nominatim")

# ─── Fallback: plain geopy Nominatim ─────────────────────────────────────────
_geopy_available = False
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
    _geopy_available = True
except ImportError:
    pass

_fallback_geocoder = None


def _get_fallback_geocoder():
    global _fallback_geocoder
    if not _geopy_available:
        return None
    if _fallback_geocoder is None:
        _fallback_geocoder = Nominatim(user_agent="BioTrace_HITL_v56_fallback")
    return _fallback_geocoder


def _plain_nominatim_lookup(locality: str) -> Optional[dict]:
    gc = _get_fallback_geocoder()
    if gc is None:
        return None
    try:
        time.sleep(1.15)
        result = gc.geocode(locality + ", India", exactly_one=True, timeout=12,
                            country_codes="in")
        if result:
            return {"lat": result.latitude, "lon": result.longitude,
                    "display_name": result.address, "strategy": "nominatim_plain",
                    "confidence": 0.5, "cached": False}
    except Exception as exc:
        logger.warning("[HITL/fallback] Nominatim: %s", exc)
    return None


def _lookup_locality(locality: str,
                     geocoder_cache_db: str = "biodiversity_data/geocoder_cache.db"
                     ) -> Optional[dict]:
    """
    Main geocoding entry point for HITL.
    Uses EnrichedGeocoder when available, falls back to plain Nominatim.
    """
    if _ENRICHED_GEOCODER_OK:
        try:
            gc = get_geocoder(cache_db=geocoder_cache_db)
            return gc.geocode(locality)
        except Exception as exc:
            logger.warning("[HITL] EnrichedGeocoder failed: %s — using fallback", exc)
    return _plain_nominatim_lookup(locality)


# ─── Cross-tab broadcast ──────────────────────────────────────────────────────

def _broadcast_update(action: str = "edit"):
    """
    Set session_state staleness flags consumed by other BioTrace tabs.

    Flags set:
      occ_map_stale    → Occurrence Map tab re-queries DB
      data_table_stale → Data tab refreshes DataFrame
      kg_dirty         → KG tab shows rebuild banner
      wiki_stale       → Wiki tab reloads article index
      hitl_last_action → Human-readable summary for status bars
    """
    try:
        import streamlit as st
        st.session_state["occ_map_stale"]    = True
        st.session_state["data_table_stale"] = True
        st.session_state["kg_dirty"]         = True
        st.session_state["wiki_stale"]       = True
        st.session_state["hitl_last_action"] = action
        st.session_state["hitl_kg_dirty"]    = True
    except Exception:
        pass  # no Streamlit context (unit-test mode)


# ─── SQLite helpers ───────────────────────────────────────────────────────────

def _resolve_table(conn: sqlite3.Connection) -> str:
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    for t in ("occurrences_v4", "occurrences"):
        if t in names:
            return t
    raise sqlite3.OperationalError("No occurrence table found.")


def _load_missing(db_path: str) -> list[dict]:
    try:
        conn = sqlite3.connect(db_path)
        table = _resolve_table(conn)
        rows = conn.execute(f"""
            SELECT id, verbatimLocality, validName, recordedName,
                   sourceCitation, geocodingSource, occurrenceType
            FROM {table}
            WHERE (decimalLatitude IS NULL OR decimalLongitude IS NULL)
              AND verbatimLocality IS NOT NULL AND verbatimLocality != ''
              AND (validationStatus IS NULL OR validationStatus != 'rejected')
            ORDER BY id""").fetchall()
        conn.close()
        return [{"id": r[0], "verbatimLocality": r[1], "validName": r[2] or r[3] or "Unknown",
                 "sourceCitation": r[4] or "", "geocodingSource": r[5] or "",
                 "occurrenceType": r[6] or ""} for r in rows]
    except Exception as exc:
        logger.error("[HITL] load_missing: %s", exc)
        return []


def _load_all_records(db_path: str) -> list[dict]:
    try:
        conn = sqlite3.connect(db_path)
        table = _resolve_table(conn)
        rows = conn.execute(f"""
            SELECT id, verbatimLocality, validName, recordedName,
                   decimalLatitude, decimalLongitude, geocodingSource,
                   occurrenceType, sourceCitation
            FROM {table} ORDER BY id""").fetchall()
        conn.close()
        return [{"id": r[0], "verbatimLocality": r[1], "validName": r[2] or r[3] or "",
                 "lat": r[4], "lon": r[5], "geocodingSource": r[6] or "",
                 "occurrenceType": r[7] or "", "sourceCitation": r[8] or ""} for r in rows]
    except Exception as exc:
        logger.error("[HITL] load_all: %s", exc)
        return []


def _fetch_record(db_path: str, row_id: int) -> Optional[dict]:
    try:
        conn = sqlite3.connect(db_path)
        table = _resolve_table(conn)
        row = conn.execute(f"SELECT * FROM {table} WHERE id=?", (row_id,)).fetchone()
        if row is None:
            conn.close()
            return None
        cols = [d[0] for d in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        conn.close()
        return dict(zip(cols, row))
    except Exception as exc:
        logger.error("[HITL] fetch_record: %s", exc)
        return None


# ─── Store 1: SQLite ──────────────────────────────────────────────────────────

def _write_coords_sqlite(db_path: str, row_id: int,
                         lat: float, lon: float, source: str) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        table = _resolve_table(conn)
        conn.execute(
            f"UPDATE {table} SET decimalLatitude=?,decimalLongitude=?,geocodingSource=? WHERE id=?",
            (round(lat, 6), round(lon, 6), source, row_id))
        conn.commit(); conn.close()
        return True
    except Exception as exc:
        logger.error("[HITL/sqlite] coords: %s", exc)
        return False


def _write_locality_sqlite(db_path: str, row_id: int, new_locality: str) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        table = _resolve_table(conn)
        conn.execute(f"UPDATE {table} SET verbatimLocality=? WHERE id=?",
                     (new_locality, row_id))
        conn.commit(); conn.close()
        return True
    except Exception as exc:
        logger.error("[HITL/sqlite] locality: %s", exc)
        return False


def _delete_sqlite(db_path: str, row_id: int) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        table = _resolve_table(conn)
        conn.execute(f"DELETE FROM {table} WHERE id=?", (row_id,))
        conn.commit(); conn.close()
        return True
    except Exception as exc:
        logger.error("[HITL/sqlite] delete: %s", exc)
        return False


# ─── Store 2: KG ─────────────────────────────────────────────────────────────

def _kg_update_locality_coords(kg_db_path: str, locality: str,
                                lat: float, lon: float) -> bool:
    try:
        from biotrace_knowledge_graph import BioTraceKnowledgeGraph, _node_id, _now
        kg = BioTraceKnowledgeGraph(kg_db_path)
        loc_nid = _node_id("Locality", locality)
        if not kg._G.has_node(loc_nid):
            kg._upsert_node("Locality", locality, {
                "decimalLatitude": lat, "decimalLongitude": lon,
                "geocodingSource": "HITL"})
        else:
            existing = json.loads(
                kg._conn.execute("SELECT properties FROM kg_nodes WHERE node_id=?",
                                 (loc_nid,)).fetchone()[0] or "{}")
            existing.update({"decimalLatitude": round(lat, 6),
                             "decimalLongitude": round(lon, 6),
                             "geocodingSource": "HITL"})
            kg._conn.execute(
                "UPDATE kg_nodes SET properties=?, updated_at=? WHERE node_id=?",
                (json.dumps(existing), _now(), loc_nid))
            kg._G.nodes[loc_nid].update({"decimalLatitude": round(lat, 6),
                                         "decimalLongitude": round(lon, 6)})
        kg._conn.commit(); kg.close()
        return True
    except Exception as exc:
        logger.warning("[HITL/KG] update coords: %s", exc)
        return False


def _kg_migrate_locality(kg_db_path: str, old_locality: str, new_locality: str,
                          lat: Optional[float] = None, lon: Optional[float] = None) -> bool:
    try:
        from biotrace_knowledge_graph import BioTraceKnowledgeGraph, _node_id, _now
        kg = BioTraceKnowledgeGraph(kg_db_path)
        old_nid = _node_id("Locality", old_locality)
        new_props = {"decimalLatitude": round(lat, 6) if lat else None,
                     "decimalLongitude": round(lon, 6) if lon else None,
                     "geocodingSource": "HITL_locality_edit"}
        new_nid = kg._upsert_node("Locality", new_locality, new_props)

        if kg._G.has_node(old_nid):
            for src, _, ed in list(kg._G.in_edges(old_nid, data=True)):
                rel  = ed.get("rel_type", "FOUND_AT")
                wt   = ed.get("weight", 1.0)
                props = {k: v for k, v in ed.items()
                         if k not in ("rel_type", "weight", "key")}
                kg._upsert_edge(src, rel, new_nid, weight=wt, props=props)
                kg._conn.execute(
                    "DELETE FROM kg_edges WHERE src_id=? AND rel_type=? AND tgt_id=?",
                    (src, rel, old_nid))
            for _, tgt, ed in list(kg._G.out_edges(old_nid, data=True)):
                rel  = ed.get("rel_type", "FOUND_AT")
                wt   = ed.get("weight", 1.0)
                props = {k: v for k, v in ed.items()
                         if k not in ("rel_type", "weight", "key")}
                kg._upsert_edge(new_nid, rel, tgt, weight=wt, props=props)
                kg._conn.execute(
                    "DELETE FROM kg_edges WHERE src_id=? AND rel_type=? AND tgt_id=?",
                    (old_nid, rel, tgt))
            kg._conn.execute("DELETE FROM kg_nodes WHERE node_id=?", (old_nid,))
            kg._G.remove_node(old_nid)
        kg._conn.commit()
        try:
            kg.detect_communities()
        except Exception as ce:
            logger.warning("[HITL/KG] Community rebuild skipped: %s", ce)
        kg.close()
        return True
    except Exception as exc:
        logger.warning("[HITL/KG] migrate locality: %s", exc)
        return False


def _kg_delete_species(kg_db_path: str, valid_name: str) -> bool:
    try:
        from biotrace_knowledge_graph import BioTraceKnowledgeGraph, _node_id
        kg = BioTraceKnowledgeGraph(kg_db_path)
        sp_nid = _node_id("Species", valid_name)
        if kg._G.has_node(sp_nid):
            kg._conn.execute(
                "DELETE FROM kg_edges WHERE src_id=? OR tgt_id=?", (sp_nid, sp_nid))
            kg._conn.execute("DELETE FROM kg_nodes WHERE node_id=?", (sp_nid,))
            kg._G.remove_node(sp_nid)
            kg._conn.commit()
        try:
            kg.detect_communities()
        except Exception:
            pass
        kg.close()
        return True
    except Exception as exc:
        logger.warning("[HITL/KG] delete species: %s", exc)
        return False


def _kg_rebuild_communities(kg_db_path: str) -> bool:
    try:
        from biotrace_knowledge_graph import BioTraceKnowledgeGraph
        kg = BioTraceKnowledgeGraph(kg_db_path)
        comms = kg.detect_communities()
        kg.close()
        logger.info("[HITL/KG] Rebuilt %d communities.", len(comms))
        return True
    except Exception as exc:
        logger.warning("[HITL/KG] rebuild_communities: %s", exc)
        return False


# ─── Store 3: MemoryBank ──────────────────────────────────────────────────────

def _mb_atom_id(species: str, locality: str, source: str) -> str:
    raw = f"{species.lower().strip()}|{locality.lower().strip()}|{source.lower().strip()[:80]}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _mb_update_coords(mb_db_path: str, valid_name: str,
                      lat: float, lon: float) -> bool:
    try:
        conn = sqlite3.connect(mb_db_path)
        conn.execute(
            "UPDATE memory_atoms SET latitude=?, longitude=? WHERE valid_name=?",
            (round(lat, 6), round(lon, 6), valid_name))
        conn.commit(); conn.close()
        return True
    except Exception as exc:
        logger.warning("[HITL/MB] update coords: %s", exc)
        return False


def _mb_migrate_locality(mb_db_path: str, valid_name: str,
                          old_locality: str, new_locality: str,
                          lat: Optional[float] = None,
                          lon: Optional[float] = None) -> bool:
    try:
        conn = sqlite3.connect(mb_db_path)
        conn.row_factory = sqlite3.Row
        atoms = conn.execute(
            "SELECT * FROM memory_atoms WHERE valid_name=? AND locality=?",
            (valid_name, old_locality)).fetchall()
        if not atoms:
            conn.close(); return True
        cols = atoms[0].keys()
        for atom in atoms:
            row = dict(atom)
            old_aid = row["atom_id"]
            row["locality"] = new_locality
            if lat is not None: row["latitude"]  = round(lat, 6)
            if lon is not None: row["longitude"] = round(lon, 6)
            new_aid = _mb_atom_id(row["valid_name"], new_locality,
                                  row.get("source_citation", ""))
            row["atom_id"] = new_aid
            conn.execute("DELETE FROM memory_atoms WHERE atom_id=?", (old_aid,))
            try:
                conn.execute(
                    f"INSERT OR IGNORE INTO memory_atoms ({', '.join(cols)}) "
                    f"VALUES ({', '.join('?'*len(cols))})",
                    [row.get(c) for c in cols])
            except Exception as ie:
                logger.warning("[HITL/MB] re-insert: %s", ie)
        conn.commit(); conn.close()
        return True
    except Exception as exc:
        logger.warning("[HITL/MB] migrate locality: %s", exc)
        return False


def _mb_delete_species(mb_db_path: str, valid_name: str) -> bool:
    try:
        conn = sqlite3.connect(mb_db_path)
        conn.execute("DELETE FROM memory_atoms WHERE valid_name=?", (valid_name,))
        conn.commit(); conn.close()
        return True
    except Exception as exc:
        logger.warning("[HITL/MB] delete: %s", exc)
        return False


# ─── Store 4: Wiki ────────────────────────────────────────────────────────────

def _wiki_update_coords(wiki_root: str, locality: str,
                        lat: float, lon: float) -> bool:
    try:
        from Unified_Wiki_Module import UnifiedWiki
        import datetime as _dt
        wiki = UnifiedWiki(wiki_root)
        slug = wiki._slugify(locality)
        fp   = wiki.root / "locality" / f"{slug}.json"
        art  = wiki._read_json(fp)
        if art:
            art.update({"decimalLatitude": round(lat, 6),
                        "decimalLongitude": round(lon, 6),
                        "geocodingSource": "HITL",
                        "last_updated": _dt.datetime.now().isoformat()})
            wiki._save_json(fp, art)
        return True
    except Exception as exc:
        logger.warning("[HITL/Wiki] update coords: %s", exc)
        return False


def _wiki_migrate_locality(wiki_root: str, valid_name: str,
                            old_locality: str, new_locality: str,
                            lat: Optional[float] = None,
                            lon: Optional[float] = None,
                            occurrence: Optional[dict] = None) -> bool:
    try:
        from Unified_Wiki_Module import UnifiedWiki
        import datetime as _dt
        wiki = UnifiedWiki(wiki_root)

        # Clean old locality article
        old_fp = wiki.root / "locality" / f"{wiki._slugify(old_locality)}.json"
        old_art = wiki._read_json(old_fp)
        if old_art:
            checklist = [s for s in old_art.get("species_checklist", [])
                         if s != valid_name]
            old_art["species_checklist"] = checklist
            old_art["last_updated"] = _dt.datetime.now().isoformat()
            if checklist:
                wiki._save_json(old_fp, old_art)
            else:
                try: old_fp.unlink(missing_ok=True)
                except Exception: pass

        # Create/update new locality article
        occ_for_wiki = dict(occurrence or {})
        if lat is not None:
            occ_for_wiki.update({"decimalLatitude": lat, "decimalLongitude": lon,
                                  "verbatimLocality": new_locality})
        wiki.update_locality_article(new_locality, valid_name, occ_for_wiki, "")

        # Update species article
        sp_fp = wiki.root / "species" / f"{wiki._slugify(valid_name)}.json"
        sp_art = wiki._read_json(sp_fp)
        if sp_art:
            for occ_entry in sp_art.get("occurrences", []):
                if occ_entry.get("verbatimLocality") == old_locality:
                    occ_entry["verbatimLocality"] = new_locality
                    if lat is not None:
                        occ_entry.update({"decimalLatitude": lat, "decimalLongitude": lon})
            sp_art["last_updated"] = _dt.datetime.now().isoformat()
            wiki._save_json(sp_fp, sp_art)

        wiki._rebuild_index()
        return True
    except Exception as exc:
        logger.warning("[HITL/Wiki] migrate locality: %s", exc)
        return False


def _wiki_delete_species(wiki_root: str, valid_name: str, locality: str) -> bool:
    try:
        from Unified_Wiki_Module import UnifiedWiki
        wiki = UnifiedWiki(wiki_root)
        slug = wiki._slugify(locality)
        fp   = wiki.root / "locality" / f"{slug}.json"
        art  = wiki._read_json(fp)
        if art:
            art["species_checklist"] = [s for s in art.get("species_checklist", [])
                                         if s != valid_name]
            wiki._save_json(fp, art)
        sp_fp = wiki.root / "species" / f"{wiki._slugify(valid_name)}.json"
        try: sp_fp.unlink(missing_ok=True)
        except Exception: pass
        wiki._rebuild_index()
        return True
    except Exception as exc:
        logger.warning("[HITL/Wiki] delete: %s", exc)
        return False


# ─── Store 4b: Spatio-temporal KG ────────────────────────────────────────────

def _st_kg_update(st_kg_db_path: str, occurrence: dict) -> bool:
    if not st_kg_db_path: return False
    try:
        from biotrace_kg_spatio_temporal import BioTraceSpatioTemporalKG
        stkg = BioTraceSpatioTemporalKG(st_kg_db_path)
        stkg.upsert_from_occurrences([occurrence])
        return True
    except Exception as exc:
        logger.warning("[HITL/STKG] %s", exc)
        return False


def _re_verify_occurrence(occurrence: dict) -> dict:
    try:
        from biotrace_unified_verifier import BioTraceUnifiedVerifier
        verifier = BioTraceUnifiedVerifier()
        enriched = verifier.verify_and_enrich([occurrence])
        if enriched: return enriched[0]
    except Exception as exc:
        logger.debug("[HITL] re-verify skipped: %s", exc)
    return occurrence


# ─── Master sync dispatcher ───────────────────────────────────────────────────

def sync_all_stores(
    meta_db_path:  str,
    kg_db_path:    str,
    mb_db_path:    str,
    wiki_root:     str,
    row_id:        int,
    valid_name:    str,
    old_locality:  str,
    lat:           Optional[float],
    lon:           Optional[float],
    source:        str,
    *,
    delete:        bool = False,
    new_locality:  Optional[str] = None,
    st_kg_db_path: str = "",
    occurrence:    Optional[dict] = None,
) -> dict[str, bool | str]:
    """
    Atomic-ish sync of all stores + cross-tab broadcast.

    After every call (regardless of action type) _broadcast_update() sets
    session_state staleness flags so occurrence maps, data tables, and KG
    panels refresh automatically on their next render cycle.
    """
    results: dict[str, bool | str] = {}

    locality_changed = (
        new_locality is not None
        and new_locality.strip()
        and new_locality.strip() != old_locality.strip()
    )
    effective_locality = new_locality.strip() if locality_changed else old_locality

    # ── DELETE ────────────────────────────────────────────────────────────────
    if delete:
        results["sqlite"] = _delete_sqlite(meta_db_path, row_id)
        if kg_db_path: results["kg"]   = _kg_delete_species(kg_db_path, valid_name)
        if mb_db_path: results["mb"]   = _mb_delete_species(mb_db_path, valid_name)
        if wiki_root:  results["wiki"] = _wiki_delete_species(wiki_root, valid_name, old_locality)
        _broadcast_update(f"delete:{valid_name}@{old_locality}")
        return results

    # ── COORD UPDATE ──────────────────────────────────────────────────────────
    if lat is not None and lon is not None:
        results["sqlite_coords"] = _write_coords_sqlite(meta_db_path, row_id, lat, lon, source)

    # ── LOCALITY TEXT MIGRATION ───────────────────────────────────────────────
    if locality_changed:
        results["sqlite_locality"] = _write_locality_sqlite(
            meta_db_path, row_id, effective_locality)
        if kg_db_path:
            results["kg_locality"] = _kg_migrate_locality(
                kg_db_path, old_locality, effective_locality, lat, lon)
        if mb_db_path:
            results["mb_locality"] = _mb_migrate_locality(
                mb_db_path, valid_name, old_locality, effective_locality, lat, lon)
        if wiki_root:
            results["wiki_locality"] = _wiki_migrate_locality(
                wiki_root, valid_name, old_locality, effective_locality,
                lat, lon, occurrence=occurrence)

        if occurrence and (kg_db_path or mb_db_path):
            updated_occ = {**occurrence,
                           "verbatimLocality": effective_locality,
                           **({"decimalLatitude": lat, "decimalLongitude": lon,
                               "geocodingSource": source} if lat is not None else {})}
            enriched_occ = _re_verify_occurrence(updated_occ)

            if kg_db_path:
                try:
                    from biotrace_knowledge_graph import BioTraceKnowledgeGraph
                    kg = BioTraceKnowledgeGraph(kg_db_path)
                    kg.ingest_occurrences([enriched_occ])
                    kg.detect_communities()
                    kg.close()
                    results["kg_reingest"] = True
                except Exception as exc:
                    logger.warning("[HITL] KG re-ingest: %s", exc)
                    results["kg_reingest"] = False

            if mb_db_path:
                try:
                    from biotrace_memory_bank import BioTraceMemoryBank
                    mb = BioTraceMemoryBank(mb_db_path)
                    mb.store_occurrences([enriched_occ])
                    mb.close()
                    results["mb_reingest"] = True
                except Exception as exc:
                    logger.warning("[HITL] MB re-ingest: %s", exc)
                    results["mb_reingest"] = False
    else:
        # Coords-only
        if kg_db_path and lat is not None:
            results["kg_coords"] = _kg_update_locality_coords(
                kg_db_path, old_locality, lat, lon)
        if mb_db_path and lat is not None:
            results["mb_coords"] = _mb_update_coords(mb_db_path, valid_name, lat, lon)
        if wiki_root and lat is not None:
            results["wiki_coords"] = _wiki_update_coords(wiki_root, old_locality, lat, lon)

    # ── Spatio-temporal KG ────────────────────────────────────────────────────
    if st_kg_db_path and lat is not None:
        st_occ = {**(occurrence or {}), "validName": valid_name,
                  "verbatimLocality": effective_locality,
                  "decimalLatitude": lat, "decimalLongitude": lon}
        results["st_kg"] = _st_kg_update(st_kg_db_path, st_occ)

    # ── Cross-tab broadcast ───────────────────────────────────────────────────
    _broadcast_update(
        f"{'locality_edit' if locality_changed else 'coord_edit'}:"
        f"{valid_name}@{effective_locality}"
    )
    return results


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

def render_hitl_geocoding_tab(
    meta_db_path:  str,
    kg_db_path:    str = "",
    mb_db_path:    str = "",
    wiki_root:     str = "",
    kg_st_db_path: str = "",
    geocoder_cache_db: str = "biodiversity_data/geocoder_cache.db",
):
    """
    Render the HITL geocoding & record management tab.

    New in v5.6:
      • Enriched geocoder with strategy/confidence display
      • Cross-tab staleness flags set on every commit
      • Geocoder cache stats expander in sidebar area
    """
    import streamlit as st

    st.subheader("📍 Human-in-the-Loop Geocoding & Record Management")
    st.caption(
        "Review ungeocoded records, confirm enriched Nominatim suggestions, edit locality "
        "text, delete false-positives. **All changes auto-propagate to Occurrence Map, "
        "Data table, KG (GraphRAG + communities), Memory Bank, Wiki, and ST-KG.**"
    )

    if not _geopy_available and not _ENRICHED_GEOCODER_OK:
        st.error("**geopy not installed.** Run: `pip install geopy`")
        return

    sync_targets = [label for label, v in [
        ("KG (GraphRAG)", kg_db_path), ("Memory Bank", mb_db_path),
        ("Wiki", wiki_root), ("ST-KG", kg_st_db_path)] if v]
    if sync_targets:
        st.info(f"🔗 Sync targets: **{', '.join(sync_targets)}**")

    # ── Stale-data banner (set by sibling tabs) ───────────────────────────────
    if st.session_state.get("hitl_last_action"):
        st.success(f"✅ Last action: `{st.session_state['hitl_last_action']}`  "
                   f"— all tabs will refresh on next render.")

    # ── Controls ──────────────────────────────────────────────────────────────
    col_mode, col_refresh, col_rebuild = st.columns([4, 1, 1])
    with col_mode:
        mode = st.radio("View:", ["🗂️ Ungeocoded queue", "📋 All records (edit / delete)"],
                        horizontal=True)
    with col_refresh:
        if st.button("🔄 Refresh"):
            for k in ("hitl_queue", "hitl_nom_cache", "hitl_all"):
                st.session_state.pop(k, None)
            st.session_state.pop("hitl_kg_dirty", None)
            st.rerun()
    with col_rebuild:
        if kg_db_path and st.session_state.get("hitl_kg_dirty"):
            if st.button("🔁 Rebuild KG", help="Re-run community detection after edits"):
                with st.spinner("Rebuilding KG communities…"):
                    ok = _kg_rebuild_communities(kg_db_path)
                st.session_state.pop("hitl_kg_dirty", None)
                st.toast("✅ KG communities rebuilt." if ok else "⚠️ KG rebuild failed.")

    # ── Geocoder cache stats ──────────────────────────────────────────────────
    if _ENRICHED_GEOCODER_OK:
        with st.expander("🗺️ Geocoder Cache Stats", expanded=False):
            try:
                stats = get_geocoder(geocoder_cache_db).cache_stats()
                c1, c2 = st.columns(2)
                c1.metric("Cached hits", stats["cached_successes"])
                c2.metric("Cached failures", stats["cached_failures"])
                if stats["by_strategy"]:
                    st.dataframe(
                        pd.DataFrame.from_dict(stats["by_strategy"], orient="index",
                                               columns=["count"]).reset_index(
                            names="strategy"),
                        use_container_width=True, height=160,
                    )
                col_clear, _ = st.columns([1, 3])
                with col_clear:
                    if st.button("🗑️ Clear failure cache"):
                        get_geocoder(geocoder_cache_db).clear_failures()
                        st.toast("Failure cache cleared — stale localities will be re-queried.")
            except Exception as exc:
                st.warning(f"Cache stats unavailable: {exc}")

    completed: set = st.session_state.get("hitl_completed", set())

    # ════════════════════════════════════════════════════════════════════════
    #  MODE A — Ungeocoded queue
    # ════════════════════════════════════════════════════════════════════════
    if mode == "🗂️ Ungeocoded queue":
        if "hitl_queue" not in st.session_state:
            with st.spinner("Loading ungeocoded records…"):
                st.session_state["hitl_queue"]     = _load_missing(meta_db_path)
                st.session_state["hitl_nom_cache"] = {}

        queue   = st.session_state["hitl_queue"]
        cache   = st.session_state["hitl_nom_cache"]
        pending = [r for r in queue if r["id"] not in completed]

        if not pending:
            st.success("✅ Queue complete — all records geocoded.")
            return

        st.info(f"**{len(pending)} record(s)** awaiting geocoding.")

        for rec in pending:
            rid      = rec["id"]
            locality = rec["verbatimLocality"]
            species  = rec["validName"]
            citation = rec.get("sourceCitation", "")

            with st.container(border=True):
                col_info, col_action = st.columns([2, 2])

                with col_info:
                    st.markdown(f"**Species:** *{species}*")
                    st.markdown(f"**Locality:** `{locality}`")
                    st.caption(citation[:120])

                with col_action:
                    # Enriched geocoder lookup (cached per locality)
                    if locality not in cache:
                        with st.spinner("Querying enriched geocoder…"):
                            cache[locality] = _lookup_locality(locality, geocoder_cache_db)
                            st.session_state["hitl_nom_cache"] = cache

                    sug = cache.get(locality)

                    if sug:
                        strategy  = sug.get("strategy", "unknown")
                        conf      = sug.get("confidence", 1.0)
                        cached_lbl = " *(cached)*" if sug.get("cached") else ""
                        st.success(
                            f"📍 **{sug['display_name'][:80]}**\n\n"
                            f"Lat `{sug['lat']:.5f}` · Lon `{sug['lon']:.5f}`  \n"
                            f"Strategy: `{strategy}` · Confidence: `{conf:.2f}`{cached_lbl}"
                        )
                        st.map(pd.DataFrame([{"lat": sug["lat"], "lon": sug["lon"]}]), zoom=6)

                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            if st.button("✅ Accept", key=f"acc_{rid}"):
                                occ = _fetch_record(meta_db_path, rid) or {}
                                r = sync_all_stores(
                                    meta_db_path, kg_db_path, mb_db_path, wiki_root,
                                    rid, species, locality,
                                    sug["lat"], sug["lon"],
                                    f"EnrichedGeocoder_{strategy}",
                                    st_kg_db_path=kg_st_db_path, occurrence=occ)
                                completed.add(rid)
                                st.session_state["hitl_completed"] = completed
                                st.toast(f"✅ Saved → {list(r.keys())}")
                                st.rerun()
                        with c2:
                            if st.button("✏️ Edit", key=f"edit_{rid}"):
                                st.session_state[f"hitl_ed_{rid}"] = True
                        with c3:
                            if st.button("🚫 Skip", key=f"skip_{rid}"):
                                completed.add(rid)
                                st.session_state["hitl_completed"] = completed
                                st.rerun()
                        with c4:
                            if st.button("🗑️ Delete", key=f"del_{rid}"):
                                st.session_state[f"hitl_cdel_{rid}"] = True
                    else:
                        st.warning("No geocoder result — opening manual edit.")
                        st.session_state[f"hitl_ed_{rid}"] = True

                    # Delete confirmation
                    if st.session_state.get(f"hitl_cdel_{rid}"):
                        st.error(
                            f"Permanently delete **{species}** @ *{locality}* from ALL databases?")
                        d1, d2 = st.columns(2)
                        with d1:
                            if st.button("⚠️ Confirm Delete", key=f"cfd_{rid}", type="primary"):
                                sync_all_stores(
                                    meta_db_path, kg_db_path, mb_db_path, wiki_root,
                                    rid, species, locality, None, None, "", delete=True,
                                    st_kg_db_path=kg_st_db_path)
                                completed.add(rid)
                                st.session_state["hitl_completed"] = completed
                                st.session_state.pop(f"hitl_cdel_{rid}", None)
                                st.toast(f"🗑️ Deleted #{rid} from all stores.")
                                st.rerun()
                        with d2:
                            if st.button("Cancel", key=f"can_{rid}"):
                                st.session_state.pop(f"hitl_cdel_{rid}", None)
                                st.rerun()

                    if st.session_state.get(f"hitl_ed_{rid}"):
                        _render_edit_form(
                            rid, species, locality, citation, sug,
                            meta_db_path, kg_db_path, mb_db_path, wiki_root,
                            kg_st_db_path, completed, geocoder_cache_db=geocoder_cache_db)

    # ════════════════════════════════════════════════════════════════════════
    #  MODE B — All records
    # ════════════════════════════════════════════════════════════════════════
    else:
        if "hitl_all" not in st.session_state:
            st.session_state["hitl_all"] = _load_all_records(meta_db_path)

        recs = st.session_state["hitl_all"]
        if not recs:
            st.info("No records found."); return

        df   = pd.DataFrame(recs)
        filt = st.text_input("🔍 Filter by species or locality:")
        if filt:
            mask = (df["validName"].str.contains(filt, case=False, na=False)
                    | df["verbatimLocality"].str.contains(filt, case=False, na=False))
            df = df[mask]

        st.dataframe(
            df[["id", "validName", "verbatimLocality", "lat", "lon",
                "occurrenceType", "geocodingSource"]],
            use_container_width=True, height=350)
        st.divider()

        st.markdown("### ✏️ Edit or Delete a record by ID")
        edit_id = int(st.number_input("Record ID:", min_value=1, step=1))
        matching = [r for r in recs if r["id"] == edit_id]
        if matching:
            rec = matching[0]
            st.info(
                f"**{rec['validName']}** @ *{rec['verbatimLocality']}*  "
                f"| type: `{rec['occurrenceType']}` | src: `{rec['geocodingSource']}`")
            tab_edit, tab_del = st.tabs(["✏️ Edit coords / locality", "🗑️ Delete"])
            with tab_edit:
                _render_edit_form(
                    rec["id"], rec["validName"], rec["verbatimLocality"],
                    rec.get("sourceCitation", ""),
                    {"lat": rec["lat"], "lon": rec["lon"], "display_name": ""}
                    if rec["lat"] else None,
                    meta_db_path, kg_db_path, mb_db_path, wiki_root,
                    kg_st_db_path, completed,
                    show_locality_edit=True,
                    geocoder_cache_db=geocoder_cache_db)
            with tab_del:
                st.warning(
                    f"This will permanently remove **{rec['validName']}** "
                    f"@ *{rec['verbatimLocality']}* from SQLite, KG, Memory Bank, and Wiki.")
                if st.button("🗑️ Confirm Delete", key=f"del_all_{edit_id}", type="primary"):
                    sync_all_stores(
                        meta_db_path, kg_db_path, mb_db_path, wiki_root,
                        rec["id"], rec["validName"], rec["verbatimLocality"],
                        None, None, "", delete=True, st_kg_db_path=kg_st_db_path)
                    st.success(f"Record #{edit_id} deleted from all stores.")
                    st.session_state.pop("hitl_all", None)
                    st.rerun()

    if completed:
        st.divider()
        st.success(f"**{len(completed)} action(s) completed this session.**")
        if st.session_state.get("hitl_kg_dirty"):
            st.warning(
                "⚠️ KG community graph has pending structural changes. "
                "Click **🔁 Rebuild KG** above to refresh GraphRAG clusters.")


# ─── Edit form ────────────────────────────────────────────────────────────────

def _render_edit_form(
    rid: int, species: str, locality: str, citation: str,
    sug: Optional[dict],
    meta_db_path: str, kg_db_path: str, mb_db_path: str,
    wiki_root: str, kg_st_db_path: str, completed: set,
    *,
    show_locality_edit: bool = False,
    geocoder_cache_db: str = "biodiversity_data/geocoder_cache.db",
):
    import streamlit as st

    with st.form(key=f"form_{rid}"):
        st.markdown("#### 📝 Manual Edit")
        lat_default = str(round(sug["lat"], 5)) if sug and sug.get("lat") else ""
        lon_default = str(round(sug["lon"], 5)) if sug and sug.get("lon") else ""
        lat_in = st.text_input("Decimal Latitude  (N positive)",  value=lat_default)
        lon_in = st.text_input("Decimal Longitude (E positive)", value=lon_default)
        new_loc_in = st.text_input(
            "Correct verbatimLocality (leave blank to keep current)",
            value="" if not show_locality_edit else locality,
            help="Editing this renames the Locality node in KG, migrates MemoryBank atoms, "
                 "updates Wiki articles, triggers community rebuild, and refreshes all tabs.")
        re_verify = st.checkbox(
            "Re-run taxonomy verification after edit", value=True,
            help="Uses BioTraceUnifiedVerifier to refresh phylum/family/WoRMS fields.")
        submitted = st.form_submit_button("💾 Commit")

    if submitted:
        try:
            lat_f = float(lat_in) if lat_in.strip() else None
            lon_f = float(lon_in) if lon_in.strip() else None
            if lat_f is not None and not (-90 <= lat_f <= 90):
                st.error("Latitude must be between -90 and 90."); return
            if lon_f is not None and not (-180 <= lon_f <= 180):
                st.error("Longitude must be between -180 and 180."); return

            new_locality = new_loc_in.strip() or None
            occ = _fetch_record(meta_db_path, rid) or {}
            if re_verify:
                occ = _re_verify_occurrence(occ)

            with st.spinner("Syncing all stores…"):
                r = sync_all_stores(
                    meta_db_path, kg_db_path, mb_db_path, wiki_root,
                    rid, species, locality, lat_f, lon_f, "Manual_HITL",
                    new_locality=new_locality,
                    st_kg_db_path=kg_st_db_path,
                    occurrence=occ)

            completed.add(rid)
            st.session_state["hitl_completed"] = completed
            st.session_state.pop(f"hitl_ed_{rid}", None)

            changed = [k for k, v in r.items() if v]
            st.toast(f"✅ Committed — stores updated: {changed}")
            st.rerun()

        except ValueError:
            st.error("Invalid numeric value for coordinates.")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            logger.exception("[HITL] form commit error")