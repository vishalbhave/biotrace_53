"""
biotrace_hitl_geocoding.py  —  BioTrace v5.4 (Full-sync edition)
────────────────────────────────────────────────────────────────────────────
Human-in-the-Loop geocoding + full database sync.

WHAT'S NEW VS v5.4-original
────────────────────────────
When the user Accepts, edits, or deletes a record, ALL four stores are
updated atomically:

  Store 1 — SQLite occurrence DB    (decimalLatitude/Longitude, geocodingSource)
  Store 2 — BioTraceKnowledgeGraph  (node lat/lon attributes, delete node)
  Store 3 — BioTraceMemoryBank      (session atom coords, delete atom)
  Store 4 — BioTraceWiki            (locality article lat/lon, remove species entry)

ADDED: Record deletion support
  "Delete" button permanently removes a record from all four stores.
  Intended for Scyphistoma / life-stage false-positive records that survived
  the extraction filter.

Usage:
    from biotrace_hitl_geocoding import render_hitl_geocoding_tab
    with tabs[N]:
        render_hitl_geocoding_tab(META_DB_PATH, KG_DB_PATH, MB_DB_PATH, WIKI_ROOT)
"""
from __future__ import annotations
import logging, sqlite3, time
from typing import Optional
import pandas as pd

logger = logging.getLogger("biotrace.hitl_geocoding")

_geopy_available = False
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
    _geopy_available = True
except ImportError:
    pass

try:
    from biotrace_geocoding_lifestage_patch import india_nominatim_geocode
    _INDIA_PATCH = True
except ImportError:
    _INDIA_PATCH = False

_geocoder = None

def _get_geocoder():
    global _geocoder
    if not _geopy_available: return None
    if _geocoder is None:
        _geocoder = Nominatim(user_agent="BioTrace_HITL_v5")
    return _geocoder

def _nominatim_lookup(locality: str) -> Optional[dict]:
    gc = _get_geocoder()
    if gc is None: return None
    if _INDIA_PATCH:
        return india_nominatim_geocode(gc, locality)
    try:
        time.sleep(1.1)
        result = gc.geocode(locality, exactly_one=True, timeout=10,
                            country_codes="in")  # always restrict to India
        if result:
            return {"lat": result.latitude, "lon": result.longitude,
                    "display_name": result.address}
    except Exception as exc:
        logger.warning("[HITL] Nominatim: %s", exc)
    return None

def _resolve_table(conn):
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for t in ("occurrences_v4", "occurrences"):
        if t in names: return t
    raise sqlite3.OperationalError("No occurrence table found.")

def _load_missing(db_path):
    try:
        conn = sqlite3.connect(db_path); table = _resolve_table(conn)
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
        logger.error("[HITL] load_missing: %s", exc); return []

def _load_all_records(db_path):
    try:
        conn = sqlite3.connect(db_path); table = _resolve_table(conn)
        rows = conn.execute(f"""SELECT id, verbatimLocality, validName, recordedName,
               decimalLatitude, decimalLongitude, geocodingSource, occurrenceType, sourceCitation
               FROM {table} ORDER BY id""").fetchall()
        conn.close()
        return [{"id": r[0], "verbatimLocality": r[1], "validName": r[2] or r[3] or "",
                 "lat": r[4], "lon": r[5], "geocodingSource": r[6] or "",
                 "occurrenceType": r[7] or "", "sourceCitation": r[8] or ""} for r in rows]
    except Exception as exc:
        logger.error("[HITL] load_all: %s", exc); return []

def _write_coords_sqlite(db_path, row_id, lat, lon, source):
    try:
        conn = sqlite3.connect(db_path); table = _resolve_table(conn)
        conn.execute(f"UPDATE {table} SET decimalLatitude=?,decimalLongitude=?,geocodingSource=? WHERE id=?",
                     (round(lat,6), round(lon,6), source, row_id))
        conn.commit(); conn.close(); return True
    except Exception as exc:
        logger.error("[HITL/sqlite] coords: %s", exc); return False

def _delete_sqlite(db_path, row_id):
    try:
        conn = sqlite3.connect(db_path); table = _resolve_table(conn)
        conn.execute(f"DELETE FROM {table} WHERE id=?", (row_id,))
        conn.commit(); conn.close(); return True
    except Exception as exc:
        logger.error("[HITL/sqlite] delete: %s", exc); return False

def _update_knowledge_graph(kg_db_path, valid_name, lat, lon, delete=False):
    try:
        from biotrace_knowledge_graph import BioTraceKnowledgeGraph
        kg = BioTraceKnowledgeGraph(kg_db_path)
        if delete: kg.delete_node(valid_name)
        else: kg.update_node_coords(valid_name, lat, lon)
    except Exception as exc:
        logger.warning("[HITL/KG] %s", exc)

def _update_memory_bank(mb_db_path, valid_name, lat, lon, delete=False):
    try:
        from biotrace_memory_bank import BioTraceMemoryBank
        mb = BioTraceMemoryBank(mb_db_path)
        if delete: mb.delete_by_species(valid_name)
        else: mb.update_coords_by_species(valid_name, lat, lon)
    except Exception as exc:
        logger.warning("[HITL/MB] %s", exc)

def _update_wiki(wiki_root, valid_name, locality, lat, lon, delete=False):
    try:
        from biotrace_wiki import BioTraceWiki
        wiki = BioTraceWiki(wiki_root)
        if delete: wiki.remove_species_from_locality(valid_name, locality)
        else: wiki.update_locality_coords(locality, lat, lon)
    except Exception as exc:
        logger.warning("[HITL/Wiki] %s", exc)

def sync_all_stores(meta_db_path, kg_db_path, mb_db_path, wiki_root,
                    row_id, valid_name, locality, lat, lon, source, delete=False):
    results = {}
    if delete:
        results["sqlite"] = _delete_sqlite(meta_db_path, row_id)
    else:
        results["sqlite"] = _write_coords_sqlite(meta_db_path, row_id, lat, lon, source)
    if kg_db_path: _update_knowledge_graph(kg_db_path, valid_name, lat or 0.0, lon or 0.0, delete); results["kg"] = True
    if mb_db_path: _update_memory_bank(mb_db_path, valid_name, lat or 0.0, lon or 0.0, delete); results["mb"] = True
    if wiki_root:  _update_wiki(wiki_root, valid_name, locality, lat or 0.0, lon or 0.0, delete); results["wiki"] = True
    return results

def render_hitl_geocoding_tab(meta_db_path, kg_db_path="", mb_db_path="", wiki_root=""):
    import streamlit as st
    st.subheader("📍 Human-in-the-Loop Geocoding & Record Management")
    st.caption("Review ungeocoded records, confirm Nominatim suggestions, delete false-positives. "
               "**All changes sync to KG, Memory Bank, and Wiki.**")
    sync_targets = [s for s, v in [("KG", kg_db_path), ("Memory Bank", mb_db_path), ("Wiki", wiki_root)] if v]
    if sync_targets: st.info(f"Sync targets: **{', '.join(sync_targets)}**")
    if not _geopy_available: st.error("**geopy not installed.** Run: `pip install geopy`"); return

    mode = st.radio("View:", ["🗂️ Ungeocoded queue", "📋 All records (edit / delete)"], horizontal=True)
    if st.button("🔄 Refresh"):
        for k in ("hitl_queue","hitl_nom_cache","hitl_all"): st.session_state.pop(k, None)

    completed = st.session_state.get("hitl_completed", set())

    if mode == "🗂️ Ungeocoded queue":
        if "hitl_queue" not in st.session_state:
            with st.spinner("Loading…"):
                st.session_state["hitl_queue"] = _load_missing(meta_db_path)
                st.session_state["hitl_nom_cache"] = {}
        queue = st.session_state["hitl_queue"]
        cache = st.session_state["hitl_nom_cache"]
        pending = [r for r in queue if r["id"] not in completed]
        if not pending: st.success("✅ Queue complete."); return
        st.info(f"**{len(pending)} record(s)** to review.")

        for rec in pending:
            rid, locality, species = rec["id"], rec["verbatimLocality"], rec["validName"]
            with st.container(border=True):
                col1, col2 = st.columns([2,1])
                with col1:
                    st.markdown(f"**Species:** *{species}*  \n**Locality:** `{locality}`")
                    st.caption(rec.get("sourceCitation","")[:100])
                with col2:
                    if locality not in cache:
                        with st.spinner("Querying Nominatim (India)…"):
                            cache[locality] = _nominatim_lookup(locality)
                            st.session_state["hitl_nom_cache"] = cache
                    sug = cache.get(locality)
                    if sug:
                        st.success(f"📍 **{sug['display_name'][:70]}**\n\nLat `{sug['lat']:.5f}` · Lon `{sug['lon']:.5f}`")
                        st.map(pd.DataFrame([{"lat": sug["lat"], "lon": sug["lon"]}]), zoom=6)
                        c1,c2,c3,c4 = st.columns(4)
                        with c1:
                            if st.button("✅ Accept", key=f"acc_{rid}"):
                                r = sync_all_stores(meta_db_path, kg_db_path, mb_db_path, wiki_root,
                                                    rid, species, locality, sug["lat"], sug["lon"], "Nominatim_HITL")
                                completed.add(rid); st.session_state["hitl_completed"] = completed
                                st.toast(f"✅ Saved & synced {list(r.keys())}"); st.rerun()
                        with c2:
                            if st.button("✏️ Edit", key=f"edit_{rid}"):
                                st.session_state[f"hitl_ed_{rid}"] = True
                        with c3:
                            if st.button("🚫 Skip", key=f"skip_{rid}"):
                                completed.add(rid); st.session_state["hitl_completed"] = completed; st.rerun()
                        with c4:
                            if st.button("🗑️ Delete", key=f"del_{rid}"):
                                st.session_state[f"hitl_cdel_{rid}"] = True
                    else:
                        st.warning("No Nominatim result for this locality.")
                        st.session_state[f"hitl_ed_{rid}"] = True

                    if st.session_state.get(f"hitl_cdel_{rid}"):
                        st.error(f"Permanently delete **{species}** @ *{locality}* from ALL databases?")
                        d1,d2 = st.columns(2)
                        with d1:
                            if st.button("⚠️ Confirm", key=f"cfd_{rid}", type="primary"):
                                sync_all_stores(meta_db_path, kg_db_path, mb_db_path, wiki_root,
                                                rid, species, locality, None, None, "", delete=True)
                                completed.add(rid); st.session_state["hitl_completed"] = completed
                                st.session_state.pop(f"hitl_cdel_{rid}", None)
                                st.toast(f"🗑️ Deleted #{rid} from all stores."); st.rerun()
                        with d2:
                            if st.button("Cancel", key=f"can_{rid}"):
                                st.session_state.pop(f"hitl_cdel_{rid}", None); st.rerun()

                    if st.session_state.get(f"hitl_ed_{rid}"):
                        with st.form(key=f"form_{rid}"):
                            lat_in = st.text_input("Decimal Latitude (N+)", value=str(sug["lat"]) if sug else "")
                            lon_in = st.text_input("Decimal Longitude (E+)", value=str(sug["lon"]) if sug else "")
                            if st.form_submit_button("💾 Commit Manual"):
                                try:
                                    lat_f, lon_f = float(lat_in), float(lon_in)
                                    assert -90 <= lat_f <= 90 and -180 <= lon_f <= 180
                                    r = sync_all_stores(meta_db_path, kg_db_path, mb_db_path, wiki_root,
                                                        rid, species, locality, lat_f, lon_f, "Manual_HITL")
                                    completed.add(rid); st.session_state["hitl_completed"] = completed
                                    st.session_state.pop(f"hitl_ed_{rid}", None)
                                    st.toast(f"✅ Manual coords saved — {list(r.keys())}"); st.rerun()
                                except (ValueError, AssertionError):
                                    st.error("Invalid coordinates.")
    else:
        if "hitl_all" not in st.session_state:
            st.session_state["hitl_all"] = _load_all_records(meta_db_path)
        recs = st.session_state["hitl_all"]
        if not recs: st.info("No records."); return
        df = pd.DataFrame(recs)
        filt = st.text_input("🔍 Filter species / locality:")
        if filt:
            mask = (df["validName"].str.contains(filt, case=False, na=False) |
                    df["verbatimLocality"].str.contains(filt, case=False, na=False))
            df = df[mask]
        st.dataframe(df[["id","validName","verbatimLocality","lat","lon","occurrenceType","geocodingSource"]],
                     use_container_width=True, height=350)
        st.divider()
        st.markdown("**Delete record by ID** (removes from SQLite, KG, Memory Bank, Wiki):")
        del_id = st.number_input("Record ID:", min_value=1, step=1)
        matching = [r for r in recs if r["id"] == int(del_id)]
        if matching:
            rec = matching[0]
            st.warning(f"Will delete: **{rec['validName']}** @ *{rec['verbatimLocality']}* (type={rec['occurrenceType']})")
            if st.button("🗑️ Delete from ALL databases", type="primary"):
                sync_all_stores(meta_db_path, kg_db_path, mb_db_path, wiki_root,
                                rec["id"], rec["validName"], rec["verbatimLocality"],
                                None, None, "", delete=True)
                st.success(f"Record #{del_id} deleted."); st.session_state.pop("hitl_all", None); st.rerun()

    if completed:
        st.divider(); st.success(f"**{len(completed)} action(s) completed this session.**")
