"""
biotrace_hitl_staging_tab.py  —  BioTrace v5.5
────────────────────────────────────────────────────────────────────────────
Persistent HITL (Human-in-the-Loop) verification tab.

WHY THE ORIGINAL KEPT RESETTING
────────────────────────────────
The v5.4 approval gate used `st.session_state` to track which records were
approved, but `st.rerun()` after each Accept/Delete action caused the entire
Streamlit script to re-execute, resetting the session state keys that were
not yet persisted. Consequently, the approved set was cleared and the user
had to restart from scratch every session.

SOLUTION: SQLITE STAGING TABLE
───────────────────────────────────
All pending records live in a `verification_staging` SQLite table (same DB
as the main occurrence store). State is NEVER stored in session_state —
only in SQLite. Every button action writes to SQLite first, then the UI re-
reads from SQLite. This means:

  • Streamlit reruns are cheap and safe — state survives.
  • Closing and reopening the browser continues where you left off.
  • "Commit to main DB" is the only action that touches occurrences_v4.

WORKFLOW
────────
  1. After extraction completes, call:
       stage_records_for_hitl(META_DB_PATH, occurrences)
     This inserts all records into verification_staging with status="pending".

  2. Open the HITL tab (render_hitl_staging_tab) to:
       • Edit any cell inline (name, locality, occurrenceType, coordinates)
       • Delete false-positives (Scyphistoma, life-stage artefacts, etc.)
       • Geocode un-georeferenced records row-by-row via Nominatim
       • Re-verify a name against WoRMS/GNV on demand
       • Mark individual rows "verified" or "rejected"

  3. Click "Commit verified records → Main DB" to move accepted rows into
     occurrences_v4 and update KG/Memory Bank/Wiki.

COLUMNS IN verification_staging
────────────────────────────────
  id, recorded_name, valid_name, taxon_rank, taxonomic_status,
  verbatim_locality, decimal_latitude, decimal_longitude, geocoding_source,
  occurrence_type, source_citation, phylum, class_, order_, family_,
  worms_id, gbif_key, col_id, unified_confidence, verification_sources,
  status (pending|verified|edited|rejected),
  notes, staged_at, committed_at

Wire into biotrace_v5.py:

    # After extraction + verification:
    from biotrace_hitl_staging_tab import stage_records_for_hitl, render_hitl_staging_tab
    n = stage_records_for_hitl(META_DB_PATH, occurrences)
    st.success(f"{n} records staged for HITL review")

    # In the HITL tab:
    with tabs[N]:
        render_hitl_staging_tab(
            meta_db_path = META_DB_PATH,
            kg_db_path   = KG_DB_PATH,
            mb_db_path   = MB_DB_PATH,
            wiki_root    = WIKI_ROOT,
        )
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import Optional

import pandas as pd

logger = logging.getLogger("biotrace.hitl_staging")

# ─────────────────────────────────────────────────────────────────────────────
#  Schema
# ─────────────────────────────────────────────────────────────────────────────

_STAGING_SCHEMA = """
CREATE TABLE IF NOT EXISTS verification_staging (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_name       TEXT,
    valid_name          TEXT,
    taxon_rank          TEXT,
    taxonomic_status    TEXT DEFAULT 'unverified',
    verbatim_locality   TEXT,
    decimal_latitude    REAL,
    decimal_longitude   REAL,
    geocoding_source    TEXT,
    occurrence_type     TEXT,
    source_citation     TEXT,
    phylum              TEXT,
    class_              TEXT,
    order_              TEXT,
    family_             TEXT,
    worms_id            TEXT,
    gbif_key            TEXT,
    col_id              TEXT,
    unified_confidence  REAL DEFAULT 0.0,
    verification_sources TEXT,
    habitat             TEXT,
    raw_text_evidence   TEXT,
    status              TEXT DEFAULT 'pending',
    notes               TEXT,
    staged_at           TEXT DEFAULT (datetime('now')),
    committed_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_vs_status ON verification_staging(status);
CREATE INDEX IF NOT EXISTS idx_vs_name   ON verification_staging(recorded_name);
"""

_STATUS_PENDING   = "pending"
_STATUS_VERIFIED  = "verified"
_STATUS_EDITED    = "edited"
_STATUS_REJECTED  = "rejected"
_STATUS_COMMITTED = "committed"


# ─────────────────────────────────────────────────────────────────────────────
#  Staging loader
# ─────────────────────────────────────────────────────────────────────────────

def ensure_staging_table(db_path: str):
    """Create staging table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.executescript(_STAGING_SCHEMA)
    conn.commit()
    conn.close()


def stage_records_for_hitl(db_path: str, occurrences: list[dict]) -> int:
    """
    Insert occurrence records into verification_staging.
    Existing uncommitted records for the same (recorded_name, verbatim_locality)
    are skipped to avoid duplicates across re-runs.

    Returns number of newly staged records.
    """
    ensure_staging_table(db_path)
    conn = sqlite3.connect(db_path)

    # Load existing pending keys to deduplicate
    existing = set()
    for row in conn.execute(
        "SELECT recorded_name, verbatim_locality FROM verification_staging WHERE status != ?",
        (_STATUS_COMMITTED,)
    ).fetchall():
        existing.add((str(row[0]).strip().lower(), str(row[1]).strip().lower()))

    staged = 0
    for occ in occurrences:
        if not isinstance(occ, dict):
            continue

        recorded = str(occ.get("recordedName") or occ.get("Recorded Name", "")).strip()
        locality = str(occ.get("verbatimLocality", "")).strip()

        # Skip __candidate_ placeholders
        if recorded.startswith("__candidate_") or recorded.startswith("_candidate"):
            continue

        # Skip empty names
        if not recorded:
            continue

        # Dedup check
        key = (recorded.lower(), locality.lower())
        if key in existing:
            continue
        existing.add(key)

        conn.execute("""
            INSERT INTO verification_staging
              (recorded_name, valid_name, taxon_rank, taxonomic_status,
               verbatim_locality, decimal_latitude, decimal_longitude,
               geocoding_source, occurrence_type, source_citation,
               phylum, class_, order_, family_,
               worms_id, gbif_key, col_id,
               unified_confidence, verification_sources,
               habitat, raw_text_evidence, status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            recorded,
            str(occ.get("validName", "")),
            str(occ.get("taxonRank", "")),
            str(occ.get("taxonomicStatus", "unverified")),
            locality,
            occ.get("decimalLatitude"),
            occ.get("decimalLongitude"),
            str(occ.get("geocodingSource", "")),
            str(occ.get("occurrenceType", "")),
            str(occ.get("sourceCitation") or occ.get("Source Citation", "")),
            str(occ.get("phylum", "")),
            str(occ.get("class_", "")),
            str(occ.get("order_", "")),
            str(occ.get("family_", "")),
            str(occ.get("wormsID", "")),
            str(occ.get("gbifKey", "")),
            str(occ.get("colID", "")),
            float(occ.get("unifiedConfidence") or occ.get("matchScore") or 0.0),
            str(occ.get("verificationSources", "")),
            str(occ.get("Habitat", "") or occ.get("habitat", "")),
            str(occ.get("Raw Text Evidence") or occ.get("rawTextEvidence", "")),
            _STATUS_PENDING,
        ))
        staged += 1

    conn.commit()
    conn.close()
    logger.info("[staging] %d new records staged", staged)
    return staged


# ─────────────────────────────────────────────────────────────────────────────
#  SQLite helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_staging(db_path: str, filter_status: Optional[str] = None) -> list[dict]:
    """Load staging records. filter_status=None loads all non-committed."""
    conn = sqlite3.connect(db_path)
    if filter_status:
        rows = conn.execute(
            "SELECT * FROM verification_staging WHERE status=? ORDER BY id",
            (filter_status,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM verification_staging WHERE status != ? ORDER BY id",
            (_STATUS_COMMITTED,)
        ).fetchall()
    cols = [d[0] for d in conn.execute("PRAGMA table_info(verification_staging)").fetchall()]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


def _update_field(db_path: str, row_id: int, field: str, value):
    """Update a single field in staging table, mark as edited."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        f"UPDATE verification_staging SET {field}=?, status=? WHERE id=?",
        (value, _STATUS_EDITED, row_id)
    )
    conn.commit()
    conn.close()


def _update_fields(db_path: str, row_id: int, updates: dict, status: str = _STATUS_EDITED):
    """Update multiple fields in one call."""
    if not updates:
        return
    set_clause = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [status, row_id]
    conn = sqlite3.connect(db_path)
    conn.execute(
        f"UPDATE verification_staging SET {set_clause}, status=? WHERE id=?", vals
    )
    conn.commit()
    conn.close()


def _set_status(db_path: str, row_id: int, status: str):
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE verification_staging SET status=? WHERE id=?", (status, row_id))
    conn.commit()
    conn.close()


def _delete_staging(db_path: str, row_id: int):
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM verification_staging WHERE id=?", (row_id,))
    conn.commit()
    conn.close()


def _get_stats(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM verification_staging GROUP BY status"
    ).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


# ─────────────────────────────────────────────────────────────────────────────
#  Geocoding helpers
# ─────────────────────────────────────────────────────────────────────────────

_geocoder = None

def _get_geocoder():
    global _geocoder
    try:
        from geopy.geocoders import Nominatim
        if _geocoder is None:
            _geocoder = Nominatim(user_agent="BioTrace_HITL_v5.5")
        return _geocoder
    except ImportError:
        return None


def _nominatim_geocode(locality: str) -> Optional[dict]:
    gc = _get_geocoder()
    if not gc:
        return None
    try:
        time.sleep(1.1)
        result = gc.geocode(locality, exactly_one=True, timeout=10, country_codes="in")
        if result:
            return {"lat": result.latitude, "lon": result.longitude,
                    "display_name": result.address}
    except Exception as exc:
        logger.warning("[HITL] Nominatim: %s", exc)
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  On-demand WoRMS re-verification
# ─────────────────────────────────────────────────────────────────────────────

def _reverify_worms(name: str) -> dict:
    """Quick WoRMS lookup for a single name. Returns dict of enrichment fields."""
    import requests
    try:
        r = requests.get(
            f"https://www.marinespecies.org/rest/AphiaRecordsByName/{requests.utils.quote(name)}",
            params={"like": "false", "marine_only": "false"},
            timeout=12,
        )
        r.raise_for_status()
        records = r.json()
        if records and isinstance(records, list):
            rec = records[0]
            return {
                "valid_name":       rec.get("valid_name", "") or rec.get("scientificname", ""),
                "worms_id":         str(rec.get("AphiaID", "")),
                "phylum":           rec.get("phylum", ""),
                "class_":           rec.get("class", ""),
                "order_":           rec.get("order", ""),
                "family_":          rec.get("family", ""),
                "taxonomic_status": "accepted" if (rec.get("status","") or "").lower() == "accepted" else "synonym",
            }
    except Exception as exc:
        logger.debug("[HITL/WoRMS] %s: %s", name, exc)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
#  Commit to main occurrences DB
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_occ_table(conn) -> str:
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for t in ("occurrences_v4", "occurrences"):
        if t in names:
            return t
    raise sqlite3.OperationalError("No occurrence table found.")


def commit_verified_to_main(
    db_path: str,
    kg_db_path: str = "",
    mb_db_path: str = "",
    wiki_root: str = "",
    statuses_to_commit: tuple = (_STATUS_VERIFIED, _STATUS_EDITED),
) -> int:
    """
    Move verified/edited records from staging to main occurrences table.
    Syncs to KG, Memory Bank, and Wiki after commit.
    Returns number of records committed.
    """
    ensure_staging_table(db_path)
    conn = sqlite3.connect(db_path)

    # Load records to commit
    placeholders = ",".join("?" for _ in statuses_to_commit)
    rows = conn.execute(
        f"SELECT * FROM verification_staging WHERE status IN ({placeholders})",
        list(statuses_to_commit)
    ).fetchall()
    cols = [d[0] for d in conn.execute("PRAGMA table_info(verification_staging)").fetchall()]
    conn.close()

    if not rows:
        return 0

    records = [dict(zip(cols, r)) for r in rows]

    # Open main DB connection
    main_conn = sqlite3.connect(db_path)
    try:
        occ_table = _resolve_occ_table(main_conn)
    except sqlite3.OperationalError:
        # Create minimal occurrences table if absent
        occ_table = "occurrences_v4"
        main_conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {occ_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recordedName TEXT, validName TEXT, taxonRank TEXT,
                taxonomicStatus TEXT, verbatimLocality TEXT,
                decimalLatitude REAL, decimalLongitude REAL,
                geocodingSource TEXT, occurrenceType TEXT,
                sourceCitation TEXT, phylum TEXT, class_ TEXT,
                order_ TEXT, family_ TEXT, wormsID TEXT,
                gbifKey TEXT, colID TEXT, unifiedConfidence REAL,
                verificationSources TEXT, habitat TEXT,
                rawTextEvidence TEXT, validationStatus TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        main_conn.commit()

    committed = 0
    for rec in records:
        try:
            main_conn.execute(f"""
                INSERT INTO {occ_table}
                  (recordedName, validName, taxonRank, taxonomicStatus,
                   verbatimLocality, decimalLatitude, decimalLongitude,
                   geocodingSource, occurrenceType, sourceCitation,
                   phylum, class_, order_, family_,
                   wormsID, gbifKey, colID, unifiedConfidence,
                   verificationSources, habitat, rawTextEvidence,
                   validationStatus)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                rec.get("recorded_name"),    rec.get("valid_name"),
                rec.get("taxon_rank"),       rec.get("taxonomic_status"),
                rec.get("verbatim_locality"),
                rec.get("decimal_latitude"), rec.get("decimal_longitude"),
                rec.get("geocoding_source"), rec.get("occurrence_type"),
                rec.get("source_citation"),
                rec.get("phylum"),   rec.get("class_"),
                rec.get("order_"),   rec.get("family_"),
                rec.get("worms_id"), rec.get("gbif_key"), rec.get("col_id"),
                rec.get("unified_confidence"),
                rec.get("verification_sources"),
                rec.get("habitat"),  rec.get("raw_text_evidence"),
                "verified",
            ))
            committed += 1
        except Exception as exc:
            logger.warning("[commit] row %s: %s", rec.get("id"), exc)

    main_conn.commit()
    main_conn.close()

    # Mark staging rows as committed
    committed_ids = [r["id"] for r in records]
    conn = sqlite3.connect(db_path)
    conn.execute(
        f"UPDATE verification_staging SET status=?, committed_at=datetime('now') "
        f"WHERE id IN ({','.join('?' for _ in committed_ids)})",
        [_STATUS_COMMITTED] + committed_ids
    )
    conn.commit()
    conn.close()

    # Sync to KG/MB/Wiki
    for rec in records:
        valid = rec.get("valid_name") or rec.get("recorded_name", "")
        lat   = rec.get("decimal_latitude")
        lon   = rec.get("decimal_longitude")
        loc   = rec.get("verbatim_locality", "")
        if kg_db_path and valid and lat and lon:
            try:
                from biotrace_knowledge_graph import BioTraceKnowledgeGraph
                BioTraceKnowledgeGraph(kg_db_path).update_node_coords(valid, lat, lon)
            except Exception as e:
                logger.debug("[commit/KG] %s", e)
        if mb_db_path and valid:
            try:
                from biotrace_memory_bank import BioTraceMemoryBank
                if lat and lon:
                    BioTraceMemoryBank(mb_db_path).update_coords_by_species(valid, lat, lon)
            except Exception as e:
                logger.debug("[commit/MB] %s", e)
        if wiki_root and loc and lat and lon:
            try:
                from biotrace_wiki import BioTraceWiki
                BioTraceWiki(wiki_root).update_locality_coords(loc, lat, lon)
            except Exception as e:
                logger.debug("[commit/Wiki] %s", e)

    logger.info("[staging] %d records committed to main DB", committed)
    return committed


# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

def render_hitl_staging_tab(
    meta_db_path: str,
    kg_db_path: str = "",
    mb_db_path: str = "",
    wiki_root: str = "",
):
    """
    Render the HITL staging tab in Streamlit.

    This tab never loses state across reruns because all state lives in SQLite.
    Each row can be independently edited, deleted, geocoded, and re-verified.
    Records only reach the main DB after an explicit "Commit" action.
    """
    import streamlit as st

    ensure_staging_table(meta_db_path)

    st.subheader("🔬 Staging Review — Verify Before Committing")
    st.caption(
        "All changes write directly to SQLite staging. "
        "Nothing reaches the main database until you click **Commit**. "
        "Closing and reopening this tab continues where you left off."
    )

    # ── Stats bar ─────────────────────────────────────────────────────────
    stats = _get_stats(meta_db_path)
    total = sum(v for k, v in stats.items() if k != _STATUS_COMMITTED)
    col_s = st.columns(5)
    col_s[0].metric("📋 Total staged",  total)
    col_s[1].metric("⏳ Pending",       stats.get(_STATUS_PENDING, 0))
    col_s[2].metric("✅ Verified",      stats.get(_STATUS_VERIFIED, 0))
    col_s[3].metric("✏️ Edited",        stats.get(_STATUS_EDITED, 0))
    col_s[4].metric("❌ Rejected",      stats.get(_STATUS_REJECTED, 0))

    # ── View + Filter controls ─────────────────────────────────────────────
    col_v1, col_v2, col_v3 = st.columns([2, 2, 1])
    with col_v1:
        filter_status = st.selectbox(
            "Filter by status:",
            ["all (non-committed)", _STATUS_PENDING, _STATUS_VERIFIED,
             _STATUS_EDITED, _STATUS_REJECTED],
            key="hitl_filter_status",
        )
    with col_v2:
        filter_text = st.text_input("🔍 Filter name / locality:", key="hitl_filter_text")
    with col_v3:
        if st.button("🔄 Refresh list", key="hitl_refresh"):
            # No rerun needed — load below always fetches fresh from SQLite
            pass

    # ── Load records ──────────────────────────────────────────────────────
    status_arg = None if filter_status.startswith("all") else filter_status
    records = _load_staging(meta_db_path, filter_status=status_arg)

    if filter_text:
        ft = filter_text.lower()
        records = [
            r for r in records
            if ft in str(r.get("recorded_name", "")).lower()
            or ft in str(r.get("valid_name", "")).lower()
            or ft in str(r.get("verbatim_locality", "")).lower()
        ]

    if not records:
        st.info("No records in this view. Adjust the filter or run extraction first.")
        return

    st.caption(f"Showing **{len(records)}** records")

    # ── BULK EDIT via data_editor ─────────────────────────────────────────
    with st.expander("📝 Bulk edit table (edit cells directly, then Save changes)", expanded=False):
        df_bulk = pd.DataFrame([{
            "id":               r["id"],
            "recorded_name":    r["recorded_name"] or "",
            "valid_name":       r["valid_name"] or "",
            "taxon_rank":       r["taxon_rank"] or "",
            "taxonomic_status": r["taxonomic_status"] or "",
            "verbatim_locality":r["verbatim_locality"] or "",
            "lat":              r["decimal_latitude"],
            "lon":              r["decimal_longitude"],
            "occurrence_type":  r["occurrence_type"] or "",
            "phylum":           r["phylum"] or "",
            "family":           r["family_"] or "",
            "worms_id":         r["worms_id"] or "",
            "confidence":       round(r.get("unified_confidence") or 0.0, 2),
            "status":           r["status"] or "",
        } for r in records])

        edited_df = st.data_editor(
            df_bulk,
            column_config={
                "id":          st.column_config.NumberColumn("ID", disabled=True, width="small"),
                "confidence":  st.column_config.ProgressColumn("Conf.", min_value=0, max_value=1, format="%.2f"),
                "status": st.column_config.SelectboxColumn(
                    "Status", options=[_STATUS_PENDING, _STATUS_VERIFIED, _STATUS_EDITED, _STATUS_REJECTED]
                ),
                "occurrence_type": st.column_config.SelectboxColumn(
                    "Type", options=["Primary", "Secondary", "Uncertain", ""]
                ),
            },
            use_container_width=True,
            hide_index=True,
            key="hitl_bulk_editor",
        )

        if st.button("💾 Save bulk edits to staging DB", key="hitl_bulk_save"):
            conn = sqlite3.connect(meta_db_path)
            saved = 0
            for _, row in edited_df.iterrows():
                rid = int(row["id"])
                orig = next((r for r in records if r["id"] == rid), None)
                if orig is None:
                    continue
                updates = {}
                field_map = {
                    "recorded_name": "recorded_name",
                    "valid_name": "valid_name",
                    "taxon_rank": "taxon_rank",
                    "taxonomic_status": "taxonomic_status",
                    "verbatim_locality": "verbatim_locality",
                    "lat": "decimal_latitude",
                    "lon": "decimal_longitude",
                    "occurrence_type": "occurrence_type",
                    "phylum": "phylum",
                    "family": "family_",
                    "worms_id": "worms_id",
                    "status": "status",
                }
                for col, db_col in field_map.items():
                    new_val = row.get(col)
                    old_val = orig.get(db_col if db_col != "status" else "status")
                    if str(new_val) != str(old_val):
                        updates[db_col] = new_val
                if updates:
                    new_status = updates.pop("status", None)
                    if updates:
                        set_cl = ", ".join(f"{k}=?" for k in updates)
                        vals = list(updates.values()) + [
                            new_status or _STATUS_EDITED, rid
                        ]
                        conn.execute(
                            f"UPDATE verification_staging SET {set_cl}, status=? WHERE id=?", vals
                        )
                    elif new_status:
                        conn.execute(
                            "UPDATE verification_staging SET status=? WHERE id=?",
                            (new_status, rid)
                        )
                    saved += 1
            conn.commit()
            conn.close()
            st.success(f"✅ Saved {saved} row(s) to staging.")

    st.divider()

    # ── Per-row cards ─────────────────────────────────────────────────────
    st.markdown("### Per-record actions")
    st.caption("Expand each record to edit fields, geocode, re-verify, or delete.")

    for rec in records:
        rid      = rec["id"]
        rname    = rec.get("recorded_name") or "Unknown"
        vname    = rec.get("valid_name") or ""
        locality = rec.get("verbatim_locality") or ""
        rstatus  = rec.get("status") or _STATUS_PENDING
        conf     = rec.get("unified_confidence") or 0.0

        # Status icon
        icon_map = {
            _STATUS_VERIFIED: "✅",
            _STATUS_EDITED:   "✏️",
            _STATUS_REJECTED: "❌",
            _STATUS_PENDING:  "⏳",
        }
        icon = icon_map.get(rstatus, "❓")

        label = (f"{icon} #{rid} — *{rname}*"
                 + (f" → {vname}" if vname and vname != rname else "")
                 + f"  |  {locality}"
                 + f"  |  conf={conf:.2f}")

        with st.expander(label, expanded=False):
            # ── Quick-action buttons ─────────────────────────────────────
            qcols = st.columns(4)
            with qcols[0]:
                if st.button("✅ Verify", key=f"verify_{rid}"):
                    _set_status(meta_db_path, rid, _STATUS_VERIFIED)
                    st.toast(f"#{rid} marked verified")
            with qcols[1]:
                if st.button("❌ Reject", key=f"reject_{rid}"):
                    _set_status(meta_db_path, rid, _STATUS_REJECTED)
                    st.toast(f"#{rid} rejected")
            with qcols[2]:
                if st.button("🗑️ Delete", key=f"del_{rid}"):
                    st.session_state[f"_cdel_{rid}"] = True
            with qcols[3]:
                if st.button("🔬 Re-verify WoRMS", key=f"reverify_{rid}"):
                    with st.spinner("Querying WoRMS…"):
                        wres = _reverify_worms(vname or rname)
                    if wres:
                        _update_fields(meta_db_path, rid, {
                            "valid_name":       wres.get("valid_name", vname),
                            "worms_id":         wres.get("worms_id", ""),
                            "phylum":           wres.get("phylum", ""),
                            "class_":           wres.get("class_", ""),
                            "order_":           wres.get("order_", ""),
                            "family_":          wres.get("family_", ""),
                            "taxonomic_status": wres.get("taxonomic_status", "unverified"),
                        }, status=_STATUS_EDITED)
                        st.success(f"WoRMS: {wres.get('valid_name','—')}")
                    else:
                        st.warning("No WoRMS match found.")

            # Confirm delete dialog (survives rerun via session_state —
            # only a flag, not data, so safe)
            if st.session_state.get(f"_cdel_{rid}"):
                st.error(f"Permanently delete **#{rid} {rname}** from staging?")
                dcols = st.columns(2)
                with dcols[0]:
                    if st.button("⚠️ Confirm delete", key=f"cdel_conf_{rid}", type="primary"):
                        _delete_staging(meta_db_path, rid)
                        st.session_state.pop(f"_cdel_{rid}", None)
                        st.toast(f"Deleted staging #{rid}")
                with dcols[1]:
                    if st.button("Cancel", key=f"cdel_cancel_{rid}"):
                        st.session_state.pop(f"_cdel_{rid}", None)

            # ── Inline edit fields ───────────────────────────────────────
            st.markdown("**Edit fields:**")
            ec1, ec2 = st.columns(2)
            with ec1:
                new_recorded = st.text_input("Recorded name", value=rname, key=f"rn_{rid}")
                new_valid    = st.text_input("Valid name",    value=vname, key=f"vn_{rid}")
                new_type     = st.selectbox("Occurrence type",
                                            ["Primary","Secondary","Uncertain",""],
                                            index=["Primary","Secondary","Uncertain",""].index(
                                                rec.get("occurrence_type","") or ""),
                                            key=f"ot_{rid}")
                new_notes    = st.text_input("Notes", value=rec.get("notes","") or "", key=f"notes_{rid}")
            with ec2:
                new_locality = st.text_input("Locality", value=locality, key=f"loc_{rid}")
                new_lat      = st.text_input("Latitude",  value=str(rec.get("decimal_latitude") or ""),  key=f"lat_{rid}")
                new_lon      = st.text_input("Longitude", value=str(rec.get("decimal_longitude") or ""), key=f"lon_{rid}")
                new_worms    = st.text_input("WoRMS ID",  value=rec.get("worms_id","") or "", key=f"wid_{rid}")

            if st.button("💾 Save edits", key=f"save_{rid}"):
                updates = {
                    "recorded_name":    new_recorded,
                    "valid_name":       new_valid,
                    "occurrence_type":  new_type,
                    "verbatim_locality":new_locality,
                    "notes":            new_notes,
                    "worms_id":         new_worms,
                }
                # Parse coordinates
                try:
                    if new_lat.strip():
                        updates["decimal_latitude"]  = float(new_lat)
                        updates["geocoding_source"]  = "Manual_HITL"
                    if new_lon.strip():
                        updates["decimal_longitude"] = float(new_lon)
                except ValueError:
                    st.error("Invalid latitude/longitude — must be decimal numbers.")
                    continue
                _update_fields(meta_db_path, rid, updates)
                st.toast(f"✏️ #{rid} saved")

            # ── Geocoding sub-section ────────────────────────────────────
            with st.expander("📍 Geocode this locality"):
                geo_query = st.text_input(
                    "Nominatim query (auto-filled from locality):",
                    value=new_locality,
                    key=f"geoq_{rid}"
                )
                if st.button("🔍 Search Nominatim", key=f"geo_search_{rid}"):
                    with st.spinner("Querying Nominatim (1 req/s)…"):
                        sug = _nominatim_geocode(geo_query)
                    if sug:
                        st.success(
                            f"📍 {sug['display_name'][:80]}\n\n"
                            f"Lat `{sug['lat']:.5f}` · Lon `{sug['lon']:.5f}`"
                        )
                        if "," not in geo_query:
                            st.map(pd.DataFrame([{"lat": sug["lat"], "lon": sug["lon"]}]), zoom=7)
                        if st.button("✅ Accept these coordinates", key=f"geo_accept_{rid}"):
                            _update_fields(meta_db_path, rid, {
                                "decimal_latitude":  round(sug["lat"], 6),
                                "decimal_longitude": round(sug["lon"], 6),
                                "geocoding_source":  "Nominatim_HITL",
                            })
                            st.toast(f"Coordinates saved for #{rid}")
                    else:
                        st.warning("No result — try a broader query (e.g. add ', India').")

            # ── Raw evidence preview ────────────────────────────────────
            evidence = rec.get("raw_text_evidence", "")
            if evidence and len(evidence) > 10:
                with st.expander("📄 Raw text evidence"):
                    st.caption(str(evidence)[:800])

    # ── Commit section ────────────────────────────────────────────────────
    st.divider()
    st.subheader("📤 Commit to main database")

    v_count = stats.get(_STATUS_VERIFIED, 0) + stats.get(_STATUS_EDITED, 0)
    st.info(
        f"**{v_count}** verified/edited records ready to commit. "
        f"Pending: {stats.get(_STATUS_PENDING, 0)} | "
        f"Rejected: {stats.get(_STATUS_REJECTED, 0)}"
    )

    sync_targets = []
    if kg_db_path: sync_targets.append("Knowledge Graph")
    if mb_db_path: sync_targets.append("Memory Bank")
    if wiki_root:  sync_targets.append("Wiki")
    if sync_targets:
        st.caption(f"Will also sync to: **{', '.join(sync_targets)}**")

    if v_count == 0:
        st.warning("No verified/edited records to commit. Mark records as Verified first.")
    else:
        if st.button(
            f"🚀 Commit {v_count} records → Main DB",
            type="primary",
            key="hitl_commit_btn",
        ):
            with st.spinner("Committing to main database…"):
                committed = commit_verified_to_main(
                    db_path    = meta_db_path,
                    kg_db_path = kg_db_path,
                    mb_db_path = mb_db_path,
                    wiki_root  = wiki_root,
                )
            st.success(
                f"✅ {committed} records committed to main database"
                + (f" and synced to {', '.join(sync_targets)}" if sync_targets else "")
                + "."
            )

    # ── Danger zone: clear staging ────────────────────────────────────────
    with st.expander("⚠️ Danger zone"):
        st.warning("These actions cannot be undone.")
        if st.button("🗑️ Clear ALL rejected records from staging", key="hitl_clear_rejected"):
            conn = sqlite3.connect(meta_db_path)
            n = conn.execute(
                "DELETE FROM verification_staging WHERE status=?", (_STATUS_REJECTED,)
            ).rowcount
            conn.commit(); conn.close()
            st.success(f"Removed {n} rejected records.")

        if st.button("♻️ Reset pending records (re-stage from scratch)", key="hitl_reset_pending"):
            conn = sqlite3.connect(meta_db_path)
            conn.execute(
                "DELETE FROM verification_staging WHERE status=?", (_STATUS_PENDING,)
            )
            conn.commit(); conn.close()
            st.success("Pending staging records cleared. Re-run extraction to re-stage.")
