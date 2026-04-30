"""
biotrace_wiki_v56.py  —  BioTrace v5.6  Enhanced Wiki Module
═══════════════════════════════════════════════════════════════════════════
Drop-in subclass of BioTraceWikiUnified.  In biotrace_v53.py replace:

    from biotrace_wiki_unified import BioTraceWikiUnified, inject_css_streamlit

with:

    from biotrace_wiki_v56 import BioTraceWikiV56 as BioTraceWikiUnified, inject_css_streamlit

NEW in v5.6
───────────
1.  Dynamic cascading taxonomic filter (Kingdom→Phylum→Class→Order→Family→Genus)
    Populated live from occurrence DB; resets on HITL staleness flag.

2.  Occurrence records fetched LIVE from occurrence DB.
    Any HITL coordinate / locality edit is immediately reflected in the wiki
    map and occurrence table — no article re-extraction needed.

3.  Folium MarkerCluster map with colour-coded occurrence types
    (Primary=green, Secondary=blue, Uncertain=amber) + legend + MiniMap.
    Falls back to st.map when folium is unavailable.

4.  Full references — [:60] truncation removed from occurrence source strings
    and provenance citations throughout render_unified_page() / _to_markdown().

5.  Verification panel with WoRMS + GBIF lookup (marine_only=false → ALL taxa)
    Editable Classification fields; one "Save & Sync" propagates to:
      • Occurrence DB (SQLite)
      • Knowledge Graph
      • Memory Bank
      • Wiki article (versioned)

6.  All taxa — system prompt updated; WoRMS marine_only=false by default.

7.  Recorded name vs valid name both shown in occurrence table.

8.  DoclingWikiBridge wired by default (no separate wiki-agent run needed).

9.  MD caching hooks (see biotrace_md_cache.py).

10. HITL taxonomy fields (class, order, family) now populate correctly.
    See biotrace_hitl_v56_patch.py for the complementary HITL upgrade.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

logger = logging.getLogger("biotrace.wiki_v56")

# ── Optional deps ──────────────────────────────────────────────────────────────
try:
    import folium
    from folium.plugins import MarkerCluster, MiniMap
    _FOLIUM = True
except ImportError:
    _FOLIUM = False

try:
    import requests as _req
    _REQUESTS = True
except ImportError:
    _REQUESTS = False

try:
    import streamlit as st
    _ST = True
except ImportError:
    _ST = False

# ── Base class ─────────────────────────────────────────────────────────────────
from biotrace_wiki_unified import BioTraceWikiUnified, inject_css_streamlit


# ══════════════════════════════════════════════════════════════════════════════
#  TAXONOMY LOOKUPS — WoRMS (marine_only=false) + GBIF fallback
# ══════════════════════════════════════════════════════════════════════════════

def _lookup_worms(species_name: str) -> Optional[dict]:
    """
    WoRMS REST API — marine_only=false so terrestrial / freshwater taxa
    registered in WoRMS are also returned.
    """
    if not _REQUESTS:
        return None
    enc = urllib.parse.quote(species_name.strip())
    url = (f"https://www.marinespecies.org/rest/AphiaRecordsByName/{enc}"
           f"?like=false&marine_only=false&offset=0")
    try:
        resp = _req.get(url, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, list):
                r = data[0]
                return {
                    "kingdom":         r.get("kingdom", ""),
                    "phylum":          r.get("phylum",  ""),
                    "class_":          r.get("class",   ""),
                    "order_":          r.get("order",   ""),
                    "family_":         r.get("family",  ""),
                    "genus":           r.get("genus",   ""),
                    "authority":       r.get("authority", ""),
                    "taxonRank":       (r.get("rank",   "species") or "species").lower(),
                    "taxonomicStatus": (r.get("status", "accepted") or "accepted").lower(),
                    "wormsID":         str(r.get("AphiaID", "")),
                    "validName":       r.get("valid_name", "") or species_name,
                    "source":          "WoRMS",
                }
    except Exception as exc:
        logger.warning("[WikiV56/WoRMS] %s: %s", species_name, exc)
    return None


def _lookup_gbif(species_name: str) -> Optional[dict]:
    """GBIF species match — covers all kingdoms (plants, fungi, animals, etc.)."""
    if not _REQUESTS:
        return None
    url = (f"https://api.gbif.org/v1/species/match"
           f"?name={urllib.parse.quote(species_name.strip())}&verbose=true")
    try:
        resp = _req.get(url, timeout=12)
        if resp.status_code == 200:
            r = resp.json()
            if r.get("matchType", "NONE") != "NONE":
                return {
                    "kingdom":         r.get("kingdom", ""),
                    "phylum":          r.get("phylum",  ""),
                    "class_":          r.get("class",   ""),
                    "order_":          r.get("order",   ""),
                    "family_":         r.get("family",  ""),
                    "genus":           r.get("genus",   ""),
                    "authority":       r.get("authorship", ""),
                    "taxonRank":       (r.get("rank",   "SPECIES") or "SPECIES").lower(),
                    "taxonomicStatus": (r.get("status", "ACCEPTED") or "ACCEPTED").lower(),
                    "wormsID":         "",
                    "gbifID":          str(r.get("usageKey", "")),
                    "validName":       r.get("species", "") or species_name,
                    "source":          "GBIF",
                }
    except Exception as exc:
        logger.warning("[WikiV56/GBIF] %s: %s", species_name, exc)
    return None


def _lookup_taxonomy(species_name: str) -> Optional[dict]:
    """Try WoRMS first, fall back to GBIF."""
    return _lookup_worms(species_name) or _lookup_gbif(species_name)


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE OCCURRENCE FETCH FROM DB
# ══════════════════════════════════════════════════════════════════════════════

_OCC_COLS = [
    "id", "validName", "recordedName", "verbatimLocality",
    "decimalLatitude", "decimalLongitude", "occurrenceType",
    "sourceCitation", "habitat", "depthRange", "eventDate",
    "phylum", "class_", "order_", "family_", "genus",
    "wormsID", "geocodingSource", "taxonomicStatus", "matchScore",
]
_TAX_COLS = ["kingdom", "phylum", "class_", "order_", "family_", "genus", "validName"]


def _resolve_occ_table(conn: sqlite3.Connection) -> Optional[str]:
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    return next((t for t in ("occurrences_v4", "occurrences") if t in names), None)


def _fetch_live_occurrences(meta_db_path: str, species_name: str) -> list[dict]:
    """
    Fetch occurrence records directly from SQLite — always reflects latest
    HITL edits (coordinates, locality, taxonomy).
    """
    if not meta_db_path or not Path(meta_db_path).exists():
        return []
    try:
        conn  = sqlite3.connect(meta_db_path)
        table = _resolve_occ_table(conn)
        if not table:
            conn.close(); return []
        avail = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        sel   = [c for c in _OCC_COLS if c in avail]
        if not sel:
            conn.close(); return []
        rows = conn.execute(
            f"SELECT {', '.join(sel)} FROM {table} "
            "WHERE (validName=? OR recordedName=?) "
            "AND (validationStatus IS NULL OR validationStatus != 'rejected') "
            "ORDER BY id",
            (species_name, species_name),
        ).fetchall()
        conn.close()
        results = []
        for row in rows:
            d = dict(zip(sel, row))
            for k in ("decimalLatitude", "decimalLongitude"):
                try:
                    d[k] = float(d[k]) if d.get(k) is not None else None
                except (TypeError, ValueError):
                    d[k] = None
            results.append(d)
        return results
    except Exception as exc:
        logger.warning("[WikiV56] live_occurrences(%s): %s", species_name, exc)
        return []


def _fetch_taxa_hierarchy_raw(meta_db_path: str) -> pd.DataFrame:
    """Return unique taxonomy rows from occurrence DB for cascade filter."""
    if not meta_db_path or not Path(meta_db_path).exists():
        return pd.DataFrame(columns=_TAX_COLS)
    try:
        conn  = sqlite3.connect(meta_db_path)
        table = _resolve_occ_table(conn)
        if not table:
            conn.close()
            return pd.DataFrame(columns=_TAX_COLS)
        avail = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        sel   = [c for c in _TAX_COLS if c in avail]
        if not sel:
            conn.close()
            return pd.DataFrame(columns=_TAX_COLS)
        df = pd.read_sql_query(
            f"SELECT DISTINCT {', '.join(sel)} FROM {table} WHERE validName IS NOT NULL",
            conn,
        )
        conn.close()
        return df.fillna("")
    except Exception as exc:
        logger.warning("[WikiV56] taxa_hierarchy: %s", exc)
        return pd.DataFrame(columns=_TAX_COLS)


# ══════════════════════════════════════════════════════════════════════════════
#  FOLIUM MAP (colour-coded by occurrence type)
# ══════════════════════════════════════════════════════════════════════════════

_OCC_COLOURS = {
    "Primary":   "#3fb950",
    "Secondary": "#58a6ff",
    "Uncertain": "#e3b341",
}
_OCC_ICON_COLOURS = {
    "Primary":   "green",
    "Secondary": "blue",
    "Uncertain": "orange",
}


def _render_folium_map(occurrences: list[dict], species_name: str):
    """Render Folium MarkerCluster map.  Falls back to st.map if folium absent."""
    if not _ST:
        return
    pts = [o for o in occurrences
           if o.get("decimalLatitude") is not None
           and o.get("decimalLongitude") is not None]
    if not pts:
        st.info("No geocoded occurrence records for this species yet.")
        return

    lats   = [p["decimalLatitude"]  for p in pts]
    lons   = [p["decimalLongitude"] for p in pts]
    centre = (sum(lats) / len(lats), sum(lons) / len(lons))

    if _FOLIUM:
        m       = folium.Map(location=centre, zoom_start=5, tiles="CartoDB dark_matter")
        cluster = MarkerCluster(name="Occurrences").add_to(m)
        MiniMap(toggle_display=True).add_to(m)

        for p in pts:
            occ_t  = p.get("occurrenceType", "Uncertain") or "Uncertain"
            colour = _OCC_COLOURS.get(occ_t, "#9e9e9e")
            popup  = folium.Popup(
                f"<b><i>{species_name}</i></b><br>"
                f"<b>Locality:</b> {p.get('verbatimLocality','—')}<br>"
                f"<b>Recorded as:</b> <i>{p.get('recordedName','—')}</i><br>"
                f"<b>Type:</b> {occ_t}<br>"
                f"<b>Source:</b> {p.get('sourceCitation','—')}<br>"
                f"<b>Depth:</b> {p.get('depthRange','—')}<br>"
                f"<b>Habitat:</b> {p.get('habitat','—')}<br>"
                f"<b>Record ID:</b> #{p.get('id','')}",
                max_width=340,
            )
            folium.CircleMarker(
                location=(p["decimalLatitude"], p["decimalLongitude"]),
                radius=7,
                color=colour, fill=True, fill_color=colour, fill_opacity=0.85,
                popup=popup,
                tooltip=p.get("verbatimLocality", ""),
            ).add_to(cluster)

        # Legend overlay
        legend = (
            "<div style='position:fixed;bottom:40px;left:40px;z-index:9999;"
            "background:#1c2433;padding:10px 14px;border-radius:8px;"
            "font-family:sans-serif;font-size:12px;color:#e6edf3;'>"
            "<b>Occurrence Type</b><br>"
            "<span style='color:#3fb950'>●</span> Primary<br>"
            "<span style='color:#58a6ff'>●</span> Secondary<br>"
            "<span style='color:#e3b341'>●</span> Uncertain</div>"
        )
        m.get_root().html.add_child(folium.Element(legend))
        st.components.v1.html(m._repr_html_(), height=480, scrolling=False)
    else:
        # Fallback
        map_df = pd.DataFrame(
            [{"lat": p["decimalLatitude"], "lon": p["decimalLongitude"]} for p in pts]
        )
        st.map(map_df, zoom=4)


# ══════════════════════════════════════════════════════════════════════════════
#  SYNC CLASSIFICATION → ALL FOUR DATABASES
# ══════════════════════════════════════════════════════════════════════════════

def _sync_classification(
    species_name:  str,
    classification: dict,
    meta_db_path:  str,
    kg_db_path:    str = "",
    mb_db_path:    str = "",
    wiki_obj:      Optional["BioTraceWikiV56"] = None,
) -> dict[str, bool]:
    """
    Propagate verified classification to ALL four stores.
    Returns {store: success_bool}.
    """
    results: dict[str, bool] = {}

    # ── 1. Occurrence SQLite DB ───────────────────────────────────────────────
    try:
        conn  = sqlite3.connect(meta_db_path)
        table = _resolve_occ_table(conn)
        if table:
            avail = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            field_map = {
                "kingdom": "kingdom", "phylum": "phylum", "class_": "class_",
                "order_": "order_", "family_": "family_", "genus": "genus",
                "authority": "authority", "wormsID": "wormsID",
                "gbifID": "gbifID", "iucnStatus": "iucnStatus",
                "taxonomicStatus": "taxonomicStatus",
            }
            updates = {dst: classification[src]
                       for src, dst in field_map.items()
                       if dst in avail and classification.get(src) is not None}
            if updates:
                set_clause = ", ".join(f"{k}=?" for k in updates)
                conn.execute(
                    f"UPDATE {table} SET {set_clause} WHERE validName=?",
                    list(updates.values()) + [species_name],
                )
                conn.commit()
        conn.close()
        results["occurrence_db"] = True
    except Exception as exc:
        logger.warning("[WikiV56/sync] occurrence_db: %s", exc)
        results["occurrence_db"] = False

    # ── 2. Knowledge Graph ────────────────────────────────────────────────────
    if kg_db_path and Path(kg_db_path).exists():
        try:
            from biotrace_knowledge_graph import BioTraceKnowledgeGraph, _node_id, _now
            kg  = BioTraceKnowledgeGraph(kg_db_path)
            nid = _node_id("Species", species_name)
            row = kg._conn.execute(
                "SELECT properties FROM kg_nodes WHERE node_id=?", (nid,)
            ).fetchone()
            if row:
                props = json.loads(row[0] or "{}")
                props.update({k: v for k, v in classification.items() if v})
                kg._conn.execute(
                    "UPDATE kg_nodes SET properties=?, updated_at=? WHERE node_id=?",
                    (json.dumps(props), _now(), nid),
                )
                if kg._G.has_node(nid):
                    kg._G.nodes[nid].update(props)
                kg._conn.commit()
            kg.close()
            results["knowledge_graph"] = True
        except Exception as exc:
            logger.warning("[WikiV56/sync] KG: %s", exc)
            results["knowledge_graph"] = False

    # ── 3. Memory Bank ────────────────────────────────────────────────────────
    if mb_db_path and Path(mb_db_path).exists():
        try:
            conn  = sqlite3.connect(mb_db_path)
            avail = {r[1] for r in conn.execute(
                "PRAGMA table_info(memory_atoms)"
            ).fetchall()}
            for field in ("phylum", "class_", "order_", "family_", "kingdom"):
                if field in avail and classification.get(field):
                    conn.execute(
                        f"UPDATE memory_atoms SET {field}=? WHERE valid_name=?",
                        (classification[field], species_name),
                    )
            conn.commit(); conn.close()
            results["memory_bank"] = True
        except Exception as exc:
            logger.warning("[WikiV56/sync] MB: %s", exc)
            results["memory_bank"] = False

    # ── 4. Wiki article ───────────────────────────────────────────────────────
    if wiki_obj:
        try:
            art = wiki_obj.get_species_article(species_name) or {}
            if art:
                art.update({k: v for k, v in classification.items()
                             if v is not None and k != "source"})
                wiki_obj._write(
                    "species", wiki_obj._slug(species_name), species_name, art,
                    change_note=(
                        f"Classification verified via "
                        f"{classification.get('source','manual')}"
                    ),
                )
                results["wiki"] = True
            else:
                results["wiki"] = False
        except Exception as exc:
            logger.warning("[WikiV56/sync] wiki: %s", exc)
            results["wiki"] = False

    # Broadcast staleness flags
    try:
        st.session_state.update({
            "occ_map_stale":    True,
            "data_table_stale": True,
            "kg_dirty":         True,
            "wiki_stale":       True,
        })
    except Exception:
        pass

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PATCHED _to_markdown — removes [:60] truncation on source / citation
# ══════════════════════════════════════════════════════════════════════════════

def _to_markdown_v56(art: dict) -> str:
    """
    Drop-in replacement for BioTraceWikiUnified._to_markdown().
    Fixes: full-length citations (no [:60] truncation),
           recorded_name shown alongside valid_name in occurrence list.
    """
    lines: list[str] = []
    title   = art.get("title", "Unknown")
    auth    = art.get("authority", "")
    rank    = art.get("taxonRank", "species")
    kingdom = art.get("kingdom", "")
    phylum  = art.get("phylum", "")
    cls     = art.get("class_", "")
    order   = art.get("order_", "")
    family  = art.get("family_", "")
    genus   = art.get("genus", "")

    lines.extend([f"# *{title}*{' ' + auth if auth else ''}", ""])
    if any([kingdom, phylum, cls, order, family]):
        lines.append(
            f"**Kingdom:** {kingdom} | **Phylum:** {phylum} | "
            f"**Class:** {cls} | **Order:** {order} | **Family:** {family}"
        )
        lines.append("")

    secs = art.get("sections", {})
    for key, heading in [
        ("lead",                "## Overview"),
        ("taxonomy_phylogeny",  "## Taxonomy & Phylogeny"),
        ("morphology",          "## Morphology"),
        ("distribution_habitat","## Distribution & Habitat"),
        ("ecology_behaviour",   "## Ecology & Behaviour"),
        ("conservation",        "## Conservation"),
        ("specimen_records",    "## Specimen Records"),
    ]:
        txt = secs.get(key, "")
        if txt:
            lines.extend([heading, "", txt, ""])

    # Occurrence summary — full citation, recorded name shown
    occ_pts = art.get("occurrence_points", [])
    if occ_pts:
        lines.extend(["## Documented Occurrences", ""])
        for pt in occ_pts[:50]:
            loc     = pt.get("locality", "—")
            ot      = pt.get("occurrenceType", "?")
            src     = str(pt.get("source", ""))          # ← NO [:60] truncation
            rec_n   = pt.get("recordedName", "")
            dep     = pt.get("depth_m")
            dep_s   = f" · {dep} m depth" if dep else ""
            rec_s   = f" [recorded as *{rec_n}*]" if rec_n and rec_n != pt.get("validName","") else ""
            lines.append(f"- **{loc}** ({ot}{dep_s}){rec_s} — _{src}_")
        lines.append("")

    # References — full citations, no truncation
    provs = art.get("provenance", [])
    if provs:
        lines.extend(["## References", ""])
        for i, p in enumerate(provs, 1):
            enhanced = " ✨" if p.get("enhanced") else ""
            citation = str(p.get("citation", ""))          # ← FULL citation
            date_str = str(p.get("date", ""))[:10]
            lines.append(f"{i}. {citation} ({date_str}){enhanced}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENHANCED WIKI CLASS
# ══════════════════════════════════════════════════════════════════════════════

class BioTraceWikiV56(BioTraceWikiUnified):
    """
    BioTrace v5.6 Wiki — all v5.5 features plus the 10 enhancements above.
    """

    # Patch _to_markdown at class level so render_unified_page() also benefits
    _to_markdown = staticmethod(_to_markdown_v56)

    # ── System prompt: all taxa (not marine-only) ──────────────────────────────
    _WIKI_ARCHITECT_SYSTEM_V56 = """\
You are a Professional Taxonomist and Wiki Editor with expertise in all clades
of life — marine invertebrates, vertebrates, plants, fungi, freshwater taxa,
terrestrial arthropods, and microbial eukaryotes.

Your task: given (a) the CURRENT wiki article JSON and (b) NEW source text
(a PDF extract / chunk), produce an UPDATED article JSON that:

1. NEVER deletes existing valid data — only appends or refines.
2. Resolves conflicts by listing BOTH sources with inline citations, e.g.
   "Bhave (2011) reports 5 m; Smith (2024) reports 12 m."
3. Fills in blank fields (authority, taxon rank, synonyms, depth, etc.)
   when the new text provides them.
4. Appends new localities, vouchers, and ecological notes.
5. Updates the "sections" dict — each key maps to a markdown string.
6. Uses italics (*Name*) for binomial nomenclature.
7. Uses **bold** for key technical terms.
8. Respects Wikipedia-style neutral tone.

Return ONLY a valid JSON object — no prose, no markdown fences.
The JSON must follow EXACTLY the schema of the input CURRENT article.
"""

    # ── Rendering helper — called by render_unified_page ──────────────────────
    # (overrides parent to fix [:60] and add recorded_name display)

    def render_streamlit_tab(
        self,
        provider:    str = "",
        model_sel:   str = "",
        api_key:     str = "",
        ollama_url:  str = "http://localhost:11434",
        meta_db:     str = "",
        call_llm_fn: Optional[Callable] = None,
        kg_db_path:  str = "",
        mb_db_path:  str = "",
    ):
        """Full Tab 7 replacement — call inside  `with tabs[6]:`."""
        if not _ST:
            return

        inject_css_streamlit(self.css_path)
        st.subheader("📖 Wiki — Universal Biodiversity Knowledge Base")
        st.caption("Wikipedia-style · LLM-enhanced · Git-versioned · All taxa · Live DB-linked occurrences")

        # ── Index metrics ───────────────────────────────────────────────────────
        idx = self.index_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Articles",  idx["total_articles"])
        c2.metric("Species",         idx["by_section"].get("species",  0))
        c3.metric("Localities",      idx["by_section"].get("locality", 0))
        c4.metric("Saved Versions",  self._count_versions())

        # Staleness flag → clear taxa cache
        if st.session_state.get("wiki_stale"):
            st.toast("📡 DB updated — taxonomic filter refreshed")
            st.session_state["wiki_stale"] = False
            if "_taxa_cache" in st.session_state:
                del st.session_state["_taxa_cache"]

        st.divider()

        # ── Fetch / cache taxa hierarchy (120 s) ───────────────────────────────
        cache_key = f"_taxa_cache_{meta_db}"
        taxa_df: pd.DataFrame
        if cache_key not in st.session_state:
            taxa_df = _fetch_taxa_hierarchy_raw(meta_db) if meta_db else pd.DataFrame()
            st.session_state[cache_key] = taxa_df
        else:
            taxa_df = st.session_state[cache_key]

        # ── Cascading taxonomic filter ──────────────────────────────────────────
        with st.expander("🔬 Taxonomic Filter (cascading — populated from DB)", expanded=False):
            def _opts(col: str, df: pd.DataFrame) -> list[str]:
                if df.empty or col not in df.columns:
                    return []
                return sorted(x for x in df[col].dropna().unique() if str(x).strip())

            filt_df = taxa_df.copy() if not taxa_df.empty else pd.DataFrame(columns=_TAX_COLS)

            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                sel_kingdom = st.multiselect("Kingdom", _opts("kingdom", filt_df),
                                              key="wf56_kingdom")
            if sel_kingdom:
                filt_df = filt_df[filt_df["kingdom"].isin(sel_kingdom)]
            with r1c2:
                sel_phylum  = st.multiselect("Phylum", _opts("phylum", filt_df),
                                              key="wf56_phylum")
            if sel_phylum:
                filt_df = filt_df[filt_df["phylum"].isin(sel_phylum)]
            with r1c3:
                sel_class   = st.multiselect("Class", _opts("class_", filt_df),
                                              key="wf56_class")
            if sel_class:
                filt_df = filt_df[filt_df["class_"].isin(sel_class)]

            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                sel_order   = st.multiselect("Order", _opts("order_", filt_df),
                                              key="wf56_order")
            if sel_order:
                filt_df = filt_df[filt_df["order_"].isin(sel_order)]
            with r2c2:
                sel_family  = st.multiselect("Family", _opts("family_", filt_df),
                                              key="wf56_family")
            if sel_family:
                filt_df = filt_df[filt_df["family_"].isin(sel_family)]
            with r2c3:
                sel_genus   = st.multiselect("Genus", _opts("genus", filt_df),
                                              key="wf56_genus")
            if sel_genus:
                filt_df = filt_df[filt_df["genus"].isin(sel_genus)]

            active = any([sel_kingdom, sel_phylum, sel_class,
                          sel_order, sel_family, sel_genus])
            filtered_sp_set: Optional[set] = (
                set(filt_df["validName"].dropna().unique().tolist())
                if (active and not filt_df.empty) else None
            )
            if active:
                st.caption(
                    f"🔎 Filter matches **{len(filtered_sp_set or [])}** species in occurrence DB."
                )

        # ── Species list (apply filter + text search) ───────────────────────────
        sp_list = self.list_species()
        if filtered_sp_set is not None and sp_list:
            sp_list = [
                s for s in sp_list
                if s in filtered_sp_set
                or any(s.lower().startswith(n.split()[0].lower())
                       for n in filtered_sp_set)
            ]

        search_q = st.text_input("🔍 Search species name:", key="wiki_v56_search")
        if search_q:
            sp_list = [s for s in sp_list if search_q.lower() in s.lower()]

        if not sp_list:
            msg = ("No wiki articles match the current filter."
                   if active else
                   "No wiki articles yet — run an extraction to populate.")
            st.info(msg)
            return

        # ── Species selector ────────────────────────────────────────────────────
        def _strip_auth(name: str) -> str:
            return re.sub(
                r"\s+[A-Z][A-Za-z\-'']+(?:\s+(?:and|&|et)\s+[A-Z][A-Za-z\-'']+)?"
                r"[,.]?\s*\d{4}.*$",
                "",
                name,
            ).strip()

        display_map = {_strip_auth(s): s for s in sorted(sp_list)}
        selected_display = st.selectbox(
            f"Select species ({len(display_map)} matching):",
            sorted(display_map.keys()),
            key="wiki_v56_sp_sel",
        )
        selected_sp = display_map.get(selected_display, selected_display)
        if not selected_sp:
            return

        art = self.get_species_article(selected_sp) or {}

        # ── Status bar ──────────────────────────────────────────────────────────
        ts    = art.get("taxonomicStatus", "unverified")
        iucn  = art.get("iucnStatus", "")
        badge = "🟢" if ts == "verified" else "🟡" if ts == "unverified" else "🔴"
        ca, cb, cc, cd = st.columns(4)
        ca.markdown(f"{badge} **Status:** `{ts}`")
        cb.markdown(f"**Family:** *{art.get('family_','—')}*")
        cc.markdown(f"**WoRMS:** {art.get('wormsID','—')}")
        cd.markdown(f"**IUCN:** {iucn if iucn else '—'}")

        # ── Five view tabs ──────────────────────────────────────────────────────
        view_tab, map_tab, verify_tab, raw_tab, ver_tab = st.tabs([
            "📄 Wiki Page",
            "🗺️ Live Occurrence Map",
            "🔬 Verify & Classify",
            "🗂️ Raw Article",
            "📜 Version History",
        ])

        # Live occurrences — fetched once, shared across map + verify tabs
        live_occs = _fetch_live_occurrences(meta_db, selected_sp) if meta_db else []

        # ── TAB 1: Wiki Page ───────────────────────────────────────────────────
        with view_tab:
            html_page = self.render_unified_page(selected_sp)
            st.components.v1.html(html_page, height=920, scrolling=True)

        # ── TAB 2: Live Occurrence Map ─────────────────────────────────────────
        with map_tab:
            geocoded = [o for o in live_occs
                        if o.get("decimalLatitude") is not None]
            st.caption(
                f"**{len(live_occs)} total records** · "
                f"**{len(geocoded)} geocoded** · "
                "Live from occurrence DB — HITL edits reflected immediately."
            )
            _render_folium_map(live_occs, selected_sp)

            if live_occs:
                st.markdown("#### Occurrence Records (full detail)")
                occ_df   = pd.DataFrame(live_occs)
                show_col = [c for c in [
                    "id", "validName", "recordedName", "verbatimLocality",
                    "decimalLatitude", "decimalLongitude", "occurrenceType",
                    "sourceCitation", "depthRange", "habitat", "eventDate",
                    "phylum", "family_",
                ] if c in occ_df.columns]
                st.dataframe(occ_df[show_col], use_container_width=True, height=300)

        # ── TAB 3: Verify & Classify ───────────────────────────────────────────
        with verify_tab:
            self._render_verification_panel(
                art, selected_sp, meta_db, kg_db_path, mb_db_path
            )

        # ── TAB 4: Raw Article ─────────────────────────────────────────────────
        with raw_tab:
            st.json(art, expanded=False)
            if call_llm_fn:
                st.markdown("#### 🤖 LLM Wiki Enhancement")
                enhance_text = st.text_area(
                    "Paste PDF chunk / text to enhance this article:",
                    height=180, key="wiki_v56_enh_txt",
                )
                enhance_cite = st.text_input("Citation:", key="wiki_v56_enh_cite")
                if st.button("✨ Enhance Article", key="wiki_v56_enh_btn"):
                    if enhance_text.strip():
                        with st.spinner("Wiki Architect enhancing…"):
                            try:
                                self._enhance_with_llm(
                                    selected_sp, enhance_text, enhance_cite, call_llm_fn
                                )
                                st.success("Article enhanced and versioned ✅")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Enhancement failed: {exc}")
                    else:
                        st.warning("Paste some text first.")

        # ── TAB 5: Version History ─────────────────────────────────────────────
        with ver_tab:
            versions = self.list_versions("species", selected_sp)
            if not versions:
                st.info("No previous versions — article will be versioned after next update.")
            else:
                st.write(f"**{len(versions)} saved versions** (newest first):")
                st.dataframe(
                    pd.DataFrame(versions), use_container_width=True, hide_index=True
                )
                max_v  = max(v["version"] for v in versions)
                rb_ver = st.number_input(
                    "Rollback to version:", min_value=1, max_value=max_v,
                    step=1, key="wiki_v56_rb_ver",
                )
                if st.button("⏪ Rollback", key="wiki_v56_rb_btn"):
                    ok = self.rollback("species", selected_sp, int(rb_ver))
                    if ok:
                        st.success(f"Rolled back to v{rb_ver} ✅")
                        st.rerun()
                    else:
                        st.error("Rollback failed — version not found.")

        st.divider()

        # ── Locality checklist ──────────────────────────────────────────────────
        with st.expander("📍 Locality Species Checklist"):
            loc_list = self.list_localities()
            if loc_list:
                sel_loc = st.selectbox("Locality:", loc_list, key="wiki_v56_loc")
                if sel_loc:
                    loc_art = self._read("locality", self._slug(sel_loc)) or {}
                    sps     = loc_art.get("species_checklist", [])
                    st.write(f"**{len(sps)} species recorded at {sel_loc}:**")
                    cols2 = st.columns(2)
                    for i, sp in enumerate(sps):
                        cols2[i % 2].markdown(f"• *{_strip_auth(sp)}*")
                    lat = loc_art.get("decimalLatitude")
                    lon = loc_art.get("decimalLongitude")
                    if lat and lon:
                        st.map(pd.DataFrame([{"lat": lat, "lon": lon}]), zoom=7)
            else:
                st.info("No locality articles yet.")

    # ── Verification & classification panel ────────────────────────────────────

    def _render_verification_panel(
        self,
        art:          dict,
        species_name: str,
        meta_db_path: str,
        kg_db_path:   str = "",
        mb_db_path:   str = "",
    ):
        """
        Verify unverified species and update Scientific Classification.
        WoRMS + GBIF lookup (all taxa — marine_only=false).
        Syncs to ALL four databases on Save.
        """
        ts  = art.get("taxonomicStatus", "unverified")
        badge = {"verified": "✅", "unverified": "⏳", "synonym": "🔁",
                 "invalid": "❌"}.get(ts, "⏳")
        st.markdown(f"**Current status:** {badge} `{ts}`")
        st.markdown(
            "Use the lookup buttons below to auto-fill classification fields from "
            "**WoRMS** (all taxa) or **GBIF** (all kingdoms). "
            "Edit manually if needed, then **Save & Sync** to propagate to "
            "Occurrence DB · Knowledge Graph · Memory Bank · Wiki."
        )

        # ── Lookup row ──────────────────────────────────────────────────────────
        col_w, col_g, col_b = st.columns(3)
        with col_w:
            if st.button("🌊 WoRMS Lookup", key=f"worms_{species_name}"):
                with st.spinner("Querying WoRMS (marine_only=false)…"):
                    r = _lookup_worms(species_name)
                if r:
                    st.session_state[f"_taxlookup_{species_name}"] = r
                    st.success(f"WoRMS: *{r.get('validName', species_name)}*")
                else:
                    st.warning("Not found in WoRMS — try GBIF.")
        with col_g:
            if st.button("🌿 GBIF Lookup", key=f"gbif_{species_name}"):
                with st.spinner("Querying GBIF…"):
                    r = _lookup_gbif(species_name)
                if r:
                    st.session_state[f"_taxlookup_{species_name}"] = r
                    st.success(f"GBIF: *{r.get('validName', species_name)}*")
                else:
                    st.warning("Not found in GBIF.")
        with col_b:
            if st.button("🔄 Try Both", key=f"both_{species_name}"):
                with st.spinner("WoRMS → GBIF fallback…"):
                    r = _lookup_taxonomy(species_name)
                if r:
                    st.session_state[f"_taxlookup_{species_name}"] = r
                    st.success(
                        f"{r.get('source','?')}: *{r.get('validName', species_name)}*"
                    )
                else:
                    st.warning("Not found in WoRMS or GBIF — fill manually.")

        # Show last lookup result
        lookup = st.session_state.get(f"_taxlookup_{species_name}", {})
        if lookup:
            st.info(
                f"📌 **{lookup.get('source','')}** → "
                f"*{lookup.get('validName','')}* | "
                f"Status: `{lookup.get('taxonomicStatus','')}` | "
                f"Rank: `{lookup.get('taxonRank','')}` | "
                f"WoRMS ID: `{lookup.get('wormsID','—')}`"
            )

        # ── Classification editor form ─────────────────────────────────────────
        st.markdown("#### Scientific Classification")

        def _v(field: str, fb: str = "") -> str:
            return str(lookup.get(field) or art.get(field) or fb)

        with st.form(key=f"classify_form_{species_name}"):
            fc1, fc2 = st.columns(2)
            kingdom  = fc1.text_input("Kingdom",   value=_v("kingdom"))
            phylum   = fc2.text_input("Phylum",    value=_v("phylum"))
            fc3, fc4 = st.columns(2)
            class_   = fc3.text_input("Class",     value=_v("class_"))
            order_   = fc4.text_input("Order",     value=_v("order_"))
            fc5, fc6 = st.columns(2)
            family_  = fc5.text_input("Family",    value=_v("family_"))
            genus    = fc6.text_input("Genus",     value=_v("genus"))
            fc7, fc8 = st.columns(2)
            authority = fc7.text_input("Authority", value=_v("authority"))
            wormsID   = fc8.text_input("WoRMS ID",  value=_v("wormsID"))
            fc9, fc10 = st.columns(2)
            gbifID    = fc9.text_input("GBIF ID",   value=_v("gbifID"))
            iucn_opts = ["", "LC", "NT", "VU", "EN", "CR", "EW", "EX", "DD", "NE"]
            cur_iucn  = _v("iucnStatus")
            iucn_idx  = iucn_opts.index(cur_iucn) if cur_iucn in iucn_opts else 0
            iucn = fc10.selectbox("IUCN Status", iucn_opts, index=iucn_idx)

            valid_name = st.text_input(
                "Valid / Accepted Name (will update occurrence DB)",
                value=lookup.get("validName", "") or species_name,
            )
            recorded_name = st.text_input(
                "Recorded Name (as it appears in source documents)",
                value=_v("recordedName"),
            )
            status_opts = ["verified", "unverified", "accepted", "synonym",
                           "invalid", "uncertain"]
            cur_status  = _v("taxonomicStatus", "unverified")
            status_idx  = status_opts.index(cur_status) if cur_status in status_opts else 1
            new_status  = st.selectbox("Taxonomic Status", status_opts, index=status_idx)
            iucn_url    = st.text_input("IUCN URL (optional)", value=_v("iucnURL"))

            submitted = st.form_submit_button(
                "💾 Save & Sync to All Databases", type="primary"
            )

        if submitted:
            classification = {
                "kingdom": kingdom, "phylum": phylum, "class_": class_,
                "order_": order_, "family_": family_, "genus": genus,
                "authority": authority, "wormsID": wormsID, "gbifID": gbifID,
                "iucnStatus": iucn, "iucnURL": iucn_url,
                "validName": valid_name, "recordedName": recorded_name,
                "taxonomicStatus": new_status,
                "source": lookup.get("source", "manual"),
            }
            with st.spinner("Syncing classification to all databases…"):
                results = _sync_classification(
                    species_name, classification,
                    meta_db_path, kg_db_path, mb_db_path, self,
                )
            ok  = [k for k, v in results.items() if v]
            err = [k for k, v in results.items() if not v]
            if ok:
                st.success(f"✅ Synced to: {' · '.join(ok)}")
            if err:
                st.warning(f"⚠️ Sync failed for: {' · '.join(err)}")
            # Clear lookup cache so updated data is shown
            st.session_state.pop(f"_taxlookup_{species_name}", None)
            st.rerun()

    # ── Override _to_markdown to guarantee no truncation ─────────────────────
    # (already set at class level via staticmethod — shown here for clarity)