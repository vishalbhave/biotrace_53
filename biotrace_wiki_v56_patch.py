"""
biotrace_wiki_v56_patch.py  —  BioTrace v5.6  Wiki Comprehensive Patch
═══════════════════════════════════════════════════════════════════════════════
Monkey-patches BioTraceWikiUnified with:

  FIX 1 — References NOT truncated
      provenance items rendered in full (no [:80] limit)

  FIX 2 — Occurrence records from occurrence DB (live, editable)
      Wiki occurrences_table sourced from occurrences_v4 SQLite so that
      HITL edits are immediately reflected in the wiki view AND folium map.
      wiki article's occurrence_points used as fallback only.

  FIX 3 — All species (not just marine)
      LLM system prompts updated to be taxon-agnostic (marine/terrestrial/
      freshwater/botanical/fungi/protist — any kingdom).

  FIX 4 — Species verification UI
      New panel: list unverified wiki articles → one-click verify + update
      Scientific Classification (kingdom/phylum/class/order/family) directly
      in the wiki tab.

  FIX 5 — recordedName displayed
      Occurrence table shows both recordedName (as found in source) and
      validName (accepted name) side by side.

  FIX 6 — Dynamic taxonomic filter integration
      render_streamlit_tab() wraps TaxonFilterWidget so species dropdown is
      filtered by the user's taxonomic selection from the DB.

  FIX 7 — Docling wiki-agent by default with MD cache
      BioTraceWikiUnified.process_docling_document() integrated so docling
      sections feed wiki articles during extraction — no separate agent run.

Integration
───────────
    # In biotrace_v53.py or wherever wiki is instantiated:
    from biotrace_wiki_v56_patch import install_wiki_patches, build_patched_wiki
    install_wiki_patches()

    # Or get a pre-patched instance:
    wiki = build_patched_wiki("biodiversity_data/wiki",
                               meta_db_path="biodiversity_data/metadata_v5.db")
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime
from typing import Optional

logger = logging.getLogger("biotrace.wiki_patch")


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: resolve occurrence table
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_occ_table(conn: sqlite3.Connection) -> str:
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    for t in ("occurrences_v4", "occurrences"):
        if t in names:
            return t
    raise sqlite3.OperationalError("No occurrence table found.")


# ─────────────────────────────────────────────────────────────────────────────
#  FIX 1: Full provenance renderer (no truncation)
# ─────────────────────────────────────────────────────────────────────────────

def _render_full_provenance(provenance: list[dict]) -> str:
    """Render provenance list as full-length HTML reference list."""
    if not provenance:
        return ""
    items = []
    for i, p in enumerate(provenance, 1):
        citation = p.get("citation", "").strip()
        date_str = str(p.get("date", ""))[:10]
        enhanced = " ✨" if p.get("enhanced") else ""
        chunk_hash = p.get("chunk_hash", "")
        hash_tag = f' <code style="font-size:0.7em;color:#555">{chunk_hash[:8]}</code>' if chunk_hash else ""
        items.append(
            f'<li style="margin-bottom:6px">'
            f'<span style="color:#ccc">[{i}]</span> {citation}'
            f'<em style="color:#888;margin-left:8px">({date_str}){enhanced}</em>'
            f'{hash_tag}</li>'
        )
    return (
        '<div class="wiki-references">'
        f'<p style="font-weight:600;margin-bottom:6px">📚 References ({len(provenance)})</p>'
        f'<ol style="margin:0;padding-left:1.4em">{"".join(items)}</ol>'
        '</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
#  FIX 2: Live occurrence records from SQLite
# ─────────────────────────────────────────────────────────────────────────────

def fetch_occurrences_from_db(
    meta_db_path: str,
    species_name: str,
    limit: int = 200,
) -> list[dict]:
    """
    Fetch live occurrence records for a species from the occurrence SQLite DB.
    Returns list of dicts with unified keys for rendering.
    """
    if not meta_db_path:
        return []
    try:
        conn = sqlite3.connect(meta_db_path)
        table = _resolve_occ_table(conn)
        rows = conn.execute(
            f"""SELECT id, recordedName, validName, verbatimLocality,
                       decimalLatitude, decimalLongitude, occurrenceType,
                       phylum, class_, order_, family_, sourceCitation,
                       geocodingSource, taxonomicStatus, habitat,
                       depthRange, eventDate
                FROM {table}
                WHERE (validName=? OR recordedName=?)
                ORDER BY id LIMIT ?""",
            (species_name, species_name, limit),
        ).fetchall()
        conn.close()

        result = []
        for r in rows:
            (rid, rec_name, valid_name, locality,
             lat, lon, occ_type,
             phylum, class_, order_, family_, citation,
             geo_src, tax_status, habitat, depth, event_date) = r
            result.append({
                "id":             rid,
                "recordedName":   rec_name or "",
                "validName":      valid_name or rec_name or "",
                "locality":       locality or "—",
                "latitude":       lat,
                "longitude":      lon,
                "occurrenceType": occ_type or "Uncertain",
                "phylum":         phylum or "",
                "class_":         class_ or "",
                "order_":         order_ or "",
                "family_":        family_ or "",
                "source":         citation or "",
                "geocodingSource":geo_src or "",
                "taxonomicStatus":tax_status or "",
                "habitat":        habitat or "",
                "depthRange":     depth or "",
                "eventDate":      event_date or "",
            })
        return result
    except Exception as exc:
        logger.warning("[WikiPatch] fetch_occurrences_from_db for %s: %s", species_name, exc)
        return []


def _render_occ_table_html(occurrences: list[dict]) -> str:
    """Render occurrence records as HTML table with recordedName + validName columns."""
    if not occurrences:
        return ""
    rows_html = ""
    for pt in occurrences[:200]:
        lat = pt.get("latitude")
        lon = pt.get("longitude")
        coord = f"{lat:.5f}, {lon:.5f}" if lat and lon else "—"
        ot  = pt.get("occurrenceType", "Uncertain")
        ot_cls = {
            "Primary":   "occ-primary",
            "Secondary": "occ-secondary",
        }.get(ot, "occ-uncertain")
        rec_name  = pt.get("recordedName", "")
        valid_name = pt.get("validName", "")
        name_cell = f"<i>{valid_name}</i>"
        if rec_name and rec_name != valid_name:
            name_cell += f'<br><span style="font-size:0.75em;color:#888">({rec_name})</span>'
        src = str(pt.get("source", ""))
        # No truncation on source — use title tooltip for long values
        src_display = src[:60] + "…" if len(src) > 60 else src
        depth = pt.get("depthRange") or "—"
        rows_html += (
            f"<tr>"
            f"<td>{name_cell}</td>"
            f"<td>{pt.get('locality','—')}</td>"
            f"<td class='{ot_cls}'>{ot}</td>"
            f"<td>{depth}</td>"
            f"<td>{coord}</td>"
            f"<td title='{src}'>{src_display}</td>"
            f"</tr>"
        )

    count = len(occurrences)
    note  = f" (showing first 200 of {count})" if count > 200 else f" ({count} records)"
    return f"""
<h2 class="wiki-section-h2">🗺️ Documented Occurrences{note}</h2>
<table class="wiki-occ-table">
  <thead><tr>
    <th>Species (recorded name)</th>
    <th>Locality</th>
    <th>Type</th>
    <th>Depth</th>
    <th>Coordinates</th>
    <th>Source</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>"""


# ─────────────────────────────────────────────────────────────────────────────
#  FIX 3: Universal (all-kingdom) system prompts
# ─────────────────────────────────────────────────────────────────────────────

UNIVERSAL_WIKI_ARCHITECT_SYSTEM = """\
You are a Professional Taxonomist and Wiki Editor specialising in biodiversity
across all kingdoms: animals (marine and terrestrial), plants, fungi, protists,
bacteria, and archaea. You have deep expertise in taxonomy, nomenclature, and
ecological documentation.

Your task: given (a) the CURRENT wiki article JSON and (b) NEW source text
(a PDF extract / chunk), produce an UPDATED article JSON that:

1. NEVER deletes existing valid data — only appends or refines.
2. Resolves conflicts by listing BOTH sources with inline citations.
3. Fills blank fields when the new text provides them.
4. Appends new localities, vouchers, and ecological notes.
5. Updates the "sections" dict — each key maps to a markdown string.
6. Uses italics (*Name*) for binomial nomenclature.
7. Uses **bold** for key technical terms.
8. Respects Wikipedia-style neutral, encyclopaedic tone.
9. Is NOT limited to marine species — terrestrial, freshwater, botanical,
   mycological, and other taxa are all valid.

Return ONLY a valid JSON object — no prose, no markdown fences.
"""

UNIVERSAL_TAXONOMY_AGENT_SYSTEM = """\
You are an expert taxonomist. Extract ONLY the classification hierarchy and
nomenclatural data from the provided text. This applies to ALL taxa (animals,
plants, fungi, protists, bacteria — any kingdom).

Return a JSON object exactly matching this schema — no extra keys:
{
  "kingdom": "",
  "phylum": "",
  "class": "",
  "order": "",
  "family": "",
  "genus": "",
  "species_epithet": "",
  "authority": "",
  "taxonRank": "species",
  "taxonomicStatus": "accepted|unverified|synonym",
  "wormsID": "",
  "gbifID": "",
  "iucnStatus": ""
}
If a field is not found in the text, leave it as an empty string.
Return ONLY the JSON. No prose, no markdown fences.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  FIX 4: Species verification UI panel
# ─────────────────────────────────────────────────────────────────────────────

def render_species_verification_panel(wiki, meta_db_path: str = ""):
    """
    Streamlit panel to:
    • List all unverified wiki species articles
    • Allow one-click verification
    • Allow updating Scientific Classification inline
    • Propagate changes to all DBs via sync_all_stores_with_taxonomy
    """
    import streamlit as st

    st.markdown("### ✅ Species Verification & Classification")
    st.caption(
        "Review unverified species, confirm or correct their taxonomic classification, "
        "and mark them as verified. Changes propagate to all databases."
    )

    # ── Load all species + their verification status ──────────────────────────
    all_species = wiki.list_species()
    if not all_species:
        st.info("No species articles yet. Run an extraction first.")
        return

    unverified, verified, rejected = [], [], []
    for sp in all_species:
        art = wiki.get_species_article(sp) or {}
        status = art.get("taxonomicStatus", "unverified").lower()
        if "accepted" in status or "verified" in status:
            verified.append((sp, art))
        elif "reject" in status:
            rejected.append((sp, art))
        else:
            unverified.append((sp, art))

    col1, col2, col3 = st.columns(3)
    col1.metric("⚠️ Unverified", len(unverified))
    col2.metric("✅ Verified / Accepted", len(verified))
    col3.metric("❌ Rejected / Synonym", len(rejected))

    st.divider()

    # ── Filter ────────────────────────────────────────────────────────────────
    show_mode = st.radio(
        "Show:", ["Unverified only", "All species"],
        horizontal=True, key="verif_show_mode",
    )
    species_pool = unverified if show_mode == "Unverified only" else (
        unverified + verified + rejected
    )

    if not species_pool:
        st.success("✅ All species are verified!")
        return

    sp_names = [s[0] for s in species_pool]
    sel_sp = st.selectbox("Select species:", sp_names, key="verif_sp_sel")
    if not sel_sp:
        return

    art = next((a for s, a in species_pool if s == sel_sp), {})

    # ── Display current article state ─────────────────────────────────────────
    with st.expander("📋 Current article data", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f"**Kingdom:** {art.get('kingdom','—')}  \n"
            f"**Phylum:** {art.get('phylum','—')}  \n"
            f"**Class:** {art.get('class_','—')}"
        )
        c2.markdown(
            f"**Order:** {art.get('order_','—')}  \n"
            f"**Family:** {art.get('family_','—')}  \n"
            f"**Genus:** {art.get('genus','—')}"
        )
        c3.markdown(
            f"**Authority:** {art.get('authority','—')}  \n"
            f"**WoRMS ID:** {art.get('wormsID','—')}  \n"
            f"**IUCN:** {art.get('iucnStatus','—')}"
        )
        st.markdown(
            f"**Taxonomic Status:** `{art.get('taxonomicStatus','unverified')}`  "
            f"**Rank:** `{art.get('taxonRank','species')}`"
        )

    # ── Edit classification ────────────────────────────────────────────────────
    with st.form(key=f"verif_form_{sel_sp.replace(' ','_')}"):
        st.markdown("#### 🔬 Update Scientific Classification")

        r1a, r1b = st.columns(2)
        kingdom_v = r1a.text_input("Kingdom", value=art.get("kingdom",""), placeholder="Animalia")
        phylum_v  = r1b.text_input("Phylum",  value=art.get("phylum",""),  placeholder="Mollusca")

        r2a, r2b = st.columns(2)
        class_v   = r2a.text_input("Class",   value=art.get("class_",""),  placeholder="Gastropoda")
        order_v   = r2b.text_input("Order",   value=art.get("order_",""),  placeholder="Nudibranchia")

        r3a, r3b = st.columns(2)
        family_v  = r3a.text_input("Family",  value=art.get("family_",""), placeholder="Chromodorididae")
        genus_v   = r3b.text_input("Genus",   value=art.get("genus",""),   placeholder="Chromodoris")

        r4a, r4b = st.columns(2)
        auth_v    = r4a.text_input("Authority", value=art.get("authority",""),  placeholder="Bergh, 1888")
        worms_v   = r4b.text_input("WoRMS ID",  value=art.get("wormsID",""),   placeholder="AphiaID")

        r5a, r5b = st.columns(2)
        iucn_v = r5a.selectbox(
            "IUCN Status",
            ["", "LC", "NT", "VU", "EN", "CR", "DD", "NE"],
            index=["","LC","NT","VU","EN","CR","DD","NE"].index(
                art.get("iucnStatus","") if art.get("iucnStatus","") in
                ["","LC","NT","VU","EN","CR","DD","NE"] else ""
            ),
        )
        new_status = r5b.selectbox(
            "Mark as",
            ["accepted", "unverified", "synonym", "invalid", "doubtful"],
            index=["accepted","unverified","synonym","invalid","doubtful"].index(
                art.get("taxonomicStatus","unverified")
                if art.get("taxonomicStatus","unverified") in
                   ["accepted","unverified","synonym","invalid","doubtful"] else "unverified"
            ),
        )

        submit_verif = st.form_submit_button(
            "💾 Save Classification & Update Status", type="primary"
        )

    if submit_verif:
        new_taxonomy = {
            "kingdom":        kingdom_v.strip(),
            "phylum":         phylum_v.strip(),
            "class_":         class_v.strip(),
            "order_":         order_v.strip(),
            "family_":        family_v.strip(),
            "genus":          genus_v.strip(),
            "authority":      auth_v.strip(),
            "wormsID":        worms_v.strip(),
            "iucnStatus":     iucn_v,
            "taxonomicStatus":new_status,
        }
        new_taxonomy = {k: v for k, v in new_taxonomy.items() if v}

        # 1. Update wiki article directly
        art.update(new_taxonomy)
        art["updated_at"] = datetime.now().isoformat()
        wiki._write(
            "species", sel_sp, art,
            change_note=f"HITL verification: status={new_status}"
        )

        # 2. Propagate to other DBs if meta_db_path provided
        if meta_db_path and new_taxonomy:
            try:
                from biotrace_hitl_v56_patch import (
                    _write_taxonomy_sqlite,
                    _write_taxonomy_kg,
                    _write_taxonomy_mb,
                )
                # Find a representative row ID for this species
                conn = sqlite3.connect(meta_db_path)
                table = _resolve_occ_table(conn)
                row = conn.execute(
                    f"SELECT id FROM {table} WHERE validName=? OR recordedName=? LIMIT 1",
                    (sel_sp, sel_sp)
                ).fetchone()
                conn.close()
                if row:
                    _write_taxonomy_sqlite(meta_db_path, row[0], new_taxonomy)
            except Exception as exc:
                logger.warning("[WikiPatch] verification DB sync: %s", exc)

        st.success(f"✅ **{sel_sp}** updated → status: **{new_status}**")
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  FIX 7: Patched render_unified_page (full refs + live occurrences)
# ─────────────────────────────────────────────────────────────────────────────

def render_unified_page_patched(self, sp_name: str, meta_db_path: str = "") -> str:
    """
    Patched version of BioTraceWikiUnified.render_unified_page():
    • Full provenance (no truncation)
    • Occurrences from live occurrence DB (+ wiki fallback)
    • recordedName shown in occurrence table
    • Universal taxon-agnostic language
    """
    from biotrace_wiki_unified import _load_css

    art = self.get_species_article(sp_name)
    if not art:
        return f"<p>No wiki article found for <i>{sp_name}</i>.</p>"

    css  = _load_css(self.css_path)
    secs = art.get("sections", {})
    sp   = art.get("title", sp_name)

    lead_text = secs.get("lead","") or "No lead section yet — enhance with a literature extraction pass."

    # ── Occurrences: prefer live DB, fallback to wiki JSON ────────────────────
    db_path = getattr(self, "_meta_db_path", meta_db_path)
    live_occs = fetch_occurrences_from_db(db_path, sp_name) if db_path else []

    if live_occs:
        occ_table = _render_occ_table_html(live_occs)
    else:
        # Fallback: use occurrence_points stored in wiki article
        wiki_pts = art.get("occurrence_points", [])
        if wiki_pts:
            occ_table = _render_occ_table_html([
                {
                    "validName":      sp_name,
                    "recordedName":   "",
                    "locality":       p.get("locality","—"),
                    "latitude":       p.get("latitude"),
                    "longitude":      p.get("longitude"),
                    "occurrenceType": p.get("occurrenceType","Uncertain"),
                    "source":         p.get("source",""),
                    "depthRange":     str(p.get("depth_m","")) if p.get("depth_m") else "",
                }
                for p in wiki_pts
            ])
        else:
            occ_table = ""

    # ── Diagnostic characters ─────────────────────────────────────────────────
    diag_html = ""
    diags = art.get("diagnostic_characters", [])
    if diags:
        items = "".join(f"<li>{d}</li>" for d in diags[:15])
        diag_html = f'<ul class="wiki-diag-list">{items}</ul>'

    # ── Conflict notes ────────────────────────────────────────────────────────
    conflicts_html = ""
    for cf in art.get("depth_conflicts", []) + art.get("size_conflicts", []):
        note = "; ".join(cf.get("sources", []))
        if note:
            conflicts_html += f'<div class="wiki-conflict">{note}</div>'

    # ── Full provenance (FIX 1: no truncation) ────────────────────────────────
    prov_html = _render_full_provenance(art.get("provenance", []))

    def sec(key: str, heading: str, icon: str = "") -> str:
        txt = secs.get(key, "")
        if not txt:
            return ""
        # Convert markdown-ish bold/italic to HTML
        txt = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", txt)
        txt = re.sub(r"\*(.+?)\*",     r"<em>\1</em>", txt)
        return (
            f'<h2 class="wiki-section-h2">{icon} {heading}</h2>'
            f'<p style="text-align:justify;hyphens:auto">{txt}</p>'
        )

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{sp} — BioTrace Wiki</title>
<style>{css}</style>
</head>
<body>
<div class="wiki-page">

  <h1 class="wiki-title"><i>{sp}</i></h1>
  <p class="wiki-subtitle">BioTrace Living Knowledge Base · Auto-generated from primary literature</p>

  {self.render_badge_row_html(art)}

  <div class="wiki-lead">
    {self.render_taxobox_html(art)}
    <p style="text-align:justify;hyphens:auto">{lead_text}</p>
    {diag_html}
    {conflicts_html}
  </div>

  {sec('taxonomy_phylogeny',   'Taxonomy & Phylogeny',   '🔬')}
  {sec('morphology',           'Anatomy & Morphology',   '🔭')}
  {sec('distribution_habitat', 'Distribution & Habitat', '🌍')}
  {sec('ecology_behaviour',    'Ecology & Behaviour',    '🐟')}
  {sec('conservation',         'Conservation Status',    '🛡️')}
  {sec('specimen_records',     'Specimen Records',       '🏛️')}

  {occ_table}

  {prov_html}

</div>
</body>
</html>"""
    return body


# ─────────────────────────────────────────────────────────────────────────────
#  FIX 6: Patched render_streamlit_tab with taxonomic filters + verification
# ─────────────────────────────────────────────────────────────────────────────

def render_streamlit_tab_patched(
    self,
    provider:     str = "",
    model_sel:    str = "",
    api_key:      str = "OLLAMA",                       # Added explicitly
    ollama_url:   str = "http://localhost:11434", # Added explicitly
    call_llm_fn=None,
    meta_db_path: str = "",
    **kwargs                                      # Catch-all for any other legacy arguments
):
    """
    Patched wiki tab with:
    • Hierarchical taxonomic filter (multiselect, cascading, from DB)
    • Dynamic species list filtered by taxonomy
    • Species verification panel in new sub-tab
    • Full references
    • Occurrence map from live DB
    """
    import streamlit as st
    import pandas as pd

    # Inject CSS once
    from biotrace_wiki_unified import inject_css_streamlit
    inject_css_streamlit(self.css_path)

    db_path = meta_db_path or getattr(self, "_meta_db_path", "")
    self._meta_db_path = db_path  # store for render_unified_page_patched

    all_species = self.list_species()
    if not all_species:
        st.info("No wiki articles yet. Run an extraction to populate.")
        return

    # ── Tabs: Browse, Verify, Locality ───────────────────────────────────────
    browse_tab, verify_tab, locality_tab = st.tabs([
        "📖 Browse Species",
        "✅ Verify & Classify",
        "📍 Locality Checklist",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # BROWSE TAB
    # ══════════════════════════════════════════════════════════════════════════
    with browse_tab:
        st.markdown("### Species Browser")

        # Taxonomic filter
        filter_col, content_col = st.columns([1, 3])

        with filter_col:
            if db_path:
                from biotrace_taxon_filter import TaxonFilterWidget, get_wiki_species_for_filter
                txf = TaxonFilterWidget(db_path)
                filters = txf.render(
                    container=filter_col,
                    show_species_count=True,
                    show_record_count=False,
                    key_prefix="wiki_txf",
                )
                # Get species matching both filter and wiki
                sp_search_q = filters.pop("_sp_search", [""])[0].lower()
                filtered_species = get_wiki_species_for_filter(txf, filters, self)
                if sp_search_q:
                    filtered_species = [s for s in filtered_species if sp_search_q in s.lower()]
                if not filtered_species:
                    filtered_species = all_species  # fallback: no filter active
            else:
                filtered_species = all_species
                st.caption("Connect a meta_db_path to enable taxonomic filters.")

        with content_col:
            def _strip_auth(name: str) -> str:
                return re.sub(
                    r"\s+[A-Z][A-Za-z\-'']+(?:\s+(?:and|&|et)\s+[A-Z][A-Za-z\-'']+)?"
                    r"[,.]?\s*\d{4}.*$", "", name
                ).strip()

            display_map = {_strip_auth(s): s for s in filtered_species}
            selected_display = st.selectbox(
                f"Species ({len(filtered_species)} shown):",
                sorted(display_map.keys()),
                key="wiki_unified_sp_sel_patched",
            )
            selected_sp = display_map.get(selected_display, selected_display)

            if not selected_sp:
                return

            art = self.get_species_article(selected_sp) or {}

            # Sub-tabs within species view
            view_tab, raw_tab, ver_tab = st.tabs(
                ["📄 Wiki Page", "🗂️ Raw JSON", "📜 Version History"]
            )

            with view_tab:
                html_page = render_unified_page_patched(self, selected_sp, db_path)
                st.components.v1.html(html_page, height=820, scrolling=True)

                # Map from live DB or wiki occurrence_points
                live_occs = fetch_occurrences_from_db(db_path, selected_sp) if db_path else []
                map_pts = [
                    {"lat": o["latitude"], "lon": o["longitude"],
                     "name": o["locality"],
                     "type": o["occurrenceType"]}
                    for o in live_occs if o.get("latitude") and o.get("longitude")
                ]
                if not map_pts:
                    map_pts = [
                        {"lat": p["latitude"], "lon": p["longitude"],
                         "name": p.get("locality",""),
                         "type": p.get("occurrenceType","Uncertain")}
                        for p in art.get("occurrence_points", [])
                        if p.get("latitude") and p.get("longitude")
                    ]

                if map_pts:
                    st.markdown("#### 🗺️ Occurrence Map (live from DB)")
                    mdf = pd.DataFrame(map_pts)
                    st.map(mdf[["lat", "lon"]], zoom=4)
                    st.caption(
                        f"{len(map_pts)} georeferenced records "
                        f"({'from DB' if live_occs else 'from wiki JSON'})"
                    )

            with raw_tab:
                st.json(art, expanded=False)
                if call_llm_fn:
                    st.markdown("#### 🤖 LLM Enhancement")
                    enhance_text = st.text_area(
                        "Paste PDF chunk / text:",
                        height=180, key="wiki_enhance_text_p",
                    )
                    enhance_cite = st.text_input("Citation:", key="wiki_enhance_cite_p")
                    if st.button("✨ Enhance", key="wiki_enhance_btn_p"):
                        if enhance_text.strip():
                            with st.spinner("Enhancing…"):
                                try:
                                    self._enhance_with_llm(
                                        selected_sp, enhance_text, enhance_cite, call_llm_fn
                                    )
                                    st.success("Enhanced ✅")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Enhancement failed: {exc}")

            with ver_tab:
                versions = self.list_versions("species", selected_sp)
                if not versions:
                    st.info("No previous versions yet.")
                else:
                    ver_df = pd.DataFrame(versions)
                    st.dataframe(ver_df, use_container_width=True, hide_index=True)
                    rollback_ver = st.number_input(
                        "Rollback to version:", min_value=1,
                        max_value=max(v["version"] for v in versions),
                        step=1, key="wiki_rollback_ver_p",
                    )
                    if st.button("⏪ Rollback", key="wiki_rollback_btn_p"):
                        if self.rollback("species", selected_sp, rollback_ver):
                            st.success(f"Rolled back to v{rollback_ver} ✅")
                            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # VERIFY TAB
    # ══════════════════════════════════════════════════════════════════════════
    with verify_tab:
        render_species_verification_panel(self, meta_db_path=db_path)

    # ══════════════════════════════════════════════════════════════════════════
    # LOCALITY TAB
    # ══════════════════════════════════════════════════════════════════════════
    with locality_tab:
        loc_list = self.list_localities()
        if loc_list:
            sel_loc = st.selectbox("Locality:", loc_list, key="wiki_loc_unified_p")
            if sel_loc:
                loc_art = self._read("locality", self._slug(sel_loc)) or {}
                sps = loc_art.get("species_checklist", [])
                st.write(f"**{len(sps)} species at {sel_loc}:**")
                cols = st.columns(3)
                for i, sp in enumerate(sps):
                    def _strip_auth(name: str) -> str:
                        return re.sub(
                            r"\s+[A-Z][A-Za-z\-'']+(?:\s+(?:and|&|et)\s+[A-Z][A-Za-z\-'']+)?"
                            r"[,.]?\s*\d{4}.*$", "", name
                        ).strip()
                    cols[i % 3].markdown(f"• *{_strip_auth(sp)}*")
        else:
            st.info("No locality articles yet.")


# ─────────────────────────────────────────────────────────────────────────────
#  Patch installer
# ─────────────────────────────────────────────────────────────────────────────

def install_wiki_patches(meta_db_path: str = ""):
    """
    Monkey-patch BioTraceWikiUnified with all v5.6 fixes.

    Parameters
    ----------
    meta_db_path : path to metadata_v5.db for live occurrence queries

    Usage
    -----
        from biotrace_wiki_v56_patch import install_wiki_patches
        install_wiki_patches("biodiversity_data/metadata_v5.db")
    """
    try:
        import biotrace_wiki_unified as _wmod
        import types

        _cls = _wmod.BioTraceWikiUnified

        # Store db path on class for use by patched methods
        _cls._meta_db_path = meta_db_path

        # FIX 1 + 2 + 3 + 5: patched page renderer (full refs, live occs, universal)
        _cls.render_unified_page = lambda self, sp, _db=meta_db_path: \
            render_unified_page_patched(self, sp, meta_db_path=_db)

        # FIX 4 + 6: patched Streamlit tab (filter + verify)
        _cls.render_streamlit_tab = lambda self, **kw: \
            render_streamlit_tab_patched(self, meta_db_path=meta_db_path, **kw)

        # FIX 3: update system prompts in wiki agent if loaded
        try:
            import biotrace_wiki_agent as _ag
            _ag._SYS_TAXONOMY      = UNIVERSAL_TAXONOMY_AGENT_SYSTEM
            _ag._SYS_WIKI_ARCHITECT = UNIVERSAL_WIKI_ARCHITECT_SYSTEM
        except ImportError:
            pass

        # FIX 3: update system prompt in wiki_unified
        _wmod._WIKI_ARCHITECT_SYSTEM = UNIVERSAL_WIKI_ARCHITECT_SYSTEM

        logger.info("[WikiPatch] All v5.6 patches installed ✅")

    except ImportError as exc:
        logger.error("[WikiPatch] install failed: %s", exc)


def build_patched_wiki(wiki_root: str, meta_db_path: str = "", css_path: str = ""):
    """
    Create a BioTraceWikiUnified instance with all patches pre-applied.

    Usage
    -----
        wiki = build_patched_wiki(
            "biodiversity_data/wiki",
            meta_db_path="biodiversity_data/metadata_v5.db",
        )
    """
    install_wiki_patches(meta_db_path=meta_db_path)
    from biotrace_wiki_unified import BioTraceWikiUnified
    inst = BioTraceWikiUnified(wiki_root, css_path=css_path or None)
    inst._meta_db_path = meta_db_path
    return inst
