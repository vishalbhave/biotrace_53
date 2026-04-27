"""
biotrace_v5_enhancements.py  —  BioTrace v5.2
────────────────────────────────────────────────────────────────────────────
Drop-in UI enhancement module for biotrace_v5.py.

Adds new tabs to the Streamlit app:
  • TNR Engine tab      — Hybrid taxonomic name recognition on raw text
  • Verification Table  — Editable Detected Species | Classification |
                          Verified Locality | Source Statement | Flag
  • Locality Expander   — Interactive "Narara → full admin string" tool
  • Schema Diagnostics  — Pydantic validation error viewer

Also provides:
  • render_verification_table()  — editable Streamlit dataframe
  • render_tnr_tab()             — TNR controls + results
  • render_locality_tab()        — locality NER + expander UI
  • render_schema_diagnostics()  — parse error log display

These functions are imported and called from biotrace_v5.py main tab block.

Standalone usage for testing:
    streamlit run biotrace_v5_enhancements.py
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("biotrace.enhancements")

# ─────────────────────────────────────────────────────────────────────────────
#  OPTIONAL MODULE IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
_NER_AVAILABLE = False
TaxonNER = None
try:
    from biotrace_ner import TaxonNER, regex_scan, trigram_score, extract_taxa
    _NER_AVAILABLE = True
    logger.info("[enhancements] biotrace_ner loaded")
except ImportError:
    logger.warning("[enhancements] biotrace_ner.py not found")

_LOC_NER_AVAILABLE = False
LocalityNER = None
try:
    from biotrace_locality_ner import LocalityNER, segregate_locality_string
    _LOC_NER_AVAILABLE = True
    logger.info("[enhancements] biotrace_locality_ner loaded")
except ImportError:
    logger.warning("[enhancements] biotrace_locality_ner.py not found")

_SCHEMA_AVAILABLE = False
try:
    from biotrace_schema import (
        parse_llm_response, safe_parse_json,
        OccurrenceRecord, records_to_dicts,
        SCHEMA_JSON_EXAMPLE, SCHEMA_VALIDATION_RULES,
    )
    _SCHEMA_AVAILABLE = True
    logger.info("[enhancements] biotrace_schema loaded")
except ImportError:
    logger.warning("[enhancements] biotrace_schema.py not found")

_GNV_AVAILABLE = False
try:
    from biotrace_gnv import GNVEnrichedVerifier, LocalitySplitter, dedup_occurrences
    _GNV_AVAILABLE = True
except ImportError:
    pass

DATA_DIR    = "biodiversity_data"
META_DB_5   = os.path.join(DATA_DIR, "metadata_v5.db")
GEONAMES_DB = os.path.join(DATA_DIR, "geonames_india.db")
PINCODE_TXT = os.path.join(DATA_DIR, "IN_pin.txt")


# ─────────────────────────────────────────────────────────────────────────────
#  VERIFICATION TABLE
# ─────────────────────────────────────────────────────────────────────────────
VERIF_COLS = [
    "Detected Species",
    "Full Classification",
    "Verified Locality",
    "Source Statement",
    "Flag",              # Primary | Secondary | Uncertain
    "WoRMS ID",
    "Match Score",
    "Validation",        # human-editable: Accept | Reject | Review
    "Notes",
]


def occurrences_to_verification_df(occurrences: list[dict]) -> pd.DataFrame:
    """Convert occurrence dicts to the verification table format."""
    rows = []
    for occ in occurrences:
        if not isinstance(occ, dict):
            continue
        name     = (occ.get("validName") or occ.get("recordedName") or
                    occ.get("Valid Name") or occ.get("Recorded Name","")).strip()
        if not name:
            continue

        phylum = occ.get("phylum","")
        class_ = occ.get("class_","") or occ.get("class","")
        order_ = occ.get("order_","") or occ.get("order","")
        family = occ.get("family_","") or occ.get("family","")
        tax_parts = [p for p in [phylum, class_, order_, family] if p]
        classification = " > ".join(tax_parts) if tax_parts else "—"

        locality = (
            occ.get("expandedLocality")
            or occ.get("verbatimLocality","")
        )

        evidence = str(
            occ.get("Raw Text Evidence") or occ.get("rawTextEvidence","")
        )[:200]

        worms = occ.get("wormsID","")
        score = occ.get("matchScore", occ.get("match_score", 0)) or 0

        rows.append({
            "Detected Species":   name,
            "Full Classification":classification,
            "Verified Locality":  locality,
            "Source Statement":   evidence,
            "Flag":               occ.get("occurrenceType","Uncertain"),
            "WoRMS ID":           worms,
            "Match Score":        round(float(score), 3),
            "Validation":         "Review",
            "Notes":              "",
        })

    return pd.DataFrame(rows, columns=VERIF_COLS)


def render_verification_table(
    occurrences: list[dict],
    key_prefix: str = "verif",
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Render an editable verification table in Streamlit.
    Returns (edited_df, updated_occurrences).

    Editable columns: Flag, Validation, Notes
    """
    try:
        import streamlit as st
    except ImportError:
        return pd.DataFrame(), occurrences

    df = occurrences_to_verification_df(occurrences)
    if df.empty:
        st.info("No records to verify.")
        return df, occurrences

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total records",    len(df))
    col2.metric("Primary",          int((df["Flag"]=="Primary").sum()))
    col3.metric("Secondary",        int((df["Flag"]=="Secondary").sum()))
    col4.metric("WoRMS verified",   int((df["WoRMS ID"] != "").sum()))

    st.caption(
        "✏️ Edit the **Flag**, **Validation**, and **Notes** columns directly. "
        "Click a row to expand the Source Statement."
    )

    # Column configuration for Streamlit data_editor
    col_config = {
        "Detected Species":   st.column_config.TextColumn("Species", width="medium"),
        "Full Classification":st.column_config.TextColumn("Classification", width="large"),
        "Verified Locality":  st.column_config.TextColumn("Locality", width="medium"),
        "Source Statement":   st.column_config.TextColumn("Evidence", width="large"),
        "Flag":               st.column_config.SelectboxColumn(
                                  "Flag", options=["Primary","Secondary","Uncertain"], width="small"
                              ),
        "WoRMS ID":           st.column_config.LinkColumn(
                                  "WoRMS",
                                  display_text=r"(\d+)",
                                  help="Click to open WoRMS record",
                              ) if df["WoRMS ID"].any() else st.column_config.TextColumn("WoRMS"),
        "Match Score":        st.column_config.ProgressColumn(
                                  "Score", min_value=0, max_value=1, format="%.2f"
                              ),
        "Validation":         st.column_config.SelectboxColumn(
                                  "Validation",
                                  options=["Accept","Reject","Review"],
                                  width="small",
                              ),
        "Notes":              st.column_config.TextColumn("Notes", width="medium"),
    }

    # Format WoRMS IDs as links
    df["WoRMS ID"] = df["WoRMS ID"].apply(
        lambda x: f"https://www.marinespecies.org/aphia.php?p=taxdetails&id={x}" if x else ""
    )

    edited_df = st.data_editor(
        df,
        column_config=col_config,
        disabled=["Detected Species","Full Classification","Source Statement","Match Score"],
        use_container_width=True,
        num_rows="fixed",
        key=f"{key_prefix}_editor",
        height=min(600, 60 + 35 * len(df)),
    )

    # Apply edits back to occurrences
    if edited_df is not None and not edited_df.empty:
        species_to_flag = dict(zip(edited_df["Detected Species"], edited_df["Flag"]))
        species_to_valid= dict(zip(edited_df["Detected Species"], edited_df["Validation"]))
        species_to_notes= dict(zip(edited_df["Detected Species"], edited_df["Notes"]))

        for occ in occurrences:
            sp = (occ.get("validName") or occ.get("recordedName","")).strip()
            if sp in species_to_flag:
                occ["occurrenceType"]   = species_to_flag[sp]
                occ["validationStatus"] = species_to_valid[sp]
                occ["notes"]            = species_to_notes.get(sp,"")

    return edited_df, occurrences


# ─────────────────────────────────────────────────────────────────────────────
#  TNR TAB RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def render_tnr_tab():
    """
    Render the TNR Engine tab in Streamlit.
    Allows paste/upload of text → runs hybrid TNR → shows verification table.
    """
    try:
        import streamlit as st
    except ImportError:
        return

    st.subheader("🔬 Taxonomic Name Recognition (TNR) Engine")
    st.caption(
        "BHL-style three-phase pipeline: "
        "Regex discovery → GNA Finder → GNA Verifier (WoRMS/ITIS/CoL) → Disambiguation"
    )

    # Status indicators
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            f"{'✅' if _NER_AVAILABLE else '❌'} **biotrace_ner** — Regex + GNA finder"
        )
    with col_b:
        try:
            import spacy
            models = spacy.util.get_installed_models()
            st.markdown(f"{'✅' if models else '⚠️'} **spaCy** — {'`' + models[0] + '`' if models else 'no model installed'}")
        except Exception:
            st.markdown("❌ **spaCy** — not installed")
    with col_c:
        st.markdown(
            f"{'✅' if _SCHEMA_AVAILABLE else '❌'} **Pydantic schema** — validation"
        )

    if not _NER_AVAILABLE:
        st.error(
            "biotrace_ner.py is required for this tab. "
            "Place it alongside biotrace_v5.py."
        )
        return

    st.divider()

    # ── Input ──────────────────────────────────────────────────────────────────
    input_method = st.radio(
        "Input method:", ["Paste text", "From database (last extraction)"],
        horizontal=True,
    )

    text_input = ""
    if input_method == "Paste text":
        text_input = st.text_area(
            "Paste document text or section:",
            height=200,
            placeholder="Paste any biodiversity text — supports marine, terrestrial, freshwater taxa...",
        )
    else:
        # Load from last extracted markdown session
        session_files = sorted(
            Path("biodiversity_data/extractions_v5").glob("*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if session_files:
            selected = st.selectbox("Session:", [f.name for f in session_files[:10]])
            # Read raw text evidence as a proxy
            try:
                df_s = pd.read_csv(session_files[0])
                evidences = df_s["rawTextEvidence"].dropna().tolist()
                text_input = " ".join(evidences[:50])
                st.caption(f"Loaded {len(evidences)} evidence strings from {selected}")
            except Exception:
                text_input = ""
        else:
            st.info("No extraction sessions found. Run an extraction first.")

    # ── Options ────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        use_gna_finder  = st.checkbox("GNA Finder (network)", value=True)
    with col2:
        use_gna_verify  = st.checkbox("GNA Verifier (network)", value=True)
    with col3:
        use_trigram     = st.checkbox("Trigram filter", value=True)

    col4, col5 = st.columns(2)
    with col4:
        min_score = st.slider("Min GNA score", 0.5, 1.0, 0.80, 0.05)
    with col5:
        run_disambig = st.checkbox("Abbreviation resolver (Phase 3)", value=True)

    run_btn = st.button(
        "🔍 Run TNR", type="primary",
        disabled=(not text_input.strip()),
    )

    if run_btn and text_input.strip():
        with st.spinner("Running TNR pipeline…"):
            ner = TaxonNER(
                use_gna_finder    = use_gna_finder,
                use_gna_verify    = use_gna_verify,
                use_trigram_filter= use_trigram,
                min_gna_score     = min_score,
            )
            candidates = ner.extract(
                text_input,
                source_label="manual_input",
                run_disambig=run_disambig,
            )

        st.success(f"Found **{len(candidates)}** taxa")

        if candidates:
            # Build DataFrame for display
            rows = []
            for c in candidates:
                rows.append({
                    "Verbatim":         c.verbatim,
                    "Canonical":        c.canonical,
                    "Valid Name":       c.valid_name or "—",
                    "Family":           c.family_ or "—",
                    "Phylum":           c.phylum or "—",
                    "Detection":        c.source,
                    "GNA ✓":           "✅" if c.gna_valid else "⚠️",
                    "Score":            round(c.match_score, 2),
                    "WoRMS":            c.worms_id or "—",
                    "Flag":             c.occurrence_type,
                    "Evidence":         c.context[:150] + "…" if len(c.context) > 150 else c.context,
                })

            df_tnr = pd.DataFrame(rows)
            st.dataframe(
                df_tnr,
                use_container_width=True,
                height=400,
                column_config={
                    "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1),
                    "Flag":  st.column_config.SelectboxColumn(
                        "Flag", options=["Primary","Secondary","Uncertain"]
                    ),
                },
            )

            # Download
            csv_bytes = df_tnr.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download TNR results (CSV)",
                data=csv_bytes,
                file_name="tnr_results.csv",
                mime="text/csv",
            )

            # Trigram score chart
            with st.expander("📊 Trigram score distribution"):
                st.bar_chart(df_tnr.set_index("Canonical")["Score"])

            # COPIOUS boost from Memory Bank
            mb_path = os.path.join(DATA_DIR, "memory_bank.db")
            if os.path.exists(mb_path):
                from biotrace_ner import COPIOUSFilter
                cop = COPIOUSFilter()
                cop.load_from_memory_bank(mb_path)
                candidates = cop.filter(candidates)
                known_count = sum(1 for c in candidates if cop.is_known(c.canonical))
                if known_count:
                    st.info(
                        f"🧠 COPIOUS boost: {known_count} names matched in Memory Bank "
                        f"(confidence +0.05)"
                    )


# ─────────────────────────────────────────────────────────────────────────────
#  LOCALITY EXPANDER TAB
# ─────────────────────────────────────────────────────────────────────────────
def render_locality_tab():
    """Render the Locality NER + Expander tab."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.subheader("📍 Locality NER & Administrative Expander")
    st.caption(
        '"Narara" → "Narara, Jamnagar, Gulf of Kutch, Gujarat, India" · '
        "spaCy GPE + GeoNames IN + Nominatim (1 req/sec)"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"{'✅' if _LOC_NER_AVAILABLE else '❌'} **biotrace_locality_ner**"
        )
        geonames_ok = os.path.exists(GEONAMES_DB)
        st.markdown(f"{'✅' if geonames_ok else '❌'} **GeoNames IN DB** — `{GEONAMES_DB}`")
    with col_b:
        try:
            from geopy.geocoders import Nominatim
            st.markdown("✅ **geopy / Nominatim** installed")
        except ImportError:
            st.markdown("❌ **geopy** — `pip install geopy`")

    if not _LOC_NER_AVAILABLE:
        st.error("biotrace_locality_ner.py required.")
        return

    st.divider()

    # ── Single locality expansion ─────────────────────────────────────────────
    st.subheader("Single Locality Expansion")
    single_loc = st.text_input(
        "Enter place name:",
        placeholder="e.g. Narara Island",
    )
    use_nominatim = st.checkbox("Enable Nominatim fallback (1 req/sec)", value=False)

    if st.button("🌍 Expand", type="primary") and single_loc:
        with st.spinner("Expanding…"):
            lner = LocalityNER(
                geonames_db   = GEONAMES_DB  if geonames_ok else "",
                pincode_txt   = PINCODE_TXT  if os.path.exists(PINCODE_TXT) else "",
                use_nominatim = use_nominatim,
            )
            rec = lner._expand(single_loc)
            if rec:
                st.success(f"**Expanded:** {rec.expanded}")
                col_l, col_r = st.columns(2)
                col_l.metric("State / Province", rec.admin1 or "—")
                col_l.metric("District", rec.admin2 or "—")
                col_r.metric("Latitude",  f"{rec.latitude:.5f}"  if rec.latitude  else "—")
                col_r.metric("Longitude", f"{rec.longitude:.5f}" if rec.longitude else "—")
                st.caption(f"Source: {rec.source}  |  Confidence: {rec.confidence:.2f}")
                if rec.latitude:
                    import pandas as _pd
                    st.map(_pd.DataFrame({"lat":[rec.latitude],"lon":[rec.longitude]}))
            else:
                st.warning(f"No expansion found for '{single_loc}'. Check GeoNames DB path.")

    st.divider()

    # ── Locality segregation demo ──────────────────────────────────────────────
    st.subheader("Locality Segregation")
    st.caption("Distinguish comma-separated sub-localities vs. multiple distinct sites")
    seg_input = st.text_input(
        "Comma-separated locality string:",
        value="Narara Island, Pirotan, Beyt Dwarka",
    )
    if seg_input:
        parts = segregate_locality_string(seg_input)
        if len(parts) == 1:
            st.info(f"🔗 **Single locality** (hierarchical context detected): `{parts[0]}`")
        else:
            st.success(f"📍 **{len(parts)} distinct sites** detected:")
            for i, p in enumerate(parts, 1):
                st.write(f"  {i}. {p}")

    st.divider()

    # ── Batch text extraction ──────────────────────────────────────────────────
    st.subheader("Extract Localities from Text")
    text_in = st.text_area(
        "Paste section text:",
        height=150,
        placeholder="Paste Methods or Results text to auto-extract localities…",
    )
    if st.button("Extract Localities") and text_in:
        with st.spinner("Extracting…"):
            lner = LocalityNER(
                geonames_db   = GEONAMES_DB if geonames_ok else "",
                use_nominatim = False,
            )
            recs = lner.extract_localities(text_in)

        if recs:
            st.success(f"Found {len(recs)} localities")
            rows = [r.to_dict() for r in recs]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No localities detected. Try a different text.")


# ─────────────────────────────────────────────────────────────────────────────
#  SCHEMA DIAGNOSTICS TAB
# ─────────────────────────────────────────────────────────────────────────────
def render_schema_diagnostics(
    error_log: list[str] | None = None,
    occurrences: list[dict] | None = None,
):
    """Render the Pydantic schema diagnostics tab."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.subheader("🛡️ Schema Validation & JSON Diagnostics")
    st.caption(
        "Pydantic v2 strict enforcement · json_repair auto-fix · "
        "Field normalisation (dates, depths, occurrence types)"
    )

    if not _SCHEMA_AVAILABLE:
        st.error("biotrace_schema.py required.")
        return

    col1, col2 = st.columns(2)
    with col1:
        import pydantic
        st.markdown(f"✅ **Pydantic** v{pydantic.__version__}")
    with col2:
        try:
            import json_repair
            st.markdown("✅ **json_repair** installed")
        except ImportError:
            st.markdown("❌ **json_repair** — `pip install json-repair`")

    # Error log
    if error_log:
        st.divider()
        st.subheader("Validation Errors (last extraction)")
        for e in error_log:
            if "[Schema]" in e:
                st.warning(e)
            else:
                st.info(e)
    elif error_log is not None:
        st.success("✅ No schema validation errors in last extraction")

    # Live JSON tester
    st.divider()
    st.subheader("Live JSON Repair Tester")
    raw_in = st.text_area(
        "Paste raw LLM output (may be malformed):",
        height=150,
        placeholder='```json\n[{"Recorded Name": "Acanthurus triostegus", ...}]',
    )
    if st.button("🔧 Parse & Validate") and raw_in:
        recs, errs = parse_llm_response(raw_in, source_citation="test")
        if recs:
            st.success(f"✅ {len(recs)} valid records")
            for r in recs:
                with st.expander(f"• {r.display_name}"):
                    d = r.model_dump()
                    st.json(d)
        if errs:
            for e in errs:
                st.warning(e)

    # Taxonomy completeness audit
    if occurrences:
        st.divider()
        st.subheader("Taxonomy Completeness Audit")
        import pandas as _pd
        df = _pd.DataFrame(occurrences)
        audit = {}
        for field in ["phylum","class_","order_","family_","wormsID","taxonomicStatus"]:
            if field in df.columns:
                n_filled = (df[field].fillna("") != "").sum()
                audit[field] = f"{n_filled}/{len(df)} ({100*n_filled//len(df)}%)"
        st.table(_pd.DataFrame(list(audit.items()), columns=["Field","Coverage"]))


# ─────────────────────────────────────────────────────────────────────────────
#  OLLAMA MODEL COMBOBOX  (used in sidebar)
# ─────────────────────────────────────────────────────────────────────────────
def render_ollama_model_selector(
    base_url: str = "http://localhost:11434",
    key: str = "ollama_model",
) -> str:
    """
    Render an Ollama model combobox in Streamlit sidebar.
    Fetches live model list; shows offline fallback if Ollama is down.
    Returns selected model name string.
    """
    try:
        import streamlit as st
        import requests as _req
    except ImportError:
        return "llama3.2"

    # Live fetch
    live_models: list[str] = []
    try:
        r = _req.get(f"{base_url}/api/tags", timeout=2)
        live_models = sorted(m["name"] for m in r.json().get("models",[]))
    except Exception:
        pass

    FALLBACK = [
        "gemma4","gemma3","gemma3:12b",
        "llama3.2","llama3.3","llama4:scout",
        "qwen2.5:7b","qwen3","mistral",
        "phi4","deepseek-r1:8b",
        "llava","llava:13b","moondream",
    ]

    all_models = live_models if live_models else FALLBACK
    label      = "Model" + (" (live)" if live_models else " (offline)")
    status     = "✅ Ollama live" if live_models else "⚠️ Ollama offline — fallback list"
    st.caption(status)

    # Combobox: selectbox with free text option
    selected = st.selectbox(label, all_models, key=key)

    # Custom model text field
    custom = st.text_input(
        "Or type a model name:",
        placeholder="e.g. llama3.2:70b",
        key=f"{key}_custom",
    )
    return custom.strip() if custom.strip() else selected


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE STREAMLIT APP (for testing)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import streamlit as st
    except ImportError:
        print("Install streamlit: pip install streamlit")
        raise SystemExit(1)

    st.set_page_config(
        page_title="BioTrace v5.2 — Enhancement Modules",
        page_icon="🔬",
        layout="wide",
    )
    st.title("🔬 BioTrace v5.2 — Enhancement Module Preview")
    st.caption("TNR · Locality NER · Pydantic Schema · Verification Table")

    tabs = st.tabs([
        "🔬 TNR Engine",
        "📍 Locality Expander",
        "🛡️ Schema Diagnostics",
        "✏️ Verification Table Demo",
    ])

    with tabs[0]:
        render_tnr_tab()

    with tabs[1]:
        render_locality_tab()

    with tabs[2]:
        render_schema_diagnostics()

    with tabs[3]:
        st.subheader("✏️ Verification Table Demo")
        st.caption("Paste JSON occurrence records to preview the editable table")

        demo_json = st.text_area(
            "Paste occurrence JSON array:",
            height=150,
            value=json.dumps([
                {
                    "recordedName": "Acanthurus triostegus",
                    "validName":    "Acanthurus triostegus",
                    "phylum":       "Chordata",
                    "class_":       "Actinopterygii",
                    "order_":       "Acanthuriformes",
                    "family_":      "Acanthuridae",
                    "verbatimLocality": "Narara Island",
                    "occurrenceType":   "Primary",
                    "wormsID":          "219635",
                    "matchScore":       0.98,
                    "rawTextEvidence":  "We collected Acanthurus triostegus from the intertidal reef at Narara Island.",
                },
                {
                    "recordedName": "Siganus javus",
                    "validName":    "Siganus javus",
                    "phylum":       "Chordata",
                    "class_":       "Actinopterygii",
                    "order_":       "Perciformes",
                    "family_":      "Siganidae",
                    "verbatimLocality": "Gulf of Kutch",
                    "occurrenceType":   "Secondary",
                    "wormsID":          "218032",
                    "matchScore":       0.92,
                    "rawTextEvidence":  "As reported by Pillai (1985), Siganus javus is common in Gulf of Kutch.",
                },
            ], indent=2),
        )

        if demo_json.strip():
            try:
                occs = json.loads(demo_json)
                _, updated = render_verification_table(occs, key_prefix="demo")
            except json.JSONDecodeError as e:
                st.error(f"JSON error: {e}")
