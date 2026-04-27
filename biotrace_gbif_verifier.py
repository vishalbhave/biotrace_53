"""
biotrace_gbif_verifier.py  —  BioTrace v5.4
────────────────────────────────────────────────────────────────────────────
GBIF Species API verifier with HITL approval integration.

References:
  https://techdocs.gbif.org/en/openapi/v1/species#/
  https://pygbif.readthedocs.io/en/latest/modules/species.html

Install:
  pip install pygbif requests

Usage:
  from biotrace_gbif_verifier import gbif_verify_batch, render_approval_table
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Optional

import requests

logger = logging.getLogger("biotrace.gbif")

# ─────────────────────────────────────────────────────────────────────────────
#  GBIF API constants
# ─────────────────────────────────────────────────────────────────────────────
GBIF_MATCH_URL  = "https://api.gbif.org/v1/species/match"
GBIF_SEARCH_URL = "https://api.gbif.org/v1/species"
GBIF_USAGE_URL  = "https://api.gbif.org/v1/species/{key}"
_TIMEOUT        = 10
_RATE_SLEEP     = 0.15   # 6-7 req/sec stays within GBIF's anonymous tier

# Minimum GBIF confidence to consider a match "approved"
DEFAULT_MIN_CONFIDENCE = 80


# ─────────────────────────────────────────────────────────────────────────────
#  Core match function
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def gbif_match_name(name: str, kingdom: str = "Animalia") -> dict:
    """
    Match a scientific name against the GBIF Backbone Taxonomy.

    Parameters
    ----------
    name    : scientific name (binomial or trinomial)
    kingdom : restrict match to a kingdom (default Animalia for marine fauna)

    Returns dict with keys:
      gbifKey, gbifName, gbifRank, gbifStatus, gbifConfidence,
      gbifMatchType, gbifPhylum, gbifClass, gbifOrder, gbifFamily,
      gbifKingdom, gbifProfileURL
    """
    try:
        params = {
            "name":    name,
            "kingdom": kingdom,
            "verbose": "false",
            "strict":  "false",
        }
        r = requests.get(GBIF_MATCH_URL, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        usage_key = data.get("usageKey", "")
        return {
            "gbifKey":       str(usage_key) if usage_key else "",
            "gbifName":      data.get("canonicalName") or data.get("scientificName", ""),
            "gbifRank":      data.get("rank", ""),
            "gbifStatus":    data.get("status", "UNKNOWN"),    # ACCEPTED | SYNONYM | DOUBTFUL
            "gbifConfidence":int(data.get("confidence", 0)),
            "gbifMatchType": data.get("matchType", "NONE"),    # EXACT | FUZZY | HIGHERRANK | NONE
            "gbifPhylum":    data.get("phylum", ""),
            "gbifClass":     data.get("class", ""),
            "gbifOrder":     data.get("order", ""),
            "gbifFamily":    data.get("family", ""),
            "gbifKingdom":   data.get("kingdom", ""),
            "gbifProfileURL":(
                f"https://www.gbif.org/species/{usage_key}"
                if usage_key else ""
            ),
        }
    except Exception as exc:
        logger.debug("[GBIF match] %s → %s", name, exc)
        return {
            "gbifKey":        "",
            "gbifName":       "",
            "gbifStatus":     "UNVERIFIED",
            "gbifConfidence": 0,
            "gbifMatchType":  "NONE",
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Batch verifier
# ─────────────────────────────────────────────────────────────────────────────

def gbif_verify_batch(
    occurrences:    list[dict],
    min_confidence: int  = DEFAULT_MIN_CONFIDENCE,
    kingdom:        str  = "Animalia",
    rate_sleep:     float = _RATE_SLEEP,
) -> list[dict]:
    """
    Enrich each occurrence dict with GBIF match fields.

    Adds to each occurrence:
      gbifKey, gbifName, gbifStatus, gbifConfidence, gbifMatchType,
      gbifPhylum, gbifClass, gbifOrder, gbifFamily, gbifProfileURL,
      gbifApproved  (bool — True only for ACCEPTED + confidence >= min_confidence)

    Parameters
    ----------
    occurrences    : list of occurrence dicts
    min_confidence : minimum GBIF confidence (0-100) to auto-approve
    kingdom        : GBIF kingdom filter
    rate_sleep     : delay between API calls (seconds)

    Returns the enriched occurrence list.
    """
    name_cache: dict[str, dict] = {}

    for occ in occurrences:
        if not isinstance(occ, dict):
            continue

        name = (occ.get("validName") or occ.get("recordedName") or
                occ.get("Recorded Name", "")).strip()

        # Strip open-nomenclature tokens before matching
        name_clean = name
        for tok in ("cf. ", "cf.", "aff. ", "aff.", " sp.", " spp.", " n. sp."):
            name_clean = name_clean.replace(tok, " ").strip()
        name_clean = name_clean.split("(")[0].strip()    # drop authority parentheticals

        if not name_clean:
            occ["gbifApproved"] = False
            continue

        # Use cache to avoid re-querying the same name
        if name_clean not in name_cache:
            name_cache[name_clean] = gbif_match_name(name_clean, kingdom)
            time.sleep(rate_sleep)

        result = name_cache[name_clean]
        occ.update(result)

        # Auto-approval rule: ACCEPTED + confidence threshold
        occ["gbifApproved"] = (
            result.get("gbifStatus") in ("ACCEPTED",)
            and result.get("gbifConfidence", 0) >= min_confidence
            and result.get("gbifMatchType") in ("EXACT", "FUZZY")
        )

        # If WoRMS ID is present but GBIF is SYNONYM, still allow approval
        # (WoRMS authority supersedes GBIF for marine species)
        if occ.get("wormsID") and result.get("gbifMatchType") in ("EXACT", "FUZZY"):
            occ["gbifApproved"] = True

        logger.debug(
            "[GBIF] %s → %s (conf=%d, match=%s, approved=%s)",
            name_clean,
            result.get("gbifStatus", "—"),
            result.get("gbifConfidence", 0),
            result.get("gbifMatchType", "—"),
            occ["gbifApproved"],
        )

    approved = sum(1 for o in occurrences if isinstance(o, dict) and o.get("gbifApproved"))
    logger.info("[GBIF] %d/%d species auto-approved (confidence ≥ %d)",
                approved, len(occurrences), min_confidence)
    return occurrences


# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit HITL approval table  (call from biotrace_v5.py)
# ─────────────────────────────────────────────────────────────────────────────

def render_approval_table(occurrences: list[dict]) -> list[dict] | None:
    """
    Render a Streamlit data_editor approval table.
    Returns the approved subset of occurrences, or None if the user
    has not yet confirmed (i.e., do not proceed yet).

    Wire into biotrace_v5.py after GBIF verification:

        from biotrace_gbif_verifier import render_approval_table
        approved = render_approval_table(occurrences)
        if approved is None:
            st.stop()   # wait for confirmation
        occurrences = approved
    """
    try:
        import streamlit as st
        import pandas as pd
    except ImportError:
        logger.error("[GBIF] Streamlit not available for approval table")
        return occurrences

    st.subheader("🔬 HITL Species Approval Gate")
    st.caption(
        "Review extracted species before they are saved to the database, "
        "knowledge graph, and memory bank. "
        "✅ = GBIF auto-approved | ⚠️ = needs review | ❌ = doubtful"
    )

    rows = []
    for i, occ in enumerate(occurrences):
        if not isinstance(occ, dict):
            continue

        confidence = occ.get("gbifConfidence", 0)
        status     = occ.get("gbifStatus", "—")
        match_type = occ.get("gbifMatchType", "—")
        auto_approved = occ.get("gbifApproved", False)

        # Visual status icon
        if auto_approved:
            icon = "✅"
        elif match_type == "NONE":
            icon = "❌"
        else:
            icon = "⚠️"

        rows.append({
            "approve":         auto_approved,
            "status":          f"{icon} {status}",
            "recordedName":    occ.get("recordedName") or occ.get("Recorded Name", ""),
            "gbifName":        occ.get("gbifName", ""),
            "wormsID":         occ.get("wormsID", ""),
            "gbifConf":        confidence,
            "family":          occ.get("family_") or occ.get("gbifFamily", ""),
            "locality":        occ.get("verbatimLocality", ""),
            "occType":         occ.get("occurrenceType", ""),
            "gbifURL":         occ.get("gbifProfileURL", ""),
            "_orig_idx":       i,
        })

    if not rows:
        st.warning("No occurrences to approve.")
        return []

    df = pd.DataFrame(rows)
    display_cols = ["approve", "status", "recordedName", "gbifName",
                    "wormsID", "gbifConf", "family", "locality", "occType"]

    edited = st.data_editor(
        df[display_cols],
        column_config={
            "approve": st.column_config.CheckboxColumn(
                "✅ Approve", default=True, help="Uncheck to exclude from DB/KG/Memory"
            ),
            "gbifConf": st.column_config.ProgressColumn(
                "GBIF conf.", min_value=0, max_value=100, format="%d%%"
            ),
            "status": st.column_config.TextColumn("GBIF Status", width="medium"),
        },
        use_container_width=True,
        hide_index=True,
        key="hitl_gbif_approval",
    )

    # Show GBIF profile links separately (data_editor doesn't render links)
    with st.expander("🔗 GBIF Profile Links"):
        for row in rows:
            if row["gbifURL"]:
                st.markdown(
                    f"- **{row['recordedName']}**: "
                    f"[GBIF {row['_orig_idx']+1}]({row['gbifURL']})"
                )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        confirmed = st.button(
            f"✅ Confirm {edited['approve'].sum()} species",
            type="primary",
            key="hitl_confirm_btn",
        )
    with col_b:
        st.caption(
            f"{edited['approve'].sum()} approved · "
            f"{(~edited['approve']).sum()} excluded · "
            f"{len(rows)} total"
        )

    if not confirmed:
        return None   # signal: don't proceed yet

    approved_indices = edited.index[edited["approve"]].tolist()
    approved_orig_indices = [rows[i]["_orig_idx"] for i in approved_indices]
    approved_occurrences  = [occurrences[i] for i in approved_orig_indices]

    st.success(
        f"✅ {len(approved_occurrences)} species approved for database insertion."
    )
    return approved_occurrences
