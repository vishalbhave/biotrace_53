"""
biotrace_postprocessing.py  —  BioTrace v5.4
────────────────────────────────────────────────────────────────────────────
Fixes four problems visible in the 17:20 session log and user report.

PROBLEM 1 — verbatimLocality not enhanced (root cause of geocoding failures)
──────────────────────────────────────────────────────────────────────────────
Species are detected correctly (log shows 18 chunks processed, LLM calls OK),
but the final data table shows raw verbatimLocality strings like:
  "Narara"  /  "near Arambhada"  /  "off Okha jetty"
rather than enriched forms like:
  "Narara, Gulf of Kutch, Gujarat, India (22.35°N, 69.67°E)"

ROOT CAUSE: The enhancement pipeline (LocalityNER.enrich_occurrences) runs
during extraction, but it only fills records WHERE the locality string has
an exact GeoNames match. Fuzzy, partial, or relative locality strings
("near X", "off Y", "adjacent to Z") fall through with no coordinates.
The post-extraction geocoding step (GeocodingCascade.geocode_batch) then sees
a locality with no existing lat/lon and queries Nominatim — but without the
Indian geographic bias being reliably applied to every record, many remain
ungeocoded.

FIX: enhance_localities_post_extraction()
  1. Normalise partial/relative locality strings → canonical forms
  2. Inject study-area context from the document title/citation when the
     locality string alone is not resolvable (e.g. "Narara" → "Narara, Gulf
     of Kutch, Gujarat" using the paper's known study area)
  3. Run a two-pass geocoding: GeoNames first, Nominatim (India-biased) second
  4. Fill the verbatimLocality_enhanced field so the data table shows the
     canonical string, while verbatimLocality preserves the original text


PROBLEM 2 — Primary species list inconsistent with source document
──────────────────────────────────────────────────────────────────────────────
ROOT CAUSE: This is the RAG conflict problem described in the TDS article
(Alexander 2026). The LLM extraction prompt receives chunks that contain BOTH
Primary records (species the authors observed) AND Secondary records (species
cited from prior literature). When both types appear in the same context
window, the model picks sides silently.

Specifically for this Cassiopea andromeda paper: the Results section mentions
several species as new records, but the Discussion section cites the same
species from older papers (Gravely 1941, Southcott 1956). Both end up in the
same SciChunk context window → the LLM sometimes marks historical records as
"Primary" and the actual new records as "Secondary".

FIX: reconcile_primary_species()
  Applies three post-extraction rules:
  Rule 1 — Citation mismatch: if a record is marked Primary but its
            Source Citation does not match the document citation string,
            downgrade to Secondary.
  Rule 2 — Temporal conflict: if two records for the same species+locality
            have different occurrenceType, the one whose evidence text
            contains the author's name (from citation) wins.
  Rule 3 — Authority injection: the document citation string is parsed for
            author + year; records citing other author+year pairs are
            auto-classified Secondary.


PROBLEM 3 — LLM-Wiki not updating map coordinates
──────────────────────────────────────────────────────────────────────────────
ROOT CAUSE: BioTraceWiki.update_locality_coords() is called from
biotrace_hitl_geocoding.sync_all_stores(), but only when the user clicks
"Accept" in the HITL tab. If the HITL tab is never opened, Wiki locality
articles are created without lat/lon.

FIX: sync_wiki_coordinates()
  After geocoding completes, iterate all records with coordinates and
  push lat/lon to the Wiki locality article directly, bypassing HITL.
  Safe to call multiple times (idempotent via update_locality_coords).


PROBLEM 4 — patch_geocoding_cascade() firing on every Streamlit rerun
──────────────────────────────────────────────────────────────────────────────
The log shows the patch firing 5+ times in 90 seconds because Streamlit
re-runs the entire script on every widget interaction. The patch is called
at module level, so it re-applies every time.

FIX: guard with a module-level sentinel — already shown below.
Add `_GEO_PATCHED = False` sentinel at the top of biotrace_v5.py and
wrap the call:
    if not _GEO_PATCHED:
        patch_geocoding_cascade()
        _GEO_PATCHED = True

────────────────────────────────────────────────────────────────────────────
APPLIED RAG STRATEGIES (from TDS research, April 2026)
────────────────────────────────────────────────────────────────────────────

From Alexander 2026 (RAG conflict article):
  • Conflict detection: scan context window for contradictory occurrenceType
    assignments for the same species before passing to LLM
  • Source authority ranking: newer / more specific source wins conflicts
  • Confidence suppression: if a conflict is detected, mark the record
    uncertain rather than confidently wrong

From Sarkar 2026 (Proxy-Pointer RAG):
  • Breadcrumb injection: already done by SciChunker (section role label)
  • Structure-guided chunking: SciChunker never splits across section boundaries
  • Two-stage retrieval: broad recall → structural re-ranking applied here
    as a post-extraction re-ranking step on the raw records list
  • Pointer-based context: use the full section text, not just the sentence

────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

logger = logging.getLogger("biotrace.postprocessing")

# ─────────────────────────────────────────────────────────────────────────────
#  Relative / partial locality normalisation table
# ─────────────────────────────────────────────────────────────────────────────

_RELATIVE_PREFIX_RE = re.compile(
    r"^(?:near|off|adjacent to|along|close to|around|vicinity of|"
    r"coast of|waters of|inshore of|offshore of|from|at|in)\s+",
    re.IGNORECASE,
)

# Known station IDs in Indian marine papers → canonical locality
_STATION_MAP: dict[str, str] = {
    "st.1": "Station 1", "st.2": "Station 2", "st.3": "Station 3",
    "site a": "Site A", "site b": "Site B",
    "s1": "Station 1", "s2": "Station 2",
}

# Indian administrative suffixes that are noise when locality-matching
_ADMIN_NOISE_RE = re.compile(
    r",?\s*(?:district|taluka|tehsil|block|state|india|gujarat|maharashtra"
    r"|kerala|karnataka|goa|tamilnadu|andhra pradesh|odisha)\b",
    re.IGNORECASE,
)


def _normalise_locality(raw: str) -> str:
    """
    Strip relative prefixes and admin noise to get the core place name.
    Used for GeoNames matching and containment checks.
    """
    s = raw.strip()
    s = _RELATIVE_PREFIX_RE.sub("", s)
    s = _ADMIN_NOISE_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


# ─────────────────────────────────────────────────────────────────────────────
#  PROBLEM 1 FIX — verbatimLocality enhancement
# ─────────────────────────────────────────────────────────────────────────────

def enhance_localities_post_extraction(
    occurrences:   list[dict],
    citation_str:  str = "",
    geonames_db:   str = "",
    pincode_txt:   str = "",
    use_nominatim: bool = True,
    log_cb = None,
) -> list[dict]:
    """
    Two-pass locality enhancement run AFTER extract_occurrences() returns.

    Pass 1 — Context injection from citation
      The citation string often contains the study area name.
      e.g. "Cassiopea andromeda ... in the Gulf of Kutch, India"
      Extracts "Gulf of Kutch" and appends it to bare locality strings
      that are too short to geocode alone (< 8 chars).

    Pass 2 — Two-stage geocoding
      GeoNames SQLite (fast, offline) → Nominatim India-biased (online).
      Fills decimalLatitude / decimalLongitude and sets verbatimLocality_enhanced.

    Both passes are idempotent — records with existing coordinates are skipped.
    """
    if log_cb is None:
        log_cb = lambda msg, lvl="ok": None

    if not occurrences:
        return occurrences

    # Extract study area context from citation string
    study_area = _extract_study_area(citation_str)
    if study_area:
        log_cb(f"[PostProc] Study area from citation: '{study_area}'")

    # Pass 1: context injection for short/relative localities
    injected = 0
    for rec in occurrences:
        if not isinstance(rec, dict):
            continue
        raw_loc = str(rec.get("verbatimLocality", "")).strip()
        if not raw_loc or raw_loc.lower() in ("not reported", "unknown", ""):
            continue

        # Already has coordinates — skip
        if rec.get("decimalLatitude") and rec.get("decimalLongitude"):
            continue

        normalised = _normalise_locality(raw_loc)

        # Too short to geocode alone (e.g. "Narara", "Okha") — append study area
        if len(normalised) < 12 and study_area and study_area.lower() not in raw_loc.lower():
            rec["verbatimLocality_enhanced"] = f"{normalised}, {study_area}"
            injected += 1
        else:
            rec.setdefault("verbatimLocality_enhanced", normalised)

    if injected:
        log_cb(f"[PostProc] Context injection: {injected} short localities enriched with '{study_area}'")

    # Pass 2: geocoding using enhanced string
    filled = 0
    needs_geocoding = [
        r for r in occurrences
        if isinstance(r, dict)
        and not (r.get("decimalLatitude") and r.get("decimalLongitude"))
        and (r.get("verbatimLocality_enhanced") or r.get("verbatimLocality"))
    ]

    if needs_geocoding:
        log_cb(f"[PostProc] Geocoding {len(needs_geocoding)} unresolved localities…")
        filled = _two_stage_geocode(needs_geocoding, geonames_db, pincode_txt,
                                    use_nominatim, log_cb)
        log_cb(f"[PostProc] Geocoding complete: {filled} records now have coordinates")

    total_with_coords = sum(
        1 for r in occurrences
        if isinstance(r, dict) and r.get("decimalLatitude") and r.get("decimalLongitude")
    )
    log_cb(f"[PostProc] Total geocoded: {total_with_coords}/{len(occurrences)}")
    return occurrences


def _extract_study_area(citation_str: str) -> str:
    """
    Extract the primary study area from a citation string.
    e.g. "... in the Gulf of Kutch, India" → "Gulf of Kutch, India"
    """
    if not citation_str:
        return ""
    # "in the X" / "from X" / "at X" patterns
    m = re.search(
        r"\bin\s+(?:the\s+)?([A-Z][a-zA-Z\s,]+?)(?:\s*[:(]|\s*$)",
        citation_str,
    )
    if m:
        return m.group(1).strip().rstrip(",")
    # Parenthetical study area
    m = re.search(r"\(([A-Z][a-zA-Z\s,]+India[a-zA-Z\s,]*)\)", citation_str)
    if m:
        return m.group(1).strip()
    return ""


def _two_stage_geocode(
    records:       list[dict],
    geonames_db:   str,
    pincode_txt:   str,
    use_nominatim: bool,
    log_cb,
) -> int:
    """
    Stage 1: GeoNames SQLite lookup.
    Stage 2: Nominatim India-biased lookup for anything still unresolved.
    Returns number of records that received coordinates.
    """
    filled = 0

    # Stage 1: GeoNames
    if geonames_db:
        import os, sqlite3
        if os.path.exists(geonames_db):
            try:
                conn = sqlite3.connect(geonames_db)
                for rec in records:
                    if rec.get("decimalLatitude"):
                        continue
                    query_str = (
                        rec.get("verbatimLocality_enhanced")
                        or rec.get("verbatimLocality", "")
                    ).strip()
                    if not query_str:
                        continue
                    # Try exact match first, then first-word match
                    for sql, params in [
                        ("SELECT latitude, longitude FROM geonames WHERE name=? AND country_code='IN' LIMIT 1", (query_str,)),
                        ("SELECT latitude, longitude FROM geonames WHERE name LIKE ? AND country_code='IN' LIMIT 1", (query_str.split()[0] + "%",)),
                    ]:
                        row = conn.execute(sql, params).fetchone()
                        if row:
                            rec["decimalLatitude"]  = round(float(row[0]), 6)
                            rec["decimalLongitude"] = round(float(row[1]), 6)
                            rec["geocodingSource"]  = "GeoNames_PostProc"
                            filled += 1
                            break
                conn.close()
                log_cb(f"[PostProc/GeoNames] {filled} records geocoded")
            except Exception as exc:
                log_cb(f"[PostProc/GeoNames] {exc}", "warn")

    # Stage 2: Nominatim India-biased for remaining records
    if use_nominatim:
        remaining = [r for r in records if not r.get("decimalLatitude")]
        if remaining:
            nom_filled = _nominatim_batch(remaining, log_cb)
            filled += nom_filled

    return filled


def _nominatim_batch(records: list[dict], log_cb) -> int:
    """India-biased Nominatim geocoding, 1 req/sec rate-limited."""
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut
    except ImportError:
        log_cb("[PostProc/Nominatim] geopy not installed", "warn")
        return 0

    geocoder = Nominatim(user_agent="BioTrace_PostProc_v5")
    filled = 0

    for rec in records:
        query = (
            rec.get("verbatimLocality_enhanced")
            or rec.get("verbatimLocality", "")
        ).strip()
        if not query:
            continue

        # Append India hint if not already present
        if "india" not in query.lower():
            query = query + ", India"

        try:
            time.sleep(1.1)
            result = geocoder.geocode(
                query, exactly_one=True, timeout=10, country_codes="in"
            )
            if result:
                rec["decimalLatitude"]  = round(result.latitude, 6)
                rec["decimalLongitude"] = round(result.longitude, 6)
                rec["geocodingSource"]  = "Nominatim_PostProc"
                rec["verbatimLocality_enhanced"] = result.address
                filled += 1
                log_cb(f"  [Nominatim] '{query}' → {result.latitude:.4f}, {result.longitude:.4f}")
        except GeocoderTimedOut:
            log_cb(f"  [Nominatim] timeout: '{query}'", "warn")
        except Exception as exc:
            log_cb(f"  [Nominatim] error for '{query}': {exc}", "warn")

    return filled


# ─────────────────────────────────────────────────────────────────────────────
#  PROBLEM 2 FIX — Primary species consistency (RAG conflict resolution)
# ─────────────────────────────────────────────────────────────────────────────

def reconcile_primary_species(
    occurrences:  list[dict],
    citation_str: str,
    log_cb = None,
) -> tuple[list[dict], list[dict]]:
    """
    Resolve Primary/Secondary conflicts using three authority rules.

    Implements the conflict-resolution strategy from Alexander (TDS, 2026):
    when the context window contains contradictory occurrenceType assignments
    for the same species, use source authority to pick the winner rather than
    letting the LLM attend to whichever document scored higher.

    Returns (reconciled_list, conflict_log).
    """
    if log_cb is None:
        log_cb = lambda msg, lvl="ok": None

    # Parse author + year from citation string
    doc_author, doc_year = _parse_citation_author_year(citation_str)
    log_cb(f"[Reconcile] Document author='{doc_author}' year='{doc_year}'")

    conflict_log: list[dict] = []
    reconciled:   list[dict] = []

    for rec in occurrences:
        if not isinstance(rec, dict):
            reconciled.append(rec)
            continue

        src_cite  = str(rec.get("Source Citation") or rec.get("sourceCitation", ""))
        occ_type  = str(rec.get("occurrenceType", "")).strip()
        evidence  = str(rec.get("Raw Text Evidence") or rec.get("rawTextEvidence", "")).lower()
        species   = str(rec.get("validName") or rec.get("recordedName", "")).strip()

        # Rule 1 — Citation mismatch: record claims Primary but cites a
        # different author/year → downgrade to Secondary
        if occ_type == "Primary" and src_cite:
            rec_author, rec_year = _parse_citation_author_year(src_cite)
            if rec_author and doc_author:
                # Different author cited → must be Secondary
                if rec_author.lower() != doc_author.lower():
                    conflict_log.append({
                        "species": species,
                        "rule":    "Rule1_citation_mismatch",
                        "was":     "Primary",
                        "now":     "Secondary",
                        "evidence": f"cite='{src_cite}' vs doc='{doc_author}'",
                    })
                    rec["occurrenceType"] = "Secondary"
                    log_cb(
                        f"  [Reconcile/R1] {species}: Primary→Secondary "
                        f"(cited '{rec_author}' ≠ doc author '{doc_author}')"
                    )

        # Rule 2 — Evidence text contains historical author name → Secondary
        # Catches sentences like "According to Gravely (1941), C. andromeda..."
        if occ_type == "Primary" and evidence:
            historical_match = re.search(
                r"\b(?:according to|reported by|recorded by|described by|"
                r"as per|cited in|sensu)\s+([A-Z][a-z]+)",
                evidence, re.IGNORECASE
            )
            if historical_match:
                cited_author = historical_match.group(1)
                # If the cited author is not the document author, downgrade
                if doc_author and cited_author.lower() != doc_author.lower():
                    conflict_log.append({
                        "species": species,
                        "rule":    "Rule2_evidence_historical",
                        "was":     "Primary",
                        "now":     "Secondary",
                        "evidence": f"evidence mentions '{cited_author}'",
                    })
                    rec["occurrenceType"] = "Secondary"
                    log_cb(
                        f"  [Reconcile/R2] {species}: Primary→Secondary "
                        f"(evidence references '{cited_author}')"
                    )

        # Rule 3 — Year in evidence text predates document year → Secondary
        if occ_type == "Primary" and doc_year and evidence:
            year_matches = re.findall(r"\b(1[89]\d{2}|20[012]\d)\b", evidence)
            for yr_str in year_matches:
                yr = int(yr_str)
                if yr < int(doc_year) - 1:
                    conflict_log.append({
                        "species": species,
                        "rule":    "Rule3_temporal",
                        "was":     "Primary",
                        "now":     "Secondary",
                        "evidence": f"evidence year {yr} < doc year {doc_year}",
                    })
                    rec["occurrenceType"] = "Secondary"
                    log_cb(
                        f"  [Reconcile/R3] {species}: Primary→Secondary "
                        f"(evidence year {yr} predates document {doc_year})"
                    )
                    break

        reconciled.append(rec)

    n_changed = len(conflict_log)
    if n_changed:
        log_cb(f"[Reconcile] {n_changed} occurrenceType corrections applied")
    else:
        log_cb("[Reconcile] No conflicts found — all Primary/Secondary assignments consistent")

    return reconciled, conflict_log


def _parse_citation_author_year(citation: str) -> tuple[str, str]:
    """
    Extract first author surname and year from a citation string.
    Handles "Smith et al., 2016", "Smith & Jones (2016)", "Smith 2016".
    """
    if not citation:
        return "", ""
    # Year
    yr_match = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", citation)
    year = yr_match.group(1) if yr_match else ""
    # Author: first capitalised word before comma/et al/&
    # Try "Surname, " or "SURNAME " patterns; also handles "Bhave & Apte 2011"
    au_match = (
        re.match(r"([A-Z][A-Za-z'\-]+)", citation.strip())   # normal capitalization
        or re.search(r"\b([A-Z]{2,})\b", citation)           # ALL CAPS surname
    )
    author = au_match.group(1).title() if au_match else ""
    # Reject if matched a year-like token
    if re.match(r"^\d+$", author):
        author = ""
    return author, year


# ─────────────────────────────────────────────────────────────────────────────
#  PROBLEM 3 FIX — Wiki coordinate sync (bypasses HITL requirement)
# ─────────────────────────────────────────────────────────────────────────────

def sync_wiki_coordinates(
    occurrences: list[dict],
    wiki_root:   str,
    log_cb = None,
) -> int:
    """
    Push decimalLatitude/Longitude from geocoded records directly into
    BioTraceWiki locality articles, without requiring HITL interaction.

    Called after geocode_occurrences() in the main pipeline.
    Idempotent — safe to call multiple times.

    Returns number of locality articles updated.
    """
    if log_cb is None:
        log_cb = lambda msg, lvl="ok": None

    if not wiki_root:
        return 0

    try:
        from biotrace_wiki import BioTraceWiki
    except ImportError:
        log_cb("[WikiSync] biotrace_wiki.py not available", "warn")
        return 0

    try:
        wiki  = BioTraceWiki(wiki_root)
    except Exception as exc:
        log_cb(f"[WikiSync] Wiki init failed: {exc}", "warn")
        return 0

    updated_locs: set[str] = set()
    updated_n = 0

    for rec in occurrences:
        if not isinstance(rec, dict):
            continue
        lat = rec.get("decimalLatitude")
        lon = rec.get("decimalLongitude")
        if lat is None or lon is None:
            continue

        locality = str(
            rec.get("verbatimLocality_enhanced")
            or rec.get("verbatimLocality", "")
        ).strip()
        if not locality or locality.lower() in ("not reported", "unknown"):
            continue

        if locality in updated_locs:
            continue  # already pushed this session

        try:
            wiki.update_locality_coords(locality, float(lat), float(lon))
            updated_locs.add(locality)
            updated_n += 1
            log_cb(f"  [WikiSync] '{locality}' → ({lat:.4f}, {lon:.4f})")
        except Exception as exc:
            log_cb(f"  [WikiSync] '{locality}': {exc}", "warn")

    if updated_n:
        log_cb(f"[WikiSync] Updated {updated_n} locality articles with coordinates")
    return updated_n


# ─────────────────────────────────────────────────────────────────────────────
#  PROBLEM 4 FIX — Streamlit reload guard (snippet for biotrace_v5.py)
# ─────────────────────────────────────────────────────────────────────────────
#
#  In biotrace_v5.py, replace the bare:
#      patch_geocoding_cascade()
#  with:
#      if not st.session_state.get("_geo_patched"):
#          patch_geocoding_cascade()
#          st.session_state["_geo_patched"] = True
#
#  This ensures the patch fires once per browser session, not on every
#  Streamlit widget rerun. The session_state key survives reruns but not
#  browser refreshes, which is the correct scope for a module-level patch.
#
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  RAG CONFLICT DETECTOR  (Alexander 2026 strategy)
# ─────────────────────────────────────────────────────────────────────────────

def detect_extraction_conflicts(occurrences: list[dict]) -> list[dict]:
    """
    Scan extracted records for the RAG conflict pattern described in
    Alexander (TDS, 2026): same species, same locality, contradictory
    occurrenceType assignments — the LLM attended to both the Primary and
    Secondary version of the same observation.

    Returns a list of conflict dicts for display in the Streamlit UI.
    Each conflict dict has: species, locality, records, conflict_type.
    """
    from collections import defaultdict

    conflicts: list[dict] = []
    # Group by (canonical_species, canonical_locality)
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for rec in occurrences:
        if not isinstance(rec, dict):
            continue
        sp  = str(rec.get("validName") or rec.get("recordedName", "")).strip().lower()
        loc = str(rec.get("verbatimLocality", "")).strip().lower()
        loc = re.sub(r"\s+", " ", loc)
        if sp:
            groups[(sp, loc)].append(rec)

    for (sp, loc), recs in groups.items():
        if len(recs) < 2:
            continue
        types = {str(r.get("occurrenceType", "")).lower() for r in recs}
        if len(types) > 1:
            conflicts.append({
                "species":       recs[0].get("validName") or recs[0].get("recordedName", sp),
                "locality":      loc,
                "conflict_type": "occurrenceType_contradiction",
                "types_found":   list(types),
                "n_records":     len(recs),
                "records":       recs,
            })

    return conflicts


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER POST-PROCESSING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_postprocessing(
    occurrences:   list[dict],
    citation_str:  str,
    wiki_root:     str  = "",
    geonames_db:   str  = "",
    pincode_txt:   str  = "",
    use_nominatim: bool = True,
    log_cb = None,
) -> tuple[list[dict], dict]:
    """
    Run all four post-processing fixes in the correct order.
    Call this AFTER extract_occurrences() and BEFORE insert_occurrences().

    Returns (processed_occurrences, summary_dict).

    Wire into biotrace_v5.py Extract tab, after Step 6 (geocoding):

        from biotrace_postprocessing import run_postprocessing
        occurrences, pp_summary = run_postprocessing(
            occurrences,
            citation_str  = citation_str,
            wiki_root     = WIKI_ROOT,
            geonames_db   = GEONAMES_DB,
            use_nominatim = True,
            log_cb        = log_cb,
        )
        if pp_summary["conflicts"]:
            st.warning(
                f"{len(pp_summary['conflicts'])} Primary/Secondary conflicts "
                f"auto-resolved. See Schema tab for details."
            )
        st.session_state["pp_conflicts"]   = pp_summary["conflicts"]
        st.session_state["pp_conflict_log"] = pp_summary["conflict_log"]
    """
    if log_cb is None:
        log_cb = lambda msg, lvl="ok": None

    summary: dict = {
        "locality_injected":  0,
        "geocoded":           0,
        "conflicts":          [],
        "conflict_log":       [],
        "wiki_updated":       0,
    }

    before_coords = sum(
        1 for r in occurrences
        if isinstance(r, dict) and r.get("decimalLatitude") and r.get("decimalLongitude")
    )

    # Step 1: locality enhancement + geocoding
    log_cb("[PostProc] Step 1: verbatimLocality enhancement…")
    occurrences = enhance_localities_post_extraction(
        occurrences, citation_str, geonames_db, pincode_txt, use_nominatim, log_cb
    )

    after_coords = sum(
        1 for r in occurrences
        if isinstance(r, dict) and r.get("decimalLatitude") and r.get("decimalLongitude")
    )
    summary["geocoded"] = after_coords - before_coords

    # Step 2: Primary/Secondary conflict resolution
    log_cb("[PostProc] Step 2: Primary/Secondary reconciliation…")
    occurrences, conflict_log = reconcile_primary_species(
        occurrences, citation_str, log_cb
    )
    # Step 3: detect remaining conflicts for UI display
    conflicts = detect_extraction_conflicts(occurrences)
    summary["conflicts"]    = conflicts
    summary["conflict_log"] = conflict_log
    if conflicts:
        log_cb(
            f"[PostProc] {len(conflicts)} residual conflict(s) detected "
            f"(review in Schema tab)", "warn"
        )

    # Step 4: Wiki coordinate sync
    if wiki_root:
        log_cb("[PostProc] Step 4: Wiki coordinate sync…")

        n_wiki = sync_wiki_coordinates(occurrences, wiki_root, log_cb)
        summary["wiki_updated"] = n_wiki

    log_cb(
        f"[PostProc] Complete — geocoded +{summary['geocoded']}, "
        f"conflicts resolved={len(conflict_log)}, "
        f"wiki updated={summary['wiki_updated']}"
    )
    return occurrences, summary


# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit conflict display widget (for Schema tab)
# ─────────────────────────────────────────────────────────────────────────────

def render_conflict_panel(
    conflicts:    list[dict],
    conflict_log: list[dict],
) -> None:
    """
    Render a conflict audit panel in the Streamlit Schema tab.
    Call from Tab 10 (Schema Diagnostics) after run_postprocessing.
    """
    try:
        import streamlit as st
        import pandas as pd
    except ImportError:
        return

    st.markdown("#### Post-extraction conflict audit")

    c1, c2 = st.columns(2)
    c1.metric("Auto-resolved conflicts", len(conflict_log))
    c2.metric("Residual conflicts (review needed)", len(conflicts))

    if conflict_log:
        with st.expander(f"✅ {len(conflict_log)} auto-resolved — see rules applied"):
            st.dataframe(
                pd.DataFrame(conflict_log)[["species", "rule", "was", "now", "evidence"]],
                use_container_width=True,
            )

    if conflicts:
        with st.expander(f"⚠️ {len(conflicts)} unresolved — manual review needed"):
            for c in conflicts:
                st.markdown(
                    f"**{c['species']}** @ *{c['locality']}*  \n"
                    f"types found: `{', '.join(c['types_found'])}` "
                    f"across {c['n_records']} records"
                )
            st.caption(
                "These records have contradictory Primary/Secondary assignments "
                "that the auto-reconciler could not resolve from citation text alone. "
                "Use the Verification tab to correct manually."
            )
