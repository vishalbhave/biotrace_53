"""
biotrace_dedup_patch.py  —  BioTrace v5.4  (patched)
────────────────────────────────────────────────────────────────────────────
Exports required by biotrace_v5.py and biotrace_traiter_prepass.py:

  LIFE_STAGE_TERMS            — frozenset  (used by traiter_prepass & filters)
  dedup_occurrences()         — Stage 1+2 intra-document deduplication
  suppress_regional_duplicates() — Stage 3 dedup (FIXED: checklist_mode param)

PATCH v5.4 changes vs previous version
────────────────────────────────────────
  • suppress_regional_duplicates() gains checklist_mode=False parameter.
    When True: preserves 'Genus cf. species', 'Genus sp.', and authority-
    suffixed forms as DISTINCT entries — correct for annotated checklist papers.
    When False (default): existing aggressive behaviour unchanged.

  • _canon() logic extracted into shared helper so both functions use
    the same normalisation (avoids divergence bugs).

  • dedup_occurrences() now also filters __candidate_* placeholder names
    produced by failed NER lookups.

ROOT CAUSE OF SUPPRESSED SPECIES (Gulf of Kutch checklist):
  'Berthellina citrina' was collapsed into 'Berthellina cf. citrina'
  'Elysia obtusa' was collapsed into 'Elysia obtusa Baba, 1938'
  'Gymnodoris alba' was collapsed into 'Gymnodoris sp.'
  'Plocamopherus ceylonicus' was fuzzy-matched to another Plocamopherus congener.
  Fix: pass checklist_mode=True (set via UI toggle in biotrace_v5.py).
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("biotrace.dedup")


# ─────────────────────────────────────────────────────────────────────────────
#  LIFE-STAGE TERMS
#  Used by: biotrace_traiter_prepass.py, post_parse_lifestage_filter()
# ─────────────────────────────────────────────────────────────────────────────

LIFE_STAGE_TERMS: frozenset[str] = frozenset({
    # Cnidarian / scyphozoan
    "medusa", "medusae", "ephyra", "ephyrae", "strobila", "strobilae",
    "scyphistoma", "scyphistomae", "polyp", "polyps",
    # Molluscan / opisthobranch
    "veliger", "veligers", "trochophore", "trochophores",
    "larva", "larvae", "larval",
    "juvenile", "juveniles",
    "adult", "adults",
    "egg", "eggs", "egg mass", "egg masses",
    "embryo", "embryos",
    "spat", "post-larva", "post-larvae", "post-larval",
    "recruit", "recruits",
    "metamorphose", "metamorphosis",
    # Crustacean / echinoderm
    "nauplius", "nauplii",
    "zoea", "zoeas", "megalopa",
    "pluteus", "bipinnaria", "auricularia",
    "doliolaria",
    # General
    "immature", "mature", "subadult",
    "hatchling", "hatchlings",
    "nymph", "nymphs",
    "pup", "pups", "fry",
    # Abbreviated size classes sometimes mistaken for taxa
    "small", "large", "tiny", "minute",
})


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED CANONICALISATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

# Regex: abbreviated genus "C. something" or just bare "C." — single capital + period
_ABBREV_GENUS_RE = re.compile(r"^[A-Z]\.\s*\w*$")

# Pattern to detect __candidate_* placeholders from failed NER
_CANDIDATE_RE = re.compile(r"^__candidate_\d+_\d+$")

# Open-nomenclature tokens
_OPEN_NOMEN_RE = re.compile(r"\b(?:cf|aff|sp|spp|ssp|n\.?\s*sp|var|subsp|f)\.\s*", re.I)
# Author + year: "Linnaeus, 1758" or "(Rüppell and Leuckart, 1828)"
_AUTHORITY_RE  = re.compile(
    r"[\(\[]?[A-Z][A-Za-zÀ-ÿ\-'']+(?:\s+(?:and|&|et)\s+[A-Z][A-Za-zÀ-ÿ\-'']+)?[,.]?\s*\d{4}[\)\]]?",
)


def _canon_name(name: str, strict: bool = False) -> str:
    """
    Canonicalise a scientific name for deduplication keying.

    strict=True  — keep cf./aff./sp./authority intact (checklist_mode).
                   Only lowercases and strips surrounding whitespace.
    strict=False — removes open nomenclature, authority strings, extra words.
                   Reduces 'Elysia obtusa Baba, 1938' → 'elysia obtusa'.
    """
    if not name:
        return ""
    name = name.strip()
    if not strict:
        # Strip authority (author + year)
        name = _AUTHORITY_RE.sub("", name)
        # Strip parenthetical subgenus
        name = re.sub(r"\s*\([^)]+\)", "", name)
        # Strip open nomenclature tokens
        name = _OPEN_NOMEN_RE.sub("", name)
        # Collapse whitespace
        name = re.sub(r"\s{2,}", " ", name).strip()
    return name.lower().strip(".,;- ")


def _canon_locality(loc: str) -> str:
    """Canonicalise a locality string for dedup keying."""
    if not loc:
        return ""
    loc = loc.lower().strip()
    # Remove punctuation variants
    loc = re.sub(r"[,;.]+", " ", loc)
    loc = re.sub(r"\s{2,}", " ", loc).strip()
    # Expand common abbreviations
    loc = loc.replace(" dist.", " district").replace(" dist ", " district ")
    return loc


# Occurrence-type priority: Primary wins over Secondary wins over Uncertain
_OCC_PRIORITY: dict[str, int] = {
    "primary":   0,
    "secondary": 1,
    "uncertain": 2,
    "":          3,
}


def _is_non_taxon(name: str) -> Optional[str]:
    """
    Return a reason string if `name` should be excluded as non-taxonomic,
    or None if it looks like a valid scientific name.

    Catches:
      • Life-stage terms (Scyphistoma, Medusa, Ephyra …)
      • Single abbreviated genus references ("C. andromeda")  — caller should
        resolve these rather than exclude; we return reason="abbreviation"
        to distinguish.
    """
    clean = name.strip()
    # Bare life stage
    if _canon(clean) in LIFE_STAGE_TERMS:
        return "life_stage"
    # First word is a life stage (e.g. "Scyphistoma polyp")
    first_word = clean.split()[0] if clean.split() else ""
    if _canon(first_word) in LIFE_STAGE_TERMS:
        return "life_stage"
    # Abbreviated genus: single capital + period (optionally followed by epithet)
    if _ABBREV_GENUS_RE.match(clean):
        return "abbreviation"
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1+2 — dedup_occurrences
#  Removes within-document duplicates produced when the same species+locality
#  appears in multiple sections (Introduction, Results, Discussion, captions).
# ─────────────────────────────────────────────────────────────────────────────

def dedup_occurrences(
    occurrences:    list[dict],
    keep_secondary: bool = True,
) -> tuple[list[dict], int]:
    """
    Stage 1+2 intra-document deduplication.

    Deduplication key:  (canon_name, canon_locality)
    Resolution priority: Primary > Secondary > Uncertain > ""
    Tiebreak: longer rawTextEvidence wins.

    When keep_secondary=True (default): unique Secondary records whose
    (name, locality) do NOT appear in any Primary record are retained.
    This preserves cited historical observations.

    Returns (deduplicated_list, n_removed).
    """
    if not occurrences:
        return occurrences, 0

    best: dict[str, dict] = {}   # key → best occurrence so far

    for occ in occurrences:
        if not isinstance(occ, dict):
            continue

        name = (occ.get("validName") or occ.get("recordedName") or
                occ.get("Recorded Name", "")).strip()

        # Drop __candidate_* placeholder names from failed NER passes
        if not name or _CANDIDATE_RE.match(name):
            continue

        loc  = str(occ.get("verbatimLocality") or "").strip()
        key  = f"{_canon_name(name)}||{_canon_locality(loc)}"

        occ_type_raw = str(occ.get("occurrenceType") or "").lower().strip()
        priority     = _OCC_PRIORITY.get(occ_type_raw, 3)

        if key not in best:
            best[key] = occ
            continue

        incumbent = best[key]
        inc_type  = str(incumbent.get("occurrenceType") or "").lower().strip()
        inc_prio  = _OCC_PRIORITY.get(inc_type, 3)

        # Primary beats anything lower-priority
        if priority < inc_prio:
            best[key] = occ
        elif priority == inc_prio:
            # Tiebreak: longer evidence string wins (more context)
            new_ev = len(str(occ.get("rawTextEvidence") or occ.get("Raw Text Evidence", "")))
            inc_ev = len(str(incumbent.get("rawTextEvidence") or
                            incumbent.get("Raw Text Evidence", "")))
            if new_ev > inc_ev:
                best[key] = occ

    result   = list(best.values())
    n_removed = len(occurrences) - len(result)

    if n_removed:
        logger.info(
            "[dedup 1+2] %d → %d records (%d duplicates removed)",
            len(occurrences), len(result), n_removed,
        )
    return result, n_removed


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 3 — suppress_regional_duplicates
#  Removes species that appear at both a finer locality AND the broader
#  regional study area when only one was actually recorded (e.g. "Gulf of Kutch"
#  as fallback locality assigned to a record already geocoded to "Narara").
#
#  CHECKLIST MODE FIX (v5.4):
#  When checklist_mode=True the canonicalisation keeps cf./sp./authority
#  intact so that table rows that are legitimately separate (e.g. Table 1
#  rows 5 and 6: "Berthellina citrina" ≠ "Berthellina cf. citrina") are
#  NOT collapsed.
# ─────────────────────────────────────────────────────────────────────────────

# Regional / study-area locality strings that are used as fallback placeholders.
# If a species has one record at a sub-locality AND one at the regional label,
# the regional-only record is suppressed (it's likely a duplicate at lower res).
_REGIONAL_PLACEHOLDERS: frozenset[str] = frozenset({
    "gulf of kutch", "gulf of mannar", "palk bay", "lakshadweep",
    "andaman islands", "andaman sea", "arabian sea", "bay of bengal",
    "indian ocean", "india", "gujarat", "tamil nadu", "kerala",
    "karnataka coast", "west coast of india", "east coast of india",
    "not reported", "unknown", "study area", "collection site",
})


def suppress_regional_duplicates(
    occurrences:    list[dict],
    checklist_mode: bool = False,
) -> tuple[list[dict], int]:
    """
    Stage 3 deduplication: remove regional-placeholder duplicates.

    A record is suppressed when:
      1. Its canonicalised locality is a known regional placeholder.
      2. Another record for the SAME species exists with a more specific
         (non-placeholder) locality.

    checklist_mode=True — use STRICT canonicalisation: 'Genus cf. species',
       'Genus sp.', and authority-suffixed forms are treated as DISTINCT
       from their plain-name counterparts.  Use for annotated checklist papers
       where the table explicitly lists open-nomenclature forms as separate rows.

    checklist_mode=False (default) — LOOSE canonicalisation: strips cf./aff./
       sp./authority before keying.  Preserves previous behaviour.

    Returns (filtered_list, n_suppressed).
    """
    if not occurrences:
        return occurrences, 0

    # Build two sets per species (strict or loose key depending on mode):
    #   specific_localities  — non-placeholder localities
    #   regional_localities  — placeholder localities
    specific_keys:  set[str] = set()
    regional_idxs:  list[int] = []

    for idx, occ in enumerate(occurrences):
        if not isinstance(occ, dict):
            continue
        name = (occ.get("validName") or occ.get("recordedName") or
                occ.get("Recorded Name", "")).strip()
        if not name or _CANDIDATE_RE.match(name):
            continue

        name_key = _canon_name(name, strict=checklist_mode)
        loc      = _canon_locality(str(occ.get("verbatimLocality") or ""))

        if loc in _REGIONAL_PLACEHOLDERS or not loc:
            regional_idxs.append(idx)
        else:
            specific_keys.add(name_key)

    # Suppress regional-only records that also appear at a specific locality
    suppressed = 0
    result: list[dict] = []

    for idx, occ in enumerate(occurrences):
        if not isinstance(occ, dict):
            result.append(occ)
            continue

        if idx in regional_idxs:
            name = (occ.get("validName") or occ.get("recordedName") or
                    occ.get("Recorded Name", "")).strip()
            name_key = _canon_name(name, strict=checklist_mode)
            if name_key in specific_keys:
                # Suppress — a more specific locality record exists
                logger.debug(
                    "[dedup 3] Suppressed regional duplicate: '%s' @ '%s'",
                    name, occ.get("verbatimLocality", ""),
                )
                suppressed += 1
                continue

        result.append(occ)

    if suppressed:
        logger.info(
            "[dedup 3] %d regional duplicates suppressed "
            "(checklist_mode=%s)",
            suppressed, checklist_mode,
        )
    return result, suppressed
