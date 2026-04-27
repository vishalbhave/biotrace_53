"""
biotrace_locality_guard_patch.py  —  BioTrace v5.4
────────────────────────────────────────────────────────────────────────────
Fixes two classes of bad verbatimLocality values that the existing patches
do not catch:

  CLASS A — MORPHOLOGY AS LOCALITY
    Source:  LLM grabs descriptive/diagnosis text when the document section
             has no named place nearby.
    Example: "Umbrella circular, flattened, 10 cm to 15 cm in diameter,
              oral arms 8–9 in number, branched…"
    Signal:  Contains measurement units (cm, mm, m) OR anatomical keywords.

  CLASS B — HABITAT DESCRIPTOR AS LOCALITY
    Source:  LLM uses a habitat phrase when the prompt forces non-blank output.
    Example: "intertidal area of dead coral reef"
    Signal:  Contains only habitat keywords, zero geographic proper nouns.

ROOT CAUSE IN PROMPT
────────────────────
The verbatimLocality field instruction currently reads:
    "CRITICAL: Cannot be blank or Unknown. … If no micro-locality, use the
     broadest study area from Title/Introduction."

This creates a coercive fallback that makes the LLM fill the field with
ANY nearby text when no named place is present.

TWO-PART FIX
────────────
PART 1 — PROMPT_LOCALITY_GUARD
    Splice into _SCHEMA_PROMPT to replace the verbatimLocality definition.
    Adds explicit prohibitions with examples.

PART 2 — post_parse_locality_filter()
    Post-LLM filter (run after post_parse_lifestage_filter, before dedup).
    Classifies each verbatimLocality string and either:
      • Quarantines CLASS A into a `_morphology_text` side-field and
        sets verbatimLocality = "Not Reported" (record is kept; HITL can
        decide later).
      • Moves CLASS B text to the `habitat` field if habitat is blank,
        and sets verbatimLocality = "Not Reported".
    Records with verbatimLocality = "Not Reported" are NOT geocoded.

HOW TO INTEGRATE INTO biotrace_v5.py
────────────────────────────────────
# 1. Import (add alongside existing patch imports at top of file)
from biotrace_locality_guard_patch import (
    PROMPT_LOCALITY_GUARD,
    post_parse_locality_filter,
)

# 2. Splice the prompt guard ONCE at startup (add after the lifestage guard
#    splice, around line 803):
if PROMPT_LOCALITY_GUARD not in _SCHEMA_PROMPT:
    _SCHEMA_PROMPT = _SCHEMA_PROMPT.replace(
        '  \"verbatimLocality\"   — Place name exactly as written.',
        PROMPT_LOCALITY_GUARD,
    )

# 3. Call filter inside _process_batch_text(), after post_parse_lifestage_filter:
#    (approximately line 981 — add immediately after the LS-filter block)
if data:
    data, loc_quarantined = post_parse_locality_filter(data)
    if loc_quarantined:
        log_cb(f"  [Loc-filter] {len(loc_quarantined)} morphology/habitat localities quarantined")
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("biotrace.locality_guard")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 1 — Prompt guard replacement for verbatimLocality field
# ─────────────────────────────────────────────────────────────────────────────
#  This string REPLACES the existing verbatimLocality line in _SCHEMA_PROMPT.
#  The splice target in biotrace_v5.py is:
#      '  "verbatimLocality"   — Place name exactly as written.'

PROMPT_LOCALITY_GUARD = '''\
  "verbatimLocality"   — GEOGRAPHIC PLACE NAME only. Must be a named location
                         (coastal site, island, bay, estuary, city, district,
                         GPS station ID, reef name, or similar proper place).

                         ✗ FORBIDDEN — do NOT put any of these in verbatimLocality:
                           • Morphological descriptions:
                               "Umbrella circular, flattened, 10 cm to 15 cm…"
                               "oral arms 8–9 in number, branched, with filaments"
                           • Habitat descriptors without a named place:
                               "intertidal area of dead coral reef"
                               "subtidal rocky substrate"
                               "open sandy bottom"
                           • Depth or measurement strings alone:
                               "0–5 m depth", "15 cm below MHWL"
                           • Behavioural or ecological phrases:
                               "found in association with seagrass"

                         ✓ ALLOWED examples:
                           "Narara, Gulf of Kutch"
                           "Arambhada coast"
                           "Lakshadweep Islands"
                           "St. 1"  (station ID — resolve from Methods if possible)
                           "Gulf of Kutch"  (when no finer site is named)

                         If the sentence contains a habitat phrase AND a place
                         name, extract ONLY the place name:
                           "intertidal area of dead coral reef at Narara"
                               → verbatimLocality = "Narara"
                               → habitat          = "Intertidal dead coral reef"

                         If NO geographic place name can be found anywhere in
                         the document section for this record, output exactly:
                           "Not Reported"
                         Do NOT use the broadest study area as a fallback unless
                         the paper explicitly states the species was collected
                         there.  "Not Reported" is correct and safe.

                         ONE location per record — never merge multiple sites.
'''


# ─────────────────────────────────────────────────────────────────────────────
#  PART 2 — Post-parse locality classifier
# ─────────────────────────────────────────────────────────────────────────────

# ── Class A: Morphology signals ──────────────────────────────────────────────

# Measurement pattern: digit + unit, e.g. "10 cm", "15mm", "8–9 in number"
_MEASUREMENT_RE = re.compile(
    r"\b\d+[\d\s\-–]*\s*"
    r"(cm|mm|m\b|km|in\b|ft|µm|um|mg|g\b|kg|l\b|ml|%|ppt|ppm|°[CF])",
    re.IGNORECASE,
)

# Anatomical / morphological keywords that never appear in place names
_MORPHOLOGY_TERMS: frozenset[str] = frozenset({
    # Cnidarian / jellyfish anatomy
    "umbrella", "subumbrella", "subumbrellar", "exumbrella", "bell",
    "oral arm", "oral arms", "manubrium", "tentacle", "tentacles",
    "nematocyst", "nematocysts", "rhopalia", "rhopalium",
    "gonad", "gonads", "radial canal", "ring canal",
    "strobila", "ephyra",  # developmental — also in lifestage list
    "zooxanthellae",        # symbiont — appears in description sections
    "filament", "filaments", "zooid", "zooids",
    "mesoglea", "ectoderm", "endoderm",
    # General morphology
    "diameter", "radius", "branched", "unbranched",
    "dorsal", "ventral", "lateral", "proximal", "distal",
    "anterior", "posterior",
    "in number", "in length", "in width", "in height",
    "flattened", "convex", "concave", "circular", "ovate",
    "pigment", "pigmented", "coloured", "coloration",
    "symmetry", "radial symmetry", "bilateral",
})

# ── Class B: Habitat-only signals ────────────────────────────────────────────

# Habitat lead words — when the string STARTS with one of these AND contains
# no geographic proper noun, it is habitat-only.
_HABITAT_LEAD_RE = re.compile(
    r"^(intertidal|subtidal|supratidal|littoral|infralittoral|eulittoral"
    r"|open\s+sea|open\s+ocean|pelagic|benthic|demersal"
    r"|rocky\s|sandy\s|muddy\s|silty\s|soft\s+bottom|hard\s+bottom"
    r"|coral\s+reef|dead\s+coral|seagrass|mangrove\s+edge|estuarine\s"
    r"|reef\s+flat|reef\s+slope|lagoon\s+floor"
    r"|shallow\s+water|deep\s+water|offshore\s+water"
    r"|found\s+(in|on|at|near|among|between|beneath|under)"
    r"|collected\s+(in|from|on|at|near)"
    r"|associated\s+with)",
    re.IGNORECASE,
)

# Pure habitat body-words — ALL tokens in the string are from this list →
# definitely habitat, even if it doesn't match _HABITAT_LEAD_RE.
_HABITAT_BODY_WORDS: frozenset[str] = frozenset({
    "intertidal", "subtidal", "reef", "coral", "dead", "live",
    "rocky", "sandy", "muddy", "soft", "hard",
    "seagrass", "mangrove", "lagoon", "estuary", "estuarine",
    "shore", "shoreface", "shoreline",
    "area", "zone", "habitat", "environment", "substrate",
    "water", "waters", "sea", "ocean", "bay", "gulf", "coast",
    "bottom", "bed", "flat", "slope", "edge",
    "shallow", "deep", "surface", "column",
    "offshore", "nearshore", "inshore", "littoral",
    "pelagic", "benthic", "demersal",
    "of", "the", "a", "an", "and", "or", "with", "in", "on",
    "at", "near", "among", "between", "beneath", "under", "from",
    "not", "reported", "unknown",
})

# Already-correct sentinel values — skip filtering
_SKIP_VALUES: frozenset[str] = frozenset({
    "not reported", "unknown", "", "n/a", "na",
})


# ─────────────────────────────────────────────────────────────────────────────
#  Classifier
# ─────────────────────────────────────────────────────────────────────────────

def _has_geographic_proper_noun(text: str) -> bool:
    """
    Heuristic: does the string contain a word that looks like a proper noun
    (Title-Cased, not a sentence opener, not a known stop-word)?

    We strip the first word (which may be capitalised as sentence-start) and
    look for any remaining capitalised token not in a stop-list.
    """
    stop = {
        "the", "a", "an", "in", "on", "at", "of", "and", "or",
        "near", "off", "from", "with", "between", "not", "reported",
        "area", "zone", "reef", "coast", "sea", "ocean", "bay", "gulf",
        "island", "islands", "waters", "shore",
    }
    tokens = text.split()
    # Skip first token (capitalised due to sentence position)
    for tok in tokens[1:]:
        clean = re.sub(r"[^a-zA-Z]", "", tok)
        if clean and clean[0].isupper() and clean.lower() not in stop:
            return True
    return False


def _classify_locality(loc: str) -> Optional[str]:
    """
    Classify a verbatimLocality string.

    Returns:
      "morphology"  — Class A: morphological/anatomical description
      "habitat"     — Class B: habitat descriptor, no named place
      None          — Looks geographic; leave as-is
    """
    if not loc or loc.strip().lower() in _SKIP_VALUES:
        return None

    text = loc.strip()

    # ── Class A: measurement pattern → definitely morphology ──────────────────
    if _MEASUREMENT_RE.search(text):
        return "morphology"

    # ── Class A: anatomical keyword ───────────────────────────────────────────
    text_lower = text.lower()
    for term in _MORPHOLOGY_TERMS:
        if term in text_lower:
            return "morphology"

    # ── Class B: starts with a habitat lead word ──────────────────────────────
    if _HABITAT_LEAD_RE.match(text):
        if _has_geographic_proper_noun(text):
            return "habitat_with_place"   # extract place, move habitat text
        return "habitat"

    # ── Class B: every token is a habitat/stop word (no proper noun) ──────────
    tokens = [re.sub(r"[^a-zA-Z]", "", w).lower() for w in text.split()]
    tokens = [t for t in tokens if t]   # remove empty
    if tokens and all(t in _HABITAT_BODY_WORDS for t in tokens):
        return "habitat"

    return None


def _extract_place_from_habitat(loc: str) -> Optional[str]:
    """
    If a habitat+place string like "intertidal area of dead coral reef at Narara"
    contains a detectable place name after a preposition, return it.
    """
    m = re.search(
        r"\b(?:at|near|in|off|from)\s+([A-Z][a-zA-Z\s,]+?)(?:\s*[,;.]|$)",
        loc,
    )
    if m:
        candidate = m.group(1).strip().rstrip(",;.")
        if len(candidate) >= 3:
            return candidate
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Main filter
# ─────────────────────────────────────────────────────────────────────────────

def post_parse_locality_filter(
    occurrences: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Post-LLM filter applied after post_parse_lifestage_filter, before dedup.

    For each record:
      CLASS A (morphology):
        • Saves original verbatimLocality to `_morphology_text` (for audit).
        • Sets verbatimLocality = "Not Reported".
        • Record is KEPT (biologist may assign coordinates manually in HITL).

      CLASS B (habitat without place):
        • If locality contains an embedded place name (e.g. "at Narara"),
          extracts it → verbatimLocality = "Narara".
        • Otherwise moves text to `habitat` field (if habitat is blank) and
          sets verbatimLocality = "Not Reported".
        • Record is KEPT.

    Returns (processed_occurrences, quarantined_list).
    quarantined_list = records that had their locality rewritten (for logging).
    """
    processed:    list[dict] = []
    quarantined:  list[dict] = []

    for occ in occurrences:
        if not isinstance(occ, dict):
            continue

        raw_loc = str(
            occ.get("verbatimLocality") or occ.get("Verbatim Locality") or ""
        ).strip()

        classification = _classify_locality(raw_loc)

        if classification == "morphology":
            logger.info(
                "[loc_filter] CLASS A (morphology) → quarantined locality: '%s…'",
                raw_loc[:80],
            )
            occ["_morphology_text"] = raw_loc        # preserve for audit
            occ["verbatimLocality"] = "Not Reported"
            quarantined.append(occ)

        elif classification in ("habitat", "habitat_with_place"):
            # Try to rescue a place name embedded after a preposition
            rescued = _extract_place_from_habitat(raw_loc)
            if rescued:
                logger.info(
                    "[loc_filter] CLASS B (habitat+place) → extracted '%s' from '%s'",
                    rescued, raw_loc[:80],
                )
                occ["verbatimLocality"] = rescued
                # Move the habitat phrase to the habitat field if empty
                existing_habitat = str(occ.get("Habitat") or occ.get("habitat") or "").strip()
                if not existing_habitat or existing_habitat.lower() in ("not reported", ""):
                    occ["Habitat"] = raw_loc
            else:
                logger.info(
                    "[loc_filter] CLASS B (habitat-only) → locality quarantined: '%s'",
                    raw_loc[:80],
                )
                existing_habitat = str(occ.get("Habitat") or occ.get("habitat") or "").strip()
                if not existing_habitat or existing_habitat.lower() in ("not reported", ""):
                    occ["Habitat"] = raw_loc     # preserve as habitat
                occ["verbatimLocality"] = "Not Reported"
                quarantined.append(occ)

        processed.append(occ)

    logger.info(
        "[loc_filter] %d records processed, %d localities quarantined "
        "(morphology or habitat-only)",
        len(processed), len(quarantined),
    )
    return processed, quarantined


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TEST_RECORDS = [
        # CLASS A — morphology
        {
            "recordedName": "Cassiopea andromeda",
            "verbatimLocality": (
                "Umbrella circular, flattened, 10 cm to 15 cm in diameter, "
                "usually with subumbrellar surface facing upwards and exumbrella "
                "facing downwards; oral arms 8-9 in number, slightly longer than "
                "umbrella radius, branched, with about 4 side branches supporting "
                "many filaments with zooxanthellae"
            ),
        },
        # CLASS B — pure habitat
        {
            "recordedName": "Cassiopea andromeda",
            "verbatimLocality": "intertidal area of dead coral reef",
            "Habitat": "",
        },
        # CLASS B — habitat with embedded place → should rescue "Narara"
        {
            "recordedName": "Cassiopea andromeda",
            "verbatimLocality": "intertidal area of dead coral reef at Narara",
            "Habitat": "",
        },
        # GOOD — geographic, should pass unchanged
        {
            "recordedName": "Cassiopea andromeda",
            "verbatimLocality": "Arambhada, Gulf of Kutch",
        },
        # GOOD — geographic, should pass unchanged
        {
            "recordedName": "Cassiopea andromeda",
            "verbatimLocality": "Narara, Gulf of Kutch",
        },
    ]

    results, quarantined = post_parse_locality_filter(TEST_RECORDS)

    print(f"\n{'─'*70}")
    print(f"{'Record':<25} | {'Result verbatimLocality':<35} | Note")
    print(f"{'─'*70}")
    for r in results:
        note = ""
        if r.get("_morphology_text"):
            note = "[CLASS A → Not Reported]"
        elif r.get("verbatimLocality") == "Not Reported" and r.get("Habitat"):
            note = f"[CLASS B → habitat='{r['Habitat'][:30]}']"
        print(
            f"{r['recordedName'][:24]:<25} | "
            f"{r['verbatimLocality'][:34]:<35} | {note}"
        )
    print(f"{'─'*70}")
    print(f"\n{len(quarantined)} record(s) had locality quarantined.\n")
