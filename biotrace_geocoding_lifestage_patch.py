"""
biotrace_geocoding_lifestage_patch.py  —  BioTrace v5.4
────────────────────────────────────────────────────────────────────────────
TWO PATCHES IN ONE FILE.

PATCH A — Nominatim India geographic bias
────────────────────────────────────────────────────────────────────────────
ROOT CAUSE of "Narara → Narara, NSW, Australia":
  The geopy Nominatim.geocode() call in geocoding_cascade.py (and in
  biotrace_hitl_geocoding.py) passes the raw locality string with NO country
  restriction.  Nominatim returns the globally highest-ranked match for
  "Narara", which happens to be the suburb in New South Wales.

FIX:
  1. Pass `country_codes="in"` to Nominatim for all India-context searches.
  2. Append ", India" to the query string as a soft geographic hint when the
     locality string does not already name a country.
  3. In GeoNames SQLite, the existing `country_code='IN'` filter already works
     correctly — no change needed there.

How to integrate into geocoding_cascade.py
  Replace the Nominatim geocode call in NominatimEnrichedGeocoder.geocode_missing()
  (and any direct geocoder.geocode() calls) with _india_geocode() from this file,
  or apply the monkey-patch at the bottom of this file.


PATCH B — LLM prompt life-stage / abbreviation guard
────────────────────────────────────────────────────────────────────────────
ROOT CAUSE of "Scyphistoma" and "C. andromeda" in extracted records:
  The LLM extraction prompt lists rules like "ONE SPECIES PER RECORD" but
  never explicitly tells the model:
    (a) Life-stage terms (Scyphistoma, Medusa, Ephyra …) are NOT taxa — do not
        extract them as "Recorded Name".
    (b) Abbreviated genus references ("C. andromeda") MUST be expanded to the
        full binomial using earlier context in the same document.

  Without these instructions the LLM treats "Scyphistomae from Narara" as a
  species occurrence and "C. andromeda" as a valid recordedName.

FIX:
  A. PROMPT_LIFESTAGE_GUARD — a paragraph to splice into _SCHEMA_PROMPT just
     before the field definitions.  Paste it into biotrace_v5.py.
  B. post_parse_lifestage_filter() — a post-LLM filter that catches any
     life-stage names that slipped through (belt-and-suspenders with dedup patch).

Both fixes are required; the prompt fix reduces upstream noise, the filter
catches LLM non-compliance.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

logger = logging.getLogger("biotrace.geo_ls_patch")

# ─────────────────────────────────────────────────────────────────────────────
#  PATCH A — Nominatim India-bias geocoding
# ─────────────────────────────────────────────────────────────────────────────

# Country codes for Nominatim to search first/only.
# "in" = India.  Extend with "lk" (Sri Lanka), "bd" (Bangladesh) if needed.
_NOMINATIM_COUNTRY_CODES = "in"

# Admin-context terms that signal the locality is already country-qualified
_INDIA_SIGNALS = re.compile(
    r"\b(india|gujarat|kerala|maharashtra|karnataka|goa|tamilnadu|tamil\s*nadu"
    r"|andhra|telangana|odisha|rajasthan|andaman|lakshadweep|bengal)\b",
    re.IGNORECASE,
)


def _append_india_hint(locality: str) -> str:
    """Return locality + ', India' unless it already names an Indian state/country."""
    if not _INDIA_SIGNALS.search(locality):
        return locality.strip().rstrip(",") + ", India"
    return locality


def india_nominatim_geocode(geocoder, locality: str) -> Optional[dict]:
    """
    Wrapper around a geopy Nominatim geocoder that enforces India context.

    Parameters
    ----------
    geocoder : geopy.geocoders.Nominatim instance
    locality : raw verbatimLocality string

    Returns
    -------
    {"lat": float, "lon": float, "display_name": str} or None
    """
    qualified = _append_india_hint(locality)

    try:
        time.sleep(1.1)  # Nominatim 1 req/s
        result = geocoder.geocode(
            qualified,
            exactly_one=True,
            timeout=10,
            country_codes=_NOMINATIM_COUNTRY_CODES,   # ← the critical fix
        )
        if result:
            logger.debug(
                "[geo_patch] '%s' → '%s' (%.5f, %.5f)",
                qualified, result.address, result.latitude, result.longitude,
            )
            return {
                "lat":          result.latitude,
                "lon":          result.longitude,
                "display_name": result.address,
            }

        # If India-restricted search returns nothing, log and return None
        # (do NOT fall back to unrestricted global search — that's how
        #  "Narara, NSW, Australia" crept in).
        logger.info(
            "[geo_patch] Nominatim India-restricted returned no result for '%s'", qualified
        )
    except Exception as exc:
        logger.warning("[geo_patch] Nominatim error for '%s': %s", qualified, exc)

    return None


def patch_geocoding_cascade() -> None:
    """
    Monkey-patch geocoding_cascade.GeocodingCascade._nominatim lookup to use
    india_nominatim_geocode().

    Call this ONCE at startup in biotrace_v5.py after importing GeocodingCascade:

        from biotrace_geocoding_lifestage_patch import patch_geocoding_cascade
        patch_geocoding_cascade()
    """
    try:
        import geocoding_cascade as _gc
        _orig_init = _gc.GeocodingCascade.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            # Wrap the Nominatim geocoder if it was initialised
            if self._nominatim:
                _inner = self._nominatim
                class _WrappedNominatim:
                    def geocode_missing(self_, records):
                        for occ in records:
                            loc = str(occ.get("verbatimLocality", "")).strip()
                            if not loc:
                                continue
                            result = india_nominatim_geocode(
                                _inner._geocoder, loc   # access underlying geopy instance
                            )
                            if result:
                                occ["decimalLatitude"]  = result["lat"]
                                occ["decimalLongitude"] = result["lon"]
                                occ["geocodingSource"]  = "Nominatim_IN"
                        return records
                self._nominatim = _WrappedNominatim()

        _gc.GeocodingCascade.__init__ = _patched_init
        logger.info("[geo_patch] GeocodingCascade patched with India Nominatim bias.")
    except ImportError:
        logger.warning("[geo_patch] geocoding_cascade not importable — patch skipped.")


# ─────────────────────────────────────────────────────────────────────────────
#  PATCH B — LLM prompt snippet + post-parse filter
# ─────────────────────────────────────────────────────────────────────────────

# ── B1: Prompt guard snippet ──────────────────────────────────────────────────
# Splice this string into biotrace_v5.py's _SCHEMA_PROMPT, immediately before
# the "For EACH species x locality x event, return a JSON object" line.

PROMPT_LIFESTAGE_GUARD = """
CRITICAL EXCLUSION RULES (violations will break the dataset):
  • LIFE-STAGE TERMS ARE NOT SPECIES — Never put the following words in
    "Recorded Name": scyphistoma, scyphistomae, ephyra, ephyrae, medusa,
    medusae, polyp, polyps, planula, planulae, larva, larvae, juvenile,
    zooid, strobila, spat, nauplius, cypris, zoea, veliger, trochophore,
    bipinnaria, brachiolaria, doliolaria, alevin, fry, fingerling.
    If a sentence says "Scyphistomae from Narara, Gulf of Kutch", the
    ORGANISM is the species named in the nearest preceding sentence
    (e.g. Cassiopea andromeda), NOT "Scyphistoma". Extract THAT species
    at THAT locality.

  • ABBREVIATED GENUS MUST BE EXPANDED — Never output an abbreviation like
    "C. andromeda" as the Recorded Name. Scan back through the document
    for the first occurrence of a full name whose genus starts with that
    letter and expand it (e.g. "C. andromeda" → "Cassiopea andromeda").
    If the full genus cannot be resolved, skip the record entirely.

  • TAXONOMIC AUTHORITY IS NOT PART OF THE NAME — "(Forsskål, 1775)" and
    similar authority strings must NOT be included in "Recorded Name".
    Write only the binomial or trinomial: "Cassiopea andromeda".
"""

# ── B2: Post-parse filter ─────────────────────────────────────────────────────

from biotrace_dedup_patch import LIFE_STAGE_TERMS, _ABBREV_GENUS_RE, _is_non_taxon  # reuse

_AUTHORITY_RE = re.compile(r"\s*\([^)]*\d{4}[^)]*\)\s*$")  # "(Forsskål, 1775)"


def _clean_recorded_name(name: str) -> str:
    """Strip taxonomic authority suffix from recorded name."""
    return _AUTHORITY_RE.sub("", name).strip()


def post_parse_lifestage_filter(
    occurrences: list[dict],
    genus_context: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Post-LLM extraction filter applied BEFORE dedup.

    Actions per record:
      1. Strip authority from recordedName ("Cassiopea andromeda (Forsskål)" →
         "Cassiopea andromeda").
      2. If recordedName is a life-stage term → discard (log reason).
      3. If recordedName is an abbreviated genus ("C. andromeda") →
         attempt expansion using genus_context dict {"C": "Cassiopea"}.
         If expansion fails → discard and warn.

    Parameters
    ----------
    occurrences   : raw list of occurrence dicts from LLM parse
    genus_context : optional dict mapping single-letter genus abbrev to full
                    genus name, built by scan_genus_context() below.

    Returns
    -------
    (kept, discarded)  —  two lists
    """
    kept:      list[dict] = []
    discarded: list[dict] = []

    for occ in occurrences:
        if not isinstance(occ, dict):
            continue

        raw_name = str(
            occ.get("recordedName") or occ.get("Recorded Name") or
            occ.get("validName", "")
        ).strip()

        # Step 1: strip authority suffix
        cleaned = _clean_recorded_name(raw_name)
        if cleaned != raw_name:
            logger.debug("[ls_filter] Authority stripped: '%s' → '%s'", raw_name, cleaned)
            occ["recordedName"] = cleaned

        reason = _is_non_taxon(cleaned)

        # Step 2: life-stage → discard
        if reason == "life_stage":
            logger.info(
                "[ls_filter] Discarding life-stage record: '%s' @ '%s'",
                cleaned, occ.get("verbatimLocality", ""),
            )
            discarded.append(occ)
            continue

        # Step 3: abbreviated genus → try expansion
        if reason == "abbreviation":
            expanded = _try_expand_abbreviation(cleaned, genus_context or {})
            if expanded:
                logger.info(
                    "[ls_filter] Expanded abbreviation: '%s' → '%s'", cleaned, expanded
                )
                occ["recordedName"] = expanded
            else:
                logger.warning(
                    "[ls_filter] Cannot expand abbreviation '%s' — discarding record", cleaned
                )
                discarded.append(occ)
                continue

        kept.append(occ)

    logger.info(
        "[ls_filter] %d kept, %d discarded (life-stage or unresolvable abbreviation)",
        len(kept), len(discarded),
    )
    return kept, discarded


def _try_expand_abbreviation(name: str, genus_context: dict[str, str]) -> Optional[str]:
    """
    Attempt to expand an abbreviated name like "C. andromeda".
    genus_context: {"C": "Cassiopea", "H": "Holothuria", …}
    """
    m = re.match(r"^([A-Z])\.\s+(.+)$", name)
    if not m:
        return None
    letter  = m.group(1)
    epithet = m.group(2).strip()
    full_genus = genus_context.get(letter)
    if full_genus:
        return f"{full_genus} {epithet}"
    return None


def scan_genus_context(text: str) -> dict[str, str]:
    """
    Scan document text for full binomials and build a letter → genus map.
    Used to populate genus_context for post_parse_lifestage_filter().

    Example:
        text contains "Cassiopea andromeda" → {"C": "Cassiopea"}
        text contains "Holothuria scabra"   → {"H": "Holothuria"}
    """
    context: dict[str, str] = {}
    # Find any capitalised genus + lowercase epithet binomials
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\s+[a-z]{2,}", text):
        genus = m.group(1)
        letter = genus[0]
        # First occurrence wins (most likely the full name introduction)
        context.setdefault(letter, genus)
    return context


# ─────────────────────────────────────────────────────────────────────────────
#  HOW TO WIRE INTO biotrace_v5.py
# ─────────────────────────────────────────────────────────────────────────────
"""
# 1. Import at top of biotrace_v5.py
from biotrace_geocoding_lifestage_patch import (
    patch_geocoding_cascade,
    PROMPT_LIFESTAGE_GUARD,
    post_parse_lifestage_filter,
    scan_genus_context,
)
from biotrace_dedup_patch import dedup_occurrences   # replaces gnv version

# 2. Apply geocoding patch once (after GeocodingCascade import)
patch_geocoding_cascade()

# 3. Splice into _SCHEMA_PROMPT (once, at module load):
_SCHEMA_PROMPT = _SCHEMA_PROMPT.replace(
    'For EACH species x locality x event, return a JSON object',
    PROMPT_LIFESTAGE_GUARD +
    'For EACH species x locality x event, return a JSON object'
)

# 4. In _process_batch_text(), after LLM parse, before dedup:
genus_ctx = scan_genus_context(text)         # text = current chunk
data, discarded = post_parse_lifestage_filter(data, genus_ctx)
if discarded:
    log_cb(f"  [LS-filter] {len(discarded)} life-stage/abbreviated records discarded")
"""
