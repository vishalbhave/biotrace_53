"""
scripts/nominatim_geocoder.py
─────────────────────────────────────────────────────────────────────────────
BioTrace · Nominatim Geocoder with Locality Enrichment
─────────────────────────────────────────────────────────────────────────────

Two-phase geocoding strategy:
  Phase 1 · ENRICH  → build a standardised, district+state-qualified address
                       string from the raw LLM verbatimLocality using the
                       GeoNames IN SQLite DB and the Pincode BBox index.
  Phase 2 · QUERY   → send enriched string to Nominatim (rate-limited,
                       1 req/sec, India-constrained, address-detail mode).

Key design decisions
  • Only records with NULL decimalLatitude OR decimalLongitude are processed.
  • Unique locality strings are deduplicated before querying → a paper with
    50 species at "Chilika Lake" costs ONE API call, not 50.
  • Results are cached per (enriched_string) within a session.
  • geocodingSource is set to "Nominatim" so downstream validation can
    distinguish GeoNames, Pincode, LLM and Nominatim fills.
  • A standalone batch function targets the SQLite occurrences table
    directly — safe to run as a background job after extraction.

Install dependency:
    pip install geopy
"""

import re
import sqlite3
import logging
from typing import Optional

logger = logging.getLogger("biotrace.nominatim")


# ─── Indian state code → full name (GeoNames admin1_code) ────────────────────
ADMIN1_MAP: dict[str, str] = {
    "01": "Andhra Pradesh",  "36": "Telangana",      "02": "Arunachal Pradesh",
    "03": "Assam",           "04": "Bihar",           "34": "Chhattisgarh",
    "30": "Goa",             "05": "Gujarat",         "06": "Haryana",
    "07": "Himachal Pradesh","08": "Jammu and Kashmir","20": "Jharkhand",
    "09": "Karnataka",       "10": "Kerala",          "37": "Ladakh",
    "12": "Madhya Pradesh",  "13": "Maharashtra",     "14": "Manipur",
    "15": "Meghalaya",       "16": "Mizoram",         "17": "Nagaland",
    "18": "Odisha",          "11": "Punjab",          "19": "Rajasthan",
    "26": "Sikkim",          "21": "Tamil Nadu",      "22": "Tripura",
    "23": "Uttar Pradesh",   "24": "Uttarakhand",     "25": "West Bengal",
    "35": "Andaman and Nicobar Islands", "28": "Chandigarh",
    "29": "Delhi",           "32": "Daman and Diu",   "33": "Lakshadweep",
    "31": "Puducherry",
}

_ALL_STATES = set(ADMIN1_MAP.values())

# Noise words to strip before locality lookup
_NOISE = re.compile(
    r'\b(near|off|at|from|around|along|in|the|a|an|coast|of|and|'
    r'village|taluk|tehsil|block|mandal|district|dt|dist|area|'
    r'region|sector|ward|zone|locality|town|city|nagar|pur|wadi)\b',
    re.I,
)


def _resolve_occurrence_table(conn: sqlite3.Connection) -> str:
    """
    Prefer the current v4 schema while remaining compatible with older DBs.
    """
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    names = {row[0] for row in rows}
    if "occurrences_v4" in names:
        return "occurrences_v4"
    if "occurrences" in names:
        return "occurrences"
    raise sqlite3.OperationalError(
        "No occurrence table found (expected `occurrences_v4` or `occurrences`)."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main geocoder class
# ─────────────────────────────────────────────────────────────────────────────

class NominatimEnrichedGeocoder:
    """
    Enrich → Deduplicate → Query Nominatim for occurrences missing coordinates.

    Parameters
    ----------
    geonames_db_path : str
        Path to the BioTrace GeoNames SQLite DB (geonames_india.db).
    pincode_index : dict, optional
        Output of ``build_pincode_bbox_index()`` from coord_utils — used to
        resolve district/state for village names via pincode lookup.
    user_agent : str
        Nominatim ToS require a unique, identifiable user-agent string.
    min_delay : float
        Minimum seconds between Nominatim requests (must be ≥ 1.0 per ToS).
    """

    def __init__(
        self,
        geonames_db_path: str = "",
        pincode_index: dict = None,
        user_agent: str = "BioTrace_biodiversity_extractor_v2",
        min_delay: float = 1.1,
    ):
        self.geonames_db   = geonames_db_path
        self.pincode_index = pincode_index or {}
        self.min_delay     = max(min_delay, 1.0)      # never below ToS limit
        self._available    = False
        self._geocode      = None
        self._session_cache: dict[str, Optional[tuple]] = {}

        try:
            from geopy.geocoders import Nominatim
            from geopy.extra.rate_limiter import RateLimiter

            _geo = Nominatim(user_agent=user_agent, timeout=10)
            self._geocode = RateLimiter(
                _geo.geocode,
                min_delay_seconds=self.min_delay,
                error_wait_seconds=10.0,
                max_retries=2,
                swallow_exceptions=True,
            )
            self._available = True
            logger.info("[Nominatim] geopy loaded, rate-limit=%.1fs", self.min_delay)
        except ImportError:
            logger.warning(
                "[Nominatim] geopy not installed — run: pip install geopy\n"
                "Nominatim geocoding will be skipped."
            )

    # ── Internal: GeoNames lookup ────────────────────────────────────────────

    def _geonames_lookup(self, token: str) -> dict:
        """
        Return the best GeoNames row for *token* restricted to India.
        Prioritises populated places (feature_class='P') over admin areas ('A').
        Returns {} on miss or error.
        """
        if not token or not self.geonames_db:
            return {}
        try:
            conn = sqlite3.connect(self.geonames_db, check_same_thread=False)
            row = conn.execute(
                """
                SELECT name, admin1_code, admin2_code, latitude, longitude
                FROM   geonames
                WHERE  (name=? OR asciiname=? OR alternatenames LIKE ?)
                AND    country_code='IN'
                ORDER BY
                    CASE feature_class WHEN 'P' THEN 1 WHEN 'A' THEN 2 ELSE 3 END,
                    CAST(population AS INTEGER) DESC
                LIMIT 1
                """,
                (token, token, f"%{token}%"),
            ).fetchone()
            conn.close()
            if row:
                return {
                    "name":    row[0],
                    "state":   ADMIN1_MAP.get(str(row[1]).zfill(2), ""),
                    "admin2":  row[2] or "",
                    "lat":     row[3],
                    "lon":     row[4],
                }
        except Exception as exc:
            logger.debug("[GeoNames lookup] %s → %s", token, exc)
        return {}

    # ── Internal: Pincode lookup ─────────────────────────────────────────────

    def _pincode_lookup(self, token: str) -> dict:
        """
        Match *token* against the pincode index (officeName field).
        Tries exact match first, then substring.  Returns {} on miss.
        """
        if not token or not self.pincode_index:
            return {}
        tok_low = token.lower().strip()
        best = {}
        for _pin, info in self.pincode_index.items():
            if not isinstance(info, dict):
                continue
            area = info.get("officeName", "").lower()
            if area == tok_low:
                return {
                    "district": info.get("districtName", ""),
                    "state":    info.get("stateName", ""),
                }
            if not best and (tok_low in area or area in tok_low):
                best = {
                    "district": info.get("districtName", ""),
                    "state":    info.get("stateName", ""),
                }
        return best

    # ── Public: enrichment ───────────────────────────────────────────────────

    def enrich_locality(self, verbatim: str) -> str:
        """
        Build a standardised address string for Nominatim from raw verbatim text.

        Output format (filled as available):
            <primary token(s)>, <district>, <state>, India

        Examples
        --------
        "Aasood"                     → "Aasood, Sindhudurg, Maharashtra, India"
        "near Chilika Lake, Odisha"  → "Chilika Lake, Odisha, India"
        "station at 17°47'N, Malvan" → "Malvan, Sindhudurg, Maharashtra, India"
        """
        if not verbatim:
            return ""

        raw = str(verbatim).strip()

        # ── 1. Split into comma-tokens and clean up ───────────────────────
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        cleaned_tokens = []
        for t in tokens:
            # drop pure coordinate tokens  (e.g. "17°47'N")
            if re.search(r"\d+[°\u00b0]\s*\d+", t):
                continue
            c = _NOISE.sub(" ", t).strip()
            c = re.sub(r"\s{2,}", " ", c)
            if len(c) > 2:
                cleaned_tokens.append(c)

        primary = cleaned_tokens[0] if cleaned_tokens else raw

        # ── 2. Detect state already present in verbatim ──────────────────
        found_state = ""
        for st in _ALL_STATES:
            if re.search(r"\b" + re.escape(st) + r"\b", raw, re.I):
                found_state = st
                break

        # ── 3. Look up primary token in both DBs ─────────────────────────
        geo  = self._geonames_lookup(primary)
        pin  = self._pincode_lookup(primary)

        # ── 4. Resolve district + state ──────────────────────────────────
        district = pin.get("district") or geo.get("admin2") or ""
        state    = found_state or pin.get("state") or geo.get("state") or ""

        # ── 5. Assemble enriched string ───────────────────────────────────
        parts: list[str] = []
        for t in cleaned_tokens:
            parts.append(t)

        if district and not any(district.lower() in p.lower() for p in parts):
            parts.append(district)

        if state and not any(state.lower() in p.lower() for p in parts):
            parts.append(state)

        if "india" not in " ".join(parts).lower():
            parts.append("India")

        enriched = ", ".join(parts)
        logger.debug("[enrich] '%s'  →  '%s'", verbatim[:60], enriched)
        return enriched

    # ── Internal: single Nominatim call ─────────────────────────────────────

    def _query(self, address: str) -> Optional[tuple[float, float, str]]:
        """
        Query Nominatim.  Returns (lat, lon, display_address) or None.
        Results are session-cached to avoid duplicate API calls.
        """
        if not self._available or not address:
            return None
        if address in self._session_cache:
            return self._session_cache[address]

        result = None
        try:
            loc = self._geocode(
                address,
                country_codes="IN",
                language="en",
                addressdetails=True,
                exactly_one=True,
            )
            if loc:
                result = (loc.latitude, loc.longitude, loc.address)
        except Exception as exc:
            logger.warning("[Nominatim] '%s' failed: %s", address[:80], exc)

        self._session_cache[address] = result
        return result

    # ── Public: geocode a list of occurrence dicts ───────────────────────────

    def geocode_missing(self, occurrences: list) -> list:
        """
        Fill missing lat/lon in-place for occurrence dicts that have a
        verbatimLocality but no coordinates.

        Already-geocoded records are untouched.
        Adds ``enrichedLocality`` field to show the query string used.

        Parameters
        ----------
        occurrences : list[dict]
            Standard BioTrace occurrence dicts.

        Returns
        -------
        list[dict]
            Same list, coordinates filled where Nominatim found a match.
        """
        if not self._available:
            return occurrences

        # Collect records that actually need geocoding
        missing = [
            occ for occ in occurrences
            if isinstance(occ, dict)
            and (occ.get("decimalLatitude") is None or occ.get("decimalLongitude") is None)
            and occ.get("verbatimLocality")
        ]

        if not missing:
            return occurrences

        logger.info("[Nominatim] %d records with missing coordinates", len(missing))

        # Build enriched strings (deduplicated)
        enriched_map: dict[str, str] = {}
        for occ in missing:
            vl = str(occ["verbatimLocality"]).strip()
            if vl not in enriched_map:
                enriched_map[vl] = self.enrich_locality(vl)

        # Geocode unique enriched strings
        for enriched in set(enriched_map.values()):
            if enriched in self._session_cache:
                continue
            result = self._query(enriched)
            # Fallback: strip to bare "locality, India" if enriched failed
            if result is None:
                raw_keys = [k for k, v in enriched_map.items() if v == enriched]
                if raw_keys:
                    fallback = raw_keys[0] + ", India"
                    result = self._query(fallback)
            self._session_cache[enriched] = result

        # Apply results back
        filled = 0
        for occ in occurrences:
            if not isinstance(occ, dict):
                continue
            if (occ.get("decimalLatitude") is not None
                    and occ.get("decimalLongitude") is not None):
                continue   # already has coordinates

            vl = str(occ.get("verbatimLocality", "")).strip()
            if not vl:
                continue

            enriched = enriched_map.get(vl, "")
            result   = self._session_cache.get(enriched)

            if result:
                lat, lon, display = result
                occ["decimalLatitude"]   = round(lat, 6)
                occ["decimalLongitude"]  = round(lon, 6)
                occ["geocodingSource"]   = "Nominatim"
                occ["enrichedLocality"]  = enriched
                filled += 1
                logger.debug(
                    "[Nominatim] '%s' → (%.4f, %.4f)", vl[:50], lat, lon
                )

        logger.info(
            "[Nominatim] filled %d / %d missing coordinates", filled, len(missing)
        )
        return occurrences


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone batch function — for the "Geocode Missing" button in the UI
# ─────────────────────────────────────────────────────────────────────────────

def batch_geocode_db_missing(
    meta_db_path: str,
    geonames_db_path: str,
    pincode_index: dict = None,
    state_filter: str = "",
    user_agent: str = "BioTrace_biodiversity_extractor_v2",
    progress_callback=None,
) -> int:
    """
    Scan the SQLite ``occurrences`` table for rows where decimalLatitude or
    decimalLongitude is NULL, geocode them with Nominatim, and update in-place.

    Designed to be called as a one-click background operation from the
    Global Dataset tab — NOT called in the real-time extraction loop.

    Parameters
    ----------
    meta_db_path : str
        Path to BioTrace metadata.db.
    geonames_db_path : str
        Path to geonames_india.db.
    pincode_index : dict, optional
        From ``build_pincode_bbox_index()``.
    state_filter : str, optional
        Restrict to rows whose verbatimLocality contains this string
        (e.g. "Maharashtra") — speeds up large databases.
    progress_callback : callable, optional
        Called as ``progress_callback(done, total)`` for Streamlit progress bar.

    Returns
    -------
    int
        Number of rows updated.
    """
    geocoder = NominatimEnrichedGeocoder(
        geonames_db_path=geonames_db_path,
        pincode_index=pincode_index or {},
        user_agent=user_agent,
    )
    if not geocoder._available:
        logger.warning("[batch_geocode] geopy unavailable — aborting")
        return 0

    conn = sqlite3.connect(meta_db_path, check_same_thread=False)
    table = _resolve_occurrence_table(conn)

    q = """
        SELECT id, verbatimLocality
        FROM   {table}
        WHERE  (decimalLatitude IS NULL OR decimalLongitude IS NULL)
        AND    verbatimLocality IS NOT NULL
        AND    verbatimLocality != ''
        AND    validationStatus != 'rejected'
    """.format(table=table)
    if state_filter:
        q += f" AND verbatimLocality LIKE '%{state_filter}%'"

    rows = conn.execute(q).fetchall()
    total = len(rows)
    logger.info("[batch_geocode] %d rows to geocode", total)

    if total == 0:
        conn.close()
        return 0

    # ── Build enriched strings and deduplicate ────────────────────────────
    enriched_map: dict[str, str] = {}
    for _, vl in rows:
        vl = str(vl).strip()
        if vl not in enriched_map:
            enriched_map[vl] = geocoder.enrich_locality(vl)

    # Geocode unique enriched strings (with progress reporting)
    unique_queries = list(set(enriched_map.values()))
    for i, enriched in enumerate(unique_queries):
        if enriched not in geocoder._session_cache:
            result = geocoder._query(enriched)
            if result is None:
                # Fallback: find a raw vl for this enriched and try bare form
                for raw_vl, enr in enriched_map.items():
                    if enr == enriched:
                        geocoder._session_cache[enriched] = geocoder._query(
                            raw_vl + ", India"
                        )
                        break
        if progress_callback:
            progress_callback(i + 1, len(unique_queries))

    # ── Update rows ───────────────────────────────────────────────────────
    updated = 0
    for row_id, vl in rows:
        vl = str(vl).strip()
        enriched = enriched_map.get(vl, "")
        result   = geocoder._session_cache.get(enriched)
        if result:
            lat, lon, _ = result
            conn.execute(
                """
                UPDATE {table}
                SET    decimalLatitude=?, decimalLongitude=?, geocodingSource=?
                WHERE  id=?
                """.format(table=table),
                (round(lat, 6), round(lon, 6), "Nominatim", row_id),
            )
            updated += 1
        if updated % 100 == 0 and updated > 0:
            conn.commit()

    conn.commit()
    conn.close()
    logger.info("[batch_geocode] complete — %d/%d rows updated", updated, total)
    return updated
