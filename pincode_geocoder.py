"""
pincode_geocoder.py — Indian Pincode Geocoder for BioTrace
===========================================================
Resolves Indian locality names from extracted PDF text to GPS coordinates
using a local pgeocode-style tab-separated file (IN_pin.txt).

Key capability: phonetic + fuzzy matching for Indian village names that
appear under many transliteration variants in English-language documents:
  Aasood / Asud / Aasud / Asood / Azood → same Ratnagiri village

Matching pipeline (each stage only runs if the previous stage fails):
  Stage 1 · Exact match         (normalised lowercase)
  Stage 2 · Canonical form      (Indian-specific vowel/consonant rules)
  Stage 3 · Token-sorted fuzzy  (rapidfuzz, threshold configurable)
  Stage 4 · Metaphone phonetic  (double-metaphone via doublemetaphone)
  Stage 5 · Postal-code lookup  (if a 6-digit code appears in the query)

Usage
-----
    from pincode_geocoder import IndianPincodeGeocoder

    geo = IndianPincodeGeocoder("biodiversity_data/IN_pin.txt")
    result = geo.geocode("Aasood")
    # → GeoResult(place_name='Aasud', postal_code='415712',
    #             state='Maharashtra', lat=17.7869, lon=73.1538,
    #             match_type='fuzzy', score=91.2)

Drop-in replacement for geocode_india_locality() in biotrace_app2.py:
    lat, lon = geo.geocode_coords("Aasood")   # → (17.7869, 73.1538) or (None, None)

Integration (biotrace_app2.py sidebar):
    geo = IndianPincodeGeocoder(PINCODE_FILE_PATH)
    # replace refine_occurrences_with_geonames() call with:
    file_occurrences = refine_occurrences_with_pincode(file_occurrences, geo)
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process as rf_process

logger = logging.getLogger("biotrace.geocoder")

# ──────────────────────────────────────────────────────────────────────────────
#  DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GeoResult:
    place_name:   str
    postal_code:  str
    state:        str
    county:       str
    latitude:     Optional[float]
    longitude:    Optional[float]
    match_type:   str           # exact | canonical | fuzzy | metaphone | pincode
    score:        float         # 0–100; 100 = perfect


@dataclass
class _Row:
    postal_code:  str
    place_name:   str
    state_name:   str
    county_name:  str
    latitude:     Optional[float]
    longitude:    Optional[float]
    # pre-computed search keys (added at index time)
    norm_key:     str = field(default="")   # normalised name
    canon_key:    str = field(default="")   # canonical transliteration


# ──────────────────────────────────────────────────────────────────────────────
#  INDIAN TRANSLITERATION NORMALISER
# ──────────────────────────────────────────────────────────────────────────────

# Vowel length variants (many Indian languages write long vowels as doubled)
_VOWEL_LONG = [
    (r"aa", "a"),   # Aasood → Asood
    (r"ee", "i"),   # Veer → Vir
    (r"oo", "u"),   # Aasood → Asud
    (r"ii", "i"),
    (r"uu", "u"),
]

# Aspirated stop variants (readers often drop the h)
_ASPIRATES = [
    (r"kh", "k"),
    (r"gh", "g"),
    (r"ch", "c"),
    (r"jh", "j"),
    (r"th", "t"),   # Thane → Tane (but keep 'th' in some contexts)
    (r"dh", "d"),
    (r"ph", "p"),   # Phadke → Padke (but ph→f in some styles)
    (r"bh", "b"),
]

# Consonant interchangeability common in Devanagari → Latin
_INTERCHANGEABLE = [
    (r"v",  "b"),   # Marathi v/b: Vasai/Basai
    (r"w",  "v"),   # Wada → Vada
    (r"sh", "s"),   # Shrivardhan → Srivardhan
    (r"ss", "s"),
    (r"tt", "t"),
    (r"nn", "n"),
    (r"ll", "l"),
    (r"rr", "r"),
    (r"mm", "m"),
]

# Suffix abbreviations (taluka / village suffixes)
_SUFFIX_MAP = {
    " tal": "", " tahsil": "", " taluka": "",
    " dist": "", " district": "",
    " vill": "", " village": "",
    " post": "", " p.o": "", " po ": " ",
}


def normalise_indian(name: str) -> str:
    """
    Normalise an Indian place name for fuzzy comparison.
    Strips diacritics → lowercase → removes common suffixes →
    collapses whitespace → strips punctuation.

    Example: 'Aasood (Dapoli Tahsil)' → 'asud'
    """
    if not name:
        return ""
    # Unicode decompose + strip combining chars (handles ā, ī, ū etc.)
    text = unicodedata.normalize("NFD", name)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower().strip()

    # Remove parenthetical qualifiers
    text = re.sub(r"\(.*?\)", "", text)

    # Strip administrative suffixes
    for sfx, repl in _SUFFIX_MAP.items():
        text = text.replace(sfx, repl)

    # Collapse spaces before applying character rules
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_form(name: str) -> str:
    """
    Apply Indian-specific transliteration rules to produce a canonical key
    that collapses the most common variant spellings into one form.

    Aasood → Asud:   aa→a, oo→u
    Aasud  → Asud:   aa→a
    Vada   → Bada:   v→b
    Shrivardhan → Sribardhan: sh→s, v→b

    All rules are applied in a fixed order to avoid cascades.
    """
    text = normalise_indian(name)

    # 1. Long vowels first (order matters: 'aa' before plain 'a')
    for pattern, repl in _VOWEL_LONG:
        text = re.sub(pattern, repl, text)

    # 2. Aspirated stops (apply before stripping plain 'h')
    for pattern, repl in _ASPIRATES:
        text = re.sub(pattern, repl, text)

    # 3. Consonant interchangeability
    for pattern, repl in _INTERCHANGEABLE:
        text = re.sub(pattern, repl, text)

    # 4. Remove residual isolated 'h' after vowels (e.g. "ah" → "a")
    text = re.sub(r"(?<=[aeiou])h", "", text)

    # 5. Collapse repeated characters (after substitutions)
    text = re.sub(r"(.)\1+", r"\1", text)

    # 6. Remove all spaces (village names sometimes written as one word)
    text = text.replace(" ", "")

    return text


# ──────────────────────────────────────────────────────────────────────────────
#  OPTIONAL DOUBLE METAPHONE
# ──────────────────────────────────────────────────────────────────────────────

def _metaphone(text: str) -> str:
    """Double metaphone if available, else return canonical_form as fallback."""
    try:
        from doublemetaphone import doublemetaphone
        p1, p2 = doublemetaphone(text)
        return p1 or p2 or canonical_form(text)
    except ImportError:
        return canonical_form(text)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN GEOCODER CLASS
# ──────────────────────────────────────────────────────────────────────────────

class IndianPincodeGeocoder:
    """
    Geocoder for Indian localities using a pgeocode-style IN_pin.txt file.

    Parameters
    ----------
    txt_path : str
        Path to the tab-separated IN_pin.txt file with columns:
        postal_code, country_code, place_name, state_name, state_code,
        county_name, county_code, community_name, community_code,
        latitude, longitude, accuracy
    db_path : str, optional
        SQLite cache path. Defaults to same directory as txt_path.
    fuzzy_threshold : float
        Minimum rapidfuzz score (0–100) to accept a fuzzy match.
        85 = strict (fewer false positives), 70 = lenient (catches more
        variants but may introduce noise for short place names < 5 chars).
    state_filter : str | None
        If set (e.g. "Maharashtra"), restrict fuzzy search to this state.
        Dramatically reduces false positives for Ratnagiri district surveys.
    """

    def __init__(
        self,
        txt_path: str,
        db_path: Optional[str] = None,
        fuzzy_threshold: float = 80.0,
        state_filter: Optional[str] = None,
    ):
        self.txt_path        = txt_path
        self.db_path         = db_path or txt_path.replace(".txt", "_geocache.db")
        self.fuzzy_threshold = fuzzy_threshold
        self.state_filter    = state_filter.lower().strip() if state_filter else None
        self._rows: list[_Row] = []
        self._norm_index:  dict[str, _Row] = {}   # normalised  → row
        self._canon_index: dict[str, _Row] = {}   # canonical   → row
        self._meta_index:  dict[str, _Row] = {}   # metaphone   → row
        self._pincode_index: dict[str, _Row] = {} # postal_code → row
        self._fuzzy_choices: list[str] = []        # canonical keys for rapidfuzz
        self._result_cache: dict[str, Optional[GeoResult]] = {}

        self._load_or_build()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _load_or_build(self) -> None:
        """Load from SQLite cache if fresh, else parse txt and rebuild."""
        if (os.path.exists(self.db_path) and
                os.path.getmtime(self.db_path) > os.path.getmtime(self.txt_path)):
            logger.info("[pincode] Loading geocache from SQLite…")
            self._load_from_db()
        else:
            logger.info("[pincode] Building geocache from %s…", self.txt_path)
            self._parse_txt_and_build()
        self._build_runtime_indices()

    def _parse_txt_and_build(self) -> None:
        """Parse IN_pin.txt, compute keys, persist to SQLite."""
        df = pd.read_csv(
            self.txt_path,
            sep="\t",
            encoding='utf-8-sig',
            header=0,
            dtype=str,
            keep_default_na=False,
        )
        # Normalise column names: strip spaces, lowercase
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        required = {"postal_code", "place_name", "state_name", "latitude", "longitude"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"IN_pin.txt is missing columns: {missing}\n"
                f"Found: {list(df.columns)}"
            )

        rows = []
        for _, r in df.iterrows():
            try:
                lat = float(r["latitude"])  if r.get("latitude")  else None
                lon = float(r["longitude"]) if r.get("longitude") else None
            except ValueError:
                lat = lon = None

            row = _Row(
                postal_code = str(r.get("postal_code", "")).strip(),
                place_name  = str(r.get("place_name",  "")).strip(),
                state_name  = str(r.get("state_name",  "")).strip(),
                county_name = str(r.get("county_name", "")).strip(),
                latitude    = lat,
                longitude   = lon,
                norm_key    = normalise_indian(r.get("place_name", "")),
                canon_key   = canonical_form(r.get("place_name",  "")),
            )
            rows.append(row)

        self._rows = rows
        self._persist_to_db()

    def _persist_to_db(self) -> None:
        con = sqlite3.connect(self.db_path)
        con.execute("DROP TABLE IF EXISTS pincode_geocache")
        con.execute("""
            CREATE TABLE pincode_geocache (
                postal_code TEXT,
                place_name  TEXT,
                state_name  TEXT,
                county_name TEXT,
                latitude    REAL,
                longitude   REAL,
                norm_key    TEXT,
                canon_key   TEXT
            )
        """)
        con.executemany(
            "INSERT INTO pincode_geocache VALUES (?,?,?,?,?,?,?,?)",
            [
                (r.postal_code, r.place_name, r.state_name, r.county_name,
                 r.latitude, r.longitude, r.norm_key, r.canon_key)
                for r in self._rows
            ]
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_norm  ON pincode_geocache(norm_key)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_canon ON pincode_geocache(canon_key)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_pin   ON pincode_geocache(postal_code)")
        con.commit()
        con.close()
        logger.info("[pincode] Persisted %d rows to %s", len(self._rows), self.db_path)

    def _load_from_db(self) -> None:
        con = sqlite3.connect(self.db_path)
        cur = con.execute("SELECT * FROM pincode_geocache")
        self._rows = [
            _Row(
                postal_code=r[0], place_name=r[1], state_name=r[2],
                county_name=r[3], latitude=r[4], longitude=r[5],
                norm_key=r[6], canon_key=r[7]
            )
            for r in cur.fetchall()
        ]
        con.close()

    def _build_runtime_indices(self) -> None:
        """Build in-memory dicts for O(1) exact/canonical lookups and fuzzy list."""
        for row in self._rows:
            # State filter (if set, only index that state)
            if self.state_filter and self.state_filter not in row.state_name.lower():
                continue
            self._norm_index.setdefault(row.norm_key,  row)
            self._canon_index.setdefault(row.canon_key, row)
            self._pincode_index[row.postal_code] = row
            meta = _metaphone(row.place_name)
            self._meta_index.setdefault(meta, row)

        self._fuzzy_choices = list(self._canon_index.keys())
        logger.info(
            "[pincode] Index ready: %d places, state_filter=%s",
            len(self._canon_index), self.state_filter or "all"
        )

    # ── MATCHING PIPELINE ────────────────────────────────────────────────────

    def geocode(self, query: str) -> Optional[GeoResult]:
        """
        Full pipeline — returns GeoResult or None.

        Stage 1: Exact normalised match
        Stage 2: Canonical transliteration match
        Stage 3: rapidfuzz token-sorted ratio on canonical keys
        Stage 4: Double-Metaphone phonetic match
        Stage 5: 6-digit postal code detected in query string
        """
        if not query or not isinstance(query, str):
            return None

        q = query.strip()
        if q in self._result_cache:
            return self._result_cache[q]

        result = (
            self._stage1_exact(q)
            or self._stage2_canonical(q)
            or self._stage3_fuzzy(q)
            or self._stage4_metaphone(q)
            or self._stage5_pincode(q)
        )
        self._result_cache[q] = result
        return result

    def geocode_coords(self, query: str) -> tuple[Optional[float], Optional[float]]:
        """Convenience wrapper — returns (lat, lon) or (None, None)."""
        r = self.geocode(query)
        return (r.latitude, r.longitude) if r else (None, None)

    def geocode_source(self, query: str) -> str:
        """Returns the source/match_type tag for the geocodingSource column."""
        r = self.geocode(query)
        if not r:
            return ""
        return f"IN_Pincode_{r.match_type}_{r.score:.0f}"

    # ── Stages ───────────────────────────────────────────────────────────────

    def _stage1_exact(self, query: str) -> Optional[GeoResult]:
        key = normalise_indian(query)
        row = self._norm_index.get(key)
        if row:
            logger.debug("[S1-exact] '%s' → '%s'", query, row.place_name)
            return self._to_result(row, "exact", 100.0)
        return None

    def _stage2_canonical(self, query: str) -> Optional[GeoResult]:
        key = canonical_form(query)
        row = self._canon_index.get(key)
        if row:
            logger.debug("[S2-canon] '%s' → '%s'", query, row.place_name)
            return self._to_result(row, "canonical", 95.0)
        return None

    def _stage3_fuzzy(self, query: str) -> Optional[GeoResult]:
        """
        rapidfuzz token_sort_ratio on canonical keys.

        token_sort_ratio is preferred over simple ratio because it handles
        multi-word queries like "Aasood village Dapoli" gracefully — it
        sorts the tokens before comparing, so word order doesn't matter.

        For very short names (≤ 4 chars) we raise the threshold to 92 to
        avoid spurious matches (e.g. "Tal" matching "Tal" from a suffix).
        """
        if not self._fuzzy_choices:
            return None

        q_canon = canonical_form(query)
        if not q_canon:
            return None

        threshold = max(self.fuzzy_threshold,
                        92.0 if len(q_canon) <= 4 else self.fuzzy_threshold)

        match = rf_process.extractOne(
            q_canon,
            self._fuzzy_choices,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if match:
            matched_key, score, _ = match
            row = self._canon_index[matched_key]
            logger.debug("[S3-fuzzy] '%s' → '%s' (score=%.1f)", query, row.place_name, score)
            return self._to_result(row, "fuzzy", float(score))
        return None

    def _stage4_metaphone(self, query: str) -> Optional[GeoResult]:
        """
        Double-Metaphone phonetic matching.
        Catches cases where even the canonical form diverges but pronunciation
        is the same: Wada / Vada / Wada → WAT / FAT → both map to same code.
        Requires:  pip install doublemetaphone
        Falls back to a secondary canonical-based metaphone approximation.
        """
        q_meta = _metaphone(canonical_form(query))
        row = self._meta_index.get(q_meta)
        if row:
            logger.debug("[S4-meta] '%s' → '%s' (meta=%s)", query, row.place_name, q_meta)
            return self._to_result(row, "metaphone", 75.0)
        return None

    def _stage5_pincode(self, query: str) -> Optional[GeoResult]:
        """
        If the query or its surrounding context contains a 6-digit postal code,
        use it directly. Useful when LLM extracts e.g. 'Ratnagiri 415612'.
        """
        pins = re.findall(r"\b([1-9]\d{5})\b", query)
        for pin in pins:
            row = self._pincode_index.get(pin)
            if row:
                logger.debug("[S5-pin] '%s' → '%s' via %s", query, row.place_name, pin)
                return self._to_result(row, "pincode", 90.0)
        return None

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_result(row: _Row, match_type: str, score: float) -> GeoResult:
        return GeoResult(
            place_name  = row.place_name,
            postal_code = row.postal_code,
            state       = row.state_name,
            county      = row.county_name,
            latitude    = row.latitude,
            longitude   = row.longitude,
            match_type  = match_type,
            score       = score,
        )

    def explain(self, query: str) -> str:
        """
        Human-readable explanation of how a query was resolved.
        Useful for debugging in the BioTrace Review tab.

        Example:
            geo.explain("Aasood")
            →
            Query       : "Aasood"
            Normalised  : "aasood"
            Canonical   : "asud"
            Stage       : fuzzy (score=91.2)
            Matched     : "Aasud" (415712) — Maharashtra / Ratnagiri
            Coords      : 17.7869, 73.1538
        """
        result = self.geocode(query)
        norm   = normalise_indian(query)
        canon  = canonical_form(query)
        meta   = _metaphone(canon)
        if result:
            return (
                f'Query       : "{query}"\n'
                f'Normalised  : "{norm}"\n'
                f'Canonical   : "{canon}"\n'
                f'Metaphone   : "{meta}"\n'
                f'Stage       : {result.match_type} (score={result.score:.1f})\n'
                f'Matched     : "{result.place_name}" ({result.postal_code})'
                f' — {result.state} / {result.county}\n'
                f'Coords      : {result.latitude}, {result.longitude}'
            )
        return (
            f'Query       : "{query}"\n'
            f'Normalised  : "{norm}"\n'
            f'Canonical   : "{canon}"\n'
            f'Metaphone   : "{meta}"\n'
            f'Result      : NO MATCH (threshold={self.fuzzy_threshold})'
        )

    def batch_geocode(
        self,
        queries: list[str],
        min_score: float = 0.0,
    ) -> pd.DataFrame:
        """
        Geocode a list of locality strings.
        Returns a DataFrame with columns:
          query, place_name, postal_code, state, county,
          latitude, longitude, match_type, score
        """
        records = []
        for q in queries:
            r = self.geocode(q)
            if r and r.score >= min_score:
                records.append({
                    "query":       q,
                    "place_name":  r.place_name,
                    "postal_code": r.postal_code,
                    "state":       r.state,
                    "county":      r.county,
                    "latitude":    r.latitude,
                    "longitude":   r.longitude,
                    "match_type":  r.match_type,
                    "score":       round(r.score, 1),
                })
            else:
                records.append({
                    "query":       q,
                    "place_name":  None, "postal_code": None,
                    "state":       None, "county":      None,
                    "latitude":    None, "longitude":   None,
                    "match_type":  "no_match", "score": 0.0,
                })
        return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
#  BIOTRACE INTEGRATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def refine_occurrences_with_pincode(
    occurrences: list[dict],
    geocoder: IndianPincodeGeocoder,
    overwrite_existing: bool = False,
) -> list[dict]:
    """
    Drop-in replacement for refine_occurrences_with_geonames().

    For each occurrence record that lacks coordinates (or has only LLM-guessed
    ones), attempts to resolve decimalLatitude / decimalLongitude via the
    pincode geocoder.

    Parameters
    ----------
    overwrite_existing : bool
        If True, replace existing LLM-inferred coordinates with pincode-derived
        ones where the pincode match score > 80. Useful when LLM hallucinated
        coordinates.
    """
    refined = []
    for occ in occurrences:
        if not isinstance(occ, dict):
            continue
        lat = occ.get("decimalLatitude")
        lon = occ.get("decimalLongitude")
        has_coords = (lat is not None and lon is not None
                      and str(lat).strip() not in ("", "null", "None")
                      and str(lon).strip() not in ("", "null", "None"))

        if has_coords and not overwrite_existing:
            refined.append(occ)
            continue

        locality = occ.get("verbatimLocality", "")
        if not locality:
            refined.append(occ)
            continue

        result = geocoder.geocode(str(locality))
        if result and result.latitude is not None:
            occ["decimalLatitude"]  = result.latitude
            occ["decimalLongitude"] = result.longitude
            occ["geocodingSource"]  = f"IN_Pincode_{result.match_type}_{result.score:.0f}"
            occ["_matchedPlace"]    = result.place_name
            occ["_postalCode"]      = result.postal_code
        refined.append(occ)
    return refined


# ──────────────────────────────────────────────────────────────────────────────
#  CLI DEMO
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pincode_geocoder.py <path/to/IN_pin.txt> [query ...]")
        print("\nExample test queries (Ratnagiri district variants):")
        test_names = [
            "Aasood", "Asud", "Aasud", "Asood", "Agar Vaigani",
            "Agar Vaygani", "Dapoli", "Guhagar", "Velas", "415712",
        ]
        sys.exit(0)

    txt_path = sys.argv[1]
    queries  = sys.argv[2:] if len(sys.argv) > 2 else [
        "Aasood", "Asud", "Aasud", "Agar Vaigani", "Agar Vaygani",
        "Dapoli", "Guhagar", "Shrivardhan", "Srivardhan",
    ]

    print(f"\nLoading geocoder from: {txt_path}")
    geo = IndianPincodeGeocoder(
        txt_path,
        fuzzy_threshold=78.0,
        state_filter="Maharashtra",   # restrict to one state for speed
    )
    print(f"Index size: {len(geo._canon_index):,} canonical keys\n")
    print("=" * 70)

    for q in queries:
        print(geo.explain(q))
        print("-" * 70)
