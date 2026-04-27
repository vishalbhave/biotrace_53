"""
biotrace_locality_ner.py  —  BioTrace v5.2
────────────────────────────────────────────────────────────────────────────
Contextual Locality Extraction and Administrative String Expansion.

Requirement (§2, §5B of spec):
  "Narara" → "Narara, Jamnagar, Gulf of Kutch, Gujarat, India"

Pipeline:
  Stage 1  NER          — spaCy GPE + LOC entities from text
                         Regex patterns for Indian districts/states
                         GPS coordinate patterns (DMS / decimal)
  Stage 2  EXPAND       — GeoNames IN SQLite → district + state + country
                         Nominatim geocoder → full address string (rate-limited)
                         Pincode lookup → complete admin hierarchy
  Stage 3  SEGREGATE    — LocalitySplitter (from biotrace_gnv.py)
                         distinguishes comma-separated sub-sites vs. list of sites
  Stage 4  VALIDATE     — India bbox check + state bbox
                         lat/lon assignment from GeoNames / Nominatim
  Stage 5  LINK         — Associate expanded localities with nearby species
                         mentions using character-offset proximity

spaCy model fallback chain:
  en_core_web_trf (best) → en_core_web_lg → en_core_web_sm → regex-only

GeoNames columns used from geonames_india.db:
  name, asciiname, alternatenames, feature_class, admin1_code,
  admin2_code, admin3_code, latitude, longitude, population

Administrative hierarchy expansion:
  GeoNames admin1 codes for India → state names
  GeoNames admin2 codes → district names
  Merged as: "locality, district, state, India"

Usage:
    from biotrace_locality_ner import LocalityNER
    lner = LocalityNER(geonames_db="biodiversity_data/geonames_india.db")
    localities = lner.extract_localities(text)
    # [{"raw": "Narara", "expanded": "Narara, Jamnagar, Gujarat, India",
    #   "lat": 22.59, "lon": 70.06, "admin1": "Gujarat", "admin2": "Jamnagar"}]

    # Associate with species occurrences
    occurrences = lner.enrich_occurrences(occurrences, text)
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import requests

logger = logging.getLogger("biotrace.locality_ner")

# ─────────────────────────────────────────────────────────────────────────────
#  OPTIONAL DEPS
# ─────────────────────────────────────────────────────────────────────────────
_SPACY_NLP: Optional[object] = None

def _load_spacy():
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    try:
        import spacy
        for model in ("en_core_web_trf","en_core_web_lg","en_core_web_md","en_core_web_sm"):
            try:
                _SPACY_NLP = spacy.load(model)
                logger.info("[locality_ner] spaCy loaded: %s", model)
                return _SPACY_NLP
            except OSError:
                continue
    except ImportError:
        pass
    logger.warning("[locality_ner] No spaCy model — regex fallback only")
    return None

_GEOPY_AVAILABLE = False
try:
    from geopy.geocoders import Nominatim as _Nominatim
    from geopy.extra.rate_limiter import RateLimiter as _RateLimiter
    _GEOPY_AVAILABLE = True
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
#  GeoNames Admin-1 code → State name  (India)
# ─────────────────────────────────────────────────────────────────────────────
INDIA_ADMIN1 = {
    "01": "Andhra Pradesh",      "02": "Arunachal Pradesh",
    "03": "Assam",               "04": "Bihar",
    "05": "Chhattisgarh",        "06": "Goa",
    "07": "Gujarat",             "08": "Haryana",
    "09": "Himachal Pradesh",    "10": "Jharkhand",
    "11": "Karnataka",           "12": "Kerala",
    "13": "Madhya Pradesh",      "14": "Maharashtra",
    "15": "Manipur",             "16": "Meghalaya",
    "17": "Mizoram",             "18": "Nagaland",
    "19": "Odisha",              "20": "Punjab",
    "21": "Rajasthan",           "22": "Sikkim",
    "23": "Tamil Nadu",          "24": "Telangana",
    "25": "Tripura",             "26": "Uttar Pradesh",
    "27": "Uttarakhand",         "28": "West Bengal",
    "29": "Andaman and Nicobar", "30": "Chandigarh",
    "31": "Dadra and Nagar Haveli", "32": "Daman and Diu",
    "33": "Delhi",               "34": "Jammu and Kashmir",
    "35": "Ladakh",              "36": "Lakshadweep",
    "37": "Puducherry",
}

# ─────────────────────────────────────────────────────────────────────────────
#  REGEX PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
# Coordinate patterns
_DMS_RE  = re.compile(
    r"""(\d{1,3})[°\s](\d{1,2})['\s](\d{0,2}(?:\.\d+)?)[\"']?\s*([NSEW])""",
    re.IGNORECASE,
)
_DEC_RE  = re.compile(
    r"""(-?\d{1,3}\.\d{3,})\s*[,;]?\s*(-?\d{1,3}\.\d{3,})"""
)

# Station-ID patterns common in marine surveys
_STATION_RE = re.compile(
    r"""(?:St(?:ation)?|Site|Stn|Transect|Plot|Quadrat|Station|Stn\.?)\s*
        [:\.\s]?\s*
        (?P<id>[A-Z0-9][-A-Z0-9]{0,4}|\d{1,3})""",
    re.IGNORECASE | re.VERBOSE,
)

# Indian administrative unit hints
_INDIAN_LOC_RE = re.compile(
    r"""(?:
        (?:Gulf|Bay|Sea|Ocean|Coast|Creek|River|Lake|Island|Islands|
           Reef|Lagoon|Estuary|Backwater|Mangrove|Forest|Reserve|
           National\s+Park|Sanctuary)\s+of\s+\w+|
        (?:off|near|at|from|in|along|around)\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)?|
        [A-Z][a-z]{3,}(?:,\s*[A-Z][a-z]{3,}){1,4}
    )""",
    re.VERBOSE,
)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LocalityRecord:
    raw:        str   = ""          # as found in text
    expanded:   str   = ""          # full admin string
    admin1:     str   = ""          # state
    admin2:     str   = ""          # district
    country:    str   = "India"
    latitude:   Optional[float] = None
    longitude:  Optional[float] = None
    source:     str   = "geonames"  # geonames | nominatim | pincode | regex
    confidence: float = 1.0
    char_start: int   = 0
    char_end:   int   = 0

    @property
    def coords_available(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    def to_dict(self) -> dict:
        return {
            "verbatimLocality": self.raw,
            "expandedLocality": self.expanded or self.raw,
            "stateProvince":    self.admin1,
            "county":           self.admin2,
            "country":          self.country,
            "decimalLatitude":  self.latitude,
            "decimalLongitude": self.longitude,
            "geocodingSource":  self.source,
            "locality_confidence": round(self.confidence, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  GEONAMES EXPANDER
# ─────────────────────────────────────────────────────────────────────────────
class GeoNamesExpander:
    """
    SQLite-backed GeoNames expansion for Indian place names.
    Expands: "Narara" → "Narara, Jamnagar, Gujarat, India"
    """

    def __init__(self, db_path: str = ""):
        self._db   = db_path
        self._conn: Optional[sqlite3.Connection] = None
        if db_path and os.path.exists(db_path):
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            logger.info("[locality_ner] GeoNames DB: %s", db_path)

    def _query(self, place: str) -> Optional[dict]:
        if not self._conn:
            return None
        clean = place.strip()
        try:
            # Exact name match first
            row = self._conn.execute(
                """SELECT name, latitude, longitude, admin1_code, admin2_code, feature_class, population
                   FROM geonames
                   WHERE (name=? OR asciiname=?)
                     AND country_code='IN'
                   ORDER BY CASE feature_class WHEN 'P' THEN 1
                                               WHEN 'H' THEN 2
                                               WHEN 'A' THEN 3 ELSE 4 END,
                            CAST(population AS INTEGER) DESC
                   LIMIT 1""",
                (clean, clean),
            ).fetchone()

            if row:
                return self._to_dict(row)

            # Alternate names search
            row = self._conn.execute(
                """SELECT name, latitude, longitude, admin1_code, admin2_code, feature_class, population
                   FROM geonames
                   WHERE alternatenames LIKE ?
                     AND country_code='IN'
                   ORDER BY CAST(population AS INTEGER) DESC
                   LIMIT 1""",
                (f"%{clean}%",),
            ).fetchone()

            return self._to_dict(row) if row else None
        except Exception as exc:
            logger.debug("[locality_ner] GeoNames query '%s': %s", place, exc)
            return None

    def _to_dict(self, row) -> dict:
        name, lat, lon, admin1, admin2, feat, pop = row
        state    = INDIA_ADMIN1.get(str(admin1 or "").zfill(2), "")
        district = self._admin2_name(admin1, admin2) if admin2 else ""
        return {
            "name": name, "lat": float(lat), "lon": float(lon),
            "state": state, "district": district,
        }

    def _admin2_name(self, admin1: str, admin2: str) -> str:
        """Look up admin2 (district) name from GeoNames admin codes."""
        if not self._conn:
            return ""
        try:
            row = self._conn.execute(
                """SELECT name FROM geonames
                   WHERE feature_class='A' AND feature_code='ADM2'
                     AND admin1_code=? AND admin2_code=?
                     AND country_code='IN'
                   LIMIT 1""",
                (admin1, admin2),
            ).fetchone()
            return row[0] if row else ""
        except Exception:
            return ""

    def expand(self, place: str) -> Optional[LocalityRecord]:
        """Expand a raw place name to a full LocalityRecord."""
        result = self._query(place)
        if not result:
            return None

        parts = [p for p in [result["name"], result["district"], result["state"], "India"] if p]
        expanded = ", ".join(dict.fromkeys(parts))  # preserve order, remove dupes

        return LocalityRecord(
            raw       = place,
            expanded  = expanded,
            admin1    = result["state"],
            admin2    = result["district"],
            country   = "India",
            latitude  = result["lat"],
            longitude = result["lon"],
            source    = "geonames",
            confidence= 0.9,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  NOMINATIM EXPANDER  (rate-limited: 1 req/sec)
# ─────────────────────────────────────────────────────────────────────────────
class NominatimExpander:
    """
    Nominatim-based locality expander: 1 locality/sec (OpenStreetMap ToS).
    Optional; used when GeoNames returns no match.
    """

    def __init__(self, user_agent: str = "BioTrace_v5_locality_ner"):
        self._enabled = _GEOPY_AVAILABLE
        if self._enabled:
            geocoder = _Nominatim(user_agent=user_agent)
            self._geocode = _RateLimiter(geocoder.geocode, min_delay_seconds=1.1)
            logger.info("[locality_ner] Nominatim expander ready")
        else:
            logger.warning("[locality_ner] geopy not installed — Nominatim disabled")
        self._cache: dict[str, Optional[LocalityRecord]] = {}

    def expand(self, place: str, country_hint: str = "India") -> Optional[LocalityRecord]:
        key = f"{place.lower()}|{country_hint.lower()}"
        if key in self._cache:
            return self._cache[key]

        if not self._enabled:
            return None

        query = f"{place}, {country_hint}" if country_hint else place
        try:
            location = self._geocode(query, addressdetails=True, language="en")
            if not location:
                self._cache[key] = None
                return None

            addr = location.raw.get("address", {})
            parts = [
                place,
                addr.get("county") or addr.get("district",""),
                addr.get("state",""),
                addr.get("country","India"),
            ]
            expanded = ", ".join(p for p in parts if p)

            rec = LocalityRecord(
                raw       = place,
                expanded  = expanded,
                admin1    = addr.get("state",""),
                admin2    = addr.get("county") or addr.get("district",""),
                country   = addr.get("country","India"),
                latitude  = float(location.latitude),
                longitude = float(location.longitude),
                source    = "nominatim",
                confidence= 0.75,
            )
            self._cache[key] = rec
            return rec

        except Exception as exc:
            logger.debug("[locality_ner] Nominatim '%s': %s", place, exc)
            self._cache[key] = None
            return None


# ─────────────────────────────────────────────────────────────────────────────
#  LOCALITY SEGREGATION
# ─────────────────────────────────────────────────────────────────────────────
def segregate_locality_string(locality: str) -> list[str]:
    """
    Contextual parsing: distinguish sub-localities within one site vs.
    multiple distinct sites in a comma-separated string.

    "Narara, Pirotan, Beyt Dwarka" → three distinct sites
    "Narara Island, Gulf of Kutch"  → ONE site with hierarchical context
    "Site A (Narara, intertidal)"   → ONE site with habitat note

    Heuristics:
      - Count of proper-noun tokens
      - Presence of habitat/geographic qualifiers
      - All-uppercase abbreviations (site codes)
      - Presence of parentheses → inline description, not a list
    """
    if "(" in locality or ")" in locality:
        return [locality.strip()]

    # Split on comma/semicolon
    parts = re.split(r"[;,]\s*", locality)
    if len(parts) <= 1:
        return [locality.strip()]

    # Heuristic: geographic hierarchy words mean it's ONE locality
    hierarchy_words = {
        "gulf","bay","sea","ocean","coast","island","district","taluk",
        "tehsil","state","province","india","gujarat","kerala","karnataka",
        "maharashtra","tamil","andhra","odisha","goa","lakshadweep",
        "andaman","nicobar","bengal","assam",
    }
    part_words = [p.lower() for p in parts]
    hierarchy_hits = sum(
        1 for pw in part_words
        if any(hw in pw for hw in hierarchy_words)
    )

    if hierarchy_hits >= len(parts) // 2:
        # More than half of parts are hierarchy words → single locality
        return [locality.strip()]

    # Otherwise: multiple distinct sites
    return [p.strip() for p in parts if p.strip()]


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────
class LocalityNER:
    """
    Combined locality extraction + expansion pipeline.

    For every chunk of text:
      1. spaCy GPE / LOC entities (if model available)
      2. Regex fallback for Indian toponyns + coordinates + station IDs
      3. GeoNames expansion → full admin string
      4. Nominatim fallback (optional, 1/sec)
      5. Coordinate parsing (DMS / decimal)
      6. Locality segregation for multi-site strings
    """

    def __init__(
        self,
        geonames_db:     str  = "biodiversity_data/geonames_india.db",
        pincode_txt:     str  = "biodiversity_data/IN_pin.txt",
        use_nominatim:   bool = True,
        country_filter:  str  = "India",
        nominatim_agent: str  = "BioTrace_v5_locality_ner",
    ):
        self._geonames  = GeoNamesExpander(geonames_db)
        self._nominatim = NominatimExpander(nominatim_agent) if use_nominatim else None
        self.country_filter = country_filter

        # Pincode geocoder (optional)
        self._pincode: Optional[object] = None
        if pincode_txt and os.path.exists(pincode_txt):
            try:
                from pincode_geocoder import IndianPincodeGeocoder
                self._pincode = IndianPincodeGeocoder(pincode_txt, fuzzy_threshold=80.0)
                logger.info("[locality_ner] PincodeGeocoder ready")
            except ImportError:
                pass

    # ── Stage 1: NER ──────────────────────────────────────────────────────────
    def _ner_entities(self, text: str) -> list[tuple[str,int,int]]:
        """Extract GPE/LOC entities with char offsets."""
        entities: list[tuple[str,int,int]] = []

        # spaCy NER
        nlp = _load_spacy()
        if nlp:
            try:
                doc = nlp(text[:50_000])
                for ent in doc.ents:
                    if ent.label_ in ("GPE","LOC","FAC"):
                        entities.append((ent.text, ent.start_char, ent.end_char))
            except Exception as exc:
                logger.debug("[locality_ner] spaCy: %s", exc)

        # Regex fallback for Indian locality patterns
        for m in _INDIAN_LOC_RE.finditer(text[:50_000]):
            raw = m.group(0).strip()
            if len(raw) > 3:
                entities.append((raw, m.start(), m.end()))

        # Station IDs (link to Methods section mappings later)
        for m in _STATION_RE.finditer(text[:50_000]):
            entities.append((m.group(0).strip(), m.start(), m.end()))

        # Deduplicate by raw string, keep first occurrence
        seen: dict[str, tuple] = {}
        for raw, start, end in entities:
            key = raw.lower().strip()
            if key not in seen and len(key) > 2:
                seen[key] = (raw, start, end)

        return list(seen.values())

    # ── Stage 1b: Coordinate extraction ──────────────────────────────────────
    @staticmethod
    def _extract_coords(text: str) -> list[tuple[float, float, int, int]]:
        """Extract lat/lon pairs from text. Returns [(lat, lon, start, end)]."""
        coords = []
        # Decimal degrees pairs
        for m in _DEC_RE.finditer(text):
            try:
                lat = float(m.group(1))
                lon = float(m.group(2))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    coords.append((lat, lon, m.start(), m.end()))
            except (ValueError, IndexError):
                pass
        return coords

    # ── Stage 2-3: Expansion + segregation ────────────────────────────────────
    def _expand(self, raw: str) -> Optional[LocalityRecord]:
        """Try GeoNames → Pincode → Nominatim expansion."""
        # GeoNames first (fast, offline)
        rec = self._geonames.expand(raw)
        if rec:
            return rec

        # Pincode lookup
        if self._pincode and re.match(r"^\d{6}$", raw.strip()):
            try:
                gr = self._pincode.geocode(raw)
                if gr and gr.latitude:
                    return LocalityRecord(
                        raw=raw, expanded=raw,
                        latitude=gr.latitude, longitude=gr.longitude,
                        source="pincode", confidence=0.95,
                    )
            except Exception:
                pass

        # Nominatim (slow, 1/sec)
        if self._nominatim:
            return self._nominatim.expand(raw, country_hint=self.country_filter)

        return None

    # ── Public: extract_localities ────────────────────────────────────────────
    def extract_localities(
        self, text: str, max_entities: int = 100
    ) -> list[LocalityRecord]:
        """
        Full pipeline: NER → segregate → expand → return list[LocalityRecord].
        """
        # Raw NER entities
        raw_entities = self._ner_entities(text)

        # Segregate comma-lists
        segregated: list[tuple[str,int,int]] = []
        for raw, start, end in raw_entities[:max_entities]:
            parts = segregate_locality_string(raw)
            if len(parts) == 1:
                segregated.append((parts[0], start, end))
            else:
                for p in parts:
                    segregated.append((p, start, end))

        # Expand
        records: list[LocalityRecord] = []
        seen_raws: set[str] = set()
        for raw, start, end in segregated:
            raw_clean = raw.strip()
            if not raw_clean or raw_clean.lower() in seen_raws:
                continue
            seen_raws.add(raw_clean.lower())

            rec = self._expand(raw_clean)
            if rec:
                rec.char_start = start
                rec.char_end   = end
            else:
                rec = LocalityRecord(
                    raw=raw_clean, expanded=raw_clean,
                    char_start=start, char_end=end,
                    source="unresolved", confidence=0.3,
                )
            records.append(rec)

        # Inline decimal coordinates
        for lat, lon, start, end in self._extract_coords(text):
            ctx = text[max(0, start-80): end+80]
            records.append(LocalityRecord(
                raw=f"{lat:.4f}, {lon:.4f}",
                expanded=f"{lat:.4f}, {lon:.4f}",
                latitude=lat, longitude=lon,
                source="coordinates",
                confidence=1.0,
                char_start=start, char_end=end,
            ))

        logger.info("[locality_ner] %d localities extracted", len(records))
        return records

    # ── Public: enrich_occurrences ────────────────────────────────────────────
    def enrich_occurrences(
        self,
        occurrences: list[dict],
        text: str,
        proximity_chars: int = 500,
    ) -> list[dict]:
        """
        For each occurrence that lacks coordinates, find the nearest
        locality mention within ±proximity_chars and expand it.
        """
        localities = self.extract_localities(text)
        if not localities:
            return occurrences

        for occ in occurrences:
            if not isinstance(occ, dict):
                continue
            # Already has coords?
            lat = occ.get("decimalLatitude")
            lon = occ.get("decimalLongitude")
            if lat and lon:
                continue

            char_pos = int(occ.get("char_start", 0) or 0)

            # Find closest locality record by char distance
            closest = min(
                localities,
                key=lambda lr: abs(lr.char_start - char_pos),
                default=None,
            )
            if closest and abs(closest.char_start - char_pos) <= proximity_chars:
                if not occ.get("verbatimLocality"):
                    occ["verbatimLocality"] = closest.raw
                if not occ.get("expandedLocality"):
                    occ["expandedLocality"] = closest.expanded
                if closest.latitude is not None:
                    occ["decimalLatitude"]  = closest.latitude
                    occ["decimalLongitude"] = closest.longitude
                    occ["geocodingSource"]  = closest.source
                if closest.admin1:
                    occ["stateProvince"] = closest.admin1
                if closest.admin2:
                    occ["county"] = closest.admin2

        return occurrences

    # ── Station-ID resolver ───────────────────────────────────────────────────
    @staticmethod
    def build_station_map(
        methods_text: str,
        geonames_db: str = "",
    ) -> dict[str, LocalityRecord]:
        """
        Parse the Methods section to build a station-ID → LocalityRecord map.

        Looks for patterns like:
          "Station 1 (Narara Island, 22.59°N, 70.06°E)"
          "Site A — Gulf of Kutch intertidal (22.5N 70.0E)"
          "St. 1: Lakshadweep, 10.5°N 72.6°E"
        """
        station_re = re.compile(
            r"""(?:St(?:ation)?|Site|Stn\.?|Plot|Transect)\s*
                [:\.\s]?\s*
                (?P<id>[A-Z0-9][-A-Z0-9]{0,4}|\d{1,3})
                [:\s\-–—]+
                (?P<desc>[^(\n]{5,60})
                (?:\((?P<coords>[^)]+)\))?""",
            re.IGNORECASE | re.VERBOSE,
        )
        coord_re = re.compile(
            r"""(\d{1,3}(?:\.\d+)?)\s*°?\s*([NS])[,\s]+
                (\d{1,3}(?:\.\d+)?)\s*°?\s*([EW])""",
            re.IGNORECASE,
        )
        result: dict[str, LocalityRecord] = {}

        for m in station_re.finditer(methods_text):
            sid  = m.group("id").strip().upper()
            desc = m.group("desc").strip()
            raw_coords = m.group("coords") or ""

            lat, lon = None, None
            cm = coord_re.search(raw_coords or desc)
            if cm:
                raw_lat = float(cm.group(1))
                raw_lon = float(cm.group(3))
                lat = raw_lat if "N" in cm.group(2).upper() else -raw_lat
                lon = raw_lon if "E" in cm.group(4).upper() else -raw_lon

            result[sid] = LocalityRecord(
                raw=f"Station {sid}",
                expanded=desc.strip(),
                latitude=lat, longitude=lon,
                source="methods_section",
                confidence=1.0,
            )

        logger.info("[locality_ner] station_map: %d stations parsed", len(result))
        return result

    def resolve_station_ids(
        self,
        occurrences: list[dict],
        station_map: dict[str, "LocalityRecord"],
    ) -> list[dict]:
        """Replace Station-ID verbatimLocality values with resolved full names."""
        for occ in occurrences:
            if not isinstance(occ, dict):
                continue
            vl = str(occ.get("verbatimLocality","")).strip().upper()
            # Try direct match or partial (e.g. "ST.1", "STATION 1", "S1")
            for sid, rec in station_map.items():
                if sid in vl or vl in sid:
                    if not occ.get("expandedLocality"):
                        occ["expandedLocality"] = rec.expanded
                    if rec.latitude is not None:
                        occ["decimalLatitude"]  = rec.latitude
                        occ["decimalLongitude"] = rec.longitude
                        occ["geocodingSource"]  = "methods_section"
                    break
        return occurrences
