"""
coord_utils.py — Robust coordinate parsing and geographic validation for BioTrace.

Handles:
  • DMS → DD conversion with OCR error correction
  • India state / district bounding-box validation
  • Ocean / out-of-country coordinate flagging
  • Pincode-centroid-based coordinate sanity checking
"""

from __future__ import annotations
import re
import logging
from typing import Optional

logger = logging.getLogger("biotrace.coord")

# ─────────────────────────────────────────────────────────────────────────────
#  INDIA NATIONAL & STATE BOUNDING BOXES
#  Format: (lat_min, lat_max, lon_min, lon_max)
# ─────────────────────────────────────────────────────────────────────────────
INDIA_BBOX = (6.0, 37.6, 68.0, 97.5)

INDIA_STATE_BBOX: dict[str, tuple[float, float, float, float]] = {
    "andhra pradesh":       (12.6, 19.9, 76.7, 84.8),
    "arunachal pradesh":    (26.6, 29.5, 91.5, 97.4),
    "assam":                (24.1, 27.9, 89.7, 96.0),
    "bihar":                (24.3, 27.5, 83.3, 88.3),
    "chhattisgarh":         (17.8, 24.1, 80.2, 84.4),
    "goa":                  (14.9, 15.8, 73.7, 74.3),
    "gujarat":              (20.1, 24.7, 68.2, 74.5),
    "haryana":              (27.7, 30.9, 74.5, 77.6),
    "himachal pradesh":     (30.4, 33.2, 75.6, 79.0),
    "jharkhand":            (21.9, 25.3, 83.3, 87.9),
    "karnataka":            (11.6, 18.5, 74.1, 78.6),
    "kerala":               (8.3,  12.8, 74.8, 77.4),
    "madhya pradesh":       (21.1, 26.9, 74.0, 82.8),
    "maharashtra":          (15.6, 22.0, 72.6, 80.9),
    "manipur":              (23.8, 25.7, 93.0, 94.8),
    "meghalaya":            (25.0, 26.1, 89.8, 92.8),
    "mizoram":              (21.9, 24.5, 92.3, 93.5),
    "nagaland":             (25.2, 27.0, 93.3, 95.3),
    "odisha":               (17.8, 22.6, 81.4, 87.5),
    "punjab":               (29.5, 32.5, 73.9, 76.9),
    "rajasthan":            (23.0, 30.2, 69.5, 78.3),
    "sikkim":               (27.1, 28.1, 88.0, 88.9),
    "tamil nadu":           (8.1,  13.6, 76.2, 80.4),
    "telangana":            (15.8, 19.9, 77.2, 81.3),
    "tripura":              (22.9, 24.5, 91.2, 92.3),
    "uttar pradesh":        (23.9, 30.4, 77.1, 84.6),
    "uttarakhand":          (28.7, 31.5, 77.6, 81.1),
    "west bengal":          (21.5, 27.2, 85.8, 89.9),
    "andaman nicobar":      (6.8,  13.7, 92.2, 93.9),
    "lakshadweep":          (8.0,  12.5, 71.8, 74.0),
    "jammu kashmir":        (32.3, 37.1, 73.9, 80.4),
    "ladakh":               (32.3, 36.2, 75.8, 80.4),
}

# Neighbouring country rough bboxes to detect cross-border errors
NEIGHBOUR_BBOX: dict[str, tuple[float, float, float, float]] = {
    "pakistan":   (23.6, 37.1, 60.9, 77.8),
    "china":      (18.2, 53.6, 73.7, 135.1),
    "nepal":      (26.4, 30.4, 80.1, 88.2),
    "bangladesh": (20.7, 26.6, 88.0, 92.7),
    "myanmar":    (9.8,  28.5, 92.2, 101.2),
    "sri lanka":  (5.9,  9.9,  79.7, 81.9),
}

# ─────────────────────────────────────────────────────────────────────────────
#  MARINE KEYWORDS — indicates an oceanic location IS valid
# ─────────────────────────────────────────────────────────────────────────────
MARINE_KEYWORDS = frozenset([
    "sea", "ocean", "marine", "offshore", "pelagic", "benthic", "reef",
    "coral", "deep", "trawl", "demersal", "littoral", "intertidal",
    "subtidal", "mangrove", "estuary", "coastal", "gulf", "bay",
    "island", "archipelago", "atoll", "seamount", "abyssal",
])

# ─────────────────────────────────────────────────────────────────────────────
#  OCR ERROR PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
# Map OCR noise → clean DMS
_OCR_FIXES: list[tuple[re.Pattern, str]] = [
    # "N 17047'13.3" → "N 17°47'13.3""  (zero before 2-digit minute is really °)
    (re.compile(r"([NS])\s*(\d{1,2})0(\d{2}['\u2019\u02bc])"), r"\1 \2°\3"),
    # "E 73009'13.8" → "E 73°09'13.8""
    (re.compile(r"([EW])\s*(\d{1,3})0(\d{2}['\u2019\u02bc])"), r"\1 \2°\3"),
    # letter O → degree symbol when sandwiched between digits
    (re.compile(r"(\d)O(\d)"), r"\1°\2"),
    # "17d47m13s" style
    (re.compile(r"(\d+)\s*[dD°]\s*(\d+)\s*[mM']\s*([\d.]+)\s*[sS\"]"), r"\1°\2'\3\""),
    # Unicode prime variants → standard quote
    (re.compile(r"[\u2032\u02bc\u2019]"), "'"),
    (re.compile(r"[\u2033\u201d]"), '"'),
    # Remove invisible chars / non-breaking spaces
    (re.compile(r"[\u00a0\u200b\u200c\u200d\ufeff]"), " "),
]

# ─────────────────────────────────────────────────────────────────────────────
#  DMS REGEX — matches after OCR cleanup
#  Captures: hemisphere, degrees, minutes (optional), seconds (optional)
# ─────────────────────────────────────────────────────────────────────────────
_DMS_RE = re.compile(
    r"""
    (?P<hemi>[NSEWnsew])\s*          # hemisphere letter
    (?P<deg>\d{1,3})                 # degrees
    [°\s]                            # separator
    (?P<min>\d{1,2})                 # minutes
    ['′\s]                           # separator
    (?:(?P<sec>[\d.]+)[\"″\s]?)?     # optional seconds
    \s*(?P<hemi2>[NSEWnsew])?        # optional trailing hemisphere
    """,
    re.VERBOSE,
)

_DECIMAL_RE = re.compile(
    r"(?P<hemi>[NSEWnsew])?\s*(?P<sign>[-−–])?(?P<val>\d{1,3}\.\d+)\s*°?\s*(?P<hemi2>[NSEWnsew])?",
    re.IGNORECASE,
)


def _apply_ocr_fixes(s: str) -> str:
    for pattern, repl in _OCR_FIXES:
        s = pattern.sub(repl, s)
    return s


def _hemisphere_sign(h: str | None) -> float:
    if h and h.upper() in ("S", "W"):
        return -1.0
    return 1.0


def parse_dms(raw: str) -> Optional[float]:
    """
    Parse a DMS or decimal-degree string → float decimal degrees.
    Applies OCR error corrections before parsing.
    Returns None on failure.
    """
    if not raw:
        return None
    cleaned = _apply_ocr_fixes(str(raw).strip())

    # Try DMS first
    m = _DMS_RE.search(cleaned)
    if m:
        hemi = (m.group("hemi") or m.group("hemi2") or "").upper()
        try:
            deg  = float(m.group("deg"))
            mins = float(m.group("min") or 0)
            secs = float(m.group("sec") or 0)
            dd = (deg + mins / 60.0 + secs / 3600.0) * _hemisphere_sign(hemi)
            if _is_plausible_dd(dd, hemi):
                return round(dd, 7)
        except (TypeError, ValueError):
            pass

    # Try plain decimal
    m2 = _DECIMAL_RE.search(cleaned)
    if m2:
        hemi = (m2.group("hemi") or m2.group("hemi2") or "").upper()
        sign = -1.0 if (m2.group("sign") or "") in ("-", "−", "–") else 1.0
        try:
            dd = float(m2.group("val")) * sign * _hemisphere_sign(hemi or None)
            if _is_plausible_dd(dd, hemi):
                return round(dd, 7)
        except (TypeError, ValueError):
            pass

    return None


def _is_plausible_dd(dd: float, hemi: str = "") -> bool:
    """Sanity-check that a decimal-degree value is within valid range."""
    if hemi in ("N", "S", ""):
        return -90.0 <= dd <= 90.0
    if hemi in ("E", "W"):
        return -180.0 <= dd <= 180.0
    return -180.0 <= dd <= 180.0


# ─────────────────────────────────────────────────────────────────────────────
#  GEOGRAPHIC VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def point_in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    lat_min, lat_max, lon_min, lon_max = bbox
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def infer_state_from_text(text: str) -> Optional[str]:
    """Scan locality / habitat text for an Indian state name."""
    if not text:
        return None
    t = text.lower()
    for state in INDIA_STATE_BBOX:
        if state in t:
            return state
    return None


def is_marine_context(occ: dict) -> bool:
    """Return True if the occurrence context strongly implies a marine location."""
    fields = " ".join(filter(None, [
        str(occ.get("habitat", "")),
        str(occ.get("samplingProtocol", "")),
        str(occ.get("verbatimLocality", "")),
        str(occ.get("sourceSentence", "")),
        str(occ.get("environmentalData", "")),
        str(occ.get("depth", "")),
    ])).lower()
    return any(kw in fields for kw in MARINE_KEYWORDS)


def validate_occurrence_coordinates(occ: dict) -> dict:
    """
    Validate and correct coordinates in a single occurrence dict.

    Rules (applied in order):
    1.  Re-parse lat/lon from raw strings if the LLM returned non-numeric values.
    2.  If coordinates are outside the global valid range → null them.
    3.  If the study context is Indian but coordinates fall outside India AND
        not in a neighbour country that's plausible → null them.
    4.  If coordinates place the record in the open ocean AND the context is
        NOT marine → null them so geocoding can fill in a proper land point.
    5.  If a state name is detectable in the locality string, check that the
        coordinates fall within that state's bounding box; if not → null.
    6.  Record the validation outcome in a new field `coordValidationNote`.
    """
    lat = occ.get("decimalLatitude")
    lon = occ.get("decimalLongitude")

    # ── 1. Re-parse if string coordinates slipped through ────────────────────
    if isinstance(lat, str) and lat.strip():
        lat = parse_dms(lat)
        occ["decimalLatitude"] = lat
    if isinstance(lon, str) and lon.strip():
        lon = parse_dms(lon)
        occ["decimalLongitude"] = lon

    # Normalise to float | None
    try:
        lat = float(lat) if lat is not None else None
    except (TypeError, ValueError):
        lat = None
    try:
        lon = float(lon) if lon is not None else None
    except (TypeError, ValueError):
        lon = None

    # ── 2. Global range check ─────────────────────────────────────────────────
    if lat is not None and not (-90 <= lat <= 90):
        occ["coordValidationNote"] = f"INVALID_RANGE lat={lat}"
        lat = lon = None
    if lon is not None and not (-180 <= lon <= 180):
        occ["coordValidationNote"] = f"INVALID_RANGE lon={lon}"
        lat = lon = None

    occ["decimalLatitude"]  = lat
    occ["decimalLongitude"] = lon

    if lat is None or lon is None:
        return occ  # nothing more to validate

    # ── 3. India bounds check ────────────────────────────────────────────────
    in_india = point_in_bbox(lat, lon, INDIA_BBOX)
    if not in_india:
        if is_marine_context(occ):
            # Offshore / sea collection — allow it
            occ.setdefault("coordValidationNote", "MARINE_OFFSHORE_ALLOWED")
        else:
            # Check if point falls in a plausible neighbouring country
            in_neighbour = any(
                point_in_bbox(lat, lon, b) for b in NEIGHBOUR_BBOX.values()
            )
            if not in_neighbour:
                occ["coordValidationNote"] = (
                    f"OUTSIDE_INDIA_BBOX lat={lat:.4f} lon={lon:.4f} — nulled"
                )
                occ["decimalLatitude"] = occ["decimalLongitude"] = None
                return occ

    # ── 4. Ocean check (rough heuristic for Indian Ocean region) ─────────────
    if not in_india and not is_marine_context(occ):
        # Indian Ocean region: lat 0–25°N, lon 55–90°E but outside land bbox
        if 0 <= lat <= 25 and 55 <= lon <= 90:
            occ["coordValidationNote"] = (
                f"POSSIBLE_OCEAN lat={lat:.4f} lon={lon:.4f} — nulled (no marine context)"
            )
            occ["decimalLatitude"] = occ["decimalLongitude"] = None
            return occ

    # ── 5. State-level bounds check ───────────────────────────────────────────
    locality_text = " ".join(filter(None, [
        str(occ.get("verbatimLocality", "")),
        str(occ.get("habitat", "")),
        str(occ.get("sourceSentence", "")),
    ]))
    state = infer_state_from_text(locality_text)
    if state:
        sbbox = INDIA_STATE_BBOX[state]
        # Expand state box by 0.5° to account for border regions
        expanded = (sbbox[0] - 0.5, sbbox[1] + 0.5, sbbox[2] - 0.5, sbbox[3] + 0.5)
        if not point_in_bbox(lat, lon, expanded):
            occ["coordValidationNote"] = (
                f"OUTSIDE_STATE_BBOX state={state} lat={lat:.4f} lon={lon:.4f} — nulled"
            )
            occ["decimalLatitude"] = occ["decimalLongitude"] = None

    return occ


def validate_occurrence_batch(occurrences: list[dict]) -> list[dict]:
    """Apply validate_occurrence_coordinates() to every record in a list."""
    return [validate_occurrence_coordinates(occ) for occ in occurrences if isinstance(occ, dict)]


# ─────────────────────────────────────────────────────────────────────────────
#  PINCODE-BASED BOUNDING BOX VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def build_pincode_bbox_index(pincode_csv_path: str) -> dict[str, tuple[float, float, float, float]]:
    """
    Build a dict mapping each 6-digit pincode → bounding box
    (lat_min, lat_max, lon_min, lon_max).

    Expected CSV columns (any order): Pincode, Latitude, Longitude
    Also builds district-level aggregated boxes keyed by
    "STATE::DISTRICT" (uppercased).

    Returns {} if the file is missing or unreadable.
    """
    import os, pandas as pd

    if not os.path.exists(pincode_csv_path):
        return {}
    try:
        df = pd.read_csv(pincode_csv_path, dtype=str, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]

        # Normalise column names
        col_map = {}
        for c in df.columns:
            if "pin" in c:           col_map[c] = "pincode"
            elif "lat" in c:         col_map[c] = "lat"
            elif "lon" in c or "lng" in c: col_map[c] = "lon"
            elif "district" in c:    col_map[c] = "district"
            elif "state" in c:       col_map[c] = "state"
        df = df.rename(columns=col_map)

        required = {"pincode", "lat", "lon"}
        if not required.issubset(df.columns):
            logger.warning(f"[pincode_bbox] Missing columns {required - set(df.columns)}")
            return {}

        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat", "lon"])

        PAD = 0.15  # ~15 km padding around each pincode centroid
        index: dict[str, tuple[float, float, float, float]] = {}

        for _, row in df.iterrows():
            lat, lon = float(row["lat"]), float(row["lon"])
            pin = str(row["pincode"]).strip().zfill(6)
            index[pin] = (lat - PAD, lat + PAD, lon - PAD, lon + PAD)

        # District-level aggregated boxes
        if "district" in df.columns and "state" in df.columns:
            for (state, dist), grp in df.groupby(
                [df["state"].str.upper(), df["district"].str.upper()], sort=False
            ):
                key = f"{state}::{dist}"
                index[key] = (
                    grp["lat"].min() - 0.1,
                    grp["lat"].max() + 0.1,
                    grp["lon"].min() - 0.1,
                    grp["lon"].max() + 0.1,
                )

        logger.info(f"[pincode_bbox] Loaded {len(index)} entries from {pincode_csv_path}")
        return index

    except Exception as exc:
        logger.warning(f"[pincode_bbox] Failed to build index: {exc}")
        return {}


def validate_with_pincode_bbox(
    occ: dict,
    pincode_index: dict[str, tuple[float, float, float, float]],
) -> dict:
    """
    If a pincode or district string is present in the locality text, validate
    that the coordinates fall within the corresponding bounding box.
    If the coordinates are absent, attempt to fill them from the pincode centroid.
    """
    if not pincode_index:
        return occ

    locality = str(occ.get("verbatimLocality", ""))
    lat = occ.get("decimalLatitude")
    lon = occ.get("decimalLongitude")

    # Look for a 6-digit pincode in the locality string
    pin_match = re.search(r"\b(\d{6})\b", locality)
    if pin_match:
        pin = pin_match.group(1)
        bbox = pincode_index.get(pin)
        if bbox:
            if lat is None or lon is None:
                # Fill from centroid of the pincode bbox
                occ["decimalLatitude"]  = round((bbox[0] + bbox[1]) / 2, 5)
                occ["decimalLongitude"] = round((bbox[2] + bbox[3]) / 2, 5)
                occ["geocodingSource"]  = f"PincodeDB_{pin}"
            else:
                # Validate existing coordinates against pincode bbox
                if not point_in_bbox(lat, lon, bbox):
                    occ["coordValidationNote"] = (
                        f"PINCODE_MISMATCH pin={pin} lat={lat:.4f} lon={lon:.4f} — nulled"
                    )
                    occ["decimalLatitude"] = occ["decimalLongitude"] = None

    return occ
