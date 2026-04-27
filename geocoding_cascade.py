"""
geocoding_cascade.py  —  BioTrace v3.1
────────────────────────────────────────────────────────────────────────────
Unified geocoding pipeline for occurrence records.

Four tools in strict priority order:
  1. coord_utils.parse_dms()     — DMS/OCR strings → decimal degrees (always first)
  2. IndianPincodeGeocoder       — 5-stage fuzzy Indian place matching
  3. GeoNames IN SQLite          — fast local lookup
  4. NominatimEnrichedGeocoder   — district+state-qualified, network fallback

After any coordinate is filled:
  coord_utils.validate_occurrence_coordinates() checks India bbox,
  state-level bbox, ocean context, and pincode mismatch.

Usage
-----
    geo = GeocodingCascade(
        geonames_db   = "biodiversity_data/geonames_india.db",
        pincode_txt   = "biodiversity_data/IN_pin.txt",
        use_nominatim = True,
    )
    occurrences = geo.geocode_batch(occurrences)
"""
from __future__ import annotations
import logging, os, sqlite3
from typing import Optional
logger = logging.getLogger("biotrace.geocoding")


def _to_float(v) -> Optional[float]:
    if v is None: return None
    try:
        f = float(str(v).strip())
        return None if str(v).strip() in ("0","") else f
    except (ValueError, TypeError):
        return None

def _has_coords(occ: dict) -> bool:
    return _to_float(occ.get("decimalLatitude")) is not None \
       and _to_float(occ.get("decimalLongitude")) is not None


def _resolve_occurrence_table(conn: sqlite3.Connection) -> str:
    """
    Prefer the current v4 schema while remaining backward-compatible with
    older databases that still use the legacy `occurrences` table.
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


class GeocodingCascade:
    """Four-tool geocoding cascade for BioTrace occurrence records."""

    def __init__(
        self,
        geonames_db:    str  = "",
        pincode_txt:    str  = "",
        pincode_state:  Optional[str] = None,
        use_nominatim:  bool = False,
        nominatim_agent:str  = "BioTrace_v3_biodiversity_extractor",
    ):
        self.geonames_db   = geonames_db
        self.use_nominatim = use_nominatim

        # Tool 2 — IndianPincodeGeocoder
        self._pincode = None
        if pincode_txt and os.path.exists(pincode_txt):
            try:
                from pincode_geocoder import IndianPincodeGeocoder
                self._pincode = IndianPincodeGeocoder(pincode_txt,
                                                      fuzzy_threshold=80.0,
                                                      state_filter=pincode_state)
                logger.info("[geocoding] PincodeGeocoder ready (%s)", pincode_txt)
            except ImportError:
                logger.warning("[geocoding] pincode_geocoder unavailable — pip install rapidfuzz")
            except Exception as exc:
                logger.warning("[geocoding] PincodeGeocoder init: %s", exc)

        # Tool 4 — NominatimEnrichedGeocoder
        self._nominatim = None
        if use_nominatim:
            try:
                from nominatim_geocoder import NominatimEnrichedGeocoder
                self._nominatim = NominatimEnrichedGeocoder(
                    geonames_db_path=geonames_db,
                    user_agent=nominatim_agent,
                )
                logger.info("[geocoding] NominatimGeocoder ready")
            except ImportError:
                logger.warning("[geocoding] nominatim_geocoder unavailable — pip install geopy")
            except Exception as exc:
                logger.warning("[geocoding] Nominatim init: %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    #  Tool 1 · DMS string parsing
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_dms(occ: dict) -> dict:
        try:
            from coord_utils import parse_dms
        except ImportError:
            return occ
        for field in ("decimalLatitude","decimalLongitude"):
            val = occ.get(field)
            if isinstance(val, str) and val.strip():
                parsed = parse_dms(val.strip())
                if parsed is not None:
                    occ[field] = parsed
        return occ

    # ─────────────────────────────────────────────────────────────────────────
    #  Tool 3 · GeoNames IN SQLite
    # ─────────────────────────────────────────────────────────────────────────
    def _geonames(self, locality: str) -> Optional[tuple[float,float]]:
        if not locality or not self.geonames_db or not os.path.exists(self.geonames_db):
            return None
        try:
            conn = sqlite3.connect(self.geonames_db, check_same_thread=False)
            res  = conn.execute(
                """SELECT latitude,longitude FROM geonames
                   WHERE (name=? OR asciiname=? OR alternatenames LIKE ?)
                   AND country_code='IN'
                   ORDER BY CASE feature_class WHEN 'P' THEN 1 WHEN 'A' THEN 2 ELSE 3 END,
                   CAST(population AS INTEGER) DESC LIMIT 1""",
                (locality, locality, f"%{locality}%")
            ).fetchone()
            conn.close()
            return (float(res[0]), float(res[1])) if res else None
        except Exception as exc:
            logger.debug("[GeoNames] %s → %s", locality, exc)
        return None

    # ─────────────────────────────────────────────────────────────────────────
    #  Coordinate validation
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _validate(occ: dict) -> dict:
        try:
            from coord_utils import validate_occurrence_coordinates
            return validate_occurrence_coordinates(occ)
        except ImportError:
            return occ
        except Exception as exc:
            logger.debug("[validate] %s", exc)
            return occ

    # ─────────────────────────────────────────────────────────────────────────
    #  Public: geocode_batch
    # ─────────────────────────────────────────────────────────────────────────
    def geocode_batch(self, occurrences: list[dict]) -> list[dict]:
        """
        Run the 4-tool cascade on every record in the list.

        geocodingSource values set by each tool:
          "LLM"               — coordinates from LLM extraction
          "IN_Pincode_*"      — pincode geocoder (match_type + score)
          "GeoNames_IN"       — GeoNames SQLite
          "Nominatim"         — Nominatim
        """
        if not occurrences: return occurrences
        result = []

        for occ in occurrences:
            if not isinstance(occ, dict):
                result.append(occ); continue

            # Step 1: Parse DMS strings
            occ = self._parse_dms(occ)

            # If LLM already provided valid numeric coords → validate and pass through
            if _has_coords(occ):
                occ.setdefault("geocodingSource","LLM")
                occ = self._validate(occ)
                result.append(occ); continue

            locality = str(occ.get("verbatimLocality","")).strip()

            # Step 2: Pincode geocoder
            if self._pincode and locality:
                try:
                    gr = self._pincode.geocode(locality)
                    if gr and gr.latitude is not None:
                        occ["decimalLatitude"]  = gr.latitude
                        occ["decimalLongitude"] = gr.longitude
                        occ["geocodingSource"]  = f"IN_Pincode_{gr.match_type}_{gr.score:.0f}"
                        occ = self._validate(occ)
                        result.append(occ); continue
                except Exception as exc:
                    logger.debug("[pincode] %s",exc)

            # Step 3: GeoNames IN
            if locality:
                coords = self._geonames(locality)
                if coords:
                    occ["decimalLatitude"]  = coords[0]
                    occ["decimalLongitude"] = coords[1]
                    occ["geocodingSource"]  = "GeoNames_IN"
                    occ = self._validate(occ)
                    result.append(occ); continue

            result.append(occ)

        # Step 4: Nominatim (batch, deduplicated per unique locality)
        if self._nominatim:
            missing = [o for o in result
                       if isinstance(o,dict) and not _has_coords(o) and o.get("verbatimLocality")]
            if missing:
                logger.info("[geocoding] Nominatim: %d remaining unresolved", len(missing))
                try:
                    geocoded = self._nominatim.geocode_missing(missing)
                    geocoded = [self._validate(o) for o in geocoded]
                    id_map   = {id(o): o for o in geocoded}
                    result   = [id_map.get(id(o), o) for o in result]
                except Exception as exc:
                    logger.warning("[geocoding] Nominatim batch: %s", exc)

        filled = sum(1 for o in result if isinstance(o,dict) and _has_coords(o))
        logger.info("[geocoding] %d/%d records geocoded", filled, len(result))
        return result

    def geocode_single(self, occ: dict) -> dict:
        return self.geocode_batch([occ])[0]

    # ─────────────────────────────────────────────────────────────────────────
    #  Batch DB update (for "Geocode Missing" button)
    # ─────────────────────────────────────────────────────────────────────────
    def batch_geocode_db(self, meta_db_path: str, progress_callback=None) -> int:
        conn = sqlite3.connect(meta_db_path, check_same_thread=False)
        table = _resolve_occurrence_table(conn)
        rows = conn.execute(
            f"""SELECT id,verbatimLocality FROM {table}
               WHERE (decimalLatitude IS NULL OR decimalLongitude IS NULL)
               AND verbatimLocality IS NOT NULL AND verbatimLocality != ''
               AND validationStatus != 'rejected'"""
        ).fetchall()
        if not rows:
            conn.close(); return 0
        logger.info("[geocoding/db] %d rows to geocode", len(rows))
        updated = 0
        for i,(row_id,vl) in enumerate(rows):
            occ = {"verbatimLocality":vl,"decimalLatitude":None,"decimalLongitude":None}
            occ = self.geocode_single(occ)
            lat = _to_float(occ.get("decimalLatitude"))
            lon = _to_float(occ.get("decimalLongitude"))
            if lat is not None and lon is not None:
                conn.execute(
                    f"UPDATE {table} SET decimalLatitude=?,decimalLongitude=?,geocodingSource=? WHERE id=?",
                    (lat,lon,occ.get("geocodingSource",""),row_id))
                updated += 1
            if updated % 50 == 0 and updated > 0: conn.commit()
            if progress_callback: progress_callback(i+1,len(rows))
        conn.commit(); conn.close()
        logger.info("[geocoding/db] %d/%d updated", updated, len(rows))
        return updated
