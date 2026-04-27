# """
# biotrace_schema.py  —  BioTrace v5.2
# ────────────────────────────────────────────────────────────────────────────
# Pydantic v2 strict schema enforcement for BioTrace occurrence records.

# Solves:
#   • "1st chunk JSON parse failures" — LLM sometimes returns malformed JSON;
#     json_repair library attempts auto-repair before Pydantic validation.
#   • Field coercion — "22.5 N" → 22.5, "2022-2023" → "2022/2023"
#   • Missing required fields → default values, not crashes
#   • Extra fields from LLM → silently stripped to keep schema clean
#   • occurrenceType enforcement — only Primary | Secondary | Uncertain
#   • Date normalisation — ISO 8601, or YYYY/YYYY for ranges
#   • Depth coercion — "10-15m" → "10-15"
#   • Coordinates validation — India bbox soft-warn, not reject

# Pydantic v2 features used:
#   • model_validator (mode='before') for pre-processing
#   • field_validator with mode='before'
#   • ConfigDict(extra='ignore', str_strip_whitespace=True)
#   • model_json_schema() for LLM prompt injection

# Usage:
#     from biotrace_schema import OccurrenceRecord, parse_llm_response

#     # Parse raw LLM JSON string (may be malformed):
#     records = parse_llm_response(raw_llm_string, source_citation="Author 2024")

#     # Validate a single dict:
#     rec = OccurrenceRecord.model_validate(raw_dict)

#     # Batch validate a list:
#     records, errors = validate_batch(list_of_dicts)
# """
# from __future__ import annotations

# import json
# import logging
# import re
# from datetime import datetime
# from typing import Any, Literal, Optional

# from pydantic import (
#     BaseModel, ConfigDict, Field,
#     field_validator, model_validator,
# )

# logger = logging.getLogger("biotrace.schema")

# # ─────────────────────────────────────────────────────────────────────────────
# #  JSON REPAIR (optional)
# # ─────────────────────────────────────────────────────────────────────────────
# _JSON_REPAIR_AVAILABLE = False
# try:
#     from json_repair import repair_json as _repair_json
#     _JSON_REPAIR_AVAILABLE = True
#     logger.info("[schema] json_repair available")
# except ImportError:
#     logger.warning("[schema] json_repair not installed — pip install json-repair")


# def safe_parse_json(raw: str) -> list[dict] | None:
#     """
#     Attempt to parse a potentially malformed JSON string from an LLM.
#     Tries in order:
#       1. Direct json.loads
#       2. Strip markdown fences + retry
#       3. json_repair (if available)
#       4. Regex extraction of JSON arrays
#     Returns list[dict] or None if all attempts fail.
#     """
#     if not raw or not raw.strip():
#         return None

#     # Strip Gemma 4 / thinking-model artifacts
#     raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

#     # Attempt 1: direct
#     try:
#         data = json.loads(raw)
#         return data if isinstance(data, list) else [data] if isinstance(data, dict) else None
#     except json.JSONDecodeError:
#         pass

#     # Attempt 2: strip fences
#     cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
#     cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
#     try:
#         data = json.loads(cleaned)
#         return data if isinstance(data, list) else [data] if isinstance(data, dict) else None
#     except json.JSONDecodeError:
#         pass

#     # Attempt 3: json_repair
#     if _JSON_REPAIR_AVAILABLE:
#         try:
#             repaired = _repair_json(cleaned, ensure_ascii=False)
#             data = json.loads(repaired)
#             logger.info("[schema] json_repair fixed malformed JSON")
#             return data if isinstance(data, list) else [data] if isinstance(data, dict) else None
#         except Exception as exc:
#             logger.debug("[schema] json_repair failed: %s", exc)

#     # Attempt 4: regex extract JSON array
#     m = re.search(r"\[.*\]", cleaned, re.DOTALL)
#     if m:
#         try:
#             data = json.loads(m.group(0))
#             return data if isinstance(data, list) else None
#         except json.JSONDecodeError:
#             pass

#     # Attempt 5: extract individual objects
#     objects: list[dict] = []
#     for obj_m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", cleaned, re.DOTALL):
#         try:
#             obj = json.loads(obj_m.group(0))
#             if isinstance(obj, dict):
#                 objects.append(obj)
#         except json.JSONDecodeError:
#             pass
#     if objects:
#         logger.info("[schema] Extracted %d objects via regex fallback", len(objects))
#         return objects

#     logger.warning("[schema] All JSON parse attempts failed for %d-char string", len(raw))
#     return None


# # ─────────────────────────────────────────────────────────────────────────────
# #  DATE NORMALISATION
# # ─────────────────────────────────────────────────────────────────────────────
# _MONTH_MAP = {
#     "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
#     "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
#     "january":1,"february":2,"march":3,"april":4,"june":6,
#     "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
# }


# def normalise_date(raw: str | None) -> str:
#     """
#     Normalise various date formats to ISO 8601 or YYYY/YYYY for ranges.
#     Returns empty string if unparseable.
#     """
#     if not raw:
#         return ""
#     raw = str(raw).strip()

#     # Already ISO
#     if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
#         return raw

#     # Range: 2022-2023 or 2022/2023
#     if re.match(r"^\d{4}[-/]\d{4}$", raw):
#         return raw.replace("-", "/")

#     # Year only
#     if re.match(r"^\d{4}$", raw):
#         return raw

#     # Month Year: "March 2022", "Mar 2022"
#     m = re.match(r"^(\w+)\s+(\d{4})$", raw)
#     if m:
#         month_name = m.group(1).lower()
#         year       = m.group(2)
#         month      = _MONTH_MAP.get(month_name)
#         if month:
#             return f"{year}-{month:02d}"

#     # DD Month YYYY or DD/MM/YYYY
#     m = re.match(r"^(\d{1,2})[-/\s](\w+)[-/\s](\d{4})$", raw)
#     if m:
#         day   = m.group(1)
#         month_raw = m.group(2)
#         year  = m.group(3)
#         if month_raw.isdigit():
#             return f"{year}-{int(month_raw):02d}-{int(day):02d}"
#         month = _MONTH_MAP.get(month_raw.lower())
#         if month:
#             return f"{year}-{month:02d}-{int(day):02d}"

#     # Return as-is if we can't normalise (don't lose data)
#     return raw


# def normalise_depth(raw: str | None) -> str:
#     """Clean depth strings: '10-15m', '10m', '>20', 'subtidal'."""
#     if not raw:
#         return ""
#     s = str(raw).strip().lower()
#     s = re.sub(r"[m\s]+$", "", s)       # strip trailing 'm' or spaces
#     s = re.sub(r"\s*(meters?|metres?)", "", s, flags=re.IGNORECASE)
#     return s.strip()


# # ─────────────────────────────────────────────────────────────────────────────
# #  SAMPLING EVENT SUB-MODEL
# # ─────────────────────────────────────────────────────────────────────────────
# class SamplingEvent(BaseModel):
#     model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

#     date:       str = Field(default="", description="ISO 8601 date or YYYY/YYYY range")
#     depth_m:    str = Field(default="", description="Depth in metres (numeric or range)")
#     method:     str = Field(default="", description="Sampling method")
#     tide_stage: str = Field(default="", description="Tide stage")

#     @field_validator("date", mode="before")
#     @classmethod
#     def norm_date(cls, v):
#         return normalise_date(v)

#     @field_validator("depth_m", mode="before")
#     @classmethod
#     def norm_depth(cls, v):
#         return normalise_depth(v)

#     @classmethod
#     def from_any(cls, v: Any) -> "SamplingEvent":
#         """Accept dict, string, or None."""
#         if v is None:
#             return cls()
#         if isinstance(v, str):
#             # Try JSON first
#             try:
#                 d = json.loads(v)
#                 if isinstance(d, dict):
#                     return cls.model_validate(d)
#             except Exception:
#                 pass
#             # Parse flat string: "2022-03-15, 10m, seine net"
#             parts = [p.strip() for p in re.split(r"[,;]", v)]
#             d = {}
#             for p in parts:
#                 if re.match(r"\d{4}", p):
#                     d["date"] = p
#                 elif "m" in p.lower() and re.search(r"\d", p):
#                     d["depth_m"] = p
#                 elif p:
#                     d.setdefault("method", p)
#             return cls.model_validate(d)
#         if isinstance(v, dict):
#             return cls.model_validate(v)
#         return cls()


# # ─────────────────────────────────────────────────────────────────────────────
# #  HIGHER TAXONOMY SUB-MODEL
# # ─────────────────────────────────────────────────────────────────────────────
# class HigherTaxonomy(BaseModel):
#     model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

#     domain:  str = ""
#     kingdom: str = ""
#     phylum:  str = ""
#     class_:  str = Field(default="", alias="class")
#     order_:  str = Field(default="", alias="order")
#     family:  str = ""

#     model_config = ConfigDict(extra="ignore", str_strip_whitespace=True, populate_by_name=True)


# # ─────────────────────────────────────────────────────────────────────────────
# #  OCCURRENCE RECORD — MAIN MODEL
# # ─────────────────────────────────────────────────────────────────────────────
# OccurrenceType = Literal["Primary", "Secondary", "Uncertain"]

# class OccurrenceRecord(BaseModel):
#     """
#     Pydantic v2 model for a single species occurrence record.
#     Accepts the LLM's raw field names (with spaces / capitalization)
#     and normalises to Darwin Core–compatible names.

#     Key aliases accepted:
#       "Recorded Name"     → recordedName
#       "Valid Name"        → validName
#       "Higher Taxonomy"   → higherTaxonomy (dict)
#       "Source Citation"   → sourceCitation
#       "Habitat"           → habitat
#       "Sampling Event"    → samplingEvent (dict or string)
#       "Raw Text Evidence" → rawTextEvidence
#       "verbatimLocality"  → verbatimLocality
#       "occurrenceType"    → occurrenceType
#     """
#     model_config = ConfigDict(
#         extra="ignore",
#         str_strip_whitespace=True,
#         populate_by_name=True,
#     )

#     # ── Core taxonomic fields ─────────────────────────────────────────────────
#     recordedName:     str  = Field(default="", alias="Recorded Name")
#     validName:        str  = Field(default="", alias="Valid Name")
#     taxonRank:        str  = ""
#     taxonomicStatus:  str  = "unverified"
#     nameAccordingTo:  str  = ""
#     matchScore:       float = 0.0
#     wormsID:          str  = ""
#     itisID:           str  = ""

#     # ── Higher taxonomy ───────────────────────────────────────────────────────
#     domain:   str = ""
#     kingdom:  str = ""
#     phylum:   str = ""
#     class_:   str = Field(default="", alias="class")
#     order_:   str = Field(default="", alias="order")
#     family_:  str = Field(default="", alias="family")

#     # ── Provenance ────────────────────────────────────────────────────────────
#     sourceCitation:  str = Field(default="", alias="Source Citation")

#     # ── Occurrence metadata ───────────────────────────────────────────────────
#     occurrenceType:  OccurrenceType  = Field(default="Uncertain", alias="occurrenceType")
#     habitat:         str = Field(default="", alias="Habitat")
#     rawTextEvidence: str = Field(default="", alias="Raw Text Evidence")
#     verbatimLocality:str = ""
#     expandedLocality:str = ""
#     stateProvince:   str = ""
#     county:          str = ""
#     country:         str = "India"

#     # ── Coordinates ───────────────────────────────────────────────────────────
#     decimalLatitude:  Optional[float] = None
#     decimalLongitude: Optional[float] = None
#     geocodingSource:  str = ""

#     # ── Sampling event ────────────────────────────────────────────────────────
#     samplingDate:   str = ""
#     depthM:         str = ""
#     method:         str = ""

#     # ── Internal tracking ─────────────────────────────────────────────────────
#     detectionSource: str = "llm"
#     chunkId:         int = 0
#     section:         str = ""

#     # ── Pre-processing validator ──────────────────────────────────────────────
#     @model_validator(mode="before")
#     @classmethod
#     def preprocess(cls, values: Any) -> dict:
#         """Normalise raw dict before field validation."""
#         if not isinstance(values, dict):
#             return values

#         # Flatten "Sampling Event" (dict or string)
#         se_raw = (
#             values.pop("Sampling Event", None)
#             or values.pop("samplingEvent", None)
#             or values.pop("sampling_event", None)
#         )
#         if se_raw:
#             se = SamplingEvent.from_any(se_raw)
#             values["samplingDate"] = se.date
#             values["depthM"]       = se.depth_m
#             values["method"]       = se.method

#         # Flatten "Higher Taxonomy" dict
#         ht_raw = (
#             values.pop("Higher Taxonomy", None)
#             or values.pop("higherTaxonomy", None)
#             or values.pop("higher_taxonomy", None)
#         )
#         if isinstance(ht_raw, dict):
#             for dk in ("domain","kingdom","phylum","class","order","family"):
#                 if not values.get(dk) and ht_raw.get(dk):
#                     values[dk] = ht_raw[dk]
#             # Handle aliased keys
#             for alias_pair in [("class_","class"),("order_","order"),("family_","family")]:
#                 src, dst = alias_pair
#                 if ht_raw.get(dst) and not values.get(src) and not values.get(dst):
#                     values[dst] = ht_raw[dst]
#         elif isinstance(ht_raw, str) and ht_raw.strip():
#             # Sometimes LLM returns a single string like "Phylum: Chordata"
#             for line in ht_raw.split("\n"):
#                 m = re.match(r"(\w+)\s*:\s*(.+)", line)
#                 if m:
#                     key = m.group(1).lower().rstrip("_")
#                     val = m.group(2).strip()
#                     if key in ("domain","kingdom","phylum","class","order","family"):
#                         values[key] = val

#         # Normalise occurrenceType variants
#         occ_raw = str(
#             values.get("occurrenceType")
#             or values.get("occurrence_type")
#             or values.get("Flag")
#             or "Uncertain"
#         ).strip().title()
#         if occ_raw.startswith("P"):
#             values["occurrenceType"] = "Primary"
#         elif occ_raw.startswith("S"):
#             values["occurrenceType"] = "Secondary"
#         else:
#             values["occurrenceType"] = "Uncertain"

#         # Alias: "Recorded Name" → recordedName (also set scientificName)
#         if not values.get("recordedName") and not values.get("Recorded Name"):
#             # Try common LLM key variants
#             for k in ("species","species_name","scientificName","name","taxon"):
#                 if values.get(k):
#                     values["Recorded Name"] = values[k]
#                     break

#         # Locality aliases
#         for loc_key in ("locality","location","site","Locality","Location","Site"):
#             if values.get(loc_key) and not values.get("verbatimLocality"):
#                 values["verbatimLocality"] = values[loc_key]
#                 break

#         # Trim overlong fields
#         for field in ("rawTextEvidence","Raw Text Evidence"):
#             if len(str(values.get(field,"") or "")) > 1000:
#                 values[field] = str(values[field])[:1000]

#         return values

#     # ── Field validators ──────────────────────────────────────────────────────
#     @field_validator("decimalLatitude", "decimalLongitude", mode="before")
#     @classmethod
#     def coerce_coord(cls, v):
#         if v is None or str(v).strip() in ("","null","None","N/A","Not Reported"):
#             return None
#         try:
#             # "22.5 N" → 22.5; "-7.3 S" → -7.3 (South negates)
#             s = str(v).strip()
#             m = re.match(r"(-?\d+\.?\d*)\s*([NSEWnsew]?)", s)
#             if m:
#                 val = float(m.group(1))
#                 hemi = m.group(2).upper()
#                 if hemi in ("S","W"):
#                     val = -abs(val)
#                 return val
#             return float(s)
#         except (ValueError, TypeError):
#             return None

#     @field_validator("samplingDate", mode="before")
#     @classmethod
#     def norm_date_field(cls, v):
#         return normalise_date(v)

#     @field_validator("depthM", mode="before")
#     @classmethod
#     def norm_depth_field(cls, v):
#         return normalise_depth(v)

#     @field_validator("matchScore", mode="before")
#     @classmethod
#     def coerce_score(cls, v):
#         try:
#             return float(v or 0)
#         except (ValueError, TypeError):
#             return 0.0

#     # ── Computed properties ───────────────────────────────────────────────────
#     @property
#     def display_name(self) -> str:
#         return self.validName or self.recordedName or "Unknown"

#     @property
#     def has_coords(self) -> bool:
#         return self.decimalLatitude is not None and self.decimalLongitude is not None

#     @property
#     def full_taxonomy_str(self) -> str:
#         parts = [
#             self.domain, self.kingdom, self.phylum,
#             self.class_, self.order_, self.family_,
#         ]
#         return " > ".join(p for p in parts if p)

#     def to_dict(self) -> dict:
#         """Return Darwin Core–compatible flat dict."""
#         d = self.model_dump(by_alias=False)
#         # Rename class_ / order_ / family_ to Darwin Core names
#         d["class"]  = d.pop("class_",  "")
#         d["order"]  = d.pop("order_",  "")
#         d["family"] = d.pop("family_", "")
#         d["Sampling Event"] = {
#             "date":    self.samplingDate,
#             "depth_m": self.depthM,
#             "method":  self.method,
#         }
#         d["Higher Taxonomy"] = {
#             "domain":  self.domain,
#             "kingdom": self.kingdom,
#             "phylum":  self.phylum,
#             "class":   self.class_,
#             "order":   self.order_,
#             "family":  self.family_,
#         }
#         return d


# # ─────────────────────────────────────────────────────────────────────────────
# #  BATCH PARSE AND VALIDATE
# # ─────────────────────────────────────────────────────────────────────────────
# def parse_llm_response(
#     raw: str,
#     source_citation: str = "",
#     chunk_id: int        = 0,
#     section: str         = "",
# ) -> tuple[list[OccurrenceRecord], list[str]]:
#     """
#     Parse and validate a raw LLM response string.

#     Returns:
#         (validated_records, error_messages)

#     Error messages are human-readable strings suitable for the UI log.
#     """
#     errors: list[str] = []

#     data = safe_parse_json(raw)
#     if data is None:
#         errors.append(f"[Schema] Chunk {chunk_id}: JSON parse failed entirely")
#         return [], errors

#     records: list[OccurrenceRecord] = []
#     for i, item in enumerate(data):
#         if not isinstance(item, dict):
#             errors.append(f"[Schema] Chunk {chunk_id} item {i}: not a dict (got {type(item).__name__})")
#             continue

#         # Inject provenance
#         if source_citation and not item.get("Source Citation") and not item.get("sourceCitation"):
#             item["Source Citation"] = source_citation
#         if chunk_id:
#             item["chunkId"] = chunk_id
#         if section:
#             item["section"] = section

#         try:
#             rec = OccurrenceRecord.model_validate(item)
#             # Must have at least a name
#             if not rec.recordedName.strip() and not (item.get("Recorded Name","") or "").strip():
#                 errors.append(f"[Schema] Chunk {chunk_id} item {i}: no species name — skipped")
#                 continue
#             records.append(rec)
#         except Exception as exc:
#             errors.append(f"[Schema] Chunk {chunk_id} item {i}: validation error — {exc}")

#     return records, errors


# def validate_batch(
#     raw_dicts: list[dict],
#     source_citation: str = "",
# ) -> tuple[list[OccurrenceRecord], list[str]]:
#     """Validate a pre-parsed list of dicts."""
#     records: list[OccurrenceRecord] = []
#     errors:  list[str] = []

#     for i, item in enumerate(raw_dicts):
#         if not isinstance(item, dict):
#             errors.append(f"Item {i}: not a dict")
#             continue
#         if source_citation and not item.get("sourceCitation") and not item.get("Source Citation"):
#             item["Source Citation"] = source_citation
#         try:
#             rec = OccurrenceRecord.model_validate(item)
#             records.append(rec)
#         except Exception as exc:
#             errors.append(f"Item {i}: {exc}")

#     return records, errors


# def records_to_dicts(records: list[OccurrenceRecord]) -> list[dict]:
#     """Convert validated records back to flat dicts for downstream use."""
#     return [r.to_dict() for r in records]


# # ─────────────────────────────────────────────────────────────────────────────
# #  LLM SCHEMA PROMPT FRAGMENT
# # ─────────────────────────────────────────────────────────────────────────────
# SCHEMA_JSON_EXAMPLE = """{
#   "Recorded Name": "Acanthurus triostegus",
#   "Valid Name": "",
#   "Higher Taxonomy": {
#     "domain":  "Eukaryota",
#     "kingdom": "Animalia",
#     "phylum":  "Chordata",
#     "class":   "Actinopterygii",
#     "order":   "Acanthuriformes",
#     "family":  "Acanthuridae"
#   },
#   "Source Citation": "Author, Title, Year",
#   "Habitat": "Coral reef, intertidal",
#   "Sampling Event": {
#     "date":    "2022-03-15",
#     "depth_m": "0-5",
#     "method":  "belt transect"
#   },
#   "Raw Text Evidence": "Verbatim sentence from paper",
#   "verbatimLocality": "Narara Island",
#   "occurrenceType": "Primary"
# }"""

# SCHEMA_VALIDATION_RULES = """
# STRICT OUTPUT RULES:
# 1. Return ONLY a valid JSON array — no markdown fences, no commentary.
# 2. occurrenceType must be exactly: "Primary", "Secondary", or "Uncertain"
# 3. "Sampling Event".date → ISO 8601 (YYYY-MM-DD) or YYYY/YYYY for year ranges
# 4. "Sampling Event".depth_m → numeric string only ("10", "10-15"), no units
# 5. "Recorded Name" → verbatim binomial as in text (never invent)
# 6. "verbatimLocality" → exact place name; resolve station IDs if possible
# 7. Missing data → empty string "", never null or "Not Reported"
# 8. One object per unique Species × Locality × Date event
# """



"""
biotrace_schemas.py — BioTrace v5.4
Pydantic models for occurrence records and relation triples.
Inspired by Hyper-Extract's typed AutoModel extraction pattern.
"""
from __future__ import annotations
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional
import re

class SpatioTemporalBbox(BaseModel):
    """Spatial bounding box + temporal range for a species at a locality."""
    lat_min: Optional[float] = 4.79
    lat_max: Optional[float] = 37.1
    lon_min: Optional[float] = 65.64
    lon_max: Optional[float] = 97.4
    date_start: Optional[str] = 1800  # YYYY or YYYY-MM-DD
    date_end: Optional[str] = 2026

class OccurrenceRecord(BaseModel):
    recorded_name: str
    valid_name: str = ""
    higher_taxonomy: str = ""
    source_citation: str = ""
    habitat: str = "Not Reported"
    sampling_event: dict = {}
    raw_text_evidence: str = ""
    verbatim_locality: str = "Not Reported"
    occurrence_type: str = "Uncertain"
    bbox: SpatioTemporalBbox = SpatioTemporalBbox()
    col_id: Optional[str] = None
    worms_id: Optional[str] = None

    @field_validator("occurrence_type")
    @classmethod
    def validate_type(cls, v):
        allowed = {"Primary", "Secondary", "Uncertain"}
        return v if v in allowed else "Uncertain"

    @field_validator("verbatim_locality")
    @classmethod
    def not_blank(cls, v):
        return v.strip() or "Not Reported"

    def to_dict(self) -> dict:
        """Backward-compatible dict for existing insert_occurrences() pipeline."""
        return {
            "recordedName": self.recorded_name,
            "validName": self.valid_name,
            "sourceCitation": self.source_citation,
            "Habitat": self.habitat,
            "Sampling Event": self.sampling_event,
            "Raw Text Evidence": self.raw_text_evidence,
            "verbatimLocality": self.verbatim_locality,
            "occurrenceType": self.occurrence_type,
            "_bbox": self.bbox.model_dump(),
        }

class RelationTriple(BaseModel):
    """
    A cross-sentence relation triple — DeepKE Document RE format.
    Maps to species_relations SQLite table.
    """
    subject: str           # species name (head entity)
    relation: str          # FOUND_AT | CO_OCCURS_WITH | INHABITS | FEEDS_ON | PARASITE_OF | OBSERVED_AT_DEPTH
    object: str            # locality / habitat / depth / co-occurring species
    evidence_text: str     # verbatim supporting sentence(s)
    source_citation: str = ""
    confidence: float = 1.0