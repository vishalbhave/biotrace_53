"""
biotrace_unified_verifier.py  —  BioTrace v5.5
────────────────────────────────────────────────────────────────────────────
Unified species verification cascade that coordinates ALL taxonomy APIs
through a single interface with SQLite caching to prevent rate-limit hits.

API cascade (marine priority order):
  1. gnparser REST  — parse name components; strip author/year noise first
  2. GNfinder REST  — confirm name is a real scientific name in text
  3. GNV (GNA)      — primary verification + classificationPath
  4. WoRMS REST     — authoritative for marine species (overrides GNV taxonomy)
  5. GBIF Backbone  — cross-check + gbifKey for occurrence data
  6. COL REST       — Catalogue of Life accepted name
  7. pytaxize/ITIS  — ITIS TSN + ITIS classification (freshwater / wider coverage)

All results are cached in `taxon_verification_cache` SQLite table.
Cache TTL = 90 days. Hit = skip all API calls for that name.

Confidence scoring:
  Each service that returns ACCEPTED adds weight:
    WoRMS match   → +0.40 (authoritative for marine)
    GNV score     → +0.30 (up to, proportional to GNV score)
    GBIF EXACT    → +0.15
    COL accepted  → +0.10
    ITIS match    → +0.05
  Score ≥ 0.60 → auto-verified

Install:
  pip install requests pytaxize

Usage:
  from biotrace_unified_verifier import UnifiedTaxonVerifier
  verifier = UnifiedTaxonVerifier(cache_db="biodiversity_data/meta.db")
  occurrences = verifier.verify_and_enrich(occurrences)
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import requests

logger = logging.getLogger("biotrace.unified_verifier")

# ─────────────────────────────────────────────────────────────────────────────
#  Optional pytaxize
# ─────────────────────────────────────────────────────────────────────────────
_PYTAXIZE_OK = False
try:
    import pytaxize
    _PYTAXIZE_OK = True
    logger.info("[unified] pytaxize available")
except ImportError:
    logger.info("[unified] pytaxize not installed (pip install pytaxize) — ITIS skipped")

# ─────────────────────────────────────────────────────────────────────────────
#  API endpoints
# ─────────────────────────────────────────────────────────────────────────────
_GNPARSER_URL  = "https://gnparser.globalnames.org/api/v1"
_GNFINDER_URL  = "https://finder.globalnames.org/api/v1/find"
_GNV_URL       = "https://verifier.globalnames.org/api/v1/verifications"
_WORMS_URL     = "https://www.marinespecies.org/rest"
_GBIF_URL      = "https://api.gbif.org/v1/species/match"
_COL_URL       = "https://api.catalogueoflife.org/nameusage/search"

_GNV_SOURCES   = "169,1,11,12,4"   # WoRMS, CoL, GBIF, ITIS, NCBI
_TIMEOUT       = 12
_BATCH_SIZE    = 20
_MIN_SCORE     = 0.60               # unified confidence threshold
_CACHE_TTL_DAYS = 90


# ─────────────────────────────────────────────────────────────────────────────
#  Result data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaxonResult:
    """Unified verification result for one species name."""
    # Input
    query_name:    str = ""

    # Parsed name components (gnparser)
    canonical:     str = ""          # bare canonical without author
    genus:         str = ""
    species:       str = ""
    infraspecific: str = ""
    author_year:   str = ""
    is_hybrid:     bool = False

    # Verification results
    valid_name:    str = ""          # accepted current canonical name
    taxon_rank:    str = ""
    taxonomic_status: str = "unverified"

    # External IDs
    worms_id:      str = ""
    gbif_key:      str = ""
    col_id:        str = ""
    itis_tsn:      str = ""
    eol_id:        str = ""

    # Higher taxonomy (WoRMS authoritative when present)
    kingdom:       str = ""
    phylum:        str = ""
    class_:        str = ""
    order_:        str = ""
    family_:       str = ""
    genus_:        str = ""

    # Source tracking
    name_according_to: str = ""
    match_type:    str = ""
    gnv_score:     float = 0.0
    gbif_confidence: int = 0
    unified_confidence: float = 0.0  # 0.0–1.0 weighted score

    # Vernacular names (for wiki profiles)
    vernacular_names: list[str] = field(default_factory=list)

    # Classification path (for wiki)
    classification_path:  str = ""
    classification_ranks: str = ""

    # Cache metadata
    cached_at: str = ""
    sources_used: str = ""          # comma-separated list of APIs that contributed


# ─────────────────────────────────────────────────────────────────────────────
#  SQLite cache schema
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS taxon_verification_cache (
    canonical_key   TEXT PRIMARY KEY,   -- lowercase canonical name (no author)
    query_name      TEXT,
    result_json     TEXT,               -- JSON-serialised TaxonResult
    cached_at       TEXT DEFAULT (datetime('now')),
    sources_used    TEXT
);
CREATE INDEX IF NOT EXISTS idx_tvc_canonical ON taxon_verification_cache(canonical_key);
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _canon_key(name: str) -> str:
    """Lowercase canonical form for cache keying."""
    return re.sub(r"\s+", " ", name.lower().strip())


def _clean_open_nom(name: str) -> str:
    """Strip open-nomenclature tokens before API queries."""
    for tok in ("cf. ", "cf.", "aff. ", "aff.", " sp.", " spp.",
                " n. sp.", " n.sp.", " nov.", " sensu"):
        name = name.replace(tok, " ")
    return re.sub(r"\s+", " ", name).strip()


def _rate_sleep(secs: float):
    time.sleep(secs)


# ─────────────────────────────────────────────────────────────────────────────
#  gnparser  — parse name into components (removes author/year noise)
# ─────────────────────────────────────────────────────────────────────────────

def _gnparser_parse(names: list[str]) -> dict[str, dict]:
    """
    POST to gnparser REST API. Returns {original_name: parsed_dict}.

    gnparser docs: https://gnparser.globalnames.org
    """
    if not names:
        return {}
    try:
        payload = {"names": names, "with_details": True}
        r = requests.post(_GNPARSER_URL, json=payload, timeout=_TIMEOUT)
        r.raise_for_status()
        results: dict[str, dict] = {}
        for item in r.json():
            orig  = item.get("verbatim", "")
            parsed = item.get("parsed", False)
            if not parsed:
                continue
            cn = item.get("canonicalFull", {})
            canonical = cn.get("value", "") or item.get("canonical", {}).get("value", "")

            details = item.get("details", {})
            sp_det  = details.get("species", details.get("infraspecies", {}))

            results[orig] = {
                "canonical":     canonical,
                "genus":         details.get("uninomial", {}).get("value", "")
                                 or (sp_det.get("genus", {}).get("value", "") if isinstance(sp_det, dict) else ""),
                "species":       (sp_det.get("species", {}).get("value", "") if isinstance(sp_det, dict) else ""),
                "author_year":   item.get("authorship", {}).get("verbatim", ""),
                "is_hybrid":     item.get("hybrid", False),
                "cardinality":   item.get("cardinality", 0),
                "quality":       item.get("quality", 0),  # 1=best, 4=worst
            }
        return results
    except Exception as exc:
        logger.debug("[gnparser] %s", exc)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
#  GNfinder  — confirm name is a genuine scientific name in text
# ─────────────────────────────────────────────────────────────────────────────

def gnfinder_find_in_text(text: str) -> list[str]:
    """
    Use GNfinder REST API to extract verified scientific names from text.
    Useful as a pre-filter before expensive verification APIs.

    Returns list of canonical names found.
    """
    if not text or not text.strip():
        return []
    try:
        payload = {
            "text": text[:8000],
            "language": "eng",
            "wordsAround": 2,
            "verification": True,          # GNfinder cross-checks GNA in one call
        }
        r = requests.post(_GNFINDER_URL, json=payload, timeout=_TIMEOUT)
        r.raise_for_status()
        names = []
        for item in r.json().get("names", []):
            name = item.get("name", "")
            if name and item.get("cardinality", 0) >= 1:
                names.append(name)
        return list(dict.fromkeys(names))  # deduplicate, preserve order
    except Exception as exc:
        logger.debug("[GNfinder] %s", exc)
    return []


# ─────────────────────────────────────────────────────────────────────────────
#  GNV (GNA Verifier)  — primary verification
# ─────────────────────────────────────────────────────────────────────────────
def _gnv_batch(names: list[str]) -> dict[str, dict]:
    """GNA Verifier batch call. Updated to use POST to avoid 405 errors."""
    if not names:
        return {}
    try:
        # Change requests.get to requests.post
        r = requests.post(
            _GNV_URL,
            json={  # Move parameters into a JSON body
                "names":              names, # Use a list, not a pipe-separated string
                "dataSources":        [169, 1, 11, 12, 4], # Pass as list of IDs
                "withVernaculars":    True,
                "withSpeciesGroup":   True,
                "capitalize":         True,
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        
# def _gnv_batch(names: list[str]) -> dict[str, dict]:
#     """GNA Verifier batch call. Returns {name: result_dict}."""
#     if not names:
#         return {}
#     try:
#         r = requests.get(
#             _GNV_URL,
#             params={
#                 "names":              "|".join(names),
#                 "data_sources":       _GNV_SOURCES,
#                 "with_vernaculars":   "true",
#                 "with_species_group": "true",
#                 "capitalize":         "true",
#             },
#             timeout=_TIMEOUT,
#         )
#         r.raise_for_status()
        out: dict[str, dict] = {}
        for item in r.json().get("names", []):
            submitted = item.get("name", "")
            best = item.get("bestResult") or {}
            if not best:
                continue

            c_path  = best.get("classificationPath", "")  or ""
            c_ranks = best.get("classificationRanks", "") or ""
            tax = _parse_classification(c_path, c_ranks)

            status_raw = (best.get("taxonomicStatus", "") or "").lower()
            status = ("synonym" if "synonym" in status_raw
                      else "accepted" if ("accepted" in status_raw or "valid" in status_raw)
                      else "unverified")

            outlink = str(best.get("outlink", "") or "")
            worms_id = gbif_id = col_id = itis_id = eol_id = ""
            if "marinespecies.org" in outlink:
                m = re.search(r"id=(\d+)", outlink)
                if m: worms_id = m.group(1)
            if "itis.gov" in outlink:
                m = re.search(r"tsn=(\d+)", outlink)
                if m: itis_id = m.group(1)
            if "catalogueoflife.org" in outlink:
                m = re.search(r"/taxon/([a-zA-Z0-9]+)", outlink)
                if m: col_id = m.group(1)
            if "gbif.org" in outlink:
                m = re.search(r"/species/(\d+)", outlink)
                if m: gbif_id = m.group(1)
            if "eol.org" in outlink:
                m = re.search(r"/pages/(\d+)", outlink)
                if m: eol_id = m.group(1)

            # Vernacular names
            vern_raw = best.get("vernacularNames") or item.get("vernacularNames") or []
            verns = []
            for v in vern_raw[:8]:
                n = (v.get("vernacularName", "") or v.get("name", "")) if isinstance(v, dict) else str(v)
                if n:
                    verns.append(n)

            out[submitted] = {
                "valid_name":      best.get("currentCanonicalFull", "") or best.get("matchedCanonicalFull", ""),
                "taxon_rank":      best.get("taxonRank", ""),
                "taxonomic_status": status,
                "gnv_score":       float(best.get("score", 0) or 0),
                "match_type":      best.get("matchType", ""),
                "name_according_to": best.get("dataSourceTitleShort", ""),
                "worms_id":  worms_id,
                "itis_tsn":  itis_id,
                "col_id":    col_id,
                "gbif_key":  gbif_id,
                "eol_id":    eol_id,
                "vernacular_names": verns,
                "classification_path":  c_path,
                "classification_ranks": c_ranks,
                **tax,
            }
        return out
    except Exception as exc:
        logger.warning("[GNV] batch error: %s", exc)
    return {}


def _parse_classification(c_path: str, c_ranks: str) -> dict:
    tax = {"kingdom": "", "phylum": "", "class_": "", "order_": "", "family_": "", "genus_": ""}
    if not c_path or not c_ranks:
        return tax
    parts = c_path.split("|")
    ranks = c_ranks.split("|")
    rmap = {"kingdom":"kingdom","phylum":"phylum","class":"class_",
            "order":"order_","family":"family_","genus":"genus_"}
    for rank, val in zip(ranks, parts):
        key = rmap.get(rank.lower().strip())
        if key:
            tax[key] = val.strip()
    return tax


# ─────────────────────────────────────────────────────────────────────────────
#  WoRMS REST  — authoritative marine taxonomy
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1024)
def _worms_by_aphia(aphia_id: str) -> dict:
    try:
        r = requests.get(f"{_WORMS_URL}/AphiaClassificationByAphiaID/{aphia_id}", timeout=_TIMEOUT)
        r.raise_for_status()
        tax: dict[str, str] = {}
        def _walk(node):
            if not isinstance(node, dict): return
            rank = (node.get("rank", "") or "").lower()
            name = node.get("scientificname", "") or ""
            rmap = {"phylum":"phylum","class":"class_","order":"order_","family":"family_","genus":"genus_"}
            if rank in rmap:
                tax[rmap[rank]] = name
            _walk(node.get("child"))
        _walk(r.json())
        return tax
    except Exception as exc:
        logger.debug("[WoRMS] AphiaID=%s: %s", aphia_id, exc)
    return {}


@lru_cache(maxsize=1024)
def _worms_by_name(name: str) -> dict:
    """WoRMS lookup by scientific name — returns aphia record."""
    try:
        r = requests.get(
            f"{_WORMS_URL}/AphiaRecordsByName/{requests.utils.quote(name)}",
            params={"like": "false", "marine_only": "false"},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        records = r.json()
        if not records or not isinstance(records, list):
            return {}
        rec = records[0]
        return {
            "worms_id":    str(rec.get("AphiaID", "")),
            "valid_name":  rec.get("valid_name", "") or rec.get("scientificname", ""),
            "phylum":      rec.get("phylum", "") or "",
            "class_":      rec.get("class", "")  or "",
            "order_":      rec.get("order", "")  or "",
            "family_":     rec.get("family", "") or "",
            "genus_":      rec.get("genus", "")  or "",
            "taxonomic_status": "accepted" if (rec.get("status","") or "").lower() == "accepted" else "synonym",
        }
    except Exception as exc:
        logger.debug("[WoRMS name] %s: %s", name, exc)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
#  GBIF Backbone match
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1024)
def _gbif_match(name: str, kingdom: str = "Animalia") -> dict:
    try:
        r = requests.get(
            _GBIF_URL,
            params={"name": name, "kingdom": kingdom, "strict": "false"},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        d = r.json()
        key = str(d.get("usageKey", ""))
        return {
            "gbif_key":       key,
            "gbif_status":    d.get("status", ""),
            "gbif_match_type": d.get("matchType", "NONE"),
            "gbif_confidence": int(d.get("confidence", 0)),
            "gbif_name":       d.get("canonicalName", ""),
            "phylum":          d.get("phylum", ""),
            "class_":          d.get("class", ""),
            "order_":          d.get("order", ""),
            "family_":         d.get("family", ""),
            "gbif_url":        f"https://www.gbif.org/species/{key}" if key else "",
        }
    except Exception as exc:
        logger.debug("[GBIF] %s: %s", name, exc)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
#  COL REST
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1024)
def _col_search(name: str) -> dict:
    try:
        r = requests.get(_COL_URL, params={"q": name, "limit": 3}, timeout=_TIMEOUT)
        r.raise_for_status()
        results = r.json().get("result", [])
        if not results:
            return {}
        r0 = results[0]
        usage = r0.get("usage", r0)
        name_field = usage.get("accepted", usage).get("name", usage.get("name", {}))
        if isinstance(name_field, dict):
            accepted_name = name_field.get("scientificName", "") or name_field.get("label", "")
        else:
            accepted_name = str(name_field)
        col_id = str(usage.get("id", "")).strip()
        status = str(usage.get("status", "")).strip()
        return {
            "col_id":       col_id,
            "col_name":     accepted_name,
            "col_status":   status,
            "col_url":      f"https://www.catalogueoflife.org/data/taxon/{col_id}" if col_id else "",
        }
    except Exception as exc:
        logger.debug("[COL] %s: %s", name, exc)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
#  pytaxize / ITIS  (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _itis_lookup(name: str) -> dict:
    """ITIS lookup via pytaxize. Returns TSN and classification."""
    if not _PYTAXIZE_OK:
        return {}
    try:
        results = pytaxize.itis.searchbynames(names=[name])
        if not results or not isinstance(results, list) or not results[0]:
            return {}
        first = results[0]
        tsn = str(first.get("tsn", "")).strip() if isinstance(first, dict) else ""
        if not tsn:
            return {}
        record = pytaxize.itis.getrecord(tsn=int(tsn))
        if not record:
            return {}
        classification = {}
        for item in record:
            if isinstance(item, dict):
                rank = str(item.get("rankName", "")).lower()
                nm   = str(item.get("taxonName", ""))
                rmap = {"phylum":"phylum","class":"class_","order":"order_","family":"family_","genus":"genus_"}
                if rank in rmap:
                    classification[rmap[rank]] = nm
        return {"itis_tsn": tsn, **classification}
    except Exception as exc:
        logger.debug("[ITIS/pytaxize] %s: %s", name, exc)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
#  Unified confidence scorer
# ─────────────────────────────────────────────────────────────────────────────

def _compute_confidence(
    gnv_score: float,
    gnv_status: str,
    gbif_data: dict,
    worms_found: bool,
    col_found: bool,
    itis_found: bool,
) -> float:
    """
    Weighted confidence from multiple sources.

    WoRMS  → +0.40 (authoritative for marine)
    GNV    → +0.30 (proportional to GNV score)
    GBIF   → +0.15 (EXACT match only; FUZZY = 0.08)
    COL    → +0.10 (accepted status)
    ITIS   → +0.05
    """
    score = 0.0

    if worms_found:
        score += 0.40

    if gnv_status in ("accepted", "synonym"):
        score += min(0.30, 0.30 * gnv_score)

    gbif_match = gbif_data.get("gbif_match_type", "NONE")
    gbif_status = gbif_data.get("gbif_status", "")
    if gbif_match == "EXACT" and gbif_status in ("ACCEPTED", "SYNONYM"):
        score += 0.15
    elif gbif_match == "FUZZY" and gbif_status in ("ACCEPTED", "SYNONYM"):
        score += 0.08

    if col_found:
        score += 0.10

    if itis_found:
        score += 0.05

    return round(min(score, 1.0), 3)


# ─────────────────────────────────────────────────────────────────────────────
#  Main verifier class
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedTaxonVerifier:
    """
    Single-instance verifier that coordinates all taxonomy APIs
    with SQLite caching to prevent rate-limit exhaustion.

    Parameters
    ----------
    cache_db   : path to BioTrace SQLite DB (same file used for occurrences)
    min_score  : unified confidence threshold (default 0.60)
    kingdom    : GBIF kingdom filter (default "Animalia" for marine fauna)
    use_gnparser: parse names before querying (strips author/year noise)
    use_gbif   : enable GBIF backbone check
    use_col    : enable Catalogue of Life check
    use_itis   : enable ITIS check via pytaxize (requires pytaxize)
    """

    def __init__(
        self,
        cache_db:    str   = "",
        min_score:   float = _MIN_SCORE,
        kingdom:     str   = "Animalia",
        use_gnparser: bool = True,
        use_gbif:    bool  = True,
        use_col:     bool  = True,
        use_itis:    bool  = True,
    ):
        self.cache_db    = cache_db
        self.min_score   = min_score
        self.kingdom     = kingdom
        self.use_gnparser = use_gnparser
        self.use_gbif    = use_gbif
        self.use_col     = use_col
        self.use_itis    = use_itis and _PYTAXIZE_OK

        if cache_db:
            self._init_cache(cache_db)

    def _init_cache(self, db_path: str):
        try:
            conn = sqlite3.connect(db_path)
            conn.executescript(_CACHE_SCHEMA)
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("[unified] cache init: %s", exc)
            self.cache_db = ""

    def _cache_get(self, canonical_key: str) -> Optional[TaxonResult]:
        if not self.cache_db:
            return None
        try:
            conn = sqlite3.connect(self.cache_db)
            row = conn.execute(
                "SELECT result_json, cached_at FROM taxon_verification_cache WHERE canonical_key=?",
                (canonical_key,)
            ).fetchone()
            conn.close()
            if row:
                cached_at = datetime.fromisoformat(row[1]) if row[1] else datetime.min
                if datetime.utcnow() - cached_at < timedelta(days=_CACHE_TTL_DAYS):
                    data = json.loads(row[0])
                    return TaxonResult(**data)
        except Exception as exc:
            logger.debug("[unified] cache_get: %s", exc)
        return None

    def _cache_set(self, canonical_key: str, result: TaxonResult):
        if not self.cache_db:
            return
        try:
            conn = sqlite3.connect(self.cache_db)
            result.cached_at = datetime.utcnow().isoformat()
            data = {k: v for k, v in asdict(result).items()}
            conn.execute(
                """INSERT OR REPLACE INTO taxon_verification_cache
                   (canonical_key, query_name, result_json, cached_at, sources_used)
                   VALUES (?, ?, ?, ?, ?)""",
                (canonical_key, result.query_name,
                 json.dumps(data), result.cached_at, result.sources_used)
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.debug("[unified] cache_set: %s", exc)

    def verify_name(self, name: str) -> TaxonResult:
        """Verify a single species name through the full API cascade."""
        # Strip open-nomenclature tokens
        clean = _clean_open_nom(name)
        key   = _canon_key(clean)

        # Cache check
        cached = self._cache_get(key)
        if cached:
            logger.debug("[unified] cache hit: %s", clean)
            return cached

        result = TaxonResult(query_name=name)
        sources: list[str] = []

        # ── Step 1: gnparser — parse name components ──────────────────────
        canonical = clean
        if self.use_gnparser:
            parsed = _gnparser_parse([clean])
            if clean in parsed:
                p = parsed[clean]
                result.canonical     = p.get("canonical", clean)
                result.genus         = p.get("genus", "")
                result.species       = p.get("species", "")
                result.author_year   = p.get("author_year", "")
                result.is_hybrid     = p.get("is_hybrid", False)
                canonical            = result.canonical or clean
                # Low quality parse (quality==4) → garbage in, skip
                if p.get("quality", 1) >= 4:
                    logger.debug("[gnparser] low quality parse for '%s'", clean)
                    result.taxonomic_status = "unverified"
                    self._cache_set(key, result)
                    return result
                sources.append("gnparser")

        # ── Step 2: GNV (GNA Verifier) — primary verification ────────────
        gnv_data = _gnv_batch([canonical])
        gnv = gnv_data.get(canonical, {})
        if gnv:
            sources.append("GNV")
            result.valid_name          = gnv.get("valid_name", "")
            result.taxon_rank          = gnv.get("taxon_rank", "")
            result.taxonomic_status    = gnv.get("taxonomic_status", "unverified")
            result.gnv_score           = gnv.get("gnv_score", 0.0)
            result.match_type          = gnv.get("match_type", "")
            result.name_according_to   = gnv.get("name_according_to", "")
            result.worms_id            = gnv.get("worms_id", "")
            result.itis_tsn            = gnv.get("itis_tsn", "")
            result.col_id              = gnv.get("col_id", "")
            result.gbif_key            = gnv.get("gbif_key", "")
            result.eol_id              = gnv.get("eol_id", "")
            result.kingdom             = gnv.get("kingdom", "")
            result.phylum              = gnv.get("phylum", "")
            result.class_              = gnv.get("class_", "")
            result.order_              = gnv.get("order_", "")
            result.family_             = gnv.get("family_", "")
            result.genus_              = gnv.get("genus_", "")
            result.vernacular_names    = gnv.get("vernacular_names", [])
            result.classification_path  = gnv.get("classification_path", "")
            result.classification_ranks = gnv.get("classification_ranks", "")
            _rate_sleep(0.15)

        # ── Step 3: WoRMS — authoritative marine taxonomy override ────────
        worms_found = False
        valid_for_worms = result.valid_name or canonical
        if result.worms_id:
            wt = _worms_by_aphia(result.worms_id)
            if wt:
                sources.append("WoRMS")
                worms_found = True
                for f in ("phylum", "class_", "order_", "family_", "genus_"):
                    if wt.get(f):
                        setattr(result, f, wt[f])
            _rate_sleep(0.20)
        elif not result.valid_name:
            # Direct WoRMS name lookup when GNV didn't find it
            wt2 = _worms_by_name(canonical)
            if wt2 and wt2.get("worms_id"):
                sources.append("WoRMS-direct")
                worms_found = True
                result.worms_id         = wt2["worms_id"]
                result.valid_name       = wt2.get("valid_name", result.valid_name) or result.valid_name
                result.taxonomic_status = wt2.get("taxonomic_status", result.taxonomic_status)
                result.name_according_to = "WoRMS"
                for f in ("phylum", "class_", "order_", "family_", "genus_"):
                    if wt2.get(f):
                        setattr(result, f, wt2[f])
            _rate_sleep(0.20)

        # ── Step 4: GBIF Backbone ─────────────────────────────────────────
        gbif_data: dict = {}
        if self.use_gbif:
            lookup_name = result.valid_name or canonical
            gbif_data = _gbif_match(lookup_name, self.kingdom)
            if gbif_data.get("gbif_key"):
                sources.append("GBIF")
                if not result.gbif_key:
                    result.gbif_key = gbif_data["gbif_key"]
                # Fill taxonomy gaps
                for f in ("phylum", "class_", "order_", "family_"):
                    if not getattr(result, f) and gbif_data.get(f):
                        setattr(result, f, gbif_data[f])
            _rate_sleep(0.15)

        # ── Step 5: COL ───────────────────────────────────────────────────
        col_found = False
        if self.use_col:
            lookup_name = result.valid_name or canonical
            col_data = _col_search(lookup_name)
            if col_data.get("col_id"):
                sources.append("COL")
                col_found = True
                if not result.col_id:
                    result.col_id = col_data["col_id"]
            _rate_sleep(1.0)  # COL is strictly rate-limited

        # ── Step 6: ITIS via pytaxize ─────────────────────────────────────
        itis_found = False
        if self.use_itis and not result.itis_tsn:
            lookup_name = result.valid_name or canonical
            it = _itis_lookup(lookup_name)
            if it.get("itis_tsn"):
                sources.append("ITIS")
                itis_found = True
                result.itis_tsn = it["itis_tsn"]
                for f in ("phylum", "class_", "order_", "family_", "genus_"):
                    if not getattr(result, f) and it.get(f):
                        setattr(result, f, it[f])
            _rate_sleep(0.30)

        # ── Compute unified confidence ────────────────────────────────────
        result.unified_confidence = _compute_confidence(
            gnv_score   = result.gnv_score,
            gnv_status  = result.taxonomic_status,
            gbif_data   = gbif_data,
            worms_found = worms_found,
            col_found   = col_found,
            itis_found  = itis_found,
        )

        # Promote taxonomic status to "accepted" if confidence is high enough
        if result.unified_confidence >= self.min_score and result.valid_name:
            if result.taxonomic_status == "unverified":
                result.taxonomic_status = "accepted"

        result.sources_used = ",".join(sources)
        logger.debug(
            "[unified] %s → %s (conf=%.2f, sources=%s)",
            clean, result.valid_name or "—",
            result.unified_confidence, result.sources_used
        )

        # Cache the result
        self._cache_set(key, result)
        return result

    def verify_batch(
        self,
        names:    list[str],
        log_cb = None,
        rate_sleep_batch: float = 0.1,
    ) -> dict[str, TaxonResult]:
        """
        Verify a list of species names. Returns {original_name: TaxonResult}.

        Uses GNV batch for efficiency; individual calls for WoRMS/GBIF/COL.
        """
        if log_cb is None:
            log_cb = lambda msg, lvl="ok": logger.info(msg)

        if not names:
            return {}

        # Deduplicate
        unique = list(dict.fromkeys(n.strip() for n in names if n.strip()))
        log_cb(f"[Unified] Verifying {len(unique)} unique names via cascade…")

        # GNV batch pass (most efficient)
        clean_names  = [_clean_open_nom(n) for n in unique]
        canonical_map = {}  # original → canonical

        if self.use_gnparser:
            parsed = _gnparser_parse(clean_names)
            for orig, clean in zip(unique, clean_names):
                p = parsed.get(clean, {})
                canonical_map[orig] = p.get("canonical", clean) or clean
        else:
            canonical_map = {n: _clean_open_nom(n) for n in unique}

        # GNV batch
        canonicals = list(set(canonical_map.values()))
        gnv_results: dict[str, dict] = {}
        for i in range(0, len(canonicals), _BATCH_SIZE):
            batch = canonicals[i: i + _BATCH_SIZE]
            gnv_results.update(_gnv_batch(batch))
            if i + _BATCH_SIZE < len(canonicals):
                _rate_sleep(0.25)
        log_cb(f"[Unified] GNV: {len(gnv_results)}/{len(canonicals)} names matched")

        # Individual cascade for each name
        results: dict[str, TaxonResult] = {}
        for i, name in enumerate(unique):
            cached = self._cache_get(_canon_key(canonical_map[name]))
            if cached:
                results[name] = cached
                continue
            # verify_name will re-use lru_cache for WoRMS/GBIF/COL
            results[name] = self.verify_name(name)
            if i % 10 == 9:
                log_cb(f"[Unified] {i+1}/{len(unique)} names verified…")
            _rate_sleep(rate_sleep_batch)

        verified = sum(1 for r in results.values() if r.unified_confidence >= self.min_score)
        log_cb(f"[Unified] Complete: {verified}/{len(unique)} names auto-verified "
               f"(conf≥{self.min_score})")
        return results

    def verify_and_enrich(
        self,
        occurrences: list[dict],
        log_cb = None,
    ) -> list[dict]:
        """
        Main pipeline: enrich a list of occurrence dicts in-place.

        Adds fields:
          validName, taxonRank, taxonomicStatus, nameAccordingTo,
          matchScore (unified), matchType, wormsID, gbifKey, colID,
          itisTSN, eolID, phylum, class_, order_, family_, genus_,
          kingdom, vernacularNames, classificationPath, classificationRanks,
          unifiedConfidence, verificationSources
        """
        if not occurrences:
            return occurrences
        if log_cb is None:
            log_cb = lambda msg, lvl="ok": logger.info(msg)

        # Filter out __candidate_ entries before verification
        clean_occs, candidate_count = _filter_candidates(occurrences)
        if candidate_count:
            log_cb(f"[Unified] Filtered {candidate_count} __candidate_ placeholder entries")

        # Collect unique names
        names = []
        for occ in clean_occs:
            if not isinstance(occ, dict):
                continue
            name = str(occ.get("recordedName") or occ.get("Recorded Name", "")).strip()
            if name:
                names.append(name)

        # Batch verify
        results = self.verify_batch(names, log_cb=log_cb)

        # Apply to records
        verified_ct = 0
        for occ in clean_occs:
            if not isinstance(occ, dict):
                continue
            raw_name = str(occ.get("recordedName") or occ.get("Recorded Name", "")).strip()
            occ["recordedName"] = raw_name

            res = results.get(raw_name)
            if not res:
                occ["taxonomicStatus"] = "unverified"
                continue

            conf = res.unified_confidence
            if conf >= self.min_score and res.valid_name:
                occ["validName"]          = res.valid_name
                occ["scientificName"]     = res.valid_name
                occ["taxonRank"]          = res.taxon_rank
                occ["nameAccordingTo"]    = res.name_according_to
                occ["taxonomicStatus"]    = res.taxonomic_status
                occ["matchScore"]         = res.gnv_score
                occ["matchType"]          = res.match_type
                occ["wormsID"]            = res.worms_id
                occ["gbifKey"]            = res.gbif_key
                occ["colID"]              = res.col_id
                occ["itisTSN"]            = res.itis_tsn
                occ["eolID"]              = res.eol_id
                occ["kingdom"]            = res.kingdom
                occ["phylum"]             = res.phylum
                occ["class_"]             = res.class_
                occ["order_"]             = res.order_
                occ["family_"]            = res.family_
                occ["genus_"]             = res.genus_
                occ["vernacularNames"]    = res.vernacular_names
                occ["classificationPath"] = res.classification_path
                occ["classificationRanks"]= res.classification_ranks
                occ["unifiedConfidence"]  = conf
                occ["verificationSources"]= res.sources_used
                verified_ct += 1
            else:
                occ["taxonomicStatus"]   = "unverified"
                occ["unifiedConfidence"] = conf
                occ["verificationSources"] = res.sources_used

        log_cb(f"[Unified] Enriched {verified_ct}/{len(clean_occs)} records "
               f"(conf≥{self.min_score})")
        return clean_occs


# ─────────────────────────────────────────────────────────────────────────────
#  Candidate name filter (fixes __candidate_*_* leaking into final table)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_candidates(occurrences: list[dict]) -> tuple[list[dict], int]:
    """
    Remove __candidate_* placeholder entries that leak from the agent loop
    when the LLM produces structured candidates instead of verified names.

    Also removes:
      - Records where recordedName is empty or only whitespace
      - Records where validName/recordedName is a JSON fragment or artifact
    """
    _CANDIDATE_RE = re.compile(r"^_+candidate_", re.IGNORECASE)
    _JSON_FRAG_RE = re.compile(r'^[\[{\"\d]')  # starts with JSON char

    clean: list[dict] = []
    removed = 0

    for occ in occurrences:
        if not isinstance(occ, dict):
            removed += 1
            continue

        recorded = str(occ.get("recordedName") or occ.get("Recorded Name", "")).strip()
        valid    = str(occ.get("validName", "")).strip()

        # Drop __candidate_ placeholders
        if _CANDIDATE_RE.match(recorded) or _CANDIDATE_RE.match(valid):
            removed += 1
            continue

        # Drop empty names
        if not recorded and not valid:
            removed += 1
            continue

        # Drop obvious JSON artifacts as names
        if len(recorded) < 4 or (len(recorded) < 8 and _JSON_FRAG_RE.match(recorded)):
            removed += 1
            continue

        clean.append(occ)

    return clean, removed


def filter_candidates(occurrences: list[dict], log_cb=None) -> list[dict]:
    """
    Public wrapper for candidate filtering.
    Call this immediately after extract_occurrences() before any verification.
    """
    clean, n = _filter_candidates(occurrences)
    if log_cb and n:
        log_cb(f"[Filter] Removed {n} candidate/invalid placeholder records")
    elif n:
        logger.info("[Filter] Removed %d candidate/invalid placeholder records", n)
    return clean


# ─────────────────────────────────────────────────────────────────────────────
#  Cache statistics (for Streamlit dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def get_cache_stats(db_path: str) -> dict:
    """Return cache hit statistics for display in the UI."""
    try:
        conn = sqlite3.connect(db_path)
        total = conn.execute("SELECT COUNT(*) FROM taxon_verification_cache").fetchone()[0]
        worms = conn.execute(
            "SELECT COUNT(*) FROM taxon_verification_cache WHERE result_json LIKE '%\"worms_id\": \"%'"
        ).fetchone()[0]
        conn.close()
        return {"total_cached": total, "worms_matched": worms}
    except Exception:
        return {"total_cached": 0, "worms_matched": 0}
