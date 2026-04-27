"""
species_verifier.py  —  BioTrace v3.1
────────────────────────────────────────────────────────────────────────────
Species name verification and higher taxonomy retrieval.

Two-step process per the Refined Data Extraction Protocol:
  Step 1 · GNA Verifier  → resolves recordedName to valid accepted name
                          → classificationPath gives Phylum/Class/Order/Family
  Step 2 · WoRMS REST    → richer marine taxonomy when GNA matched WoRMS (ID 169)

Darwin Core fields populated:
  recordedName     — verbatim name from text (preserved, never overwritten)
  validName        — accepted / current canonical name
  taxonRank        — species | genus | family | etc.
  phylum / class_ / order_ / family_  — higher taxonomy
  nameAccordingTo  — checklist used (WoRMS | CoL | GBIF | ITIS)
  taxonomicStatus  — accepted | synonym | unverified
  matchScore       — GNA confidence 0–1
  wormsID          — WoRMS AphiaID (hyperlink in map popup)
  itisID           — ITIS TSN if matched via ITIS

Data source priority (GNA data_sources parameter):
  169 = WoRMS  (marine species — first priority for marine biology work)
  1   = Catalogue of Life
  11  = GBIF Backbone Taxonomy
  12  = ITIS
  4   = NCBI
"""
from __future__ import annotations
import logging, re, time
from functools import lru_cache
import requests

logger = logging.getLogger("biotrace.verifier")

GNA_FINDER_URL   = "https://finder.globalnames.org/api/v1/find"
GNA_VERIFIER_URL = "https://verifier.globalnames.org/api/v1/verifications"
WORMS_REST_URL   = "https://www.marinespecies.org/rest"

GNA_DATA_SOURCES = "169,1,11,12,4"   # WoRMS first
_TIMEOUT          = 15
_BATCH            = 20
_MIN_SCORE        = 0.80


# ─────────────────────────────────────────────────────────────────────────────
#  GNA Finder  — detect scientific names in free text
# ─────────────────────────────────────────────────────────────────────────────
def find_names_in_text(text: str) -> list[dict]:
    """
    Returns list of {scientificName, verbatimName, cardinality} for genus/species names.
    """
    if not text or not text.strip():
        return []
    try:
        r = requests.post(GNA_FINDER_URL,
                          json={"text": text[:8000]},
                          headers={"Content-Type":"application/json"},
                          timeout=_TIMEOUT)
        r.raise_for_status()
        hits = []
        for n in r.json().get("names", []):
            cardinality = int(n.get("cardinality", 0) or 0)
            if cardinality <= 0:
                continue
            card_label = "genus" if cardinality <= 1 else "species"
            hits.append(
                {
                    "scientificName": n.get("name", ""),
                    "verbatimName": n.get("verbatim", n.get("name", "")),
                    "cardinality": card_label,
                    "cardinalityValue": 1 if card_label == "genus" else 2,
                }
            )
        return hits
    except Exception as exc:
        logger.debug("[GNA Finder] %s",exc)
    return []


# ─────────────────────────────────────────────────────────────────────────────
#  GNA Verifier  — batch name resolution
# ─────────────────────────────────────────────────────────────────────────────
def _call_verifier(names: list[str]) -> dict[str, dict]:
    if not names: return {}
    try:
        r = requests.get(GNA_VERIFIER_URL,
                         params={"names":"|".join(names),
                                 "data_sources": GNA_DATA_SOURCES,
                                 "with_vernaculars":"false",
                                 "with_species_group":"true"},
                         timeout=_TIMEOUT)
        r.raise_for_status()
        results = {}
        for item in r.json().get("names",[]):
            submitted = item.get("name","")
            best      = item.get("bestResult") or {}
            results[submitted] = _parse_gna_result(best)
        return results
    except Exception as exc:
        logger.debug("[GNA Verifier] %s",exc)
    return {}


def _parse_gna_result(best: dict) -> dict:
    """Extract taxonomy fields from a GNA bestResult object."""
    # Parse classificationPath "Animalia|Chordata|Actinopterygii|Perciformes|Acanthuridae"
    # and classificationRanks "kingdom|phylum|class|order|family"
    tax = {"phylum":"","class_":"","order_":"","family_":""}
    c_path  = best.get("classificationPath","") or ""
    c_ranks = best.get("classificationRanks","") or ""
    if c_path and c_ranks:
        parts = c_path.split("|")
        ranks = c_ranks.split("|")
        for rank, val in zip(ranks, parts):
            rank_lower = rank.lower().strip()
            if rank_lower == "phylum":  tax["phylum"]  = val.strip()
            elif rank_lower == "class": tax["class_"]  = val.strip()
            elif rank_lower == "order": tax["order_"]  = val.strip()
            elif rank_lower == "family":tax["family_"] = val.strip()

    status_raw = (best.get("taxonomicStatus","") or "").lower()
    if   "synonym" in status_raw:          status = "synonym"
    elif "accepted" in status_raw or "valid" in status_raw: status = "accepted"
    else:                                  status = "unverified"

    # Extract WoRMS AphiaID or ITIS TSN from outlink
    outlink = str(best.get("outlink","") or "")
    worms_id = ""
    itis_id  = ""
    if "marinespecies.org" in outlink:
        m = re.search(r"id=(\d+)", outlink)
        if m: worms_id = m.group(1)
    elif "itis.gov" in outlink:
        m = re.search(r"tsn=(\d+)", outlink)
        if m: itis_id = m.group(1)

    return {
        "validName":      best.get("currentCanonicalFull", "") or best.get("matchedCanonicalFull",""),
        "matchedName":    best.get("matchedCanonicalFull",""),
        "taxonRank":      best.get("taxonRank",""),
        "nameAccordingTo": best.get("dataSourceTitleShort","") or "",
        "taxonomicStatus": status,
        "matchScore":     float(best.get("score",0) or 0),
        "matchType":      best.get("matchType",""),
        "wormsID":        worms_id,
        "itisID":         itis_id,
        **tax,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  WoRMS REST  — higher taxonomy for marine species
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=512)
def _worms_by_aphia(aphia_id: str) -> dict:
    """Fetch full WoRMS classification for a given AphiaID."""
    try:
        r = requests.get(f"{WORMS_REST_URL}/AphiaClassificationByAphiaID/{aphia_id}",
                         timeout=_TIMEOUT)
        r.raise_for_status()
        # Response is a nested {AphiaID, rank, scientificname, child: {...}}
        # Walk the chain to extract Phylum/Class/Order/Family
        taxonomy = {"phylum":"","class_":"","order_":"","family_":""}
        def _walk(node):
            if not isinstance(node,dict): return
            rank = (node.get("rank","") or "").lower()
            name = node.get("scientificname","") or ""
            if rank == "phylum":  taxonomy["phylum"]  = name
            elif rank == "class": taxonomy["class_"]  = name
            elif rank == "order": taxonomy["order_"]  = name
            elif rank == "family":taxonomy["family_"] = name
            _walk(node.get("child"))
        _walk(r.json())
        return taxonomy
    except Exception as exc:
        logger.debug("[WoRMS REST] AphiaID=%s %s",aphia_id,exc)
    return {}

@lru_cache(maxsize=512)
def _worms_by_name(name: str) -> dict:
    """Fetch WoRMS record by scientific name — returns phylum/class/order/family + AphiaID."""
    try:
        r = requests.get(f"{WORMS_REST_URL}/AphiaRecordsByName/{requests.utils.quote(name)}",
                         params={"like":"false","marine_only":"false"},
                         timeout=_TIMEOUT)
        r.raise_for_status()
        records = r.json()
        if not records or not isinstance(records,list):
            return {}
        rec = records[0]
        return {
            "phylum":   rec.get("phylum","") or "",
            "class_":   rec.get("class","")  or "",
            "order_":   rec.get("order","")  or "",
            "family_":  rec.get("family","") or "",
            "wormsID":  str(rec.get("AphiaID","")) or "",
            "validName":rec.get("valid_name","") or rec.get("scientificname",""),
            "taxonomicStatus": "accepted" if rec.get("status","").lower()=="accepted" else "synonym",
        }
    except Exception as exc:
        logger.debug("[WoRMS by name] %s %s",name,exc)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC: verify_occurrence_names
# ─────────────────────────────────────────────────────────────────────────────
def verify_occurrence_names(occurrences: list[dict]) -> list[dict]:
    """
    Enrich occurrence dicts with:
      recordedName     — original name from text (always preserved)
      validName        — accepted canonical name from GNA/WoRMS
      phylum / class_ / order_ / family_   — higher taxonomy
      taxonRank, nameAccordingTo, taxonomicStatus, matchScore
      wormsID, itisID

    For marine species (wormsID present), a second call to WoRMS REST
    fills/corrects the higher taxonomy using the official WoRMS classification.
    """
    if not occurrences: return occurrences

    # Collect unique names (from recordedName; fall back to scientificName)
    name_map: dict[str,str] = {}   # recordedName → key for de-dup
    for occ in occurrences:
        if not isinstance(occ,dict): continue
        name = str(occ.get("recordedName") or occ.get("scientificName","")).strip()
        if name: name_map.setdefault(name,name)

    unique_names = list(name_map.keys())
    if not unique_names: return occurrences
    logger.info("[verifier] Verifying %d unique names…",len(unique_names))

    # Batch GNA verification
    verification: dict[str,dict] = {}
    for i in range(0, len(unique_names), _BATCH):
        batch = unique_names[i:i+_BATCH]
        verification.update(_call_verifier(batch))
        if i+_BATCH < len(unique_names):
            time.sleep(0.3)

    # Apply to records
    verified_ct = 0
    for occ in occurrences:
        if not isinstance(occ,dict): continue
        raw_name = str(occ.get("recordedName") or occ.get("scientificName","")).strip()

        # Preserve verbatim name
        occ["recordedName"] = raw_name
        # Also keep backward-compat scientificName field
        occ.setdefault("scientificName", raw_name)

        ver = verification.get(raw_name,{})
        if not ver:
            occ["taxonomicStatus"] = "unverified"
            continue

        score      = float(ver.get("matchScore",0) or 0)
        valid_name = ver.get("validName","") or ""
        worms_id   = ver.get("wormsID","")

        if valid_name and score >= _MIN_SCORE:
            occ["validName"]       = valid_name
            occ["scientificName"]  = valid_name   # Darwin Core canonical field
            occ["taxonRank"]       = ver.get("taxonRank","")
            occ["nameAccordingTo"] = ver.get("nameAccordingTo","")
            occ["taxonomicStatus"] = ver.get("taxonomicStatus","unverified")
            occ["matchScore"]      = round(score,3)
            occ["wormsID"]         = worms_id
            occ["itisID"]          = ver.get("itisID","")

            # Higher taxonomy from GNA classificationPath
            occ["phylum"]  = ver.get("phylum","")
            occ["class_"]  = ver.get("class_","")
            occ["order_"]  = ver.get("order_","")
            occ["family_"] = ver.get("family_","")

            # For marine species, upgrade with authoritative WoRMS REST taxonomy
            if worms_id:
                wt = _worms_by_aphia(worms_id)
                if wt:
                    occ["phylum"]  = wt.get("phylum",  occ["phylum"])  or occ["phylum"]
                    occ["class_"]  = wt.get("class_",  occ["class_"])  or occ["class_"]
                    occ["order_"]  = wt.get("order_",  occ["order_"])  or occ["order_"]
                    occ["family_"] = wt.get("family_", occ["family_"]) or occ["family_"]
                    time.sleep(0.2)   # WoRMS rate politeness

            verified_ct += 1
        else:
            occ["taxonomicStatus"] = "unverified"
            occ["matchScore"]      = round(score,3)
            # Attempt direct WoRMS lookup for unverified marine names
            if not valid_name:
                wt = _worms_by_name(raw_name)
                if wt:
                    occ["validName"]       = wt.get("validName","")
                    occ["scientificName"]  = wt.get("validName","") or raw_name
                    occ["phylum"]          = wt.get("phylum","")
                    occ["class_"]          = wt.get("class_","")
                    occ["order_"]          = wt.get("order_","")
                    occ["family_"]         = wt.get("family_","")
                    occ["wormsID"]         = wt.get("wormsID","")
                    occ["taxonomicStatus"] = wt.get("taxonomicStatus","unverified")
                    occ["nameAccordingTo"] = "WoRMS"
                    time.sleep(0.2)

    logger.info("[verifier] %d/%d names resolved (score≥%.2f)",
                verified_ct, len(unique_names), _MIN_SCORE)
    return occurrences


# ── End of species_verifier.py ──
