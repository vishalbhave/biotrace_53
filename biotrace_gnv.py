# # """
# # biotrace_gnv.py  —  BioTrace v5.1
# # ────────────────────────────────────────────────────────────────────────────
# # Enhanced Global Names Verifier (GNV) integration with:
# #   • Full GNV JSON response mapping (currentName, classificationPath,
# #     classificationRanks, taxonomicStatus, outlinks, vernacularNames)
# #   • WoRMS REST authoritative override for marine species
# #   • Wiki-enriched species profiles from GNV metadata
# #   • Intra-document deduplication (species × locality × source)
# #   • Locality segregation: comma-separated locality strings split into
# #     individual geographic entries using contextual heuristics + GeoNames
# #   • JSON repair using json_repair library for LLM parse errors
# #   • Gemma 4 response extraction (handles thinking-block artifacts)

# # GNV API documentation reference:
# #   verifier.globalnames.org/api/v1/verifications
# #   Response fields captured:
# #     names[].name                       — submitted name
# #     names[].bestResult.currentCanonicalFull  — accepted valid name
# #     names[].bestResult.matchedCanonicalFull  — matched canonical
# #     names[].bestResult.classificationPath    — "Animalia|Chordata|…"
# #     names[].bestResult.classificationRanks   — "kingdom|phylum|class|…"
# #     names[].bestResult.taxonomicStatus       — "accepted" | "synonym" | …
# #     names[].bestResult.outlink               — WoRMS / ITIS / CoL URL
# #     names[].bestResult.dataSourceTitleShort  — "WoRMS" | "CoL" | "GBIF"
# #     names[].bestResult.score                 — confidence 0–1
# #     names[].bestResult.matchType             — "Exact" | "Fuzzy" | …
# #     names[].bestResult.taxonRank             — "species" | "genus" | …

# # Usage:
# #     from biotrace_gnv import GNVEnrichedVerifier, LocalitySplitter
# #     verifier = GNVEnrichedVerifier()
# #     occurrences = verifier.verify_and_enrich(occurrences)
# #     occurrences = LocalitySplitter().split_localities(occurrences)
# # """
# # from __future__ import annotations

# # import json
# # import logging
# # import re
# # import time
# # from collections import defaultdict
# # from functools import lru_cache
# # from typing import Any, Optional

# # import requests

# # logger = logging.getLogger("biotrace.gnv")

# # # ─────────────────────────────────────────────────────────────────────────────
# # #  OPTIONAL DEPS
# # # ─────────────────────────────────────────────────────────────────────────────
# # _JSON_REPAIR_AVAILABLE = False
# # try:
# #     from json_repair import repair_json as _repair_json
# #     _JSON_REPAIR_AVAILABLE = True
# #     logger.info("[gnv] json_repair available")
# # except ImportError:
# #     pass

# # _GEOPY_AVAILABLE = False
# # try:
# #     from geopy.geocoders import Nominatim as _Nominatim
# #     _GEOPY_AVAILABLE = True
# # except ImportError:
# #     pass

# # # ─────────────────────────────────────────────────────────────────────────────
# # #  CONSTANTS
# # # ─────────────────────────────────────────────────────────────────────────────
# # GNV_VERIFIER_URL = "https://verifier.globalnames.org/api/v1/verifications"
# # GNV_FINDER_URL   = "https://finder.globalnames.org/api/v1/find"
# # WORMS_REST_URL   = "https://www.marinespecies.org/rest"

# # GNV_DATA_SOURCES = "169,1,11,12,4"   # WoRMS, CoL, GBIF, ITIS, NCBI
# # _BATCH_SIZE      = 20
# # _TIMEOUT         = 15
# # _MIN_SCORE       = 0.80
# # _RATE_SLEEP      = 0.25   # seconds between WoRMS calls


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  JSON REPAIR  (fix LLM parse errors)
# # # ─────────────────────────────────────────────────────────────────────────────
# # def safe_parse_json(raw: str) -> list[dict] | None:
# #     """
# #     Parse a raw LLM response into a JSON list.

# #     Handles:
# #       1. Clean JSON                    → json.loads()
# #       2. Fenced JSON (```json … ```)   → strip fences then parse
# #       3. Thinking blocks (<think>…)    → strip Gemma 4 chain-of-thought
# #       4. Truncated / malformed JSON    → json_repair library
# #       5. JSON object instead of array  → wrap in list
# #       6. Embedded text before/after    → regex extraction
# #     """
# #     if not raw or not raw.strip():
# #         return None

# #     text = raw.strip()

# #     # Remove Gemma 4 / Qwen thinking blocks
# #     text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
# #     text = re.sub(r"<\|thinking\|>.*?<\|/thinking\|>", "", text, flags=re.DOTALL).strip()

# #     # Strip markdown code fences
# #     text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
# #     text = re.sub(r"\s*```\s*$", "", text)
# #     text = text.strip()

# #     # Attempt 1: direct parse
# #     try:
# #         result = json.loads(text)
# #         if isinstance(result, list):
# #             return [r for r in result if isinstance(r, dict)]
# #         if isinstance(result, dict):
# #             # LLM wrapped array in an outer dict key
# #             for v in result.values():
# #                 if isinstance(v, list):
# #                     return [r for r in v if isinstance(r, dict)]
# #             return [result]
# #     except json.JSONDecodeError:
# #         pass

# #     # Attempt 2: find JSON array anywhere in the text
# #     array_match = re.search(r"\[[\s\S]*\]", text)
# #     if array_match:
# #         try:
# #             result = json.loads(array_match.group())
# #             if isinstance(result, list):
# #                 return [r for r in result if isinstance(r, dict)]
# #         except json.JSONDecodeError:
# #             pass

# #     # Attempt 3: json_repair
# #     if _JSON_REPAIR_AVAILABLE:
# #         try:
# #             repaired = _repair_json(text, return_objects=True)
# #             if isinstance(repaired, list):
# #                 return [r for r in repaired if isinstance(r, dict)]
# #             if isinstance(repaired, dict):
# #                 for v in repaired.values():
# #                     if isinstance(v, list):
# #                         return [r for r in v if isinstance(r, dict)]
# #                 return [repaired]
# #         except Exception as exc:
# #             logger.debug("[gnv] json_repair failed: %s", exc)

# #     # Attempt 4: extract individual JSON objects
# #     objects: list[dict] = []
# #     for m in re.finditer(r"\{[^{}]+\}", text):
# #         try:
# #             obj = json.loads(m.group())
# #             if isinstance(obj, dict) and len(obj) >= 2:
# #                 objects.append(obj)
# #         except json.JSONDecodeError:
# #             pass
# #     if objects:
# #         logger.debug("[gnv] recovered %d partial objects via regex", len(objects))
# #         return objects

# #     logger.warning("[gnv] safe_parse_json: all strategies exhausted — dropping chunk")
# #     return None


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  INTRA-DOCUMENT DEDUPLICATION
# # # ─────────────────────────────────────────────────────────────────────────────
# # def dedup_occurrences(
# #     occurrences: list[dict],
# #     keep_secondary: bool = True,
# # ) -> tuple[list[dict], int]:
# #     """
# #     Remove duplicate species × locality entries extracted from different
# #     document sections (Introduction / Results / Discussion / captions).

# #     Deduplication key: (canonicalised_name, canonicalised_locality)
# #     — occurrenceType is NOT part of the key so the same event described in
# #       multiple sections collapses to one canonical record.

# #     Resolution priority within a collision group:
# #       1. occurrenceType: Primary > Secondary > Uncertain > ""
# #       2. Among equal priority: longer rawTextEvidence string wins.

# #     When keep_secondary=True (default), a unique Secondary record whose
# #     (name, locality) does NOT appear in any Primary record is kept as-is.
# #     This preserves cited historical observations.

# #     Returns (deduplicated_list, n_removed).
# #     """
# #     # Priority: Primary > Secondary > Uncertain > unknown
# #     _priority = {"primary": 0, "secondary": 1, "uncertain": 2, "": 3}

# #     def _key(occ: dict) -> str:
# #         name = _canon(occ.get("validName") or occ.get("recordedName") or
# #                       occ.get("Recorded Name", ""))
# #         loc  = _canon(occ.get("verbatimLocality") or "")
# #         return f"{name}||{loc}"

# #     def _ev_len(occ: dict) -> int:
# #         return len(str(
# #             occ.get("Raw Text Evidence") or occ.get("rawTextEvidence", "")
# #         ))

# #     seen:    dict[str, dict] = {}
# #     removed = 0

# #     for occ in occurrences:
# #         if not isinstance(occ, dict):
# #             continue
# #         k = _key(occ)
# #         if k not in seen:
# #             seen[k] = occ
# #         else:
# #             existing  = seen[k]
# #             prio_new  = _priority.get(str(occ.get("occurrenceType","")).lower(), 3)
# #             prio_old  = _priority.get(str(existing.get("occurrenceType","")).lower(), 3)
# #             if prio_new < prio_old:
# #                 seen[k] = occ   # higher priority type wins
# #             elif prio_new == prio_old and _ev_len(occ) > _ev_len(existing):
# #                 seen[k] = occ   # same priority, richer evidence wins
# #             removed += 1

# #     result = list(seen.values())
# #     logger.info("[gnv] dedup: %d → %d records (%d removed)",
# #                 len(occurrences), len(result), removed)
# #     return result, removed


# # def _canon(s: str) -> str:
# #     """Canonical form for dedup comparison: lowercase, strip, normalise spaces."""
# #     return re.sub(r"\s+", " ", str(s).lower().strip())


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  GNV VERIFIER
# # # ─────────────────────────────────────────────────────────────────────────────
# # class GNVEnrichedVerifier:
# #     """
# #     Drop-in enhancement of species_verifier.py with full GNV field coverage
# #     and automatic wiki-profile enrichment.

# #     Key enhancements over base verifier:
# #       • Captures ALL GNV taxonomy ranks (not just phylum/class/order/family)
# #       • Stores genus, superfamily, suborder, infraorder when present
# #       • Parses outlinks to identify WoRMS, ITIS, CoL, EOL, GBIF source IDs
# #       • Stores vernacularNames list for wiki profiles
# #       • Merges result into BioTraceWiki article if wiki instance is supplied
# #     """

# #     def __init__(
# #         self,
# #         data_sources: str  = GNV_DATA_SOURCES,
# #         min_score:    float = _MIN_SCORE,
# #         wiki          = None,   # optional BioTraceWiki instance
# #     ):
# #         self.data_sources = data_sources
# #         self.min_score    = min_score
# #         self.wiki         = wiki

# #     # ── GNV batch call ────────────────────────────────────────────────────────
# #     def _call_gnv_batch(self, names: list[str]) -> dict[str, dict]:
# #         """Call GNV verifier for a batch of names. Returns {name: parsed_result}."""
# #         if not names:
# #             return {}
# #         try:
# #             r = requests.get(
# #                 GNV_VERIFIER_URL,
# #                 params={
# #                     "names":             "|".join(names),
# #                     "data_sources":      self.data_sources,
# #                     "with_vernaculars":  "true",
# #                     "with_species_group":"true",
# #                     "capitalize":        "true",
# #                 },
# #                 timeout=_TIMEOUT,
# #             )
# #             r.raise_for_status()
# #             results: dict[str, dict] = {}
# #             for item in r.json().get("names", []):
# #                 submitted = item.get("name", "")
# #                 best      = item.get("bestResult") or {}
# #                 if best:
# #                     results[submitted] = self._parse_gnv_result(item, best)
# #             return results
# #         except Exception as exc:
# #             logger.warning("[gnv] GNV batch call failed: %s", exc)
# #         return {}

# #     # ── Full GNV response parser ──────────────────────────────────────────────
# #     def _parse_gnv_result(self, item: dict, best: dict) -> dict:
# #         """
# #         Parse a complete GNV bestResult object into a flat enrichment dict.

# #         Maps ALL GNV fields including:
# #           currentCanonicalFull, matchedCanonicalFull, classificationPath,
# #           classificationRanks, taxonomicStatus, outlink, dataSourceTitleShort,
# #           score, matchType, taxonRank, vernacularNames
# #         """
# #         # ── Taxonomy from classificationPath + classificationRanks ─────────
# #         c_path  = best.get("classificationPath", "")  or ""
# #         c_ranks = best.get("classificationRanks", "") or ""
# #         tax     = self._parse_classification(c_path, c_ranks)

# #         # ── Taxonomic status ─────────────────────────────────────────────────
# #         status_raw = (best.get("taxonomicStatus", "") or "").lower()
# #         if   "synonym"  in status_raw:                      status = "synonym"
# #         elif "accepted" in status_raw or "valid" in status_raw: status = "accepted"
# #         else:                                               status = "unverified"

# #         # ── Outlink parsing: WoRMS / ITIS / CoL / GBIF / EOL ────────────────
# #         outlink  = str(best.get("outlink", "") or "")
# #         worms_id = itis_id = col_id = gbif_id = eol_id = ""

# #         if "marinespecies.org" in outlink:
# #             m = re.search(r"id=(\d+)", outlink)
# #             if m: worms_id = m.group(1)
# #         if "itis.gov" in outlink:
# #             m = re.search(r"tsn=(\d+)", outlink)
# #             if m: itis_id = m.group(1)
# #         if "catalogueoflife.org" in outlink:
# #             m = re.search(r"/taxon/([a-zA-Z0-9]+)", outlink)
# #             if m: col_id = m.group(1)
# #         if "gbif.org" in outlink:
# #             m = re.search(r"/species/(\d+)", outlink)
# #             if m: gbif_id = m.group(1)
# #         if "eol.org" in outlink:
# #             m = re.search(r"/pages/(\d+)", outlink)
# #             if m: eol_id = m.group(1)

# #         # ── Vernacular names (for wiki profiles) ────────────────────────────
# #         vern_raw   = best.get("vernacularNames") or item.get("vernacularNames") or []
# #         vernaculars = []
# #         if isinstance(vern_raw, list):
# #             for v in vern_raw[:8]:
# #                 if isinstance(v, dict):
# #                     n = v.get("vernacularName", "") or v.get("name", "")
# #                     if n:
# #                         vernaculars.append({
# #                             "name": n,
# #                             "language": v.get("language", ""),
# #                         })
# #                 elif isinstance(v, str) and v:
# #                     vernaculars.append({"name": v, "language": ""})

# #         return {
# #             # Core name fields
# #             "validName":       best.get("currentCanonicalFull", "") or best.get("matchedCanonicalFull", ""),
# #             "matchedName":     best.get("matchedCanonicalFull", ""),
# #             "currentName":     best.get("currentCanonicalFull", ""),
# #             "taxonRank":       best.get("taxonRank", ""),
# #             "taxonomicStatus": status,
# #             "matchScore":      float(best.get("score", 0) or 0),
# #             "matchType":       best.get("matchType", ""),
# #             "nameAccordingTo": best.get("dataSourceTitleShort", "") or "",

# #             # External IDs
# #             "wormsID":  worms_id,
# #             "itisID":   itis_id,
# #             "colID":    col_id,
# #             "gbifID":   gbif_id,
# #             "eolID":    eol_id,
# #             "outlink":  outlink,

# #             # Full taxonomy dict (all ranks captured)
# #             **tax,

# #             # Vernacular names for wiki
# #             "vernacularNames": vernaculars,

# #             # Raw classification for wiki
# #             "classificationPath":  c_path,
# #             "classificationRanks": c_ranks,
# #         }

# #     @staticmethod
# #     def _parse_classification(c_path: str, c_ranks: str) -> dict:
# #         """
# #         Parse GNV classificationPath + classificationRanks into a flat dict.

# #         Captures all ranks: kingdom, phylum, class, order, family, genus,
# #         superfamily, suborder, infraorder, tribe, subtribe.
# #         """
# #         tax: dict[str, str] = {
# #             "kingdom": "", "phylum": "", "class_": "", "subclass": "",
# #             "superorder": "", "order_": "", "suborder": "",
# #             "superfamily": "", "family_": "", "subfamily": "",
# #             "tribe": "", "genus_": "",
# #         }
# #         if not c_path or not c_ranks:
# #             return tax

# #         parts = c_path.split("|")
# #         ranks = c_ranks.split("|")

# #         rank_map = {
# #             "kingdom":     "kingdom",
# #             "phylum":      "phylum",
# #             "class":       "class_",
# #             "subclass":    "subclass",
# #             "superorder":  "superorder",
# #             "order":       "order_",
# #             "suborder":    "suborder",
# #             "superfamily": "superfamily",
# #             "family":      "family_",
# #             "subfamily":   "subfamily",
# #             "tribe":       "tribe",
# #             "genus":       "genus_",
# #         }
# #         for rank_raw, val in zip(ranks, parts):
# #             key = rank_map.get(rank_raw.lower().strip())
# #             if key:
# #                 tax[key] = val.strip()

# #         return tax

# #     # ── WoRMS REST authority override ─────────────────────────────────────────
# #     @lru_cache(maxsize=512)
# #     def _worms_by_aphia(self, aphia_id: str) -> dict:
# #         if not aphia_id:
# #             return {}
# #         try:
# #             r = requests.get(
# #                 f"{WORMS_REST_URL}/AphiaClassificationByAphiaID/{aphia_id}",
# #                 timeout=_TIMEOUT,
# #             )
# #             r.raise_for_status()
# #             tax: dict[str, str] = {}
# #             def _walk(node):
# #                 if not isinstance(node, dict): return
# #                 rank = (node.get("rank", "") or "").lower()
# #                 name = node.get("scientificname", "") or ""
# #                 rmap = {"phylum":"phylum","class":"class_","order":"order_","family":"family_","genus":"genus_"}
# #                 if rank in rmap:
# #                     tax[rmap[rank]] = name
# #                 _walk(node.get("child"))
# #             _walk(r.json())
# #             return tax
# #         except Exception as exc:
# #             logger.debug("[gnv] WoRMS AphiaID=%s: %s", aphia_id, exc)
# #         return {}

# #     # ── Main enrichment pipeline ─────────────────────────────────────────────
# #     def verify_and_enrich(
# #         self,
# #         occurrences: list[dict],
# #         update_wiki: bool = True,
# #     ) -> list[dict]:
# #         """
# #         Full pipeline:
# #           1. Batch GNV verification
# #           2. WoRMS authority override for marine species
# #           3. Wiki profile update (if wiki instance supplied)
# #           4. Intra-document deduplication
# #         """
# #         if not occurrences:
# #             return occurrences

# #         # Collect unique names
# #         unique_names: list[str] = []
# #         seen_names: set[str]    = set()
# #         for occ in occurrences:
# #             if not isinstance(occ, dict): continue
# #             name = str(occ.get("recordedName") or occ.get("Recorded Name", "")).strip()
# #             if name and name not in seen_names:
# #                 unique_names.append(name)
# #                 seen_names.add(name)

# #         logger.info("[gnv] Verifying %d unique names…", len(unique_names))

# #         # Batch GNV calls
# #         verification: dict[str, dict] = {}
# #         for i in range(0, len(unique_names), _BATCH_SIZE):
# #             batch = unique_names[i: i + _BATCH_SIZE]
# #             verification.update(self._call_gnv_batch(batch))
# #             if i + _BATCH_SIZE < len(unique_names):
# #                 time.sleep(_RATE_SLEEP)

# #         verified_ct = 0

# #         for occ in occurrences:
# #             if not isinstance(occ, dict): continue

# #             raw_name = str(occ.get("recordedName") or occ.get("Recorded Name", "")).strip()
# #             occ["recordedName"] = raw_name
# #             occ.setdefault("scientificName", raw_name)

# #             ver = verification.get(raw_name, {})
# #             if not ver:
# #                 occ["taxonomicStatus"] = "unverified"
# #                 continue

# #             score      = float(ver.get("matchScore", 0) or 0)
# #             valid_name = ver.get("validName", "")
# #             worms_id   = ver.get("wormsID", "")

# #             if valid_name and score >= self.min_score:
# #                 # Apply all GNV fields
# #                 occ["validName"]       = valid_name
# #                 occ["scientificName"]  = valid_name
# #                 occ["taxonRank"]       = ver.get("taxonRank", "")
# #                 occ["nameAccordingTo"] = ver.get("nameAccordingTo", "")
# #                 occ["taxonomicStatus"] = ver.get("taxonomicStatus", "unverified")
# #                 occ["matchScore"]      = round(score, 3)
# #                 occ["matchType"]       = ver.get("matchType", "")

# #                 # All external IDs
# #                 occ["wormsID"] = worms_id
# #                 occ["itisID"]  = ver.get("itisID", "")
# #                 occ["colID"]   = ver.get("colID", "")
# #                 occ["gbifID"]  = ver.get("gbifID", "")
# #                 occ["eolID"]   = ver.get("eolID", "")
# #                 occ["outlink"] = ver.get("outlink", "")

# #                 # Full taxonomy from GNV classification
# #                 for f in ("kingdom","phylum","class_","subclass","superorder",
# #                           "order_","suborder","superfamily","family_","subfamily",
# #                           "tribe","genus_"):
# #                     if ver.get(f):
# #                         occ[f] = ver[f]

# #                 # Vernacular names for wiki
# #                 occ["vernacularNames"]     = ver.get("vernacularNames", [])
# #                 occ["classificationPath"]  = ver.get("classificationPath", "")
# #                 occ["classificationRanks"] = ver.get("classificationRanks", "")

# #                 # WoRMS authority override for marine species
# #                 if worms_id:
# #                     wt = self._worms_by_aphia(worms_id)
# #                     if wt:
# #                         for f, key in [("phylum","phylum"),("class_","class_"),
# #                                        ("order_","order_"),("family_","family_"),
# #                                        ("genus_","genus_")]:
# #                             if wt.get(key):
# #                                 occ[f] = wt[key]
# #                     time.sleep(_RATE_SLEEP)

# #                 # Wiki profile update
# #                 if update_wiki and self.wiki is not None:
# #                     try:
# #                         self._update_wiki_profile(occ, ver)
# #                     except Exception as exc:
# #                         logger.debug("[gnv] wiki update: %s", exc)

# #                 verified_ct += 1
# #             else:
# #                 occ["taxonomicStatus"] = "unverified"
# #                 occ["matchScore"]      = round(score, 3)

# #         logger.info(
# #             "[gnv] %d/%d names resolved (score≥%.2f)",
# #             verified_ct, len(unique_names), self.min_score,
# #         )
# #         return occurrences

# #     def _update_wiki_profile(self, occ: dict, ver: dict):
# #         """Populate wiki article with GNV-sourced fields."""
# #         sp_slug = re.sub(r"[^a-z0-9_]", "_", (occ.get("validName","") or "").lower())[:80]
# #         if not sp_slug:
# #             return

# #         wiki_path = getattr(self.wiki, "root", None)
# #         if wiki_path is None:
# #             return

# #         article_path = wiki_path / "species" / f"{sp_slug}.json"
# #         if article_path.exists():
# #             try:
# #                 article = json.loads(article_path.read_text(encoding="utf-8"))
# #             except Exception:
# #                 article = {}
# #         else:
# #             article = {}

# #         article.setdefault("gnv_profile", {})
# #         gp = article["gnv_profile"]

# #         # Full taxonomy
# #         for f in ("kingdom","phylum","class_","subclass","order_","suborder",
# #                   "superfamily","family_","subfamily","tribe","genus_"):
# #             if ver.get(f):
# #                 gp[f] = ver[f]

# #         gp["classificationPath"]  = ver.get("classificationPath", "")
# #         gp["classificationRanks"] = ver.get("classificationRanks", "")
# #         gp["wormsID"]    = ver.get("wormsID", "")
# #         gp["itisID"]     = ver.get("itisID", "")
# #         gp["colID"]      = ver.get("colID", "")
# #         gp["gbifID"]     = ver.get("gbifID", "")
# #         gp["eolID"]      = ver.get("eolID", "")
# #         gp["outlink"]    = ver.get("outlink", "")
# #         gp["matchScore"] = ver.get("matchScore", 0)
# #         gp["matchType"]  = ver.get("matchType", "")

# #         # Vernacular names
# #         existing_vn = gp.get("vernacularNames", [])
# #         new_vn      = ver.get("vernacularNames", [])
# #         existing_set = {v.get("name","").lower() for v in existing_vn}
# #         for v in new_vn:
# #             if v.get("name","").lower() not in existing_set:
# #                 existing_vn.append(v)
# #         gp["vernacularNames"] = existing_vn[:20]

# #         article["gnv_profile"] = gp
# #         article["version"]     = article.get("version", 0) + 1
# #         article["last_updated"]= __import__("datetime").datetime.utcnow().isoformat()

# #         try:
# #             wiki_path_species = wiki_path / "species"
# #             wiki_path_species.mkdir(parents=True, exist_ok=True)
# #             article_path.write_text(
# #                 json.dumps(article, indent=2, ensure_ascii=False), encoding="utf-8"
# #             )
# #         except Exception as exc:
# #             logger.debug("[gnv] wiki write: %s", exc)


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  LOCALITY SPLITTER
# # # ─────────────────────────────────────────────────────────────────────────────
# # class LocalitySplitter:
# #     """
# #     Split compound locality strings into individual geographic entities.

# #     Problem:
# #       LLMs often join multiple localities into one verbatimLocality string:
# #         "Mandapam, Tuticorin, Rameswaram and Pamban"
# #         "Gulf of Mannar, Palk Bay, Lakshadweep"
# #         "Sites A, B and C (Gulf of Mannar)"

# #     Solution:
# #       1. Detect comma/semi-colon/conjunction patterns in verbatimLocality.
# #       2. Heuristically split into candidate place names.
# #       3. Validate each candidate against GeoNames IN SQLite (fast, offline).
# #       4. If GeoNames unavailable, use Nominatim (online, geopy).
# #       5. For each valid place: clone the occurrence record with that locality.

# #     Also handles:
# #       • "Village name" → coordinates via Nominatim
# #       • Station IDs like "St. 1, St. 2" → keep as-is if no GeoNames match
# #     """

# #     # Patterns that indicate a compound locality
# #     _SPLIT_PATTERNS = [
# #         r"\s+and\s+",          # "X and Y"
# #         r"\s*;\s*",            # "X; Y"
# #         r"\s*,\s*(?=[A-Z])",   # "X, Y" where Y starts with capital
# #         r"\s*&\s*",            # "X & Y"
# #     ]
# #     _SPLIT_RE = re.compile("|".join(_SPLIT_PATTERNS))

# #     # Tokens that indicate station IDs (don't geocode)
# #     _STATION_RE = re.compile(
# #         r"^(?:St(?:ation)?\.?\s*\d+|Site\s+[A-Z0-9]+|Plot\s+\d+|Transect\s+\d+|S\d+|T\d+)$",
# #         re.IGNORECASE,
# #     )

# #     def __init__(
# #         self,
# #         geonames_db:    str  = "biodiversity_data/geonames_india.db",
# #         use_nominatim:  bool = True,
# #         nominatim_agent: str = "BioTrace_v5_biodiversity",
# #         min_name_len:   int  = 4,   # minimum chars to consider a token a place name
# #     ):
# #         self.geonames_db   = geonames_db
# #         self.min_name_len  = min_name_len
# #         self._nom_geocoder = None

# #         if use_nominatim and _GEOPY_AVAILABLE:
# #             try:
# #                 self._nom_geocoder = _Nominatim(user_agent=nominatim_agent)
# #                 logger.info("[gnv] Nominatim geocoder ready")
# #             except Exception as exc:
# #                 logger.warning("[gnv] Nominatim init: %s", exc)

# #         import os, sqlite3
# #         self._gn_conn = None
# #         if geonames_db and os.path.exists(geonames_db):
# #             try:
# #                 self._gn_conn = sqlite3.connect(geonames_db, check_same_thread=False)
# #                 logger.info("[gnv] GeoNames DB connected: %s", geonames_db)
# #             except Exception as exc:
# #                 logger.warning("[gnv] GeoNames DB: %s", exc)

# #     def _is_known_place(self, name: str) -> bool:
# #         """Quick check: is this name a known place in GeoNames IN?"""
# #         if not name or len(name) < self.min_name_len:
# #             return False
# #         if self._gn_conn:
# #             try:
# #                 res = self._gn_conn.execute(
# #                     "SELECT 1 FROM geonames WHERE (name=? OR asciiname=?) AND country_code='IN' LIMIT 1",
# #                     (name, name),
# #                 ).fetchone()
# #                 return res is not None
# #             except Exception:
# #                 pass
# #         return True  # Optimistic if no GeoNames DB available

# #     def _geocode_nominatim(self, name: str) -> Optional[tuple[float, float]]:
# #         """Geocode a place name via Nominatim. Rate-limited to 1 req/s."""
# #         if not self._nom_geocoder:
# #             return None
# #         try:
# #             result = self._nom_geocoder.geocode(
# #                 f"{name}, India",
# #                 addressdetails=True,
# #                 timeout=10,
# #             )
# #             if result:
# #                 return (result.latitude, result.longitude)
# #         except Exception as exc:
# #             logger.debug("[gnv] Nominatim '%s': %s", name, exc)
# #         time.sleep(1.1)   # Nominatim 1 req/s limit
# #         return None

# #     def _split_locality_string(self, locality: str) -> list[str]:
# #         """
# #         Split a compound locality string into individual place tokens.

# #         Handles:
# #           "Mandapam, Tuticorin, Rameswaram" → ["Mandapam", "Tuticorin", "Rameswaram"]
# #           "Gulf of Mannar and Palk Bay"       → ["Gulf of Mannar", "Palk Bay"]
# #           "Sites A, B and C"                  → ["Sites A", "B", "C"]  (station IDs kept)
# #           "Kovalam (near Chennai)"            → ["Kovalam", "Chennai"] (parenthetical)
# #         """
# #         if not locality:
# #             return []

# #         # Extract parenthetical hints
# #         paren = re.findall(r"\(([^)]{3,40})\)", locality)
# #         base  = re.sub(r"\s*\([^)]*\)", "", locality).strip()

# #         # Split on conjunctions and separators
# #         raw_parts = self._SPLIT_RE.split(base)

# #         # Post-clean
# #         parts: list[str] = []
# #         for p in raw_parts + paren:
# #             p = p.strip().strip(",").strip()
# #             if len(p) >= self.min_name_len:
# #                 parts.append(p)

# #         return list(dict.fromkeys(parts))   # preserve order, deduplicate

# #     def split_localities(
# #         self,
# #         occurrences: list[dict],
# #         geocode_new: bool = True,
# #     ) -> list[dict]:
# #         """
# #         Expand compound verbatimLocality strings into separate occurrence records.

# #         Each split locality gets a cloned occurrence record.
# #         Original record is replaced only when split succeeds (≥2 parts validated).
# #         """
# #         expanded: list[dict] = []
# #         total_split = 0

# #         for occ in occurrences:
# #             if not isinstance(occ, dict):
# #                 expanded.append(occ)
# #                 continue

# #             locality = str(occ.get("verbatimLocality", "") or "").strip()
# #             if not locality:
# #                 expanded.append(occ)
# #                 continue

# #             # Skip station IDs and short strings
# #             if self._STATION_RE.match(locality) or len(locality) < self.min_name_len + 2:
# #                 expanded.append(occ)
# #                 continue

# #             parts = self._split_locality_string(locality)
# #             valid_parts = [p for p in parts if self._is_known_place(p)]

# #             if len(valid_parts) < 2:
# #                 # No compound locality detected — pass through
# #                 expanded.append(occ)
# #                 continue

# #             # Clone for each valid part
# #             for part in valid_parts:
# #                 clone = dict(occ)
# #                 clone["verbatimLocality"]   = part
# #                 clone["originalLocality"]   = locality
# #                 clone["localitySplitFrom"]  = locality

# #                 # Try to get coords for new locality if not already geocoded
# #                 if geocode_new and (
# #                     clone.get("decimalLatitude") is None
# #                     or clone.get("decimalLongitude") is None
# #                 ):
# #                     coords = self._geocode_nominatim(part)
# #                     if coords:
# #                         clone["decimalLatitude"]  = coords[0]
# #                         clone["decimalLongitude"] = coords[1]
# #                         clone["geocodingSource"]  = "Nominatim_split"

# #                 expanded.append(clone)
# #                 total_split += 1

# #         if total_split:
# #             logger.info(
# #                 "[gnv] locality split: %d original → %d expanded records (%d splits)",
# #                 len(occurrences), len(expanded), total_split,
# #             )
# #         return expanded

# #     def close(self):
# #         if self._gn_conn:
# #             self._gn_conn.close()


# """
# species_verifier.py  —  BioTrace v5.3 Update
# ────────────────────────────────────────────────────────────────────────────
# Global Names Architecture (GNA) Integration for Species Finding and Verification.

# This module replaces local/regex NER pipelines by relying entirely on the 
# official Global Names APIs:
#   1. GNfinder   (finder.globalnames.org) -> Extracts names from text chunks
#   2. GNverifier (verifier.globalnames.org) -> Validates & maps to higher taxonomy

# Darwin Core fields populated:
#   recordedName     — verbatim name extracted from the text
#   validName        — accepted/current canonical name
#   scientificName   — full accepted scientific name
#   taxonomicStatus  — accepted | synonym | unverified
#   matchScore       — GNA exact/fuzzy sort score
#   domain/kingdom/phylum/class_/order_/family_ — Higher taxonomy path
# """

# import logging
# import urllib.parse
# import requests

# logger = logging.getLogger("biotrace.species_verifier")
# # Fallback basic config if logger isn't initialized by the main app
# logging.basicConfig(level=logging.INFO)

# def find_species_with_gnfinder(text_chunk: str) -> list[str]:
#     """
#     Sends a text chunk to GNfinder to extract unique scientific names.
    
#     Args:
#         text_chunk (str): A paragraph or sentence from the document.
        
#     Returns:
#         list[str]: A list of unique scientific names found in the text.
#     """
#     if not text_chunk or not text_chunk.strip():
#         return []
        
#     url = "https://finder.globalnames.org/api/v1/find"
#     payload = {
#         "text": text_chunk,
#         "language": "eng", 
#         "wordsAround": 0,
#         "verification": False # Verification happens in the next step
#     }
    
#     try:
#         response = requests.post(url, json=payload, timeout=15)
#         response.raise_for_status()
#         data = response.json()
        
#         found_names = set()
#         for name_obj in data.get("names", []):
#             name = name_obj.get("name")
#             if name:
#                 found_names.add(name)
                
#         return list(found_names)
        
#     except requests.exceptions.RequestException as e:
#         logger.error(f"[GNfinder] Network/API error: {e}")
#         return []
#     except Exception as e:
#         logger.error(f"[GNfinder] Unexpected error: {e}")
#         return []


# def verify_species_with_gnverifier(recorded_name: str) -> dict:
#     """
#     Validates a recorded name using the GNverifier API.
    
#     Args:
#         recorded_name (str): The raw scientific name extracted by GNfinder.
        
#     Returns:
#         dict: A dictionary of standardized Darwin Core taxonomic fields.
#     """
#     # URL encode the species name to handle spaces and special chars safely
#     safe_name = urllib.parse.quote(recorded_name.strip())
#     url = f"https://verifier.globalnames.org/api/v1/verifications/{safe_name}"
    
#     # Base Darwin Core payload (returns this if verification fails or matches nothing)
#     result = {
#         "recordedName": recorded_name,
#         "validName": None,
#         "scientificName": recorded_name,
#         "taxonomicStatus": "unverified",
#         "matchScore": 0.0,
#         "nameAccordingTo": None,
#         "domain": "",
#         "kingdom": "",
#         "phylum": "",
#         "class_": "",
#         "order_": "",
#         "family_": ""
#     }
    
#     try:
#         response = requests.get(url, timeout=15)
#         response.raise_for_status()
#         data = response.json()
        
#         names_data = data.get("names", [])
#         if not names_data:
#             return result
            
#         first_match = names_data[0]
#         best_result = first_match.get("bestResult")
        
#         if not best_result:
#             return result
        
#         # Populate verified fields based on the API response structure
#         result["validName"] = best_result.get("currentCanonicalSimple") or best_result.get("currentName")
#         result["scientificName"] = best_result.get("currentName") or recorded_name
#         result["taxonomicStatus"] = best_result.get("taxonomicStatus", "unverified")
#         result["matchScore"] = float(best_result.get("sortScore", 0.0))
#         result["nameAccordingTo"] = best_result.get("dataSourceTitleShort", "GNverifier")
        
#         # Parse Classification Path and Ranks
#         # Example ranks: "domain|kingdom|phylum|class|order|family|genus|species"
#         ranks_str = best_result.get("classificationRanks", "")
#         path_str = best_result.get("classificationPath", "")
        
#         if ranks_str and path_str:
#             ranks = ranks_str.split("|")
#             path = path_str.split("|")
            
#             # Map specific ranks to our schema by zipping them together
#             if len(ranks) == len(path):
#                 for rank, taxon in zip(ranks, path):
#                     rank = rank.lower()
#                     if rank == "domain":
#                         result["domain"] = taxon
#                     elif rank == "kingdom":
#                         result["kingdom"] = taxon
#                     elif rank == "phylum":
#                         result["phylum"] = taxon
#                     elif rank == "class":
#                         result["class_"] = taxon
#                     elif rank == "order":
#                         result["order_"] = taxon
#                     elif rank == "family":
#                         result["family_"] = taxon
                    
#     except requests.exceptions.RequestException as e:
#         logger.error(f"[GNverifier] API request failed for '{recorded_name}': {e}")
#     except Exception as e:
#         logger.error(f"[GNverifier] API parsing error for '{recorded_name}': {e}")
        
#     return result

# # Simple test block if you run the script directly
# if __name__ == "__main__":
#     test_text = "Medusa of Cassiopea andromeda was reported from the Gulf of Kutch."
#     print("1. Running GNfinder...")
#     names = find_species_with_gnfinder(test_text)
#     print(f"Found names: {names}")
    
#     if names:
#         print(f"\n2. Running GNverifier for '{names[0]}'...")
#         verified_data = verify_species_with_gnverifier(names[0])
#         import json
#         print(json.dumps(verified_data, indent=2))

"""
biotrace_gnv.py  —  BioTrace v5.1
────────────────────────────────────────────────────────────────────────────
Enhanced Global Names Verifier (GNV) integration with:
  • Full GNV JSON response mapping (currentName, classificationPath,
    classificationRanks, taxonomicStatus, outlinks, vernacularNames)
  • WoRMS REST authoritative override for marine species
  • Wiki-enriched species profiles from GNV metadata
  • Intra-document deduplication (species × locality × source)
  • Locality segregation: comma-separated locality strings split into
    individual geographic entries using contextual heuristics + GeoNames
  • JSON repair using json_repair library for LLM parse errors
  • Gemma 4 response extraction (handles thinking-block artifacts)

GNV API documentation reference:
  verifier.globalnames.org/api/v1/verifications
  Response fields captured:
    names[].name                       — submitted name
    names[].bestResult.currentCanonicalFull  — accepted valid name
    names[].bestResult.matchedCanonicalFull  — matched canonical
    names[].bestResult.classificationPath    — "Animalia|Chordata|…"
    names[].bestResult.classificationRanks   — "kingdom|phylum|class|…"
    names[].bestResult.taxonomicStatus       — "accepted" | "synonym" | …
    names[].bestResult.outlink               — WoRMS / ITIS / CoL URL
    names[].bestResult.dataSourceTitleShort  — "WoRMS" | "CoL" | "GBIF"
    names[].bestResult.score                 — confidence 0–1
    names[].bestResult.matchType             — "Exact" | "Fuzzy" | …
    names[].bestResult.taxonRank             — "species" | "genus" | …

Usage:
    from biotrace_gnv import GNVEnrichedVerifier, LocalitySplitter
    verifier = GNVEnrichedVerifier()
    occurrences = verifier.verify_and_enrich(occurrences)
    occurrences = LocalitySplitter().split_localities(occurrences)
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from functools import lru_cache
from typing import Any, Optional

import requests

logger = logging.getLogger("biotrace.gnv")

# ─────────────────────────────────────────────────────────────────────────────
#  OPTIONAL DEPS
# ─────────────────────────────────────────────────────────────────────────────
_JSON_REPAIR_AVAILABLE = False
try:
    from json_repair import repair_json as _repair_json
    _JSON_REPAIR_AVAILABLE = True
    logger.info("[gnv] json_repair available")
except ImportError:
    pass

_GEOPY_AVAILABLE = False
try:
    from geopy.geocoders import Nominatim as _Nominatim
    _GEOPY_AVAILABLE = True
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
GNV_VERIFIER_URL = "https://verifier.globalnames.org/api/v1/verifications"
GNV_FINDER_URL   = "https://finder.globalnames.org/api/v1/find"
WORMS_REST_URL   = "https://www.marinespecies.org/rest"

GNV_DATA_SOURCES = "169,1,11,12,4"   # WoRMS, CoL, GBIF, ITIS, NCBI
_BATCH_SIZE      = 20
_TIMEOUT         = 15
_MIN_SCORE       = 0.80
_RATE_SLEEP      = 0.25   # seconds between WoRMS calls


# ─────────────────────────────────────────────────────────────────────────────
#  JSON REPAIR  (fix LLM parse errors)
# ─────────────────────────────────────────────────────────────────────────────
def safe_parse_json(raw: str) -> list[dict] | None:
    """
    Parse a raw LLM response into a JSON list.

    Handles:
      1. Clean JSON                    → json.loads()
      2. Fenced JSON (```json … ```)   → strip fences then parse
      3. Thinking blocks (<think>…)    → strip reasoning chain
      4. Truncated / malformed JSON    → json_repair library
      5. JSON object instead of array  → wrap in list
      6. Embedded text before/after    → regex extraction
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Remove reasoning blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<\|thinking\|>.*?<\|/thinking\|>", "", text, flags=re.DOTALL).strip()

    # Strip markdown code fences
    # text = re.sub(r"^json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    # Attempt 1: direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [r for r in result if isinstance(r, dict)]
        if isinstance(result, dict):
            # LLM wrapped array in an outer dict key
            for v in result.values():
                if isinstance(v, list):
                    return [r for r in v if isinstance(r, dict)]
            return [result]
    except json.JSONDecodeError:
        pass

    # Attempt 2: find JSON array anywhere in the text safely
    # (Fixes the issue where an early bracket truncates the string)
    array_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", text)
    if array_match:
        try:
            result = json.loads(array_match.group(0))
            if isinstance(result, list):
                return [r for r in result if isinstance(r, dict)]
        except json.JSONDecodeError:
            pass

    # Attempt 2b: check for empty array explicitly
    if re.search(r"^\s*\[\s*\]\s*$", text):
        return []

    # Attempt 3: json_repair
    if _JSON_REPAIR_AVAILABLE:
        try:
            repaired = _repair_json(text, return_objects=True)
            if isinstance(repaired, list):
                return [r for r in repaired if isinstance(r, dict)]
            if isinstance(repaired, dict):
                for v in repaired.values():
                    if isinstance(v, list):
                        return [r for r in v if isinstance(r, dict)]
                return [repaired]
        except Exception as exc:
            logger.debug("[gnv] json_repair failed: %s", exc)

    # Attempt 4: extract individual JSON objects safely
    objects: list[dict] = []
    for m in re.finditer(r"\{[^{}]+\}", text):
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and len(obj) >= 2:
                objects.append(obj)
        except json.JSONDecodeError:
            pass
    if objects:
        logger.debug("[gnv] recovered %d partial objects via regex", len(objects))
        return objects

    logger.warning("[gnv] safe_parse_json: all strategies exhausted — dropping chunk")
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  INTRA-DOCUMENT DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────
def dedup_occurrences(
    occurrences: list[dict],
    keep_secondary: bool = True,
) -> tuple[list[dict], int]:
    """
    Remove duplicate species × locality entries extracted from different
    document sections (Introduction / Results / Discussion / captions).

    Deduplication key: (canonicalised_name, canonicalised_locality)
    — occurrenceType is NOT part of the key so the same event described in
      multiple sections collapses to one canonical record.

    Resolution priority within a collision group:
      1. occurrenceType: Primary > Secondary > Uncertain > ""
      2. Among equal priority: longer rawTextEvidence string wins.

    When keep_secondary=True (default), a unique Secondary record whose
    (name, locality) does NOT appear in any Primary record is kept as-is.
    This preserves cited historical observations.

    Returns (deduplicated_list, n_removed).
    """
    # Priority: Primary > Secondary > Uncertain > unknown
    _priority = {"primary": 0, "secondary": 1, "uncertain": 2, "": 3}

    def _key(occ: dict) -> str:
        name = _canon(occ.get("validName") or occ.get("recordedName") or
                      occ.get("Recorded Name", ""))
        loc  = _canon(occ.get("verbatimLocality") or "")
        return f"{name}||{loc}"

    def _ev_len(occ: dict) -> int:
        return len(str(
            occ.get("Raw Text Evidence") or occ.get("rawTextEvidence", "")
        ))

    seen:    dict[str, dict] = {}
    removed = 0

    for occ in occurrences:
        if not isinstance(occ, dict):
            continue
        k = _key(occ)
        if k not in seen:
            seen[k] = occ
        else:
            existing  = seen[k]
            prio_new  = _priority.get(str(occ.get("occurrenceType","")).lower(), 3)
            prio_old  = _priority.get(str(existing.get("occurrenceType","")).lower(), 3)
            if prio_new < prio_old:
                seen[k] = occ   # higher priority type wins
            elif prio_new == prio_old and _ev_len(occ) > _ev_len(existing):
                seen[k] = occ   # same priority, richer evidence wins
            removed += 1

    result = list(seen.values())
    logger.info("[gnv] dedup: %d → %d records (%d removed)",
                len(occurrences), len(result), removed)
    return result, removed


def _canon(s: str) -> str:
    """Canonical form for dedup comparison: lowercase, strip, normalise spaces."""
    return re.sub(r"\s+", " ", str(s).lower().strip())


# ─────────────────────────────────────────────────────────────────────────────
#  GNV VERIFIER
# ─────────────────────────────────────────────────────────────────────────────
class GNVEnrichedVerifier:
    """
    Drop-in enhancement of species_verifier.py with full GNV field coverage
    and automatic wiki-profile enrichment.

    Key enhancements over base verifier:
      • Captures ALL GNV taxonomy ranks (not just phylum/class/order/family)
      • Stores genus, superfamily, suborder, infraorder when present
      • Parses outlinks to identify WoRMS, ITIS, CoL, EOL, GBIF source IDs
      • Stores vernacularNames list for wiki profiles
      • Merges result into BioTraceWiki article if wiki instance is supplied
    """

    def __init__(
        self,
        data_sources: str  = GNV_DATA_SOURCES,
        min_score:    float = _MIN_SCORE,
        wiki          = None,   # optional BioTraceWiki instance
    ):
        self.data_sources = data_sources
        self.min_score    = min_score
        self.wiki         = wiki

    # ── GNV batch call ────────────────────────────────────────────────────────
    def _call_gnv_batch(self, names: list[str]) -> dict[str, dict]:
        """Call GNV verifier for a batch of names. Returns {name: parsed_result}."""
        if not names:
            return {}
        try:
            r = requests.get(
                GNV_VERIFIER_URL,
                params={
                    "names":             "|".join(names),
                    "data_sources":      self.data_sources,
                    "with_vernaculars":  "true",
                    "with_species_group":"true",
                    "capitalize":        "true",
                },
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            results: dict[str, dict] = {}
            for item in r.json().get("names", []):
                submitted = item.get("name", "")
                best      = item.get("bestResult") or {}
                if best:
                    results[submitted] = self._parse_gnv_result(item, best)
            return results
        except Exception as exc:
            logger.warning("[gnv] GNV batch call failed: %s", exc)
        return {}

    # ── Full GNV response parser ──────────────────────────────────────────────
    def _parse_gnv_result(self, item: dict, best: dict) -> dict:
        """
        Parse a complete GNV bestResult object into a flat enrichment dict.

        Maps ALL GNV fields including:
          currentCanonicalFull, matchedCanonicalFull, classificationPath,
          classificationRanks, taxonomicStatus, outlink, dataSourceTitleShort,
          score, matchType, taxonRank, vernacularNames
        """
        # ── Taxonomy from classificationPath + classificationRanks ─────────
        c_path  = best.get("classificationPath", "")  or ""
        c_ranks = best.get("classificationRanks", "") or ""
        tax     = self._parse_classification(c_path, c_ranks)

        # ── Taxonomic status ─────────────────────────────────────────────────
        status_raw = (best.get("taxonomicStatus", "") or "").lower()
        if   "synonym"  in status_raw:                      status = "synonym"
        elif "accepted" in status_raw or "valid" in status_raw: status = "accepted"
        else:                                               status = "unverified"

        # ── Outlink parsing: WoRMS / ITIS / CoL / GBIF / EOL ────────────────
        outlink  = str(best.get("outlink", "") or "")
        worms_id = itis_id = col_id = gbif_id = eol_id = ""

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

        # ── Vernacular names (for wiki profiles) ────────────────────────────
        vern_raw   = best.get("vernacularNames") or item.get("vernacularNames") or []
        vernaculars = []
        if isinstance(vern_raw, list):
            for v in vern_raw[:8]:
                if isinstance(v, dict):
                    n = v.get("vernacularName", "") or v.get("name", "")
                    if n:
                        vernaculars.append({
                            "name": n,
                            "language": v.get("language", ""),
                        })
                elif isinstance(v, str) and v:
                    vernaculars.append({"name": v, "language": ""})

        return {
            # Core name fields
            "validName":       best.get("currentCanonicalFull", "") or best.get("matchedCanonicalFull", ""),
            "matchedName":     best.get("matchedCanonicalFull", ""),
            "currentName":     best.get("currentCanonicalFull", ""),
            "taxonRank":       best.get("taxonRank", ""),
            "taxonomicStatus": status,
            "matchScore":      float(best.get("score", 0) or 0),
            "matchType":       best.get("matchType", ""),
            "nameAccordingTo": best.get("dataSourceTitleShort", "") or "",

            # External IDs
            "wormsID":  worms_id,
            "itisID":   itis_id,
            "colID":    col_id,
            "gbifID":   gbif_id,
            "eolID":    eol_id,
            "outlink":  outlink,

            # Full taxonomy dict (all ranks captured)
            **tax,

            # Vernacular names for wiki
            "vernacularNames": vernaculars,

            # Raw classification for wiki
            "classificationPath":  c_path,
            "classificationRanks": c_ranks,
        }

    @staticmethod
    def _parse_classification(c_path: str, c_ranks: str) -> dict:
        """
        Parse GNV classificationPath + classificationRanks into a flat dict.

        Captures all ranks: kingdom, phylum, class, order, family, genus,
        superfamily, suborder, infraorder, tribe, subtribe.
        """
        tax: dict[str, str] = {
            "kingdom": "", "phylum": "", "class_": "", "subclass": "",
            "superorder": "", "order_": "", "suborder": "",
            "superfamily": "", "family_": "", "subfamily": "",
            "tribe": "", "genus_": "",
        }
        if not c_path or not c_ranks:
            return tax

        parts = c_path.split("|")
        ranks = c_ranks.split("|")

        rank_map = {
            "kingdom":     "kingdom",
            "phylum":      "phylum",
            "class":       "class_",
            "subclass":    "subclass",
            "superorder":  "superorder",
            "order":       "order_",
            "suborder":    "suborder",
            "superfamily": "superfamily",
            "family":      "family_",
            "subfamily":   "subfamily",
            "tribe":       "tribe",
            "genus":       "genus_",
        }
        for rank_raw, val in zip(ranks, parts):
            key = rank_map.get(rank_raw.lower().strip())
            if key:
                tax[key] = val.strip()

        return tax

    # ── WoRMS REST authority override ─────────────────────────────────────────
    @lru_cache(maxsize=512)
    def _worms_by_aphia(self, aphia_id: str) -> dict:
        if not aphia_id:
            return {}
        try:
            r = requests.get(
                f"{WORMS_REST_URL}/AphiaClassificationByAphiaID/{aphia_id}",
                timeout=_TIMEOUT,
            )
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
            logger.debug("[gnv] WoRMS AphiaID=%s: %s", aphia_id, exc)
        return {}

    # ── Main enrichment pipeline ─────────────────────────────────────────────
    def verify_and_enrich(
        self,
        occurrences: list[dict],
        update_wiki: bool = True,
    ) -> list[dict]:
        """
        Full pipeline:
          1. Batch GNV verification
          2. WoRMS authority override for marine species
          3. Wiki profile update (if wiki instance supplied)
          4. Intra-document deduplication
        """
        if not occurrences:
            return occurrences

        # Collect unique names
        unique_names: list[str] = []
        seen_names: set[str]    = set()
        for occ in occurrences:
            if not isinstance(occ, dict): continue
            name = str(occ.get("recordedName") or occ.get("Recorded Name", "")).strip()
            if name and name not in seen_names:
                unique_names.append(name)
                seen_names.add(name)

        logger.info("[gnv] Verifying %d unique names…", len(unique_names))

        # Batch GNV calls
        verification: dict[str, dict] = {}
        for i in range(0, len(unique_names), _BATCH_SIZE):
            batch = unique_names[i: i + _BATCH_SIZE]
            verification.update(self._call_gnv_batch(batch))
            if i + _BATCH_SIZE < len(unique_names):
                time.sleep(_RATE_SLEEP)

        verified_ct = 0

        for occ in occurrences:
            if not isinstance(occ, dict): continue

            raw_name = str(occ.get("recordedName") or occ.get("Recorded Name", "")).strip()
            occ["recordedName"] = raw_name
            occ.setdefault("scientificName", raw_name)

            ver = verification.get(raw_name, {})
            if not ver:
                occ["taxonomicStatus"] = "unverified"
                continue

            score      = float(ver.get("matchScore", 0) or 0)
            valid_name = ver.get("validName", "")
            worms_id   = ver.get("wormsID", "")

            if valid_name and score >= self.min_score:
                # Apply all GNV fields
                occ["validName"]       = valid_name
                occ["scientificName"]  = valid_name
                occ["taxonRank"]       = ver.get("taxonRank", "")
                occ["nameAccordingTo"] = ver.get("nameAccordingTo", "")
                occ["taxonomicStatus"] = ver.get("taxonomicStatus", "unverified")
                occ["matchScore"]      = round(score, 3)
                occ["matchType"]       = ver.get("matchType", "")

                # All external IDs
                occ["wormsID"] = worms_id
                occ["itisID"]  = ver.get("itisID", "")
                occ["colID"]   = ver.get("colID", "")
                occ["gbifID"]  = ver.get("gbifID", "")
                occ["eolID"]   = ver.get("eolID", "")
                occ["outlink"] = ver.get("outlink", "")

                # Full taxonomy from GNV classification
                for f in ("kingdom","phylum","class_","subclass","superorder",
                          "order_","suborder","superfamily","family_","subfamily",
                          "tribe","genus_"):
                    if ver.get(f):
                        occ[f] = ver[f]

                # Vernacular names for wiki
                occ["vernacularNames"]     = ver.get("vernacularNames", [])
                occ["classificationPath"]  = ver.get("classificationPath", "")
                occ["classificationRanks"] = ver.get("classificationRanks", "")

                # WoRMS authority override for marine species
                if worms_id:
                    wt = self._worms_by_aphia(worms_id)
                    if wt:
                        for f, key in [("phylum","phylum"),("class_","class_"),
                                       ("order_","order_"),("family_","family_"),
                                       ("genus_","genus_")]:
                            if wt.get(key):
                                occ[f] = wt[key]
                    time.sleep(_RATE_SLEEP)

                # Wiki profile update
                if update_wiki and self.wiki is not None:
                    try:
                        self._update_wiki_profile(occ, ver)
                    except Exception as exc:
                        logger.debug("[gnv] wiki update: %s", exc)

                verified_ct += 1
            else:
                occ["taxonomicStatus"] = "unverified"
                occ["matchScore"]      = round(score, 3)

        logger.info(
            "[gnv] %d/%d names resolved (score≥%.2f)",
            verified_ct, len(unique_names), self.min_score,
        )
        return occurrences

    def _update_wiki_profile(self, occ: dict, ver: dict):
        """Populate wiki article with GNV-sourced fields."""
        sp_slug = re.sub(r"[^a-z0-9_]", "_", (occ.get("validName","") or "").lower())[:80]
        if not sp_slug:
            return

        wiki_path = getattr(self.wiki, "root", None)
        if wiki_path is None:
            return

        article_path = wiki_path / "species" / f"{sp_slug}.json"
        if article_path.exists():
            try:
                article = json.loads(article_path.read_text(encoding="utf-8"))
            except Exception:
                article = {}
        else:
            article = {}

        article.setdefault("gnv_profile", {})
        gp = article["gnv_profile"]

        # Full taxonomy
        for f in ("kingdom","phylum","class_","subclass","order_","suborder",
                  "superfamily","family_","subfamily","tribe","genus_"):
            if ver.get(f):
                gp[f] = ver[f]

        gp["classificationPath"]  = ver.get("classificationPath", "")
        gp["classificationRanks"] = ver.get("classificationRanks", "")
        gp["wormsID"]    = ver.get("wormsID", "")
        gp["itisID"]     = ver.get("itisID", "")
        gp["colID"]      = ver.get("colID", "")
        gp["gbifID"]     = ver.get("gbifID", "")
        gp["eolID"]      = ver.get("eolID", "")
        gp["outlink"]    = ver.get("outlink", "")
        gp["matchScore"] = ver.get("matchScore", 0)
        gp["matchType"]  = ver.get("matchType", "")

        # Vernacular names
        existing_vn = gp.get("vernacularNames", [])
        new_vn      = ver.get("vernacularNames", [])
        existing_set = {v.get("name","").lower() for v in existing_vn}
        for v in new_vn:
            if v.get("name","").lower() not in existing_set:
                existing_vn.append(v)
        gp["vernacularNames"] = existing_vn[:20]

        article["gnv_profile"] = gp
        article["version"]     = article.get("version", 0) + 1
        article["last_updated"]= __import__("datetime").datetime.utcnow().isoformat()

        try:
            wiki_path_species = wiki_path / "species"
            wiki_path_species.mkdir(parents=True, exist_ok=True)
            article_path.write_text(
                json.dumps(article, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.debug("[gnv] wiki write: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
#  LOCALITY SPLITTER
# ─────────────────────────────────────────────────────────────────────────────
class LocalitySplitter:
    """
    Split compound locality strings into individual geographic entities.

    Problem:
      LLMs often join multiple localities into one verbatimLocality string:
        "Mandapam, Tuticorin, Rameswaram and Pamban"
        "Gulf of Mannar, Palk Bay, Lakshadweep"
        "Sites A, B and C (Gulf of Mannar)"

    Solution:
      1. Detect comma/semi-colon/conjunction patterns in verbatimLocality.
      2. Heuristically split into candidate place names.
      3. Validate each candidate against GeoNames IN SQLite (fast, offline).
      4. If GeoNames unavailable, use Nominatim (online, geopy).
      5. For each valid place: clone the occurrence record with that locality.

    Also handles:
      • "Village name" → coordinates via Nominatim
      • Station IDs like "St. 1, St. 2" → keep as-is if no GeoNames match
    """

    # Patterns that indicate a compound locality
    _SPLIT_PATTERNS = [
        r"\s+and\s+",          # "X and Y"
        r"\s*;\s*",            # "X; Y"
        r"\s*,\s*(?=[A-Z])",   # "X, Y" where Y starts with capital
        r"\s*&\s*",            # "X & Y"
    ]
    _SPLIT_RE = re.compile("|".join(_SPLIT_PATTERNS))

    # Tokens that indicate station IDs (don't geocode)
    _STATION_RE = re.compile(
        r"^(?:St(?:ation)?\.?\s*\d+|Site\s+[A-Z0-9]+|Plot\s+\d+|Transect\s+\d+|S\d+|T\d+)$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        geonames_db:    str  = "biodiversity_data/geonames_india.db",
        use_nominatim:  bool = True,
        nominatim_agent: str = "BioTrace_v5_biodiversity",
        min_name_len:   int  = 4,   # minimum chars to consider a token a place name
    ):
        self.geonames_db   = geonames_db
        self.min_name_len  = min_name_len
        self._nom_geocoder = None

        if use_nominatim and _GEOPY_AVAILABLE:
            try:
                self._nom_geocoder = _Nominatim(user_agent=nominatim_agent)
                logger.info("[gnv] Nominatim geocoder ready")
            except Exception as exc:
                logger.warning("[gnv] Nominatim init: %s", exc)

        import os, sqlite3
        self._gn_conn = None
        if geonames_db and os.path.exists(geonames_db):
            try:
                self._gn_conn = sqlite3.connect(geonames_db, check_same_thread=False)
                logger.info("[gnv] GeoNames DB connected: %s", geonames_db)
            except Exception as exc:
                logger.warning("[gnv] GeoNames DB: %s", exc)

    def _is_known_place(self, name: str) -> bool:
        """Quick check: is this name a known place in GeoNames IN?"""
        if not name or len(name) < self.min_name_len:
            return False
        if self._gn_conn:
            try:
                res = self._gn_conn.execute(
                    "SELECT 1 FROM geonames WHERE (name=? OR asciiname=?) AND country_code='IN' LIMIT 1",
                    (name, name),
                ).fetchone()
                return res is not None
            except Exception:
                pass
        return True  # Optimistic if no GeoNames DB available

    def _geocode_nominatim(self, name: str) -> Optional[tuple[float, float]]:
        """Geocode a place name via Nominatim. Rate-limited to 1 req/s."""
        if not self._nom_geocoder:
            return None
        try:
            result = self._nom_geocoder.geocode(
                f"{name}, India",
                addressdetails=True,
                timeout=10,
            )
            if result:
                return (result.latitude, result.longitude)
        except Exception as exc:
            logger.debug("[gnv] Nominatim '%s': %s", name, exc)
        time.sleep(1.1)   # Nominatim 1 req/s limit
        return None

    def _split_locality_string(self, locality: str) -> list[str]:
        """
        Split a compound locality string into individual place tokens.

        Handles:
          "Mandapam, Tuticorin, Rameswaram" → ["Mandapam", "Tuticorin", "Rameswaram"]
          "Gulf of Mannar and Palk Bay"       → ["Gulf of Mannar", "Palk Bay"]
          "Sites A, B and C"                  → ["Sites A", "B", "C"]  (station IDs kept)
          "Kovalam (near Chennai)"            → ["Kovalam", "Chennai"] (parenthetical)
        """
        if not locality:
            return []

        # Extract parenthetical hints
        paren = re.findall(r"\(([^)]{3,40})\)", locality)
        base  = re.sub(r"\s*\([^)]*\)", "", locality).strip()

        # Split on conjunctions and separators
        raw_parts = self._SPLIT_RE.split(base)

        # Post-clean
        parts: list[str] = []
        for p in raw_parts + paren:
            p = p.strip().strip(",").strip()
            if len(p) >= self.min_name_len:
                parts.append(p)

        return list(dict.fromkeys(parts))   # preserve order, deduplicate

    def split_localities(
        self,
        occurrences: list[dict],
        geocode_new: bool = True,
    ) -> list[dict]:
        """
        Expand compound verbatimLocality strings into separate occurrence records.

        Each split locality gets a cloned occurrence record.
        Original record is replaced only when split succeeds (≥2 parts validated).
        """
        expanded: list[dict] = []
        total_split = 0

        for occ in occurrences:
            if not isinstance(occ, dict):
                expanded.append(occ)
                continue

            locality = str(occ.get("verbatimLocality", "") or "").strip()
            if not locality:
                expanded.append(occ)
                continue

            # Skip station IDs and short strings
            if self._STATION_RE.match(locality) or len(locality) < self.min_name_len + 2:
                expanded.append(occ)
                continue

            parts = self._split_locality_string(locality)
            valid_parts = [p for p in parts if self._is_known_place(p)]

            if len(valid_parts) < 2:
                # No compound locality detected — pass through
                expanded.append(occ)
                continue

            # Clone for each valid part
            for part in valid_parts:
                clone = dict(occ)
                clone["verbatimLocality"]   = part
                clone["originalLocality"]   = locality
                clone["localitySplitFrom"]  = locality

                # Try to get coords for new locality if not already geocoded
                if geocode_new and (
                    clone.get("decimalLatitude") is None
                    or clone.get("decimalLongitude") is None
                ):
                    coords = self._geocode_nominatim(part)
                    if coords:
                        clone["decimalLatitude"]  = coords[0]
                        clone["decimalLongitude"] = coords[1]
                        clone["geocodingSource"]  = "Nominatim_split"

                expanded.append(clone)
                total_split += 1

        if total_split:
            logger.info(
                "[gnv] locality split: %d original → %d expanded records (%d splits)",
                len(occurrences), len(expanded), total_split,
            )
        return expanded

    def close(self):
        if self._gn_conn:
            self._gn_conn.close()
