"""
Shared taxonomy orchestration for BioTrace apps.

This module consolidates scientific-name detection and verification so the
Streamlit entry points do not each carry their own fallback stacks.
"""
from __future__ import annotations

import inspect
import logging
import re
from functools import lru_cache
from typing import Callable

logger = logging.getLogger("biotrace.taxonomy")

_GENUS = "genus"
_SPECIES = "species"
_CARDINALITY_LABELS = {1: _GENUS, 2: _SPECIES}
_OPEN_NOM_TOKENS = {
    "cf",
    "cf.",
    "aff",
    "aff.",
    "sp",
    "sp.",
    "spp",
    "spp.",
    "ssp",
    "ssp.",
    "subsp",
    "subsp.",
    "var",
    "var.",
    "f",
    "f.",
    "nov",
    "nov.",
    "sensu",
}

_LEGACY_FINDER_AVAILABLE = False
_LEGACY_VERIFY_AVAILABLE = False
_UNIFIED_VERIFY_AVAILABLE = False
_RULE_NER_AVAILABLE = False
_BIODIVIZ_AVAILABLE = False

_legacy_find_names = None
_legacy_verify_occurrences = None
UnifiedTaxonVerifier = None
_extract_taxa = None
BiodiVizPipeline = None

try:
    from species_verifier import find_names_in_text as _legacy_find_names

    _LEGACY_FINDER_AVAILABLE = True
except ImportError:
    pass

try:
    from species_verifier import verify_occurrence_names as _legacy_verify_occurrences

    _LEGACY_VERIFY_AVAILABLE = True
except ImportError:
    pass

try:
    from biotrace_unified_verifier import UnifiedTaxonVerifier

    _UNIFIED_VERIFY_AVAILABLE = True
except ImportError:
    pass

try:
    from biotrace_ner import extract_taxa as _extract_taxa

    _RULE_NER_AVAILABLE = True
except ImportError:
    pass

try:
    from biotrace_hf_ner import BiodiVizPipeline

    _BIODIVIZ_AVAILABLE = True
except ImportError:
    pass


def _log(log_cb, message: str, level: str = "ok") -> None:
    if callable(log_cb):
        log_cb(message, level)
        return
    if level == "warn":
        logger.warning(message)
    else:
        logger.info(message)


def _clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", str(name or "").strip())
    return name.strip(" ,;:.")


def _cardinality_value(name: str) -> int:
    cleaned = _clean_name(name)
    if not cleaned:
        return 0

    tokens = []
    for token in cleaned.split():
        normalised = token.strip("()[]{}.,;:").lower()
        if not normalised or normalised in _OPEN_NOM_TOKENS:
            continue
        tokens.append(normalised)

    if not tokens:
        return 0
    if len(tokens) == 1:
        return 1
    return 2


def _normalise_name_hits(names: list[str], source: str) -> list[dict]:
    seen: set[str] = set()
    results: list[dict] = []

    for raw_name in names:
        cleaned = _clean_name(raw_name)
        if not cleaned:
            continue

        cardinality = _cardinality_value(cleaned)
        if cardinality not in _CARDINALITY_LABELS:
            continue

        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)

        results.append(
            {
                "scientificName": cleaned,
                "verbatimName": cleaned,
                "cardinality": _CARDINALITY_LABELS[cardinality],
                "cardinalityValue": cardinality,
                "detectionSource": source,
            }
        )

    return results


def _normalise_finder_hits(raw_hits: list[dict], source: str) -> list[dict]:
    seen: set[str] = set()
    results: list[dict] = []

    for item in raw_hits:
        if not isinstance(item, dict):
            continue

        scientific_name = _clean_name(
            item.get("scientificName") or item.get("name") or item.get("verbatimName") or item.get("verbatim")
        )
        verbatim_name = _clean_name(item.get("verbatimName") or item.get("verbatim") or scientific_name)
        if not scientific_name:
            continue

        raw_cardinality = item.get("cardinality")
        if isinstance(raw_cardinality, str):
            lowered = raw_cardinality.strip().lower()
            if lowered == _GENUS:
                cardinality = 1
            elif lowered == _SPECIES:
                cardinality = 2
            else:
                cardinality = _cardinality_value(scientific_name)
        elif isinstance(raw_cardinality, (int, float)):
            cardinality = 1 if int(raw_cardinality) <= 1 else 2
        else:
            cardinality = _cardinality_value(scientific_name)

        if cardinality not in _CARDINALITY_LABELS:
            continue

        key = scientific_name.lower()
        if key in seen:
            continue
        seen.add(key)

        results.append(
            {
                "scientificName": scientific_name,
                "verbatimName": verbatim_name or scientific_name,
                "cardinality": _CARDINALITY_LABELS[cardinality],
                "cardinalityValue": cardinality,
                "detectionSource": source,
            }
        )

    return results


@lru_cache(maxsize=1)
def _get_biodiviz_pipeline():
    if not _BIODIVIZ_AVAILABLE or BiodiVizPipeline is None:
        return None
    try:
        return BiodiVizPipeline(ner_model_path="./ner_model", re_model_path="./re_model")
    except Exception as exc:
        logger.warning("[taxonomy] BiodiViz initialisation failed: %s", exc)
        return None


def detect_scientific_names(
    text: str,
    *,
    log_cb=None,
    ner_model_getter: Callable[[], object | None] | None = None,
) -> list[dict]:
    """
    Detect scientific names with a single shared fallback order.

    Order:
      1. Global Names Finder API
      2. NER model fallback when the API fails or returns nothing useful
      3. Local rule-based taxon extraction as the final safety net

    Returned records always use genus/species cardinality labels.
    """
    if not text or not text.strip():
        return []

    gnfinder_failed = False

    if _LEGACY_FINDER_AVAILABLE and _legacy_find_names is not None:
        try:
            raw_hits = _legacy_find_names(text[:8000])
            if raw_hits:
                normalised = _normalise_finder_hits(raw_hits, source="globalnames")
                if normalised:
                    _log(log_cb, f"[Taxonomy] Global Names detected {len(normalised)} genus/species names")
                    return normalised
        except Exception as exc:
            gnfinder_failed = True
            _log(log_cb, f"[Taxonomy] Global Names finder failed: {exc} — falling back to NER", "warn")

    pipeline = None
    if callable(ner_model_getter):
        try:
            pipeline = ner_model_getter()
        except Exception as exc:
            _log(log_cb, f"[Taxonomy] NER model getter failed: {exc}", "warn")
    elif gnfinder_failed or not _LEGACY_FINDER_AVAILABLE:
        pipeline = _get_biodiviz_pipeline()

    if pipeline is not None:
        try:
            result = pipeline.extract(text[:6000])
            names = result.get("organisms", []) if isinstance(result, dict) else []
            normalised = _normalise_name_hits(names, source="biodiviz_ner")
            if normalised:
                _log(log_cb, f"[Taxonomy] NER fallback detected {len(normalised)} genus/species names")
                return normalised
        except Exception as exc:
            _log(log_cb, f"[Taxonomy] BiodiViz NER fallback failed: {exc}", "warn")

    if _RULE_NER_AVAILABLE and _extract_taxa is not None:
        try:
            candidates = _extract_taxa(
                text[:6000],
                source_label="taxonomy_fallback",
                use_gna=False,
            )
            names = [
                getattr(candidate, "valid_name", "")
                or getattr(candidate, "canonical", "")
                or getattr(candidate, "verbatim", "")
                for candidate in candidates
            ]
            normalised = _normalise_name_hits(names, source="taxon_ner")
            if normalised:
                _log(log_cb, f"[Taxonomy] Rule-based fallback detected {len(normalised)} genus/species names")
                return normalised
        except Exception as exc:
            _log(log_cb, f"[Taxonomy] Rule-based fallback failed: {exc}", "warn")

    return []


def _sync_wiki(occurrences: list[dict], wiki, log_cb) -> None:
    if wiki is None or not hasattr(wiki, "update_species_article"):
        return

    updater = getattr(wiki, "update_species_article")
    try:
        param_names = list(inspect.signature(updater).parameters)
    except (TypeError, ValueError):
        param_names = []

    updated = 0
    for occurrence in occurrences:
        if not isinstance(occurrence, dict):
            continue

        name = (
            occurrence.get("validName")
            or ""
        ).strip()
        if not name or occurrence.get("taxonomicStatus") == "unverified":
            continue

        try:
            if "sp_name" in param_names or "species_name" in param_names:
                citation = (
                    occurrence.get("sourceCitation")
                    or occurrence.get("source_citation")
                    or occurrence.get("citation")
                    or ""
                )
                updater(name, occurrence, citation)
            else:
                updater(occurrence)
            updated += 1
        except Exception as exc:
            logger.debug("[taxonomy] Wiki sync skipped for %s: %s", name, exc)

    if updated:
        _log(log_cb, f"[Taxonomy] Wiki updated for {updated} verified records")


def verify_occurrences_with_fallback(
    occurrences: list[dict],
    *,
    log_cb=None,
    wiki=None,
    cache_db: str = "",
) -> list[dict]:
    """Verify occurrence names through the unified cascade, then fall back to legacy verification."""
    if not occurrences:
        return occurrences

    if _UNIFIED_VERIFY_AVAILABLE and UnifiedTaxonVerifier is not None:
        try:
            verifier = UnifiedTaxonVerifier(cache_db=cache_db)
            enriched = verifier.verify_and_enrich(occurrences, log_cb=log_cb)
            _sync_wiki(enriched, wiki, log_cb)
            return enriched
        except Exception as exc:
            _log(log_cb, f"[Taxonomy] Unified verifier failed: {exc} — falling back to legacy verifier", "warn")

    if _LEGACY_VERIFY_AVAILABLE and _legacy_verify_occurrences is not None:
        try:
            enriched = _legacy_verify_occurrences(occurrences)
            _sync_wiki(enriched, wiki, log_cb)
            return enriched
        except Exception as exc:
            _log(log_cb, f"[Taxonomy] Legacy verifier failed: {exc}", "warn")

    _log(log_cb, "[Taxonomy] No verifier available — keeping raw occurrences", "warn")
    return occurrences
