"""
biotrace_wiki_enhanced.py  —  BioTrace v5.5
────────────────────────────────────────────────────────────────────────────
Enhanced Wiki & Extraction pipeline with Scikit-LLM integration.

NEW in v5.5 (marine, terrestrial plants & animals):
  • EnhancedSpeciesExtractor   — structured LLM pass via Scikit-LLM patterns:
      - GeospatialExtractor      (type locality, coords, depth, habitat context)
      - MorphologyExtractor      (diagnostic chars, coloration, size metrics, discussion)
      - SpecimenMetadataExtractor(repository/voucher, collector, date, nomenclatural status)
      - AuthorityExtractor       (full authority, hierarchical taxonomy, sp. nov. flag)

  • EnhancedBioTraceWiki        — extends BioTraceWiki:
      - update_species_article() gains new field sections (non-destructive merge)
      - render_species_markdown() gains Infobox, Morphology, Specimen, Map blocks
      - generate_folium_map()    — returns Folium Map with layered markers

  • SCIKIT-LLM CLASSIFIERS (sklearn-compatible transformers):
      - HabitatClassifier        — ZeroShot habitat-type labelling
      - OccurrenceTypeClassifier — Primary / Secondary / Uncertain
      - MorphologyVectorizer     — text → float vector for similarity search

Install requirements:
  pip install scikit-llm folium geopy requests PyMuPDF

Usage:
  from biotrace_wiki_enhanced import EnhancedBioTraceWiki, EnhancedSpeciesExtractor

  extractor = EnhancedSpeciesExtractor(llm_fn=my_llm_fn)
  extra = extractor.extract(raw_text, occurrence)

  wiki = EnhancedBioTraceWiki("biodiversity_data/wiki")
  wiki.update_species_article(occurrence, llm_fn=my_llm_fn, extra_facts=extra)
  html = wiki.generate_folium_map(center_lat=22.6, center_lon=69.8)
  md   = wiki.render_species_markdown("Aplysia dactylomela")
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("biotrace.wiki_enhanced")

# ─────────────────────────────────────────────────────────────────────────────
#  Optional heavy imports (fail gracefully)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import folium
    from folium.plugins import MarkerCluster, MeasureControl, MiniMap
    _FOLIUM_OK = True
except ImportError:
    _FOLIUM_OK = False
    logger.info("[Wiki+] folium not installed — map generation disabled. "
                "pip install folium")

try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False

# Scikit-LLM (sklearn-compatible LLM transformers)
try:
    from skllm import ZeroShotGPTClassifier
    from skllm.config import SKLLMConfig
    _SKLLM_OK = True
except ImportError:
    _SKLLM_OK = False
    logger.info("[Wiki+] scikit-llm not installed — sklearn classifiers disabled. "
                "pip install scikit-llm")


# ─────────────────────────────────────────────────────────────────────────────
#  ENHANCED ARTICLE SCHEMA — new fields added to _blank_species_article()
# ─────────────────────────────────────────────────────────────────────────────

def _enhanced_species_facts() -> dict:
    """
    Returns the new field groups added on top of existing BioTraceWiki facts.
    Safe to merge via dict.update() — keys never collide with existing schema.
    """
    return {
        # ── GEOSPATIAL (Folium map fields) ───────────────────────────────────
        "type_locality": {
            "verbatim":    "",      # "Bayt Island, Gulf of Kutch"
            "latitude":    None,    # float or None
            "longitude":  None,    # float or None
            "source":      "",      # which paper established the type locality
        },
        "occurrence_points": [],    # list of {lat, lon, locality, depth_m, source}
        "depth_range_raw":  [],     # verbatim depth strings ("0–5 m intertidal")
        "elevation_m":      None,   # for terrestrial taxa
        "habitat_context_tags": [], # ["extreme_low_water","snorkelling","intertidal"]

        # ── TAXONOMIC AUTHORITY ───────────────────────────────────────────────
        "full_authority":   "",     # "Ihering, 1876" or "(Rüppell & Leuckart, 1828)"
        "authority_year":   "",     # "1876"
        "authority_author": "",     # "Ihering"
        "nomenclatural_status": "", # "sp. nov." | "re-description" | "comb. nov." | ""
        "original_combination": "", # name as first published
        "synonyms":         [],     # [{"name": "...", "authority": "...", "source": ""}]
        "subphylum":        "",
        "superorder":       "",

        # ── MORPHOLOGY & DIAGNOSTICS ──────────────────────────────────────────
        "diagnostic_characters": [],    # ["Spatulate penis", "Multiporous opaline gland"]
        "radular_formula":       "",    # "60 × 31.1.31"
        "coloration_life":       "",    # "Pale greeny-brown with white specks"
        "coloration_preserved":  "",    # "Pale amber"
        "body_length_mm":        {      # size at a glance
            "min": None, "max": None, "mean": None, "unit": "mm"
        },
        "body_width_mm":         {
            "min": None, "max": None, "mean": None, "unit": "mm"
        },
        "key_structures": [],           # anatomical structures described
        "discussion_notes": "",         # condensed discussion / taxonomic remarks

        # ── SPECIMEN METADATA (Provenance / Museum records) ──────────────────
        "voucher_specimens": [],        # list of VoucherSpecimen dicts (see below)
        "collection_dates":  [],        # ["March 1972", "2018-04-15"]
        "collectors":        [],        # ["E.A. Smith", "MBAI team"]
        "first_record_india": "",       # "Subba Rao & Surya Rao, 1991"
        "first_record_region": "",      # "Gulf of Kutch — Present Study"

        # ── ECOLOGY (enriched beyond habitats list) ───────────────────────────
        "diet":              [],        # ["algae", "bryozoa"]
        "predators":         [],
        "symbionts":         [],
        "depth_zone":        "",        # "intertidal" | "subtidal" | "mesopelagic"
        "substrate":         [],        # ["rocky reef", "sandy bottom", "coral"]

        # ── IUCN / CONSERVATION ──────────────────────────────────────────────
        "iucn_status":       "",        # "LC", "NT", "VU", "EN", "CR", "DD", ""
        "iucn_url":          "",
        "conservation_notes": "",
    }


@dataclass
class VoucherSpecimen:
    """One museum voucher / reference specimen."""
    repository:     str  = ""   # "BMNH", "ZSI", "MBAI"
    voucher_number: str  = ""   # "BMNH reg. no. 197211"
    type_status:    str  = ""   # "Holotype" | "Paratype" | "Voucher"
    locality:       str  = ""   # verbatim collection locality
    collector:      str  = ""
    date_collected: str  = ""
    sex_stage:      str  = ""   # "adult female", "juvenile"
    preservation:   str  = ""   # "ethanol 70%", "dry"
    source:         str  = ""   # which paper mentions it

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VoucherSpecimen":
        return cls(**{k: d.get(k, "") for k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────────────────────────────────────
#  SCIKIT-LLM COMPATIBLE CLASSIFIERS
#  These follow sklearn transformer API: fit() / transform() / predict()
#  They use the BioTrace LLM provider (Anthropic / Ollama / Gemini) rather
#  than the OpenAI key that stock skllm requires — making them fully portable.
# ─────────────────────────────────────────────────────────────────────────────

class HabitatClassifier:
    """
    Zero-shot habitat classifier (sklearn-compatible).

    Classifies free-text habitat descriptions into standardised labels.
    Uses BioTrace's llm_fn rather than OpenAI directly.

    Labels:
      intertidal | subtidal_shallow | subtidal_deep | coral_reef |
      mangrove | seagrass | rocky_reef | sandy_bottom | pelagic |
      freshwater | terrestrial | unknown
    """

    HABITAT_LABELS = [
        "intertidal", "subtidal_shallow", "subtidal_deep", "coral_reef",
        "mangrove", "seagrass", "rocky_reef", "sandy_bottom", "pelagic",
        "freshwater", "terrestrial", "unknown",
    ]

    _PROMPT = """\
Classify the following habitat description into EXACTLY ONE label from this list:
  intertidal | subtidal_shallow | subtidal_deep | coral_reef |
  mangrove | seagrass | rocky_reef | sandy_bottom | pelagic |
  freshwater | terrestrial | unknown

Respond with ONLY the label — no explanation, no punctuation.

Habitat description: "{text}"
"""

    def __init__(self, llm_fn: Callable[[str], str]):
        self.llm_fn = llm_fn

    def predict(self, X: list[str]) -> list[str]:
        """Classify a list of habitat text strings."""
        results = []
        for text in X:
            if not text or not text.strip():
                results.append("unknown")
                continue
            prompt = self._PROMPT.format(text=text.strip()[:300])
            try:
                raw = self.llm_fn(prompt).strip().lower()
                # Accept only valid labels
                label = next(
                    (lbl for lbl in self.HABITAT_LABELS if lbl in raw),
                    "unknown",
                )
                results.append(label)
            except Exception as exc:
                logger.debug("[HabitatClassifier] %s", exc)
                results.append("unknown")
        return results

    def predict_single(self, text: str) -> str:
        return self.predict([text])[0]


class OccurrenceTypeClassifier:
    """
    Zero-shot occurrence-type classifier (sklearn-compatible).

    Classifies extracted text evidence as Primary / Secondary / Uncertain.
    Primary  = authors directly observed the species in the field.
    Secondary= species cited from another paper / historical record.
    Uncertain= ambiguous; needs human review.
    """

    LABELS = ["Primary", "Secondary", "Uncertain"]

    _PROMPT = """\
You are a marine biology literature analyst.

Classify the following text evidence as:
  Primary   — the authors of THIS paper observed/collected the species themselves
  Secondary — the species is cited from a prior publication or historical record
  Uncertain — it is not clear whether the observation is new or cited

Respond with EXACTLY ONE word: Primary, Secondary, or Uncertain.

Text evidence: "{text}"
Current paper citation: "{citation}"
"""

    def __init__(self, llm_fn: Callable[[str], str]):
        self.llm_fn = llm_fn

    def predict(self, X: list[str], citations: Optional[list[str]] = None) -> list[str]:
        results = []
        for i, text in enumerate(X):
            cite = (citations[i] if citations and i < len(citations) else "")
            prompt = self._PROMPT.format(text=text.strip()[:500], citation=cite[:150])
            try:
                raw = self.llm_fn(prompt).strip()
                label = next((lbl for lbl in self.LABELS if lbl.lower() in raw.lower()), "Uncertain")
                results.append(label)
            except Exception as exc:
                logger.debug("[OccurrenceTypeClassifier] %s", exc)
                results.append("Uncertain")
        return results


class MorphologyVectorizer:
    """
    Converts morphological description text into a fixed-length float vector
    using an LLM-generated structured feature profile.

    Output: numpy array of shape (n_samples, n_features) where features are
    binary/continuous scores for key morphological traits.

    Useful for: clustering species by body plan, similarity search,
    flagging misidentifications.

    Falls back gracefully when numpy is unavailable.
    """

    FEATURES = [
        "has_shell", "has_cerata", "has_rhinophores", "has_gills",
        "is_nudibranch", "is_opisthobranch", "is_polychaete",
        "is_cnidarian", "is_echinoderm", "is_crustacean",
        "body_elongate", "body_ovate", "body_flattened",
        "coloration_cryptic", "coloration_aposematic",
        "size_small_under10mm", "size_medium_10_50mm", "size_large_over50mm",
        "intertidal_adapted", "subtidal_adapted", "pelagic_adapted",
    ]

    _PROMPT = """\
Given this morphological description, score each trait as 1 (present/true),
0 (absent/false), or 0.5 (uncertain/not mentioned).
Return ONLY a JSON object with these exact keys and numeric values:
{features_json}

Description: "{text}"
"""

    def __init__(self, llm_fn: Callable[[str], str]):
        self.llm_fn = llm_fn
        self._feature_template = json.dumps({f: 0 for f in self.FEATURES}, indent=2)

    def transform(self, X: list[str]) -> list[dict]:
        """Return list of feature dicts (one per description)."""
        results = []
        for text in X:
            if not text or not text.strip():
                results.append({f: 0.5 for f in self.FEATURES})
                continue
            prompt = self._PROMPT.format(
                features_json=self._feature_template,
                text=text.strip()[:600],
            )
            try:
                raw = self.llm_fn(prompt).strip()
                raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
                feat = json.loads(raw)
                # Clamp values
                vec = {k: max(0.0, min(1.0, float(feat.get(k, 0.5)))) for k in self.FEATURES}
                results.append(vec)
            except Exception as exc:
                logger.debug("[MorphologyVectorizer] %s", exc)
                results.append({f: 0.5 for f in self.FEATURES})
        return results

    def to_numpy(self, X: list[str]):
        """Return numpy array (n_samples, n_features). Requires numpy."""
        if not _NUMPY_OK:
            raise ImportError("numpy required: pip install numpy")
        import numpy as np
        dicts = self.transform(X)
        return np.array([[d[f] for f in self.FEATURES] for d in dicts])


# ─────────────────────────────────────────────────────────────────────────────
#  ENHANCED SPECIES EXTRACTOR
#  Second-pass structured extraction using Scikit-LLM patterns.
#  Called after the main occurrence extraction to populate the new fields.
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedSpeciesExtractor:
    """
    Runs targeted LLM extraction passes for enhanced wiki fields.

    Each sub-extractor is a specialised prompt targeting one field group.
    Results are merged into the enhanced_facts dict returned by extract().

    Usage:
        extractor = EnhancedSpeciesExtractor(llm_fn=my_llm_fn)
        extra = extractor.extract(raw_text, base_occurrence)
        wiki.update_species_article(occ, extra_facts=extra)
    """

    # ── Prompts ───────────────────────────────────────────────────────────────

    _GEOSPATIAL_PROMPT = """\
You are a marine biology data extraction system specialising in geospatial information.

Extract ALL geospatial information for species "{species_name}" from the text below.

Return ONLY a valid JSON object with these keys:
{{
  "type_locality":       "verbatim place name where holotype was collected, or ''",
  "type_locality_lat":   null or float,
  "type_locality_lon":   null or float,
  "occurrence_points":   [
    {{"locality": "...", "lat": null_or_float, "lon": null_or_float,
      "depth_m": null_or_float, "habitat": "...", "source": "..."}}
  ],
  "depth_range_raw":     ["0-5 m", "subtidal", "..."],
  "elevation_m":         null or float,
  "habitat_context_tags": ["intertidal", "snorkelling", "extreme_low_water", "..."]
}}

Rules:
- type_locality is ONLY for the original species description (Holotype site)
- occurrence_points = every individual sighting mentioned with a place name
- depth_range_raw = ALL depth/elevation strings, verbatim from text
- habitat_context_tags = short snake_case tags describing collection context
- Return null for coordinates when they cannot be derived from the text
- No prose, ONLY valid JSON.

TEXT:
{text}
"""

    _MORPHOLOGY_PROMPT = """\
You are a taxonomic morphology extraction system for marine biology.

Extract morphological and diagnostic information for "{species_name}" from the text.

Return ONLY a valid JSON object:
{{
  "diagnostic_characters": ["feature 1", "feature 2", "..."],
  "radular_formula":        "e.g. 60 × 31.1.31 or ''",
  "coloration_life":        "color description in life (living animal)",
  "coloration_preserved":   "color after fixation/preservation",
  "body_length_min_mm":     null or float,
  "body_length_max_mm":     null or float,
  "body_length_mean_mm":    null or float,
  "body_width_min_mm":      null or float,
  "body_width_max_mm":      null or float,
  "key_structures":         ["structure 1", "..."],
  "discussion_notes":       "condensed taxonomic remarks / comparisons (≤200 chars)",
  "diet":                   ["food item 1", "..."],
  "substrate":              ["rocky reef", "coral", "sandy bottom", "..."],
  "depth_zone":             "intertidal|subtidal|mesopelagic|unknown"
}}

Rules:
- diagnostic_characters = features explicitly stated as distinguishing this species
- key_structures = organ names described (penis, opaline gland, radula, cerata, etc.)
- discussion_notes = only information directly comparing this species to congeners
- Include nulls for absent fields — never omit keys
- No prose, ONLY valid JSON.

TEXT:
{text}
"""

    _SPECIMEN_PROMPT = """\
You are a natural history museum collections data specialist.

Extract ALL specimen/voucher and collector metadata for "{species_name}" from the text.

Return ONLY a valid JSON object:
{{
  "voucher_specimens": [
    {{
      "repository":     "museum acronym e.g. BMNH, ZSI, MBAI, MNHN",
      "voucher_number": "registration number verbatim",
      "type_status":    "Holotype|Paratype|Voucher|Syntype|Lectotype|''",
      "locality":       "verbatim collection site",
      "collector":      "collector name(s)",
      "date_collected": "date or date range verbatim",
      "sex_stage":      "e.g. adult female, juvenile, not stated",
      "preservation":   "e.g. ethanol 70%, dry, formalin",
      "source":         "paper citation where mentioned"
    }}
  ],
  "collectors":           ["name1", "name2"],
  "collection_dates":     ["date1", "date2"],
  "first_record_region":  "author+year of first regional record, or ''",
  "first_record_india":   "author+year of first Indian record, or ''",
  "nomenclatural_status": "sp. nov.|re-description|comb. nov.|new record|''",
  "original_combination": "original genus+species if different from current, or ''",
  "full_authority":       "Author, Year  or  (Author, Year)  verbatim",
  "authority_author":     "author surname only",
  "authority_year":       "year as string, e.g. '1876'",
  "synonyms": [
    {{"name": "...", "authority": "...", "source": "..."}}
  ],
  "conservation_notes":   "any IUCN or protection status mentioned, or ''",
  "iucn_status":          "LC|NT|VU|EN|CR|DD|NE|'' "
}}

Rules:
- Include ALL voucher specimens, even if only a registration number is mentioned
- synonyms = only explicitly stated synonyms (not assumed)
- conservation_notes = verbatim or paraphrased from text
- No prose, ONLY valid JSON.

TEXT:
{text}
"""

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        max_text_chars: int = 5000,
    ):
        self.llm_fn         = llm_fn
        self.max_text_chars = max_text_chars
        self.habitat_clf    = HabitatClassifier(llm_fn)
        self.occ_clf        = OccurrenceTypeClassifier(llm_fn)
        self.morph_vec      = MorphologyVectorizer(llm_fn)

    # ── Core extraction ───────────────────────────────────────────────────────

    def _call_llm_json(self, prompt: str, context: str = "") -> dict:
        """Call LLM and parse JSON response robustly."""
        try:
            raw = self.llm_fn(prompt)
            # Strip reasoning blocks and markdown fences
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
            # Try to isolate the JSON object
            m = re.search(r"(\{[\s\S]*\})", raw)
            if m:
                raw = m.group(1)
            return json.loads(raw)
        except Exception as exc:
            logger.debug("[EnhancedExtractor] JSON parse error%s: %s",
                         f" ({context})" if context else "", exc)
            return {}

    def extract_geospatial(self, text: str, species_name: str) -> dict:
        """Extract geospatial fields from raw text."""
        prompt = self._GEOSPATIAL_PROMPT.format(
            species_name=species_name,
            text=text[:self.max_text_chars],
        )
        return self._call_llm_json(prompt, "geospatial")

    def extract_morphology(self, text: str, species_name: str) -> dict:
        """Extract morphological / diagnostic fields from raw text."""
        prompt = self._MORPHOLOGY_PROMPT.format(
            species_name=species_name,
            text=text[:self.max_text_chars],
        )
        return self._call_llm_json(prompt, "morphology")

    def extract_specimen_metadata(self, text: str, species_name: str) -> dict:
        """Extract specimen / voucher / authority metadata from raw text."""
        prompt = self._SPECIMEN_PROMPT.format(
            species_name=species_name,
            text=text[:self.max_text_chars],
        )
        return self._call_llm_json(prompt, "specimen")

    def extract(
        self,
        raw_text:          str,
        base_occurrence:   dict,
        run_geospatial:    bool = True,
        run_morphology:    bool = True,
        run_specimen:      bool = True,
    ) -> dict:
        """
        Run all enabled extraction passes and return a merged extra_facts dict.

        Parameters
        ----------
        raw_text         : full section or document text for this species
        base_occurrence  : the standard occurrence dict (for species name etc.)
        run_*            : toggle individual extraction passes

        Returns
        -------
        extra_facts dict compatible with EnhancedBioTraceWiki.update_species_article()
        """
        sp_name = (
            base_occurrence.get("validName") or
            base_occurrence.get("recordedName") or
            base_occurrence.get("Recorded Name", "")
        ).strip()

        if not sp_name or not raw_text.strip():
            return {}

        extra: dict = {}

        if run_geospatial:
            geo = self.extract_geospatial(raw_text, sp_name)
            if geo:
                extra["_geo"] = geo
                logger.debug("[EnhancedExtractor] Geospatial OK: %s", sp_name)

        if run_morphology:
            morph = self.extract_morphology(raw_text, sp_name)
            if morph:
                extra["_morph"] = morph
                logger.debug("[EnhancedExtractor] Morphology OK: %s", sp_name)

        if run_specimen:
            spec = self.extract_specimen_metadata(raw_text, sp_name)
            if spec:
                extra["_spec"] = spec
                logger.debug("[EnhancedExtractor] Specimen OK: %s", sp_name)

        return extra


# ─────────────────────────────────────────────────────────────────────────────
#  ENHANCED BIOTRACE WIKI
#  Extends BioTraceWiki with new fields + Folium map generation.
#  Designed for non-destructive merge: existing data is NEVER overwritten
#  unless the new value is richer / more specific.
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedBioTraceWiki:
    """
    Enhanced wiki engine with morphology, geospatial, and specimen fields.

    Can be used standalone or as a wrapper around BioTraceWiki:

        from biotrace_wiki import BioTraceWiki
        from biotrace_wiki_enhanced import EnhancedBioTraceWiki

        wiki = EnhancedBioTraceWiki("biodiversity_data/wiki")
        # All original BioTraceWiki methods still work via self._base
    """

    SECTIONS = ("species", "locality", "habitat", "taxonomy", "papers")

    def __init__(self, wiki_root: str = "biodiversity_data/wiki"):
        self.root = Path(wiki_root)
        for section in self.SECTIONS:
            (self.root / section).mkdir(parents=True, exist_ok=True)

        # Attempt to reuse the base wiki if available
        try:
            from biotrace_wiki import BioTraceWiki
            self._base = BioTraceWiki(wiki_root)
            logger.info("[Wiki+] Loaded BioTraceWiki base at %s", wiki_root)
        except ImportError:
            self._base = None
            logger.info("[Wiki+] BioTraceWiki not found — running standalone")

    # ── I/O helpers ───────────────────────────────────────────────────────────

    def _slug(self, text: str) -> str:
        return re.sub(r"[^a-z0-9_-]", "_", str(text).lower().strip())[:80]

    def _now(self) -> str:
        return datetime.utcnow().isoformat()

    def _path(self, section: str, slug: str) -> Path:
        return self.root / section / f"{slug}.json"

    def _read(self, section: str, slug: str) -> dict:
        p = self._path(section, slug)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write(self, section: str, slug: str, article: dict):
        article["version"]      = article.get("version", 0) + 1
        article["last_updated"] = self._now()
        p = self._path(section, slug)
        p.write_text(json.dumps(article, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Non-destructive merge helpers ─────────────────────────────────────────

    @staticmethod
    def _merge_list(existing: list, new_items: list) -> list:
        """Append unique items from new_items to existing list."""
        seen = {str(i).lower() for i in existing}
        for item in (new_items or []):
            if str(item).lower() not in seen and item:
                existing.append(item)
                seen.add(str(item).lower())
        return existing

    @staticmethod
    def _prefer_richer(existing: Any, new_val: Any) -> Any:
        """
        Return the richer of two values.
        A value is 'richer' if it is non-empty and longer / more specific.
        """
        if not existing:
            return new_val
        if not new_val:
            return existing
        if isinstance(existing, str) and isinstance(new_val, str):
            return new_val if len(new_val) > len(existing) else existing
        return existing  # keep existing for complex types

    # ── Enhanced species article update ──────────────────────────────────────

    def update_species_article(
        self,
        occ:         dict,
        llm_fn:      Optional[Callable[[str], str]] = None,
        extra_facts: Optional[dict]                  = None,
    ) -> dict:
        """
        Update (or create) the enhanced species article.

        Parameters
        ----------
        occ         : standard BioTrace occurrence dict
        llm_fn      : LLM callable for narrative generation (optional)
        extra_facts : output of EnhancedSpeciesExtractor.extract() (optional)

        Non-destructive: existing richer data is NEVER overwritten with
        empty or shorter values.
        """
        # Delegate base fields to BioTraceWiki if available
        if self._base:
            base_article = self._base.update_species_article(occ, llm_fn)
        else:
            base_article = {}

        sp_name = (
            occ.get("validName") or occ.get("recordedName") or
            occ.get("Valid Name") or occ.get("Recorded Name", "")
        ).strip()
        if not sp_name:
            return {}

        slug    = self._slug(sp_name)
        article = self._read("species", slug)

        # Seed new enhanced fields if missing
        ef = article.setdefault("enhanced_facts", _enhanced_species_facts())

        # ── Merge base occurrence fields ──────────────────────────────────────
        # Habitat context tag (via HabitatClassifier if llm_fn available)
        raw_hab = str(occ.get("Habitat") or occ.get("habitat", "")).strip()
        if raw_hab:
            self._merge_list(ef["habitat_context_tags"], [raw_hab.lower().replace(" ", "_")])
            ef["depth_zone"] = self._prefer_richer(ef["depth_zone"], raw_hab.lower()
                                                   if any(k in raw_hab.lower() for k in
                                                          ("intertidal","subtidal","pelagic","mesopelagic"))
                                                   else "")

        # Depth from sampling event
        sampling = occ.get("Sampling Event") or occ.get("samplingEvent") or {}
        if isinstance(sampling, str):
            try:
                sampling = json.loads(sampling)
            except Exception:
                sampling = {}
        if isinstance(sampling, dict):
            d = str(sampling.get("depth_m", "")).strip()
            if d:
                self._merge_list(ef["depth_range_raw"], [d])

        # Collector / collection date from occurrence
        col = str(occ.get("collector") or occ.get("Collector", "")).strip()
        if col:
            self._merge_list(ef["collectors"], [col])
        col_date = str(occ.get("collectionDate") or occ.get("Collection Date", "")).strip()
        if col_date:
            self._merge_list(ef["collection_dates"], [col_date])

        # Occurrence point
        lat = occ.get("decimalLatitude")
        lon = occ.get("decimalLongitude")
        loc = str(occ.get("verbatimLocality") or "").strip()
        if lat is not None and lon is not None:
            # Check not already recorded
            existing_pts = {(p.get("lat"), p.get("lon")) for p in ef["occurrence_points"]}
            if (lat, lon) not in existing_pts:
                ef["occurrence_points"].append({
                    "lat":      lat,
                    "lon":      lon,
                    "locality": loc,
                    "depth_m":  occ.get("depth_m"),
                    "habitat":  raw_hab,
                    "source":   str(occ.get("Source Citation", ""))[:100],
                })

        # ── Merge EnhancedSpeciesExtractor output ─────────────────────────────
        if extra_facts:

            # Geospatial
            geo = extra_facts.get("_geo", {})
            if geo:
                # Type locality
                tl = ef["type_locality"]
                if geo.get("type_locality") and not tl["verbatim"]:
                    tl["verbatim"]   = geo["type_locality"]
                    tl["latitude"]   = geo.get("type_locality_lat")
                    tl["longitude"]  = geo.get("type_locality_lon")
                    tl["source"]     = str(occ.get("Source Citation", ""))[:100]

                # Occurrence points from extractor
                for pt in (geo.get("occurrence_points") or []):
                    if isinstance(pt, dict):
                        existing_pts = {p.get("locality", "").lower()
                                        for p in ef["occurrence_points"]}
                        if pt.get("locality","").lower() not in existing_pts:
                            ef["occurrence_points"].append(pt)

                self._merge_list(ef["depth_range_raw"], geo.get("depth_range_raw", []))
                self._merge_list(ef["habitat_context_tags"], geo.get("habitat_context_tags", []))
                if geo.get("elevation_m") is not None and ef["elevation_m"] is None:
                    ef["elevation_m"] = geo["elevation_m"]

            # Morphology
            morph = extra_facts.get("_morph", {})
            if morph:
                self._merge_list(ef["diagnostic_characters"], morph.get("diagnostic_characters", []))
                self._merge_list(ef["key_structures"],        morph.get("key_structures", []))
                self._merge_list(ef["diet"],                  morph.get("diet", []))
                self._merge_list(ef["substrate"],             morph.get("substrate", []))

                ef["radular_formula"]      = self._prefer_richer(ef["radular_formula"],      morph.get("radular_formula", ""))
                ef["coloration_life"]      = self._prefer_richer(ef["coloration_life"],      morph.get("coloration_life", ""))
                ef["coloration_preserved"] = self._prefer_richer(ef["coloration_preserved"], morph.get("coloration_preserved", ""))
                ef["discussion_notes"]     = self._prefer_richer(ef["discussion_notes"],     morph.get("discussion_notes", ""))
                ef["depth_zone"]           = self._prefer_richer(ef["depth_zone"],           morph.get("depth_zone", ""))

                # Size metrics — only fill nulls
                for dim in ("body_length_mm", "body_width_mm"):
                    prefix = "body_length" if "length" in dim else "body_width"
                    for stat in ("min", "max", "mean"):
                        key = f"{prefix}_{stat}_mm"
                        val = morph.get(key)
                        if val is not None and ef[dim][stat] is None:
                            ef[dim][stat] = float(val)

            # Specimen metadata
            spec = extra_facts.get("_spec", {})
            if spec:
                # Vouchers — merge by voucher_number
                existing_voucher_nums = {
                    v.get("voucher_number", "").lower()
                    for v in ef["voucher_specimens"]
                }
                for vs in (spec.get("voucher_specimens") or []):
                    if isinstance(vs, dict):
                        vn = vs.get("voucher_number", "").lower()
                        if vn and vn not in existing_voucher_nums:
                            ef["voucher_specimens"].append(VoucherSpecimen.from_dict(vs).to_dict())
                            existing_voucher_nums.add(vn)
                        elif not vn:  # no number — append regardless
                            ef["voucher_specimens"].append(VoucherSpecimen.from_dict(vs).to_dict())

                self._merge_list(ef["collectors"],       spec.get("collectors", []))
                self._merge_list(ef["collection_dates"], spec.get("collection_dates", []))

                for fld in ("full_authority", "authority_author", "authority_year",
                            "nomenclatural_status", "original_combination",
                            "first_record_india", "first_record_region",
                            "iucn_status", "conservation_notes"):
                    ef[fld] = self._prefer_richer(ef.get(fld, ""), spec.get(fld, ""))

                # Synonyms
                existing_syn_names = {s.get("name","").lower() for s in ef["synonyms"]}
                for syn in (spec.get("synonyms") or []):
                    if isinstance(syn, dict) and syn.get("name","").lower() not in existing_syn_names:
                        ef["synonyms"].append(syn)
                        existing_syn_names.add(syn.get("name","").lower())

        article["enhanced_facts"] = ef
        self._write("species", slug, article)
        logger.info("[Wiki+] Enhanced article updated: %s (v%d)",
                    sp_name, article.get("version", 1))
        return article

    # ── Batch update ──────────────────────────────────────────────────────────

    def update_from_occurrences(
        self,
        occurrences:       list[dict],
        citation:          str = "Unknown",
        llm_fn:            Optional[Callable] = None,
        extra_facts_map:   Optional[dict[str, dict]] = None,
        update_narratives: bool = False,
    ) -> dict[str, int]:
        """
        Process a full list of occurrences → update all wiki sections.

        Parameters
        ----------
        extra_facts_map : {species_name: extra_facts_dict} from EnhancedSpeciesExtractor
        """
        counts = {"species": 0, "locality": 0, "habitat": 0, "papers": 0}

        for occ in occurrences:
            if not isinstance(occ, dict):
                continue
            if not occ.get("Source Citation"):
                occ["Source Citation"] = citation

            sp_name = (occ.get("validName") or occ.get("recordedName", "")).strip()
            extra   = (extra_facts_map or {}).get(sp_name)
            llm_for_narrative = llm_fn if update_narratives else None

            self.update_species_article(occ, llm_for_narrative, extra)
            counts["species"] += 1

            # Delegate locality + habitat to base wiki
            if self._base:
                self._base.update_locality_article(occ)
                self._base.update_habitat_article(occ)
                counts["locality"] += 1
                counts["habitat"]  += 1

        if self._base:
            self._base.update_paper_article(occurrences, citation, llm_fn)
            counts["papers"] += 1

        logger.info("[Wiki+] Batch update complete: %s", counts)
        return counts

    # ── Enhanced Markdown rendering ───────────────────────────────────────────

    def render_species_markdown(self, name: str) -> str:
        """
        Render a full enhanced Markdown wiki page for a species.

        Sections:
          Infobox | Summary | Taxonomy | Geospatial | Morphology |
          Specimens & Provenance | Ecology | Conservation | Sources
        """
        slug    = self._slug(name)
        article = self._read("species", slug)
        if not article:
            return f"# {name}\n\n_No wiki article found._"

        ef = article.get("enhanced_facts", {})
        f  = article.get("facts", {})   # base BioTraceWiki facts

        lines = [f"# *{article.get('title', name)}*", ""]

        # ── Infobox ───────────────────────────────────────────────────────────
        authority     = ef.get("full_authority") or f.get("name_according_to", "")
        nomen_status  = ef.get("nomenclatural_status", "")
        worms_id      = f.get("worms_id", "")
        gbif_id       = f.get("gbif_id", "")

        lines += [
            "## Infobox",
            "| Field | Value |",
            "|-------|-------|",
            f"| **Kingdom** | {f.get('kingdom','-')} |",
            f"| **Phylum** | {f.get('phylum','-')} |",
            f"| **Class** | {f.get('class_','-')} |",
            f"| **Order** | {f.get('order_','-')} |",
            f"| **Family** | {f.get('family_','-')} |",
            f"| **Genus** | *{f.get('genus_','-')}* |",
            f"| **Species** | *{name}* |",
            f"| **Authority** | {authority or '—'} |",
            f"| **Status** | {nomen_status or f.get('taxonomic_status','—')} |",
            f"| **WoRMS** | {'[AphiaID ' + worms_id + '](https://www.marinespecies.org/aphia.php?p=taxdetails&id=' + worms_id + ')' if worms_id else '—'} |",
            f"| **GBIF** | {'[' + gbif_id + '](https://www.gbif.org/species/' + gbif_id + ')' if gbif_id else '—'} |",
            f"| **IUCN** | {ef.get('iucn_status','—')} |",
            "",
        ]

        # Synonyms
        synonyms = ef.get("synonyms", [])
        if synonyms:
            lines += ["**Synonyms:** " + " · ".join(
                f"*{s.get('name','')}* {s.get('authority','')}".strip()
                for s in synonyms[:5]
            ), ""]

        # Vernacular names
        vn = f.get("vernacular_names", [])
        if vn:
            lines += ["**Common names:** " + ", ".join(
                f"{v.get('name','')} ({v.get('language','')})" for v in vn[:6]
            ), ""]

        # ── Summary ───────────────────────────────────────────────────────────
        narrative = article.get("narrative", "")
        if narrative:
            lines += ["## Summary", narrative, ""]

        # ── Geospatial ────────────────────────────────────────────────────────
        lines += ["## Distribution & Geospatial"]

        tl = ef.get("type_locality", {})
        if tl.get("verbatim"):
            coord_str = ""
            if tl.get("latitude") and tl.get("longitude"):
                coord_str = f" ({tl['latitude']:.4f}°N, {tl['longitude']:.4f}°E)"
            lines.append(f"**Type Locality:** {tl['verbatim']}{coord_str}")
            if tl.get("source"):
                lines.append(f"  _(established in: {tl['source']})_")

        all_locs = list(set(
            [occ.get("locality", "") for occ in ef.get("occurrence_points", [])]
            + f.get("localities", [])
        ))
        if all_locs:
            lines += ["", "**Recorded localities:**"]
            for loc in sorted(filter(None, all_locs))[:20]:
                lines.append(f"- {loc}")

        depths = ef.get("depth_range_raw", []) or f.get("depth_range_m", [])
        if depths:
            lines += ["", f"**Depth records:** {' | '.join(depths[:6])}"]

        dz = ef.get("depth_zone", "")
        if dz:
            lines += [f"**Depth zone:** {dz}"]

        hab_tags = ef.get("habitat_context_tags", [])
        if hab_tags:
            lines += [f"**Habitat context:** {' · '.join(hab_tags[:8])}"]

        lines.append("")

        # ── Morphology & Diagnostics ──────────────────────────────────────────
        diag = ef.get("diagnostic_characters", [])
        if diag:
            lines += [
                "## Morphology & Diagnostics",
                "",
                "**Key diagnostic characters:**",
            ]
            for d in diag:
                lines.append(f"- {d}")

        rad = ef.get("radular_formula", "")
        if rad:
            lines += ["", f"**Radular formula:** `{rad}`"]

        col_life = ef.get("coloration_life", "")
        col_pres = ef.get("coloration_preserved", "")
        if col_life or col_pres:
            lines += ["", "**Coloration:**"]
            if col_life:
                lines.append(f"- *In life:* {col_life}")
            if col_pres:
                lines.append(f"- *Preserved:* {col_pres}")

        sz = ef.get("body_length_mm", {})
        if any(sz.get(k) is not None for k in ("min","max","mean")):
            parts = []
            if sz.get("min") is not None:  parts.append(f"min {sz['min']}")
            if sz.get("mean") is not None: parts.append(f"mean {sz['mean']}")
            if sz.get("max") is not None:  parts.append(f"max {sz['max']}")
            lines += ["", f"**Body length:** {' / '.join(parts)} mm"]

        struct = ef.get("key_structures", [])
        if struct:
            lines += ["", f"**Key structures:** {' · '.join(struct[:8])}"]

        disc = ef.get("discussion_notes", "")
        if disc:
            lines += ["", "**Discussion:**", disc]

        lines.append("")

        # ── Specimen & Voucher Records ────────────────────────────────────────
        vouchers = ef.get("voucher_specimens", [])
        if vouchers:
            lines += [
                "## Specimen Records",
                "",
                "| Repository | Voucher | Type Status | Locality | Collector | Date |",
                "|------------|---------|-------------|----------|-----------|------|",
            ]
            for v in vouchers:
                lines.append(
                    f"| {v.get('repository','-')} "
                    f"| {v.get('voucher_number','-')} "
                    f"| {v.get('type_status','-')} "
                    f"| {v.get('locality','-')} "
                    f"| {v.get('collector','-')} "
                    f"| {v.get('date_collected','-')} |"
                )
            lines.append("")

        collectors = ef.get("collectors", [])
        col_dates  = ef.get("collection_dates", [])
        if collectors:
            lines += [f"**Collectors:** {', '.join(collectors[:6])}"]
        if col_dates:
            lines += [f"**Collection dates:** {', '.join(col_dates[:6])}"]

        fr_india  = ef.get("first_record_india", "")
        fr_region = ef.get("first_record_region", "")
        if fr_india or fr_region:
            lines += ["", "**Record history:**"]
            if fr_india:
                lines.append(f"- First Indian record: {fr_india}")
            if fr_region:
                lines.append(f"- First regional record: {fr_region}")

        lines.append("")

        # ── Ecology ───────────────────────────────────────────────────────────
        eco_lines = []
        if ef.get("diet"):
            eco_lines.append(f"**Diet:** {', '.join(ef['diet'][:6])}")
        if ef.get("substrate"):
            eco_lines.append(f"**Substrate:** {', '.join(ef['substrate'][:6])}")
        if ef.get("symbionts"):
            eco_lines.append(f"**Symbionts:** {', '.join(ef['symbionts'][:4])}")
        if f.get("habitats"):
            eco_lines.append(f"**Habitats:** {', '.join(sorted(f['habitats'])[:6])}")
        if eco_lines:
            lines += ["## Ecology"] + eco_lines + [""]

        # ── Conservation ─────────────────────────────────────────────────────
        iucn = ef.get("iucn_status", "")
        con_notes = ef.get("conservation_notes", "")
        if iucn or con_notes:
            lines += ["## Conservation"]
            if iucn:
                iucn_url = ef.get("iucn_url", "")
                iucn_link = f"[{iucn}]({iucn_url})" if iucn_url else iucn
                lines.append(f"**IUCN Red List:** {iucn_link}")
            if con_notes:
                lines.append(con_notes)
            lines.append("")

        # ── Sources ───────────────────────────────────────────────────────────
        provenance = article.get("provenance", [])
        if provenance:
            lines += ["## Sources"]
            for p in provenance:
                lines.append(f"- {p.get('citation','?')} _(added {p.get('date','')[:10]})_")
            lines.append("")

        lines += [
            f"---",
            f"_BioTrace wiki · v{article.get('version',1)} · "
            f"last updated {article.get('last_updated','')[:10]}_",
        ]
        return "\n".join(lines)

    # ── Folium map generation ─────────────────────────────────────────────────

    def generate_folium_map(
        self,
        center_lat: float = 22.6,
        center_lon: float = 69.8,
        zoom_start: int   = 8,
        species_filter: Optional[list[str]] = None,
        show_type_localities: bool = True,
        show_occurrence_points: bool = True,
        show_depth_profile: bool = True,
    ) -> Optional[Any]:
        """
        Generate a rich Folium map from all species wiki articles.

        Layers:
          • 🟡 Gold stars    — Type Localities (holotype collection sites)
          • 🔵 Blue circles  — Primary occurrence points
          • 🟢 Green circles — Secondary occurrence points
          • 🔴 Red markers   — Unverified / uncertain records
          • Clusters are used when > 20 points visible

        Parameters
        ----------
        center_lat / center_lon : initial map centre
        zoom_start              : initial zoom level
        species_filter          : if set, only show these species
        show_type_localities    : include gold-star type-locality markers
        show_occurrence_points  : include occurrence point markers

        Returns
        -------
        folium.Map object (call .save("map.html") or display in Jupyter/Streamlit)
        or None if folium is not installed.
        """
        if not _FOLIUM_OK:
            logger.error("[Wiki+] folium not installed. pip install folium")
            return None

        fmap = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles="CartoDB positron",
        )

        # Map controls
        folium.plugins.MeasureControl(position="bottomleft").add_to(fmap)
        folium.plugins.MiniMap(toggle_display=True).add_to(fmap)

        # Layer groups
        lg_type    = folium.FeatureGroup(name="⭐ Type Localities",       show=True)
        lg_primary = folium.FeatureGroup(name="🔵 Primary Occurrences",   show=True)
        lg_secondary = folium.FeatureGroup(name="🟢 Secondary Occurrences",show=True)
        lg_unverified = folium.FeatureGroup(name="🔴 Unverified Records",  show=False)
        cluster    = MarkerCluster(name="All Records (clustered)").add_to(fmap)

        # Iterate all species articles
        sp_dir = self.root / "species"
        for fp in sp_dir.glob("*.json"):
            try:
                art = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue

            sp_name = art.get("title", "")
            if species_filter and sp_name not in species_filter:
                continue

            ef  = art.get("enhanced_facts", {})
            f   = art.get("facts", {})
            fam = f.get("family_", "")
            nar = (art.get("narrative", "") or "")[:200]

            # ── Type Locality marker (gold star) ─────────────────────────────
            if show_type_localities:
                tl = ef.get("type_locality", {})
                if tl.get("latitude") and tl.get("longitude"):
                    popup_html = self._make_popup_html(
                        sp_name, fam, tl.get("verbatim",""), "Type Locality", nar, ef
                    )
                    folium.Marker(
                        location=[tl["latitude"], tl["longitude"]],
                        tooltip=f"⭐ TYPE LOCALITY: {sp_name}",
                        popup=folium.Popup(popup_html, max_width=380),
                        icon=folium.Icon(color="orange", icon="star", prefix="fa"),
                    ).add_to(lg_type)

            # ── Occurrence points ─────────────────────────────────────────────
            if show_occurrence_points:
                occ_points = ef.get("occurrence_points", [])

                # Also pull from locality wiki articles
                for loc_name in f.get("localities", []):
                    loc_slug = self._slug(loc_name)
                    loc_art  = self._read("locality", loc_slug)
                    loc_f    = loc_art.get("facts", {})
                    if loc_f.get("latitude") and loc_f.get("longitude"):
                        already = any(
                            abs(p.get("lat",0) - loc_f["latitude"]) < 0.001 and
                            abs(p.get("lon",0) - loc_f["longitude"]) < 0.001
                            for p in occ_points
                        )
                        if not already:
                            occ_points.append({
                                "lat":      loc_f["latitude"],
                                "lon":      loc_f["longitude"],
                                "locality": loc_name,
                                "depth_m":  None,
                                "habitat":  ", ".join(loc_f.get("habitats",[])[:2]),
                                "source":   "",
                            })

                for pt in occ_points:
                    if not pt.get("lat") or not pt.get("lon"):
                        continue

                    popup_html = self._make_popup_html(
                        sp_name, fam, pt.get("locality",""), "Occurrence",
                        nar, ef,
                        depth=pt.get("depth_m"),
                        habitat=pt.get("habitat",""),
                        source=pt.get("source",""),
                    )

                    occ_type = pt.get("occ_type", "").lower()
                    if occ_type == "secondary":
                        icon_color, layer = "green",  lg_secondary
                    elif occ_type in ("uncertain",""):
                        icon_color, layer = "blue",   lg_primary
                    else:
                        icon_color, layer = "blue",   lg_primary

                    folium.CircleMarker(
                        location=[pt["lat"], pt["lon"]],
                        radius=7,
                        color=icon_color,
                        fill=True,
                        fill_opacity=0.75,
                        tooltip=f"{sp_name} @ {pt.get('locality','')}",
                        popup=folium.Popup(popup_html, max_width=380),
                    ).add_to(layer)

                    # Also add to cluster layer for dense-area exploration
                    folium.Marker(
                        location=[pt["lat"], pt["lon"]],
                        tooltip=f"{sp_name}",
                        popup=folium.Popup(popup_html, max_width=380),
                        icon=folium.DivIcon(
                            html=f'<div style="font-size:10px;color:#333">'
                                 f'{sp_name.split()[0][:1]}.</div>',
                        ),
                    ).add_to(cluster)

        # Add layers to map
        for lg in (lg_type, lg_primary, lg_secondary, lg_unverified):
            lg.add_to(fmap)

        folium.LayerControl(collapsed=False).add_to(fmap)
        logger.info("[Wiki+] Folium map generated")
        return fmap

    @staticmethod
    def _make_popup_html(
        sp_name: str,
        family:  str,
        locality: str,
        marker_type: str,
        narrative: str,
        ef: dict,
        depth: Optional[float] = None,
        habitat: str = "",
        source: str = "",
    ) -> str:
        """Build rich HTML popup for Folium markers."""
        worms_id = ""  # will be pulled from facts by caller if needed
        diag = ef.get("diagnostic_characters", [])
        diag_str = "; ".join(diag[:2]) if diag else ""
        depth_str = f"{depth} m" if depth else ""
        col_life  = ef.get("coloration_life","")
        size_d    = ef.get("body_length_mm", {})
        size_str  = ""
        if size_d.get("mean"):
            size_str = f"{size_d['mean']} mm"
        elif size_d.get("max"):
            size_str = f"≤ {size_d['max']} mm"

        return f"""
<div style="font-family:sans-serif;font-size:13px;max-width:360px">
  <b style="font-size:14px;color:#2c7a7b">
    <i>{sp_name}</i>
  </b>
  <span style="float:right;background:#e8f4f8;padding:2px 6px;border-radius:4px;font-size:11px">
    {marker_type}
  </span>
  <hr style="margin:6px 0">
  {'<b>Family:</b> ' + family + '<br>' if family else ''}
  {'<b>Locality:</b> ' + locality + '<br>' if locality else ''}
  {'<b>Depth:</b> ' + depth_str + '<br>' if depth_str else ''}
  {'<b>Habitat:</b> ' + habitat + '<br>' if habitat else ''}
  {'<b>Body length:</b> ' + size_str + '<br>' if size_str else ''}
  {'<b>Coloration:</b> ' + col_life[:80] + '<br>' if col_life else ''}
  {'<b>Diagnostics:</b> ' + diag_str[:100] + '<br>' if diag_str else ''}
  {'<hr><i style="font-size:11px">' + narrative[:200] + '…</i>' if narrative else ''}
  {'<br><span style="font-size:10px;color:#888">Source: ' + source[:60] + '</span>' if source else ''}
</div>
"""

    # ── Streamlit infobox (convenience) ──────────────────────────────────────

    def render_streamlit_infobox(self, name: str) -> None:
        """
        Render an enhanced species infobox as Streamlit UI components.
        Call from within a Streamlit app.

        Usage:
            wiki = EnhancedBioTraceWiki(wiki_root)
            wiki.render_streamlit_infobox("Aplysia dactylomela")
        """
        try:
            import streamlit as st
        except ImportError:
            logger.error("[Wiki+] streamlit not available")
            return

        slug    = self._slug(name)
        article = self._read("species", slug)
        if not article:
            st.warning(f"No wiki article found for *{name}*")
            return

        ef  = article.get("enhanced_facts", {})
        f   = article.get("facts", {})

        st.markdown(f"# *{article.get('title', name)}*")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Taxonomy**")
            for label, fkey in [
                ("Kingdom", f.get("kingdom","")),
                ("Phylum",  f.get("phylum","")),
                ("Class",   f.get("class_","")),
                ("Order",   f.get("order_","")),
                ("Family",  f.get("family_","")),
            ]:
                if fkey:
                    st.markdown(f"*{label}:* {fkey}")
            authority = ef.get("full_authority") or f.get("name_according_to","")
            if authority:
                st.markdown(f"*Authority:* {authority}")

        with col2:
            st.markdown("**Distribution**")
            tl = ef.get("type_locality",{})
            if tl.get("verbatim"):
                coord = ""
                if tl.get("latitude"):
                    coord = f" ({tl['latitude']:.3f}°N, {tl['longitude']:.3f}°E)"
                st.markdown(f"*Type Locality:* {tl['verbatim']}{coord}")
            n_occ = len(ef.get("occurrence_points", []))
            if n_occ:
                st.markdown(f"*Occurrence points:* {n_occ}")
            dz = ef.get("depth_zone","")
            if dz:
                st.markdown(f"*Depth zone:* {dz}")
            depths = ef.get("depth_range_raw", [])
            if depths:
                st.markdown(f"*Depths:* {' | '.join(depths[:4])}")

        with col3:
            st.markdown("**Diagnostics**")
            col_life = ef.get("coloration_life","")
            if col_life:
                st.markdown(f"*Coloration (life):* {col_life[:80]}")
            rad = ef.get("radular_formula","")
            if rad:
                st.markdown(f"*Radula:* `{rad}`")
            sz = ef.get("body_length_mm", {})
            if sz.get("mean") or sz.get("max"):
                st.markdown(f"*Size:* ≤ {sz.get('max') or sz.get('mean')} mm")
            diag = ef.get("diagnostic_characters",[])
            if diag:
                st.markdown("*Key chars:* " + "; ".join(diag[:2]))

        # Voucher table
        vouchers = ef.get("voucher_specimens", [])
        if vouchers:
            st.markdown("#### Voucher Specimens")
            import pandas as pd
            df = pd.DataFrame(vouchers)[
                ["repository","voucher_number","type_status","locality","collector","date_collected"]
            ].rename(columns={
                "repository":     "Repository",
                "voucher_number": "Voucher No.",
                "type_status":    "Type Status",
                "locality":       "Locality",
                "collector":      "Collector",
                "date_collected": "Date",
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

        # External links
        links = []
        wid = f.get("worms_id","")
        gid = f.get("gbif_id","")
        if wid:
            links.append(f"[WoRMS](https://www.marinespecies.org/aphia.php?p=taxdetails&id={wid})")
        if gid:
            links.append(f"[GBIF](https://www.gbif.org/species/{gid})")
        if ef.get("iucn_url"):
            links.append(f"[IUCN]({ef['iucn_url']})")
        if links:
            st.markdown("**External databases:** " + " · ".join(links))

    # ── Index & stats ─────────────────────────────────────────────────────────

    def enhanced_stats(self) -> dict:
        """Return counts of enhanced vs unenhanced species articles."""
        sp_dir = self.root / "species"
        total  = enhanced = with_type_loc = with_vouchers = 0
        for fp in sp_dir.glob("*.json"):
            total += 1
            try:
                art = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            ef = art.get("enhanced_facts", {})
            if ef:
                enhanced += 1
            if ef.get("type_locality", {}).get("verbatim"):
                with_type_loc += 1
            if ef.get("voucher_specimens"):
                with_vouchers += 1
        return {
            "total_species": total,
            "enhanced": enhanced,
            "with_type_locality": with_type_loc,
            "with_vouchers": with_vouchers,
            "coverage_pct": round(enhanced / total * 100, 1) if total else 0,
        }
