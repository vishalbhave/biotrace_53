"""
biotrace_traiter_prepass.py — BioTrace v5.4
─────────────────────────────────────────────
spaCy rule-based annotation of raw text BEFORE LLM extraction.

Outputs a span_annotations dict and an augmented text that the LLM
extraction prompt can reference directly. This mirrors Traiter's
layered-pattern approach:
  Layer 1: Phrase matchers for known terms (taxa, localities, measurements)
  Layer 2: Rule patterns built from Layer 1 terms
  Layer 3: Entity linking (taxon ← measurement ← location)

Benefit: The LLM validates annotated spans rather than guessing from scratch,
reducing hallucination of morphology-as-locality and life-stage-as-taxon.
"""
from __future__ import annotations
import re, logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("biotrace.prepass")

# ── Span types ────────────────────────────────────────────────────────────────
@dataclass
class Span:
    start: int
    end: int
    label: str          # TAXON | LOCALITY | MEASUREMENT | DATE | HABITAT | LIFESTAGE
    text: str
    confidence: float = 1.0

@dataclass
class PrePassResult:
    spans: list[Span] = field(default_factory=list)
    taxa: list[str] = field(default_factory=list)        # candidate species strings
    localities: list[str] = field(default_factory=list)  # candidate place strings
    dates: list[str] = field(default_factory=list)
    measurements: list[str] = field(default_factory=list)
    habitats: list[str] = field(default_factory=list)
    lifestages: list[str] = field(default_factory=list)

# ── Measurement regex (Traiter-style: numeric + unit) ─────────────────────────
_MEAS_RE = re.compile(
    r"\b\d+[\d\s.–\-]*\s*"
    r"(cm|mm|m\b|km|in\b|µm|um|mg|g\b|kg|ml|l\b|%|ppt|ppm|°[CF]"
    r"|individuals?|specimens?|colonies)",
    re.IGNORECASE,
)

# ── Date regex ────────────────────────────────────────────────────────────────
_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{4}\b"
    r"|\b\d{4}[–\-]\d{2,4}\b"
    r"|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b",
    re.IGNORECASE,
)

# ── Binomial taxon regex ───────────────────────────────────────────────────────
# Capitalised genus + lowercase epithet (+ optional authority year)
_BINOMIAL_RE = re.compile(
    r"\b([A-Z][a-z]{2,})\s+([a-z]{2,})"
    r"(?:\s+(?:var\.|subsp\.|f\.)\s+[a-z]+)?"
    r"(?:\s+\([^)]*\d{4}[^)]*\))?",
)

# ── Habitat terms (from biotrace_dedup_patch extended) ───────────────────────
# _HABITAT_TERMS = {
#     "intertidal", "subtidal", "coral reef", "dead coral", "seagrass",
#     "mangrove", "lagoon", "estuary", "rocky shore", "sandy bottom",
#     "mudflat", "reef flat", "reef slope", "pelagic", "benthic",
#     "littoral", "supralittoral", "tide pool", "rock pool",
# }

_HABITAT_TERMS = {
    # --- Coastal & Supratidal ---
    "sea cliff", "offshore island", "coastal dune", "coastal cave", "karst",
    "salt pan", "shingle beach", "pebble shoreline", "strandline",
    
    # --- Marine Neritic (Continental Shelf) ---
    "neritic zone", "subtidal rock", "rocky reef", "subtidal sandy-mud",
    "macroalgal forest", "kelp forest", "maerl bed", "shellfish bed",
    "oyster reef", "bivalve reef", "eelgrass bed", "submerged aquatic vegetation",
    
    # --- Oceanic & Deep Sea Benthic ---
    "continental slope", "continental rise", "bathyal zone", "abyssal plain",
    "abyssal hill", "oceanic trench", "hadal zone", "seamount", "submarine canyon",
    "hydrothermal vent", "cold seep", "brine pool", "mid-ocean ridge",
    
    # --- Pelagic Vertical Zonation ---
    "epipelagic zone", "mesopelagic zone", "bathypelagic zone", 
    "abyssopelagic zone", "hadalpelagic zone", "photic zone", "aphotic zone",
    
    # --- Polar & Ice-Associated ---
    "sea ice ecosystem", "ice shelf", "polynya", "marine ice body",
    
    # --- Specialized Biotopes & Features ---
    "eulittoral rock", "infralittoral rock", "circalittoral rock", 
    "biogenic reef", "coralligenous platform", "surge gully", 
    "upwelling region", "surface microlayer", "anoxic mud",
    
    # --- Anthropogenic (Artificial) ---
    "artificial reef", "shipwreck", "mariculture cage", "aquaculture pond",
    "marine anthropogenic structure"
}

# ── Life-stage terms (from biotrace_dedup_patch) ──────────────────────────────
from biotrace_dedup_patch import LIFE_STAGE_TERMS  # reuse existing list


def run_prepass(text: str) -> PrePassResult:
    """
    Run the rule-based pre-pass over a text chunk.
    Returns a PrePassResult with all detected spans and extracted lists.

    Wire into biotrace_v5.py _process_batch_text() BEFORE the LLM call:

        from biotrace_traiter_prepass import run_prepass, format_annotations_for_prompt
        pre = run_prepass(text)
        annotation_block = format_annotations_for_prompt(pre)
        # prepend annotation_block to the text sent to LLM
    """
    result = PrePassResult()

    # ── Layer 1a: Binomial taxa ───────────────────────────────────────────────
    for m in _BINOMIAL_RE.finditer(text):
        genus, epithet = m.group(1), m.group(2)
        # Exclude life-stage words that happen to be capitalised
        if epithet.lower() not in LIFE_STAGE_TERMS and genus.lower() not in LIFE_STAGE_TERMS:
            result.taxa.append(f"{genus} {epithet}")
            result.spans.append(Span(m.start(), m.end(), "TAXON", m.group()))

    # ── Layer 1b: Measurements ────────────────────────────────────────────────
    for m in _MEAS_RE.finditer(text):
        result.measurements.append(m.group())
        result.spans.append(Span(m.start(), m.end(), "MEASUREMENT", m.group()))

    # ── Layer 1c: Dates ───────────────────────────────────────────────────────
    for m in _DATE_RE.finditer(text):
        result.dates.append(m.group())
        result.spans.append(Span(m.start(), m.end(), "DATE", m.group()))

    # ── Layer 1d: Life-stage terms ────────────────────────────────────────────
    text_lower = text.lower()
    for term in LIFE_STAGE_TERMS:
        idx = 0
        while True:
            pos = text_lower.find(term, idx)
            if pos == -1:
                break
            result.lifestages.append(text[pos:pos+len(term)])
            result.spans.append(Span(pos, pos+len(term), "LIFESTAGE", text[pos:pos+len(term)]))
            idx = pos + 1

    # ── Layer 1e: Habitat terms ───────────────────────────────────────────────
    for term in _HABITAT_TERMS:
        pos = text_lower.find(term)
        if pos >= 0:
            result.habitats.append(text[pos:pos+len(term)])
            result.spans.append(Span(pos, pos+len(term), "HABITAT", text[pos:pos+len(term)]))

    # ── Layer 2: Named-place heuristic ────────────────────────────────────────
    # Proper-noun sequences not already tagged as TAXON
    tagged_spans = {(s.start, s.end) for s in result.spans if s.label == "TAXON"}
    for m in re.finditer(r"\b([A-Z][a-z]{2,})(?:\s+[A-Z][a-z]{2,}){0,3}\b", text):
        if (m.start(), m.end()) not in tagged_spans:
            # Must not be a measurement unit or life-stage
            first_word = m.group().split()[0].lower()
            if first_word not in LIFE_STAGE_TERMS and not _MEAS_RE.match(m.group()):
                result.localities.append(m.group())
                result.spans.append(Span(m.start(), m.end(), "LOCALITY", m.group()))

    result.taxa    = list(dict.fromkeys(result.taxa))    # dedup preserving order
    result.localities = list(dict.fromkeys(result.localities))

    logger.debug(
        "[prepass] taxa=%d, localities=%d, measurements=%d, dates=%d, "
        "habitats=%d, lifestages=%d",
        len(result.taxa), len(result.localities), len(result.measurements),
        len(result.dates), len(result.habitats), len(result.lifestages),
    )
    return result


def format_annotations_for_prompt(pre: PrePassResult) -> str:
    """
    Format pre-pass results as a structured annotation block to prepend to
    the LLM prompt. The LLM is told to USE these spans, not invent new ones.

    Usage in biotrace_v5.py:
        annotation_block = format_annotations_for_prompt(pre)
        full_text = annotation_block + "\\n\\n" + text
    """
    lines = ["[PRE-ANNOTATED SPANS — use these, do not contradict them]"]
    if pre.taxa:
        lines.append(f"TAXA detected: {', '.join(pre.taxa[:10])}")
    if pre.localities:
        lines.append(f"LOCALITIES detected: {', '.join(pre.localities[:8])}")
    if pre.dates:
        lines.append(f"DATES detected: {', '.join(pre.dates[:5])}")
    if pre.measurements:
        lines.append(
            "MEASUREMENTS (these are morphology/size — NEVER use as verbatimLocality): "
            + ", ".join(pre.measurements[:6])
        )
    if pre.habitats:
        lines.append(
            "HABITATS (use in Habitat field — NEVER as verbatimLocality alone): "
            + ", ".join(set(pre.habitats))
        )
    if pre.lifestages:
        lines.append(
            "LIFE-STAGE TERMS (these are NOT taxa — do not use as Recorded Name): "
            + ", ".join(set(pre.lifestages))
        )
    return "\n".join(lines)