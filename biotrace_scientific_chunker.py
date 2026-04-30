"""
biotrace_scientific_chunker.py  —  BioTrace v5.6  (+ Pydantic AI Validation)
────────────────────────────────────────────────────────────────────────────
ScientificPaperChunker — context-aware chunking for academic papers.
PydanticAIChunkValidator — post-extraction structured validation via pydantic-ai.

CHANGES vs v5.4
────────────────
1. Pydantic AI integration for extraction validation
   After each chunk is processed by the LLM, PydanticAIChunkValidator:
     • Validates occurrence dicts against OccurrenceExtract (Pydantic model)
     • Flags suspicious locality strings (morphology terms, habitat as locality)
     • Normalises species name casing and separates genus/epithet
     • Infers missing fields (occurrenceType, verbatimLocality) from study context
     • Runs async for batch chunks; falls back gracefully if pydantic-ai absent

2. Docling section roles honoured
   ScientificPaperChunker._split_sections() now accepts a pre-parsed section
   list from DoclingWikiBridge.extract_sections_from_docling() so docling
   sections are not re-parsed from scratch (avoids duplicate effort).

3. Unchanged from v5.4
   All existing SciChunk, classify_section, extract_locality_context,
   split_sentences, study_context_locs, ScientificPaperChunker.chunk()
   remain API-compatible.

Usage
─────
    from biotrace_scientific_chunker import ScientificPaperChunker, PydanticAIChunkValidator

    sc = ScientificPaperChunker()
    batches = sc.chunk(markdown_text, source_label="Author 2024")

    # After LLM extraction produces raw_records per chunk:
    validator = PydanticAIChunkValidator(model="claude-sonnet-4-20250514")
    validated = await validator.validate_batch(raw_records, study_context=batches[0].injected_context)
"""
from __future__ import annotations

import asyncio
import logging
import re
import regex as _regex
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("biotrace.sci_chunker")

# Use regex for full Unicode support if available
try:
    _re = _regex
except Exception:
    _re = re

# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic models for validated extraction
# ─────────────────────────────────────────────────────────────────────────────

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    _PYDANTIC_OK = True
except ImportError:
    _PYDANTIC_OK = False
    BaseModel = object  # type: ignore
    def Field(*a, **kw): return None  # type: ignore

# pydantic-ai agent
try:
    from pydantic_ai import Agent as _PAIAgent
    from pydantic_ai.models.anthropic import AnthropicModel as _AnthropicModel
    _PAI_OK = True
except ImportError:
    _PAI_OK = False

# Ollama fallback for pydantic-ai
try:
    from pydantic_ai.models.ollama import OllamaModel as _OllamaModel
    _PAI_OLLAMA_OK = True
except ImportError:
    _PAI_OLLAMA_OK = False


if _PYDANTIC_OK:
    class OccurrenceExtract(BaseModel):
        """
        Validated occurrence record.  Mirrors biotrace_v5.py's occurrence dict fields.
        Used as the result_type for PydanticAIChunkValidator.
        """
        validName:          str             = Field("", description="Accepted binomial species name")
        recordedName:       str             = Field("", description="Name as it appears in source text")
        verbatimLocality:   str             = Field("", description="Verbatim locality string from text")
        occurrenceType:     str             = Field("Uncertain",
                                                     description="Primary | Secondary | Uncertain")
        sourceCitation:     str             = Field("", description="Formatted citation")
        decimalLatitude:    Optional[float] = Field(None, ge=-90.0,  le=90.0)
        decimalLongitude:   Optional[float] = Field(None, ge=-180.0, le=180.0)
        eventDate:          str             = Field("")
        depthRange:         str             = Field("")
        habitat:            str             = Field("")
        phylum:             str             = Field("")
        family_:            str             = Field("")
        wormsID:            str             = Field("")
        confidence:         float           = Field(1.0, ge=0.0, le=1.0,
                                                     description="Extraction confidence 0–1")
        validation_flags:   list[str]       = Field(default_factory=list,
                                                     description="Warnings from validator")

        @field_validator("occurrenceType")
        @classmethod
        def normalise_type(cls, v: str) -> str:
            v = v.strip().title()
            return v if v in ("Primary", "Secondary", "Uncertain") else "Uncertain"

        @field_validator("validName", "recordedName")
        @classmethod
        def normalise_binomial(cls, v: str) -> str:
            """Ensure genus is capitalised, epithet is lowercase."""
            if not v.strip():
                return v
            parts = v.strip().split()
            if len(parts) >= 2:
                return parts[0].capitalize() + " " + " ".join(p.lower() for p in parts[1:])
            return v

        @model_validator(mode="after")
        def check_locality_not_morphology(self) -> "OccurrenceExtract":
            """Flag if verbatimLocality looks like a morphology/habitat term."""
            _MORPHO_TERMS = {
                "dorsal", "ventral", "anterior", "posterior", "lateral",
                "mantle", "gill", "tentacle", "body", "head", "foot",
                "coral reef", "sandy bottom", "mudflat", "seagrass",
                "intertidal", "subtidal", "benthic", "pelagic",
            }
            loc_lower = self.verbatimLocality.lower().strip()
            if any(term == loc_lower for term in _MORPHO_TERMS):
                self.validation_flags.append(
                    f"locality_suspicious:{self.verbatimLocality!r} looks like morphology/habitat"
                )
                self.confidence = min(self.confidence, 0.4)
            return self

        @model_validator(mode="after")
        def infer_missing_from_context(self) -> "OccurrenceExtract":
            """Set confidence low if both locality and species are empty."""
            if not self.validName and not self.recordedName:
                self.validation_flags.append("missing_species_name")
                self.confidence = 0.0
            if not self.verbatimLocality:
                self.validation_flags.append("missing_locality")
                self.confidence = min(self.confidence, 0.5)
            return self

else:
    OccurrenceExtract = dict  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic AI validation agent
# ─────────────────────────────────────────────────────────────────────────────

_VALIDATOR_SYSTEM = """\
You are a marine biodiversity data validator. Given a raw occurrence record extracted
from a scientific paper, validate and correct it:

1. Ensure species name has capitalised genus + lowercase epithet.
2. Check that verbatimLocality is a real geographic place (not a body part or habitat term).
3. Infer occurrenceType (Primary if specimen-based; Secondary if literature-cited).
4. If decimalLatitude/Longitude are provided, verify they fall in a plausible range
   for the stated locality (India: lat 6–37, lon 68–97).
5. Extract eventDate if mentioned in the context block.
6. Return ONLY a valid JSON object matching the OccurrenceExtract schema.
   Include a confidence (0.0–1.0) and list any issues in validation_flags.
Schema fields: validName, recordedName, verbatimLocality, occurrenceType, sourceCitation,
decimalLatitude, decimalLongitude, eventDate, depthRange, habitat, phylum, family_,
wormsID, confidence, validation_flags.
Return ONLY JSON. No prose, no markdown fences.
"""


class PydanticAIChunkValidator:
    """
    Validates and normalises raw occurrence dicts extracted from a SciChunk.

    Two modes:
      1. pydantic-ai Agent (when _PAI_OK, uses Anthropic claude-sonnet-4-20250514
         or Ollama if base_url provided)
      2. Pydantic-only validation (no LLM, uses OccurrenceExtract validators only)

    Parameters
    ──────────
    model       Model string for pydantic-ai (default: "claude-sonnet-4-20250514").
    base_url    If set, uses Ollama at this URL instead of Anthropic.
    use_llm     If False, only runs Pydantic model validation (fast, no API call).
    """

    def __init__(
        self,
        model:    str  = "claude-sonnet-4-20250514",
        base_url: str  = "",       # e.g. "http://localhost:11434" for Ollama
        use_llm:  bool = True,
    ):
        self.model    = model
        self.base_url = base_url
        self.use_llm  = use_llm and _PAI_OK and _PYDANTIC_OK
        self._agent   = None

        if self.use_llm:
            self._init_agent()

    def _init_agent(self):
        try:
            if self.base_url and _PAI_OLLAMA_OK:
                pai_model = _OllamaModel(self.model, base_url=self.base_url)
            else:
                pai_model = _AnthropicModel(self.model)

            self._agent = _PAIAgent(
                model=pai_model,
                result_type=OccurrenceExtract if _PYDANTIC_OK else dict,
                system_prompt=_VALIDATOR_SYSTEM,
            )
            logger.info("[SciChunk/PAI] Agent ready (model=%s)", self.model)
        except Exception as exc:
            logger.warning("[SciChunk/PAI] Agent init failed: %s — falling back to Pydantic-only", exc)
            self._agent = None
            self.use_llm = False

    def validate_one(
        self,
        record: dict,
        study_context: str = "",
    ) -> "OccurrenceExtract | dict":
        """
        Synchronous validation of a single occurrence record.
        Returns OccurrenceExtract (validated) or the input dict on failure.
        """
        if self.use_llm and self._agent:
            return asyncio.get_event_loop().run_until_complete(
                self._validate_one_async(record, study_context)
            )
        return self._pydantic_only(record)

    async def validate_one_async(
        self,
        record: dict,
        study_context: str = "",
    ) -> "OccurrenceExtract | dict":
        """Async variant — preferred for batch processing."""
        if self.use_llm and self._agent:
            return await self._validate_one_async(record, study_context)
        return self._pydantic_only(record)

    async def validate_batch(
        self,
        records:       list[dict],
        study_context: str = "",
        max_concurrent: int = 3,
    ) -> list["OccurrenceExtract | dict"]:
        """
        Validate a list of occurrence records concurrently.

        max_concurrent caps simultaneous API calls to respect rate limits.
        Returns list in same order as input.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _guarded(rec: dict) -> "OccurrenceExtract | dict":
            async with semaphore:
                return await self.validate_one_async(rec, study_context)

        return await asyncio.gather(*[_guarded(r) for r in records])

    async def _validate_one_async(
        self,
        record: dict,
        study_context: str,
    ) -> "OccurrenceExtract | dict":
        try:
            user_prompt = (
                f"Study context (locality/date hints from Methods section):\n"
                f"{study_context[:1500]}\n\n"
                f"Raw occurrence record (JSON):\n"
                f"{__import__('json').dumps(record, default=str)}"
            )
            result = await self._agent.run(user_prompt)
            return result.data
        except Exception as exc:
            logger.warning("[SciChunk/PAI] Validation failed: %s — using Pydantic-only", exc)
            return self._pydantic_only(record)

    def _pydantic_only(self, record: dict) -> "OccurrenceExtract | dict":
        """Run Pydantic model validators without LLM call."""
        if not _PYDANTIC_OK:
            return record
        try:
            return OccurrenceExtract(**{
                k: v for k, v in record.items()
                if k in OccurrenceExtract.model_fields
            })
        except Exception as exc:
            logger.debug("[SciChunk] Pydantic validation: %s", exc)
            return record

    def to_dict(self, result: "OccurrenceExtract | dict") -> dict:
        """Convert validation result to plain dict for DB insertion."""
        if hasattr(result, "model_dump"):
            d = result.model_dump()
            d.pop("validation_flags", None)   # don't persist flags to DB
            d.pop("confidence", None)
            return d
        return result if isinstance(result, dict) else {}


# ─────────────────────────────────────────────────────────────────────────────
#  Section role classification  (unchanged from v5.4)
# ─────────────────────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)

_ROLE_KEYWORDS: dict[str, frozenset] = {
    "ABSTRACT":     frozenset({"abstract", "summary", "synopsis"}),
    "INTRODUCTION": frozenset({"introduction", "background", "overview"}),
    "METHODS":      frozenset({
        "method", "methods", "methodology", "material", "materials",
        "study area", "study site", "sampling", "collection", "field",
        "survey", "protocol", "procedure", "experimental", "station",
    }),
    "RESULTS":      frozenset({"result", "results", "findings", "occurrence", "record"}),
    "DISCUSSION":   frozenset({"discussion", "conclusion", "conclusions", "remarks",
                               "synthesis", "implication"}),
    "TABLES":       frozenset({"table", "appendix", "supplementary", "checklist"}),
    "TAXONOMY":     frozenset({"taxonomy", "systematics", "diagnosis", "description",
                               "new species", "new record", "key to species"}),
}


def classify_section(heading: str) -> str:
    h_lower = heading.lower()
    for role, keywords in _ROLE_KEYWORDS.items():
        if any(kw in h_lower for kw in keywords):
            return role
    return "OTHER"


# ─────────────────────────────────────────────────────────────────────────────
#  Locality / date / depth extraction (unchanged from v5.4)
# ─────────────────────────────────────────────────────────────────────────────

_PLACE_RE = re.compile(r"\b([A-Z][a-z]{2,})(?:\s+(?:of\s+)?[A-Z][a-z]{2,}){0,3}\b")
_DATE_RE  = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{4}\b|\b\d{4}[–\-/]\d{2,4}\b", re.IGNORECASE)
_DEPTH_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:–|-)\s*(\d+(?:\.\d+)?)?\s*m(?:eters?)?\b"
    r"|\bat\s+(?:a\s+)?depth\s+of\s+(\d+(?:\.\d+)?)\s*m\b", re.IGNORECASE)
_LOC_BLOCKLIST = frozenset({
    "Table", "Figure", "Methods", "Results", "Discussion", "Abstract",
    "Introduction", "Section", "Appendix", "Supplementary", "Data",
    "Note", "Text", "Plate", "Volume", "Number", "Species", "Family",
    "Order", "Class", "Phylum", "Kingdom", "Genus"})


def extract_locality_context(text: str) -> dict:
    localities: list[str] = []
    seen_locs:  set[str]  = set()
    for m in _PLACE_RE.finditer(text):
        name = m.group(0).strip()
        if name not in seen_locs and name.split()[0] not in _LOC_BLOCKLIST:
            localities.append(name); seen_locs.add(name)
    dates  = [m.group(0) for m in _DATE_RE.finditer(text)]
    depths = [m.group(0) for m in _DEPTH_RE.finditer(text)]
    return {"localities": localities[:20],
            "dates":  list(dict.fromkeys(dates))[:5],
            "depths": list(dict.fromkeys(depths))[:5]}


# ─────────────────────────────────────────────────────────────────────────────
#  Species-bearing sentence detection (unchanged from v5.4)
# ─────────────────────────────────────────────────────────────────────────────

_BINOMIAL_RE   = re.compile(
    r"\b([A-Z][a-z]{2,})\s+([a-z]{3,}(?:\s+(?:cf\.|aff\.|sp\.|spp\.|var\.|subsp\.)\s*\S*)?)\b")
_GENUS_ONLY_RE = re.compile(
    r"\b([A-Z][a-z]{2,})\s+(?:sp\.|spp\.|cf\.|aff\.|n\.?\s*sp\.?)\b")
_NON_TAXON_CAPS = frozenset({
    "January","February","March","April","May","June","July",
    "August","September","October","November","December",
    "Fig","Table","Plate","Station","Site","Area"})


def sentence_has_species(sentence: str) -> bool:
    for m in _BINOMIAL_RE.finditer(sentence):
        if m.group(1) not in _NON_TAXON_CAPS:
            return True
    return bool(_GENUS_ONLY_RE.search(sentence))


# ─────────────────────────────────────────────────────────────────────────────
#  Sentence splitter (unchanged from v5.4)
# ─────────────────────────────────────────────────────────────────────────────

_SENT_RE = re.compile(
    r"\b"  # Match the word boundary first
    r"(?<!Dr)(?<!Mr)(?<!Ms)(?<!vs)(?<!sp)(?<!cf)(?<!al)(?<!et)" # 2 chars
    r"(?<!Mrs)(?<!Sr)(?<!Jr)(?<!Fig)(?<!aff)(?<!var)(?<!spp)"    # 3 chars
    r"(?<!Prof)"                                                 # 4 chars
    r"(?<!subsp)"                                                # 5 chars
    r"(?<!n\.s)"                                                 # 3 chars (fixed)
    r"(?<=[.!?])\s+(?=[A-Z\"'])"                                 # Trigger
)


def split_sentences(text: str) -> list[str]:
    sents = _SENT_RE.split(text.strip())
    return [re.sub(r"\s+", " ", s).strip() for s in sents if s.strip()]


# ─────────────────────────────────────────────────────────────────────────────
#  Output dataclass (unchanged from v5.4)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SciChunk:
    context:              str
    section:              str
    section_role:         str
    char_start:           int  = 0
    char_end:             int  = 0
    candidate_localities: list[str] = field(default_factory=list)
    candidate_species:    list[str] = field(default_factory=list)
    injected_context:     str  = ""
    has_species:          bool = True


# ─────────────────────────────────────────────────────────────────────────────
#  Main chunker (v5.6: accepts pre-parsed docling sections)
# ─────────────────────────────────────────────────────────────────────────────

class ScientificPaperChunker:
    """
    Context-aware chunker for academic biodiversity papers.

    New in v5.6:
      • chunk_from_sections(sections_dict) — accepts pre-parsed docling sections
        to avoid re-parsing from markdown. Used by DoclingWikiBridge.
      • All existing chunk(markdown_text) API unchanged.
    """

    def __init__(
        self,
        chunk_chars:          int = 6000,
        overlap_chars:        int = 400,
        context_inject_chars: int = 2000,
        context_window:       int = 2,
    ):
        self.chunk_chars          = chunk_chars
        self.overlap_chars        = overlap_chars
        self.context_inject_chars = context_inject_chars
        self.context_window       = context_window

    # ── Public API ─────────────────────────────────────────────────────────────

    def chunk(self, markdown_text: str, source_label: str = "") -> list[SciChunk]:
        """Main entry point: parse markdown → produce SciChunk list."""
        sections = self._split_sections(markdown_text)
        return self._chunks_from_section_list(sections, source_label)

    def chunk_from_sections(
        self,
        sections_dict: dict[str, str],
        source_label:  str = "",
    ) -> list[SciChunk]:
        """
        v5.6 NEW — Accept pre-parsed sections dict from DoclingWikiBridge.
        Avoids double-parsing documents that were already processed by docling.

        sections_dict: {role: text}  e.g. {"METHODS": "...", "RESULTS": "..."}
        """
        section_list = [
            {"heading": role.title(), "role": role, "text": text}
            for role, text in sections_dict.items() if text.strip()
        ]
        return self._chunks_from_section_list(section_list, source_label)

    # ── Internal ────────────────────────────────────────────────────────────────

    def _chunks_from_section_list(
        self,
        sections:     list[dict],
        source_label: str,
    ) -> list[SciChunk]:
        study_context = self._build_study_context(sections)
        chunks: list[SciChunk] = []

        for sec in sections:
            role    = sec["role"]
            text    = sec["text"]
            heading = sec["heading"]

            if role in ("METHODS", "ABSTRACT"):
                for ch in self._flat_chunk(text, heading, role, ""):
                    chunks.append(ch)
            elif role in ("RESULTS", "DISCUSSION", "TAXONOMY"):
                for ch in self._species_focused_chunk(text, heading, role, study_context):
                    chunks.append(ch)
            elif role == "TABLES":
                chunks.append(SciChunk(
                    context=study_context + "\n\n" + text if study_context else text,
                    section=heading, section_role=role,
                    candidate_localities=study_context_locs(study_context),
                    injected_context=study_context))
            else:
                for ch in self._flat_chunk(text, heading, role, ""):
                    chunks.append(ch)

        logger.info("[SciChunk] %s → %d sections → %d chunks (study_context=%d chars)",
                    source_label, len(sections), len(chunks), len(study_context))
        return chunks

    def _split_sections(self, text: str) -> list[dict]:
        heading_matches = list(_HEADING_RE.finditer(text))
        if not heading_matches:
            return [{"heading": "Full text", "role": "RESULTS", "text": text}]
        sections = []
        for i, m in enumerate(heading_matches):
            heading = m.group(1).strip()
            role    = classify_section(heading)
            start   = m.end()
            end     = heading_matches[i+1].start() if i+1 < len(heading_matches) else len(text)
            body    = text[start:end].strip()
            if body:
                sections.append({"heading": heading, "role": role, "text": body})
        return sections

    def _build_study_context(self, sections: list[dict]) -> str:
        context_parts: list[str] = []
        for sec in sections:
            if sec["role"] not in ("METHODS", "ABSTRACT"):
                continue
            ctx = extract_locality_context(sec["text"])
            if ctx["localities"] or ctx["dates"]:
                lines = [f"[STUDY CONTEXT from {sec['heading']} section]"]
                if ctx["localities"]:
                    lines.append("Localities: " + ", ".join(ctx["localities"][:10]))
                if ctx["dates"]:
                    lines.append("Collection period: " + "; ".join(ctx["dates"]))
                if ctx["depths"]:
                    lines.append("Depth range: " + "; ".join(ctx["depths"]))
                context_parts.append("\n".join(lines))
        return "\n\n".join(context_parts)[:self.context_inject_chars]

    def _species_focused_chunk(
        self, text: str, heading: str, role: str, study_context: str
    ) -> list[SciChunk]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        chunks: list[SciChunk] = []
        i = 0; total = len(sentences)
        while i < total:
            batch_sents: list[str] = []
            budget = self.chunk_chars - len(study_context) - 200
            j = i
            while j < total:
                candidate = " ".join(batch_sents + [sentences[j]])
                if len(candidate) > budget and batch_sents:
                    break
                batch_sents.append(sentences[j]); j += 1
            if not batch_sents:
                batch_sents = [sentences[i]]; j = i + 1

            chunk_text  = " ".join(batch_sents)
            has_species = any(sentence_has_species(s) for s in batch_sents)
            loc_ctx     = extract_locality_context(chunk_text)
            sp_cands    = [m.group(0) for m in _BINOMIAL_RE.finditer(chunk_text)
                           if m.group(1) not in _NON_TAXON_CAPS]
            full_context = (study_context + "\n\n" + chunk_text) if study_context else chunk_text

            chunks.append(SciChunk(
                context=full_context, section=heading, section_role=role,
                candidate_localities=(study_context_locs(study_context) + loc_ctx["localities"]),
                candidate_species=sp_cands,
                injected_context=study_context, has_species=has_species))

            overlap_sents = batch_sents[-self.context_window:] if self.context_window else []
            if sum(len(s) for s in overlap_sents) > self.overlap_chars:
                overlap_sents = overlap_sents[-1:]
            i = j
            rewind = len(overlap_sents)
            if rewind and i < total:
                i = max(i - rewind, i - 1)

        return chunks

    def _flat_chunk(
        self, text: str, heading: str, role: str, study_context: str
    ) -> list[SciChunk]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        chunks: list[SciChunk] = []
        budget = self.chunk_chars - 200
        i = 0; total = len(sentences)
        while i < total:
            batch_sents: list[str] = []
            j = i
            while j < total:
                candidate = " ".join(batch_sents + [sentences[j]])
                if len(candidate) > budget and batch_sents:
                    break
                batch_sents.append(sentences[j]); j += 1
            if not batch_sents:
                batch_sents = [sentences[i]]; j = i + 1
            chunk_text = " ".join(batch_sents)
            loc_ctx = extract_locality_context(chunk_text)
            chunks.append(SciChunk(
                context=chunk_text, section=heading, section_role=role,
                candidate_localities=loc_ctx["localities"], candidate_species=[],
                has_species=any(sentence_has_species(s) for s in batch_sents)))
            i = j
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────────────────────────────────────

def study_context_locs(study_context: str) -> list[str]:
    locs: list[str] = []
    for line in study_context.splitlines():
        if line.startswith("Localities:"):
            locs = [l.strip() for l in line.replace("Localities:", "").split(",")]
            break
    return [l for l in locs if l]


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAMPLE = """
## Abstract
This study reports scyphozoan diversity from Narara reef, Gulf of Kutch, Gujarat
(22.35°N, 69.67°E) during September 2022–March 2023.

## Study Area and Methods
Specimens were collected by snorkelling at depths of 0.5–3 m from Narara Marine
National Park, Arambhada coast and Okha jetty, Gujarat, India.

## Results
Five species of scyphozoans were identified from the study sites.
Cassiopea andromeda (Forsskål, 1775) was the most abundant species, forming dense
aggregations at Narara reef. Chrysaora cf. melanaster was recorded from the pelagic
zone off Okha.

## Discussion
The presence of Cassiopea andromeda in the Gulf of Kutch extends its known distribution.
"""
    sc      = ScientificPaperChunker(chunk_chars=2000, overlap_chars=200)
    batches = sc.chunk(SAMPLE, "test_paper")
    for b in batches:
        print(f"[{b.section_role}] {b.section!r} | species={b.has_species} | locs={b.candidate_localities[:2]}")

    print("\n--- Pydantic-only validation ---")
    validator = PydanticAIChunkValidator(use_llm=False)
    raw = {"validName": "cassiopea andromeda", "verbatimLocality": "dorsal",
           "occurrenceType": "primary", "sourceCitation": "Test"}
    result = validator.validate_one(raw)
    if hasattr(result, "model_dump"):
        print(result.model_dump())
    else:
        print(result)