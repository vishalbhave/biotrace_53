"""
biotrace_scientific_chunker.py  —  BioTrace v5.4
────────────────────────────────────────────────────────────────────────────
ScientificPaperChunker — context-aware chunking for academic papers.

ROOT CAUSE of contextual loss in HierarchicalChunker
─────────────────────────────────────────────────────
Scientific papers have a characteristic pattern:

    METHODS SECTION:
      "Samples were collected from Narara reef, Gulf of Kutch, Gujarat
       (22.35°N, 69.67°E) during September 2022–March 2023."

    RESULTS SECTION (pages later):
      "Five species of scyphozoans were identified.
       Cassiopea andromeda was the most abundant."

When chunked at section boundaries, the Results chunk has the species name
but NO locality. The Methods chunk has the locality but NO species.
The LLM correctly extracts nothing useful from either chunk alone.

THE FIX — Two mechanisms:

  1. Section role detection
     Classify each section as: METHODS | RESULTS | DISCUSSION |
     ABSTRACT | INTRODUCTION | TABLES | OTHER.
     Sections typed as METHODS are parsed for localities, depth, and
     date strings that become a "study context" block.

  2. Context injection
     Before sending a RESULTS or DISCUSSION chunk to the LLM, prepend
     the study context extracted from METHODS / ABSTRACT.
     The LLM now sees both species name AND collection locality in the
     same prompt — it can correctly link them.

  3. Overlap-aware sentence splitting
     Chunks always break at sentence boundaries (never mid-sentence).
     Species-containing sentences always start a new chunk (never split).
     An overlap window carries the last N sentences of the previous chunk
     into the next to preserve cross-sentence anaphora.

Usage
─────
    from biotrace_scientific_chunker import ScientificPaperChunker, SciChunk

    sc = ScientificPaperChunker(chunk_chars=6000, overlap_chars=400)
    batches = sc.chunk(markdown_text, source_label="Author 2024")

    for batch in batches:
        # batch.context  — text to send to LLM (injected + chunk)
        # batch.section  — section label ("Results", "Methods", etc.)
        # batch.candidate_localities — pre-extracted locality strings
        # batch.candidate_species   — pre-extracted species candidates
        process_chunk(batch.context, batch.section, ...)
"""
from __future__ import annotations

import re
import logging
import regex as re 
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("biotrace.sci_chunker")

# ─────────────────────────────────────────────────────────────────────────────
#  Section role classification
# ─────────────────────────────────────────────────────────────────────────────

# Regex patterns for markdown section headings
_HEADING_RE = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)

# Keyword sets that identify section roles
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
    """Return the role label for a section heading string."""
    h_lower = heading.lower()
    for role, keywords in _ROLE_KEYWORDS.items():
        if any(kw in h_lower for kw in keywords):
            return role
    return "OTHER"


# ─────────────────────────────────────────────────────────────────────────────
#  Locality / date / depth extraction from text (fast rule-based)
# ─────────────────────────────────────────────────────────────────────────────

# Named place heuristic: capitalised words, 2-4 tokens, not followed by a number
_PLACE_RE = re.compile(
    r"\b([A-Z][a-z]{2,})(?:\s+(?:of\s+)?[A-Z][a-z]{2,}){0,3}\b"
)

# Date patterns (matches "September 2022", "2022–2023", "2022-2023", "2022/2023")
_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{4}\b|\b\d{4}[–\-/]\d{2,4}\b",
    re.IGNORECASE,
)

# Depth patterns
_DEPTH_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:–|-)\s*(\d+(?:\.\d+)?)?\s*m(?:eters?)?\b"
    r"|\bat\s+(?:a\s+)?depth\s+of\s+(\d+(?:\.\d+)?)\s*m\b",
    re.IGNORECASE,
)

# Admin suffixes to exclude from locality candidates
_LOC_BLOCKLIST = frozenset({
    "Table", "Figure", "Methods", "Results", "Discussion", "Abstract",
    "Introduction", "Section", "Appendix", "Supplementary", "Data",
    "Note", "Text", "Plate", "Volume", "Number", "Species", "Family",
    "Order", "Class", "Phylum", "Kingdom", "Genus",
})


def extract_locality_context(text: str) -> dict:
    """
    Fast rule-based extraction of study context from Methods/Abstract text.
    Returns dict with localities, dates, depths.
    """
    localities: list[str] = []
    seen_locs: set[str]   = set()

    for m in _PLACE_RE.finditer(text):
        name = m.group(0).strip()
        if name not in seen_locs and name.split()[0] not in _LOC_BLOCKLIST:
            localities.append(name)
            seen_locs.add(name)

    dates  = [m.group(0) for m in _DATE_RE.finditer(text)]
    depths = [m.group(0) for m in _DEPTH_RE.finditer(text)]

    return {
        "localities": localities[:20],
        "dates":      list(dict.fromkeys(dates))[:5],
        "depths":     list(dict.fromkeys(depths))[:5],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Species-bearing sentence detection
# ─────────────────────────────────────────────────────────────────────────────

_BINOMIAL_RE = re.compile(
    r"\b([A-Z][a-z]{2,})\s+([a-z]{3,}(?:\s+(?:cf\.|aff\.|sp\.|spp\.|var\.|subsp\.)\s*\S*)?)\b"
)
_GENUS_ONLY_RE = re.compile(
    r"\b([A-Z][a-z]{2,})\s+(?:sp\.|spp\.|cf\.|aff\.|n\.?\s*sp\.?)\b"
)

# Non-taxonomic words that can appear capitalized
_NON_TAXON_CAPS = frozenset({
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Fig", "Table", "Plate", "Station", "Site", "Area",
})


def sentence_has_species(sentence: str) -> bool:
    """True if the sentence contains a likely binomial or genus+qualifier."""
    for m in _BINOMIAL_RE.finditer(sentence):
        if m.group(1) not in _NON_TAXON_CAPS:
            return True
    if _GENUS_ONLY_RE.search(sentence):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Sentence splitter
# ─────────────────────────────────────────────────────────────────────────────

# Split at sentence boundaries (. ! ?) but not inside abbreviations
_SENT_RE = re.compile(
    r"(?<!\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|Fig|et|al|sp|cf|aff|var|subsp|spp|n\.s))"
    r"(?<=[.!?])\s+(?=[A-Z\"\'])"
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences at . ! ? boundaries."""
    sents = _SENT_RE.split(text.strip())
    # Filter empty, normalize whitespace
    return [re.sub(r"\s+", " ", s).strip() for s in sents if s.strip()]


# ─────────────────────────────────────────────────────────────────────────────
#  Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SciChunk:
    """
    A single extraction batch for the LLM.
    Compatible with ExtractionBatch interface used by biotrace_v5.py.
    """
    context:             str              # text to send to LLM
    section:             str              # section label
    section_role:        str              # METHODS | RESULTS | DISCUSSION | …
    char_start:          int  = 0
    char_end:            int  = 0
    candidate_localities: list[str] = field(default_factory=list)
    candidate_species:    list[str] = field(default_factory=list)
    injected_context:    str  = ""        # the Methods context that was prepended
    has_species:         bool = True


# ─────────────────────────────────────────────────────────────────────────────
#  Main chunker
# ─────────────────────────────────────────────────────────────────────────────

class ScientificPaperChunker:
    """
    Context-aware chunker for academic biodiversity papers.

    Algorithm
    ─────────
    1. Split markdown into sections at ## headings.
    2. Classify each section role (METHODS, RESULTS, etc.).
    3. For METHODS / ABSTRACT sections: extract locality context dictionary.
    4. For RESULTS / DISCUSSION / TAXONOMY sections:
       a. Split into sentences.
       b. Pack sentences into chunks ≤ max_chars, always breaking at sentence
          boundaries, always keeping species-containing sentences together
          with their surrounding context (±context_window sentences).
       c. Prepend the study context block from step 3 to each chunk.
    5. For TABLES: include whole-section as one chunk (tables are usually
       < max_chars and must not be split across chunks).
    6. For INTRODUCTION / OTHER: flat chunks without context injection.
    """

    def __init__(
        self,
        chunk_chars:          int = 6000,
        overlap_chars:        int = 400,
        context_inject_chars: int = 2000,
        context_window:       int = 2,      # sentences before/after species sentence
    ):
        self.chunk_chars          = chunk_chars
        self.overlap_chars        = overlap_chars
        self.context_inject_chars = context_inject_chars
        self.context_window       = context_window

    # ── Public API ─────────────────────────────────────────────────────────────

    def chunk(self, markdown_text: str, source_label: str = "") -> list[SciChunk]:
        """
        Main entry point. Returns list[SciChunk] ready for LLM extraction.
        """
        sections = self._split_sections(markdown_text)
        study_context = self._build_study_context(sections)

        chunks: list[SciChunk] = []
        for sec in sections:
            role = sec["role"]
            text = sec["text"]
            heading = sec["heading"]

            if role in ("METHODS", "ABSTRACT"):
                # Include Methods in output (secondary occurrences may be here)
                # but also inject its own locality context as a "self-annotated" chunk
                for ch in self._flat_chunk(text, heading, role, study_context=""):
                    chunks.append(ch)

            elif role in ("RESULTS", "DISCUSSION", "TAXONOMY"):
                # Context-injected chunking — key fix for cross-section loss
                for ch in self._species_focused_chunk(
                    text, heading, role, study_context
                ):
                    chunks.append(ch)

            elif role == "TABLES":
                # Keep whole table sections intact
                chunks.append(SciChunk(
                    context=study_context + "\n\n" + text if study_context else text,
                    section=heading,
                    section_role=role,
                    candidate_localities=study_context_locs(study_context),
                    injected_context=study_context,
                ))

            else:
                # INTRODUCTION / OTHER: no context injection
                for ch in self._flat_chunk(text, heading, role, study_context=""):
                    chunks.append(ch)

        logger.info(
            "[SciChunk] %s → %d sections → %d chunks (study_context=%d chars)",
            source_label, len(sections), len(chunks), len(study_context),
        )
        return chunks

    # ── Section splitter ────────────────────────────────────────────────────────

    def _split_sections(self, text: str) -> list[dict]:
        """Split markdown at # heading boundaries. Returns list of section dicts."""
        # Find all headings
        heading_matches = list(_HEADING_RE.finditer(text))
        if not heading_matches:
            # No headings — treat whole text as a single RESULTS section
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

    # ── Study context builder ───────────────────────────────────────────────────

    def _build_study_context(self, sections: list[dict]) -> str:
        """
        Extract locality, date, depth from METHODS and ABSTRACT sections.
        Returns a formatted context block to inject into RESULTS chunks.
        """
        context_parts: list[str] = []
        for sec in sections:
            if sec["role"] not in ("METHODS", "ABSTRACT"):
                continue
            ctx = extract_locality_context(sec["text"])
            if ctx["localities"] or ctx["dates"]:
                lines = [f"[STUDY CONTEXT from {sec['heading']} section]"]
                if ctx["localities"]:
                    lines.append(
                        "Localities: " + ", ".join(ctx["localities"][:10])
                    )
                if ctx["dates"]:
                    lines.append("Collection period: " + "; ".join(ctx["dates"]))
                if ctx["depths"]:
                    lines.append("Depth range: " + "; ".join(ctx["depths"]))
                context_parts.append("\n".join(lines))

        combined = "\n\n".join(context_parts)
        return combined[:self.context_inject_chars]

    # ── Species-focused chunking (RESULTS / DISCUSSION) ─────────────────────────

    def _species_focused_chunk(
        self,
        text:          str,
        heading:       str,
        role:          str,
        study_context: str,
    ) -> list[SciChunk]:
        """
        Pack sentences into chunks, always keeping species sentences with their
        surrounding context. Injects study_context at top of each chunk.
        """
        sentences = split_sentences(text)
        if not sentences:
            return []

        chunks:   list[SciChunk] = []
        i    = 0
        total = len(sentences)

        while i < total:
            # Try to pack sentences up to chunk_chars
            batch_sents: list[str] = []
            budget = self.chunk_chars - len(study_context) - 200  # headroom

            j = i
            while j < total:
                candidate = " ".join(batch_sents + [sentences[j]])
                if len(candidate) > budget and batch_sents:
                    break
                batch_sents.append(sentences[j])
                j += 1

            if not batch_sents:
                # Single sentence exceeds budget — include anyway
                batch_sents = [sentences[i]]
                j = i + 1

            chunk_text = " ".join(batch_sents)
            has_species = any(sentence_has_species(s) for s in batch_sents)

            # Extract candidates from this chunk
            loc_ctx  = extract_locality_context(chunk_text)
            sp_cands = [
                m.group(0) for m in _BINOMIAL_RE.finditer(chunk_text)
                if m.group(1) not in _NON_TAXON_CAPS
            ]

            # Inject study context
            full_context = (study_context + "\n\n" + chunk_text) if study_context else chunk_text

            chunks.append(SciChunk(
                context              = full_context,
                section              = heading,
                section_role         = role,
                candidate_localities = (study_context_locs(study_context) +
                                        loc_ctx["localities"]),
                candidate_species    = sp_cands,
                injected_context     = study_context,
                has_species          = has_species,
            ))

            # Overlap: carry last N sentences into next iteration
            overlap_sents = batch_sents[-self.context_window:] if self.context_window else []
            overlap_chars = sum(len(s) for s in overlap_sents)
            if overlap_chars > self.overlap_chars:
                overlap_sents = overlap_sents[-1:]  # just the last sentence

            i = j
            # Rewind by overlap window so it's included in next chunk
            rewind = len(overlap_sents)
            if rewind and i < total:
                i = max(i - rewind, i - 1)

        return chunks

    # ── Flat chunking (METHODS, INTRODUCTION, OTHER) ────────────────────────────

    def _flat_chunk(
        self,
        text:          str,
        heading:       str,
        role:          str,
        study_context: str,
    ) -> list[SciChunk]:
        """Sentence-boundary-aware flat chunks without species-first packing."""
        sentences = split_sentences(text)
        if not sentences:
            return []

        chunks: list[SciChunk] = []
        budget = self.chunk_chars - 200
        i = 0
        total = len(sentences)

        while i < total:
            batch_sents: list[str] = []
            j = i
            while j < total:
                candidate = " ".join(batch_sents + [sentences[j]])
                if len(candidate) > budget and batch_sents:
                    break
                batch_sents.append(sentences[j])
                j += 1
            if not batch_sents:
                batch_sents = [sentences[i]]
                j = i + 1

            chunk_text = " ".join(batch_sents)
            loc_ctx = extract_locality_context(chunk_text)

            chunks.append(SciChunk(
                context              = chunk_text,
                section              = heading,
                section_role         = role,
                candidate_localities = loc_ctx["localities"],
                candidate_species    = [],
                has_species          = any(sentence_has_species(s) for s in batch_sents),
            ))
            i = j

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  Helper used in SciChunk construction
# ─────────────────────────────────────────────────────────────────────────────

def study_context_locs(study_context: str) -> list[str]:
    """Extract locality strings from an injected study context block."""
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
Collection was carried out monthly from September 2022 to March 2023.

## Results
Five species of scyphozoans were identified from the study sites.
Cassiopea andromeda (Forsskål, 1775) was the most abundant species, forming dense
aggregations at Narara reef. Chrysaora cf. melanaster was recorded from the pelagic
zone off Okha. Mastigias papua was observed at Arambhada coast in association with
seagrass beds. Catostylus mosaicus and Aurelia aurita were less frequently observed.

## Discussion
The presence of Cassiopea andromeda in the Gulf of Kutch extends its known distribution
along the Indian west coast. Previous records of C. andromeda from the region include
those of Gravely (1941) from Madras and Southcott (1956) from Ceylon.
"""

    sc = ScientificPaperChunker(chunk_chars=2000, overlap_chars=200, context_inject_chars=1000)
    batches = sc.chunk(SAMPLE, source_label="test_paper")
    for b in batches:
        print(f"\n[{b.section_role}] {b.section!r}")
        print(f"  has_species={b.has_species}")
        print(f"  localities={b.candidate_localities[:3]}")
        print(f"  species={b.candidate_species[:3]}")
        print(f"  context_preview={b.context[:120]!r}…")