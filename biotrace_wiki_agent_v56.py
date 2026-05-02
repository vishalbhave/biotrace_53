"""
biotrace_wiki_agent_v56.py  —  BioTrace v5.6  Agentic Wiki Narrative Builder
═══════════════════════════════════════════════════════════════════════════════
Post-HITL agent that builds wiki article sections from HITL-approved species
using the Docling-converted markdown chunks stored in session state.

Architecture
────────────
  WikiNarrativeAgent
  ├── retrieve_species_chunks()   ← finds text chunks containing the species
  ├── reason_over_section()       ← single LLM call per wiki section
  ├── synthesise_article()        ← multi-section assembly + cross-chunk check
  └── commit_to_wiki()            ← writes to BioTraceWikiUnified (all DBs)

  AgentOrchestrator               ← drives the agent loop over N species
  ├── run_for_batch()             ← iterates approved species post-HITL
  └── render_streamlit_progress() ← live progress in Extract tab

Integration (see biotrace_v53.py patch notes at bottom of this file)
─────────────────────────────────────────────────────────────────────
  After HITL confirmation in the _hitl_resume block:

    from biotrace_wiki_agent_v56 import AgentOrchestrator
    orch = AgentOrchestrator(
        wiki        = get_wiki(),
        call_llm_fn = lambda p: call_llm(p, provider, model_sel, api_key, ollama_url),
        log_cb      = _lcb,
    )
    orch.run_for_batch(
        approved_occurrences = _appr,
        md_text              = st.session_state.pop("_hitl_pending_md_text", ""),
        citation             = _h_cite,
    )

Wiki sections produced
──────────────────────
  • Taxonomy & Classification   — from occurrence taxonomy fields + LLM
  • Distribution & Localities   — from verbatimLocality + lat/lon
  • Ecology & Habitat           — from habitat field + chunk evidence
  • Key Observations            — from rawTextEvidence + supporting chunks
  • References                  — from sourceCitation, deduplicated
"""
from __future__ import annotations

import logging
import re
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger("biotrace.wiki_agent_v56")

# ─────────────────────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpeciesChunkBundle:
    """All text chunks that contain evidence for one species."""
    species_name:   str
    occurrences:    list[dict]          # HITL-approved occurrence records
    chunks:         list[str]           # raw text chunks containing the species
    section_labels: list[str]           # which paper section each chunk came from
    citation:       str = ""


@dataclass
class WikiSectionDraft:
    """A single wiki section built by the LLM."""
    heading:   str
    content:   str
    confident: bool = True              # False when LLM signals low confidence


@dataclass
class WikiArticleDraft:
    """All sections assembled into a draft article for one species."""
    species_name:  str
    valid_name:    str
    sections:      list[WikiSectionDraft] = field(default_factory=list)
    references:    list[str]             = field(default_factory=list)
    taxonomy_meta: dict                  = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
#  CHUNK RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that match abbreviated genus references (e.g. "A. cornutus") by
# requiring the first character to match the species genus initial.
_ABBREV_RE = re.compile(r"\b[A-Z]\.\s+[a-z]{3,}")


def _genus_initial(species_name: str) -> str:
    """Return the first letter of the genus (uppercase), or '' if not parseable."""
    parts = species_name.strip().split()
    return parts[0][0].upper() if parts else ""


def score_chunk_for_species(chunk: str, species_name: str) -> float:
    """
    Heuristic relevance score [0.0–1.0] for a text chunk w.r.t. a species.

    Scoring:
      0.5   exact binomial match (case-insensitive)
      0.3   genus-only match
      0.15  abbreviated match (e.g. 'A. cornutus' when genus='Acanthurus')
      0.05  any single token of the name appears
    """
    name_lower  = species_name.lower()
    chunk_lower = chunk.lower()
    score       = 0.0

    # Full binomial / trinomial hit
    if name_lower in chunk_lower:
        score += 0.5

    parts = species_name.split()
    genus = parts[0] if parts else ""
    epithet = parts[1] if len(parts) > 1 else ""

    # Genus hit
    if genus and genus.lower() in chunk_lower:
        score += 0.3

    # Abbreviated genus (e.g. "A. cornutus")
    initial = _genus_initial(species_name)
    if epithet and initial:
        abbrev_pattern = re.compile(
            rf"\b{re.escape(initial)}\.\s+{re.escape(epithet)}\b", re.IGNORECASE
        )
        if abbrev_pattern.search(chunk):
            score += 0.15

    # Epithet alone (weak signal)
    if epithet and epithet.lower() in chunk_lower:
        score += 0.05

    return min(score, 1.0)


def retrieve_species_chunks(
    md_text:      str,
    species_name: str,
    chunk_chars:  int   = 4000,
    overlap:      int   = 400,
    min_score:    float = 0.25,
    max_chunks:   int   = 6,
) -> list[tuple[str, str]]:
    """
    Slide a window over the markdown and return (section_label, chunk_text)
    pairs that are relevant to *species_name*.

    Returns at most *max_chunks* pairs sorted by relevance descending.
    Falls back to section-boundary splitting when sections are available.
    """
    if not md_text.strip():
        return []

    # Prefer section-boundary splitting (looks for markdown headings)
    heading_re = re.compile(r"^#{1,4}\s+.+$", re.MULTILINE)
    boundaries = [0] + [m.start() for m in heading_re.finditer(md_text)]
    boundaries.append(len(md_text))

    candidates: list[tuple[float, str, str]] = []

    if len(boundaries) > 2:                       # at least one real section
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end   = boundaries[i + 1]
            chunk = md_text[start:end]
            if len(chunk) < 80:                   # too small — skip
                continue
            # Derive label from heading line
            first_line = chunk.strip().splitlines()[0].lstrip("#").strip()
            label = first_line[:60] or f"Section {i}"
            score = score_chunk_for_species(chunk, species_name)
            if score >= min_score:
                candidates.append((score, label, chunk[:chunk_chars]))
    else:
        # Flat sliding-window fallback
        step = max(chunk_chars - overlap, 1000)
        for i, start in enumerate(range(0, len(md_text), step)):
            chunk = md_text[start: start + chunk_chars]
            score = score_chunk_for_species(chunk, species_name)
            if score >= min_score:
                candidates.append((score, f"Chunk {i + 1}", chunk))

    # Sort by score descending, deduplicate by content prefix
    candidates.sort(key=lambda x: x[0], reverse=True)
    seen_prefixes: set[str] = set()
    results: list[tuple[str, str]] = []
    for _score, label, text in candidates:
        prefix = text[:120]
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            results.append((label, text))
        if len(results) >= max_chunks:
            break

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION-LEVEL PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

_TAXONOMY_SECTION_PROMPT = """\
You are a taxonomist writing the "Taxonomy & Classification" section of a
species wiki article.

Use the structured occurrence data AND the source text chunks below.
Write 2–5 concise sentences covering:
  • Authorship and original description year
  • Accepted scientific name and any common synonyms mentioned in the text
  • Taxonomic rank (species / subspecies / variety)
  • Kingdom → Phylum → Class → Order → Family hierarchy

OCCURRENCE TAXONOMY FIELDS (from verified database):
{taxonomy_block}

SOURCE TEXT CHUNKS:
{chunks_block}

Rules:
  • Use present tense, encyclopaedic voice.
  • If a field is empty or uncertain, omit it rather than invent.
  • Do NOT invent WoRMS IDs or authority strings not present in the data.
  • End with: [CONFIDENCE: HIGH / MEDIUM / LOW]

Write ONLY the section body text (no heading). Begin directly.
"""

_DISTRIBUTION_SECTION_PROMPT = """\
You are a marine/field biologist writing the "Distribution & Localities" section
of a species wiki article.

Use the locality records and chunk evidence below.

VERIFIED LOCALITY RECORDS:
{localities_block}

SOURCE TEXT CHUNKS:
{chunks_block}

Write 3–6 sentences covering:
  • Geographic range (countries, seas, ocean basins) inferred from localities
  • Depth range if mentioned in chunks
  • Whether records are primary (direct observation) or secondary (literature)
  • Any noteworthy locality or range extension reported

Rules:
  • Encyclopaedic voice, present tense for range, past tense for specific records.
  • Do NOT invent localities not present in the data.
  • End with: [CONFIDENCE: HIGH / MEDIUM / LOW]

Write ONLY the section body. Begin directly.
"""

_ECOLOGY_SECTION_PROMPT = """\
You are an ecologist writing the "Ecology & Habitat" section of a species wiki.

HABITAT FIELDS FROM DATABASE:
{habitat_block}

SOURCE TEXT CHUNKS (ecology-relevant):
{chunks_block}

Write 3–6 sentences covering:
  • Primary habitat types (coral reef, seagrass, mangrove, pelagic, etc.)
  • Substrate preferences or depth zones if mentioned
  • Feeding guild or trophic level if stated in the text
  • Co-occurring species or community associations mentioned

Rules:
  • Encyclopaedic voice.
  • Do not speculate beyond what the source text supports.
  • End with: [CONFIDENCE: HIGH / MEDIUM / LOW]

Write ONLY the section body. Begin directly.
"""

_KEY_OBSERVATIONS_PROMPT = """\
You are a biodiversity scientist writing the "Key Observations" section
of a species wiki, summarising the most informative evidence sentences
from a primary scientific paper.

RAW TEXT EVIDENCE FROM VERIFIED RECORDS:
{evidence_block}

ADDITIONAL SUPPORTING CHUNKS:
{chunks_block}

Write 3–8 bullet points (markdown "• ") each describing a distinct, notable
observation about this species from the source material.  Each bullet should:
  • State the observation precisely (what was found, where, when if known)
  • Attribute it to the paper (e.g. "Reported by [Author Year]")
  • Be self-contained (readable without context)

Rules:
  • Draw ONLY from the provided evidence.
  • Do not duplicate observations already covered in Taxonomy/Distribution.
  • End with: [CONFIDENCE: HIGH / MEDIUM / LOW]

Begin directly with the bullet list (no preamble).
"""

_CROSS_CHUNK_SYNTHESIS_PROMPT = """\
You are an editor reviewing a draft species wiki article for internal consistency.

DRAFT SECTIONS:
{draft_sections}

SOURCE SPECIES NAME: {species_name}

Perform a cross-section check:
1. Are there any factual contradictions between sections?
2. Is anything stated with high confidence that the source evidence does NOT support?
3. Are there any important facts mentioned in the evidence that were MISSED?

Respond ONLY with a short JSON object:
{{
  "contradictions": ["<issue 1>", ...],   // empty list if none
  "overconfident":  ["<claim>", ...],     // empty list if none
  "missed_facts":   ["<fact>", ...]       // empty list if none
}}

No prose. JSON only.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT CORE
# ─────────────────────────────────────────────────────────────────────────────

class WikiNarrativeAgent:
    """
    Builds a wiki article for a single species by reasoning over its
    HITL-approved occurrences and the relevant Docling text chunks.

    One instance per species. Stateless across species (AgentOrchestrator
    re-instantiates for each).
    """

    def __init__(
        self,
        call_llm_fn: Callable[[str], str],
        log_cb:      Callable[[str, str], None] | None = None,
    ):
        self.call_llm = call_llm_fn
        self._log     = log_cb or (lambda msg, lvl="ok": None)

    # ── Public API ─────────────────────────────────────────────────────────

    def build_article(
        self,
        bundle:   SpeciesChunkBundle,
    ) -> WikiArticleDraft | None:
        """
        Full agent pipeline for one species.
        Returns a WikiArticleDraft or None if not enough evidence.
        """
        sp   = bundle.species_name
        occs = bundle.occurrences

        if not occs and not bundle.chunks:
            self._log(f"[WikiAgent/{sp}] No occurrences or chunks — skip", "warn")
            return None

        self._log(f"[WikiAgent/{sp}] Building article from "
                  f"{len(occs)} occurrences, {len(bundle.chunks)} chunks…")

        taxonomy_meta = self._extract_taxonomy_meta(occs)
        draft         = WikiArticleDraft(
            species_name  = sp,
            valid_name    = taxonomy_meta.get("validName", sp),
            taxonomy_meta = taxonomy_meta,
        )

        # ── Section 1: Taxonomy ────────────────────────────────────────────
        tax_section = self._build_taxonomy_section(taxonomy_meta, bundle.chunks)
        if tax_section:
            draft.sections.append(tax_section)

        # ── Section 2: Distribution ───────────────────────────────────────
        dist_section = self._build_distribution_section(occs, bundle.chunks)
        if dist_section:
            draft.sections.append(dist_section)

        # ── Section 3: Ecology ────────────────────────────────────────────
        eco_section = self._build_ecology_section(occs, bundle.chunks)
        if eco_section:
            draft.sections.append(eco_section)

        # ── Section 4: Key Observations ───────────────────────────────────
        obs_section = self._build_observations_section(occs, bundle.chunks)
        if obs_section:
            draft.sections.append(obs_section)

        # ── Cross-chunk consistency check ─────────────────────────────────
        if len(draft.sections) >= 2:
            issues = self._cross_chunk_check(draft)
            if issues.get("overconfident"):
                self._log(
                    f"[WikiAgent/{sp}] Confidence issues: "
                    f"{issues['overconfident']}", "warn"
                )
            if issues.get("contradictions"):
                self._log(
                    f"[WikiAgent/{sp}] Contradictions found: "
                    f"{issues['contradictions']}", "warn"
                )

        # ── References ────────────────────────────────────────────────────
        draft.references = self._collect_references(occs)

        self._log(
            f"[WikiAgent/{sp}] Article draft: "
            f"{len(draft.sections)} sections, "
            f"{len(draft.references)} references ✅"
        )
        return draft

    # ── Section builders ──────────────────────────────────────────────────

    def _build_taxonomy_section(
        self,
        taxonomy_meta: dict,
        chunks: list[str],
    ) -> WikiSectionDraft | None:
        tax_block = "\n".join(
            f"  {k}: {v}" for k, v in taxonomy_meta.items() if v
        ) or "  (no taxonomy fields available)"

        chunks_block = self._format_chunks(chunks, max_total_chars=3000)

        prompt = _TAXONOMY_SECTION_PROMPT.format(
            taxonomy_block = tax_block,
            chunks_block   = chunks_block,
        )
        raw = self._safe_llm_call(prompt, section="Taxonomy")
        if not raw:
            return None

        content, confident = self._strip_confidence(raw)
        return WikiSectionDraft(
            heading   = "Taxonomy & Classification",
            content   = content,
            confident = confident,
        )

    def _build_distribution_section(
        self,
        occs:   list[dict],
        chunks: list[str],
    ) -> WikiSectionDraft | None:
        loc_lines = []
        for o in occs:
            loc   = o.get("verbatimLocality") or o.get("locality") or ""
            lat   = o.get("decimalLatitude")
            lon   = o.get("decimalLongitude")
            otype = o.get("occurrenceType", "Uncertain")
            coords = f"({lat:.4f}, {lon:.4f})" if (lat and lon) else ""
            loc_lines.append(f"  [{otype}] {loc} {coords}".rstrip())

        if not loc_lines:
            return None

        localities_block = "\n".join(loc_lines[:40])
        chunks_block     = self._format_chunks(chunks, max_total_chars=2500)

        prompt = _DISTRIBUTION_SECTION_PROMPT.format(
            localities_block = localities_block,
            chunks_block     = chunks_block,
        )
        raw = self._safe_llm_call(prompt, section="Distribution")
        if not raw:
            return None

        content, confident = self._strip_confidence(raw)
        return WikiSectionDraft(
            heading   = "Distribution & Localities",
            content   = content,
            confident = confident,
        )

    def _build_ecology_section(
        self,
        occs:   list[dict],
        chunks: list[str],
    ) -> WikiSectionDraft | None:
        habitats = list({
            o.get("habitat") or o.get("Habitat", "")
            for o in occs
            if o.get("habitat") or o.get("Habitat")
        })
        if not habitats and not chunks:
            return None

        habitat_block = "\n".join(f"  • {h}" for h in habitats) or "  (not recorded)"
        # Prefer chunks that mention ecology keywords
        eco_keywords  = ["habitat", "reef", "mangrove", "seagrass", "depth",
                         "trophic", "feed", "substrate", "benthic", "pelagic"]
        eco_chunks    = self._filter_chunks_by_keywords(chunks, eco_keywords, top_n=4)
        chunks_block  = self._format_chunks(eco_chunks, max_total_chars=2500)

        prompt = _ECOLOGY_SECTION_PROMPT.format(
            habitat_block = habitat_block,
            chunks_block  = chunks_block,
        )
        raw = self._safe_llm_call(prompt, section="Ecology")
        if not raw:
            return None

        content, confident = self._strip_confidence(raw)
        return WikiSectionDraft(
            heading   = "Ecology & Habitat",
            content   = content,
            confident = confident,
        )

    def _build_observations_section(
        self,
        occs:   list[dict],
        chunks: list[str],
    ) -> WikiSectionDraft | None:
        evidence_lines = [
            str(o.get("rawTextEvidence") or o.get("Raw Text Evidence") or "")
            for o in occs
            if o.get("rawTextEvidence") or o.get("Raw Text Evidence")
        ]
        if not evidence_lines and not chunks:
            return None

        evidence_block = "\n---\n".join(evidence_lines[:10])
        chunks_block   = self._format_chunks(chunks, max_total_chars=2000)

        prompt = _KEY_OBSERVATIONS_PROMPT.format(
            evidence_block = evidence_block or "(none)",
            chunks_block   = chunks_block,
        )
        raw = self._safe_llm_call(prompt, section="KeyObservations")
        if not raw:
            return None

        content, confident = self._strip_confidence(raw)
        return WikiSectionDraft(
            heading   = "Key Observations",
            content   = content,
            confident = confident,
        )

    # ── Cross-chunk consistency check ─────────────────────────────────────

    def _cross_chunk_check(self, draft: WikiArticleDraft) -> dict:
        draft_sections_str = "\n\n".join(
            f"=== {s.heading} ===\n{s.content}"
            for s in draft.sections
        )
        prompt = _CROSS_CHUNK_SYNTHESIS_PROMPT.format(
            draft_sections = draft_sections_str[:5000],
            species_name   = draft.species_name,
        )
        raw = self._safe_llm_call(prompt, section="CrossCheck")
        if not raw:
            return {}
        try:
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
            raw = raw.strip().lstrip("```json").rstrip("```").strip()
            return json.loads(raw)
        except Exception:
            return {}

    # ── Helpers ───────────────────────────────────────────────────────────

    def _extract_taxonomy_meta(self, occs: list[dict]) -> dict:
        """Aggregate taxonomy fields from the best occurrence record."""
        best = {}
        for o in occs:
            for k in ("validName", "phylum", "class_", "order_", "family_",
                      "taxonRank", "taxonomicStatus", "wormsID",
                      "nameAccordingTo", "kingdom"):
                if o.get(k) and not best.get(k):
                    best[k] = str(o[k]).strip()
            # Also check Higher Taxonomy dict
            tax = o.get("Higher Taxonomy") or o.get("higherTaxonomy") or {}
            if isinstance(tax, str):
                try:
                    tax = json.loads(tax)
                except Exception:
                    tax = {}
            for k in ("kingdom", "phylum", "class", "order", "family"):
                alt = k + "_" if k in ("class", "order", "family") else k
                if isinstance(tax, dict) and tax.get(k) and not best.get(alt):
                    best[alt] = str(tax[k]).strip()
        return best

    @staticmethod
    def _format_chunks(chunks: list[str], max_total_chars: int = 3000) -> str:
        if not chunks:
            return "(no relevant source text found)"
        parts   = []
        running = 0
        for i, ch in enumerate(chunks):
            snippet = ch[:max(200, max_total_chars // max(len(chunks), 1))]
            parts.append(f"[Chunk {i + 1}]\n{snippet}")
            running += len(snippet)
            if running >= max_total_chars:
                break
        return "\n\n".join(parts)

    @staticmethod
    def _filter_chunks_by_keywords(
        chunks: list[str],
        keywords: list[str],
        top_n: int = 4,
    ) -> list[str]:
        scored = []
        for ch in chunks:
            ch_lower = ch.lower()
            score    = sum(1 for kw in keywords if kw in ch_lower)
            scored.append((score, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ch for _, ch in scored[:top_n]]

    @staticmethod
    def _strip_confidence(text: str) -> tuple[str, bool]:
        """Remove trailing [CONFIDENCE: …] tag and return (content, is_high)."""
        text  = text.strip()
        # Strip <think> blocks from reasoning models
        text  = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        match = re.search(r"\[CONFIDENCE:\s*(HIGH|MEDIUM|LOW)\]", text, re.IGNORECASE)
        confident = True
        if match:
            level     = match.group(1).upper()
            confident = level == "HIGH"
            text      = text[:match.start()].rstrip()
        return text, confident

    @staticmethod
    def _collect_references(occs: list[dict]) -> list[str]:
        refs: list[str] = []
        seen: set[str]  = set()
        for o in occs:
            cit = str(o.get("sourceCitation") or o.get("Source Citation") or "").strip()
            if cit and cit not in seen:
                seen.add(cit)
                refs.append(cit)
        return refs

    def _safe_llm_call(self, prompt: str, section: str) -> str:
        try:
            raw = self.call_llm(prompt)
            if not raw or raw.strip().startswith("{\"error\""):
                self._log(f"[WikiAgent] LLM error on {section}", "warn")
                return ""
            return raw.strip()
        except Exception as exc:
            self._log(f"[WikiAgent] LLM call failed ({section}): {exc}", "warn")
            return ""


# ─────────────────────────────────────────────────────────────────────────────
#  WIKI COMMIT  — translates WikiArticleDraft → BioTraceWikiUnified format
# ─────────────────────────────────────────────────────────────────────────────

def commit_article_to_wiki(
    draft:       WikiArticleDraft,
    wiki,                               # BioTraceWikiUnified instance
    occurrences: list[dict],
    citation:    str,
    log_cb:      Callable[[str, str], None] | None = None,
) -> bool:
    """
    Write the draft article into the wiki store via update_from_occurrences().
    Falls back to a direct wiki.upsert_article() call if available.
    """
    _log = log_cb or (lambda msg, lvl="ok": None)
    sp   = draft.valid_name or draft.species_name

    # Build a section-keyed content dict for wiki
    narrative_sections: dict[str, str] = {}
    for sec in draft.sections:
        key = sec.heading.lower().replace(" ", "_").replace("&", "and")
        narrative_sections[key] = sec.content

    # Build a flat narrative string (fallback for wikis without section API)
    flat_narrative = "\n\n".join(
        f"## {s.heading}\n\n{s.content}"
        for s in draft.sections
    )

    try:
        # Preferred: per-species article upsert with sections
        if hasattr(wiki, "upsert_article_sections"):
            wiki.upsert_article_sections(
                species_name = sp,
                sections     = narrative_sections,
                references   = draft.references,
                taxonomy     = draft.taxonomy_meta,
                citation     = citation,
            )
            _log(f"[WikiCommit/{sp}] Article sections written ✅")
            return True

        # Fallback A: update_from_occurrences with pre-computed narrative
        if hasattr(wiki, "update_from_occurrences"):
            # Inject the narrative into a dummy LLM function that just returns
            # the pre-built text, bypassing a second LLM call.
            cached_narrative = flat_narrative

            def _prebuilt_llm(prompt: str) -> str:
                return cached_narrative

            wiki.update_from_occurrences(
                occurrences,
                citation          = citation,
                llm_fn            = _prebuilt_llm,
                update_narratives = True,
                chunk_text        = flat_narrative,
            )
            _log(f"[WikiCommit/{sp}] Committed via update_from_occurrences ✅")
            return True

        _log(f"[WikiCommit/{sp}] No compatible wiki method found", "warn")
        return False

    except Exception as exc:
        _log(f"[WikiCommit/{sp}] Commit error: {exc}", "warn")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    Drives WikiNarrativeAgent across all HITL-approved species in a batch.

    Call once per HITL confirmation event.  Thread-safe if each PDF extraction
    creates a fresh orchestrator instance (which AgentOrchestrator is).
    """

    def __init__(
        self,
        wiki,
        call_llm_fn: Callable[[str], str],
        log_cb:      Callable[[str, str], None] | None = None,
    ):
        self.wiki       = wiki
        self._llm       = call_llm_fn
        self._log       = log_cb or (lambda msg, lvl="ok": None)

    # ─────────────────────────────────────────────────────────────────────

    def run_for_batch(
        self,
        approved_occurrences: list[dict],
        md_text:              str,
        citation:             str,
        chunk_chars:          int   = 4000,
        overlap:              int   = 400,
        min_chunk_score:      float = 0.25,
        max_chunks_per_sp:    int   = 6,
    ) -> dict[str, bool]:
        """
        Build and commit wiki articles for every unique species in
        *approved_occurrences*.

        Parameters
        ──────────
        approved_occurrences : HITL-approved occurrence dicts
        md_text              : Full markdown from the source document
        citation             : Bibliographic citation string
        chunk_chars          : Window size for chunk retrieval
        overlap              : Overlap between windows
        min_chunk_score      : Minimum relevance score to include a chunk
        max_chunks_per_sp    : Cap on chunks used per species (LLM context budget)

        Returns
        ───────
        {species_name: True/False}  — True = article committed successfully
        """
        if not self.wiki:
            self._log("[AgentOrch] Wiki unavailable — skipping", "warn")
            return {}

        # Group occurrences by valid/recorded name
        species_map: dict[str, list[dict]] = {}
        for occ in approved_occurrences:
            sp = (occ.get("validName") or occ.get("recordedName") or "").strip()
            if sp:
                species_map.setdefault(sp, []).append(occ)

        if not species_map:
            self._log("[AgentOrch] No species found in batch", "warn")
            return {}

        self._log(
            f"[AgentOrch] Running wiki agent for "
            f"{len(species_map)} species…"
        )

        results: dict[str, bool] = {}
        agent = WikiNarrativeAgent(call_llm_fn=self._llm, log_cb=self._log)

        for sp, occs in species_map.items():
            # Retrieve text chunks relevant to this species
            chunk_pairs = retrieve_species_chunks(
                md_text      = md_text,
                species_name = sp,
                chunk_chars  = chunk_chars,
                overlap      = overlap,
                min_score    = min_chunk_score,
                max_chunks   = max_chunks_per_sp,
            )
            chunks         = [text for _label, text in chunk_pairs]
            section_labels = [label for label, _text in chunk_pairs]

            bundle = SpeciesChunkBundle(
                species_name   = sp,
                occurrences    = occs,
                chunks         = chunks,
                section_labels = section_labels,
                citation       = citation,
            )

            self._log(
                f"[AgentOrch/{sp}] "
                f"{len(occs)} occurrences, "
                f"{len(chunks)} relevant chunks "
                f"(sections: {', '.join(section_labels[:3]) or 'none'})"
            )

            draft = agent.build_article(bundle)
            if draft is None:
                results[sp] = False
                continue

            ok = commit_article_to_wiki(
                draft       = draft,
                wiki        = self.wiki,
                occurrences = occs,
                citation    = citation,
                log_cb      = self._log,
            )
            results[sp] = ok

        n_ok   = sum(1 for v in results.values() if v)
        n_fail = len(results) - n_ok
        self._log(
            f"[AgentOrch] Done — "
            f"{n_ok}/{len(results)} articles committed "
            f"({n_fail} failed)"
        )
        return results

    # ─────────────────────────────────────────────────────────────────────
    #  Streamlit helper (optional)
    # ─────────────────────────────────────────────────────────────────────

    def render_streamlit_progress(
        self,
        results:  dict[str, bool],
        container = None,
    ) -> None:
        """
        Render a compact Streamlit table showing per-species agent status.
        Pass a st.container() to scope the output, or leave None for global.
        """
        try:
            import streamlit as st
            _st = container or st
        except ImportError:
            return

        if not results:
            return

        rows = [
            {"Species": sp, "Wiki": "✅ Committed" if ok else "⚠️ Failed"}
            for sp, ok in results.items()
        ]
        import pandas as pd
        _st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  PATCH NOTES FOR biotrace_v53.py
# ─────────────────────────────────────────────────────────────────────────────
"""
Required changes to biotrace_v53.py
════════════════════════════════════

## Change 1 — Store md_text in session state when saving HITL checkpoint
   (inside `if run_btn and uploaded:` → HITL checkpoint block)

   Before:
       st.session_state["_hitl_pending_occurrences"] = occurrences
       st.session_state["_hitl_pending_hash"]        = file_hash
       ...

   After:
       st.session_state["_hitl_pending_occurrences"] = occurrences
       st.session_state["_hitl_pending_hash"]        = file_hash
       st.session_state["_hitl_pending_md_text"]     = md_text   # ← ADD THIS
       ...


## Change 2 — After HITL confirmation, run AgentOrchestrator
   (inside `_hitl_resume` block, after ingest_into_v5_systems call)

   Add these lines:

       # ── Agentic wiki narrative building (post-HITL) ──────────────────
       _md_text = st.session_state.pop("_hitl_pending_md_text", "")
       if _md_text and use_wiki and _WIKI_AGENT_AVAILABLE:
           try:
               from biotrace_wiki_agent_v56 import AgentOrchestrator
               _orch = AgentOrchestrator(
                   wiki        = wiki_inst or get_wiki(),
                   call_llm_fn = lambda p: call_llm(p, provider, model_sel, api_key, ollama_url),
                   log_cb      = _lcb,
               )
               _agent_results = _orch.run_for_batch(
                   approved_occurrences = _appr,
                   md_text              = _md_text,
                   citation             = _h_cite,
               )
               _orch.render_streamlit_progress(_agent_results)
           except Exception as _ae:
               _lcb(f"[WikiAgent] {_ae}", "warn")


## Change 3 — Fix duplicate wiki update in ingest_into_v5_systems()
   The function currently calls get_wiki() and wiki.update_from_occurrences()
   TWICE (the second block is a leftover copy at the bottom).

   Remove the second block (lines starting with:
       "# Wiki (BioTraceWikiUnified — versioned, CSS-styled, LLM-enhanced)"
       wiki = get_wiki() if use_wiki else None
       if wiki:
           try:
               counts = wiki.update_from_occurrences( ...
   This avoids the wiki article being overwritten immediately by a call with
   no chunk_text, which wipes out the narrative the agent just built.

## Change 4 — Pass wiki_inst to ingest_into_v5_systems() in the HITL resume block
   The HITL resume currently calls ingest_into_v5_systems without a pre-built
   wiki instance; the function calls get_wiki() internally.  If you want the
   agent to use the same patched singleton, pass it explicitly or rely on
   get_wiki() returning the v56 patched instance (which it does after
   install_v56_patches() runs at startup).  No code change needed here as long
   as install_v56_patches() has been called.
"""
