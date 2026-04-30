"""
biotrace_wiki_agent.py  —  BioTrace v5.5  Standalone Wiki Architect Agent
──────────────────────────────────────────────────────────────────────────────
Agentic, Ollama-powered, pydantic-ai-based pipeline for dynamic wiki creation.

Architecture
────────────
  OllamaWikiAgent
    ├── WikiArchitectAgent   — master editor  (writes / merges wiki sections)
    ├── TaxonomyAgent        — fills Taxobox + Authority from WoRMS / text
    ├── ConflictAgent        — detects source disagreements, emits ConflictNote
    └── CitationAgent        — normalises citations, deduplicates references

All agents communicate via structured Pydantic models.
The orchestrate() method calls them in sequence and writes the result to
BioTraceWikiUnified.

Usage
─────
    from biotrace_wiki_agent import OllamaWikiAgent
    from biotrace_wiki_unified import BioTraceWikiUnified

    wiki  = BioTraceWikiUnified("biodiversity_data/wiki")
    agent = OllamaWikiAgent(wiki=wiki, model="qwen2.5:14b")

    # Enhance from a PDF chunk
    result = agent.orchestrate(
        species_name = "Anteaeolidiella indica",
        chunk_text   = open("chunk.txt").read(),
        citation     = "Bhave & Apte (2011) J. Bombay Nat. Hist. Soc. 108(3)",
    )
    print(result.summary)
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

logger = logging.getLogger("biotrace.wiki_agent")

# ── Required: pydantic ────────────────────────────────────────────────────────
try:
    from pydantic import BaseModel, Field, validator
except ImportError:
    raise ImportError("pydantic is required: pip install pydantic>=2.0")

# ── Required: ollama Python client ───────────────────────────────────────────
try:
    import ollama as _ollama
    _OLLAMA_OK = True
except ImportError:
    _OLLAMA_OK = False
    logger.warning("[WikiAgent] ollama package not found: pip install ollama")

# ── Optional: pydantic-ai ─────────────────────────────────────────────────────
try:
    from pydantic_ai import Agent as _PAIAgent
    from pydantic_ai.models.ollama import OllamaModel as _PAIOllamaModel
    _PAI_OK = True
except ImportError:
    _PAI_OK = False
    logger.info("[WikiAgent] pydantic-ai not installed — using raw Ollama client")


# ─────────────────────────────────────────────────────────────────────────────
#  PYDANTIC DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

class TaxonBox(BaseModel):
    """Structured taxobox populated by TaxonomyAgent."""
    kingdom:         str = ""
    phylum:          str = ""
    class_:          str = Field("", alias="class")
    order_:          str = Field("", alias="order")
    family_:         str = Field("", alias="family")
    genus:           str = ""
    species_epithet: str = ""
    authority:       str = ""    # "Bergh, 1888"
    taxonRank:       str = "species"
    taxonomicStatus: str = "unverified"
    wormsID:         str = ""
    iucnStatus:      str = ""

    model_config = {"populate_by_name": True}


class ConflictNote(BaseModel):
    """A single source conflict detected by ConflictAgent."""
    field:   str         # e.g. "depth_range_m"
    values:  list[str]   # ["Bhave (2011): 5 m", "Smith (2024): 12 m"]


class WikiSections(BaseModel):
    """Free-text markdown sections for a species article."""
    lead:                 str = ""
    taxonomy_phylogeny:   str = ""
    morphology:           str = ""
    distribution_habitat: str = ""
    ecology_behaviour:    str = ""
    conservation:         str = ""
    specimen_records:     str = ""


class AgentResult(BaseModel):
    """Final result returned by OllamaWikiAgent.orchestrate()."""
    species_name:  str
    taxobox:       TaxonBox
    sections:      WikiSections
    conflicts:     list[ConflictNote]   = []
    citations:     list[str]            = []
    summary:       str                  = ""
    tokens_used:   int                  = 0
    elapsed_s:     float                = 0.0
    errors:        list[str]            = []


# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

_SYS_TAXONOMY = """\
You are a marine taxonomist. Extract ONLY the classification hierarchy and
nomenclatural data from the provided text.
Return a JSON object exactly matching this schema — no extra keys:
{
  "kingdom": "", "phylum": "", "class": "", "order": "",
  "family": "", "genus": "", "species_epithet": "",
  "authority": "",          // "Surname, YYYY" format
  "taxonRank": "species",
  "taxonomicStatus": "accepted|unverified|synonym",
  "wormsID": "",
  "iucnStatus": ""          // "LC|NT|VU|EN|CR|DD" or ""
}
If a field is not found in the text, leave it as an empty string.
Return ONLY the JSON. No prose, no markdown fences.
"""

_SYS_CONFLICT = """\
You are a scientific editor checking for conflicting data between sources.
Given (a) existing article data and (b) new source text, identify ANY fields
where the two sources report DIFFERENT values (e.g. different depth ranges,
different size measurements, different type localities).
Return a JSON array of conflict objects:
[{"field": "depth_range_m", "values": ["Source A: 5 m", "Source B: 12 m"]}, ...]
If no conflicts, return [].
Return ONLY the JSON array. No prose, no markdown fences.
"""

_SYS_WIKI_ARCHITECT = """\
You are a Professional Taxonomist and Wikipedia-style Wiki Editor specialising
in marine invertebrates and Indian Ocean biodiversity.

TASK: Given the CURRENT article sections JSON and NEW source text, update each
section following these rules:
1. NEVER delete existing valid data — only append or refine.
2. Conflict resolution: if a field differs between sources, list BOTH with
   inline citations: "Bhave (2011) reports 5 m; Smith (2024) reports 12 m."
3. Fill blank fields when the new text provides data.
4. Use *italics* for binomial nomenclature.
5. Use **bold** for key technical terms.
6. Maintain neutral, encyclopaedic tone.

Return a JSON object with keys: lead, taxonomy_phylogeny, morphology,
distribution_habitat, ecology_behaviour, conservation, specimen_records.
Each value is a markdown string (or "" if no content).
Return ONLY the JSON object. No prose, no markdown fences.
"""

_SYS_CITATION = """\
You are a scientific librarian. Normalise the following citation string to
the format:  AuthorSurname et al. (YEAR) Journal Title Volume(Issue): Pages.
Return ONLY the normalised citation as a plain string, no explanation.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  LOW-LEVEL OLLAMA CALL  (works without pydantic-ai)
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_call(
    model:      str,
    system:     str,
    user:       str,
    base_url:   str = "http://localhost:11434",
    temperature:float = 0.15,
    timeout_s:  int   = 120,
) -> tuple[str, int]:
    """
    Call Ollama chat API, return (response_text, total_tokens).
    Falls back gracefully if ollama SDK is missing.
    """
    if not _OLLAMA_OK:
        raise RuntimeError("ollama package not installed: pip install ollama")

    client = _ollama.Client(host=base_url)
    resp   = client.chat(
        model   = model,
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user[:12000]},   # guard context window
        ],
        options = {"temperature": temperature},
    )
    text   = resp["message"]["content"]
    tokens = resp.get("eval_count", 0) + resp.get("prompt_eval_count", 0)
    return text, tokens


def _strip_fences(text: str) -> str:
    """Remove ```json / ``` fences from LLM output."""
    text = re.sub(r"^```+(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```+$",          "", text.strip(), flags=re.MULTILINE)
    return text.strip()


def _safe_json(text: str, fallback: Any = None) -> Any:
    """Parse JSON from LLM output, tolerating common errors."""
    try:
        return json.loads(_strip_fences(text))
    except Exception:
        # Try extracting the first {...} or [...] block
        for pat in (r"(\{.*\})", r"(\[.*\])"):
            m = re.search(pat, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
#  PYDANTIC-AI WRAPPER  (optional — richer tracing + retries)
# ─────────────────────────────────────────────────────────────────────────────

class _PAIWrapper:
    """
    Thin wrapper around pydantic-ai Agent, used when pydantic-ai is installed.
    Provides automatic structured output parsing + built-in retry logic.
    """

    def __init__(self, model: str, system: str, result_model, base_url: str):
        if not _PAI_OK:
            raise RuntimeError("pydantic-ai not installed: pip install pydantic-ai")
        ollama_model = _PAIOllamaModel(model, base_url=base_url)
        self._agent  = _PAIAgent(
            model         = ollama_model,
            system_prompt = system,
            result_type   = result_model,
            retries       = 2,
        )

    def run(self, user: str):
        result = self._agent.run_sync(user)
        return result.data


# ─────────────────────────────────────────────────────────────────────────────
#  SPECIALISED AGENTS
# ─────────────────────────────────────────────────────────────────────────────

class _TaxonomyAgent:
    """Extracts and validates taxobox fields using TaxonBox Pydantic model."""

    def __init__(self, model: str, base_url: str, use_pai: bool):
        self.model    = model
        self.base_url = base_url
        self.use_pai  = use_pai and _PAI_OK

    def run(self, text: str, existing: TaxonBox) -> tuple[TaxonBox, int]:
        user = (
            f"EXISTING TAXON DATA:\n{existing.model_dump_json(indent=2)}\n\n"
            f"NEW SOURCE TEXT:\n{text[:4000]}"
        )
        if self.use_pai:
            try:
                wrapper = _PAIWrapper(self.model, _SYS_TAXONOMY, TaxonBox, self.base_url)
                result  = wrapper.run(user)
                return result, 0
            except Exception as exc:
                logger.warning("[TaxonomyAgent/PAI] %s — falling back to raw", exc)

        raw, toks = _ollama_call(self.model, _SYS_TAXONOMY, user, self.base_url)
        data      = _safe_json(raw, {})
        if not data:
            return existing, toks
        # Merge: only fill blank fields in existing
        merged = existing.model_dump(by_alias=False)
        for k, v in data.items():
            norm_k = k.rstrip("_")   # "class" → "class"
            # map back to pydantic field names
            field_map = {"class": "class_", "order": "order_", "family": "family_"}
            fk = field_map.get(norm_k, norm_k)
            if fk in merged and not merged[fk] and v:
                merged[fk] = str(v).strip()
        try:
            return TaxonBox(**merged), toks
        except Exception:
            return existing, toks


class _ConflictAgent:
    """Detects field conflicts between existing article data and new text."""

    def __init__(self, model: str, base_url: str, use_pai: bool):
        self.model    = model
        self.base_url = base_url
        self.use_pai  = use_pai and _PAI_OK

    def run(self, existing_json: str, new_text: str) -> tuple[list[ConflictNote], int]:
        user = (
            f"EXISTING ARTICLE (JSON):\n{existing_json[:3000]}\n\n"
            f"NEW SOURCE TEXT:\n{new_text[:3000]}"
        )
        raw, toks = _ollama_call(self.model, _SYS_CONFLICT, user, self.base_url)
        data      = _safe_json(raw, [])
        if not isinstance(data, list):
            return [], toks
        conflicts = []
        for item in data:
            if isinstance(item, dict) and item.get("field") and item.get("values"):
                conflicts.append(ConflictNote(**item))
        return conflicts, toks


class _WikiArchitectAgent:
    """
    Core Wiki Architect — merges new text into existing sections.
    This is the primary creative / editorial agent.
    """

    def __init__(self, model: str, base_url: str, use_pai: bool):
        self.model    = model
        self.base_url = base_url
        self.use_pai  = use_pai and _PAI_OK

    def run(
        self,
        existing_sections: WikiSections,
        new_text:          str,
        citation:          str,
        taxobox:           TaxonBox,
    ) -> tuple[WikiSections, int]:
        user = (
            f"CITATION: {citation}\n\n"
            f"CURRENT SECTIONS (JSON):\n{existing_sections.model_dump_json(indent=2)[:4000]}\n\n"
            f"TAXOBOX CONTEXT:\n{taxobox.model_dump_json(indent=2)[:1000]}\n\n"
            f"NEW SOURCE TEXT:\n{new_text[:5000]}"
        )
        if self.use_pai:
            try:
                wrapper = _PAIWrapper(self.model, _SYS_WIKI_ARCHITECT,
                                      WikiSections, self.base_url)
                result  = wrapper.run(user)
                return result, 0
            except Exception as exc:
                logger.warning("[WikiArchitect/PAI] %s — falling back to raw", exc)

        raw, toks = _ollama_call(self.model, _SYS_WIKI_ARCHITECT, user,
                                 self.base_url, temperature=0.25)
        data      = _safe_json(raw, {})
        if not data or not isinstance(data, dict):
            return existing_sections, toks

        # Non-destructive merge: append new content where existing is blank,
        # or append non-duplicate content at the end of existing sections.
        merged = existing_sections.model_dump()
        for k, new_val in data.items():
            if k not in merged or not new_val:
                continue
            existing_val = merged[k]
            if not existing_val:
                merged[k] = new_val
            elif new_val.strip() and new_val.strip() not in existing_val:
                merged[k] = existing_val.rstrip() + "\n\n" + new_val.strip()
        try:
            return WikiSections(**merged), toks
        except Exception:
            return existing_sections, toks


class _CitationAgent:
    """Normalises a raw citation string to a standard format."""

    def __init__(self, model: str, base_url: str):
        self.model    = model
        self.base_url = base_url

    def run(self, raw_citation: str) -> str:
        if not raw_citation.strip():
            return raw_citation
        try:
            text, _ = _ollama_call(
                self.model, _SYS_CITATION,
                f"RAW CITATION: {raw_citation}",
                self.base_url, temperature=0.0,
            )
            return text.strip()[:300] or raw_citation
        except Exception:
            return raw_citation


# ─────────────────────────────────────────────────────────────────────────────
#  ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class OllamaWikiAgent:
    """
    Top-level orchestrator for multi-agent wiki creation and enhancement.

    Parameters
    ──────────
    wiki        BioTraceWikiUnified instance (handles storage & versioning)
    model       Ollama model tag, e.g. "qwen2.5:14b", "llama3.2:8b"
    base_url    Ollama server URL (default: http://localhost:11434)
    use_pai     If True and pydantic-ai is installed, use it for structured
                output + automatic retries (recommended).
    """

    def __init__(
        self,
        wiki,              # BioTraceWikiUnified — typed loosely to avoid circular import
        model:     str   = "qwen2.5:14b",
        base_url:  str   = "http://localhost:11434",
        use_pai:   bool  = True,
    ):
        self.wiki     = wiki
        self.model    = model
        self.base_url = base_url
        self.use_pai  = use_pai and _PAI_OK

        # Instantiate sub-agents
        self._tax_agent    = _TaxonomyAgent(model, base_url, self.use_pai)
        self._conf_agent   = _ConflictAgent(model, base_url, self.use_pai)
        self._arch_agent   = _WikiArchitectAgent(model, base_url, self.use_pai)
        self._cite_agent   = _CitationAgent(model, base_url)

        logger.info("[WikiAgent] Initialised — model=%s pai=%s", model, self.use_pai)

    # ── Public API ────────────────────────────────────────────────────────────

    def orchestrate(
        self,
        species_name: str,
        chunk_text:   str,
        citation:     str  = "",
        verbose:      bool = True,
    ) -> AgentResult:
        """
        Full agentic pipeline for one species + one text chunk.

        Stages
        ──────
        1. CitationAgent    — normalise citation string
        2. Load existing    — pull current article from wiki SQLite store
        3. TaxonomyAgent    — fill / update taxobox fields
        4. ConflictAgent    — detect source disagreements
        5. WikiArchitectAgent — merge new text into article sections
        6. Write back       — versioned update via BioTraceWikiUnified
        """
        t0          = time.time()
        total_toks  = 0
        errors: list[str] = []

        def _log(msg: str):
            if verbose:
                logger.info("[WikiAgent] %s", msg)

        # ── Stage 1: Normalise citation ───────────────────────────────────
        _log("Stage 1 — CitationAgent")
        try:
            citation = self._cite_agent.run(citation or species_name)
        except Exception as exc:
            errors.append(f"CitationAgent: {exc}")

        # ── Stage 2: Load existing article ───────────────────────────────
        _log("Stage 2 — Load existing article")
        existing_art = self.wiki.get_species_article(species_name) or {}
        existing_secs_dict = existing_art.get("sections", {})
        existing_sections  = WikiSections(**{
            k: existing_secs_dict.get(k, "")
            for k in WikiSections.model_fields
        })
        existing_taxobox = TaxonBox(
            kingdom         = existing_art.get("kingdom",          ""),
            phylum          = existing_art.get("phylum",           ""),
            **{"class":       existing_art.get("class_",          "")},
            **{"order":       existing_art.get("order_",          "")},
            **{"family":      existing_art.get("family_",         "")},
            genus           = existing_art.get("genus",            ""),
            species_epithet = existing_art.get("species_epithet",  ""),
            authority       = existing_art.get("authority",        ""),
            taxonRank       = existing_art.get("taxonRank",        "species"),
            taxonomicStatus = existing_art.get("taxonomicStatus",  "unverified"),
            wormsID         = existing_art.get("wormsID",          ""),
            iucnStatus      = existing_art.get("iucnStatus",       ""),
        )

        # ── Stage 3: TaxonomyAgent ────────────────────────────────────────
        _log("Stage 3 — TaxonomyAgent")
        try:
            updated_taxobox, t_toks = self._tax_agent.run(chunk_text, existing_taxobox)
            total_toks += t_toks
        except Exception as exc:
            errors.append(f"TaxonomyAgent: {exc}")
            updated_taxobox = existing_taxobox

        # ── Stage 4: ConflictAgent ────────────────────────────────────────
        _log("Stage 4 — ConflictAgent")
        conflicts: list[ConflictNote] = []
        try:
            existing_json = json.dumps(
                {k: existing_art.get(k, "")
                 for k in ("phylum","order_","family_","authority",
                            "depth_range_raw","body_length_mm")},
                indent=2,
            )
            conflicts, c_toks = self._conf_agent.run(existing_json, chunk_text)
            total_toks += c_toks
            if conflicts:
                _log(f"  {len(conflicts)} conflicts detected")
        except Exception as exc:
            errors.append(f"ConflictAgent: {exc}")

        # ── Stage 5: WikiArchitectAgent ───────────────────────────────────
        _log("Stage 5 — WikiArchitectAgent (core)")
        try:
            updated_sections, a_toks = self._arch_agent.run(
                existing_sections, chunk_text, citation, updated_taxobox
            )
            total_toks += a_toks
        except Exception as exc:
            errors.append(f"WikiArchitectAgent: {exc}")
            updated_sections = existing_sections

        # ── Stage 6: Write back to wiki store ────────────────────────────
        _log("Stage 6 — Write back to wiki store")
        try:
            # Reconstruct article dict from current store and merge updates
            from biotrace_wiki_unified import _blank_species_article
            art = existing_art or _blank_species_article(species_name)

            # Taxobox fields
            tb_data = updated_taxobox.model_dump(by_alias=False)
            for k, v in tb_data.items():
                if v and not art.get(k):
                    art[k] = v

            # Sections (non-destructive)
            sec_data = updated_sections.model_dump()
            art.setdefault("sections", {})
            for k, v in sec_data.items():
                old = art["sections"].get(k, "")
                if not old:
                    art["sections"][k] = v
                elif v and v.strip() and v.strip() not in old:
                    art["sections"][k] = old.rstrip() + "\n\n" + v.strip()

            # Conflicts
            if conflicts:
                art.setdefault("depth_conflicts",  [])
                for cf in conflicts:
                    if "depth" in cf.field.lower():
                        art["depth_conflicts"].append({"sources": cf.values})

            # Provenance
            art.setdefault("provenance", [])
            if not any(p.get("citation") == citation for p in art["provenance"]):
                art["provenance"].append({
                    "citation": citation,
                    "date":     __import__("datetime").datetime.now().isoformat(),
                    "agent":    self.model,
                    "enhanced": True,
                })

            slug = self.wiki._slug(species_name)
            self.wiki._write("species", slug, species_name, art,
                             change_note=f"agent-enhanced: {citation[:60]}")
            _log("  Article written (versioned)")
        except Exception as exc:
            errors.append(f"WriteBack: {exc}")

        elapsed = round(time.time() - t0, 2)
        _log(f"Done in {elapsed}s — {total_toks} tokens — {len(errors)} errors")

        return AgentResult(
            species_name  = species_name,
            taxobox       = updated_taxobox,
            sections      = updated_sections,
            conflicts     = conflicts,
            citations     = [citation],
            summary       = (
                f"Processed '{species_name}' in {elapsed}s using {self.model}. "
                f"Sections updated: "
                f"{[k for k,v in updated_sections.model_dump().items() if v]}. "
                f"Conflicts: {len(conflicts)}. Errors: {len(errors)}."
            ),
            tokens_used   = total_toks,
            elapsed_s     = elapsed,
            errors        = errors,
        )

    def batch_enhance(
        self,
        species_chunk_pairs: list[tuple[str, str, str]],
        verbose: bool = True,
    ) -> list[AgentResult]:
        """
        Batch mode: list of (species_name, chunk_text, citation) tuples.
        Processes sequentially; safe to call from background thread.
        """
        results = []
        for i, (sp, chunk, cite) in enumerate(species_chunk_pairs, 1):
            logger.info("[WikiAgent] Batch %d/%d — %s", i, len(species_chunk_pairs), sp)
            try:
                r = self.orchestrate(sp, chunk, cite, verbose=verbose)
                results.append(r)
            except Exception as exc:
                logger.error("[WikiAgent] Batch error for %s: %s", sp, exc)
                results.append(AgentResult(
                    species_name=sp,
                    taxobox=TaxonBox(),
                    sections=WikiSections(),
                    errors=[str(exc)],
                ))
        return results

    # ── Streamlit UI panel ────────────────────────────────────────────────────

    def render_agent_panel(self):
        """
        Render a compact Streamlit control panel for the agent.
        Call inside any Streamlit tab or expander.
        """
        try:
            import streamlit as st
        except ImportError:
            return

        st.markdown("#### 🤖 Ollama Wiki Architect Agent")
        pai_badge = "✅ pydantic-ai" if _PAI_OK else "⚠️ raw ollama"
        st.caption(f"Model: `{self.model}` · {pai_badge} · {self.base_url}")

        sp_input = st.text_input("Species name:", key="agent_sp")
        chunk_input = st.text_area("PDF chunk / text:", height=160, key="agent_chunk")
        cite_input  = st.text_input("Citation:", key="agent_cite")

        if st.button("🚀 Run Wiki Agent", key="agent_run_btn"):
            if sp_input and chunk_input:
                with st.spinner(f"Running {self.model} — multi-stage pipeline…"):
                    result = self.orchestrate(sp_input, chunk_input, cite_input)
                st.success(result.summary)
                if result.errors:
                    st.warning("Errors: " + "; ".join(result.errors))
                if result.conflicts:
                    with st.expander(f"⚠️ {len(result.conflicts)} Conflicts Detected"):
                        for cf in result.conflicts:
                            st.write(f"**{cf.field}:** " + " vs ".join(cf.values))
                with st.expander("Taxobox"):
                    st.json(result.taxobox.model_dump())
                with st.expander("Sections Preview"):
                    for k, v in result.sections.model_dump().items():
                        if v:
                            st.markdown(f"**{k}**\n\n{v[:300]}…")
                st.metric("Tokens", result.tokens_used)
                st.metric("Elapsed (s)", result.elapsed_s)
            else:
                st.warning("Enter a species name and text chunk.")
