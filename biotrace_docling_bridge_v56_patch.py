"""
biotrace_docling_bridge_v56_patch.py  —  BioTrace v5.6
═══════════════════════════════════════════════════════════════════════════════
Patches DoclingWikiBridge so that:

  1. Wiki agent runs by DEFAULT during extraction using docling sections
     (run_wiki_agent=False → sections are fed directly; no agent is called
      for every chunk — only for species flagged as "priority")

  2. MD cache is checked before docling conversion to avoid reprocessing
     Integrates DoclingMDCache into the PDF → Markdown pipeline.

  3. PydanticAIChunkValidator is called after each LLM extraction to
     clean and structure occurrence records before ingest.

Integration
───────────
    from biotrace_docling_bridge_v56_patch import patch_docling_bridge
    patch_docling_bridge()

Or use the convenience wrapper:
    from biotrace_docling_bridge_v56_patch import convert_pdf_cached

    md, sections = convert_pdf_cached(pdf_path, docling_converter, cache)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("biotrace.docling_bridge_patch")


# ─────────────────────────────────────────────────────────────────────────────
#  FIX: MD-cached PDF → Markdown conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_pdf_cached(
    pdf_path:         str,
    docling_converter,       # DoclingDocumentConverter instance
    cache,                   # DoclingMDCache instance (or None to skip cache)
    section_extractor=None,  # optional callable(DoclingDocument) → sections dict
) -> tuple[str, dict]:
    """
    Convert a PDF to Markdown using docling, with MD-cache check first.

    Returns
    -------
    (markdown_text, sections_dict)
        sections_dict: {section_role: text, ...}  e.g.
            {"body": "...", "table": "...", "reference": "...", "title": "..."}

    If the cache already has this PDF's hash, returns cached result immediately.
    Otherwise runs docling, stores the result, and returns it.
    """
    # 1. Cache hit
    if cache is not None:
        cached_md, cached_secs = cache.get(pdf_path)
        if cached_md is not None:
            logger.info("[DoclingPatch] Cache HIT for %s", Path(pdf_path).name)
            return cached_md, cached_secs

    # 2. Run docling
    logger.info("[DoclingPatch] Running docling on %s", Path(pdf_path).name)
    try:
        result  = docling_converter.convert(pdf_path)
        doc     = result.document
        md_text = doc.export_to_markdown()

        # 3. Extract sections
        sections: dict[str, str] = {}
        if section_extractor:
            sections = section_extractor(doc)
        else:
            sections = _default_section_extractor(doc, md_text)

        # 4. Store in cache
        if cache is not None:
            page_count = getattr(doc, "num_pages", 0) or 0
            cache.put(pdf_path, md_text, sections, page_count=page_count)

        return md_text, sections

    except Exception as exc:
        logger.error("[DoclingPatch] Docling conversion failed for %s: %s", pdf_path, exc)
        return "", {}


def _default_section_extractor(doc, markdown: str) -> dict[str, str]:
    """
    Extract sections from a DoclingDocument by element type.
    Falls back to heuristic markdown splitting if docling elements unavailable.
    """
    sections: dict[str, list[str]] = {}

    # Try docling's element-level API
    try:
        for elem in doc.elements:
            role = getattr(elem, "label", "body") or "body"
            text = getattr(elem, "text", "") or ""
            if text.strip():
                sections.setdefault(role, []).append(text.strip())

        return {k: "\n\n".join(v) for k, v in sections.items() if v}
    except Exception:
        pass

    # Heuristic fallback: split markdown by common section headers
    import re
    current_section = "body"
    current_lines:  list[str] = []
    result: dict[str, list[str]] = {}

    _SEC_MAP = {
        r"reference[s]?":     "reference",
        r"method[s]?":        "methods",
        r"material[s]?":      "materials",
        r"result[s]?":        "results",
        r"discussion":        "discussion",
        r"introduction":      "introduction",
        r"abstract":          "abstract",
        r"acknowledge?ment[s]?": "acknowledgement",
        r"appendix":          "appendix",
        r"table":             "table",
        r"figure":            "figure",
    }

    for line in markdown.splitlines():
        if line.startswith("#"):
            # Flush
            if current_lines:
                result.setdefault(current_section, []).extend(current_lines)
                current_lines = []
            # Detect new section
            heading_text = re.sub(r"^#+\s*", "", line).strip().lower()
            detected = "body"
            for pattern, role in _SEC_MAP.items():
                if re.search(pattern, heading_text):
                    detected = role
                    break
            current_section = detected
        else:
            if line.strip():
                current_lines.append(line)

    if current_lines:
        result.setdefault(current_section, []).extend(current_lines)

    return {k: "\n".join(v) for k, v in result.items() if v}


# ─────────────────────────────────────────────────────────────────────────────
#  FIX: process_document using cached sections → wiki update
# ─────────────────────────────────────────────────────────────────────────────

def process_document_patched(
    self,
    doc_sections:   dict,
    occurrences:    list[dict],
    citation:       str,
    species_names:  list[str],
    # NEW params:
    run_wiki_agent: Optional[bool] = None,
    priority_species: Optional[list[str]] = None,
):
    """
    Patched DoclingWikiBridge.process_document().

    Changes vs original:
    • run_wiki_agent defaults to False (docling sections feed wiki directly)
    • priority_species: run full wiki agent only for these species
    • PydanticAIChunkValidator applied to occurrences before wiki update
    """
    # Override instance flag if caller passes explicit value
    if run_wiki_agent is not None:
        self.run_wiki_agent = run_wiki_agent

    # 1. Validate occurrences with PydanticAI if available
    validated_occs = _validate_occurrences(occurrences, doc_sections)

    # 2. Run the original process_document with validated occs
    try:
        _orig = _ORIG_PROCESS_DOCUMENT
        if _orig:
            _orig(self, doc_sections, validated_occs, citation, species_names)
    except Exception as exc:
        logger.warning("[DoclingPatch] original process_document error: %s", exc)

    # 3. For priority species only, run the full wiki agent
    if priority_species:
        body_text = doc_sections.get("body", "") + "\n" + doc_sections.get("results", "")
        for sp in priority_species:
            if sp in species_names and body_text.strip():
                _run_wiki_agent_for_species(self, sp, body_text, citation)


def _validate_occurrences(
    occurrences: list[dict],
    doc_sections: dict,
) -> list[dict]:
    """
    Run PydanticAI validation on extracted occurrences.
    Falls back to original list if pydantic-ai is not available.
    """
    try:
        from biotrace_scientific_chunker import PydanticAIChunkValidator
        import asyncio

        study_context = (
            doc_sections.get("abstract", "")
            + doc_sections.get("methods", "")
        )[:1500]

        validator = PydanticAIChunkValidator()
        loop = asyncio.new_event_loop()
        validated = loop.run_until_complete(
            validator.validate_batch(occurrences, study_context=study_context)
        )
        loop.close()
        logger.info(
            "[DoclingPatch] Pydantic validation: %d → %d records",
            len(occurrences), len(validated),
        )
        return validated
    except Exception as exc:
        logger.debug("[DoclingPatch] PydanticAI validation skipped: %s", exc)
        return occurrences


def _run_wiki_agent_for_species(self, sp: str, text: str, citation: str):
    """Run OllamaWikiAgent for a single priority species."""
    try:
        from biotrace_wiki_agent import OllamaWikiAgent
        agent = OllamaWikiAgent(wiki=self.wiki)
        result = agent.orchestrate(
            species_name=sp,
            chunk_text=text[:6000],
            citation=citation,
        )
        logger.info(
            "[DoclingPatch] WikiAgent run for %s: %s",
            sp, result.summary[:80] if result.summary else "done",
        )
    except Exception as exc:
        logger.warning("[DoclingPatch] WikiAgent for %s failed: %s", sp, exc)


# ─────────────────────────────────────────────────────────────────────────────
#  Patch installer
# ─────────────────────────────────────────────────────────────────────────────

_ORIG_PROCESS_DOCUMENT = None


def patch_docling_bridge():
    """
    Monkey-patch DoclingWikiBridge with v5.6 improvements.

    Call once at app startup, before any DoclingWikiBridge is instantiated.
    """
    global _ORIG_PROCESS_DOCUMENT
    try:
        import biotrace_docling_wiki_bridge as _mod
        _ORIG_PROCESS_DOCUMENT = _mod.DoclingWikiBridge.process_document
        _mod.DoclingWikiBridge.process_document = process_document_patched
        logger.info("[DoclingPatch] DoclingWikiBridge.process_document patched ✅")
    except ImportError as exc:
        logger.warning("[DoclingPatch] Could not patch DoclingWikiBridge: %s", exc)


def build_cached_bridge(
    wiki_root:      str,
    kg_db_path:     str,
    cache_dir:      str = "biodiversity_data/md_cache",
    run_wiki_agent: bool = False,
):
    """
    Convenience factory: DoclingWikiBridge + MD cache pre-wired.

    Usage
    -----
        bridge, cache = build_cached_bridge(
            wiki_root  = "biodiversity_data/wiki",
            kg_db_path = "biodiversity_data/knowledge_graph.db",
        )

        # In extraction loop:
        md, sections = convert_pdf_cached(pdf_path, converter, cache)
        bridge.process_document(sections, occurrences, citation, species)
    """
    patch_docling_bridge()

    from biotrace_docling_wiki_bridge import DoclingWikiBridge
    from biotrace_md_cache import DoclingMDCache

    bridge = DoclingWikiBridge(
        wiki_root=wiki_root,
        kg_db_path=kg_db_path,
        run_wiki_agent=run_wiki_agent,
    )
    cache = DoclingMDCache(cache_dir)
    return bridge, cache
