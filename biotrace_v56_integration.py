"""
biotrace_v56_integration.py  —  BioTrace v5.6  Master Patch Installer
═══════════════════════════════════════════════════════════════════════════════
Single entry-point to apply all v5.6 patches.  Add ONE import to the top
of biotrace_v53.py (after the existing patch imports) and call install().

Changes applied
───────────────
  [WIKI-1]  References not truncated — full citations rendered
  [WIKI-2]  Occurrence records live from occurrence DB → folium map reflects edits
  [WIKI-3]  All species supported (not just marine) — universal LLM prompts
  [WIKI-4]  Species verification UI: unverified → verified + classification edit
  [WIKI-5]  recordedName shown alongside validName in occurrence table
  [WIKI-6]  Dynamic taxonomic filter (kingdom→phylum→class→order→family cascade)
  [WIKI-7]  Docling wiki-agent runs by default via sections (no separate step)

  [HITL-1]  Taxonomic classification fields in edit form (kingdom/phylum/class/order/family)
  [HITL-2]  All databases (SQLite + KG + Memory Bank + Wiki) updated on taxonomy edit
  [HITL-3]  re-verify checkbox uses BioTraceUnifiedVerifier with taxonomy propagation

  [CACHE-1] Docling-converted MD saved to disk; skip re-processing on re-run
  [CACHE-2] Sections JSON also cached alongside MD

  [SCHUNK]  PydanticAI chunk validator called after each LLM extraction

Usage
─────
    # At the top of biotrace_v53.py, alongside existing patch imports:

    from biotrace_v56_integration import install_v56_patches
    install_v56_patches(
        meta_db_path = META_DB_PATH,     # "biodiversity_data/metadata_v5.db"
        kg_db_path   = KG_DB_PATH,       # "biodiversity_data/knowledge_graph.db"
        wiki_root    = WIKI_ROOT,        # "biodiversity_data/wiki"
        cache_dir    = "biodiversity_data/md_cache",
    )

    # That's it. All existing API calls remain unchanged.

Wiki tab usage (in your Tab 7 / wiki section):
    from biotrace_v56_integration import get_patched_wiki
    wiki = get_patched_wiki()
    wiki.render_streamlit_tab(meta_db_path=META_DB_PATH, call_llm_fn=your_llm_fn)

    # The tab now shows:
    #   Browse Species (with cascading taxon filters)
    #   Verify & Classify (unverified species review)
    #   Locality Checklist

Docling / extraction loop usage:
    from biotrace_v56_integration import get_bridge_and_cache
    bridge, md_cache = get_bridge_and_cache()

    # In PDF loop:
    from biotrace_docling_bridge_v56_patch import convert_pdf_cached
    md, sections = convert_pdf_cached(pdf_path, docling_converter, md_cache)
    bridge.process_document(sections, raw_occurrences, citation, species_names)
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("biotrace.v56_integration")

# Module-level singletons so callers always get the same patched instances
_WIKI_INSTANCE   = None
_BRIDGE_INSTANCE = None
_CACHE_INSTANCE  = None
_FILTER_INSTANCE = None
_INSTALLED       = False


def install_v56_patches(
    meta_db_path: str = "",
    kg_db_path:   str = "",
    wiki_root:    str = "",
    cache_dir:    str = "biodiversity_data/md_cache",
    css_path:     str = "",
) -> None:
    """
    Apply all BioTrace v5.6 patches.  Safe to call multiple times (idempotent).

    Parameters
    ----------
    meta_db_path : path to metadata_v5.db (occurrence SQLite database)
    kg_db_path   : path to knowledge_graph.db
    wiki_root    : path to wiki root directory (contains wiki_unified.db)
    cache_dir    : directory for docling MD cache files
    css_path     : path to biotrace_wiki.css (optional, auto-detected otherwise)
    """
    global _WIKI_INSTANCE, _BRIDGE_INSTANCE, _CACHE_INSTANCE, _FILTER_INSTANCE, _INSTALLED

    if _INSTALLED:
        logger.debug("[v56] Already installed — skipping.")
        return

    # ── Patch 1: HITL taxonomy edit form ─────────────────────────────────────
    try:
        from biotrace_hitl_v56_patch import install_hitl_patches
        install_hitl_patches()
        logger.info("[v56] [HITL-1,2,3] HITL taxonomy patches ✅")
    except Exception as exc:
        logger.warning("[v56] HITL patches skipped: %s", exc)

    # ── Patch 2: Wiki patches (refs, live occs, universal, verify, filter) ────
    try:
        from biotrace_wiki_v56_patch import install_wiki_patches, build_patched_wiki
        install_wiki_patches(meta_db_path=meta_db_path)
        _WIKI_INSTANCE = build_patched_wiki(
            wiki_root    = wiki_root  or "biodiversity_data/wiki",
            meta_db_path = meta_db_path,
            css_path     = css_path,
        )
        logger.info("[v56] [WIKI-1..7] Wiki patches ✅")
    except Exception as exc:
        logger.warning("[v56] Wiki patches skipped: %s", exc)

    # ── Patch 3: Docling bridge + MD cache ────────────────────────────────────
    try:
        from biotrace_md_cache import DoclingMDCache
        _CACHE_INSTANCE = DoclingMDCache(cache_dir)

        from biotrace_docling_bridge_v56_patch import patch_docling_bridge
        patch_docling_bridge()
        logger.info("[v56] [CACHE-1,2] [SCHUNK] Docling bridge + MD cache ✅")
    except Exception as exc:
        logger.warning("[v56] Docling/cache patches skipped: %s", exc)

    # ── Patch 4: Taxonomic filter singleton ──────────────────────────────────
    if meta_db_path:
        try:
            from biotrace_taxon_filter import TaxonFilterWidget
            _FILTER_INSTANCE = TaxonFilterWidget(meta_db_path)
            logger.info("[v56] [WIKI-6] Taxonomic filter widget ✅")
        except Exception as exc:
            logger.warning("[v56] TaxonFilter skipped: %s", exc)

    _INSTALLED = True
    logger.info("[v56] All BioTrace v5.6 patches installed successfully ✅")


# ─────────────────────────────────────────────────────────────────────────────
#  Singleton accessors
# ─────────────────────────────────────────────────────────────────────────────

def get_patched_wiki(
    wiki_root:    str = "biodiversity_data/wiki",
    meta_db_path: str = "",
    css_path:     str = "",
):
    """Return the patched BioTraceWikiUnified singleton."""
    global _WIKI_INSTANCE
    if _WIKI_INSTANCE is None:
        from biotrace_wiki_v56_patch import build_patched_wiki
        _WIKI_INSTANCE = build_patched_wiki(
            wiki_root=wiki_root,
            meta_db_path=meta_db_path,
            css_path=css_path,
        )
    return _WIKI_INSTANCE


def get_bridge_and_cache(
    wiki_root:  str = "biodiversity_data/wiki",
    kg_db_path: str = "biodiversity_data/knowledge_graph.db",
    cache_dir:  str = "biodiversity_data/md_cache",
):
    """Return (DoclingWikiBridge, DoclingMDCache) singletons."""
    global _BRIDGE_INSTANCE, _CACHE_INSTANCE

    if _CACHE_INSTANCE is None:
        from biotrace_md_cache import DoclingMDCache
        _CACHE_INSTANCE = DoclingMDCache(cache_dir)

    if _BRIDGE_INSTANCE is None:
        from biotrace_docling_bridge_v56_patch import build_cached_bridge
        _BRIDGE_INSTANCE, _ = build_cached_bridge(
            wiki_root=wiki_root,
            kg_db_path=kg_db_path,
            cache_dir=cache_dir,
        )

    return _BRIDGE_INSTANCE, _CACHE_INSTANCE


def get_taxon_filter(meta_db_path: str = ""):
    """Return the TaxonFilterWidget singleton."""
    global _FILTER_INSTANCE
    if _FILTER_INSTANCE is None and meta_db_path:
        from biotrace_taxon_filter import TaxonFilterWidget
        _FILTER_INSTANCE = TaxonFilterWidget(meta_db_path)
    return _FILTER_INSTANCE


# ─────────────────────────────────────────────────────────────────────────────
#  Quick-reference: diff of changes to existing files
# ─────────────────────────────────────────────────────────────────────────────

INTEGRATION_NOTES = """
BioTrace v5.6 Integration Notes
================================

## Files added (new)
  biotrace_md_cache.py              — Docling MD/section cache (hash-keyed, SQLite manifest)
  biotrace_taxon_filter.py          — Dynamic cascading taxonomic multiselect widget
  biotrace_hitl_v56_patch.py        — HITL taxonomy fields + all-DB sync
  biotrace_wiki_v56_patch.py        — Wiki fixes (refs, live occs, verify, filter, universal)
  biotrace_docling_bridge_v56_patch.py — Docling bridge: default wiki update + MD cache
  biotrace_v56_integration.py       — Master installer (this file)

## Changes needed in biotrace_v53.py

### 1. Add at top (after existing patch imports):
    from biotrace_v56_integration import install_v56_patches
    install_v56_patches(
        meta_db_path = META_DB_PATH,
        kg_db_path   = KG_DB_PATH,
        wiki_root    = WIKI_ROOT,
    )

### 2. Wiki tab (Tab 7 or wherever wiki is rendered):
    # Replace:
    wiki = BioTraceWikiUnified(WIKI_ROOT)
    wiki.render_streamlit_tab(...)

    # With:
    from biotrace_v56_integration import get_patched_wiki
    wiki = get_patched_wiki(WIKI_ROOT, META_DB_PATH)
    wiki.render_streamlit_tab(call_llm_fn=your_llm_fn)

### 3. Docling extraction loop:
    # Replace raw docling call:
    result = converter.convert(pdf_path)
    md = result.document.export_to_markdown()

    # With cached version:
    from biotrace_v56_integration import get_bridge_and_cache
    from biotrace_docling_bridge_v56_patch import convert_pdf_cached
    bridge, cache = get_bridge_and_cache(WIKI_ROOT, KG_DB_PATH)
    md, sections = convert_pdf_cached(pdf_path, converter, cache)
    bridge.process_document(sections, occurrences, citation, species_names)

### 4. HITL tab:
    # No changes needed — install_v56_patches() monkey-patches _render_edit_form
    # automatically.  The edit form now has a "Taxonomy" tab alongside "Coordinates".

## Wiki tab new features (auto-enabled after patch)
  • Browse Species tab: cascading kingdom→phylum→class→order→family multiselects
    populated dynamically from occurrence DB; species list filtered accordingly
  • Verify & Classify tab: list unverified species, update classification,
    mark as accepted/synonym/invalid — all DBs updated atomically
  • Occurrence table: shows both recordedName (source) and validName (accepted)
  • References section: full citations — no 80-character truncation
  • Occurrence map: reads from live occurrence DB (edits via HITL reflected immediately)

## HITL tab new features (auto-enabled after patch)
  • Edit form has TWO tabs: "Coordinates & Locality" | "Taxonomy & Classification"
  • Taxonomy tab: kingdom, phylum, class, order, family, genus, authority,
    WoRMS ID, IUCN status, taxon rank, taxonomic status, valid name override
  • On commit: SQLite + KG + Memory Bank + Wiki taxobox all updated atomically

## MD Cache
  • First run converts PDF via docling (slow)
  • Subsequent runs load from <cache_dir>/<sha256>.md (instant)
  • Sections JSON cached separately at <cache_dir>/<sha256>.sections.json
  • Cache panel: wiki.get_md_cache().render_streamlit_panel()
"""
