"""
biotrace_v5.py  —  BioTrace v5.2  (Full Edition)
────────────────────────────────────────────────────────────────────────────
Universal Biodiversity Record Extractor
GraphRAG + Memory Bank + LLM-Wiki + TNR Engine + Locality NER + Pydantic Schema

New in v5.2 vs v5.0:
  ✦ TNR Engine (biotrace_ner.py)
      BHL-style three-phase: Regex → GNA Finder → GNA Verifier → Disambiguation
      Universal: marine, terrestrial, freshwater, botanical taxa.
  ✦ Locality NER (biotrace_locality_ner.py)
      spaCy GPE + GeoNames + Nominatim; "Narara" → full admin string.
      Station-ID resolver, locality segregation heuristic.
  ✦ Pydantic Schema (biotrace_schema.py)
      5-attempt JSON repair; field coercion; Gemma4 artifact strip.
  ✦ Verification Table (biotrace_v5_enhancements.py)
      Editable Species | Classification | Locality | Evidence | Flag | Validation.
  ✦ Ollama Model Combobox
      Live model list from /api/tags with custom model text field.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import requests
import urllib.parse



# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("biotrace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS & DIRS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR      = "biodiversity_data"
CSV_DIR       = os.path.join(DATA_DIR, "extractions_v5")
PDF_DIR       = os.path.join(DATA_DIR, "pdfs_v5")
META_DB_PATH  = os.path.join(DATA_DIR, "metadata_v5.db")
KG_DB_PATH    = os.path.join(DATA_DIR, "knowledge_graph.db")
MB_DB_PATH    = os.path.join(DATA_DIR, "memory_bank.db")
WIKI_ROOT     = os.path.join(DATA_DIR, "wiki")
GEONAMES_DB   = os.path.join(DATA_DIR, "geonames_india.db")
PINCODE_TXT   = os.path.join(DATA_DIR, "IN_pin.txt")

for _d in [DATA_DIR, CSV_DIR, PDF_DIR]:
    os.makedirs(_d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  ENHANCEMENT IMPORTS — patch 18042026 + enhancement plan 19042026
#  Order matters: all patch imports must come before optional module imports
#  so the dedup override at the end of the gnv block takes correct precedence.
# ─────────────────────────────────────────────────────────────────────────────

# [ENHANCEMENT: biotrace_geocoding_lifestage_patch] — Patch A: Nominatim India
# bias + Patch B: LLM prompt life-stage guard + post-parse filter
from biotrace_geocoding_lifestage_patch import (
    patch_geocoding_cascade,
    PROMPT_LIFESTAGE_GUARD,
    post_parse_lifestage_filter,
    scan_genus_context,
)

# [ENHANCEMENT: biotrace_locality_guard_patch] — Locality guard: rejects
# morphology-as-locality and habitat-as-locality from LLM output
from biotrace_locality_guard_patch import PROMPT_LOCALITY_GUARD, post_parse_locality_filter

# [ENHANCEMENT: biotrace_traiter_prepass] — Stage 0: rule-based spaCy pre-pass
# annotates taxa, localities, measurements, habitats before LLM sees the text
from biotrace_traiter_prepass import run_prepass, format_annotations_for_prompt

# NOTE: biotrace_dedup_patch.dedup_occurrences is imported AFTER the biotrace_gnv
# try/except block below (~line 225) so it correctly overrides the gnv version.
# DO NOT import it here — a premature import would be silently overwritten by gnv.

if not st.session_state.get("_geo_patched"):
    patch_geocoding_cascade()
    st.session_state["_geo_patched"] = True
    
    
from biotrace_dedup_patch import dedup_occurrences
from biotrace_dedup_patch import suppress_regional_duplicates
from biotrace_progress_logger import BioTraceLogger, render_species_progress_panel

from biotrace_v56_integration import install_v56_patches
install_v56_patches(meta_db_path=META_DB_PATH,
                    kg_db_path=KG_DB_PATH, 
                    wiki_root=WIKI_ROOT)


from biotrace_patch57_update import install_v57_patches
install_v57_patches(
    meta_db_path = META_DB_PATH,    # e.g. "biodiversity_data/metadata_v5.db"
    wiki_root    = WIKI_ROOT,       # e.g. "biodiversity_data/wiki"
    kg_db_path   = KG_DB_PATH,      # e.g. "biodiversity_data/knowledge_graph.db"
    ollama_url   = "http://localhost:11434",
    ollama_model = "gemma4",        # or whichever model you use
)

# ─────────────────────────────────────────────────────────────────────────────
#  OPTIONAL IMPORTS — ALL GRACEFULLY DEGRADED
# ─────────────────────────────────────────────────────────────────────────────

# ── V4 core modules ───────────────────────────────────────────────────────────
_VERIFIER_AVAILABLE = False
detect_scientific_names = None
verify_occurrences_with_fallback = None
try:
    from biotrace_taxonomy import (
        detect_scientific_names,
        verify_occurrences_with_fallback,
    )

    _VERIFIER_AVAILABLE = True
    logger.info("[v5] biotrace_taxonomy loaded (shared detection + verification)")
except ImportError:
    logger.warning("[v5] biotrace_taxonomy.py not found")

_GEOCODER_AVAILABLE = False
GeocodingCascade = None
try:
    from geocoding_cascade import GeocodingCascade
    _GEOCODER_AVAILABLE = True
    logger.info("[v5] geocoding_cascade loaded")
except ImportError:
    logger.warning("[v5] geocoding_cascade.py not found")

# ── V5 new modules ────────────────────────────────────────────────────────────
# _KG_AVAILABLE = False
# BioTraceKnowledgeGraph = None
# try:
#     from biotrace_knowledge_graph import BioTraceKnowledgeGraph
#     _KG_AVAILABLE = True
#     logger.info("[v5] KnowledgeGraph loaded")
# except ImportError:
#     logger.warning("[v5] biotrace_knowledge_graph.py not found")

# _MB_AVAILABLE = False
# BioTraceMemoryBank = None
# try:
#     from biotrace_memory_bank import BioTraceMemoryBank
#     _MB_AVAILABLE = True
#     logger.info("[v5] MemoryBank loaded")
# except ImportError:
#     logger.warning("[v5] biotrace_memory_bank.py not found")

# _WIKI_AVAILABLE = False
# BioTraceWiki = None
# try:
#     from biotrace_wiki import BioTraceWiki
#     _WIKI_AVAILABLE = True
#     logger.info("[v5] Wiki loaded")
# except ImportError:
#     logger.warning("[v5] biotrace_wiki.py not found")
#28042026
_KG_AVAILABLE = False
BioTraceKnowledgeGraph = None
_KG_ERROR = ""
try:
    from biotrace_knowledge_graph import BioTraceKnowledgeGraph
    _KG_AVAILABLE = True
    logger.info("[v5] KnowledgeGraph loaded")
except Exception as _exc:
    _KG_ERROR = str(_exc)
    logger.warning("[v5] KnowledgeGraph import failed: %s", _exc)
 
_MB_AVAILABLE = False
BioTraceMemoryBank = None
_MB_ERROR = ""
try:
    from biotrace_memory_bank import BioTraceMemoryBank
    _MB_AVAILABLE = True
    logger.info("[v5] MemoryBank loaded")
except Exception as _exc:
    _MB_ERROR = str(_exc)
    logger.warning("[v5] MemoryBank import failed: %s", _exc)
 
_WIKI_AVAILABLE = False
BioTraceWikiUnified = None
try:
    # from biotrace_wiki_unified import BioTraceWikiUnified, inject_css_streamlit
    from biotrace_wiki_v56 import BioTraceWikiV56 as BioTraceWikiUnified, inject_css_streamlit
    _WIKI_AVAILABLE = True
    logger.info("[v5.5] BioTraceWikiUnified loaded (versioned, LLM-enhanced)")
except ImportError:
    logger.warning("[v5.5] biotrace_wiki_unified.py not found — Wiki tab disabled")

# Optional: Ollama Wiki Architect Agent
_WIKI_AGENT_AVAILABLE = False
OllamaWikiAgent = None
try:
    from biotrace_wiki_agent import OllamaWikiAgent
    _WIKI_AGENT_AVAILABLE = True
    logger.info("[v5.5] OllamaWikiAgent loaded")
except ImportError:
    logger.info("[v5.5] biotrace_wiki_agent.py not found — agent panel disabled")

# ── PDF parsers ───────────────────────────────────────────────────────────────
_PYMUPDF_AVAILABLE = False
try:
    import pymupdf4llm as _pymupdf4llm
    _PYMUPDF_AVAILABLE = True
except ImportError:
    pass

_MARKITDOWN_AVAILABLE = False
try:
    from markitdown import MarkItDown as _MarkItDown
    _MARKITDOWN_AVAILABLE = True
except ImportError:
    pass

_DOCLING_AVAILABLE = False
try:
    from docling.document_converter import DocumentConverter as _DoclingConverter
    _DOCLING_AVAILABLE = True
except ImportError:
    pass

# ── Plotting ──────────────────────────────────────────────────────────────────
_PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    pass

# ── LLM clients ───────────────────────────────────────────────────────────────
_OLLAMA_AVAILABLE = False
try:
    import ollama as _ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    pass

_OPENAI_AVAILABLE = False
try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    pass

_GEMINI_AVAILABLE = False
try:
    import google.generativeai as _genai
    _GEMINI_AVAILABLE = True
except ImportError:
    pass

_ANTHROPIC_AVAILABLE = False
try:
    import anthropic as _anthropic_sdk
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    pass

# ── v5.1 NEW MODULES ──────────────────────────────────────────────────────────
_CHUNKER_AVAILABLE = False
DocumentChunker = None
try:
    from biotrace_chunker import DocumentChunker, availability_report as chunker_avail
    _CHUNKER_AVAILABLE = True
    logger.info("[v5] DocumentChunker loaded")
except ImportError:
    logger.warning("[v5] biotrace_chunker.py not found")

_GNV_AVAILABLE = False
LocalitySplitter    = None
safe_parse_json     = None

try:
    from biotrace_gnv import (
        LocalitySplitter,
        safe_parse_json,
    )
    _GNV_AVAILABLE = True
    logger.info("[v5] LocalitySplitter + safe_parse_json loaded")
except ImportError:
    logger.warning("[v5] biotrace_gnv.py not found")

# # after the biotrace_gnv try/except block (after line ~216)
# from biotrace_dedup_patch import dedup_occurrences   # always overrides gnv version

# ── OCR pipeline (DocTR + multimodal + Tesseract) ─────────────────────────────
_OCR_AVAILABLE = False
OCRPipeline = None
is_scanned_pdf = None
try:
    from biotrace_ocr import OCRPipeline, is_scanned_pdf
    _OCR_AVAILABLE = True
    logger.info("[v5] OCRPipeline loaded (doctr/tesseract/multimodal)")
except ImportError:
    logger.warning("[v5] biotrace_ocr.py not found")

# ── v5.2 NEW MODULES ──────────────────────────────────────────────────────────
_NER_AVAILABLE = False
TaxonNER = None
try:
    from biotrace_ner import TaxonNER, extract_taxa, COPIOUSFilter
    _NER_AVAILABLE = True
    logger.info("[v5.2] TaxonNER loaded")
except ImportError:
    logger.warning("[v5.2] biotrace_ner.py not found")

_LOC_NER_AVAILABLE = False
LocalityNER = None
try:
    from biotrace_locality_ner import LocalityNER, segregate_locality_string
    _LOC_NER_AVAILABLE = True
    logger.info("[v5.2] LocalityNER loaded")
except ImportError:
    logger.warning("[v5.2] biotrace_locality_ner.py not found")

_SCHEMA52_AVAILABLE = False
_parse_llm_response = None
_records_to_dicts   = None
try:
    from biotrace_schema import (
        parse_llm_response  as _parse_llm_response,
        records_to_dicts    as _records_to_dicts,
        SCHEMA_VALIDATION_RULES,
        SCHEMA_JSON_EXAMPLE,
    )
    _SCHEMA52_AVAILABLE = True
    logger.info("[v5.2] Pydantic schema loaded")
except ImportError:
    logger.warning("[v5.2] biotrace_schema.py not found")

_ENH_AVAILABLE = False
_render_verification_table = None
_render_tnr_tab             = None
_render_locality_tab        = None
_render_schema_diagnostics  = None
_render_ollama_model_selector = None
try:
    from biotrace_v5_enhancements import (
        render_verification_table       as _render_verification_table,
        render_tnr_tab                  as _render_tnr_tab,
        render_locality_tab             as _render_locality_tab,
        render_schema_diagnostics       as _render_schema_diagnostics,
        render_ollama_model_selector    as _render_ollama_model_selector,
        occurrences_to_verification_df,
    )
    _ENH_AVAILABLE = True
    logger.info("[v5.2] UI enhancements loaded")
except ImportError:
    logger.warning("[v5.2] biotrace_v5_enhancements.py not found")

# ── v5.3 NEW MODULES ──────────────────────────────────────────────────────────
_PDF_META_AVAILABLE = False
PaperMetaFetcher = None
try:
    from biotrace_pdf_meta import PaperMetaFetcher, availability_report as pdf_meta_avail
    _PDF_META_AVAILABLE = True
    logger.info("[v5.3] PaperMetaFetcher loaded (S2 + Crossref)")
except ImportError:
    logger.warning("[v5.3] biotrace_pdf_meta.py not found")

_HIER_CHUNKER_AVAILABLE = False
HierarchicalChunker = None
try:
    from biotrace_hierarchical_chunker import HierarchicalChunker, ExtractionBatch
    _HIER_CHUNKER_AVAILABLE = True
    logger.info("[v5.3] HierarchicalChunker loaded (3-level hierarchy)")
except ImportError:
    logger.warning("[v5.3] biotrace_hierarchical_chunker.py not found")


# ── v5.4 BiodiViz NER Integration ──────────────────────────────────────────────
_BIODIVIZ_AVAILABLE = False
_SCICHUNKER_AVAILABLE = False
try:
    from biotrace_scientific_chunker import ScientificPaperChunker
    _SCICHUNKER_AVAILABLE = True
    logger.info("[v5.4] ScientificPaperChunker loaded")
except ImportError:
    logger.warning("[v5.4] biotrace_scientific_chunker.py not found")

# try:
#     from biotrace_scientific_chunker import ScientificPaperChunker
#     _SCICHUNKER_AVAILABLE = True
#     logger.info("[v5.4] ScientificPaperChunker loaded")
# except ImportError:
#     logger.warning("[v5.4] biotrace_scientific_chunker.py not found")


_AGENTIC_AVAILABLE = False
try:
    from biotrace_agentic_chunker import build_agentic_pipeline, records_to_dicts
    _AGENTIC_AVAILABLE = True
    logger.info("[v5.7] AgenticExtractionPipeline loaded")
except ImportError:
    logger.warning("[v5.7] biotrace_agentic_chunker.py not found")

BiodiVizPipeline = None

try:
    from biotrace_hf_ner import BiodiVizPipeline
    _BIODIVIZ_AVAILABLE = True
    logger.info("[v5.4] BiodiViz HF Pipeline loaded")
except ImportError:
    logger.warning("[v5.4] biotrace_hf_ner.py not found or missing HF deps")

@st.cache_resource
def get_biodiviz_pipeline() -> "BiodiVizPipeline | None":
    if not _BIODIVIZ_AVAILABLE:
        return None
    try:
        return BiodiVizPipeline(ner_model_path="./ner_model", re_model_path="./re_model")
    except Exception as exc:
        logger.error("[v5.4] BiodiViz Init Error: %s", exc)
        return None

#290426

# ── v5.5 pydantic-ai agentic extraction ───────────────────────────────────────
_PAI_AVAILABLE = False
try:
    from pydantic_ai import Agent as _PAIAgent
    from pydantic_ai.models.ollama import OllamaModel as _PAIOllamaModel
    _PAI_AVAILABLE = True
    logger.info("[v5.5] pydantic-ai loaded — agentic extraction available")
except Exception:
    logger.info("[v5.5] pydantic-ai not installed (pip install pydantic-ai) — optional")
    
    
# ─────────────────────────────────────────────────────────────────────────────
#  PROMPTS  (v5.2 — high-recall, GNA-aware)
# ─────────────────────────────────────────────────────────────────────────────

_THINKER_PROMPT = """
You are an expert biologist working in the field of biodiversity informatics, tasked with reading a section of a scientific paper.

YOUR TASK — perform TWO steps in order:

STEP 1 — SPECIES INVENTORY (THINK ALOUD)
Carefully scan every sentence, table cell, figure caption, and parenthetical citation.
List EVERY scientific biological name you find, including:
  • Fully written binomials/trinomials
  • Genus-only references
  • Abbreviated names (expand to the full genus if context makes it clear)
  • Open nomenclature (sp., cf., aff., spp., var., ssp., n. sp.)
  • Names in parentheses, footnotes, supplementary lists, and tables
  • Names mentioned as previously recorded, synonyms, or cited from literature

For each name found, note the following in plain text:
  (a) The exact sentence(s) it appears in (include adjacent preceding/succeeding sentences if needed to establish context).
  (b) The observation status: Is this a "Primary" record (data/observations generated exclusively by this paper's authors in the present study. Generally, primary observation localities are mentioned in the Title, Abstract, or Methods section) or a "Secondary" record (authors are citing prior work from a different author/year)?
  (c) Missing context check: Look at sentences adjacent to the scientific name. If a locality is mentioned using a life stage (e.g., "polyps from the reef", "medusae from the coast") or pronoun referring back to the species, make sure you note these primary localities!

STEP 2 — COMPLETENESS CHECK
Confirm your inventory is exhaustive. Explicitly ask and answer:
  "Did I miss any names in table rows, figure legends, footnotes, or dense discussion paragraphs?"

Output STEP 1 and STEP 2 as plain text thinking. Do NOT produce JSON yet.
The structured JSON extraction will happen in the next step using your inventory.

TEXT TO ANALYSE:
"""

# ── Main extraction prompt ─────────────────────────────────────────────────────

_SCHEMA_PROMPT = """\
You are a biodiversity data extraction expert.

Your job is to produce COMPLETE structured JSON for EVERY species-occurrence event found.

# NOTE: PROMPT_LIFESTAGE_GUARD is spliced in at runtime (see extract_occurrences()).
# NOTE: PROMPT_LOCALITY_GUARD replaces the verbatimLocality field definition at runtime.
# Do NOT add format placeholders here — both are injected via str.replace().

CONTEXT: The species inventory from your prior analysis step has already identified the biological names present.
Your job is to produce the COMPLETE structured JSON for EVERY species-occurrence event.

[CURRENT_DOCUMENT_METADATA]: {document_citation_string}

DETECTION RULES:
  • GENUS-ONLY NAMES: Extract genus-only occurrences (e.g. "Acropora sp.", "Elysia",
    "Glossodoris spp.") as valid records. Use the verbatim form as "Recorded Name".
    Do NOT discard names simply because a species epithet is absent.
  • ABBREVIATIONS: Expand abbreviated genus fully based on prior text (e.g. "A. cornutus"
    → "Acanthurus cornutus"). Include authority strings if present.
  • OPEN NOMENCLATURE: Treat cf., aff., sp. n., n. sp., sp., spp., var., subsp. as
    valid qualifiers. Include them verbatim in "Recorded Name".
  • PRONOUNS & LIFE STAGES: If a sentence discusses a locality but refers to the organism
    by a life stage (medusa, polyp, larva, spat, juvenile) or pronoun, MUST extract
    those localities and associate with the scientific name in preceding sentences.
  • ONE LOCALITY PER RECORD (STRICT): Multiple localities in one sentence → separate
    JSON objects for EACH. NEVER merge with semicolons or commas in verbatimLocality.
  • ONE SPECIES PER RECORD (STRICT): Multiple species in one sentence → separate JSON
    records for EACH, duplicating locality/date data as needed.
For EACH species x locality x event, return a JSON object with EXACTLY these keys:
  "Recorded Name"      — Scientific name exactly as written (never correct spelling). Expand abbreviations if genus is known. Include authority if present.
  "Valid Name"         — Leave as empty string "" (taxonomy enrichment fills this later).
  "Higher Taxonomy"    — Leave as empty string "" (taxonomy enrichment fills this later).
  "Source Citation"    — PRIMARY/UNCERTAIN records: output the exact string from [CURRENT_DOCUMENT_METADATA] above. Do not guess author/year from text. Do NOT use inline citations from other authors.
                         SECONDARY records: use the historical cited work. Extract FULL reference string from Bibliography if present, otherwise use exact inline citation (e.g. "Browne, 1916").
  "Habitat"            — Specific ecosystem type (Coral reef, Mangrove, Rocky intertidal, Seagrass bed, Pelagic, Estuarine). Use "Not Reported" if unspecified.
  "Sampling Event"     — JSON string: {"date": "YYYY-MM-DD", "depth_m": "N", "method": "..."}. Use "YYYY/YYYY" for year ranges. Use "Not Reported" for missing fields.
  "Raw Text Evidence"  — EXACT verbatim sentence from the paper proving this occurrence, PLUS exactly one preceding and one succeeding sentence for context. Copy word-for-word.
  "verbatimLocality"   — Place name exactly as written. ONE location per record.
                         CRITICAL: Cannot be blank or "Unknown". Resolve station IDs (St.1, Site A) from Methods tables. If no micro-locality, use the broadest study area from Title/Introduction.
  "occurrenceType"     — EXACTLY one of:
                           "Primary"   — Authors themselves collected/observed it in this study.
                           "Secondary" — Cited from a prior publication. Treat historical records in comparative statements as separate Secondary records.
                           "Uncertain" — Ambiguous; cannot determine if directly observed or cited.

MANDATORY COMPLETENESS CHECK before returning JSON:
  □ Every row of every TABLE has been processed into a record.
  □ Every figure caption mentioning a species has a corresponding record.
  □ Every life-stage locality in Methods/Abstract has been paired with its target species.

FORMATTING RULES:
  • Return ONLY a valid JSON array of objects: [ {...}, {...} ]
  • No markdown fences, no prose, no comments.
  • If ZERO species found in this chunk, return: []
  • Never invent or hallucinate data not explicitly in the text.
"""



# ── GNA name-finder pre-pass prompt ──────────────────────────────────────────
_GNA_FALLBACK_PROMPT = """\
Extract every scientific biological name (including Genus + species, trinomial subspecies, and open nomenclature like sp., cf., aff., and spp.) from the text below.

RULES:
1. Output EXACTLY ONE scientific name per line.
2. Output nothing else (no bullet points, no introductory text).
3. If a name is abbreviated and the full genus was established earlier in the text, write the FULL expanded form.
4. Ignore common vernacular names unless part of a scientific name string.
5. Include taxonomic authorities (e.g., "Linnaeus, 1758") if attached to the name in the text.

TEXT:
"""

# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE  (v5 schema — backward-compatible with v4)
# ─────────────────────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(META_DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS occurrences_v4 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT, recordedName TEXT, validName TEXT,
            higherTaxonomy TEXT, sourceCitation TEXT, habitat TEXT,
            samplingEvent TEXT, rawTextEvidence TEXT,
            decimalLatitude REAL, decimalLongitude REAL,
            verbatimLocality TEXT, occurrenceType TEXT,
            geocodingSource TEXT, phylum TEXT, class_ TEXT,
            order_ TEXT, family_ TEXT, wormsID TEXT, itisID TEXT,
            taxonRank TEXT, nameAccordingTo TEXT, taxonomicStatus TEXT,
            matchScore REAL,
            validationStatus TEXT DEFAULT 'pending',
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            session_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_v4_hash   ON occurrences_v4(file_hash);
        CREATE INDEX IF NOT EXISTS idx_v4_sp     ON occurrences_v4(validName);
        CREATE INDEX IF NOT EXISTS idx_v4_loc    ON occurrences_v4(verbatimLocality);
        CREATE INDEX IF NOT EXISTS idx_v4_family ON occurrences_v4(family_);
    """)
    con.commit()
    con.close()


# def _to_float(val):
#     if val is None:
#         return None
#     try:
#         f = float(str(val).strip())
#         return None if str(val).strip() in ("0", "") else f
#     except (ValueError, TypeError):
#         return None


def _to_float(val) -> float | None:
    """FIX: zero is a valid coordinate (equatorial specimens were silently dropped)."""
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None



def insert_occurrences(occurrences, file_hash, source_title, session_id):

    """Insert extracted occurrences into SQLite."""
    if not occurrences:
        return 0
        
    con = sqlite3.connect(META_DB_PATH)
    inserted = 0
    for occ in occurrences:
        if not isinstance(occ, dict):
            continue
            
        # BUG FIX: Now checking validName AND recordedName AND Recorded Name
        sp = (occ.get("validName") or occ.get("recordedName") or occ.get("Recorded Name", "")).strip()
        if not sp:
            continue
            
        sampling = occ.get("Sampling Event") or occ.get("samplingEvent") or {}
        if isinstance(sampling, dict):
            sampling_str = json.dumps(sampling)
        else:
            sampling_str = str(sampling)

        tax = occ.get("Higher Taxonomy") or {}
        if isinstance(tax, str):
            try:
                tax = json.loads(tax)
            except Exception:
                tax = {}

        # PATCHED-R2: R2-taxonomy-fallback — unpack Higher Taxonomy JSON when
        # flat fields are empty (happens when GNA verifier skips or fails)
        def _tax_field(key_variants: list, fallback: str = "") -> str:
            for k in key_variants:
                v = occ.get(k)
                if v and str(v).strip():
                    return str(v).strip()[:100]
            # Fallback: pull from Higher Taxonomy dict
            if isinstance(tax, dict):
                for k in key_variants:
                    v = tax.get(k) or tax.get(k.rstrip("_"))
                    if v and str(v).strip():
                        return str(v).strip()[:100]
            return fallback

        _phylum  = _tax_field(["phylum",  "Phylum"])
        _class   = _tax_field(["class_",  "class", "Class"])
        _order   = _tax_field(["order_",  "order", "Order"])
        _family  = _tax_field(["family_", "family","Family"])

        # Citation: prefer per-record "Source Citation", then session citation,
        # then source_title (which is now citation_str, not doc_title)
        _citation = (
            str(occ.get("Source Citation") or occ.get("sourceCitation") or "").strip()
            or source_title   # source_title is now citation_str (see line ~2384 fix)
        )

        con.execute("""
            INSERT INTO occurrences_v4 (
                file_hash, recordedName, validName, higherTaxonomy,
                sourceCitation, habitat, samplingEvent, rawTextEvidence,
                decimalLatitude, decimalLongitude, verbatimLocality,
                occurrenceType, geocodingSource, phylum, class_, order_,
                family_, wormsID, itisID, taxonRank, nameAccordingTo,
                taxonomicStatus, matchScore, session_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            file_hash,
            str(occ.get("recordedName") or occ.get("Recorded Name",""))[:300],
            sp[:300],
            json.dumps(tax),
            _citation[:500],
            str(occ.get("Habitat") or occ.get("habitat",""))[:300],
            sampling_str,
            str(occ.get("Raw Text Evidence") or occ.get("rawTextEvidence",""))[:1000],
            _to_float(occ.get("decimalLatitude")),
            _to_float(occ.get("decimalLongitude")),
            str(occ.get("verbatimLocality",""))[:300],
            str(occ.get("occurrenceType",""))[:50],
            str(occ.get("geocodingSource",""))[:100],
            _phylum, _class, _order, _family,
            str(occ.get("wormsID",""))[:20],
            str(occ.get("itisID",""))[:20],
            str(occ.get("taxonRank",""))[:50],
            str(occ.get("nameAccordingTo",""))[:100],
            str(occ.get("taxonomicStatus",""))[:50],
            float(occ.get("matchScore", 0) or 0),
            session_id,
        ))
        inserted += 1
        
    con.commit()
    con.close()
    return inserted

# ─────────────────────────────────────────────────────────────────────────────
#  PDF → MARKDOWN
# ─────────────────────────────────────────────────────────────────────────────
def pdf_to_markdown(pdf_path: str, parser: str = "pymupdf4llm") -> str:
    if "pymupdf" in parser and _PYMUPDF_AVAILABLE:
        try:
            return _pymupdf4llm.to_markdown(pdf_path)
        except Exception as exc:
            logger.warning("[pdf] pymupdf4llm: %s", exc)

    if "markitdown" in parser and _MARKITDOWN_AVAILABLE:
        try:
            md = _MarkItDown()
            return md.convert(pdf_path).text_content
        except Exception as exc:
            logger.warning("[pdf] markitdown: %s", exc)

    if "docling" in parser and _DOCLING_AVAILABLE:
        try:
            conv = _DoclingConverter()
            return conv.convert(pdf_path).document.export_to_markdown()
        except Exception as exc:
            logger.warning("[pdf] docling: %s", exc)

    # Raw text fallback
    try:
        import fitz
        doc  = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception:
        pass

    return "[Could not extract text — install pymupdf4llm]"


# ─────────────────────────────────────────────────────────────────────────────
#  LLM CALL  (multi-provider)
# ─────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass as _dc

@_dc
class LLMConfig:
    provider: str
    model:    str
    api_key:  str = ""
    base_url: str = "http://localhost:11434"

def _call_ollama(prompt, cfg):
    resp = _ollama.chat(model=cfg.model,
                        messages=[{"role":"user","content":prompt}],
                        options={"num_predict":2048})
    return resp.message.content if hasattr(resp,"message") else resp["message"]["content"]

def _call_anthropic_via_ollama(prompt, cfg):
    # Use user-provided api_key from config; default to "ollama" for local instances if blank
    key = cfg.api_key if cfg.api_key else "ollama"
    client = _anthropic_sdk.Anthropic(base_url=cfg.base_url, api_key=key)
    return client.messages.create(model=cfg.model, max_tokens=2048,
           messages=[{"role":"user","content":prompt}]).content[0].text

def _call_openai(prompt, cfg):
    return _OpenAI(api_key=cfg.api_key).chat.completions.create(
           model=cfg.model, messages=[{"role":"user","content":prompt}],
           max_tokens=2048).choices[0].message.content

def _call_gemini(prompt, cfg):
    _genai.configure(api_key=cfg.api_key)
    return _genai.GenerativeModel(cfg.model).generate_content(prompt).text

_LLM_REGISTRY = {
    "Ollama (Local)":       _call_ollama,
    "Anthropic via Ollama": _call_anthropic_via_ollama,
    "OpenAI":               _call_openai,
    "Gemini":               _call_gemini,
}

def call_llm(prompt: str, provider: str, model_sel: str,
             api_key: str = "", ollama_base_url: str = "http://localhost:11434") -> str:
    """Registry dispatch — adding a new provider = one dict entry."""
    cfg = LLMConfig(provider=provider, model=model_sel,
                    api_key=api_key, base_url=ollama_base_url)
    fn  = _LLM_REGISTRY.get(provider, lambda p, c: '{"error":"No provider"}')
    try:
        return fn(prompt, cfg)
    except Exception as exc:
        logger.warning("[LLM] %s error: %s", provider, exc)
        return "[]"



# ─────────────────────────────────────────────────────────────────────────────
#  OLLAMA MODEL DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """
    Fetch the list of locally available Ollama models from the REST API.
    Returns a sorted list of model name strings, or a fallback list if
    Ollama is unreachable.
    """
    try:
        import requests as _req
        r = _req.get(f"{base_url}/api/tags", timeout=3)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if models:
            return sorted(models)
    except Exception as exc:
        logger.debug("[v5] Ollama model list: %s", exc)
    return [
        "gemma4", "gemma3", "gemma3:12b",
        "llama3.2", "llama3.3", "qwen2.5:7b",
        "mistral", "phi4", "deepseek-r1:8b",
        "llava", "llava:13b", "moondream",
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACT THINKER  — three-layer species name discovery (v5.4)
# ─────────────────────────────────────────────────────────────────────────────

def extract_thinker(
    chunk_text: str,
    provider: str,
    model_sel: str,
    api_key: str,
    ollama_base_url: str,
    log_cb,
) -> list[str]:
    """
    Multi-layer species name pre-inventory.  Returns a deduplicated list of
    name strings (binomials, genus-only, open nomenclature) found in the chunk.
    """
    found: list[str] = []

    if _VERIFIER_AVAILABLE and detect_scientific_names is not None:
        try:
            detected = detect_scientific_names(
                chunk_text,
                log_cb=log_cb,
                ner_model_getter=get_biodiviz_pipeline,
            )
            if detected:
                found.extend(
                    hit.get("scientificName") or hit.get("verbatimName", "")
                    for hit in detected
                    if (hit.get("scientificName") or hit.get("verbatimName", "")).strip()
                )
        except Exception as exc:
            logger.debug("[thinker/taxonomy] %s", exc)

    # ── Layer 3: LLM plain-text extraction (no regex post-processing) ─────────
    if not found:
        try:
            llm_prompt = (
                "List every scientific biological name in the text below.\n"
                "Rules:\n"
                "1. Output ONE name per line, nothing else.\n"
                "2. Include binomials, genus-only, open nomenclature "
                "(sp., cf., aff., spp., var., n. sp.).\n"
                "3. Expand abbreviated genera (e.g. 'A. cornutus' → "
                "'Acanthurus cornutus') when the full genus appears earlier.\n"
                "4. Do NOT include common names or author citations.\n\n"
                "TEXT:\n" + chunk_text[:4000]
            )
            raw = call_llm(llm_prompt, provider, model_sel, api_key, ollama_base_url)
            # Strip <think> blocks
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
            llm_names = [
                ln.strip().lstrip("•-–* ")
                for ln in raw.splitlines()
                if ln.strip() and re.match(r"^[A-Z]", ln.strip())
                and len(ln.strip()) >= 4
                and not ln.strip().startswith("Text")
                and not ln.strip().startswith("Rule")
                and not ln.strip().startswith("Note")
            ]
            if llm_names:
                found.extend(llm_names)
                log_cb(f"    [Thinker/LLM] {len(llm_names)} names from LLM")
        except Exception as exc:
            logger.debug("[thinker/LLM] %s", exc)

    # ── Layer 4: Relaxed regex fallback ──────────────────────────────────────
    if not found:
        _binomial_strict = re.compile(
            r"\b([A-Z][a-z]{2,})\s+([a-z]{3,}(?:\s+(?:var\.|subsp\.|f\.)\s+[a-z]+)?)\b"
        )
        _thinker_blocklist = frozenset({
            "Table","Figure","Methods","Results","Discussion","Abstract",
            "Introduction","Conclusion","Appendix","Section","Supplementary",
            "Material","Study","Sample","Station","Site","Area","Data",
            "Note","Rule","Text","Plate","Part","Volume","Number",
        })
        regex_names = [
            m.group(0) for m in _binomial_strict.finditer(chunk_text)
            if m.group(1) not in _thinker_blocklist
        ]
        if regex_names:
            found.extend(regex_names)
            log_cb(f"    [Thinker/regex] {len(regex_names)} names (relaxed fallback)")

    # Deduplicate preserving order
    # PATCHED: P2-candidate-filter-thinker — drop NER placeholder IDs and non-taxon strings
    _CAND_ID_RE = re.compile(r"^__candidate_\d+_\d+$")
    _TAXON_START_RE = re.compile(r"^[A-Z][a-z]{2,}")

    seen:   set[str]  = set()
    unique: list[str] = []
    for n in found:
        n_strip = n.strip()
        if (n_strip
                and n_strip not in seen
                and not _CAND_ID_RE.match(n_strip)
                and _TAXON_START_RE.match(n_strip)):
            seen.add(n_strip)
            unique.append(n_strip)

    if unique:
        log_cb(f"    🔍 thinker: {len(unique)} unique candidate names")
    return unique


def build_schema_prompt(cite_str: str) -> str:
    """Assembles the full prompt from immutable parts — no global mutation."""
    base = _SCHEMA_PROMPT
    # Inject guards once, cleanly, without modifying the global
    if PROMPT_LIFESTAGE_GUARD not in base:
        base = base.replace(
            'For EACH species x locality x event',
            PROMPT_LIFESTAGE_GUARD + 'For EACH species x locality x event'
        )
    if PROMPT_LOCALITY_GUARD not in base:
        base = base.replace(
            '  \"verbatimLocality\"   — Place name exactly as written.',
            PROMPT_LOCALITY_GUARD,
        )
    return base.replace("{document_citation_string}", cite_str)


from dataclasses import dataclass as _dc, field as _f

@_dc
class _ChunkResult:
    records:  list   = _f(default_factory=list)
    status:   str    = "ok"     # "ok" | "error" | "skip"
    error:    str    = ""


# ── Module-level chunk processor (replaces _process_batch_text closure) ───────

def process_chunk(
    text:               str,
    section_label:      str,
    schema_prompt:      str,
    cite_str:           str,
    provider:           str,
    model_sel:          str,
    api_key:            str,
    ollama_base_url:    str,
    use_thinker:        bool,
    candidate_locs:     list | None = None,
    log_cb              = None,
) -> _ChunkResult:
    """
    Stateless module-level function — no nonlocal, no closure captures.
    Returns _ChunkResult(records, status, error).
    The caller owns error_ct / skip_ct and increments them from .status.
    """
    if log_cb is None:
        log_cb = lambda msg, lvl="ok": None

    if not text.strip():
        return _ChunkResult(status="skip")

    if candidate_locs is None:
        candidate_locs = []

    # ── Pre-pass annotation ───────────────────────────────────────────────────
    pre              = run_prepass(text)
    annotation_block = format_annotations_for_prompt(pre)
    augmented_text   = annotation_block + "\n\n" + text

    # ── Species hint (BiodiViz NER or Thinker — mutually exclusive) ──────────
    species_hint_str   = ""
    biodiviz_relations = []
    use_hf_ner         = st.session_state.get("use_biodiviz", False)

    if use_hf_ner and _BIODIVIZ_AVAILABLE:
        hf_pipeline = get_biodiviz_pipeline()
        if hf_pipeline:
            hf_results        = hf_pipeline.extract(augmented_text)
            organisms         = hf_results.get("organisms", [])
            biodiviz_relations = hf_results.get("relations", [])
            if organisms:
                species_hint_str = "\n".join(f"  • {n}" for n in organisms[:30])
                log_cb(f"    [BiodiViz NER] {len(organisms)} organisms")
            if biodiviz_relations:
                log_cb(f"    [BiodiViz RE] {len(biodiviz_relations)} locality relations")

    elif use_thinker:
        hints = extract_thinker(
            augmented_text, provider, model_sel, api_key, ollama_base_url, log_cb
        )
        if hints:
            species_hint_str = "\n".join(f"  • {n}" for n in hints[:25])
            log_cb(f"    [Thinker] {len(hints)} names pre-identified")

    # ── Assemble prompt ───────────────────────────────────────────────────────
    species_hint = (
        "\n\n[CONFIRMED SPECIES INVENTORY — extract structured JSON for these]:\n"
        + species_hint_str
    ) if species_hint_str else ""

    re_hint = (
        "\n\n[CONFIRMED SPECIES-LOCALITY LINKS from NER — use these verbatimLocality values]:\n"
        + "\n".join(f"  • {r}" for r in biodiviz_relations[:15])
    ) if biodiviz_relations else ""

    locality_hint = (
        "\n\n[PRE-LINKED LOCALITIES from Methods section]:\n"
        + "\n".join(f"  • {l}" for l in candidate_locs[:5])
    ) if candidate_locs else ""

    prompt = (
        f"{schema_prompt}"
        f"{locality_hint}"
        f"{species_hint}"
        f"{re_hint}"
        f"\n\nSECTION: \"{section_label}\"\n\nTEXT:\n{augmented_text}"
        f"\n\nCRITICAL: Output EXACTLY ONE valid JSON array. "
        f"No markdown code blocks, no prose, no explanations."
    )

    # ── LLM call ─────────────────────────────────────────────────────────────
    try:
        raw = call_llm(prompt, provider, model_sel, api_key, ollama_base_url)

        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"<\|thinking\|>.*?<\|/thinking\|>", "", raw, flags=re.DOTALL).strip()

        fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
        if fence:
            raw = fence.group(1).strip()
        else:
            arr = re.search(r"(\[\s*\{.*\}\s*\])", raw, re.DOTALL)
            if arr:
                raw = arr.group(1).strip()
            elif re.match(r"^\s*\[\s*\]\s*$", raw):
                raw = "[]"
            else:
                s, e = raw.find("["), raw.rfind("]")
                if s != -1 and e > s:
                    raw = raw[s: e + 1]

        # ── Parse ─────────────────────────────────────────────────────────────
        data = None

        if _SCHEMA52_AVAILABLE and _parse_llm_response:
            recs, errs = _parse_llm_response(raw, source_citation=cite_str)
            for e in errs:
                log_cb(f"  [{section_label}] schema: {e}", "warn")
            data = [r.to_dict() for r in recs] if recs is not None else None

            if data:
                genus_ctx = scan_genus_context(augmented_text)
                data, discarded = post_parse_lifestage_filter(data, genus_ctx)
                if discarded:
                    log_cb(f"  [LS-filter] {len(discarded)} life-stage records discarded")
                data, loc_quar = post_parse_locality_filter(data)
                if loc_quar:
                    log_cb(f"  [Loc-filter] {len(loc_quar)} morphology/habitat localities quarantined")
                # Feed filter results into tracker if log_cb is a BioTraceLogger
                if hasattr(log_cb, 'log_filter_result'):
                    log_cb.log_filter_result(data, discarded, loc_quar)

            if data is None and errs:
                return _ChunkResult(status="error", error="schema parse failed")

        else:
            if _GNV_AVAILABLE and safe_parse_json:
                data = safe_parse_json(raw)
            else:
                cleaned = raw.replace("```json", "").replace("```", "").strip()
                try:
                    parsed = json.loads(cleaned)
                    data   = parsed if isinstance(parsed, list) else None
                except json.JSONDecodeError:
                    data = None

        if data is None:
            msg = f"JSON parse failed (preview: {raw[:60].replace(chr(10), ' ')}…)"
            log_cb(f"  [{section_label}] {msg}", "warn")
            return _ChunkResult(status="error", error=msg)

        # ── Normalise ─────────────────────────────────────────────────────────
        for rec in data:
            if isinstance(rec, dict):
                if not rec.get("recordedName") and rec.get("Recorded Name"):
                    rec["recordedName"] = rec["Recorded Name"]
                if not rec.get("verbatimLocality") and candidate_locs:
                    rec["verbatimLocality"] = candidate_locs[0]

        log_cb(f"  [{section_label}] {len(data)} records")
        return _ChunkResult(records=data)

    except Exception as exc:
        log_cb(f"  [{section_label}] error: {exc}", "warn")
        return _ChunkResult(status="error", error=str(exc))
def extract_occurrences(
    markdown_text:    str,
    doc_title:        str,
    provider:         str,
    model_sel:        str,
    api_key:          str,
    ollama_base_url:  str,
    log_cb,
    chunk_strategy:   str  = "section",
    chunk_chars:      int  = 6000,
    overlap_chars:    int  = 400,
    batch_mode:       bool = False,
    citation_string:  str  = "",
    use_hierarchical: bool = True,
    use_scientific:   bool = True,    # NEW — ScientificPaperChunker
    use_agentic:      bool = False,   # NEW — AgenticExtractionPipeline
    use_thinker:      bool = True,
    use_auto_loc_ner: bool = True,
    geonames_db:      str  = "",
) -> list[dict]:
    """
    Orchestrator — thin and readable.
    All per-chunk work is delegated to process_chunk() which owns
    pre-pass, NER, LLM call, JSON cleaning, Pydantic parse, and filters.
    Stateless — no closures, no global mutation.
    """
    if not markdown_text.strip():
        return []
 
    cite_str = citation_string or doc_title or "Unknown Source"
 
    # ── Prompt assembly — no global mutation ─────────────────────────────────
    # Build a local copy of the prompt; the module-level _SCHEMA_PROMPT is
    # never modified — local copy only.
    _prompt_base = _SCHEMA_PROMPT
    if PROMPT_LIFESTAGE_GUARD not in _prompt_base:
        _prompt_base = _prompt_base.replace(
            'For EACH species x locality x event',
            PROMPT_LIFESTAGE_GUARD + 'For EACH species x locality x event',
        )
    if PROMPT_LOCALITY_GUARD not in _prompt_base:
        _prompt_base = _prompt_base.replace(
            '  "verbatimLocality"   — Place name exactly as written.',
            PROMPT_LOCALITY_GUARD,
        )
    schema_prompt = _prompt_base.replace("{document_citation_string}", cite_str)
 

    # ── Agentic Chunking Pipeline (v5.7) ─────────────────────────────────────
    if use_agentic and _AGENTIC_AVAILABLE:
        try:
            log_cb("[Agentic] Starting AgenticExtractionPipeline...")
            pipeline = build_agentic_pipeline(
                meta_db_path    = META_DB_PATH,
                wiki_root       = WIKI_ROOT if 'WIKI_ROOT' in globals() else "./wiki",
                provider        = provider,
                model_sel       = model_sel,
                ollama_url      = ollama_base_url,
                use_wiki_writer = True, # Enables wiki writing side by side
                external_llm_fn = lambda p: call_llm(p, provider, model_sel, api_key, ollama_base_url),
                log_cb          = log_cb,
            )
            records = pipeline.run(
                markdown_text    = markdown_text,
                source_citation  = cite_str,
                log_cb           = log_cb,
            )
            return records_to_dicts(records)
        except Exception as exc:
            log_cb(f"[Agentic] Pipeline failed: {exc} — falling back to standard extraction", "warn")

    # ── Chunking — 4-priority waterfall ──────────────────────────────────────
    #
    #  Each priority populates EITHER `batches` (objects with .context)
    #  OR `flat_chunks` (objects with .text).  Exactly one list ends up
    #  non-empty; the loop below unifies them with getattr() defaults.
    #
    #  Priority 1  ScientificPaperChunker  — injects Methods localities into
    #              Results chunks (fixes the cross-section context-loss bug).
    #  Priority 2  HierarchicalChunker     — 3-level section/para/sentence.
    #  Priority 3  DocumentChunker         — strategy-based flat chunks.
    #  Priority 4  Naive fixed-size slices — last resort, always works.
 
    batches     = []
    flat_chunks = []
    use_flat    = True
 
    # Priority 1: ScientificPaperChunker
    if use_scientific and _SCICHUNKER_AVAILABLE:
        try:
            sc      = ScientificPaperChunker(
                chunk_chars          = chunk_chars,
                overlap_chars        = overlap_chars,
                context_inject_chars = min(2000, chunk_chars // 3),
            )
            batches = sc.chunk(markdown_text, source_label=doc_title)
            log_cb(
                f"[SciChunk] {len(batches)} context-aware batches "
                f"(Methods→Results locality injection active)"
            )
            use_flat = False
        except Exception as exc:
            log_cb(f"[SciChunk] {exc} — falling back to HierarchicalChunker", "warn")
            batches  = []
 
    # Priority 2: HierarchicalChunker
    if use_flat and use_hierarchical and _HIER_CHUNKER_AVAILABLE and HierarchicalChunker:
        try:
            hier     = HierarchicalChunker(db_path=os.path.join(DATA_DIR, "chunks.db"))
            doc_hash = hier.ingest(markdown_text, source_label=doc_title)
            stats    = hier.doc_stats(doc_hash)
            log_cb(
                f"[HChunk] {stats.get('sections',0)} sections, "
                f"{stats.get('paragraphs',0)} paragraphs, "
                f"{stats.get('sentences',0)} sentences "
                f"({stats.get('species_sentences',0)} with species signal)"
            )
            batches = list(hier.extraction_batches(
                doc_hash,
                window_sentences = 5,
                max_batch_chars  = chunk_chars,
                species_only     = True,
            ))
            hier.close()
            log_cb(f"[HChunk] {len(batches)} extraction batches")
            use_flat = False
        except Exception as exc:
            log_cb(f"[HChunk] {exc} — falling back to flat chunks", "warn")
            batches  = []
 
    # Priority 3: DocumentChunker flat
    if use_flat:
        if _CHUNKER_AVAILABLE and DocumentChunker:
            chunker = DocumentChunker(
                strategy      = chunk_strategy,
                chunk_chars   = chunk_chars,
                overlap_chars = overlap_chars,
                model_name    = model_sel,
                batch_mode    = batch_mode,
            )
            flat_chunks, c_stats = chunker.chunk_markdown(
                markdown_text, source_label=doc_title
            )
            log_cb(
                f"[Chunk] {c_stats.total_chunks} chunks "
                f"| strategy={c_stats.strategy_used}"
            )
        else:
            # Priority 4: naive fixed-size slices
            step = max(chunk_chars - overlap_chars, 1000)
 
            class _FC:
                def __init__(self, i, t):
                    self.chunk_id            = i
                    self.text                = t
                    self.section             = f"Chunk {i + 1}"
                    self.has_species         = True
                    self.candidate_localities = []
                    self.candidate_species   = []
 
            flat_chunks = [
                _FC(i, markdown_text[s: s + chunk_chars])
                for i, s in enumerate(
                    range(0, min(len(markdown_text), 20_000), step)
                )
            ]
            log_cb(f"[Chunk] {len(flat_chunks)} naive fallback chunks")
 
    # ── Extraction loop — delegates entirely to process_chunk() ──────────────
    #
    #  process_chunk() owns: pre-pass → NER/Thinker → LLM → JSON clean
    #                        → Pydantic parse → LS-filter → Loc-filter
    #                        → field normalisation
    #
    #  This loop owns:  chunk iteration, skip/error counting, result collection.
    #  No inlined LLM calls.  No inlined JSON parsing.
 
    results:  list[dict] = []
    error_ct = skip_ct   = 0
 
    all_chunks = batches if batches else flat_chunks
 
    for chunk in all_chunks:
 
        text           = getattr(chunk, "context", None) or getattr(chunk, "text", str(chunk))
        section_label  = getattr(chunk, "section", f"chunk-{getattr(chunk, 'chunk_id', 0) + 1}")
        candidate_locs = list(getattr(chunk, "candidate_localities", []))
 
        # Skip non-species chunks in large documents (LLM call budget)
        if len(all_chunks) > 10 and not getattr(chunk, "has_species", True):
            skip_ct += 1
            continue
    
    
        result = process_chunk(
            text=text, 
            section_label=section_label,
            schema_prompt=schema_prompt, 
            cite_str=cite_str,
            provider=provider, 
            model_sel=model_sel,
            api_key=api_key, 
            ollama_base_url=ollama_base_url,
            use_thinker=use_thinker, 
            candidate_locs=candidate_locs, 
            log_cb=log_cb,
        )
 
        if result.status == "skip":
            skip_ct += 1
        elif result.status == "error":
            error_ct += 1
        else:
            results.extend(result.records)
 
    log_cb(
        f"[Extract] Raw total: {len(results)} "
        f"| errors: {error_ct} | skipped: {skip_ct}"
    )
 
    # ── Dedup pipeline (stages 1-3) ───────────────────────────────────────────
    # suppress_regional_duplicates is imported at module top — no late import.
 
    removed_records: list[dict] = []
    if dedup_occurrences and len(results) > 1:
        before = list(results)
        results, n_removed = dedup_occurrences(results)
        if n_removed:
            removed_records = [r for r in before if r not in results]
            log_cb(f"[Dedup] Removed {n_removed} duplicates (stages 1+2)")

    # PATCHED: P5-checklist-mode-suppress — honour checklist mode in stage-3 dedup
    _checklist_mode = st.session_state.get("is_checklist", False) if 'st' in dir() else False
    results, n_suppressed = suppress_regional_duplicates(
        results, checklist_mode=_checklist_mode
    )
    if n_suppressed:
        log_cb(f"[Dedup/Stage3] Suppressed {n_suppressed} regional-level duplicates"
               f" (checklist_mode={_checklist_mode})")

    # Feed dedup result into tracker
    if hasattr(log_cb, 'log_dedup_result'):
        log_cb.log_dedup_result(results, removed_records)
 
    # ── Auto Locality NER enrichment (v5.3) ───────────────────────────────────
    if use_auto_loc_ner and _LOC_NER_AVAILABLE and LocalityNER:
        try:
            log_cb("[LocalityNER] Auto-enriching localities…")
            _gdb = geonames_db or (GEONAMES_DB if os.path.exists(GEONAMES_DB) else "")
            lner = LocalityNER(
                geonames_db   = _gdb,
                pincode_txt   = PINCODE_TXT if os.path.exists(PINCODE_TXT) else "",
                use_nominatim = False,
            )
            results = lner.enrich_occurrences(results, markdown_text, proximity_chars=600)
            filled  = sum(
                1 for r in results
                if isinstance(r, dict)
                and r.get("decimalLatitude")
                and r.get("decimalLongitude")
            )
            log_cb(f"[LocalityNER] {filled}/{len(results)} records have coordinates")
        except Exception as exc:
            log_cb(f"[LocalityNER] {exc}", "warn")
 
    return results


def enrich_taxonomy(occurrences: list[dict], log_cb, wiki=None) -> list[dict]:
    """Shared taxonomy verification with unified cascade fallback."""
    if not occurrences:
        return occurrences

    if verify_occurrences_with_fallback is None:
        log_cb("[Taxonomy] verifier unavailable — skipped", "warn")
        return occurrences

    log_cb(f"[Taxonomy] Verifying {len(occurrences)} records with unified cascade…")
    try:
        return verify_occurrences_with_fallback(
            occurrences,
            log_cb=log_cb,
            wiki=wiki,
            cache_db=META_DB_PATH,
        )
    except Exception as exc:
        log_cb(f"[Taxonomy] Error: {exc}", "warn")
        return occurrences


def split_localities(occurrences: list[dict], log_cb,
                     geonames_db: str = GEONAMES_DB) -> list[dict]:
    """v5.1: Expand compound locality strings into individual occurrence records."""
    if not _GNV_AVAILABLE or LocalitySplitter is None:
        return occurrences
    try:
        splitter   = LocalitySplitter(geonames_db=geonames_db, use_nominatim=True)
        expanded   = splitter.split_localities(occurrences, geocode_new=True)
        n_added    = len(expanded) - len(occurrences)
        if n_added > 0:
            log_cb(f"[Locality] Split {n_added} compound localities → {len(expanded)} records")
        splitter.close()
        return expanded
    except Exception as exc:
        log_cb(f"[Locality] Split error: {exc}", "warn")
        return occurrences


def geocode_occurrences(occurrences: list[dict], log_cb) -> list[dict]:
    if not occurrences or GeocodingCascade is None:
        if GeocodingCascade is None:
            log_cb("[Geocoding] geocoding_cascade unavailable — skipped", "warn")
        return occurrences
    try:
        geo = GeocodingCascade(
            geonames_db   = GEONAMES_DB  if os.path.exists(GEONAMES_DB) else "",
            pincode_txt   = PINCODE_TXT  if os.path.exists(PINCODE_TXT) else "",
            use_nominatim = True,
        )
        log_cb(f"[Geocoding] Processing {len(occurrences)} records…")
        return geo.geocode_batch(occurrences)
    except Exception as exc:
        log_cb(f"[Geocoding] Error: {exc}", "warn")
        return occurrences


# ─────────────────────────────────────────────────────────────────────────────
#  V5 ENHANCED PIPELINE — KG + Memory Bank + Wiki
# ─────────────────────────────────────────────────────────────────────────────


# @st.cache_resource
from typing import NamedTuple
 
class _ModuleResult(NamedTuple):
    """Thin wrapper so tabs can show real errors, not generic 'Install' messages."""
    instance: object        # None if unavailable
    error:    str = ""      # human-readable error string
 
    def __bool__(self):
        return self.instance is not None
 
    def __getattr__(self, name):
        # Proxy attribute access to the wrapped instance so callers like
        # `kg.stats()` still work without unwrapping.
        if self.instance is not None:
            return getattr(self.instance, name)
        raise AttributeError(f"Module unavailable ({self.error}): .{name}")
 
 
@st.cache_resource
def get_knowledge_graph() -> _ModuleResult:
    if not _KG_AVAILABLE:
        return _ModuleResult(None, _KG_ERROR or "biotrace_knowledge_graph.py not importable")
    try:
        return _ModuleResult(BioTraceKnowledgeGraph(KG_DB_PATH))
    except Exception as exc:
        logger.error("[v5] KG init: %s", exc)
        return _ModuleResult(None, str(exc))
 
 
@st.cache_resource
def get_memory_bank() -> _ModuleResult:
    if not _MB_AVAILABLE:
        return _ModuleResult(None, _MB_ERROR or "biotrace_memory_bank.py not importable")
    try:
        return _ModuleResult(BioTraceMemoryBank(MB_DB_PATH))
    except Exception as exc:
        logger.error("[v5] MemoryBank init: %s", exc)
        return _ModuleResult(None, str(exc))
 


# @st.cache_resource
@st.cache_resource
def get_wiki() -> "BioTraceWikiUnified | None":
    """Returns the singleton BioTraceWikiUnified instance (versioned SQLite store)."""
    if not _WIKI_AVAILABLE:
        return None
    try:
        css_path = os.path.join(os.path.dirname(__file__), "biotrace_wiki.css")
        return BioTraceWikiUnified(
            root_dir = WIKI_ROOT,
            css_path = css_path if os.path.exists(css_path) else None,
        )
    except Exception as exc:
        logger.error("[v5.5] Wiki init: %s", exc)
        return None


@st.cache_resource
def get_wiki_agent() -> "OllamaWikiAgent | None":
    """Returns an OllamaWikiAgent bound to the shared wiki store."""
    if not _WIKI_AGENT_AVAILABLE:
        return None
    wiki = get_wiki()
    if not wiki:
        return None
    # model / URL come from sidebar — read from session_state with defaults
    model    = st.session_state.get("ollama_model_sel", "gemma4")
    base_url = st.session_state.get("ollama_url",       "http://localhost:11434")
    try:
        return OllamaWikiAgent(wiki=wiki, model=model, base_url=base_url)
    except Exception as exc:
        logger.error("[v5.5] WikiAgent init: %s", exc)
        return None


def ingest_into_v5_systems(
    occurrences: list[dict],
    citation: str,
    session_id: str,
    log_cb,
    provider: str = "",
    model_sel: str = "",
    api_key: str  = "",
    ollama_base_url: str = "http://localhost:11434",
    update_wiki_narratives: bool = False,
    chunk_text: str = "",       # v5.5 — passed to wiki for LLM section updates
    use_kg: bool = True,
    use_mb: bool = True,
    use_wiki: bool = True,
):
    """Push verified/geocoded occurrences into KG + Memory Bank + Wiki."""
    llm_fn = None
    if update_wiki_narratives:
        def llm_fn(prompt: str) -> str:
            return call_llm(prompt, provider, model_sel, api_key, ollama_base_url)

    # Knowledge Graph
    kg = get_knowledge_graph() if use_kg else None
    if kg:
        try:
            added = kg.ingest_occurrences(occurrences)
            log_cb(f"[KG] +{added} nodes. Total: {kg.stats()['total_nodes']}")
        except Exception as exc:
            log_cb(f"[KG] Ingest error: {exc}", "warn")

    # Memory Bank
    mb = get_memory_bank() if use_mb else None
    if mb:
        try:
            r = mb.store_occurrences(
                occurrences, session_id=session_id,
                session_title=citation, source_file=session_id,
            )
            log_cb(
                f"[MemoryBank] inserted={r['inserted']} merged={r['merged']} "
                f"conflicts={r['conflicts']}"
            )
        except Exception as exc:
            log_cb(f"[MemoryBank] Store error: {exc}", "warn")

    # Wiki — BioTraceWikiUnified (versioned, CSS-styled, LLM-enhanced)
    # wiki = get_wiki() if use_wiki else None
    # if wiki:
    #     try:
    #         counts = wiki.update_from_occurrences(
    #             occurrences,
    #             citation           = citation,
    #             llm_fn             = llm_fn,
    #             update_narratives  = update_wiki_narratives,
    #             chunk_text         = chunk_text,
    #         )
    #         log_cb(f"[Wiki] Updated: {counts}")
    #     except Exception as exc:
    #         log_cb(f"[Wiki] Update error: {exc}", "warn")    
    
    
    # Wiki (BioTraceWikiUnified — versioned, CSS-styled, LLM-enhanced)
    wiki = get_wiki() if use_wiki else None
    if wiki:
        try:
            counts = wiki.update_from_occurrences(
                occurrences,
                citation           = citation,
                llm_fn             = llm_fn,
                update_narratives  = update_wiki_narratives,
                chunk_text         = chunk_text,    # NEW: enables targeted section update
            )
            log_cb(f"[Wiki] Updated: {counts}")

            # ── NEW: Auto-Run Wiki Architect Agent on Docling Chunks ──
            # This uses the Docling sections (chunk_text) to automatically 
            # run the agent loop for newly extracted species.
            if update_wiki_narratives and chunk_text and _WIKI_AGENT_AVAILABLE:
                agent = get_wiki_agent()
                if agent:
                    # Get unique species names from this batch
                    unique_sps = list({
                        (o.get("validName") or o.get("recordedName")) 
                        for o in occurrences 
                        if (o.get("validName") or o.get("recordedName"))
                    })
                    
                    log_cb(f"[WikiAgent] Auto-running architect on {len(unique_sps)} species using Docling sections...")
                    
                    for sp in unique_sps:
                        try:
                            # Invoke the primary agent extraction method
                            # (Adjust 'enhance_article' to match your agent's actual entry point if different)
                            if hasattr(agent, "enhance_article"):
                                agent.enhance_article(sp, chunk_text)
                            elif hasattr(agent, "run_agent_pipeline"):
                                agent.run_agent_pipeline(sp, chunk_text)
                        except Exception as e_agent:
                            log_cb(f"[WikiAgent] Error processing {sp}: {e_agent}", "warn")
            # ──────────────────────────────────────────────────────────

        except Exception as exc:
            log_cb(f"[Wiki] Update error: {exc}", "warn")

    # Wiki
    # wiki = get_wiki() if use_wiki else None  # FIX 2: respect toggle
    # if wiki:
    #     try:
    #         counts = wiki.update_from_occurrences(
    #             occurrences, citation=citation,
    #             llm_fn=llm_fn,
    #             update_narratives=update_wiki_narratives,
    #         )
    #         log_cb(f"[Wiki] Updated: {counts}")
    #     except Exception as exc:
    #         log_cb(f"[Wiki] Update error: {exc}", "warn")
    
    #29042026
    # Wiki (BioTraceWikiUnified — versioned, CSS-styled, LLM-enhanced)
    wiki = get_wiki() if use_wiki else None
    if wiki:
        try:
            counts = wiki.update_from_occurrences(
                occurrences,
                citation           = citation,
                llm_fn             = llm_fn,
                update_narratives  = update_wiki_narratives,
                chunk_text         = chunk_text,    # NEW: enables targeted section update
            )
            log_cb(f"[Wiki] Updated: {counts}")
        except Exception as exc:
            log_cb(f"[Wiki] Update error: {exc}", "warn")
            
            
    


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: load all occurrences from SQLite
# ─────────────────────────────────────────────────────────────────────────────
def db_load_all() -> pd.DataFrame:
    try:
        con = sqlite3.connect(META_DB_PATH)
        df  = pd.read_sql_query("SELECT * FROM occurrences_v4 ORDER BY id DESC", con)
        con.close()
        return df
    except Exception:
        return pd.DataFrame()



# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioTrace v5 — Marine Biodiversity Extractor",
    page_icon="🐚",
    layout="wide",
)

# v5.5: Inject wiki CSS once at startup so it's available in any tab that
# renders wiki HTML via st.components.v1.html().
try:
    inject_css_streamlit()
except Exception:
    pass

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #0d1117; color: #e6edf3; }
  .stSidebar { background: #161b22; }
  .metric-card {
    background: #21262d; border-radius: 8px;
    padding: 12px 18px; margin: 6px 0;
    border-left: 4px solid #2E86AB;
  }
  .v5-badge {
    background: linear-gradient(135deg, #2E86AB, #44BBA4);
    color: white; padding: 2px 8px; border-radius: 12px;
    font-size: 0.75em; font-weight: bold;
  }
</style>
""", unsafe_allow_html=True)




# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown(
    "# 🐚 BioTrace <span class='v5-badge'>v5.0</span>",
    unsafe_allow_html=True,
)
st.caption(
    "Marine Biodiversity Record Extractor · "
    "GraphRAG + Memory Bank + LLM-Wiki Edition"
)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR — CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # LLM provider
    st.subheader("LLM Provider")
    _providers = ["Ollama (Local)", "Anthropic via Ollama", "OpenAI", "Gemini"]
    provider   = st.selectbox("Provider", _providers,key="_providers")
    api_key    = ""
    if provider in ("OpenAI", "Gemini", "Anthropic via Ollama"):
        api_key = st.text_input("API Key", type="password",
                                placeholder="ollama" if provider == "Anthropic via Ollama" else "")
    ollama_url = st.text_input("Ollama URL", "http://localhost:11434")

    # Model — live Ollama combobox or static input for cloud providers
    if provider in ("Ollama (Local)", "Anthropic via Ollama"):
        if _ENH_AVAILABLE and _render_ollama_model_selector:
            model_sel = _render_ollama_model_selector(
                base_url=ollama_url,
                key="main_model_sel",
            )
        else:
            model_sel = st.text_input("Model", "gemma4")
    else:
        _default_cloud = {"OpenAI": "gpt-4o-mini", "Gemini": "gemini-2.0-flash"}
        model_sel = st.text_input("Model", _default_cloud.get(provider, "gpt-4o-mini"))

    st.divider()
    with st.expander("🧩 Chunking & Extraction Options", expanded=False):
        use_thinker_cb = st.checkbox("extract_thinker pre-pass", value=True,
                                help="LLM inventory step before structured extraction")
        
        use_biodiviz = st.checkbox(
            "Use BiodiViz NER + RE",
            value=_BIODIVIZ_AVAILABLE,
            key="use_biodiviz",
            help="Use local Hugging Face models to map species to localities",
        )
    st.divider()
    # Gemma 4 / large-context hint
    _is_large_ctx = any(x in model_sel.lower() for x in ["gemma4","gemma3","llama3.3","qwen2.5"])
    
    if _is_large_ctx:
        st.caption("🚀 Large-context model — batch mode enabled")
    st.divider()
    
    # Chunking strategy
    st.subheader("📐 Chunking Strategy")
    chunk_strategy = st.selectbox(
        "Strategy",
        ["section (recommended)", "paragraph", "fixed"],
        help="section: split on Markdown headings (best for academic papers)"
    ).split(" ")[0]
    col_ck1, col_ck2 = st.columns(2)
    chunk_chars   = col_ck1.number_input("Chunk size (chars)", 2000, 50000, 6000, 500)
    overlap_chars = col_ck2.number_input("Overlap (chars)", 0, 2000, 400, 50)

    # PDF parser
    st.subheader("PDF Parser")
    available_parsers = []
    if _PYMUPDF_AVAILABLE:   available_parsers.append("pymupdf4llm (fast fallback)")
    if _MARKITDOWN_AVAILABLE:available_parsers.append("markitdown (lightweight)")
    if _DOCLING_AVAILABLE:   available_parsers.append("docling (IBM, structured)")
    if not available_parsers:available_parsers = ["None — install pymupdf4llm"]
    parser_choice = st.selectbox("PDF Parser", available_parsers)

    st.divider()

    # V5 features
    st.subheader("🔮 v5 Features")
    col1, col2 = st.columns(2)
    with col1:
        use_kg   = st.toggle("Knowledge Graph",  value=_KG_AVAILABLE)
        use_mb   = st.toggle("Memory Bank",       value=_MB_AVAILABLE)
    with col2:
        use_wiki = st.toggle("LLM-Wiki",          value=_WIKI_AVAILABLE)
        wiki_narr= st.toggle("Wiki Narratives",   value=True)

    st.divider()

    # System status
    st.subheader("📡 System Status")
    def _status(ok: bool, name: str, hint: str = "", is_local: bool = False):
        icon = "✅" if ok else "❌"
        if not ok and hint:
            tip = (
                f" — copy `{hint}` into project folder"
                if is_local
                else f" — `pip install {hint}`"
            )
        else:
            tip = ""
        st.caption(f"{icon} {name}{tip}")

    st.caption("**pip packages**")
    _status(_PYMUPDF_AVAILABLE,    "pymupdf4llm",       "pymupdf4llm")
    _status(_MARKITDOWN_AVAILABLE, "markitdown",        "markitdown")
    _status(_OLLAMA_AVAILABLE,     "ollama client",     "ollama")
    _status(_OPENAI_AVAILABLE,     "openai",            "openai")
    _status(_PLOTLY_AVAILABLE,     "plotly",            "plotly")

    st.caption("**local modules** (place alongside biotrace_v5.py)")
    _status(_VERIFIER_AVAILABLE,   "biotrace_taxonomy.py",         "biotrace_taxonomy.py",         is_local=True)
    _status(_GEOCODER_AVAILABLE,   "geocoding_cascade.py",         "geocoding_cascade.py",         is_local=True)
    _status(_CHUNKER_AVAILABLE,    "biotrace_chunker.py",          "biotrace_chunker.py",          is_local=True)
    _status(_GNV_AVAILABLE,        "biotrace_gnv.py",              "biotrace_gnv.py",              is_local=True)
    _status(_KG_AVAILABLE,         "biotrace_knowledge_graph.py",  "biotrace_knowledge_graph.py",  is_local=True)
    _status(_MB_AVAILABLE,         "biotrace_memory_bank.py",      "biotrace_memory_bank.py",      is_local=True)
    _status(_WIKI_AVAILABLE,       "biotrace_wiki.py",             "biotrace_wiki.py",             is_local=True)
    _status(_NER_AVAILABLE,        "biotrace_ner.py",              "biotrace_ner.py",              is_local=True)
    _status(_LOC_NER_AVAILABLE,    "biotrace_locality_ner.py",     "biotrace_locality_ner.py",     is_local=True)
    _status(_SCHEMA52_AVAILABLE,   "biotrace_schema.py",           "biotrace_schema.py",           is_local=True)
    _status(_ENH_AVAILABLE,        "biotrace_v5_enhancements.py",  "biotrace_v5_enhancements.py",  is_local=True)
    _status(_PDF_META_AVAILABLE,   "biotrace_pdf_meta.py",         "biotrace_pdf_meta.py",         is_local=True)
    _status(_HIER_CHUNKER_AVAILABLE,"biotrace_hierarchical_chunker.py","biotrace_hierarchical_chunker.py",is_local=True)

    st.caption("**enhancement patch modules** (place alongside biotrace_v5.py)")
    # [ENHANCEMENT MODULE STATUS] — new patch + enhancement modules
    _status(True,  "biotrace_geocoding_lifestage_patch.py", "biotrace_geocoding_lifestage_patch.py", is_local=True)
    _status(True,  "biotrace_locality_guard_patch.py",      "biotrace_locality_guard_patch.py",      is_local=True)
    _status(True,  "biotrace_dedup_patch.py",               "biotrace_dedup_patch.py",               is_local=True)
    _status(True,  "biotrace_traiter_prepass.py",           "biotrace_traiter_prepass.py",           is_local=True)
    _status(True,  "biotrace_col_client.py",                "biotrace_col_client.py",                is_local=True)
    _status(True,  "biotrace_relation_extractor.py",        "biotrace_relation_extractor.py",        is_local=True)
    _status(True,  "biotrace_kg_spatio_temporal.py",        "biotrace_kg_spatio_temporal.py",        is_local=True)
    # PATCHED: SIDEBAR-new-module-status
    _status(True,  "biotrace_gbif_verifier.py",             "biotrace_gbif_verifier.py",             is_local=True)
    _status(True,  "biotrace_agent_loop.py",                "biotrace_agent_loop.py",                is_local=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
init_db()
tabs = st.tabs([
    "📄 Extract",
    "🔬 TNR Engine",
    "📍 Locality NER",
    "✏️ Verification & Coordinates",
    "🕸️ Knowledge Graph",
    "🧠 Memory Bank",
    "📖 Wiki",
    "🔍 GraphRAG Query",
    "📊 Database",
    "🛡️ Schema",
    "📥 Export",
])

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — EXTRACT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Upload & Extract Occurrence Records")

    uploaded = st.file_uploader(
        "Upload PDF or Markdown",
        type=["pdf","md","txt"],
        help="Any biodiversity paper — marine, terrestrial, freshwater, botanical",
    )

    # ── Auto-Titling Extraction Logic ─────────────────────────────────────────
    if uploaded:
        if 'auto_title' not in st.session_state or st.session_state.get('last_uploaded') != uploaded.name:
            st.session_state['last_uploaded'] = uploaded.name
            # st.session_state['auto_title'] = uploaded.name 
            st.session_state['auto_title'] = ""   # don't pre-fill with filename
            if uploaded.name.lower().endswith('.pdf'):
                file_bytes = uploaded.getvalue()
                def dummy_llm_title(p):
                    return call_llm(p, provider, model_sel, api_key, ollama_url)

                try:
                    import title_extractor
                    extracted = title_extractor.extract_title(file_bytes, None, uploaded.name, dummy_llm_title)
                    if extracted and len(extracted.strip()) > 3:
                        st.session_state['auto_title'] = extracted
                except Exception as e:
                    logger.warning(f"Title extraction failed: {e}")
                    
            # PATCHED: P6-pdf-backup-gate — use original filename; skip if already saved
            backup_dir = os.path.join(DATA_DIR, "backup_manuscripts")
            os.makedirs(backup_dir, exist_ok=True)
            # Use the original uploaded filename (not auto_title which may be blank)
            safe_backup_name = re.sub(r'[\\/*?:"<>|]', "_", uploaded.name)
            backup_path = os.path.join(backup_dir, safe_backup_name)
            if not os.path.exists(backup_path):
                with open(backup_path, "wb") as f:
                    f.write(uploaded.getvalue())
                logger.info("[upload] Backup saved: %s", backup_path)
            else:
                logger.debug("[upload] Backup already exists, skipping: %s", backup_path)
            st.session_state['backup_path'] = backup_path

    # ── PDF Metadata panel ────────────────────────────────────────────────────
    with st.expander("📑 PDF Metadata & Rename", expanded=False):
        col_m1, col_m2 = st.columns([2,1])
        with col_m1:
            meta_email = st.text_input(
                "Contact email (Crossref polite pool):",
                placeholder="researcher@institution.edu",
                key="meta_email",
            )
            s2_key = st.text_input(
                "Semantic Scholar API key (optional — raises rate limit):",
                type="password", key="s2_key",
            )
        with col_m2:
            do_rename_pdf  = st.checkbox("Auto-rename PDF after fetch", value=True)
            doi_override   = st.text_input("DOI override:", placeholder="10.xxxx/xxxx")

        if _PDF_META_AVAILABLE:
            st.caption(
                "✅ PaperMetaFetcher — Semantic Scholar (80 calls/5 min) + "
                "Crossref + PDF DOI extraction"
            )
        else:
            st.caption("❌ biotrace_pdf_meta.py not found — place it alongside biotrace_v5.py")

    # PATCHED: P4-checklist-hitl-toggles — add checklist mode and HITL approval
    col_a, col_b, col_c = st.columns([3,1,1])
    with col_a:
        doc_title = st.text_input(
            "Document Title / Citation (auto-filled from metadata):",
            value=st.session_state.get('auto_title', '') if uploaded else ''
        )
    with col_b:
        primary_only = st.checkbox("Primary records only", value=False)
        is_checklist = st.checkbox(
            "📋 Checklist paper mode",
            value=False,
            help="Keeps 'cf.', 'sp.', and authority forms as separate entries. "
                 "Use for annotated checklists where the table lists them distinctly.",
            key="is_checklist",
        )
    with col_c:
        do_split_loc = st.checkbox("Split localities", value=True,
                                   help="Expand 'Site A, B and C' → 3 records")
        use_hitl = st.checkbox(
            "🔬 Approve before saving",
            value=True,
            help="HITL gate: review + approve species before they enter DB/KG/Memory.",
            key="use_hitl_approval",
        )

    # ── v5.3 chunking controls ────────────────────────────────────────────────
    with st.expander("🧩 Chunking & Extraction Options", expanded=False):
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            # use_hierarchical = st.checkbox(
            #     "Hierarchical late-chunking (v5.3 — recommended)",
            #     value=_HIER_CHUNKER_AVAILABLE,
            #     help="3-level (section→paragraph→sentence) solves species/locality separation",
            # )
            # v5.5: Hierarchical chunking is always-on internally (Priority-2 fallback).
            # The user checkbox has been removed to simplify the UI.
            use_hierarchical = _HIER_CHUNKER_AVAILABLE   # always True when module present
        
            use_scientific = st.checkbox(
                "Scientific paper chunker (v5.4)",
                value=_SCICHUNKER_AVAILABLE,
                key="use_scientific",
                help="Injects Methods-section localities into Results chunks — "
                    "fixes species/locality split across section boundaries",
            )
            use_agentic = st.checkbox(
                "Agentic chunking (v5.7)",
                value=_AGENTIC_AVAILABLE,
                key="use_agentic",
                help="Uses pydantic-ai agent to chunk text based on semantic boundaries.",
            )
            use_thinker_cb = st.checkbox("extract_thinker pre-pass", value=True,
                                          help="LLM inventory step before structured extraction")
        with col_h2:
            use_auto_loc = st.checkbox(
                "Auto Locality NER after extraction",
                value=_LOC_NER_AVAILABLE,
                help="Automatically expand localities and fill lat/lon using GeoNames",
            )
            use_nominatim_auto = st.checkbox("Include Nominatim (1 req/sec)", value=False)

    run_btn = st.button("🚀 Extract + Enrich", type="primary", disabled=(uploaded is None))


    # ── HITL Resume — MUST be outside run_btn guard ───────────────────────────
    # When the user clicks Confirm in the approval table, Streamlit re-runs the
    # entire script. On that re-run run_btn is False, so the old resume logic
    # (nested inside `if run_btn`) was silently skipped — occurrences never
    # reached insert_occurrences / ingest_into_v5_systems / SpatioKG.
    # FIX 1: This block now fires on every script run, checks the checkpoint,
    # and short-circuits with st.stop() after persisting the approved records.
    _hitl_resume = st.session_state.get("_hitl_pending_occurrences")
    if _hitl_resume is not None and uploaded is not None:
        _log_c = st.container()
        _se: list[str] = []
        _lcb  = BioTraceLogger(_log_c, _se)
        try:
            from biotrace_gbif_verifier import render_approval_table as _rat
            _appr = _rat(_hitl_resume)
        except ImportError:
            _appr = _hitl_resume  # verifier absent — treat all as approved

        if _appr is None:
            st.info("⏳ Review the species above and click **Confirm** to save.")
            st.stop()  # biologist hasn't confirmed yet

        # Biologist confirmed — drain checkpoint first to avoid re-entry
        _h_hash  = st.session_state.pop("_hitl_pending_hash",     "")
        _h_sess  = st.session_state.pop("_hitl_pending_session",  "session_resumed")
        _h_cite  = st.session_state.pop("_hitl_pending_citation", uploaded.name)
        _h_text  = st.session_state.pop("_hitl_pending_text",     "")
        st.session_state.pop("_hitl_pending_title",       None)
        st.session_state.pop("_hitl_pending_occurrences", None)

        _lcb(f"[HITL] {len(_appr)} species approved — persisting now…")

        # Geocode
        _appr = geocode_occurrences(_appr, _lcb)

        # Post-processing
        try:
            from biotrace_postprocessing import run_postprocessing
            _appr, _pp = run_postprocessing(
                _appr, citation_str=_h_cite,
                wiki_root=WIKI_ROOT, geonames_db=GEONAMES_DB,
                use_nominatim=True, log_cb=_lcb,
            )
            st.session_state["pp_conflicts"]    = _pp.get("conflicts", [])
            st.session_state["pp_conflict_log"] = _pp.get("conflict_log", [])
        except Exception as _exc:
            _lcb(f"[Post] {_exc}", "warn")

        # Occurrence DB
        _n = insert_occurrences(_appr, _h_hash, _h_cite, _h_sess)
        _lcb(f"[DB] {_n} records saved (HITL-approved, session {_h_sess})")

        # KG + Memory Bank + Wiki — respect sidebar toggles
        if any([use_kg, use_mb, use_wiki]):
            ingest_into_v5_systems(
                _appr, citation=_h_cite, session_id=_h_sess, log_cb=_lcb,
                provider=provider, model_sel=model_sel,
                api_key=api_key, ollama_base_url=ollama_url,
                update_wiki_narratives=wiki_narr,
                chunk_text=_h_text,
                use_kg=use_kg, use_mb=use_mb, use_wiki=use_wiki,
            )

        # SpatioTemporal KG
        try:
            from biotrace_kg_spatio_temporal import BioTraceSpatioTemporalKG
            _stkg = BioTraceSpatioTemporalKG(META_DB_PATH)
            _stkg.upsert_from_occurrences(_appr)
            _lcb(f"[SpatioKG] {len(_appr)} species nodes upserted after HITL approval")
        except Exception as _exc:
            _lcb(f"[SpatioKG] {_exc}", "warn")

        st.success(f"✅ {len(_appr)} occurrence records saved after approval.")
        _df = pd.DataFrame(_appr)
        st.dataframe(_df[[c for c in
            ["recordedName","validName","family_","phylum","verbatimLocality",
             "occurrenceType","wormsID"] if c in _df.columns]],
            width=True, height=350)
        st.stop()  # done — prevent fall-through to run_btn pipeline

    if run_btn and uploaded:
        log_container = st.container()
        # logs: list[str] = []
        # _schema_errors: list[str] = []

        # def log_cb(msg: str, lvl: str = "ok"):
        #     logs.append(f"[{lvl.upper()}] {msg}")
        #     if lvl == "warn":
        #         _schema_errors.append(msg)
        #     with log_container:
        #         icon = {"ok":"✅","warn":"⚠️","error":"❌"}.get(lvl,"ℹ️")
        #         st.write(f"{icon} {msg}")
        
        _schema_errors: list[str] = []
        log_inst  = BioTraceLogger(log_container, _schema_errors)
        log_cb    = log_inst   # drop-in: same __call__ signature
        progress_placeholder = st.empty()



        with st.spinner("Processing…"):
            clean_title = st.session_state.get('auto_title', uploaded.name)
            ts = int(time.time())
            suffix = Path(uploaded.name).suffix
            
            # PATCHED: P7-pdf-hash-filename — content hash prevents re-extraction duplicates
            safe_title = re.sub(r'[\\/*?:"<>|]', "_", clean_title or Path(uploaded.name).stem)
            # Pre-compute hash here so we can use it in the filename
            _pre_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()[:8]
            filename = f"{safe_title}_{_pre_hash}{suffix}"
            tmp_path = os.path.join(PDF_DIR, filename)
            
            raw_bytes = uploaded.getvalue()
            with open(tmp_path, "wb") as f:
                f.write(raw_bytes)

            file_hash  = hashlib.sha256(raw_bytes).hexdigest()[:16]
            session_id = f"session_{ts}"

            log_cb(f"File saved dynamically as: {filename}")

            # ── Step 1: PDF Metadata fetch ────────────────────────────────────
            citation_str = doc_title or ""
            final_pdf_path = tmp_path

            if _PDF_META_AVAILABLE and PaperMetaFetcher and suffix.lower() == ".pdf":
                log_cb("[Meta] Fetching paper metadata (S2 → Crossref → PDF)…")
                try:
                    llm_fn_meta = lambda p: call_llm(p, provider, model_sel, api_key, ollama_url)
                    fetcher = PaperMetaFetcher(
                        email      = meta_email or "biotrace@example.com",
                        s2_api_key = s2_key or "",
                        llm_fn     = llm_fn_meta,
                    )
                    meta, final_pdf_path = fetcher.fetch_and_rename(
                        tmp_path,
                        dest_dir  = PDF_DIR,
                        doi_hint  = doi_override or "",
                        title_hint= doc_title or "",
                    )
                    if meta.is_complete():
                        citation_str = meta.citation_string
                        if not doc_title:
                            doc_title = meta.title
                        log_cb(
                            f"[Meta] ✓ {meta.source}: {meta.title[:60]}… "
                            f"({meta.year}) via {meta.source}"
                        )
                        log_cb(f"[Meta] Citation: {citation_str[:120]}…")
                        if do_rename_pdf and final_pdf_path != tmp_path:
                            log_cb(f"[Meta] PDF renamed → {Path(final_pdf_path).name}")
                    else:
                        log_cb(f"[Meta] Partial metadata only (source={meta.source})", "warn")
                        extracted_title = st.session_state.get('auto_title', '') or doc_title
                        citation_str = meta.citation_string or extracted_title or uploaded.name
                    
                    if not doc_title and extracted_title:
                        doc_title = extracted_title

                except Exception as exc:
                    log_cb(f"[Meta] Failed: {exc} — using filename as citation", "warn")
                    citation_str = doc_title or uploaded.name
            elif not citation_str:
                citation_str = doc_title or uploaded.name

            # ── Step 2: PDF → Markdown ────────────────────────────────────────
            if suffix.lower() == ".pdf":
                log_cb(f"[PDF] Parsing with {parser_choice}…")
                md_text = pdf_to_markdown(final_pdf_path, parser_choice)
            else:
                md_text = raw_bytes.decode("utf-8", errors="replace")

            if not doc_title:
                doc_title = st.session_state.get('auto_title', '') or citation_str or uploaded.name
            
            log_cb(f"[Extract] Text: {len(md_text):,} chars | Citation: {citation_str[:60]}…")

            # ── Step 3: LLM Extraction (hierarchical) ─────────────────────────
            # PATCHED: P9-agent-loop — self-correcting extraction with species-count check
            log_cb("[Extract] Running v5.3 hierarchical extraction…")

            def _run_standard_extraction(text_input):
                return extract_occurrences(
                    text_input, doc_title, provider, model_sel,
                    api_key, ollama_url, log_cb,
                    chunk_strategy   = chunk_strategy,
                    chunk_chars      = chunk_chars,
                    overlap_chars    = overlap_chars,
                    batch_mode       = False,
                    citation_string  = citation_str,
                    use_hierarchical = use_hierarchical,
                    use_scientific   = use_scientific,
                    use_agentic      = use_agentic,
                    use_thinker      = use_thinker_cb,
                    use_auto_loc_ner = use_auto_loc and _LOC_NER_AVAILABLE,
                    geonames_db      = GEONAMES_DB,
                )

            try:
                from biotrace_agent_loop import agent_extract_with_correction
                _llm_partial = lambda p: call_llm(p, provider, model_sel, api_key, ollama_url)
                occurrences = agent_extract_with_correction(
                    full_text  = md_text,
                    extract_fn = _run_standard_extraction,
                    llm_fn     = _llm_partial,
                    log_cb     = log_cb,
                    max_retries = 2,
                )
            except ImportError:
                log_cb("[Agent] biotrace_agent_loop.py not found — standard extraction", "warn")
                occurrences = _run_standard_extraction(md_text)
            
            if hasattr(log_inst, 'log_extraction_result'):
                log_inst.log_extraction_result("document", occurrences)
                
            with progress_placeholder.container():
                render_species_progress_panel(log_inst.tracker)

            log_cb(f"[Extract] Raw records: {len(occurrences)}")

            if not occurrences:
                st.warning("No occurrences extracted. Check LLM provider settings.")
                st.stop()

            # ── Step 4: Taxonomy enrichment ───────────────────────────────────
            wiki_inst   = get_wiki() if use_wiki else None
            occurrences = enrich_taxonomy(occurrences, log_cb, wiki=wiki_inst)

            # ── Step 5: Locality splitting ────────────────────────────────────
            if do_split_loc:
                occurrences = split_localities(occurrences, log_cb)

            # ── Step 6: Primary filter ────────────────────────────────────────
            if primary_only:
                before = len(occurrences)
                occurrences = [
                    o for o in occurrences
                    if isinstance(o, dict) and
                       str(o.get("occurrenceType","")).lower() == "primary"
                ]
                log_cb(f"[Filter] Primary only: {len(occurrences)}/{before}")

            
            # PATCHED-R2: R1b-hitl-checkpoint — save checkpoint before gate fires
            if st.session_state.get("use_hitl_approval", True):
                try:
                    from biotrace_gbif_verifier import gbif_verify_batch, render_approval_table
                    log_cb("[GBIF] Verifying species against GBIF Backbone Taxonomy…")
                    occurrences = gbif_verify_batch(occurrences, min_confidence=80)
                    n_auto = sum(1 for o in occurrences if o.get("gbifApproved"))
                    log_cb(f"[GBIF] {n_auto}/{len(occurrences)} auto-approved")

                    # Save checkpoint so HITL confirm can resume without re-running pipeline
                    st.session_state["_hitl_pending_occurrences"] = occurrences
                    st.session_state["_hitl_pending_hash"]        = file_hash
                    st.session_state["_hitl_pending_session"]     = session_id
                    st.session_state["_hitl_pending_citation"]    = citation_str
                    st.session_state["_hitl_pending_title"]       = doc_title
                    st.session_state["_hitl_pending_text"]        = md_text

                    approved = render_approval_table(occurrences)
                    if approved is None:
                        st.stop()   # wait for biologist to confirm
                    # Confirmed — clear checkpoint and continue
                    del st.session_state["_hitl_pending_occurrences"]
                    occurrences = approved
                    log_cb(f"[HITL] {len(occurrences)} species approved for DB/KG/Memory")
                except ImportError:
                    log_cb("[GBIF] biotrace_gbif_verifier.py not found — skipping HITL gate",
                           "warn")

            # [ENHANCEMENT: biotrace_col_client] — Stage 5: COL taxonomy enrichment
            # Queries Catalogue of Life API; results cached in col_taxonomy_cache.
            # Runs after Step 4 so validName is already populated.
            from biotrace_col_client import enrich_records_with_col
            occurrences = enrich_records_with_col(occurrences, META_DB_PATH)
            log_cb(f"[COL] Enrichment complete ({len(occurrences)} records)")

            # [ENHANCEMENT: biotrace_relation_extractor] — Stage 3 (DeepKE-inspired)
            # Second LLM pass: cross-sentence relation triples per document.
            # FOUND_AT | CO_OCCURS_WITH | INHABITS | FEEDS_ON | PARASITE_OF
            # Stored in species_relations SQLite table; passed to SpatioKG below.
            from biotrace_relation_extractor import extract_relations
            species_in_batch = list({
                r.get("validName") or r.get("recordedName", "")
                for r in occurrences
                if r.get("validName") or r.get("recordedName")
            })
            relation_triples = []  # always defined; KG block below never NameErrors
            if species_in_batch:
                relation_triples = extract_relations(
                    text=md_text,                           # full parsed markdown
                    known_species=species_in_batch,
                    source_citation=citation_str,           # fixed: was cite_str (inner scope)
                    file_hash=file_hash,
                    llm_call_fn=lambda p: call_llm(         # fixed: was _call_claude (undefined)
                        p, provider, model_sel, api_key, ollama_url
                    ),
                    meta_db_path=META_DB_PATH,
                )
                log_cb(f"[RE] Extracted {len(relation_triples)} relation triples")
            
            
            
            # ── Step 7: Geocoding cascade ─────────────────────────────────────
            occurrences = geocode_occurrences(occurrences, log_cb)
            if hasattr(log_inst, 'log_geocoded'):
                log_inst.log_geocoded(occurrences)
            
            # After: occurrences = geocode_occurrences(occurrences, log_cb)
            from biotrace_postprocessing import run_postprocessing
            occurrences, pp_summary = run_postprocessing(
                    occurrences, citation_str=citation_str,
                    wiki_root=WIKI_ROOT, geonames_db=GEONAMES_DB,
                    use_nominatim=True, log_cb=log_cb,)
                
            if pp_summary["conflicts"]:
                st.warning(f"{len(pp_summary['conflicts'])} unresolved conflicts — see Schema tab")
            st.session_state["pp_conflicts"]    = pp_summary["conflicts"]
            st.session_state["pp_conflict_log"] = pp_summary["conflict_log"]

            # PATCHED-R2: R3-citation-fix — stamp full citation_str on each record
            # and pass citation_str (not doc_title) as source_title to insert
            for _occ in occurrences:
                if isinstance(_occ, dict):
                    # Overwrite only if the per-record citation looks like a raw filename
                    _rec_cit = str(_occ.get("Source Citation") or _occ.get("sourceCitation","")).strip()
                    _looks_like_filename = (
                        not _rec_cit
                        or _rec_cit == doc_title
                        or _rec_cit.lower().endswith((".pdf",".p65",".md",".txt"))
                        or len(_rec_cit) < 20
                    )
                    if _looks_like_filename and citation_str and len(citation_str) > 20:
                        _occ["Source Citation"] = citation_str
                        _occ["sourceCitation"]  = citation_str

            # ── Step 8: Save to SQLite ────────────────────────────────────────
            n = insert_occurrences(occurrences, file_hash, citation_str, session_id)
            log_cb(f"[DB] {n} records saved (session {session_id})")
            
            if hasattr(log_inst, 'log_saved'):
                log_inst.log_saved(n)
            # Final panel render with all stages populated
            # PATCHED: P3-tracker-purge — remove NER placeholder IDs before render
            if hasattr(log_inst, 'tracker') and hasattr(log_inst.tracker, 'species'):
                _CAND_PURGE_RE = re.compile(r"^__candidate_\d+_\d+$")
                log_inst.tracker.species = {
                    k: v for k, v in log_inst.tracker.species.items()
                    if not _CAND_PURGE_RE.match(str(k))
                }

            with progress_placeholder.container():
                render_species_progress_panel(log_inst.tracker)
            
            log_cb(f"[DB] {n} records saved (session {session_id})")

            # ── Step 9: v5 knowledge systems ─────────────────────────────────
            if any([use_kg, use_mb, use_wiki]):
                ingest_into_v5_systems(
                    occurrences, citation=citation_str,
                    session_id=session_id, log_cb=log_cb,
                    provider=provider, model_sel=model_sel,
                    api_key=api_key, ollama_base_url=ollama_url,
                    update_wiki_narratives=wiki_narr,
                    chunk_text=md_text,
                    use_kg=use_kg, use_mb=use_mb, use_wiki=use_wiki,  # FIX 2c
                )

            
            # [ENHANCEMENT: biotrace_kg_spatio_temporal] — Stage 7 (Hyper-Extract-inspired)
            # Incrementally upserts species nodes (lat/lon bbox, temporal range) and
            # relation edges into kg_nodes / kg_edges SQLite tables (FTS5-queryable).
            # relation_triples always defined above (→ [] when RE found nothing).
            from biotrace_kg_spatio_temporal import BioTraceSpatioTemporalKG
            _stkg = BioTraceSpatioTemporalKG(META_DB_PATH)
            _stkg.upsert_from_occurrences(occurrences)   # fixed: was results (undefined)
            if relation_triples:
                _stkg.upsert_from_relations(relation_triples)
            log_cb(
                f"[SpatioKG] Updated: {len(occurrences)} species nodes, "
                f"{len(relation_triples)} relation edges"
            )
            
            
            # ── Step 10: Save CSV ─────────────────────────────────────────────
            df = pd.DataFrame(occurrences)
            csv_path = os.path.join(CSV_DIR, f"{session_id}.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8")

            st.session_state["schema_errors"]     = _schema_errors
            st.session_state["last_occurrences"]  = occurrences
            st.session_state["last_session_id"]   = session_id

            st.success(f"✅ {len(occurrences)} occurrence records extracted.")

            # ── Results preview ───────────────────────────────────────────────
            show_cols = [
                "recordedName","validName","family_","phylum",
                "verbatimLocality","occurrenceType","decimalLatitude",
                "decimalLongitude","wormsID","matchScore",
            ]
            show_cols = [c for c in show_cols if c in df.columns]
            st.dataframe(df[show_cols], width=True, height=350)


            with st.expander("📋 Extraction Log"):
                st.code("\n".join(log_inst.logs))

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — TNR ENGINE  (v5.2)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    if _ENH_AVAILABLE and _render_tnr_tab:
        _render_tnr_tab()
    else:
        st.subheader("🔬 TNR Engine")
        st.info(
            "Install `biotrace_ner.py` and `biotrace_v5_enhancements.py` "
            "alongside this file to enable the TNR Engine tab."
        )

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — LOCALITY NER  (v5.2)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    if _ENH_AVAILABLE and _render_locality_tab:
        _render_locality_tab()
    else:
        st.subheader("📍 Locality NER")
        st.info(
            "Install `biotrace_locality_ner.py` and `biotrace_v5_enhancements.py` "
            "alongside this file to enable the Locality NER tab."
        )

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — VERIFICATION & COORDINATES (v5.2 / Combined)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("✏️ Verification Table — Human-in-the-Loop Review & Coordinate Editor")
    st.caption(
        "Edit Flag (Primary/Secondary/Uncertain), Validation (Accept/Reject/Review), "
        "Notes, and Coordinates. Changes are saved back to the database immediately."
    )
    
    # [ENHANCEMENT: biotrace_hitl_geocoding] — HITL geocoding tab with full
    # multi-store sync (SQLite + KG + MemoryBank + Wiki). All 4 path args required.
    from biotrace_hitl_geocoding import render_hitl_geocoding_tab
    render_hitl_geocoding_tab(META_DB_PATH, KG_DB_PATH, MB_DB_PATH, WIKI_ROOT)
    
    with st.expander("old editor"):
        
        if not _ENH_AVAILABLE or _render_verification_table is None:
            st.info("Install `biotrace_v5_enhancements.py` to enable this tab.")
        else:
            try:
                import sqlite3 as _sql
                _con = _sql.connect(META_DB_PATH)
                _df_v = pd.read_sql_query(
                    "SELECT * FROM occurrences_v4 ORDER BY id DESC LIMIT 500", _con
                )
                _con.close()
            except Exception:
                _df_v = pd.DataFrame()

            if _df_v.empty:
                st.info("No records yet. Run an extraction first.")
            else:
                _occs_v = _df_v.to_dict("records")
                for _o in _occs_v:
                    _o.setdefault("validName",      _o.get("validname",""))
                    _o.setdefault("recordedName",   _o.get("recordedname",""))
                    _o.setdefault("family_",        _o.get("family_",""))
                    _o.setdefault("class_",         _o.get("class_",""))
                    _o.setdefault("order_",         _o.get("order_",""))
                    _o.setdefault("occurrenceType", _o.get("occurrencetype","Uncertain"))
                    _o.setdefault("wormsID",        _o.get("wormsid",""))
                    _o.setdefault("matchScore",     _o.get("matchscore",0))

                with st.expander("Filters"):
                    _col1, _col2, _col3 = st.columns(3)
                    _fam_opts = sorted(_df_v["family_"].dropna().unique().tolist())
                    _sel_fam  = _col1.multiselect("Family:", _fam_opts, key="verif_fam")
                    _sel_flag = _col2.multiselect(
                        "Flag:", ["Primary","Secondary","Uncertain"], key="verif_flag"
                    )
                    _min_score= _col3.slider(
                        "Min match score:", 0.0, 1.0, 0.0, 0.05, key="verif_score"
                    )

                _filt_occs = [
                    o for o in _occs_v
                    if (not _sel_fam  or o.get("family_","") in _sel_fam)
                    and (not _sel_flag or o.get("occurrenceType","") in _sel_flag)
                    and float(o.get("matchScore",0) or 0) >= _min_score
                ]
                
                _missing = _df_v[_df_v["decimalLatitude"].isna() | _df_v["decimalLongitude"].isna()]
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                col_stat1.metric("Total records shown", len(_filt_occs))
                col_stat2.metric("Total in DB with coords", int((~_df_v["decimalLatitude"].isna()).sum()))
                col_stat3.metric("Total in DB missing coords", len(_missing))

                # if len(_missing) > 0:
                #     with st.expander(f"⚠️ {len(_missing)} records missing coordinates — click to geocode"):
                #         _nom_btn = st.button(
                #             "🌍 Geocode missing via GeoNames + Nominatim",
                #             key="geocode_missing_btn_verif",
                #         )
                #         if _nom_btn and _GEOCODER_AVAILABLE and GeocodingCascade:
                #             with st.spinner("Geocoding…"):
                #                 try:
                #                     geo = GeocodingCascade(
                #                         geonames_db   = GEONAMES_DB if os.path.exists(GEONAMES_DB) else "",
                #                         pincode_txt   = PINCODE_TXT if os.path.exists(PINCODE_TXT) else "",
                #                         use_nominatim = True,
                #                     )
                #                     n_updated = geo.batch_geocode_db(META_DB_PATH)
                #                     st.success(f"✅ Geocoded {n_updated} records.")
                #                     st.rerun()
                #                 except Exception as _ge:
                #                     st.error(f"Geocoding failed: {_ge}")
                
                if len(_missing) > 0:
                    with st.expander(f"⚠️ {len(_missing)} records missing coordinates — click to geocode"):
                        _nom_btn = st.button(
                            "🌍 Geocode missing via GeoNames + Nominatim",
                            key="geocode_missing_btn_verif",
                        )
                        
                        # Check if button clicked AND geocoder is ready
                        if _nom_btn and _GEOCODER_AVAILABLE:
                            with st.spinner("Geocoding..."):
                                try:
                                    geo = GeocodingCascade(
                                        geonames_db=GEONAMES_DB if os.path.exists(GEONAMES_DB) else "",
                                        pincode_txt=PINCODE_TXT if os.path.exists(PINCODE_TXT) else "",
                                        use_nominatim=True,
                                    )
                                    
                                    n_updated = geo.batch_geocode_db(META_DB_PATH)
                                    
                                    if n_updated > 0:
                                        # 1. Show a persistent notification
                                        st.toast(f"✅ Success! Geocoded {n_updated} records.", icon="🌍")
                                        
                                        # 2. IMPORTANT: Clear cache so new data is loaded on rerun
                                        st.cache_data.clear()
                                        
                                        # 3. Slight delay so the user registers the success state
                                        time.sleep(1.2)
                                        
                                        # 4. Refresh to hide this expander and show new data
                                        st.rerun()
                                    else:
                                        st.warning("⚠️ No records were updated. Check connection or address format.")
                                        
                                except Exception as _ge:
                                    st.error(f"Geocoding failed: {_ge}")

                # Create dataframe for the editor which includes both verification and coordinate columns
                _df_for_editor = pd.DataFrame(_filt_occs)
                
                _edited_df = st.data_editor(
                    _df_for_editor,
                    column_config={
                        "id":               st.column_config.NumberColumn("ID", disabled=True),
                        "recordedName":     st.column_config.TextColumn("Species", disabled=True),
                        "verbatimLocality": st.column_config.TextColumn(
                                                "Locality", 
                                                help="Click a cell to edit the text",
                                                required=True,    # Prevents empty submissions
                                                default="",       # Default text for new rows
                                                max_chars=60      # Limit input length
                                            ),
                        "occurrenceType":   st.column_config.SelectboxColumn("Flag", options=["Primary", "Secondary", "Uncertain"]),
                        "validationStatus": st.column_config.SelectboxColumn("Validation", options=["Accept", "Reject", "Review"]),
                        "notes":            st.column_config.TextColumn("Notes"),
                        "decimalLatitude":  st.column_config.NumberColumn(
                            "Latitude", min_value=-90, max_value=90, format="%.5f"
                        ),
                        "decimalLongitude": st.column_config.NumberColumn(
                            "Longitude", min_value=-180, max_value=180, format="%.5f"
                        ),
                        "geocodingSource":  st.column_config.TextColumn("Source"),
                    },
                    disabled=["id","recordedName"],
                    width=True,
                    key="combined_editor",
                    height=min(400, 60 + 35 * len(_filt_occs)),
                )

                # PATCHED-R2: R5-edit-delete — full-field UPDATE + DELETE support
                _col_save, _col_del = st.columns([3,1])

                with _col_save:
                    if st.button("💾 Save All Edits to Database", key="save_combined"):
                        try:
                            _save_con = _sql.connect(META_DB_PATH)
                            _updated_count = 0
                            for _, row in _edited_df.iterrows():
                                _rid = row.get("id")
                                if pd.notna(_rid):
                                    _save_con.execute(
                                        """UPDATE occurrences_v4 SET
                                            recordedName=?, validName=?,
                                            verbatimLocality=?, occurrenceType=?,
                                            validationStatus=?, notes=?,
                                            habitat=?, sourceCitation=?,
                                            phylum=?, class_=?, order_=?, family_=?,
                                            wormsID=?, taxonRank=?,
                                            decimalLatitude=?, decimalLongitude=?,
                                            geocodingSource=?
                                        WHERE id=?""",
                                        (
                                            str(row.get("recordedName",""))[:300],
                                            str(row.get("validName",""))[:300],
                                            str(row.get("verbatimLocality",""))[:300],
                                            str(row.get("occurrenceType","Uncertain"))[:50],
                                            str(row.get("validationStatus","Review"))[:50],
                                            str(row.get("notes",""))[:1000],
                                            str(row.get("habitat",""))[:300],
                                            str(row.get("sourceCitation",""))[:500],
                                            str(row.get("phylum",""))[:100],
                                            str(row.get("class_",""))[:100],
                                            str(row.get("order_",""))[:100],
                                            str(row.get("family_",""))[:100],
                                            str(row.get("wormsID",""))[:20],
                                            str(row.get("taxonRank",""))[:50],
                                            _to_float(row.get("decimalLatitude")),
                                            _to_float(row.get("decimalLongitude")),
                                            str(row.get("geocodingSource","manual"))[:100],
                                            int(_rid),
                                        ),
                                    )
                                    _updated_count += 1
                            _save_con.commit()
                            _save_con.close()
                            st.success(f"✅ Saved {_updated_count} edits to database.")
                            st.rerun()
                        except Exception as _e:
                            st.error(f"Save failed: {_e}")

                with _col_del:
                    # Row-level delete: select by ID
                    _del_ids_str = st.text_input(
                        "Delete record IDs (comma-separated):",
                        placeholder="e.g. 12, 47, 93",
                        key="delete_ids_input",
                    )
                    if st.button("🗑️ Delete Selected", key="delete_selected_btn",
                                 type="secondary"):
                        if _del_ids_str.strip():
                            try:
                                _del_ids = [
                                    int(x.strip())
                                    for x in _del_ids_str.split(",")
                                    if x.strip().isdigit()
                                ]
                                if _del_ids:
                                    _del_con = _sql.connect(META_DB_PATH)
                                    _del_con.executemany(
                                        "DELETE FROM occurrences_v4 WHERE id=?",
                                        [(i,) for i in _del_ids],
                                    )
                                    _del_con.commit()
                                    _del_con.close()
                                    st.success(
                                        f"🗑️ Deleted {len(_del_ids)} records: "
                                        f"{_del_ids}"
                                    )
                                    st.rerun()
                            except Exception as _de:
                                st.error(f"Delete failed: {_de}")
                        else:
                            st.warning("Enter at least one record ID to delete.")
                        
                _with_coords = _edited_df.dropna(subset=["decimalLatitude","decimalLongitude"])
                if not _with_coords.empty:
                    st.map(_with_coords.rename(columns={
                        "decimalLatitude":"lat","decimalLongitude":"lon"
                    })[["lat","lon"]])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("🕸️ Species Knowledge Graph (GraphRAG)")

    # kg = get_knowledge_graph()
    # if not kg:
    #     st.warning("Install `biotrace_knowledge_graph.py` alongside this file.")
    kg = get_knowledge_graph()
    if not kg:
        st.error(f"⚠️ Knowledge Graph unavailable")
        with st.expander("Show error details"):
            st.code(kg.error, language="text")
            st.markdown(
                "**Common fixes:**\n"
                "- `pip install networkx` (required by knowledge graph)\n"
                "- `pip install pyvis` (optional, needed for graph viz)\n"
                "- Confirm `biotrace_knowledge_graph.py` is in the same folder as this file"
            )
    else:
        stats = kg.stats()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nodes", stats["total_nodes"])
        col2.metric("Edges", stats["total_edges"])
        col3.metric("Species", stats["node_types"].get("Species",0))
        col4.metric("Localities", stats["node_types"].get("Locality",0))

        with st.expander("Node & Edge Breakdown"):
            col_l, col_r = st.columns(2)
            with col_l:
                st.write("**Node types:**")
                for k,v in stats["node_types"].items():
                    st.write(f"  {k}: {v}")
            with col_r:
                st.write("**Edge types:**")
                for k,v in stats["edge_types"].items():
                    st.write(f"  {k}: {v}")

        st.divider()

        col_viz, col_comm = st.columns([2,1])
        with col_viz:
            if st.button("🗺️ Generate Interactive Graph (PyVis)"):
                with st.spinner("Rendering…"):
                    out = kg.export_pyvis_html("kg_viz.html", max_nodes=150)
                    if out and os.path.exists(out):
                        html_content = Path(out).read_text(encoding="utf-8")
                        st.components.v1.html(html_content, height=600, scrolling=True, )
                        # st.components.v1.html(html, width=None, height=None, scrolling=False, *, tab_index=None)


                    else:
                        st.warning("PyVis HTML generation failed.")

        with col_comm:
            if st.button("🔍 Detect Communities"):
                with st.spinner("Running community detection…"):
                    comms = kg.detect_communities()
                    for cid, members in list(comms.items())[:5]:
                        with st.expander(f"Community {cid} ({len(members)} nodes)"):
                            st.write(", ".join(members[:12]))

        st.divider()

        st.subheader("Neighbourhood Query")
        q_loc = st.text_input("Species at locality:", placeholder="e.g. Gulf of Mannar")
        if q_loc:
            sps = kg.get_species_at_locality(q_loc)
            if sps:
                st.success(f"**{len(sps)} species** found at '{q_loc}':")
                st.write(", ".join(sps))
            else:
                st.info("No species found for that locality.")

        q_sp = st.text_input("Co-occurring species with:", placeholder="e.g. Acanthurus triostegus")
        if q_sp:
            cos = kg.get_co_occurring_species(q_sp)
            if cos:
                st.success(f"**{len(cos)} co-occurring species:**")
                st.write(", ".join(cos[:20]))
            else:
                st.info("No co-occurrences found.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — MEMORY BANK
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("🧠 Persistent Memory Bank")

    # mb = get_memory_bank()
    # if not mb:
    #     st.warning("Install `biotrace_memory_bank.py` alongside this file.")
    mb = get_memory_bank()
    
    if not mb:
        st.error(f"⚠️ Memory Bank unavailable")
        with st.expander("Show error details"):
            st.code(mb.error, language="text")
            st.markdown(
                "**Common fixes:**\n"
                "- Confirm `biotrace_memory_bank.py` is in the same folder as this file\n"
                "- Check for any missing dependency reported above"
            )
    else:
        ms = mb.stats()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Memory Atoms", ms["total_atoms"])
        col2.metric("Unique Species", ms["unique_species"])
        col3.metric("Localities", ms["unique_localities"])
        col4.metric("Sessions", ms["total_sessions"])

        if ms.get("top_species_by_confirmation"):
            st.write("**Most confirmed species:**")
            top_df = pd.DataFrame(
                ms["top_species_by_confirmation"],
                columns=["Species","Confirmations"],
            )
            st.dataframe(top_df, width=True, hide_index=True)

        st.divider()

        st.subheader("Semantic Recall")
        recall_q = st.text_input("Query:", placeholder="coral reef species Acanthurus")
        col_flt1, col_flt2, col_flt3 = st.columns(3)
        flt_loc = col_flt1.text_input("Filter locality:")
        flt_fam = col_flt2.text_input("Filter family:")
        flt_hab = col_flt3.text_input("Filter habitat:")

        if recall_q:
            atoms = mb.recall(
                recall_q, top_k=15,
                filter_locality=flt_loc or None,
                filter_family=flt_fam or None,
                filter_habitat=flt_hab or None,
            )
            if atoms:
                st.success(f"{len(atoms)} relevant records recalled.")
                st.dataframe(pd.DataFrame(atoms)[[
                    "valid_name","family_","phylum","locality",
                    "habitat","occurrence_type","confidence","times_confirmed","source_citation",
                ]], width=True)
            else:
                st.info("No matching records found.")

        st.divider()
        if st.button("📥 Export Darwin Core CSV"):
            out_path = os.path.join(DATA_DIR, "memory_bank_export.csv")
            n = mb.export_darwin_core_csv(out_path)
            st.success(f"Exported {n} records → {out_path}")
            st.download_button(
                "⬇️ Download",
                data=open(out_path, "rb").read(),
                file_name="memory_bank_export.csv",
                mime="text/csv",
            )
#29042026

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 7 — UNIFIED WIKI  (v5.5)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    # Inject wiki CSS once per session
    inject_css_streamlit()

    wiki = get_wiki()
    if not wiki:
        st.warning(
            "📖 Wiki unavailable — place **biotrace_wiki_unified.py** and "
            "**biotrace_wiki.css** alongside this file and restart."
        )
    else:
        # ── Build the LLM callable (used by manual enhance + auto-narrate) ──
        def _wiki_llm_fn(prompt: str) -> str:
            return call_llm(
                prompt, provider, model_sel, api_key, ollama_url
            )

        # ── Delegate all UI to BioTraceWikiUnified.render_streamlit_tab() ──
        wiki.render_streamlit_tab(
            provider      = provider,
            model_sel     = model_sel,
            api_key       = api_key,
            ollama_url    = ollama_url,
            meta_db       = META_DB_PATH,
            call_llm_fn   = _wiki_llm_fn,
        )

        # ── Optional: Ollama Wiki Architect Agent panel ──────────────────
        agent = get_wiki_agent()
        if agent and _WIKI_AGENT_AVAILABLE:
            st.divider()
            with st.expander("🤖 Ollama Wiki Architect Agent (optional / agentic)", expanded=False):
                agent.render_agent_panel()


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 8 — GRAPHRAG QUERY
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.subheader("🔍 GraphRAG Natural Language Query")
    st.caption(
        "Questions are answered using the Knowledge Graph + Memory Bank + Wiki "
        "as multi-hop context, then sent to your configured LLM."
    )

    gq = st.text_area(
        "Ask a question:",
        placeholder=(
            "e.g. Which species were found in mangrove habitats?\n"
            "e.g. What families co-occur at rocky intertidal zones?\n"
            "e.g. List species from Gulf of Mannar with WoRMS IDs"
        ),
        height=100,
    )

    context_sources = st.multiselect(
        "Context sources:",
        ["Knowledge Graph", "Memory Bank", "Wiki"],
        default=["Knowledge Graph", "Memory Bank", "Wiki"],
    )

    if st.button("🔎 Run GraphRAG Query", type="primary") and gq.strip():
        with st.spinner("Building context + querying LLM…"):
            context_parts = []

            if "Knowledge Graph" in context_sources:
                kg = get_knowledge_graph()
                if kg:
                    context_parts.append(kg.build_rag_context(gq, top_k=8))

            if "Memory Bank" in context_sources:
                mb = get_memory_bank()
                if mb:
                    context_parts.append(mb.build_memory_context(gq, top_k=10))

            if "Wiki" in context_sources:
                wiki = get_wiki()
                if wiki:
                    context_parts.append(wiki.build_wiki_context(gq, top_k=5))

            full_context = "\n\n".join(filter(None, context_parts))

            if not full_context.strip():
                st.warning("No context built — extract some papers first.")
            else:
                full_prompt = (
                    "You are a marine biodiversity expert. Use the structured context "
                    "below (from a knowledge graph, memory bank, and wiki) to answer "
                    "the question accurately. Cite species with scientific names.\n\n"
                    f"{full_context}\n\n"
                    f"QUESTION: {gq}\n\n"
                    "Answer with taxonomic precision, citing WoRMS IDs where available."
                )
                response = call_llm(
                    full_prompt, provider, model_sel, api_key, ollama_url
                )

                st.markdown("### Answer")
                st.markdown(response)

                with st.expander("📋 Full Context Injected"):
                    st.code(full_context, language="text")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 9 — DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.subheader("📊 Occurrence Database")

    df_all = db_load_all()
    if df_all.empty:
        st.info("No records yet. Run an extraction first.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df_all))
        col2.metric("Unique Species", df_all["validName"].nunique())
        col3.metric("Localities", df_all["verbatimLocality"].nunique())

        with st.expander("Filters"):
            fam_opts = sorted(df_all["family_"].dropna().unique())
            sel_fam  = st.multiselect("Family:", fam_opts)
            occ_type = st.multiselect("Occurrence type:", ["Primary","Secondary","Uncertain"])
            if sel_fam:
                df_all = df_all[df_all["family_"].isin(sel_fam)]
            if occ_type:
                df_all = df_all[df_all["occurrenceType"].isin(occ_type)]

        cols_show = [
            "validName","family_","phylum","verbatimLocality",
            "habitat","occurrenceType","wormsID","taxonomicStatus","matchScore",
            "decimalLatitude","decimalLongitude","sourceCitation",
        ]
        cols_show = [c for c in cols_show if c in df_all.columns]
        st.dataframe(df_all[cols_show], width=True, height=450)

        with_coords = df_all.dropna(subset=["decimalLatitude","decimalLongitude"])
        if not with_coords.empty:
            st.subheader("🗺️ Occurrence Map")
            map_df = with_coords.rename(columns={
                "decimalLatitude":"lat","decimalLongitude":"lon"
            })[["lat","lon","validName"]].copy()
            st.map(map_df)



# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 10 — SCHEMA DIAGNOSTICS  (v5.2)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[9]:
    from biotrace_postprocessing import render_conflict_panel
    render_conflict_panel(
        st.session_state.get("pp_conflicts", []),
        st.session_state.get("pp_conflict_log", []),
    )
    # else:
    #     st.subheader("🛡️ Schema Diagnostics")
    #     st.info("Install `biotrace_schema.py` and `biotrace_v5_enhancements.py`.")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 11 — EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[10]:
    st.subheader("📥 Export")

    df_all = db_load_all()
    if df_all.empty:
        st.info("No data to export.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Darwin Core CSV**")
            csv_bytes = df_all.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download CSV",
                data=csv_bytes,
                file_name="biotrace_v5_occurrences.csv",
                mime="text/csv",
            )

        with col2:
            st.write("**JSON (occurrence list)**")
            json_bytes = df_all.to_json(orient="records", indent=2).encode()
            st.download_button(
                "⬇️ Download JSON",
                data=json_bytes,
                file_name="biotrace_v5_occurrences.json",
                mime="application/json",
            )

        st.divider()
        kg = get_knowledge_graph()
        if kg:
            st.write("**Knowledge Graph Export**")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Export GraphML (Gephi/Cytoscape)"):
                    gml_path = os.path.join(DATA_DIR, "kg_export.graphml")
                    kg.export_graphml(gml_path)
                    st.download_button(
                        "⬇️ Download GraphML",
                        data=open(gml_path,"rb").read(),
                        file_name="biotrace_kg.graphml",
                        mime="application/xml",
                    )
            with col_b:
                if st.button("Export PyVis HTML"):
                    html_path = os.path.join(DATA_DIR, "kg_visualization.html")
                    kg.export_pyvis_html(html_path, max_nodes=200)
                    st.download_button(
                        "⬇️ Download HTML",
                        data=open(html_path,"rb").read(),
                        file_name="biotrace_kg.html",
                        mime="text/html",
                    )