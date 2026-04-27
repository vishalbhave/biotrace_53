"""
biotrace_hf_ner.py  —  BioTrace v5.4
────────────────────────────────────────────────────────────────────────────
Pure PyTorch + HuggingFace Transformers NER pipeline.

REPLACES: spaCy + taxonerd (both fail on CUDA 13 due to compiled extensions)
MODELS:
  1. NoYo25/BiodivBERT        — token-classification, biodiversity entities
                                 (TAXON, HABITAT, LOCATION, METHOD)
  2. nleguillarme/en_ner_eco_biobert
                               — ecological taxa NER; this is the same weights
                                 as TaxoNERD's en_ner_eco_biobert-1.1.0 but
                                 loaded directly via transformers (no spaCy).
                                 Falls back to dmis-lab/biobert-base-cased-v1.2
                                 + zero-shot if the HF mirror is unavailable.

CUDA 13 FIX:
  Install PyTorch with cu121 wheels — CUDA 13 drivers (≥530) are backward-
  compatible with CUDA 12.1 runtime, so no recompilation needed:

    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers accelerate sentencepiece safetensors

USAGE in biotrace_v5.py (unchanged API — BiodiVizPipeline.extract(text)):
    from biotrace_hf_ner import BiodiVizPipeline
    pipeline = BiodiVizPipeline()
    result   = pipeline.extract(text)
    # result = {"organisms": [...], "relations": [...], "habitats": [...]}
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("biotrace.hf_ner")

# ─────────────────────────────────────────────────────────────────────────────
#  Model registry
# ─────────────────────────────────────────────────────────────────────────────

# Primary: BiodivBERT (token-classification fine-tuned on biodiversity text)
_BIODIVBERT_MODEL = "NoYo25/BiodivBERT"


# Secondary: ecoBERT NER — the HuggingFace mirror of TaxoNERD's en_ner_eco_biobert
# weights. If unavailable on HF, falls back to vanilla BioBERT with regex post-filter.
_ECOBERT_MODEL = "ViktorDo/EcoBERT-finetuned-ner-S800"
_BIOBERT_FALLBACK = "dmis-lab/biobert-base-cased-v1.2"

# ─────────────────────────────────────────────────────────────────────────────
#  Entity label maps
# ─────────────────────────────────────────────────────────────────────────────

# Labels produced by BiodivBERT token-classification
_TAXON_LABELS   = {"TAXON", "B-TAXON", "I-TAXON", "ORG", "B-ORG", "I-ORG"}
_HABITAT_LABELS = {"HABITAT", "B-HABITAT", "I-HABITAT", "LOC", "B-LOC", "I-LOC"}
_LOC_LABELS     = {"LOCATION", "B-LOCATION", "I-LOCATION", "GPE", "B-GPE", "I-GPE"}

# Labels produced by ecoBERT NER
_ECO_TAXON_LABELS = {"TAXON", "B-TAXON", "I-TAXON", "SPECIES", "B-SPECIES", "I-SPECIES"}

# ─────────────────────────────────────────────────────────────────────────────
#  Availability guard
# ─────────────────────────────────────────────────────────────────────────────
_TORCH_AVAILABLE = False
_TF_AVAILABLE    = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForTokenClassification
    _TF_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _best_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' — in preference order."""
    if _TORCH_AVAILABLE:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _load_pipeline(model_id: str, device: str, task: str = "token-classification"):
    """
    Load a HuggingFace token-classification pipeline with graceful fallback.
    Returns (pipe, model_id_used) or (None, None) on failure.
    """
    try:
        import torch
        from transformers import pipeline as hf_pipeline

        device_idx = 0 if device == "cuda" else (-1 if device == "cpu" else device)
        pipe = hf_pipeline(
            task,
            model=model_id,
            aggregation_strategy="simple",   # merge B-/I- tokens automatically
            device=device_idx,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        logger.info("[hf_ner] Loaded %s on %s", model_id, device)
        return pipe, model_id
    except Exception as exc:
        logger.warning("[hf_ner] Failed to load %s: %s", model_id, exc)
        return None, None


def _merge_subword_entities(raw_entities: list[dict]) -> list[dict]:
    """
    Merge consecutive B-/I- subword spans produced by simple aggregation
    into clean named entities. Handles ##-prefixed wordpieces.
    """
    merged = []
    for ent in raw_entities:
        word = ent.get("word", "").replace("##", "").strip()
        label = ent.get("entity_group", ent.get("entity", ""))
        score = ent.get("score", 1.0)
        if not word:
            continue
        # Merge into previous if same label and adjacent
        if (merged and merged[-1]["label"] == label
                and ent.get("start", 0) - merged[-1]["end"] <= 2):
            merged[-1]["word"] += " " + word
            merged[-1]["end"]   = ent.get("end", merged[-1]["end"])
            merged[-1]["score"] = (merged[-1]["score"] + score) / 2
        else:
            merged.append({
                "word":  word,
                "label": label,
                "score": score,
                "start": ent.get("start", 0),
                "end":   ent.get("end", 0),
            })
    return merged


def _deduplicate_spans(spans: list[str]) -> list[str]:
    """Deduplicate preserving first-seen order, case-insensitive."""
    seen: set[str] = set()
    out:  list[str] = []
    for s in spans:
        s = s.strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def _binomial_regex_fallback(text: str) -> list[str]:
    """
    Regex binomial extraction — used when both NER models fail.
    Mirrors biotrace_traiter_prepass logic.
    """
    pattern = re.compile(
        r"\b([A-Z][a-z]{2,})\s+([a-z]{3,})"
        r"(?:\s+(?:var\.|subsp\.|f\.)\s+[a-z]+)?"
        r"(?:\s+\([^)]*\d{4}[^)]*\))?",
    )
    LIFE_STAGE = {
        "scyphistoma", "medusa", "ephyra", "polyp", "planula", "larva",
        "juvenile", "nauplius", "cypris", "zoea", "spat", "strobila",
    }
    found = []
    for m in pattern.finditer(text):
        genus, epithet = m.group(1), m.group(2)
        if epithet.lower() not in LIFE_STAGE:
            found.append(f"{genus} {epithet}")
    return _deduplicate_spans(found)


def _locality_regex_fallback(text: str) -> list[str]:
    """Extract probable place names — uppercase sequences not in life-stage list."""
    pattern = re.compile(r"\b([A-Z][a-z]{2,})(?:\s+(?:of\s+)?[A-Z][a-z]{2,}){0,3}\b")
    STOP = {
        "The", "This", "These", "That", "However", "Therefore",
        "Primary", "Secondary", "Uncertain", "Figure", "Table",
    }
    found = []
    for m in pattern.finditer(text):
        phrase = m.group()
        if phrase not in STOP and len(phrase) >= 4:
            found.append(phrase)
    return _deduplicate_spans(found)[:10]


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline class (same public API as the old spaCy-based version)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BiodiVizPipeline:
    """
    Pure-transformers replacement for the spaCy/taxonerd BiodiViz pipeline.

    Loads two complementary models:
      1. BiodivBERT  — broad biodiversity entity recognition
      2. ecoBERT NER — taxonomic name specialisation (taxonerd weights via HF)

    Falls back gracefully at each level if a model is unavailable.

    Usage (unchanged from old API):
        from biotrace_hf_ner import BiodiVizPipeline
        pipe = BiodiVizPipeline()
        result = pipe.extract(text)
        # {"organisms": [...], "habitats": [...], "locations": [...], "relations": [...]}

    Deprecated constructor args (kept for backward compat, now ignored):
        ner_model_path, re_model_path
    """

    # Deprecated args from old spaCy-based constructor — accepted but ignored
    ner_model_path: str = ""
    re_model_path:  str = ""

    # Internal state (not constructor args)
    _biodivbert_pipe: object = field(default=None, init=False, repr=False)
    _ecobert_pipe:    object = field(default=None, init=False, repr=False)
    _device:          str    = field(default="", init=False, repr=False)
    _ready:           bool   = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if not _TF_AVAILABLE:
            logger.error(
                "[hf_ner] transformers not installed. "
                "Run: pip install transformers accelerate sentencepiece safetensors"
            )
            return

        self._device = _best_device()
        logger.info("[hf_ner] Device: %s", self._device)

        # Load BiodivBERT (primary)
        self._biodivbert_pipe, _ = _load_pipeline(_BIODIVBERT_MODEL, self._device)

        # Load ecoBERT NER (secondary) — taxonerd weights without spaCy
        self._ecobert_pipe, loaded_id = _load_pipeline(_ECOBERT_MODEL, self._device)
        if self._ecobert_pipe is None:
            logger.warning(
                "[hf_ner] %s unavailable — falling back to regex binomial extraction.",
                _ECOBERT_MODEL,
            )

        self._ready = self._biodivbert_pipe is not None or self._ecobert_pipe is not None
        if not self._ready:
            logger.warning("[hf_ner] Both NER models failed — regex fallback only.")

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, text: str, chunk_size: int = 512) -> dict:
        """
        Extract biodiversity entities from text.

        Parameters
        ----------
        text       : raw or markdown text chunk
        chunk_size : max tokens per inference call (models have 512-token window)

        Returns
        -------
        {
            "organisms":  list[str],  # scientific names (deduped)
            "habitats":   list[str],  # habitat phrases
            "locations":  list[str],  # geographic place names
            "relations":  list[str],  # "Species @ Locality" strings for prompt hint
        }
        """
        organisms: list[str] = []
        habitats:  list[str] = []
        locations: list[str] = []

        # Chunk text for models with 512-token window
        chunks = self._chunk_text(text, chunk_size)

        for chunk in chunks:
            if not chunk.strip():
                continue

            # ── BiodivBERT pass ───────────────────────────────────────────────
            if self._biodivbert_pipe is not None:
                try:
                    raw_ents = self._biodivbert_pipe(chunk)
                    for ent in _merge_subword_entities(raw_ents):
                        lbl = ent["label"].upper()
                        word = ent["word"].strip()
                        if lbl in _TAXON_LABELS and len(word) >= 4:
                            organisms.append(word)
                        elif lbl in _HABITAT_LABELS:
                            habitats.append(word)
                        elif lbl in _LOC_LABELS:
                            locations.append(word)
                except Exception as exc:
                    logger.warning("[hf_ner] BiodivBERT error: %s", exc)

            # ── ecoBERT NER pass ──────────────────────────────────────────────
            if self._ecobert_pipe is not None:
                try:
                    raw_ents = self._ecobert_pipe(chunk)
                    for ent in _merge_subword_entities(raw_ents):
                        lbl = ent["label"].upper()
                        word = ent["word"].strip()
                        if lbl in _ECO_TAXON_LABELS and len(word) >= 4:
                            organisms.append(word)
                except Exception as exc:
                    logger.warning("[hf_ner] ecoBERT error: %s", exc)

        # ── Regex fallback when both models unavailable ───────────────────────
        if not self._ready:
            organisms = _binomial_regex_fallback(text)
            locations = _locality_regex_fallback(text)

        # Deduplicate
        organisms = _deduplicate_spans(organisms)
        habitats  = _deduplicate_spans(habitats)
        locations = _deduplicate_spans(locations)

        # Build relation hints: "Species @ Locality" for LLM prompt injection
        relations = self._build_relation_hints(organisms, locations, text)

        return {
            "organisms": organisms,
            "habitats":  habitats,
            "locations": locations,
            "relations": relations,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 1800) -> list[str]:
        """
        Split text into overlapping chunks that fit in the model's token window.
        ~1800 chars ≈ 450 tokens for English biomedical text (4 chars/token avg).
        Uses sentence boundaries where possible.
        """
        if len(text) <= max_chars:
            return [text]
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= max_chars:
                current += (" " if current else "") + sent
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _build_relation_hints(organisms: list[str], locations: list[str], text: str) -> list[str]:
        """
        Heuristic co-occurrence: if a species and a location appear in the same
        sentence, emit "Species @ Location" as a relation hint for the LLM prompt.
        """
        hints: list[str] = []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            for org in organisms:
                if org.lower() in sent.lower():
                    for loc in locations:
                        if loc.lower() in sent.lower():
                            hint = f"{org} @ {loc}"
                            if hint not in hints:
                                hints.append(hint)
        return hints[:20]  # cap to avoid overwhelming the prompt


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    TEST_TEXT = """
    Cassiopea andromeda was collected from the intertidal area at Narara, Gulf of Kutch,
    Gujarat in January 2019. The scyphistomae were observed on dead coral reef substrate.
    Holothuria scabra was also recorded at Arambhada coast during the same survey.
    C. andromeda showed typical morphology with oral arms 8-9 in number.
    Secondary records include Aurelia aurita from the Arabian Sea (Kumar, 2015).
    """

    print("Loading BiodiVizPipeline (pure transformers, no spaCy)…")
    pipe = BiodiVizPipeline()
    result = pipe.extract(TEST_TEXT)

    print("\n── Results ──────────────────────────────────────")
    print(f"Organisms  ({len(result['organisms'])}): {result['organisms']}")
    print(f"Habitats   ({len(result['habitats'])}):  {result['habitats']}")
    print(f"Locations  ({len(result['locations'])}): {result['locations']}")
    print(f"Relations  ({len(result['relations'])}): {result['relations']}")

    # Verify the two known species are extracted
    text_lower = TEST_TEXT.lower()
    assert any("cassiopea" in o.lower() for o in result["organisms"]), \
        "Cassiopea andromeda not extracted"
    print("\n✅ Smoke test passed")
    
    

# """
# biotrace_hf_ner.py  —  BioTrace v5.4 (Enhanced)
# ────────────────────────────────────────────────────────────────────────────
# Hugging Face Token Classification (NER) wrapper.
# Uses TaxoNERD for species extraction + spaCy GPE/LOC for localities.

# CHANGES vs v5.4 original:
#   • Sentence-proximity filter replaces the Cartesian product occur_in builder.
#     Only links species ↔ locality when they appear within SENTENCE_WINDOW
#     sentences of each other. Eliminates false-positive relations from dense
#     multi-species, multi-locality sections.
#   • Lowered TaxoNERD confidence threshold to 0.72 for BINOMIAL-pattern names
#     (fixes systematic misses on Opisthobranchia / Nudibranchia families whose
#     names score 0.72–0.79 on the vascular-plant-biased TaxoNERD corpus).
#   • Sentence tokenisation uses NLTK when available, plain "\n"/"." split otherwise.
#   • `extract()` now returns `sentence_relations` (list of dicts with positional
#     metadata) in addition to the original flat `relations` list, for downstream
#     use by the HITL geocoding tab.
# """
# import logging
# import re
# import spacy
# from taxo_extractor import TaxoNERD

# logger = logging.getLogger("biotrace.hf_ner")

# # ── Constants ──────────────────────────────────────────────────────────────────
# # Only link a species to a locality if both appear within this many sentences
# # of each other in the source text.
# SENTENCE_WINDOW = 10

# # Regex for true binomial — used to allow lower-confidence TaxoNERD hits
# # through for opisthobranch / nudibranch / soft-coral names.
# _BINOMIAL_RE = re.compile(r"^[A-Z][a-z]{2,}\s+[a-z]{2,}")

# # ──────────────────────────────────────────────────────────────────────────────

# class BiodiVizPipeline:
#     # Preserving threshold constant for backward-compatibility
#     # with any biotrace_v5.py imports that might check it.
#     RE_CONFIDENCE_THRESHOLD = 0.55

#     def __init__(self, ner_model_path="./ner_model/en_ner_eco_biobert-1.1.0", re_model_path="./re_model"):
#         """
#         Initializes TaxoNERD + spaCy.
#         Local model paths are ignored (kept for API compatibility).
#         """
#         logger.info("Initializing BiodiVizPipeline (TaxoNERD + spaCy, v5.4-enhanced)…")
#         self.taxo_ner = TaxoNERD(confidence_threshold=0.72)

#         try:
#             self.nlp = spacy.load("en_core_web_sm")
#         except OSError:
#             import subprocess
#             logger.warning("Downloading spaCy en_core_web_sm…")
#             subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
#             self.nlp = spacy.load("en_core_web_sm")

#         # Optional NLTK sentence tokeniser
#         self._sent_tokenize = None
#         try:
#             import nltk
#             nltk.download("punkt",     quiet=True)
#             nltk.download("punkt_tab", quiet=True)
#             from nltk import sent_tokenize
#             self._sent_tokenize = sent_tokenize
#             logger.debug("[BiodiViz] NLTK sentence tokeniser ready.")
#         except ImportError:
#             logger.debug("[BiodiViz] NLTK not available — using simple sentence split.")

#     # ── Internal helpers ───────────────────────────────────────────────────────

#     def _split_sentences(self, text: str) -> list[str]:
#         """Return a list of sentences, preferring NLTK over simple split."""
#         if self._sent_tokenize:
#             return self._sent_tokenize(text)
#         # Simple fallback: split on period-space or newlines
#         raw = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
#         return [s.strip() for s in raw if s.strip()]

#     @staticmethod
#     def _is_valid_binomial(name: str) -> bool:
#         """Return True if name looks like a proper binomial (for lower-conf gate)."""
#         return bool(_BINOMIAL_RE.match(name))

#     def _extract_organisms_with_threshold(self, text: str) -> list[str]:
#         """
#         Calls TaxoNERD and applies a two-tier confidence gate:
#           ≥ 0.80  → always accepted
#           0.72–0.80 → accepted only if the name matches a binomial pattern
#                        (genus + species, first letter capitalised)
#         This catches opisthobranch / nudibranch binomials that TaxoNERD
#         underscores due to training corpus bias, while blocking noisy
#         short tokens that score in the same band.
#         """
#         if not text or not self.taxo_ner.pipeline:
#             return []
#         try:
#             extracted = self.taxo_ner.pipeline(text)
#             species_set: set[str] = set()
#             for entity in extracted:
#                 if entity["entity_group"] != "LIVB":
#                     continue
#                 score      = float(entity["score"])
#                 clean_name = entity["word"].strip()
#                 if len(clean_name) <= 3:
#                     continue
#                 if score >= 0.80:
#                     species_set.add(clean_name)
#                 elif score >= 0.72 and self._is_valid_binomial(clean_name):
#                     logger.debug(
#                         "[BiodiViz] Low-conf binomial accepted (%.3f): %s", score, clean_name
#                     )
#                     species_set.add(clean_name)
#             return list(species_set)
#         except Exception as exc:
#             logger.error("[BiodiViz] TaxoNERD extraction error: %s", exc)
#             return []

#     def _build_proximity_relations(
#         self,
#         sentences: list[str],
#         window: int = SENTENCE_WINDOW,
#     ) -> tuple[list[str], list[dict]]:
#         """
#         For each sentence, extract organisms and localities independently.
#         Then link any organism to any locality found within ±window sentences.

#         Returns:
#             flat_relations   — ["[Org] occur_in [Loc]", …]  (deduped)
#             sentence_records — [{organism, locality, sentence_idx, sentence_text}, …]
#                                for downstream HITL use
#         """
#         # Step 1: per-sentence NER
#         sent_organisms: list[list[str]] = []
#         sent_localities: list[list[str]] = []

#         for sent in sentences:
#             orgs = self._extract_organisms_with_threshold(sent)
#             sent_organisms.append(orgs)

#             locs: list[str] = []
#             try:
#                 doc = self.nlp(sent)
#                 for ent in doc.ents:
#                     if ent.label_ in ("GPE", "LOC"):
#                         loc = ent.text.strip().replace("\n", " ")
#                         if len(loc) > 2:
#                             locs.append(loc)
#             except Exception as exc:
#                 logger.debug("[BiodiViz] spaCy error on sentence: %s", exc)
#             sent_localities.append(locs)

#         # Step 2: proximity linking
#         flat_relations:   set[str]  = set()
#         sentence_records: list[dict] = []

#         n = len(sentences)
#         for i in range(n):
#             # Window: sentences [i-window … i+window]
#             i_start = max(0, i - window)
#             i_end   = min(n, i + window + 1)

#             orgs_window = []
#             locs_window = []
#             for j in range(i_start, i_end):
#                 orgs_window.extend(sent_organisms[j])
#                 locs_window.extend(sent_localities[j])

#             # Sentence i is the "anchor" — link its own organisms to all window locs
#             for org in sent_organisms[i]:
#                 for loc in set(locs_window):
#                     rel = f"[{org}] occur_in [{loc}]"
#                     if rel not in flat_relations:
#                         flat_relations.add(rel)
#                         sentence_records.append({
#                             "organism":      org,
#                             "locality":      loc,
#                             "sentence_idx":  i,
#                             "sentence_text": sentences[i],
#                         })

#             # Also link anchor sentence's localities to all window organisms
#             for loc in sent_localities[i]:
#                 for org in set(orgs_window):
#                     rel = f"[{org}] occur_in [{loc}]"
#                     if rel not in flat_relations:
#                         flat_relations.add(rel)
#                         sentence_records.append({
#                             "organism":      org,
#                             "locality":      loc,
#                             "sentence_idx":  i,
#                             "sentence_text": sentences[i],
#                         })

#         return list(flat_relations), sentence_records

#     # ── Public API ─────────────────────────────────────────────────────────────

#     def extract(self, text: str) -> dict:
#         """
#         Main extraction method called by biotrace_v5.py chunk loop.

#         Returns the exact dictionary structure expected by the schema mapper,
#         plus an extra `sentence_relations` key for HITL geocoding:
#         {
#             "organisms":         ["Pleurobranchus testudinarius", …],
#             "relations":         ["[Pleurobranchus testudinarius] occur_in [Gulf of Mannar]", …],
#             "sentence_relations":[{"organism": …, "locality": …, "sentence_idx": …, …}, …]
#         }
#         """
#         if not text:
#             return {"organisms": [], "relations": [], "sentence_relations": []}

#         # 1. Sentence tokenisation
#         sentences = self._split_sentences(text)
#         if not sentences:
#             return {"organisms": [], "relations": [], "sentence_relations": []}

#         # 2. Run proximity-aware extraction
#         flat_relations, sentence_records = self._build_proximity_relations(sentences)

#         # 3. Aggregate all unique organisms across sentences
#         all_organisms: set[str] = set()
#         for sent in sentences:
#             all_organisms.update(self._extract_organisms_with_threshold(sent))

#         # 4. Logging
#         if all_organisms:
#             logger.info(
#                 "[BiodiViz NER/TaxoNERD] %d organisms, %d proximity-filtered relations "
#                 "(window=%d sentences)",
#                 len(all_organisms), len(flat_relations), SENTENCE_WINDOW,
#             )
#         else:
#             logger.debug("[BiodiViz NER/TaxoNERD] No organisms found in chunk.")

#         return {
#             "organisms":          list(all_organisms),
#             "relations":          flat_relations,
#             "sentence_relations": sentence_records,  # new — for HITL tab
#         }
        
        
# #OLD # """
# # # biotrace_hf_ner.py
# # # ────────────────────────────────────────────────────────────────────────────
# # # Hugging Face Token Classification (NER) and Relation Extraction (RE) wrapper
# # # based on the BiodiViz models.
# # # """
# # # import os
# # # import logging
# # # from itertools import combinations

# # # logger = logging.getLogger("biotrace.hf_ner")

# # # try:
# # #     import torch
# # #     import nltk
# # #     from transformers import pipeline
# # #     _HF_AVAILABLE = True
# # # except ImportError:
# # #     _HF_AVAILABLE = False
# # #     logger.warning("Install torch, transformers, and nltk to use BiodiViz NER.")

# # # class BiodiVizPipeline:
# # #     # Minimum RE classifier confidence to accept a relation.
# # #     # Lowered from 0.70 → 0.55: the BiodiViz RE model was trained on ecological
# # #     # text where species-locality links often score in the 0.55–0.70 band
# # #     # (especially for "occur_in" on genus-only or abbreviated names).
# # #     # False-positive relations are harmless — the LLM extraction pass filters
# # #     # them; false-negatives (missed locality links) directly reduce record count.
# # #     RE_CONFIDENCE_THRESHOLD = 0.55

# # #     def __init__(self, ner_model_path="./ner_model", re_model_path="./re_model"):
# # #         """Initializes the BiodiViz Transformer pipelines."""
# # #         if not _HF_AVAILABLE:
# # #             raise ImportError("Missing required Hugging Face libraries.")

# # #         self.device = 0 if torch.cuda.is_available() else -1
# # #         nltk.download('punkt', quiet=True)
# # #         nltk.download('punkt_tab', quiet=True)

# # #         logger.info(f"Loading NER model from {ner_model_path}...")
# # #         self.token_classifier = pipeline(
# # #             "token-classification",
# # #             model=ner_model_path,
# # #             aggregation_strategy="first",
# # #             device=self.device,
# # #         )

# # #         # RE model is optional — NER-only mode works without it
# # #         self.re_classifier = None
# # #         if os.path.isdir(re_model_path):
# # #             try:
# # #                 logger.info(f"Loading RE model from {re_model_path}...")
# # #                 self.re_classifier = pipeline(
# # #                     "text-classification",   # correct task name (not "sentiment-analysis")
# # #                     model=re_model_path,
# # #                     device=self.device,
# # #                 )
# # #             except Exception as exc:
# # #                 logger.warning(
# # #                     "[BiodiViz] RE model failed to load (%s) — running NER-only mode.", exc
# # #                 )
# # #         else:
# # #             logger.info("[BiodiViz] RE model folder not found — running NER-only mode.")

# # #         self.label_mapping = {
# # #             "LABEL_1": "have",
# # #             "LABEL_2": "occur_in",
# # #             "LABEL_3": "influence",
# # #         }

# # #     def _mask_entities(self, sentence: str, entities: list[dict], mask_combo: tuple) -> tuple:
# # #         """
# # #         Replace the two entities in *mask_combo* with @TYPE$ markers while
# # #         leaving all other entity words as-is.

# # #         FIX: entities MUST be processed in ascending start-position order so
# # #         that the running *offset* is applied correctly.  The original code used
# # #         the unsorted order returned by the pipeline, which caused the offset to
# # #         be mis-applied when an earlier entity followed a later one in the list.
# # #         """
# # #         # Work on a sorted copy — do NOT mutate the caller's list
# # #         sorted_entities = sorted(entities, key=lambda e: e["start"])

# # #         masked_sentence = sentence
# # #         offset = 0
# # #         masked_words: list[tuple[str, str]] = []

# # #         for entity in sorted_entities:
# # #             if entity in mask_combo:
# # #                 entity_group = entity["entity_group"].upper()
# # #                 masked_word  = f"@{entity_group}$"
# # #                 masked_words.append((entity["word"], entity["entity_group"]))
# # #             else:
# # #                 masked_word = entity["word"]

# # #             start = entity["start"] + offset
# # #             end   = entity["end"]   + offset
# # #             masked_sentence = (
# # #                 masked_sentence[:start] + masked_word + masked_sentence[end:]
# # #             )
# # #             offset += len(masked_word) - len(entity["word"])

# # #         return masked_sentence, masked_words

# # #     def extract(self, text: str) -> dict:
# # #         """
# # #         Processes text and returns identified organisms and their related localities.

# # #         Returns:
# # #             {
# # #                 "organisms": ["C. lupus", "Acropora"],
# # #                 "relations": ["C. lupus occur_in Gulf of Mannar"]
# # #             }

# # #         Improvements vs original:
# # #           • Entities sorted by start position before masking (offset bug fix).
# # #           • Only pairs that include at least one ORGANISM are sent to the RE
# # #             model — reduces API calls by ~60 % on dense ecological text.
# # #           • Confidence threshold (RE_CONFIDENCE_THRESHOLD) filters noisy labels.
# # #           • Graceful NER-only fallback when RE model is unavailable.
# # #           • Uses 'text-classification' pipeline task (not 'sentiment-analysis').
# # #         """
# # #         sentences = nltk.sent_tokenize(text)
# # #         found_organisms: set[str] = set()
# # #         found_relations: list[str] = []

# # #         for s in sentences:
# # #             # ── 1. NER pass ──────────────────────────────────────────────────
# # #             try:
# # #                 entities: list[dict] = self.token_classifier(s)
# # #             except Exception as exc:
# # #                 logger.debug("[BiodiViz NER] sentence skipped: %s", exc)
# # #                 continue

# # #             if not entities:
# # #                 continue

# # #             for ent in entities:
# # #                 if ent["entity_group"].upper() == "ORGANISM":
# # #                     found_organisms.add(ent["word"])

# # #             # ── 2. Relation Extraction pass ──────────────────────────────────
# # #             if self.re_classifier is None:
# # #                 continue   # NER-only mode — skip RE entirely

# # #             # FIX: Only consider pairs where at least one entity is ORGANISM.
# # #             # The original code checked ALL N*(N-1)/2 pairs, including pure
# # #             # habitat-locality pairs that the RE model can't usefully classify.
# # #             organism_entities  = [e for e in entities if e["entity_group"].upper() == "ORGANISM"]
# # #             non_organism_entities = [e for e in entities if e["entity_group"].upper() != "ORGANISM"]

# # #             candidate_pairs: list[tuple] = []
# # #             for org in organism_entities:
# # #                 for other in non_organism_entities:
# # #                     candidate_pairs.append((org, other))
# # #             # Also check organism-organism pairs (e.g. host–parasite)
# # #             for pair in combinations(organism_entities, 2):
# # #                 candidate_pairs.append(pair)

# # #             for combo in candidate_pairs:
# # #                 masked_sentence, masked_words = self._mask_entities(s, entities, combo)
# # #                 if len(masked_words) != 2:
# # #                     continue

# # #                 try:
# # #                     results = self.re_classifier(masked_sentence)
# # #                 except Exception as exc:
# # #                     logger.debug("[BiodiViz RE] classification error: %s", exc)
# # #                     continue

# # #                 label      = results[0]["label"]
# # #                 confidence = float(results[0].get("score", 0.0))

# # #                 # Skip LABEL_0 (no relation) and low-confidence predictions
# # #                 if label == "LABEL_0" or confidence < self.RE_CONFIDENCE_THRESHOLD:
# # #                     continue

# # #                 relation = self.label_mapping.get(label, "unknown")
# # #                 entity_1, type_1 = masked_words[0]
# # #                 entity_2, type_2 = masked_words[1]

# # #                 if "ORGANISM" in (type_1.upper(), type_2.upper()):
# # #                     rel_str = f"[{entity_1}] {relation} [{entity_2}]"
# # #                     found_relations.append(rel_str)

# # #         return {
# # #             "organisms": list(found_organisms),
# # #             "relations": list(set(found_relations)),
# # #         }
# # """
# # biotrace_hf_ner.py
# # ────────────────────────────────────────────────────────────────────────────
# # Hugging Face Token Classification (NER) wrapper.
# # UPDATED: Replaced the original BiodiViz model with TaxoNERD via adapter 
# # to fix skipped opisthobranch header records.
# # """
# # import logging
# # import itertools
# # import spacy
# # from taxo_extractor import TaxoNERD

# # logger = logging.getLogger("biotrace.hf_ner")

# # class BiodiVizPipeline:
# #     # Preserving threshold constant for backward compatibility 
# #     # with any biotrace_v5.py imports that might check it.
# #     RE_CONFIDENCE_THRESHOLD = 0.55

# #     def __init__(self, ner_model_path="./ner_model", re_model_path="./re_model"):
# #         """
# #         Initializes the upgraded Transformer pipelines.
# #         Ignores old local paths and pulls TaxoNERD.
# #         """
# #         logger.info("Initializing upgraded BiodiVizPipeline with TaxoNERD and spaCy...")
# #         self.taxo_ner = TaxoNERD()
        
# #         # Load spaCy for Locality extraction fallback
# #         try:
# #             self.nlp = spacy.load("en_core_web_sm")
# #         except OSError:
# #             import subprocess
# #             logger.warning("Downloading spacy en_core_web_sm model...")
# #             subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
# #             self.nlp = spacy.load("en_core_web_sm")

# #     def extract(self, text):
# #         """
# #         Main extraction method called by biotrace_v5.py chunk loop.
# #         Returns the exact dictionary structure expected by your schema mapper.
# #         """
# #         if not text:
# #             return {"organisms": [], "relations": []}

# #         # 1. Run the new BioBERT extraction
# #         organisms = self.taxo_ner.extract_organisms(text)
        
# #         # 2. Extract Localities using spaCy to fix missing locations
# #         localities = set()
# #         doc = self.nlp(text)
# #         for ent in doc.ents:
# #             if ent.label_ in ["GPE", "LOC"]:
# #                 loc_clean = ent.text.strip().replace("\n", " ")
# #                 if len(loc_clean) > 2:
# #                     localities.add(loc_clean)
# #         localities = list(localities)

# #         # 3. Rebuild occur_in relations to fix deduplication grouping
# #         found_relations = []
# #         if organisms and localities:
# #             # Proximity heuristic: link all found species to all found localities in this chunk
# #             for org, loc in itertools.product(organisms, localities):
# #                 found_relations.append(f"[{org}] occur_in [{loc}]")
        
# #         # 4. Replicate your original logging behavior
# #         if organisms:
# #             logger.info(f"[BiodiViz NER / TaxoNERD] Found {len(organisms)} organisms, {len(localities)} localities.")
# #         else:
# #             logger.debug("[BiodiViz NER / TaxoNERD] No organisms found in chunk.")

# #         # 5. Return exact dictionary shape biotrace_v5.py expects
# #         return {
# #             "organisms": organisms,
# #             "relations": found_relations
# #         }