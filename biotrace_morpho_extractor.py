"""
biotrace_morpho_extractor.py  —  BioTrace v5.5
────────────────────────────────────────────────────────────────────────────
Third LLM pass per document: morphological data + specimen metadata extraction.

Integrates with the existing pipeline via the same llm_fn pattern used by
biotrace_relation_extractor.py.  All imports are optional — the module
degrades gracefully when Scikit-LLM or its dependencies are absent.

Fields extracted
────────────────
Taxonomic / authority:
  authority_string        — "Ihering, 1876" / "(Forsskål, 1775)"
  nomenclatural_status    — "sp. nov." | "re-description" | "new record" | "established"
  full_authority          — author(s) + year with parentheses if applicable
  type_status             — "holotype" | "paratype" | "syntype" | "lectotype" | ""

Morphological / diagnostic:
  diagnostic_characters   — list[str]  key feature phrases
  coloration_life         — colour description in life (string)
  coloration_preserved    — colour description in preservation (string)
  size_length_mm          — [min, max] or single value (float list)
  size_width_mm           — [min, max] or single value (float list)
  radular_formula         — e.g. "60 x 31.1.31" for molluscs
  key_features_summary    — 3-bullet plain-English summary (LLM)

Specimen metadata (provenance):
  voucher_numbers         — list[str]  e.g. ["BMNH 197211", "ZSI/MBRC/F.2344"]
  repository              — museum/repository name string
  collector               — collector name(s)
  collection_date         — ISO date string or raw date phrase
  type_locality           — verbatim type locality sentence
  type_lat                — float | None  (geocoded from type locality)
  type_lon                — float | None

Scikit-LLM usage (optional)
────────────────────────────
When scikit-llm is installed AND openai_api_key is provided, ZeroShotGPTClassifier
is used for:
  • Habitat type normalisation  → standard ENVO categories
  • Nomenclatural status classification
  • Diagnostic character tagging

Falls back to regex + direct LLM call when Scikit-LLM is unavailable.

Wire into biotrace_v5.py
────────────────────────
    from biotrace_morpho_extractor import extract_morpho_data, classify_habitats_skllm

    morpho = extract_morpho_data(
        text           = md_text,
        species_name   = valid_name,
        source_citation= citation,
        file_hash      = file_hash,
        llm_fn         = lambda p: call_llm(p, provider, model, api_key, ollama_url),
        meta_db_path   = META_DB_PATH,
    )

    # Optional Scikit-LLM habitat normalisation
    habitat_str = classify_habitats_skllm(
        raw_habitats   = ["rocky intertidal", "below low tide mark"],
        openai_api_key = api_key,   # None → falls back to LLM
    )
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("biotrace.morpho")

# ─────────────────────────────────────────────────────────────────────────────
#  DATA MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MorphoRecord:
    """
    All morphological and specimen fields for one species mention in a paper.
    Maps to the species_morphology SQLite table.
    """
    # Identity
    species_name:        str   = ""
    source_citation:     str   = ""
    file_hash:           str   = ""

    # Taxonomic / authority
    authority_string:    str   = ""   # "Ihering, 1876"
    full_authority:      str   = ""   # "(Forsskål, 1775)"
    nomenclatural_status:str   = ""   # "sp. nov." | "re-description" | "new record" | "established"
    type_status:         str   = ""   # "holotype" | "paratype" | etc.

    # Morphological / diagnostic
    diagnostic_characters: list[str] = field(default_factory=list)
    coloration_life:        str       = ""
    coloration_preserved:   str       = ""
    size_length_mm:         list[float] = field(default_factory=list)
    size_width_mm:          list[float] = field(default_factory=list)
    radular_formula:        str         = ""
    key_features_summary:   str         = ""

    # Specimen metadata
    voucher_numbers:    list[str] = field(default_factory=list)
    repository:         str       = ""
    collector:          str       = ""
    collection_date:    str       = ""
    type_locality:      str       = ""
    type_lat:           Optional[float] = None
    type_lon:           Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
#  STANDARD HABITAT CATEGORIES  (ENVO-inspired)
# ─────────────────────────────────────────────────────────────────────────────

STANDARD_HABITATS = [
    "coral reef",
    "rocky intertidal",
    "sandy intertidal",
    "mangrove",
    "seagrass bed",
    "subtidal rocky bottom",
    "subtidal sandy bottom",
    "kelp forest",
    "estuarine",
    "deep sea",
    "pelagic open ocean",
    "brackish water",
    "freshwater",
    "terrestrial forest",
    "terrestrial grassland",
    "wetland",
    "not reported",
]

# Regex shortcuts for nomenclatural status
_STATUS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bsp(?:ecies)?\.?\s+nov(?:a|us)?\b",          re.I), "sp. nov."),
    (re.compile(r"\bn(?:ew)?\.?\s+sp(?:ecies)?\b",              re.I), "sp. nov."),
    (re.compile(r"\bre[‑\-]?descri(?:ption|bed|bes)\b",         re.I), "re-description"),
    (re.compile(r"\bnew\s+(?:record|report|occurrence)\b",       re.I), "new record"),
    (re.compile(r"\bfirst\s+(?:record|report)\b",                re.I), "new record"),
    (re.compile(r"\bredescri(?:ption|bed|bes)\b",                re.I), "re-description"),
]

# Voucher patterns: ZSI/..., BMNH ..., NHM ..., USNM ..., MNHN ..., etc.
_VOUCHER_RE = re.compile(
    r"\b(?:"
    r"ZSI/[A-Z]{2,8}/[A-Z\d\.\-/]+"
    r"|BMNH\s?[A-Z\d\.\-/\s]{4,20}"
    r"|NHM[A-Z]?\s?[A-Z\d\.\-/\s]{4,20}"
    r"|USNM\s?[A-Z\d\.\-/\s]{4,20}"
    r"|MNHN[A-Z\s\.\-/]{4,20}"
    r"|SMF\s?[A-Z\d\.\-/\s]{4,20}"
    r"|MCZ\s?[A-Z\d\.\-/\s]{4,20}"
    r"|[A-Z]{2,8}[\.\-][A-Z\d\.\-/]{3,}\d{4,}"
    r")\b",
    re.IGNORECASE,
)

# Authority pattern: "Author, 1900" or "(Author & Author, 1900)"
_AUTHORITY_RE = re.compile(
    r"[\(\[]?"
    r"(?:[A-Z][A-Za-zÀ-ÿ''\-]+(?:\s+(?:and|&|et)\s+[A-Z][A-Za-zÀ-ÿ''\-]+)?)"
    r"[,\s]*"
    r"(1[6-9]\d{2}|20[0-2]\d)"
    r"[\)\]]?",
)

# Size extraction: "12.5–18.3 mm" or "up to 45 mm"
_SIZE_RE = re.compile(
    r"(?:up\s+to\s+|approximately\s+|about\s+)?"
    r"(\d+(?:\.\d+)?)"
    r"(?:\s*[–\-]\s*(\d+(?:\.\d+)?))?"
    r"\s*mm",
    re.IGNORECASE,
)

# Radular formula: "N x M.J.M" pattern
_RADULA_RE = re.compile(
    r"\b(\d+)\s*[×x]\s*(\d+(?:\.\d+)?(?:\s*\.\s*\d+(?:\.\d+)?){1,3})\b",
    re.IGNORECASE,
)

# Collector: "coll. Smith" / "collected by J. Smith" / "leg. Smith"
_COLLECTOR_RE = re.compile(
    r"(?:coll(?:ected)?\.?\s+(?:by\s+)?|leg\.\s*)"
    r"([A-Z][A-Za-z\s\.\-]{3,40}?)(?=[,;.]|\s+on\s|\s+in\s|\s+\d)",
    re.IGNORECASE,
)

# Type locality: sentences containing "type locality"
_TYPE_LOC_RE = re.compile(
    r"(?:type\s+locality)[:\s]+([^.;]{10,200})[.;]",
    re.IGNORECASE,
)

# Repository patterns: museum abbreviations + full names
_REPO_RE = re.compile(
    r"\b(?:"
    r"Natural\s+History\s+Museum(?:\s+London)?"
    r"|Zoological\s+Survey\s+of\s+India"
    r"|Smithsonian"
    r"|MNHN|BMNH|NHM|ZSI|MCZ|USNM|SMF"
    r"|British\s+Museum"
    r"|[A-Z]{2,6}\s+Museum"
    r")\b",
    re.IGNORECASE,
)

# Type status markers
_TYPE_STATUS_RE = re.compile(
    r"\b(holotype|paratype|allotype|syntype|lectotype|neotype|"
    r"paralectotype|topotype|cotype)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACTION PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_MORPHO_PROMPT = """\
You are a marine biology extraction system performing a MORPHOLOGY AND SPECIMEN PASS.

From the TEXT below, extract all morphological descriptions and specimen data
for the species: **{species_name}**

Return a single JSON object with EXACTLY these keys (use "" or [] for missing fields):

  "authority_string"      — author(s) + year only: e.g. "Ihering, 1876" or "(Forsskål, 1775)"
  "full_authority"        — full formatted authority with parentheses if applicable
  "nomenclatural_status"  — EXACTLY one of: "sp. nov." | "re-description" | "new record" | "established"
  "type_status"           — "holotype" | "paratype" | "syntype" | "lectotype" | "" (if not stated)
  "diagnostic_characters" — list of short key-feature phrases, e.g.:
                             ["Spatulate penis", "Multiporous opaline gland", "Radular formula 60×31.1.31"]
  "coloration_life"       — colour description in life (if stated), e.g. "Pale greeny brown dorsally"
  "coloration_preserved"  — colour in preservation/alcohol (if stated)
  "size_length_mm"        — [min_mm, max_mm] or [single_mm] — floats; [] if not stated
  "size_width_mm"         — [min_mm, max_mm] or [single_mm] — floats; [] if not stated
  "radular_formula"       — e.g. "60 x 31.1.31" — "" if not stated or not molluscan
  "key_features_summary"  — 2-3 sentence plain English summary of what makes this species distinctive
  "voucher_numbers"       — list of museum accession/registration numbers, e.g. ["BMNH reg. 197211"]
  "repository"            — museum or collection name, e.g. "Natural History Museum, London"
  "collector"             — collector name(s), e.g. "J.H. Orton"
  "collection_date"       — date or date range, e.g. "March 1971" or "1969-03-15"
  "type_locality"         — verbatim sentence describing where the type specimen was collected
  "type_lat"              — decimal latitude of type locality (float) or null
  "type_lon"              — decimal longitude of type locality (float) or null

RULES:
  • Extract data ONLY for {species_name} — not for other species.
  • "diagnostic_characters" should be SHORT phrases (not full sentences), max 8 items.
  • If coloration is not described separately for life vs preserved, put it under "coloration_life".
  • For "nomenclatural_status": "sp. nov." = new species; "re-description" = same species re-described
    with new detail; "new record" = previously described species reported for new region;
    "established" = routine mention of known species.
  • Return ONLY valid JSON — no markdown, no prose.

TEXT:
{text}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  SCIKIT-LLM HABITAT CLASSIFICATION  (optional)
# ─────────────────────────────────────────────────────────────────────────────

def classify_habitats_skllm(
    raw_habitats:   list[str],
    openai_api_key: Optional[str] = None,
    openai_model:   str = "gpt-3.5-turbo",
    llm_fn:         Optional[callable] = None,
) -> list[str]:
    """
    Normalise a list of raw habitat strings to standard ENVO-inspired categories.

    Priority:
      1. Scikit-LLM ZeroShotGPTClassifier (if scikit-llm installed + API key)
      2. Direct LLM call via llm_fn (if provided)
      3. Regex keyword matching (always-available fallback)

    Parameters
    ----------
    raw_habitats    : list of verbatim habitat strings from extraction
    openai_api_key  : OpenAI API key for Scikit-LLM (optional)
    openai_model    : model name for Scikit-LLM
    llm_fn          : existing BioTrace llm_fn as fallback

    Returns
    -------
    List of normalised habitat strings (same length as input).
    """
    if not raw_habitats:
        return []

    # ── Strategy 1: Scikit-LLM ─────────────────────────────────────────────
    if openai_api_key:
        try:
            from skllm import ZeroShotGPTClassifier
            from skllm.config import SKLLMConfig
            SKLLMConfig.set_openai_key(openai_api_key)
            SKLLMConfig.set_openai_org("")

            clf = ZeroShotGPTClassifier(openai_model=openai_model)
            clf.fit(None, STANDARD_HABITATS)
            predictions = clf.predict(raw_habitats)
            logger.info("[Morpho/skllm] Classified %d habitats via Scikit-LLM", len(predictions))
            return list(predictions)
        except ImportError:
            logger.debug("[Morpho/skllm] scikit-llm not installed — skipping")
        except Exception as exc:
            logger.warning("[Morpho/skllm] Classification failed: %s", exc)

    # ── Strategy 2: Direct LLM call ─────────────────────────────────────────
    if llm_fn:
        prompt = (
            f"Classify each habitat description to the CLOSEST standard category.\n\n"
            f"Standard categories:\n"
            + "\n".join(f"  - {h}" for h in STANDARD_HABITATS)
            + f"\n\nHabitat descriptions to classify (one per line):\n"
            + "\n".join(f"{i+1}. {h}" for i, h in enumerate(raw_habitats))
            + "\n\nReturn ONLY a JSON array of the matching standard category strings, "
            f"one per input line. No prose."
        )
        try:
            raw = llm_fn(prompt)
            raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                result = json.loads(m.group())
                if isinstance(result, list) and len(result) == len(raw_habitats):
                    return [str(r) for r in result]
        except Exception as exc:
            logger.debug("[Morpho/llm] Habitat classification error: %s", exc)

    # ── Strategy 3: Regex keyword fallback ─────────────────────────────────
    def _regex_classify(text: str) -> str:
        t = text.lower()
        for cat in STANDARD_HABITATS:
            if any(kw in t for kw in cat.split()):
                return cat
        return "not reported"

    return [_regex_classify(h) for h in raw_habitats]


def classify_nomenclatural_status_skllm(
    text_excerpt:   str,
    openai_api_key: Optional[str] = None,
    openai_model:   str = "gpt-3.5-turbo",
) -> str:
    """
    Use Scikit-LLM to classify the nomenclatural status of a species mention.
    Falls back to regex matching when Scikit-LLM is unavailable.
    """
    # Regex first — fast and reliable for standard patterns
    for pattern, status in _STATUS_PATTERNS:
        if pattern.search(text_excerpt):
            return status

    # Scikit-LLM for ambiguous cases
    if openai_api_key:
        try:
            from skllm import ZeroShotGPTClassifier
            from skllm.config import SKLLMConfig
            SKLLMConfig.set_openai_key(openai_api_key)

            labels = ["sp. nov.", "re-description", "new record", "established"]
            clf = ZeroShotGPTClassifier(openai_model=openai_model)
            clf.fit(None, labels)
            return clf.predict([text_excerpt[:500]])[0]
        except Exception as exc:
            logger.debug("[Morpho/skllm] Status classification: %s", exc)

    return "established"


# ─────────────────────────────────────────────────────────────────────────────
#  REGEX PRE-PASS  (fast baseline before LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _regex_prepass(text: str, species_name: str) -> dict:
    """
    Extract what we can with regex — reduces token burden on LLM by
    pre-populating high-confidence fields.
    """
    result: dict = {
        "authority_string":     "",
        "nomenclatural_status": "",
        "type_status":          "",
        "voucher_numbers":      [],
        "repository":           "",
        "collector":            "",
        "type_locality":        "",
        "size_length_mm":       [],
        "radular_formula":      "",
    }

    # Species context window: ±500 chars around each mention
    windows: list[str] = []
    for m in re.finditer(re.escape(species_name), text, re.IGNORECASE):
        start = max(0, m.start() - 500)
        end   = min(len(text), m.end() + 500)
        windows.append(text[start:end])

    context = " ".join(windows) if windows else text[:3000]

    # Authority
    auth_m = _AUTHORITY_RE.search(context)
    if auth_m:
        result["authority_string"] = auth_m.group().strip()

    # Nomenclatural status
    for pattern, status in _STATUS_PATTERNS:
        if pattern.search(context):
            result["nomenclatural_status"] = status
            break

    # Type status
    ts_m = _TYPE_STATUS_RE.search(context)
    if ts_m:
        result["type_status"] = ts_m.group(1).lower()

    # Vouchers
    vouchers = _VOUCHER_RE.findall(context)
    result["voucher_numbers"] = list(dict.fromkeys(v.strip() for v in vouchers))[:10]

    # Repository
    repo_m = _REPO_RE.search(context)
    if repo_m:
        result["repository"] = repo_m.group().strip()

    # Collector
    coll_m = _COLLECTOR_RE.search(context)
    if coll_m:
        result["collector"] = coll_m.group(1).strip()

    # Type locality
    tl_m = _TYPE_LOC_RE.search(context)
    if tl_m:
        result["type_locality"] = tl_m.group(1).strip()

    # Size
    sizes = [
        (float(m.group(1)), float(m.group(2)) if m.group(2) else float(m.group(1)))
        for m in _SIZE_RE.finditer(context)
    ]
    if sizes:
        all_vals = [v for pair in sizes for v in pair]
        result["size_length_mm"] = [min(all_vals), max(all_vals)]

    # Radular formula
    rad_m = _RADULA_RE.search(context)
    if rad_m:
        result["radular_formula"] = f"{rad_m.group(1)} x {rad_m.group(2)}"

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  DB PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_morpho_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS species_morphology (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            species_name          TEXT NOT NULL,
            source_citation       TEXT,
            file_hash             TEXT,
            authority_string      TEXT,
            full_authority        TEXT,
            nomenclatural_status  TEXT,
            type_status           TEXT,
            diagnostic_characters TEXT,   -- JSON array
            coloration_life       TEXT,
            coloration_preserved  TEXT,
            size_length_mm        TEXT,   -- JSON array
            size_width_mm         TEXT,   -- JSON array
            radular_formula       TEXT,
            key_features_summary  TEXT,
            voucher_numbers       TEXT,   -- JSON array
            repository            TEXT,
            collector             TEXT,
            collection_date       TEXT,
            type_locality         TEXT,
            type_lat              REAL,
            type_lon              REAL,
            created_at            TEXT DEFAULT (datetime('now'))
        )
    """)
    # Partial index so we can skip re-extracting the same species+paper
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_morpho_species_hash
        ON species_morphology (species_name, file_hash)
    """)
    conn.commit()


def _persist_morpho(db_path: str, record: MorphoRecord) -> None:
    if not record.species_name:
        return
    try:
        conn = sqlite3.connect(db_path)
        _ensure_morpho_table(conn)

        # Skip if already extracted for this species+paper
        existing = conn.execute(
            "SELECT id FROM species_morphology WHERE species_name=? AND file_hash=?",
            (record.species_name, record.file_hash),
        ).fetchone()
        if existing:
            conn.close()
            logger.debug("[Morpho] Already in DB: %s", record.species_name)
            return

        conn.execute(
            """INSERT INTO species_morphology (
                species_name, source_citation, file_hash,
                authority_string, full_authority, nomenclatural_status, type_status,
                diagnostic_characters, coloration_life, coloration_preserved,
                size_length_mm, size_width_mm, radular_formula, key_features_summary,
                voucher_numbers, repository, collector, collection_date,
                type_locality, type_lat, type_lon
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record.species_name,
                record.source_citation,
                record.file_hash,
                record.authority_string,
                record.full_authority,
                record.nomenclatural_status,
                record.type_status,
                json.dumps(record.diagnostic_characters),
                record.coloration_life,
                record.coloration_preserved,
                json.dumps(record.size_length_mm),
                json.dumps(record.size_width_mm),
                record.radular_formula,
                record.key_features_summary,
                json.dumps(record.voucher_numbers),
                record.repository,
                record.collector,
                record.collection_date,
                record.type_locality,
                record.type_lat,
                record.type_lon,
            ),
        )
        conn.commit()
        conn.close()
        logger.info("[Morpho] Persisted morpho record for: %s", record.species_name)
    except Exception as exc:
        logger.error("[Morpho] DB write error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
#  GEOCODE TYPE LOCALITY  (best-effort, reuses Nominatim if available)
# ─────────────────────────────────────────────────────────────────────────────

def _geocode_type_locality(
    locality_text: str,
    timeout: int = 8,
) -> tuple[Optional[float], Optional[float]]:
    """
    Attempt to geocode a type locality string via Nominatim.
    Returns (lat, lon) or (None, None).
    """
    if not locality_text or len(locality_text) < 5:
        return None, None
    try:
        import urllib.parse
        import urllib.request
        query = locality_text[:120]
        url = (
            "https://nominatim.openstreetmap.org/search"
            f"?q={urllib.parse.quote(query)}&format=json&limit=1"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "BioTrace/5.5"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as exc:
        logger.debug("[Morpho/geocode] %s: %s", locality_text[:40], exc)
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN EXTRACTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_morpho_data(
    text:            str,
    species_name:    str,
    source_citation: str,
    file_hash:       str,
    llm_fn:          callable,
    meta_db_path:    str,
    openai_api_key:  Optional[str]  = None,
    max_text_chars:  int            = 5000,
    geocode_type_loc: bool          = True,
) -> MorphoRecord:
    """
    Extract morphological and specimen data for a single species.

    Algorithm:
      1. Regex pre-pass (fast, fills high-confidence fields)
      2. LLM extraction pass (fills remaining fields + richer descriptions)
      3. Merge: regex wins for structured fields; LLM wins for narrative fields
      4. Scikit-LLM: nomenclatural status classification (if API key provided)
      5. Geocode type locality (Nominatim, optional)
      6. Persist to DB

    Parameters
    ----------
    text            : full document text (Markdown)
    species_name    : valid scientific name to target
    source_citation : paper citation string
    file_hash       : document hash for dedup
    llm_fn          : callable(prompt: str) -> str — existing BioTrace LLM wrapper
    meta_db_path    : path to metadata SQLite DB
    openai_api_key  : optional key for Scikit-LLM enhancements
    max_text_chars  : character budget sent to LLM
    geocode_type_loc: whether to geocode the type locality string

    Returns
    -------
    Populated MorphoRecord (also persisted to DB).
    """
    logger.info("[Morpho] Extracting data for: %s", species_name)

    record = MorphoRecord(
        species_name    = species_name,
        source_citation = source_citation,
        file_hash       = file_hash,
    )

    # ── 1. Regex pre-pass ───────────────────────────────────────────────────
    pre = _regex_prepass(text, species_name)
    record.authority_string     = pre["authority_string"]
    record.nomenclatural_status = pre["nomenclatural_status"]
    record.type_status          = pre["type_status"]
    record.voucher_numbers      = pre["voucher_numbers"]
    record.repository           = pre["repository"]
    record.collector            = pre["collector"]
    record.type_locality        = pre["type_locality"]
    record.size_length_mm       = pre["size_length_mm"]
    record.radular_formula      = pre["radular_formula"]

    # ── 2. LLM extraction pass ──────────────────────────────────────────────
    prompt = _MORPHO_PROMPT.format(
        species_name = species_name,
        text         = text[:max_text_chars],
    )
    try:
        raw = llm_fn(prompt)
        # Strip thinking blocks and markdown fences
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        # Extract JSON object
        m = re.search(r"(\{[\s\S]*\})", raw)
        if m:
            raw = m.group(1)
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("[Morpho] LLM pass failed for %s: %s", species_name, exc)
        data = {}

    # ── 3. Merge: LLM wins for narrative fields; regex wins for structured ──
    def _llm_str(key: str, default: str = "") -> str:
        v = data.get(key, default)
        return str(v).strip() if v else default

    def _llm_list(key: str) -> list:
        v = data.get(key, [])
        if isinstance(v, list):
            return [str(x).strip() for x in v if x]
        return []

    def _llm_float_list(key: str) -> list[float]:
        v = data.get(key, [])
        if isinstance(v, list):
            result = []
            for x in v:
                try:
                    result.append(float(x))
                except (TypeError, ValueError):
                    pass
            return result
        return []

    # Narrative / complex fields: LLM wins
    record.full_authority        = _llm_str("full_authority",        record.authority_string)
    record.coloration_life       = _llm_str("coloration_life")
    record.coloration_preserved  = _llm_str("coloration_preserved")
    record.key_features_summary  = _llm_str("key_features_summary")
    record.collection_date       = _llm_str("collection_date")

    # Structured fields: LLM fills if regex missed
    if not record.authority_string:
        record.authority_string = _llm_str("authority_string")
    if not record.nomenclatural_status:
        record.nomenclatural_status = _llm_str("nomenclatural_status", "established")
    if not record.type_status:
        record.type_status = _llm_str("type_status")
    if not record.voucher_numbers:
        record.voucher_numbers = _llm_list("voucher_numbers")
    if not record.repository:
        record.repository = _llm_str("repository")
    if not record.collector:
        record.collector = _llm_str("collector")
    if not record.type_locality:
        record.type_locality = _llm_str("type_locality")
    if not record.size_length_mm:
        record.size_length_mm = _llm_float_list("size_length_mm")
    if not record.size_width_mm:
        record.size_width_mm = _llm_float_list("size_width_mm")
    if not record.radular_formula:
        record.radular_formula = _llm_str("radular_formula")

    # Diagnostic characters: always use LLM (richer descriptions)
    record.diagnostic_characters = _llm_list("diagnostic_characters")

    # Type coordinates from LLM
    try:
        if data.get("type_lat"):
            record.type_lat = float(data["type_lat"])
        if data.get("type_lon"):
            record.type_lon = float(data["type_lon"])
    except (TypeError, ValueError):
        pass

    # ── 4. Scikit-LLM: nomenclatural status classification ──────────────────
    if openai_api_key and not record.nomenclatural_status:
        record.nomenclatural_status = classify_nomenclatural_status_skllm(
            text[:1500], openai_api_key
        )

    # Ensure status is set
    if not record.nomenclatural_status:
        record.nomenclatural_status = "established"

    # ── 5. Geocode type locality ─────────────────────────────────────────────
    if geocode_type_loc and record.type_locality:
        if record.type_lat is None or record.type_lon is None:
            lat, lon = _geocode_type_locality(record.type_locality)
            if lat is not None:
                record.type_lat = lat
                record.type_lon = lon
                logger.info(
                    "[Morpho] Type locality geocoded: %s → (%.4f, %.4f)",
                    record.type_locality[:40], lat, lon,
                )

    # ── 6. Persist ───────────────────────────────────────────────────────────
    _persist_morpho(meta_db_path, record)
    return record


# ─────────────────────────────────────────────────────────────────────────────
#  BATCH EXTRACTION — process all species from one document
# ─────────────────────────────────────────────────────────────────────────────

def extract_morpho_batch(
    text:            str,
    species_list:    list[str],
    source_citation: str,
    file_hash:       str,
    llm_fn:          callable,
    meta_db_path:    str,
    openai_api_key:  Optional[str] = None,
    max_species:     int           = 10,
) -> dict[str, MorphoRecord]:
    """
    Run morphological extraction for a list of species from one document.

    Prioritises species that:
      • Have voucher/type locality signals in text (likely to have morpho data)
      • Are marked as sp. nov. or re-described (highest scientific value)

    Returns dict mapping species_name → MorphoRecord.
    """
    results: dict[str, MorphoRecord] = {}

    # Prioritise species with morphological signals
    def _priority_score(sp: str) -> int:
        ctx = text[max(0, text.find(sp) - 300): text.find(sp) + 300]
        score = 0
        if _VOUCHER_RE.search(ctx):       score += 3
        if _TYPE_LOC_RE.search(ctx):      score += 3
        if _TYPE_STATUS_RE.search(ctx):   score += 2
        for pat, _ in _STATUS_PATTERNS:
            if pat.search(ctx):           score += 2; break
        if _SIZE_RE.search(ctx):          score += 1
        return score

    prioritised = sorted(species_list, key=_priority_score, reverse=True)[:max_species]

    for sp in prioritised:
        if not sp.strip():
            continue
        record = extract_morpho_data(
            text            = text,
            species_name    = sp,
            source_citation = source_citation,
            file_hash       = file_hash,
            llm_fn          = llm_fn,
            meta_db_path    = meta_db_path,
            openai_api_key  = openai_api_key,
        )
        results[sp] = record

    logger.info("[Morpho] Batch complete: %d/%d species processed",
                len(results), len(species_list))
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  QUERY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_morpho_record(
    species_name: str,
    db_path:      str,
) -> Optional[MorphoRecord]:
    """
    Retrieve the most recent morpho record for a species from the DB.
    Returns None if not found.
    """
    try:
        conn = sqlite3.connect(db_path)
        _ensure_morpho_table(conn)
        row = conn.execute(
            "SELECT * FROM species_morphology WHERE species_name=? "
            "ORDER BY created_at DESC LIMIT 1",
            (species_name,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        cols = [
            "id", "species_name", "source_citation", "file_hash",
            "authority_string", "full_authority", "nomenclatural_status", "type_status",
            "diagnostic_characters", "coloration_life", "coloration_preserved",
            "size_length_mm", "size_width_mm", "radular_formula", "key_features_summary",
            "voucher_numbers", "repository", "collector", "collection_date",
            "type_locality", "type_lat", "type_lon", "created_at",
        ]
        d = dict(zip(cols, row))
        return MorphoRecord(
            species_name         = d["species_name"],
            source_citation      = d["source_citation"] or "",
            file_hash            = d["file_hash"] or "",
            authority_string     = d["authority_string"] or "",
            full_authority       = d["full_authority"] or "",
            nomenclatural_status = d["nomenclatural_status"] or "",
            type_status          = d["type_status"] or "",
            diagnostic_characters= json.loads(d["diagnostic_characters"] or "[]"),
            coloration_life      = d["coloration_life"] or "",
            coloration_preserved = d["coloration_preserved"] or "",
            size_length_mm       = json.loads(d["size_length_mm"] or "[]"),
            size_width_mm        = json.loads(d["size_width_mm"] or "[]"),
            radular_formula      = d["radular_formula"] or "",
            key_features_summary = d["key_features_summary"] or "",
            voucher_numbers      = json.loads(d["voucher_numbers"] or "[]"),
            repository           = d["repository"] or "",
            collector            = d["collector"] or "",
            collection_date      = d["collection_date"] or "",
            type_locality        = d["type_locality"] or "",
            type_lat             = d["type_lat"],
            type_lon             = d["type_lon"],
        )
    except Exception as exc:
        logger.error("[Morpho] DB read error: %s", exc)
        return None
