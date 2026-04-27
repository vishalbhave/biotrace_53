"""
biotrace_relation_extractor.py  —  BioTrace v5.4
────────────────────────────────────────────────────────────────────────────
Second LLM pass per document: cross-sentence relation triples (DeepKE-inspired).

BUG FIX (19-04-2026):
  ModuleNotFoundError: No module named 'biotrace_schemas'
  The file was imported as 'biotrace_schemas' but the actual module in the
  project is 'biotrace_schema' (no trailing 's'). To avoid this fragile
  cross-module dependency entirely, RelationTriple is now defined inline here.
  biotrace_schemas.py / biotrace_schema.py are no longer imported.

Relations extracted:
  FOUND_AT | CO_OCCURS_WITH | INHABITS | FEEDS_ON | PARASITE_OF |
  OBSERVED_AT_DEPTH | SYMBIONT_OF | COMPETITOR_OF

Output: species_relations SQLite table (created on first run).

New SQLite table schema:
  CREATE TABLE IF NOT EXISTS species_relations (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      subject_name    TEXT NOT NULL,
      relation_type   TEXT NOT NULL,
      object_value    TEXT NOT NULL,
      evidence_text   TEXT,
      source_citation TEXT,
      confidence      REAL DEFAULT 1.0,
      file_hash       TEXT,
      created_at      TEXT DEFAULT (datetime('now'))
  );
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field

logger = logging.getLogger("biotrace.relation_extractor")

# ─────────────────────────────────────────────────────────────────────────────
#  RelationTriple — defined inline to remove biotrace_schemas dependency
# ─────────────────────────────────────────────────────────────────────────────

VALID_RELATIONS = frozenset({
    "FOUND_AT",
    "CO_OCCURS_WITH",
    "INHABITS",
    "FEEDS_ON",
    "PARASITE_OF",
    "OBSERVED_AT_DEPTH",
    "SYMBIONT_OF",
    "COMPETITOR_OF",
})


@dataclass
class RelationTriple:
    """
    A cross-sentence relation triple — DeepKE Document RE format.
    Maps to the species_relations SQLite table.
    """
    subject:        str              # species name (head entity)
    relation:       str              # one of VALID_RELATIONS
    object:         str              # locality / habitat / depth / co-species / host
    evidence_text:  str  = ""        # verbatim supporting sentence(s)
    source_citation:str  = ""
    confidence:     float = 1.0

    def __post_init__(self):
        self.relation = self.relation.upper().strip()
        if self.relation not in VALID_RELATIONS:
            # Normalise near-matches
            for vr in VALID_RELATIONS:
                if vr in self.relation:
                    self.relation = vr
                    break
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


# ─────────────────────────────────────────────────────────────────────────────
#  Extraction prompt
# ─────────────────────────────────────────────────────────────────────────────

_RELATION_PROMPT = """\
You are a marine biology information extraction system.

Given the TEXT below and the list of KNOWN SPECIES, extract ALL explicit or
strongly implied relationships between species and:
  - Collection localities (named places)
  - Habitat types (coral reef, mangrove, intertidal, etc.)
  - Depth ranges
  - Co-occurring species
  - Prey / food items
  - Hosts (for parasites / symbionts)

Return a JSON array of objects with EXACTLY these keys:
  "subject"    — the species name (MUST be one of the KNOWN SPECIES listed below)
  "relation"   — EXACTLY one of:
                   FOUND_AT | CO_OCCURS_WITH | INHABITS | FEEDS_ON |
                   PARASITE_OF | OBSERVED_AT_DEPTH | SYMBIONT_OF | COMPETITOR_OF
  "object"     — locality / habitat / depth / co-occurring species / host
  "evidence"   — verbatim sentence(s) from the text that prove this relation
  "confidence" — float 0.0–1.0

RULES:
  • Only extract relations EXPLICITLY stated or very strongly implied.
  • subject MUST be a binomial from the KNOWN SPECIES list — never a life-stage.
  • For FOUND_AT: object must be a named geographic place, not a habitat phrase.
  • For INHABITS: object must be a habitat type (coral reef, mangrove, etc.)
  • Multiple relations per species are expected — extract ALL.
  • Return [] if no relations can be extracted with confidence ≥ 0.5.
  • No markdown, no prose, ONLY valid JSON array.

KNOWN SPECIES:
{species_list}

TEXT:
{text}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_relations_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS species_relations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name    TEXT NOT NULL,
            relation_type   TEXT NOT NULL,
            object_value    TEXT NOT NULL,
            evidence_text   TEXT,
            source_citation TEXT,
            confidence      REAL DEFAULT 1.0,
            file_hash       TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


def _persist_relations(
    db_path: str,
    triples: list[RelationTriple],
    file_hash: str,
) -> None:
    """Write relation triples to species_relations table."""
    if not triples:
        return
    try:
        conn = sqlite3.connect(db_path)
        _ensure_relations_table(conn)
        conn.executemany(
            """INSERT INTO species_relations
               (subject_name, relation_type, object_value, evidence_text,
                source_citation, confidence, file_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    t.subject,
                    t.relation,
                    t.object,
                    t.evidence_text,
                    t.source_citation,
                    t.confidence,
                    file_hash,
                )
                for t in triples
            ],
        )
        conn.commit()
        conn.close()
        logger.info("[RE] Persisted %d relation triples to DB", len(triples))
    except Exception as exc:
        logger.error("[RE] DB write error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
#  Main extraction function
# ─────────────────────────────────────────────────────────────────────────────

def extract_relations(
    text:            str,
    known_species:   list[str],
    source_citation: str,
    file_hash:       str,
    llm_call_fn,          # callable: fn(prompt: str) -> str
    meta_db_path:    str,
    max_species:     int = 20,
    max_text_chars:  int = 4000,
) -> list[RelationTriple]:
    """
    Run the relation extraction LLM pass and persist results.

    Parameters
    ----------
    text            : raw text chunk (full document or section)
    known_species   : valid_name strings already extracted in Stage 1
    source_citation : document citation string
    file_hash       : for provenance / dedup
    llm_call_fn     : the same LLM wrapper used by biotrace_v5.py
                      signature: fn(prompt: str) -> str
    meta_db_path    : path to the main SQLite DB
    max_species     : cap on species sent to LLM (token budget)
    max_text_chars  : cap on text sent to LLM (token budget)

    Returns
    -------
    List of RelationTriple objects (also persisted to DB).
    """
    if not known_species or not text.strip():
        return []

    # Deduplicate and filter empty strings
    species_clean = [s.strip() for s in known_species if s.strip()]
    species_clean = list(dict.fromkeys(species_clean))[:max_species]

    prompt = _RELATION_PROMPT.format(
        species_list="\n".join(f"  - {s}" for s in species_clean),
        text=text[:max_text_chars],
    )

    try:
        raw = llm_call_fn(prompt)
    except Exception as exc:
        logger.warning("[RE] LLM call failed: %s", exc)
        return []

    # Parse JSON — strip reasoning blocks and markdown fences
    import re
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    # Extract JSON array
    array_match = re.search(r"(\[\s*\{.*\}\s*\])", raw, re.DOTALL)
    if array_match:
        raw = array_match.group(1)
    elif re.match(r"^\s*\[\s*\]\s*$", raw):
        return []

    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data).__name__}")
    except Exception as exc:
        logger.warning("[RE] JSON parse error: %s | raw preview: %s", exc, raw[:80])
        return []

    triples: list[RelationTriple] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        subject  = str(item.get("subject",  "")).strip()
        relation = str(item.get("relation", "")).strip().upper()
        obj      = str(item.get("object",   "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        try:
            confidence = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0

        # Basic validation
        if not subject or not obj:
            continue
        if confidence < 0.5:
            continue

        try:
            triple = RelationTriple(
                subject=subject,
                relation=relation,
                object=obj,
                evidence_text=evidence,
                source_citation=source_citation,
                confidence=confidence,
            )
            triples.append(triple)
        except Exception as exc:
            logger.debug("[RE] Triple validation error: %s", exc)

    _persist_relations(meta_db_path, triples, file_hash)
    logger.info("[RE] Extracted %d relation triples from chunk", len(triples))
    return triples