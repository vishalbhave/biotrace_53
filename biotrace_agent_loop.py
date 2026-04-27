"""
biotrace_agent_loop.py  —  BioTrace v5.4
────────────────────────────────────────────────────────────────────────────
Lightweight self-correcting extraction agent.

Detects stated species counts in document text and retries extraction
on missed species. No external agent framework dependency — compatible
with all existing BioTrace LLM providers (Ollama, OpenAI, Gemini, Anthropic).

Algorithm:
  1. Standard extraction → N species
  2. Parse document for stated expected count (e.g. "a total of 33 species")
  3. If N < expected × threshold, run targeted "missed species" prompt
  4. Merge, dedup, return

Wire into biotrace_v5.py:

    from biotrace_agent_loop import agent_extract_with_correction
    occurrences = agent_extract_with_correction(
        full_text   = md_text,
        extract_fn  = lambda t: extract_occurrences(t, ...),
        llm_fn      = lambda p: call_llm(p, provider, model_sel, api_key, ollama_url),
        log_cb      = log_cb,
        max_retries = 2,
    )
"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

logger = logging.getLogger("biotrace.agent")

# ─────────────────────────────────────────────────────────────────────────────
#  Expected-count detection patterns
# ─────────────────────────────────────────────────────────────────────────────

_EXPECTED_PATTERNS = [
    # "a total of 33 species"  /  "33 species of opisthobranchs"
    re.compile(
        r"(?:total\s+of|recorded|listed|comprises?|identified|documented|"
        r"reported|observed)\s+(\d+)\s+"
        r"(?:species|taxa|opisthobranch|nudibranch|gastropod|mollusk)",
        re.IGNORECASE,
    ),
    # "33 new records" — less reliable; lower priority
    re.compile(r"(\d+)\s+(?:new\s+)?records?\s+for\s+(?:India|Gujarat|study\s+area)",
               re.IGNORECASE),
    # Table header: "33" at end of table = total row
    re.compile(r"^\s*(?:Total|Sum)[:,]?\s+(\d+)", re.MULTILINE),
]

_CHECKLIST_TABLE_RE = re.compile(
    r"(?:Sr\.?\s*No\.?|No\.|#)\s+(?:Species|Taxon|Name)",
    re.IGNORECASE,
)


def detect_expected_species_count(text: str) -> Optional[int]:
    """
    Scan document text for self-stated species counts.
    Returns the highest plausible stated count, or None if not found.

    Priority: explicit "total of N" > table total row > "N new records"
    """
    candidates: list[int] = []

    for pattern in _EXPECTED_PATTERNS:
        for m in pattern.finditer(text):
            val = int(m.group(1))
            if 1 < val < 5000:   # sanity bounds
                candidates.append(val)

    if not candidates:
        return None

    # Return the largest plausible number
    # (avoid picking up page numbers or small citation counts)
    return max(candidates)


def _extract_names_from_occurrences(occurrences: list[dict]) -> set[str]:
    """Return the set of canonical species names from occurrence dicts."""
    names: set[str] = set()
    for r in occurrences:
        if not isinstance(r, dict):
            continue
        name = (r.get("validName") or r.get("recordedName") or
                r.get("Recorded Name", "")).strip()
        if name and not name.startswith("__candidate_"):
            names.add(name)
    return names


def _parse_llm_json_list(raw: str) -> list[dict]:
    """Robustly parse LLM output as JSON list."""
    # Strip thinking blocks
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    # Try direct parse
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return [r for r in v if isinstance(r, dict)]
            return [data]
    except json.JSONDecodeError:
        pass

    # Try extracting array from embedded text
    m = re.search(r"(\[[\s\S]*\])", raw)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, list):
                return [r for r in data if isinstance(r, dict)]
        except json.JSONDecodeError:
            pass

    return []


# ─────────────────────────────────────────────────────────────────────────────
#  Targeted re-extraction prompts
# ─────────────────────────────────────────────────────────────────────────────

_MISSED_SPECIES_PROMPT = """\
You are a marine biology extraction system performing a CORRECTION PASS.

A previous extraction of the document found only {found} species, but the
document states there should be {expected} total species.

Already extracted species (DO NOT duplicate these):
{found_list}

Your task: Find the REMAINING {deficit} species that were MISSED.
Focus especially on:
  • Table rows (scan every row — some species may be in dense multi-column tables)
  • Figure captions mentioning species
  • Methods section species lists
  • Appendices or supplementary lists
  • Species mentioned only in comparative statements

For each missed species, return a JSON object with these keys:
  "Recorded Name"   — scientific name exactly as written
  "Valid Name"      — leave as ""
  "verbatimLocality"— best available locality (use study area if not specified)
  "occurrenceType"  — "Primary" or "Secondary"
  "Source Citation" — paper citation
  "Habitat"         — habitat type or "Not Reported"
  "Raw Text Evidence" — exact sentence(s) mentioning this species

Return ONLY a valid JSON array [{{...}}, {{...}}].
Return [] if no additional species can be found.

DOCUMENT TEXT:
{text}
"""

_TABLE_REPARSE_PROMPT = """\
The following text contains a species checklist TABLE. Extract EVERY species
from every table row, including those with only a checkmark (√) in the
Present Study column.

Rules:
  • One JSON object per species per row — even if the species spans multiple rows.
  • Use "Gulf of Kutch" as verbatimLocality unless a finer site is mentioned.
  • occurrenceType = "Primary" for rows with √ in "Present Study" column.
  • occurrenceType = "Secondary" for rows without √ in "Present Study" column.
  • Include species from ALL sections of the table (head, body, footnotes).

Return ONLY a valid JSON array. No prose.

TABLE TEXT:
{text}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Agent table detector
# ─────────────────────────────────────────────────────────────────────────────

def _detect_tables(text: str) -> list[str]:
    """
    Detect Markdown table blocks in the text and return them as strings.
    Minimum 3 rows (header + separator + 1 data row).
    """
    tables: list[str] = []
    lines  = text.splitlines()
    i = 0
    while i < len(lines):
        if re.match(r"^\|.*\|", lines[i]):
            start = i
            while i < len(lines) and (re.match(r"^\|.*\|", lines[i]) or
                                      re.match(r"^\|[-: |]+\|", lines[i])):
                i += 1
            block = "\n".join(lines[start:i])
            if block.count("\n") >= 2:   # at least 3 rows
                tables.append(block)
        else:
            i += 1
    return tables


# ─────────────────────────────────────────────────────────────────────────────
#  Main agent
# ─────────────────────────────────────────────────────────────────────────────

def agent_extract_with_correction(
    full_text:      str,
    extract_fn:     Callable[[str], list[dict]],
    llm_fn:         Callable[[str], str],
    log_cb:         Callable,
    max_retries:    int   = 2,
    completion_threshold: float = 0.95,  # accept if ≥ 95% of expected found
    focus_on_tables: bool = True,        # re-parse tables before full-text retry
) -> list[dict]:
    """
    Self-correcting extraction agent.

    Parameters
    ----------
    full_text            : full Markdown-converted document text
    extract_fn           : partial of extract_occurrences() — takes text, returns list[dict]
    llm_fn               : partial of call_llm() — takes prompt string, returns string
    log_cb               : logging callback (same signature as BioTrace log_cb)
    max_retries          : maximum correction iterations
    completion_threshold : fraction of expected count to consider "done"
    focus_on_tables      : run table-specific re-parse before general retry

    Returns
    -------
    Final merged + deduplicated occurrence list.
    """
    # ── Phase 1: Standard extraction ─────────────────────────────────────────
    log_cb("[Agent] Phase 1: Running standard extraction…")
    results: list[dict] = extract_fn(full_text)

    found_names = _extract_names_from_occurrences(results)
    log_cb(f"[Agent] Phase 1 complete: {len(found_names)} unique species found")

    # ── Detect expected count ────────────────────────────────────────────────
    expected = detect_expected_species_count(full_text)

    if expected is None:
        log_cb("[Agent] No stated species count found in document — "
               "skipping correction loop")
        return results

    log_cb(f"[Agent] Document states {expected} expected species; "
           f"found {len(found_names)} "
           f"({'✅ complete' if len(found_names) >= expected else '⚠️ incomplete'})")

    # ── Phase 2: Table-targeted re-parse (if enabled) ─────────────────────
    if focus_on_tables and len(found_names) < expected * completion_threshold:
        tables = _detect_tables(full_text)
        if tables:
            log_cb(f"[Agent] Phase 2: Re-parsing {len(tables)} detected table(s)…")
            for idx, table_text in enumerate(tables):
                if len(found_names) >= expected * completion_threshold:
                    break
                prompt = _TABLE_REPARSE_PROMPT.format(text=table_text)
                try:
                    raw = llm_fn(prompt)
                    extra = _parse_llm_json_list(raw)
                    new_names = _extract_names_from_occurrences(extra) - found_names
                    if new_names:
                        results.extend([
                            r for r in extra
                            if (_extract_names_from_occurrences([r]) & new_names)
                        ])
                        found_names |= new_names
                        log_cb(f"[Agent] Table {idx+1}: +{len(new_names)} species "
                               f"({', '.join(sorted(new_names)[:5])}…)")
                    else:
                        log_cb(f"[Agent] Table {idx+1}: no new species found")
                except Exception as exc:
                    log_cb(f"[Agent] Table {idx+1} re-parse error: {exc}", "warn")

    # ── Phase 3: General missed-species retry ────────────────────────────
    for attempt in range(max_retries):
        if len(found_names) >= expected * completion_threshold:
            log_cb(
                f"[Agent] Completion threshold met: "
                f"{len(found_names)}/{expected} species "
                f"({len(found_names)/expected:.0%})"
            )
            break

        deficit = expected - len(found_names)
        log_cb(f"[Agent] Phase 3 retry {attempt+1}: seeking {deficit} missed species…")

        found_list_str = "\n".join(f"  - {n}" for n in sorted(found_names))
        prompt = _MISSED_SPECIES_PROMPT.format(
            found    = len(found_names),
            expected = expected,
            deficit  = deficit,
            found_list = found_list_str,
            text     = full_text[:6000],
        )

        try:
            raw   = llm_fn(prompt)
            extra = _parse_llm_json_list(raw)
            new_names = _extract_names_from_occurrences(extra) - found_names

            if not new_names:
                log_cb(f"[Agent] Retry {attempt+1}: no new species found — stopping")
                break

            results.extend([
                r for r in extra
                if (_extract_names_from_occurrences([r]) & new_names)
            ])
            found_names |= new_names
            log_cb(
                f"[Agent] Retry {attempt+1}: +{len(new_names)} species: "
                + ", ".join(sorted(new_names)[:8])
                + ("…" if len(new_names) > 8 else "")
            )
        except Exception as exc:
            log_cb(f"[Agent] Retry {attempt+1} error: {exc}", "warn")
            break

    # ── Final report ──────────────────────────────────────────────────────
    gap = max(0, expected - len(found_names))
    log_cb(
        f"[Agent] Final: {len(found_names)} species found "
        f"(expected {expected}; gap={gap}; "
        f"completion={len(found_names)/expected:.0%})"
    )
    if gap > 0:
        log_cb(
            f"[Agent] ⚠️ {gap} species still unaccounted for. "
            f"Check: (1) OCR quality, (2) table extraction, "
            f"(3) secondary records in bibliography.",
            "warn",
        )

    return results
