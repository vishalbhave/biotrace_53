"""
biotrace_wiki_autopop.py  —  BioTrace Wiki Auto-Population Agent
═══════════════════════════════════════════════════════════════════════════════
Stand-alone enhancement for BioTrace v5.3+ / v5.6.
Adds ONE new capability:

    WikiAutoPopAgent.populate_species(sp_name, ...)

which:

  1. Fetches DYNAMIC DATA from live databases
       • Taxonomy  →  occurrences SQLite  (kingdom / phylum / class_ / order_ / family_)
       • Occurrences table  →  occurrences SQLite  (live, reflects HITL edits)

  2. Runs an Ollama Wiki Architect Agent chunk-by-chunk AUTOMATICALLY
       • Agent receives raw text chunks approved by HITL (or all chunks if bypass=True)
       • Each chunk triggers a targeted section-fill pass
       • Lead section is written from a synthesis pass over all chunks

  3. Writes the populated article back to wiki SQLite (versioned)

  4. Exposes a Streamlit panel:
       wiki_autopop_agent.render_streamlit_panel(wiki, meta_db_path, ...)

Design principles
─────────────────
  • No patch files — pure additive module, imported alongside existing code.
  • Idempotent  — re-running skips already-processed chunk hashes.
  • Agent loop  — each chunk is an agentic step; agent decides which sections
                  need updating and returns ONLY the affected section keys.
  • Graceful    — every DB / LLM call is wrapped; failures logged, never fatal.

Usage in biotrace_v53.py (Tab 7 wiki section)
──────────────────────────────────────────────
    from biotrace_wiki_autopop import WikiAutoPopAgent

    agent = WikiAutoPopAgent(
        ollama_url  = OLLAMA_URL,           # e.g. "http://localhost:11434"
        ollama_model= st.session_state.get("ollama_model", "llama3"),
        meta_db_path= META_DB_PATH,
        wiki        = get_patched_wiki(),   # BioTraceWikiUnified instance
    )

    # Inside Tab 7 render:
    agent.render_streamlit_panel()

    # Or call programmatically after HITL approval:
    agent.populate_species(
        sp_name        = "Chromodoris annae",
        approved_chunks= [{"text": "...", "citation": "Smith 2024"}],
        force_lead     = True,
    )
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger("biotrace.wiki_autopop")


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_AGENT_SYSTEM = """\
You are a Marine Biodiversity Wiki Architect specialised in taxonomy, ecology,
and conservation of marine invertebrates, fish, and coastal organisms.

You receive:
  (A) CURRENT WIKI ARTICLE JSON  — the article as it currently stands.
  (B) ONE SOURCE CHUNK           — a raw text excerpt from a scientific paper.
  (C) CITATION                   — bibliographic reference for the chunk.

Your task: return a JSON object with ONLY the sections that need updating.
Follow these rules strictly:
  1. NEVER delete or overwrite existing content.
  2. If a section already contains the same fact, do NOT repeat it.
  3. Write in Wikipedia-style neutral, encyclopaedic prose.
  4. Use *italics* for binomial names and **bold** for key terms.
  5. Append inline citations: "Smith (2024) reports…"
  6. Only populate sections where the chunk actually provides new information.
  7. Return a compact JSON — only keys that changed, under a "sections" key.

Example valid response:
{
  "sections": {
    "morphology": "The mantle is **orange** with white-edged gills (Bhave 2011).",
    "distribution_habitat": "Recorded in the **intertidal zone** at Lakshadweep (Bhave 2011)."
  }
}

Return ONLY valid JSON. No prose, no markdown fences, no explanation.
"""

_SECTION_AGENT_USER = """\
CURRENT ARTICLE JSON (truncated to 5000 chars):
{current_json}

SOURCE CHUNK:
{chunk_text}

CITATION:
{citation}

Which sections need updating? Return only the changed sections as JSON.
"""

_LEAD_SYNTHESIS_SYSTEM = """\
You are a Marine Biodiversity Wiki Lead-Section Writer.

You receive:
  (A) CURRENT WIKI ARTICLE JSON  — full article with all sections populated.
  (B) TAXONOMY                   — live taxonomy from occurrence database.
  (C) OCCURRENCE SUMMARY         — count and key localities from the live DB.

Your task: Write a SINGLE encyclopaedic lead paragraph (3–5 sentences) for
the species article. The lead must:
  1. Open with the species name in italics and its authority.
  2. State the higher taxonomy (family, order).
  3. Summarise the main habitat and geographic range.
  4. Include any notable ecological or conservation fact if present.
  5. Do NOT invent facts not supported by the article sections.

Return a JSON object with a single key:
{
  "lead": "Full lead paragraph here."
}

No prose outside the JSON. No markdown fences.
"""

_LEAD_SYNTHESIS_USER = """\
ARTICLE JSON (sections only, truncated 4000 chars):
{sections_json}

LIVE TAXONOMY:
  Kingdom : {kingdom}
  Phylum  : {phylum}
  Class   : {class_}
  Order   : {order_}
  Family  : {family_}
  Authority: {authority}

LIVE OCCURRENCES:
  Total records : {occ_count}
  Key localities: {key_localities}

Write the lead paragraph now.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_occ_table(conn: sqlite3.Connection) -> str:
    """Return the name of the occurrence table that exists."""
    for t in ("occurrences_v4", "occurrences"):
        try:
            conn.execute(f"SELECT 1 FROM {t} LIMIT 1")
            return t
        except sqlite3.OperationalError:
            continue
    raise sqlite3.OperationalError("No occurrence table found.")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())


def _chunk_hash(text: str) -> str:
    return hashlib.md5(text[:2000].encode()).hexdigest()[:12]


def _strip_fences(raw: str) -> str:
    """Remove ```json / ``` fences that some models emit."""
    raw = re.sub(r"^```+(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```+$",          "", raw.strip(), flags=re.MULTILINE)
    return raw.strip()


def _safe_json(raw: str) -> Optional[dict]:
    """Parse JSON with fence stripping and error swallowing."""
    try:
        return json.loads(_strip_fences(raw))
    except Exception as exc:
        logger.debug("[AutoPop] JSON parse failed: %s | raw[:200]=%s", exc, raw[:200])
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_taxonomy_from_db(meta_db_path: str, sp_name: str) -> dict:
    """
    Pull dynamic taxonomy for *sp_name* from the live occurrence DB.
    Returns a dict with keys: kingdom, phylum, class_, order_, family_,
    authority, iucnStatus, wormsID, taxonomicStatus.
    Falls back to empty strings if the species isn't found.
    """
    empty = {
        "kingdom": "", "phylum": "", "class_": "", "order_": "",
        "family_": "", "authority": "", "iucnStatus": "",
        "wormsID": "", "taxonomicStatus": "unverified",
    }
    if not meta_db_path:
        return empty
    try:
        conn  = sqlite3.connect(meta_db_path)
        table = _resolve_occ_table(conn)
        row   = conn.execute(
            f"""SELECT kingdom, phylum, class_, order_, family_,
                       nameAccordingTo, iucnStatus, wormsID, taxonomicStatus
                FROM {table}
                WHERE (validName=? OR recordedName=?)
                  AND kingdom IS NOT NULL
                ORDER BY id LIMIT 1""",
            (sp_name, sp_name),
        ).fetchone()
        conn.close()
        if row:
            return {
                "kingdom":        row[0] or "",
                "phylum":         row[1] or "",
                "class_":         row[2] or "",
                "order_":         row[3] or "",
                "family_":        row[4] or "",
                "authority":      row[5] or "",
                "iucnStatus":     row[6] or "",
                "wormsID":        row[7] or "",
                "taxonomicStatus":row[8] or "unverified",
            }
    except Exception as exc:
        logger.warning("[AutoPop] fetch_taxonomy_from_db: %s", exc)
    return empty


def fetch_occurrences_from_db_full(meta_db_path: str, sp_name: str,
                                   limit: int = 200) -> list[dict]:
    """
    Fetch live occurrences from the occurrence DB for *sp_name*.
    Returns list of dicts with all columns needed for the wiki article.
    """
    if not meta_db_path:
        return []
    try:
        conn  = sqlite3.connect(meta_db_path)
        table = _resolve_occ_table(conn)
        rows  = conn.execute(
            f"""SELECT id, recordedName, validName, verbatimLocality,
                       decimalLatitude, decimalLongitude, occurrenceType,
                       phylum, class_, order_, family_, sourceCitation,
                       geocodingSource, taxonomicStatus, habitat,
                       depthRange, eventDate
                FROM {table}
                WHERE (validName=? OR recordedName=?)
                ORDER BY id LIMIT ?""",
            (sp_name, sp_name, limit),
        ).fetchall()
        conn.close()
        result = []
        for r in rows:
            (rid, rec_name, valid_name, locality,
             lat, lon, occ_type,
             phylum, class_, order_, family_, citation,
             geo_src, tax_status, habitat, depth, event_date) = r
            result.append({
                "id":             rid,
                "recordedName":   rec_name  or "",
                "validName":      valid_name or rec_name or "",
                "locality":       locality   or "—",
                "latitude":       lat,
                "longitude":      lon,
                "occurrenceType": occ_type   or "Uncertain",
                "phylum":         phylum or "",
                "class_":         class_ or "",
                "order_":         order_ or "",
                "family_":        family_ or "",
                "source":         citation   or "",
                "geocodingSource":geo_src    or "",
                "taxonomicStatus":tax_status or "",
                "habitat":        habitat    or "",
                "depthRange":     depth      or "",
                "eventDate":      event_date or "",
            })
        return result
    except Exception as exc:
        logger.warning("[AutoPop] fetch_occurrences_from_db_full: %s", exc)
        return []


def fetch_pending_hitl_chunks(meta_db_path: str, sp_name: str) -> list[dict]:
    """
    Fetch text chunks from HITL-approved records (hitl_approved=1) that
    have not yet been wiki-processed (wiki_processed=0 or NULL).

    Expected table: hitl_chunks (or hitl_queue) with columns:
      id, species_name, chunk_text, citation, hitl_approved, wiki_processed

    If no HITL chunk table exists, returns empty list (graceful).
    """
    if not meta_db_path:
        return []
    try:
        conn = sqlite3.connect(meta_db_path)
        # Discover table name
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        table = None
        for candidate in ("hitl_chunks", "hitl_queue", "extraction_chunks"):
            if candidate in tables:
                table = candidate
                break
        if not table:
            conn.close()
            return []
        rows = conn.execute(
            f"""SELECT id, chunk_text, citation
                FROM {table}
                WHERE species_name=?
                  AND (hitl_approved=1 OR hitl_approved='1')
                  AND (wiki_processed IS NULL OR wiki_processed=0)
                ORDER BY id""",
            (sp_name,),
        ).fetchall()
        conn.close()
        return [
            {"id": r[0], "text": r[1] or "", "citation": r[2] or ""}
            for r in rows
        ]
    except Exception as exc:
        logger.debug("[AutoPop] fetch_pending_hitl_chunks: %s", exc)
        return []


def mark_chunks_wiki_processed(meta_db_path: str, chunk_ids: list[int]) -> None:
    """Mark HITL chunks as wiki-processed so they are not re-processed."""
    if not meta_db_path or not chunk_ids:
        return
    try:
        conn = sqlite3.connect(meta_db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        table = next(
            (t for t in ("hitl_chunks", "hitl_queue", "extraction_chunks")
             if t in tables), None
        )
        if table:
            conn.executemany(
                f"UPDATE {table} SET wiki_processed=1 WHERE id=?",
                [(cid,) for cid in chunk_ids],
            )
            conn.commit()
        conn.close()
    except Exception as exc:
        logger.debug("[AutoPop] mark_chunks_wiki_processed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
#  OLLAMA CALLER
# ─────────────────────────────────────────────────────────────────────────────

def call_ollama(
    prompt: str,
    ollama_url: str  = "http://localhost:11434",
    model: str       = "llama3",
    timeout: int     = 120,
    system: str      = "",
) -> str:
    """
    Call Ollama /api/generate endpoint.  Returns model response text.
    Raises on HTTP or connection errors (caller should catch).
    """
    import requests  # local import — avoids hard dependency at module load
    payload: dict = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    resp = requests.post(
        f"{ollama_url}/api/generate",
        json    = payload,
        timeout = timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

_VALID_SECTIONS = {
    "lead", "taxonomy_phylogeny", "morphology", "distribution_habitat",
    "ecology_behaviour", "conservation", "specimen_records",
}


class WikiAutoPopAgent:
    """
    Agent-driven, chunk-based wiki auto-population for BioTrace.

    Parameters
    ──────────
    wiki        : BioTraceWikiUnified instance (already initialised).
    meta_db_path: path to metadata_v5.db (occurrence + optionally HITL chunks).
    ollama_url  : Ollama server URL.
    ollama_model: model tag to use.
    llm_fn      : optional external LLM callable (str → str).  If provided,
                  overrides Ollama for all calls.
    """

    def __init__(
        self,
        wiki,
        meta_db_path:  str = "",
        ollama_url:    str = "http://localhost:11434",
        ollama_model:  str = "llama3",
        llm_fn: Optional[Callable[[str], str]] = None,
    ):
        self.wiki         = wiki
        self.meta_db_path = meta_db_path
        self.ollama_url   = ollama_url
        self.ollama_model = ollama_model
        self._llm_fn      = llm_fn

    # ── LLM dispatch ──────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, system: str = "") -> str:
        """Route to external llm_fn or Ollama."""
        if self._llm_fn:
            full = (f"{system}\n\n{prompt}") if system else prompt
            return self._llm_fn(full)
        return call_ollama(
            prompt     = prompt,
            system     = system,
            ollama_url = self.ollama_url,
            model      = self.ollama_model,
        )

    # ── Taxonomy sync from DB ─────────────────────────────────────────────────

    def sync_taxonomy_from_db(self, sp_name: str) -> dict:
        """
        Pull live taxonomy from occurrence DB and write it back into the
        wiki article (non-destructively — only fills blank fields).
        Returns the taxonomy dict for informational use.
        """
        taxon = fetch_taxonomy_from_db(self.meta_db_path, sp_name)
        art   = self.wiki.get_species_article(sp_name)
        if not art:
            logger.warning("[AutoPop] No wiki article for %s — skipping taxonomy sync", sp_name)
            return taxon

        changed = False
        field_map = {
            "kingdom":   "kingdom",
            "phylum":    "phylum",
            "class_":    "class_",
            "order_":    "order_",
            "family_":   "family_",
            "authority": "authority",
            "iucnStatus":"iucnStatus",
            "wormsID":   "wormsID",
            "taxonomicStatus": "taxonomicStatus",
        }
        for art_key, db_key in field_map.items():
            if not art.get(art_key) and taxon.get(db_key):
                art[art_key] = taxon[db_key]
                changed = True

        if changed:
            # Use wiki's internal write (versioned)
            slug = _slug(sp_name)
            art["updated_at"] = datetime.now().isoformat()
            self.wiki._write(
                "species", slug, sp_name, art,
                change_note="AutoPop: taxonomy synced from occurrence DB",
            )
            logger.info("[AutoPop] Taxonomy synced for %s", sp_name)
        return taxon

    # ── Occurrence sync from DB ───────────────────────────────────────────────

    def sync_occurrences_from_db(self, sp_name: str) -> list[dict]:
        """
        Fetch live occurrences from the occurrence DB and merge them into
        the wiki article's occurrence_points list (by id, de-duplicated).
        Returns the full occurrence list.
        """
        occs = fetch_occurrences_from_db_full(self.meta_db_path, sp_name)
        if not occs:
            return []

        art  = self.wiki.get_species_article(sp_name)
        if not art:
            return occs

        existing_ids = {
            pt.get("db_id") for pt in art.get("occurrence_points", [])
            if pt.get("db_id")
        }
        new_pts = []
        for o in occs:
            if o["id"] in existing_ids:
                continue
            new_pts.append({
                "db_id":          o["id"],
                "locality":       o["locality"],
                "latitude":       o["latitude"],
                "longitude":      o["longitude"],
                "occurrenceType": o["occurrenceType"],
                "depth_m":        o["depthRange"],
                "source":         o["source"],
                "recordedName":   o["recordedName"],
                "validName":      o["validName"],
                "habitat":        o["habitat"],
                "eventDate":      o["eventDate"],
            })

        if new_pts:
            art.setdefault("occurrence_points", []).extend(new_pts)
            art["updated_at"] = datetime.now().isoformat()
            slug = _slug(sp_name)
            self.wiki._write(
                "species", slug, sp_name, art,
                change_note=f"AutoPop: {len(new_pts)} occurrences synced from DB",
            )
            logger.info("[AutoPop] %d occurrences synced for %s", len(new_pts), sp_name)
        return occs

    # ── Single-chunk agent step ───────────────────────────────────────────────

    def _process_chunk(
        self, sp_name: str, chunk_text: str, citation: str,
    ) -> dict:
        """
        Run one agentic chunk step: send current article + chunk to LLM,
        merge only the sections the agent returned.
        Returns {"sections_updated": [...], "skipped": bool}.
        """
        ch = _chunk_hash(chunk_text)

        art = self.wiki.get_species_article(sp_name)
        if not art:
            logger.warning("[AutoPop] No article for %s — skipping chunk", sp_name)
            return {"sections_updated": [], "skipped": True}

        # Skip if chunk already processed
        if any(p.get("chunk_hash") == ch for p in art.get("provenance", [])):
            logger.debug("[AutoPop] Chunk %s already processed for %s", ch, sp_name)
            return {"sections_updated": [], "skipped": True}

        # Build compact article representation for the prompt
        art_prompt = {
            "title":    art.get("title", sp_name),
            "sections": art.get("sections", {}),
            "occurrence_count": len(art.get("occurrence_points", [])),
        }
        prompt = _SECTION_AGENT_USER.format(
            current_json = json.dumps(art_prompt, indent=2)[:5000],
            chunk_text   = chunk_text[:4000],
            citation     = citation,
        )

        try:
            raw     = self._call_llm(prompt, system=_SECTION_AGENT_SYSTEM)
            updated = _safe_json(raw)
        except Exception as exc:
            logger.warning("[AutoPop] LLM call failed for chunk %s/%s: %s", sp_name, ch, exc)
            return {"sections_updated": [], "skipped": True}

        if not updated:
            return {"sections_updated": [], "skipped": True}

        # Merge sections returned by agent
        new_secs  = updated.get("sections", {})
        merged    = []
        art_secs  = art.setdefault("sections", {})
        for sec_key, new_text in new_secs.items():
            if sec_key not in _VALID_SECTIONS or not new_text:
                continue
            existing = art_secs.get(sec_key, "")
            if not existing:
                art_secs[sec_key] = new_text
                merged.append(sec_key)
            elif new_text.strip() and new_text.strip() not in existing:
                art_secs[sec_key] = existing.rstrip() + "\n\n" + new_text.strip()
                merged.append(sec_key)

        # Record provenance
        art.setdefault("provenance", []).append({
            "citation":   citation,
            "date":       datetime.now().isoformat(),
            "chunk_hash": ch,
            "enhanced":   True,
            "agent":      "WikiAutoPopAgent",
        })
        art["updated_at"] = datetime.now().isoformat()

        slug = _slug(sp_name)
        self.wiki._write(
            "species", slug, sp_name, art,
            change_note=f"AutoPop: chunk {ch} → sections {merged}",
        )
        logger.info("[AutoPop] Chunk %s processed for %s → %s", ch, sp_name, merged)
        return {"sections_updated": merged, "skipped": False}

    # ── Lead synthesis pass ───────────────────────────────────────────────────

    def _synthesise_lead(self, sp_name: str, taxon: dict, occs: list[dict]) -> str:
        """
        After all chunks are processed, synthesise the lead paragraph from
        the full article + live taxonomy + occurrence summary.
        Returns the generated lead string (empty on failure).
        """
        art = self.wiki.get_species_article(sp_name)
        if not art:
            return ""

        key_localities = ", ".join(
            {o["locality"] for o in occs if o["locality"] != "—"}
        )[:300] or "not recorded"

        sections_snapshot = json.dumps(art.get("sections", {}), indent=2)[:4000]
        prompt = _LEAD_SYNTHESIS_USER.format(
            sections_json  = sections_snapshot,
            kingdom        = taxon.get("kingdom",  "Animalia"),
            phylum         = taxon.get("phylum",   ""),
            class_         = taxon.get("class_",   ""),
            order_         = taxon.get("order_",   ""),
            family_        = taxon.get("family_",  ""),
            authority      = taxon.get("authority",""),
            occ_count      = len(occs),
            key_localities = key_localities,
        )

        try:
            raw    = self._call_llm(prompt, system=_LEAD_SYNTHESIS_SYSTEM)
            parsed = _safe_json(raw)
        except Exception as exc:
            logger.warning("[AutoPop] Lead synthesis LLM error for %s: %s", sp_name, exc)
            return ""

        if not parsed:
            return ""

        lead = parsed.get("lead", "")
        if lead:
            art = self.wiki.get_species_article(sp_name)   # re-read (may have changed)
            art.setdefault("sections", {})["lead"] = lead
            art["updated_at"] = datetime.now().isoformat()
            slug = _slug(sp_name)
            self.wiki._write(
                "species", slug, sp_name, art,
                change_note="AutoPop: lead synthesised",
            )
            logger.info("[AutoPop] Lead written for %s (%d chars)", sp_name, len(lead))
        return lead

    # ── Main entry point ──────────────────────────────────────────────────────

    def populate_species(
        self,
        sp_name:         str,
        approved_chunks: Optional[list[dict]] = None,
        force_lead:      bool = True,
        bypass_hitl:     bool = False,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        Full auto-population pipeline for one species.

        Parameters
        ──────────
        sp_name         : species name (validName or recordedName).
        approved_chunks : list of {"text": str, "citation": str} dicts.
                          If None, chunks are fetched from HITL table in DB.
        force_lead      : always (re-)synthesise the lead section after chunks.
        bypass_hitl     : if True, use approved_chunks without checking HITL
                          table (useful for manual calls / testing).
        progress_cb     : optional callable(str) for progress messages.

        Returns
        ───────
        dict with keys:
          taxonomy_synced   : bool
          occurrences_count : int
          chunks_processed  : int
          chunks_skipped    : int
          sections_updated  : list[str]
          lead_generated    : bool
          errors            : list[str]
        """
        result = {
            "taxonomy_synced":   False,
            "occurrences_count": 0,
            "chunks_processed":  0,
            "chunks_skipped":    0,
            "sections_updated":  [],
            "lead_generated":    False,
            "errors":            [],
        }

        def _log(msg: str):
            logger.info("[AutoPop] %s", msg)
            if progress_cb:
                progress_cb(msg)

        _log(f"▶ Starting auto-population for {sp_name}")

        # ── Step 1: Sync taxonomy from DB ─────────────────────────────────────
        try:
            taxon = self.sync_taxonomy_from_db(sp_name)
            result["taxonomy_synced"] = True
            _log(f"  ✅ Taxonomy synced — family: {taxon.get('family_','?')}")
        except Exception as exc:
            result["errors"].append(f"taxonomy_sync: {exc}")
            taxon = {}
            _log(f"  ⚠️ Taxonomy sync failed: {exc}")

        # ── Step 2: Sync occurrences from DB ──────────────────────────────────
        try:
            occs = self.sync_occurrences_from_db(sp_name)
            result["occurrences_count"] = len(occs)
            _log(f"  ✅ {len(occs)} occurrences synced from DB")
        except Exception as exc:
            result["errors"].append(f"occ_sync: {exc}")
            occs = []
            _log(f"  ⚠️ Occurrence sync failed: {exc}")

        # ── Step 3: Gather approved chunks ────────────────────────────────────
        if approved_chunks is None and not bypass_hitl:
            try:
                raw_chunks = fetch_pending_hitl_chunks(self.meta_db_path, sp_name)
                chunks = [{"text": r["text"], "citation": r["citation"],
                           "_db_id": r["id"]} for r in raw_chunks]
                _log(f"  📋 {len(chunks)} HITL-approved chunks pending")
            except Exception as exc:
                result["errors"].append(f"hitl_fetch: {exc}")
                chunks = []
        else:
            chunks = approved_chunks or []
            _log(f"  📋 {len(chunks)} chunks supplied directly")

        # ── Step 4: Agent chunk loop ──────────────────────────────────────────
        all_sections: list[str] = []
        processed_db_ids: list[int] = []

        for i, ch in enumerate(chunks, 1):
            text     = ch.get("text", "")
            citation = ch.get("citation", "")
            if not text.strip():
                continue
            _log(f"  🤖 Processing chunk {i}/{len(chunks)}: {citation[:50]}…")
            try:
                step_result = self._process_chunk(sp_name, text, citation)
                if step_result["skipped"]:
                    result["chunks_skipped"] += 1
                else:
                    result["chunks_processed"] += 1
                    all_sections.extend(step_result["sections_updated"])
                    if ch.get("_db_id"):
                        processed_db_ids.append(ch["_db_id"])
            except Exception as exc:
                result["errors"].append(f"chunk_{i}: {exc}")
                _log(f"    ⚠️ Chunk {i} error: {exc}")

        result["sections_updated"] = list(set(all_sections))

        # Mark HITL chunks as processed
        if processed_db_ids:
            mark_chunks_wiki_processed(self.meta_db_path, processed_db_ids)

        # ── Step 5: Lead synthesis ────────────────────────────────────────────
        art = self.wiki.get_species_article(sp_name)
        lead_empty = not (art and art.get("sections", {}).get("lead", "").strip())
        if force_lead or lead_empty:
            _log(f"  ✍️ Synthesising lead section…")
            try:
                lead = self._synthesise_lead(sp_name, taxon, occs)
                result["lead_generated"] = bool(lead)
                if lead:
                    _log(f"  ✅ Lead written ({len(lead)} chars)")
                else:
                    _log(f"  ⚠️ Lead synthesis returned empty")
            except Exception as exc:
                result["errors"].append(f"lead_synthesis: {exc}")
                _log(f"  ⚠️ Lead synthesis error: {exc}")

        _log(
            f"▶ Done: {result['chunks_processed']} chunks processed, "
            f"{result['occurrences_count']} occurrences, "
            f"lead={'✅' if result['lead_generated'] else '—'}"
        )
        return result

    # ── Batch population ──────────────────────────────────────────────────────

    def populate_all_species(
        self,
        species_list:    Optional[list[str]] = None,
        force_lead:      bool = True,
        bypass_hitl:     bool = False,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> dict[str, dict]:
        """
        Run populate_species() for every species in *species_list*.
        If species_list is None, iterates over all species in the wiki DB.
        Returns dict mapping sp_name → result dict.
        """
        if species_list is None:
            species_list = self.wiki.list_species()
        results: dict[str, dict] = {}
        for sp in species_list:
            results[sp] = self.populate_species(
                sp_name     = sp,
                force_lead  = force_lead,
                bypass_hitl = bypass_hitl,
                progress_cb = progress_cb,
            )
        return results


# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_autopop_panel(
    wiki,
    meta_db_path:  str = "",
    ollama_url:    str = "http://localhost:11434",
    ollama_model:  str = "llama3",
    llm_fn: Optional[Callable[[str], str]] = None,
):
    """
    Render a self-contained Streamlit panel for the Wiki Auto-Population Agent.

    Drop-in for Tab 7 (or any Streamlit expander).  Exposes:
      • Species selector with live DB occurrence count
      • 🤖 Run for selected species button
      • 🌐 Run for ALL species button
      • Live progress log
      • Result summary table

    Usage
    ─────
        from biotrace_wiki_autopop import render_autopop_panel
        render_autopop_panel(wiki=get_patched_wiki(), meta_db_path=META_DB_PATH,
                             ollama_url=OLLAMA_URL, ollama_model=model_name)
    """
    import streamlit as st  # local import — avoids hard dep at module level

    st.markdown("### 🤖 Ollama Wiki Architect Agent — Auto-Population")
    st.caption(
        "Fetches live taxonomy & occurrences from the database, then runs an "
        "agentic chunk-by-chunk LLM pass to populate all wiki article sections "
        "and synthesise the lead paragraph automatically."
    )

    agent = WikiAutoPopAgent(
        wiki         = wiki,
        meta_db_path = meta_db_path,
        ollama_url   = ollama_url,
        ollama_model = ollama_model,
        llm_fn       = llm_fn,
    )

    # ── Species selection ──────────────────────────────────────────────────────
    all_species = wiki.list_species() if hasattr(wiki, "list_species") else []
    if not all_species:
        st.info("No species in wiki yet. Run an extraction first.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox(
            "Select species to auto-populate:",
            options  = all_species,
            key      = "autopop_species_sel",
        )
    with col2:
        force_lead   = st.checkbox("Force lead re-synthesis", value=True,
                                   key="autopop_force_lead")
        bypass_hitl  = st.checkbox("Bypass HITL filter", value=False,
                                   key="autopop_bypass_hitl",
                                   help="Use all available chunks, not just HITL-approved ones")

    # ── Manual chunk input ─────────────────────────────────────────────────────
    with st.expander("➕ Add raw text chunks for this run (optional)", expanded=False):
        manual_chunks_raw = st.text_area(
            "Paste one or more chunks separated by --- (triple dash):",
            height=150, key="autopop_manual_chunks",
        )
        manual_citation = st.text_input(
            "Citation for manual chunks:", key="autopop_manual_cite"
        )
        manual_chunks: list[dict] = []
        if manual_chunks_raw.strip():
            for part in manual_chunks_raw.split("---"):
                if part.strip():
                    manual_chunks.append({
                        "text": part.strip(),
                        "citation": manual_citation or "Manual input",
                    })
        if manual_chunks:
            st.caption(f"{len(manual_chunks)} manual chunk(s) will be added.")

    st.divider()

    # ── Action buttons ─────────────────────────────────────────────────────────
    btn_col1, btn_col2 = st.columns(2)
    run_single = btn_col1.button(
        f"🤖 Populate: {selected[:35]}…" if len(selected) > 35 else f"🤖 Populate: {selected}",
        key="autopop_run_single", use_container_width=True,
    )
    run_all = btn_col2.button(
        f"🌐 Populate ALL ({len(all_species)} species)",
        key="autopop_run_all", use_container_width=True,
    )

    if not (run_single or run_all):
        return

    # ── Progress container ─────────────────────────────────────────────────────
    log_box     = st.empty()
    status_area = st.status("Running Wiki Auto-Population Agent…", expanded=True)
    log_lines: list[str] = []

    def _progress(msg: str):
        log_lines.append(msg)
        log_box.code("\n".join(log_lines[-30:]), language="")

    # ── Run ────────────────────────────────────────────────────────────────────
    with status_area:
        if run_single:
            chunks_to_use = manual_chunks if manual_chunks else None
            result = agent.populate_species(
                sp_name        = selected,
                approved_chunks= chunks_to_use,
                force_lead     = force_lead,
                bypass_hitl    = bypass_hitl,
                progress_cb    = _progress,
            )
            results_map = {selected: result}

        else:  # run_all
            results_map = agent.populate_all_species(
                species_list = all_species,
                force_lead   = force_lead,
                bypass_hitl  = bypass_hitl,
                progress_cb  = _progress,
            )

    # ── Summary table ──────────────────────────────────────────────────────────
    st.markdown("#### 📊 Run Summary")
    import pandas as pd
    rows = []
    for sp, r in results_map.items():
        rows.append({
            "Species":          sp,
            "Taxonomy ✅":      "✅" if r.get("taxonomy_synced")  else "—",
            "Occurrences":      r.get("occurrences_count", 0),
            "Chunks processed": r.get("chunks_processed",  0),
            "Chunks skipped":   r.get("chunks_skipped",    0),
            "Sections updated": ", ".join(r.get("sections_updated", [])) or "—",
            "Lead generated":   "✅" if r.get("lead_generated") else "—",
            "Errors":           "; ".join(r.get("errors", [])) or "—",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    success_count = sum(
        1 for r in results_map.values()
        if r.get("lead_generated") or r.get("chunks_processed")
    )
    st.success(
        f"✅ Auto-population complete — {success_count}/{len(results_map)} "
        f"species updated. Refresh the wiki browser to see changes."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  INTEGRATION NOTES
# ─────────────────────────────────────────────────────────────────────────────
INTEGRATION_NOTES = """
biotrace_wiki_autopop.py — Integration Notes
═════════════════════════════════════════════

NO PATCH FILES REQUIRED — import and call directly.

## Minimal integration in biotrace_v53.py Tab 7

    from biotrace_wiki_autopop import render_autopop_panel

    # Inside your Tab 7 block:
    with st.expander("🤖 Auto-Populate Wiki Articles", expanded=False):
        render_autopop_panel(
            wiki         = get_patched_wiki(WIKI_ROOT, META_DB_PATH),
            meta_db_path = META_DB_PATH,
            ollama_url   = st.session_state.get("ollama_url",  "http://localhost:11434"),
            ollama_model = st.session_state.get("ollama_model","llama3"),
            llm_fn       = your_existing_llm_fn,   # or None to use Ollama directly
        )

## Programmatic use (e.g. after HITL approval callback)

    from biotrace_wiki_autopop import WikiAutoPopAgent

    agent = WikiAutoPopAgent(wiki=wiki, meta_db_path=META_DB_PATH)

    # Called automatically after HITL approves a batch:
    agent.populate_species(
        sp_name        = "Chromodoris annae",
        approved_chunks= hitl_approved_chunk_list,   # [{"text":…,"citation":…}]
        force_lead     = True,
    )

## HITL chunk table (optional but recommended)

If you store extraction chunks in a SQLite table, name it one of:
    hitl_chunks | hitl_queue | extraction_chunks

With columns:
    id              INTEGER PRIMARY KEY
    species_name    TEXT
    chunk_text      TEXT
    citation        TEXT
    hitl_approved   INTEGER   -- 1 = approved
    wiki_processed  INTEGER   -- 0 = pending; 1 = done (auto-set by agent)

The agent fetches only hitl_approved=1 AND wiki_processed=0 rows,
then marks them wiki_processed=1 after success.

## What each step does

Step 1 — Taxonomy sync
    Reads kingdom/phylum/class_/order_/family_/authority from the occurrence DB
    and fills any blank fields in the wiki article.  No LLM call.

Step 2 — Occurrence sync
    Reads all occurrence records for the species from the live DB and merges
    them into the wiki article's occurrence_points list (de-duplicated by id).
    These instantly appear in the rendered occurrence table and folium map.

Step 3 — Chunk gather
    Retrieves HITL-approved pending chunks from hitl_chunks table, OR uses
    manually supplied chunks.

Step 4 — Agent chunk loop
    For each chunk, sends (current article + chunk + citation) to the Ollama
    Wiki Architect agent.  The agent returns ONLY the sections it can update.
    Chunk hashes are recorded in provenance to prevent re-processing.

Step 5 — Lead synthesis
    Once all chunks are processed, synthesises the lead paragraph from the
    full article + live taxonomy + occurrence locality summary.
    Overwrites "No lead section yet…" placeholder.
"""
