"""
biotrace_wiki_unified.py  —  BioTrace v5.5  Unified Wiki Module
────────────────────────────────────────────────────────────────────────────
Replaces both biotrace_wiki.py and biotrace_wiki_enhanced.py.

Key design principles
─────────────────────
• Single class  BioTraceWikiUnified  — backward-compatible with BioTraceWiki API
• Git-like versioning in SQLite (wiki_versions table) — every article update
  snapshots the previous body so rollback is always possible
• LLM-driven "Wiki Architect" enhancement — incremental, conflict-aware,
  section-level merge (never replaces valid data)
• Wikipedia-style rendered output using biotrace_wiki.css
• Taxobox Authority field populated from WoRMS / occurrence records
• Status banner merged from taxonomicStatus + validationStatus
• Chunked enhancement: each new extraction chunk triggers a targeted section
  update rather than a full rewrite
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
from typing import Any, Callable, Optional


logger = logging.getLogger("biotrace.wiki_unified")

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    import folium
    from folium.plugins import MarkerCluster, MiniMap
    _FOLIUM = True
except ImportError:
    _FOLIUM = False

try:
    import requests as _requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False

try:
    import streamlit as st
    _ST = True
except ImportError:
    _ST = False


# ─────────────────────────────────────────────────────────────────────────────
#  CSS LOADER
# ─────────────────────────────────────────────────────────────────────────────
_CSS_CACHE: str = ""

def _load_css(css_path: Optional[str] = None) -> str:
    """Load biotrace_wiki.css once and cache it for the session."""
    global _CSS_CACHE
    if _CSS_CACHE:
        return _CSS_CACHE
    # Search alongside this module, then CWD
    candidates = [
        css_path,
        Path(__file__).with_name("biotrace_wiki.css"),
        Path("biotrace_wiki.css"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            _CSS_CACHE = Path(c).read_text(encoding="utf-8")
            logger.debug("[Wiki] CSS loaded from %s", c)
            return _CSS_CACHE
    logger.warning("[Wiki] biotrace_wiki.css not found — inline styles only")
    return ""


def inject_css_streamlit(css_path: Optional[str] = None):
    """Inject wiki CSS into a running Streamlit app (call once per session)."""
    if not _ST:
        return
    css = _load_css(css_path)
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SQLITE SCHEMA  — versioned articles
# ─────────────────────────────────────────────────────────────────────────────
_WIKI_SCHEMA = """
CREATE TABLE IF NOT EXISTS wiki_articles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    section      TEXT NOT NULL,           -- 'species' | 'locality' | ...
    slug         TEXT NOT NULL,
    title        TEXT NOT NULL,
    body_json    TEXT NOT NULL,           -- full article JSON (current)
    version      INTEGER DEFAULT 1,
    created_at   TEXT DEFAULT (datetime('now')),
    updated_at   TEXT DEFAULT (datetime('now')),
    UNIQUE(section, slug)
);
CREATE TABLE IF NOT EXISTS wiki_versions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id   INTEGER NOT NULL REFERENCES wiki_articles(id),
    version      INTEGER NOT NULL,
    body_json    TEXT NOT NULL,           -- snapshot BEFORE this update
    change_note  TEXT DEFAULT '',
    created_at   TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_wiki_sec_slug ON wiki_articles(section, slug);
CREATE INDEX IF NOT EXISTS idx_wiki_ver_art  ON wiki_versions(article_id);
"""

_SECTION_ORDER = [
    "lead", "taxonomy_phylogeny", "morphology", "distribution_habitat",
    "ecology_behaviour", "conservation", "specimen_records",
    "occurrences", "provenance",
]

# ─────────────────────────────────────────────────────────────────────────────
#  WIKI ARCHITECT PROMPT
# ─────────────────────────────────────────────────────────────────────────────
_WIKI_ARCHITECT_SYSTEM = """\
You are a Professional Taxonomist and Wiki Editor specialising in marine
invertebrates, coastal ecology, and Indian Ocean biodiversity.

Your task: given (a) the CURRENT wiki article JSON and (b) NEW source text
(a PDF extract / chunk), produce an UPDATED article JSON that:

1. NEVER deletes existing valid data — only appends or refines.
2. Resolves conflicts by listing BOTH sources with inline citations, e.g.
   "Bhave (2011) reports 5 m; Smith (2024) reports 12 m."
3. Fills in blank fields (authority, taxon rank, synonyms, depth, etc.)
   when the new text provides them.
4. Appends new localities, vouchers, and ecological notes.
5. Updates the "sections" dict — each key maps to a markdown string.
6. Uses italics (*Name*) for binomial nomenclature.
7. Uses **bold** for key technical terms.
8. Respects Wikipedia-style neutral tone.

Return ONLY a valid JSON object — no prose, no markdown fences.
The JSON must follow EXACTLY the schema of the input CURRENT article.
"""

_WIKI_ARCHITECT_USER = """\
CURRENT ARTICLE JSON:
{current_json}

NEW SOURCE TEXT (chunk from paper):
{new_text}

PAPER CITATION:
{citation}

Update the article using the integration rules above.
Pay special attention to:
- authority / nameAccordingTo (fill if blank)
- taxonRank (fill if blank)
- sections.lead — expand with new traits or habitat info
- sections.morphology — add diagnostic characters if present
- sections.distribution_habitat — add new localities / depth
- sections.conservation — add IUCN / legal protection if mentioned
- depth_conflicts / size_conflicts — list if sources disagree
Return only the updated JSON object.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  BLANK ARTICLE TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

def _blank_species_article(sp_name: str) -> dict:
    return {
        "title": sp_name,
        "type": "species",
        "version": 1,
        # ── Taxobox fields ──────────────────────────────────────────────────
        "kingdom":          "",
        "phylum":           "",
        "class_":           "",
        "order_":           "",
        "family_":          "",
        "genus":            "",
        "species_epithet":  "",
        "authority":        "",      # e.g. "Bergh, 1888"
        "taxonRank":        "species",
        "taxonomicStatus":  "unverified",
        "wormsID":          "",
        "gbifID":           "",
        "iucnStatus":       "",      # "LC" | "VU" | …
        "iucnURL":          "",
        "synonyms":         [],      # [{"name": "...", "authority": "...", "source": ""}]
        # ── Morphology / ecology ────────────────────────────────────────────
        "coloration_life":      "",
        "coloration_preserved": "",
        "body_length_mm":       {"min": None, "max": None, "mean": None},
        "body_width_mm":        {"min": None, "max": None, "mean": None},
        "radular_formula":      "",
        "diagnostic_characters": [],
        "diet":                 [],
        "depth_zone":           "",   # intertidal | subtidal | pelagic …
        "depth_range_raw":      [],   # verbatim strings, each with source
        "substrate":            [],
        # ── Type locality ────────────────────────────────────────────────────
        "type_locality": {"verbatim": "", "latitude": None, "longitude": None, "source": ""},
        # ── Occurrence points ─────────────────────────────────────────────────
        "occurrence_points": [],      # [{lat, lon, locality, depth_m, source, occurrenceType}]
        "habitats":          [],
        "voucher_specimens": [],      # VoucherSpecimen dicts
        "collectors":        [],
        # ── Conflicts ─────────────────────────────────────────────────────────
        "depth_conflicts":  [],       # [{"sources": ["A: 5m", "B: 12m"]}]
        "size_conflicts":   [],
        # ── Free-text wiki sections (markdown) ────────────────────────────────
        "sections": {
            "lead":                   "",
            "taxonomy_phylogeny":     "",
            "morphology":             "",
            "distribution_habitat":   "",
            "ecology_behaviour":      "",
            "conservation":           "",
            "specimen_records":       "",
        },
        # ── Provenance ────────────────────────────────────────────────────────
        "provenance":   [],    # [{citation, date, chunk_hash}]
        "created_at":   datetime.now().isoformat(),
        "updated_at":   datetime.now().isoformat(),
    }


def _blank_locality_article(locality: str) -> dict:
    return {
        "title": locality,
        "type": "locality",
        "version": 1,
        "decimalLatitude":  None,
        "decimalLongitude": None,
        "species_checklist": [],
        "habitat_types":     [],
        "sections": {"overview": ""},
        "provenance": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class BioTraceWikiUnified:
    """
    Unified, versioned, LLM-enhanceable wiki for BioTrace v5.5.

    Drop-in replacement for BioTraceWiki.  Additional capabilities:
    • SQLite versioning (wiki_versions table) with rollback support
    • section-level LLM enhancement via _enhance_with_llm()
    • Wikipedia-style HTML/CSS rendering via render_unified_page()
    • Taxobox Authority auto-populated from occurrences / WoRMS data
    • Status badges from taxonomicStatus + validationStatus
    """

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(self, root_dir: str, css_path: Optional[str] = None):
        self.root    = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.root / "wiki_unified.db")
        self.css_path = css_path
        self._init_db()

    def _init_db(self):
        con = sqlite3.connect(self.db_path)
        con.executescript(_WIKI_SCHEMA)
        con.commit()
        con.close()

    # ── Slug ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _slug(text: str) -> str:
        text = str(text).lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        return re.sub(r"[\s_]+", "-", text)[:120]

    # ── Read / Write ──────────────────────────────────────────────────────────

    def _read(self, section: str, slug: str) -> Optional[dict]:
        con = sqlite3.connect(self.db_path)
        row = con.execute(
            "SELECT id, body_json, version FROM wiki_articles WHERE section=? AND slug=?",
            (section, slug),
        ).fetchone()
        con.close()
        if not row:
            return None
        try:
            art = json.loads(row[1])
            art["_db_id"]  = row[0]
            art["version"] = row[2]
            return art
        except Exception:
            return None

    def _write(self, section: str, slug: str, title: str, art: dict,
               change_note: str = "") -> int:
        """Upsert article; snapshot old version to wiki_versions before overwriting."""
        art["updated_at"] = datetime.now().isoformat()
        body_json = json.dumps(art, ensure_ascii=False)
        con = sqlite3.connect(self.db_path)
        existing = con.execute(
            "SELECT id, body_json, version FROM wiki_articles WHERE section=? AND slug=?",
            (section, slug),
        ).fetchone()
        if existing:
            art_id, old_json, old_ver = existing
            # Snapshot old version
            con.execute(
                "INSERT INTO wiki_versions (article_id, version, body_json, change_note) "
                "VALUES (?,?,?,?)",
                (art_id, old_ver, old_json, change_note),
            )
            new_ver = old_ver + 1
            art["version"] = new_ver
            body_json = json.dumps(art, ensure_ascii=False)
            con.execute(
                "UPDATE wiki_articles SET body_json=?, version=?, updated_at=datetime('now') "
                "WHERE id=?",
                (body_json, new_ver, art_id),
            )
        else:
            art["version"] = 1
            body_json = json.dumps(art, ensure_ascii=False)
            con.execute(
                "INSERT INTO wiki_articles (section, slug, title, body_json, version) "
                "VALUES (?,?,?,?,1)",
                (section, slug, title, body_json),
            )
            art_id = con.execute(
                "SELECT id FROM wiki_articles WHERE section=? AND slug=?",
                (section, slug),
            ).fetchone()[0]
        con.commit()
        con.close()
        return art_id

    # ── Version history & rollback ────────────────────────────────────────────

    def list_versions(self, section: str, sp_name: str) -> list[dict]:
        """Return list of historical snapshots for an article (newest first)."""
        slug = self._slug(sp_name)
        con  = sqlite3.connect(self.db_path)
        rows = con.execute(
            """SELECT wv.version, wv.change_note, wv.created_at
               FROM wiki_versions wv
               JOIN wiki_articles wa ON wa.id = wv.article_id
               WHERE wa.section=? AND wa.slug=?
               ORDER BY wv.version DESC""",
            (section, slug),
        ).fetchall()
        con.close()
        return [{"version": r[0], "note": r[1], "date": r[2]} for r in rows]

    def rollback(self, section: str, sp_name: str, to_version: int) -> bool:
        """Restore article body from a historical snapshot."""
        slug = self._slug(sp_name)
        con  = sqlite3.connect(self.db_path)
        art_row = con.execute(
            "SELECT id, version FROM wiki_articles WHERE section=? AND slug=?",
            (section, slug),
        ).fetchone()
        if not art_row:
            con.close()
            return False
        art_id, cur_ver = art_row
        snap = con.execute(
            "SELECT body_json FROM wiki_versions WHERE article_id=? AND version=?",
            (art_id, to_version),
        ).fetchone()
        if not snap:
            con.close()
            return False
        # Snapshot current before rollback
        cur_body = con.execute(
            "SELECT body_json FROM wiki_articles WHERE id=?", (art_id,)
        ).fetchone()[0]
        con.execute(
            "INSERT INTO wiki_versions (article_id, version, body_json, change_note) "
            "VALUES (?,?,?,?)",
            (art_id, cur_ver, cur_body, f"auto-snapshot before rollback to v{to_version}"),
        )
        new_ver = cur_ver + 1
        con.execute(
            "UPDATE wiki_articles SET body_json=?, version=?, updated_at=datetime('now') "
            "WHERE id=?",
            (snap[0], new_ver, art_id),
        )
        con.commit()
        con.close()
        logger.info("[Wiki] Rolled back %s/%s to v%d (now v%d)", section, slug,
                    to_version, new_ver)
        return True

    # ── Public article accessors ───────────────────────────────────────────────

    def get_species_article(self, sp_name: str) -> Optional[dict]:
        return self._read("species", self._slug(sp_name))

    def list_species(self) -> list[str]:
        con = sqlite3.connect(self.db_path)
        rows = con.execute(
            "SELECT title FROM wiki_articles WHERE section='species' ORDER BY title"
        ).fetchall()
        con.close()
        return [r[0] for r in rows]

    def list_localities(self) -> list[str]:
        con = sqlite3.connect(self.db_path)
        rows = con.execute(
            "SELECT title FROM wiki_articles WHERE section='locality' ORDER BY title"
        ).fetchall()
        con.close()
        return [r[0] for r in rows]

    def index_stats(self) -> dict:
        con = sqlite3.connect(self.db_path)
        rows = con.execute(
            "SELECT section, COUNT(*) FROM wiki_articles GROUP BY section"
        ).fetchall()
        con.close()
        by_sec = {r[0]: r[1] for r in rows}
        return {
            "total_articles": sum(by_sec.values()),
            "by_section": by_sec,
        }

    def build_wiki_context(self, query: str, top_k: int = 5) -> str:
        """Return a text context string for GraphRAG queries."""
        con = sqlite3.connect(self.db_path)
        rows = con.execute(
            "SELECT title, body_json FROM wiki_articles WHERE section='species' ORDER BY updated_at DESC LIMIT ?",
            (top_k * 3,),
        ).fetchall()
        con.close()
        parts = []
        for title, body_json in rows[:top_k]:
            try:
                art = json.loads(body_json)
                lead = art.get("sections", {}).get("lead", "")[:400]
                parts.append(f"=={title}==\n{lead}")
            except Exception:
                pass
        return "\n\n".join(parts)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def update_from_occurrences(
        self,
        occurrences:        list[dict],
        citation:           str = "",
        llm_fn:             Optional[Callable] = None,
        update_narratives:  bool = False,
        chunk_text:         str = "",
        extra_facts_map:    Optional[dict] = None,
    ) -> dict:
        """
        Ingest a batch of occurrence dicts.
        If update_narratives=True and llm_fn is provided, each unique species
        article is enhanced with the Wiki Architect prompt using chunk_text.
        """
        counts = {"species": 0, "locality": 0}
        extra_facts_map = extra_facts_map or {}
        enhanced_species: set[str] = set()

        for occ in occurrences:
            sp_name = occ.get("validName") or occ.get("recordedName", "")
            if not sp_name:
                continue
            self._update_species_article(sp_name, occ, citation,
                                         extra_facts=extra_facts_map.get(sp_name, {}))
            counts["species"] += 1
            enhanced_species.add(sp_name)

            loc = occ.get("verbatimLocality", "")
            if loc and loc.lower() not in ("not reported", "unknown", ""):
                self._update_locality_article(loc, sp_name, occ, citation)
                counts["locality"] += 1

        # LLM enhancement pass — one per unique species, only if new chunk available
        if update_narratives and llm_fn and chunk_text.strip():
            for sp_name in enhanced_species:
                try:
                    self._enhance_with_llm(sp_name, chunk_text, citation, llm_fn)
                    counts.setdefault("llm_enhanced", 0)
                    counts["llm_enhanced"] = counts.get("llm_enhanced", 0) + 1
                except Exception as exc:
                    logger.warning("[Wiki] LLM enhance failed for %s: %s", sp_name, exc)

        return counts

    def _update_species_article(
        self, sp_name: str, occ: dict, citation: str, extra_facts: dict = None
    ):
        slug = self._slug(sp_name)
        art  = self._read("species", slug) or _blank_species_article(sp_name)

        # ── Taxonomy fields from occurrence ────────────────────────────────
        def _fill(art_key: str, *occ_keys):
            if not art.get(art_key):
                for k in occ_keys:
                    v = occ.get(k, "")
                    if v and str(v).strip():
                        art[art_key] = str(v).strip()
                        break

        _fill("phylum",         "phylum",  "Phylum")
        _fill("class_",         "class_",  "Class")
        _fill("order_",         "order_",  "Order")
        _fill("family_",        "family_", "Family")
        _fill("wormsID",        "wormsID")
        _fill("gbifID",         "gbifID")
        _fill("taxonRank",      "taxonRank")
        _fill("taxonomicStatus","taxonomicStatus")

        # Authority: prefer "nameAccordingTo" → "authority" occ field
        if not art.get("authority"):
            auth = (occ.get("nameAccordingTo") or occ.get("authority") or
                    occ.get("name_according_to") or "")
            if auth:
                art["authority"] = str(auth).strip()

        # ── IUCN ──────────────────────────────────────────────────────────
        _fill("iucnStatus", "iucnStatus", "iucn_status", "IUCN")

        # ── Occurrence point ───────────────────────────────────────────────
        lat = occ.get("decimalLatitude")
        lon = occ.get("decimalLongitude")
        loc = occ.get("verbatimLocality", "")
        occ_pt = {
            "locality": loc, "latitude": lat, "longitude": lon,
            "depth_m": None, "source": citation,
            "occurrenceType": occ.get("occurrenceType", "Uncertain"),
        }
        # Extract depth from samplingEvent if available
        se = occ.get("samplingEvent") or occ.get("Sampling Event", {})
        if isinstance(se, str):
            try: se = json.loads(se)
            except Exception: se = {}
        if isinstance(se, dict) and se.get("depth_m"):
            try: occ_pt["depth_m"] = float(se["depth_m"])
            except Exception: pass

        # Deduplicate by loc+citation hash
        _hash = hashlib.md5(f"{loc}_{citation}".encode()).hexdigest()[:8]
        existing_hashes = [
            hashlib.md5(f"{p['locality']}_{p['source']}".encode()).hexdigest()[:8]
            for p in art["occurrence_points"]
        ]
        if _hash not in existing_hashes and (lat or loc):
            art["occurrence_points"].append(occ_pt)

        # ── Habitat ────────────────────────────────────────────────────────
        hab = occ.get("habitat", "")
        if hab and hab not in art["habitats"]:
            art["habitats"].append(hab)

        # ── Extra facts merge ──────────────────────────────────────────────
        if extra_facts:
            self._merge_extra_facts(art, extra_facts)

        # ── Provenance ─────────────────────────────────────────────────────
        prov_entry = {"citation": citation, "date": datetime.now().isoformat()}
        if not any(p["citation"] == citation for p in art["provenance"]):
            art["provenance"].append(prov_entry)

        self._write("species", slug, sp_name, art,
                    change_note=f"occurrence ingest: {citation[:60]}")

    def _update_locality_article(
        self, locality: str, sp_name: str, occ: dict, citation: str
    ):
        slug = self._slug(locality)
        art  = self._read("locality", slug) or _blank_locality_article(locality)

        if sp_name and sp_name not in art["species_checklist"]:
            art["species_checklist"].append(sp_name)

        if not art["decimalLatitude"] and occ.get("decimalLatitude"):
            art["decimalLatitude"]  = occ["decimalLatitude"]
            art["decimalLongitude"] = occ["decimalLongitude"]

        hab = occ.get("habitat", "")
        if hab and hab not in art.get("habitat_types", []):
            art.setdefault("habitat_types", []).append(hab)

        prov = {"citation": citation, "date": datetime.now().isoformat()}
        if not any(p["citation"] == citation for p in art.get("provenance", [])):
            art.setdefault("provenance", []).append(prov)

        self._write("locality", slug, locality, art,
                    change_note=f"checklist update: {sp_name}")

    @staticmethod
    def _merge_extra_facts(art: dict, extra: dict):
        """Non-destructive merge of extra LLM-extracted facts into article."""
        for k, v in extra.items():
            if not v:
                continue
            if k not in art:
                art[k] = v
            elif isinstance(art[k], list) and isinstance(v, list):
                for item in v:
                    if item not in art[k]:
                        art[k].append(item)
            elif isinstance(art[k], dict) and isinstance(v, dict):
                for dk, dv in v.items():
                    if not art[k].get(dk) and dv:
                        art[k][dk] = dv
            elif not art[k] and v:
                art[k] = v

    # ── LLM "Wiki Architect" enhancement ─────────────────────────────────────

    def _enhance_with_llm(
        self, sp_name: str, chunk_text: str, citation: str,
        llm_fn: Callable[[str], str],
    ):
        """
        Send current article + new chunk to LLM Wiki Architect.
        The LLM returns an updated article JSON — merged non-destructively.
        Only fires if this chunk_hash hasn't been processed before.
        """
        slug = self._slug(sp_name)
        art  = self._read("species", slug)
        if not art:
            return

        chunk_hash = hashlib.md5(chunk_text[:2000].encode()).hexdigest()[:12]
        already_done = any(
            p.get("chunk_hash") == chunk_hash for p in art.get("provenance", [])
        )
        if already_done:
            logger.debug("[Wiki] chunk already processed for %s, skipping LLM", sp_name)
            return

        # Strip occurrence_points list to keep prompt small
        art_for_prompt = {
            k: v for k, v in art.items()
            if k not in ("_db_id", "provenance", "occurrence_points")
        }
        art_for_prompt["occurrence_count"] = len(art.get("occurrence_points", []))

        prompt = (
            _WIKI_ARCHITECT_SYSTEM + "\n\n" +
            _WIKI_ARCHITECT_USER.format(
                current_json=json.dumps(art_for_prompt, indent=2)[:6000],
                new_text=chunk_text[:4000],
                citation=citation,
            )
        )

        raw = llm_fn(prompt)
        # Strip markdown fences if present
        raw = re.sub(r"^```+(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r"\s*```+$", "", raw.strip(), flags=re.MULTILINE)

        try:
            updated = json.loads(raw)
        except Exception as exc:
            logger.warning("[Wiki] LLM JSON parse failed for %s: %s", sp_name, exc)
            return

        # Non-destructive field merge back into art
        for k, v in updated.items():
            if k in ("_db_id", "version", "created_at"):
                continue
            if isinstance(v, str) and v.strip():
                if not art.get(k):
                    art[k] = v
                elif k == "sections":
                    pass  # handled below
            elif isinstance(v, list) and v:
                if isinstance(art.get(k), list):
                    self._merge_extra_facts(art, {k: v})
                else:
                    art[k] = v
            elif isinstance(v, dict):
                self._merge_extra_facts(art, {k: v})

        # Section merge — append new content after existing
        updated_secs = updated.get("sections", {})
        for sec_key, new_text in updated_secs.items():
            if not new_text:
                continue
            existing = art.get("sections", {}).get(sec_key, "")
            if not existing:
                art.setdefault("sections", {})[sec_key] = new_text
            elif new_text.strip() and new_text.strip() not in existing:
                # Append new non-duplicate content
                art["sections"][sec_key] = existing.rstrip() + "\n\n" + new_text.strip()

        # Record provenance with chunk hash to prevent reprocessing
        prov_entry = {
            "citation":   citation,
            "date":       datetime.now().isoformat(),
            "chunk_hash": chunk_hash,
            "enhanced":   True,
        }
        art.setdefault("provenance", []).append(prov_entry)

        self._write("species", slug, sp_name, art,
                    change_note=f"LLM-enhanced from chunk {chunk_hash}: {citation[:50]}")
        logger.info("[Wiki] LLM-enhanced %s (chunk %s)", sp_name, chunk_hash)

    # ── Rendering ─────────────────────────────────────────────────────────────

    @staticmethod
    def _iucn_badge_class(status: str) -> str:
        return {
            "LC": "wiki-badge-iucn-lc", "NT": "wiki-badge-iucn-nt",
            "VU": "wiki-badge-iucn-vu", "EN": "wiki-badge-iucn-en",
            "CR": "wiki-badge-iucn-cr", "DD": "wiki-badge-iucn-dd",
        }.get(status.upper() if status else "", "wiki-badge-rank")

#     @staticmethod
#     def _status_badge_class(status: str) -> str:
#         s = (status or "").lower()
#         if s in ("accepted", "verified", "accept"):  return "wiki-badge-verified"
#         if s in ("rejected", "reject"):               return "wiki-badge-rejected"
#         return "wiki-badge-unverified"

#     def render_taxobox_html(self, art: dict) -> str:
#         """Render a Wikipedia-style taxobox as HTML (needs biotrace_wiki.css)."""
#         sp_name   = art.get("title", "")
#         authority = art.get("authority", "")
#         rows = [
#             ("Kingdom",  art.get("kingdom",  "") or "Animalia"),
#             ("Phylum",   art.get("phylum",   "")),
#             ("Class",    art.get("class_",   "")),
#             ("Order",    art.get("order_",   "")),
#             ("Family",   art.get("family_",  "")),
#             ("Genus",    art.get("genus",    "") or (sp_name.split()[0] if sp_name else "")),
#             ("Species",  f"<i>{sp_name}</i>" if sp_name else ""),
#             ("Authority",authority or "<span style='color:#555'>—</span>"),
#         ]
#         row_html = "".join(
#             f"<tr><td>{label}</td><td{'class=\"no-italic\"' if label in ('Kingdom','Authority','Family','Order','Class','Phylum') else ''}>{val}</td></tr>"
#             for label, val in rows if val
#         )
#         wid = art.get("wormsID", "")
#         footer = ""
#         if wid:
#             footer = (f'<div class="wiki-taxobox-footer">'
#                       f'<a href="https://www.marinespecies.org/aphia.php?p=taxdetails&id={wid}" '
#                       f'target="_blank">🔗 WoRMS AphiaID {wid}</a></div>')
#         gbif = art.get("gbifID", "")
#         if gbif:
#             footer += (f'<div class="wiki-taxobox-footer">'
#                        f'<a href="https://www.gbif.org/species/{gbif}" target="_blank">'
#                        f'🔗 GBIF {gbif}</a></div>')

#         return f"""
# <div class="wiki-taxobox">
#   <div class="wiki-taxobox-header">Scientific Classification</div>
#   <div class="wiki-taxobox-species-name"><i>{sp_name}</i></div>
#   <table>{row_html}</table>
#   {footer}
# </div>"""

    @staticmethod
    def _status_badge_class(status: str) -> str:
        s = (status or "").lower()
        if s in ("accepted", "verified", "accept"):  return "wiki-badge-verified"
        if s in ("rejected", "reject"):               return "wiki-badge-rejected"
        return "wiki-badge-unverified"

    def render_taxobox_html(self, art: dict) -> str:
        """Render a Wikipedia-style taxobox as HTML (needs biotrace_wiki.css)."""
        lineage_str = ""
        sp_name   = art.get("title", "")
        authority = art.get("authority", "")
        # 1. Build the lineage string properly formatted with delimiters
        lineage_ranks = [
            art.get("kingdom", "Animalia"), 
            art.get("phylum", ""), 
            art.get("class_", ""), 
            art.get("order_", ""), 
            art.get("family_", "")
        ]
        # Filter out empty strings and join with a clean separator
        lineage_str = " &rsaquo; ".join([r for r in lineage_ranks if r])

        rows = [
            ("Kingdom",  art.get("kingdom",  "") or "Animalia"),
            ("Phylum",   art.get("phylum",   "")),
            ("Class",    art.get("class_",   "")),
            ("Order",    art.get("order_",   "")),
            ("Family",   art.get("family_",  "")),
            ("Genus",    art.get("genus",    "") or (sp_name.split()[0] if sp_name else "")),
            ("Species",  f"<i>{sp_name}</i>" if sp_name else ""),
            ("Authority",authority or "<span style='color:#555'>—</span>"),
        ]
        
        # Added a space before 'class' to ensure proper HTML attributes
        row_html = "".join(
            f"<tr><td>{label}</td><td{' class=\"no-italic\"' if label in ('Kingdom','Authority','Family','Order','Class','Phylum') else ''}>{val}</td></tr>"
            for label, val in rows if val
        )
        
        wid = art.get("wormsID", "")
        footer = ""
        if wid:
            footer = (f'<div class="wiki-taxobox-footer">'
                      f'<a href="https://www.marinespecies.org/aphia.php?p=taxdetails&id={wid}" '
                      f'target="_blank">🔗 WoRMS AphiaID {wid}</a></div>')
        gbif = art.get("gbifID", "")
        if gbif:
            footer += (f'<div class="wiki-taxobox-footer">'
                       f'<a href="https://www.gbif.org/species/{gbif}" target="_blank">'
                       f'🔗 GBIF {gbif}</a></div>')

        # 2. Inject the lineage_str into the HTML output
        return f"""
        <div class="wiki-taxobox">
        <div class="wiki-taxobox-header">Scientific Classification</div>
        <div class="wiki-taxobox-species-name"><i>{sp_name}</i></div>
        <div class="lineage" style="text-align: center; font-size: 0.85em; padding: 4px 10px; color: #ddd; word-wrap: break-word;">
            {lineage_str}
        </div>
        <table>{row_html}</table>
        {footer}
        </div>"""

    def render_badge_row_html(self, art: dict) -> str:
        """Render verification status, rank, and IUCN badges."""
        status    = art.get("taxonomicStatus", "unverified")
        rank      = art.get("taxonRank", "species")
        iucn      = art.get("iucnStatus", "")
        auth      = art.get("authority", "")

        badges = [
            f'<span class="wiki-badge {self._status_badge_class(status)}">● {status}</span>',
            f'<span class="wiki-badge wiki-badge-rank">Rank: {rank}</span>',
        ]
        if auth:
            badges.append(f'<span class="wiki-badge wiki-badge-rank">Authority: {auth}</span>')
        if iucn:
            badges.append(
                f'<span class="wiki-badge {self._iucn_badge_class(iucn)}">IUCN {iucn}</span>'
            )
        ver = art.get("version", 1)
        updated = art.get("updated_at", "")[:10]
        badges.append(
            f'<span class="wiki-version-chip">v{ver} · {updated}</span>'
        )
        return f'<div class="wiki-badge-row">{"".join(badges)}</div>'

    def render_unified_page(self, sp_name: str) -> str:
        """
        Render a full Wikipedia-style HTML page for a species.
        Embeds biotrace_wiki.css inline.
        """
        art = self.get_species_article(sp_name)
        if not art:
            return f"<p>No wiki article found for <i>{sp_name}</i>.</p>"

        css  = _load_css(self.css_path)
        secs = art.get("sections", {})
        sp   = art.get("title", sp_name)

        lead_text = secs.get("lead", "") or "No lead section yet — enhance with an LLM pass."

        # Occurrence table
        occ_rows = ""
        for pt in art.get("occurrence_points", [])[:40]:
            lat = pt.get("latitude")
            lon = pt.get("longitude")
            coord = f"{lat:.4f}, {lon:.4f}" if lat and lon else "—"
            ot  = pt.get("occurrenceType", "Uncertain")
            ot_cls = {"Primary": "occ-primary", "Secondary": "occ-secondary"}.get(ot, "occ-uncertain")
            src = str(pt.get("source",""))[:60]
            occ_rows += (
                f"<tr><td>{pt.get('locality','—')}</td>"
                f"<td class='{ot_cls}'>{ot}</td>"
                f"<td>{pt.get('depth_m') or '—'}</td>"
                f"<td>{coord}</td>"
                f"<td title='{src}'>{src[:40]}{'…' if len(src)>40 else ''}</td></tr>"
            )

        occ_table = ""
        if occ_rows:
            occ_table = f"""
            <h2 class="wiki-section-h2">🗺️ Documented Occurrences</h2>
            <table class="wiki-occ-table">
            <thead><tr>
                <th>Locality</th><th>Type</th><th>Depth (m)</th>
                <th>Coordinates</th><th>Source</th>
            </tr></thead>
            <tbody>{occ_rows}</tbody>
            </table>"""

        # Diagnostic characters
        diag_html = ""
        diags = art.get("diagnostic_characters", [])
        if diags:
            items = "".join(f"<li>{d}</li>" for d in diags[:10])
            diag_html = f'<ul class="wiki-diag-list">{items}</ul>'

        # Conflict notes
        conflicts_html = ""
        for cf in art.get("depth_conflicts", []) + art.get("size_conflicts", []):
            note = "; ".join(cf.get("sources", []))
            if note:
                conflicts_html += f'<div class="wiki-conflict">{note}</div>'

        # Provenance
        prov_items = "".join(
            f"<li>{p.get('citation','')[:80]} <em>({p.get('date','')[:10]})"
            f"{'✨' if p.get('enhanced') else ''}</em></li>"
            for p in art.get("provenance", [])
        )
        prov_html = f'<div class="wiki-references"><ol>{prov_items}</ol></div>' if prov_items else ""

        # Section rendering helper
        def sec(key: str, heading: str, icon: str = "") -> str:
            txt = secs.get(key, "")
            if not txt:
                return ""
            return f'<h2 class="wiki-section-h2">{icon} {heading}</h2><p>{txt}</p>'

        body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{sp} — BioTrace Wiki</title>
<style>{css}</style>
</head>
<body>
<div class="wiki-page">

  <!-- Title -->
  <h1 class="wiki-title"><i>{sp}</i></h1>
  <p class="wiki-subtitle">BioTrace Living Knowledge Base · Auto-generated from literature</p>

  <!-- Badges -->
  {self.render_badge_row_html(art)}

  <!-- Lead + Taxobox -->
  <div class="wiki-lead">
    {self.render_taxobox_html(art)}
    <p>{lead_text}</p>
    {diag_html}
    {conflicts_html}
  </div>

  <!-- Sections -->
  {sec('taxonomy_phylogeny',   'Taxonomy & Phylogeny',        '🔬')}
  {sec('morphology',           'Anatomy & Morphology',        '🔭')}
  {sec('distribution_habitat', 'Distribution & Habitat',      '🌍')}
  {sec('ecology_behaviour',    'Ecology & Behaviour',         '🐟')}
  {sec('conservation',         'Conservation Status',         '🛡️')}
  {sec('specimen_records',     'Specimen Records',            '🏛️')}

  <!-- Occurrences table -->
  {occ_table}

  <!-- References -->
  <h2 class="wiki-section-h2">📚 References</h2>
  {prov_html}

</div>
</body>
</html>"""
        return body

    def render_species_markdown(self, sp_name: str) -> str:
        """
        Backward-compatible markdown render (used by biotrace_v53.py Tab 7).
        Returns Wikipedia-style markdown with populated Authority & status.
        """
        art = self.get_species_article(sp_name)
        if not art:
            return f"*Article not found for {sp_name}.*"

        sp       = art.get("title", sp_name)
        auth     = art.get("authority", "—")
        phylum   = art.get("phylum",  "")
        class_   = art.get("class_",  "")
        order_   = art.get("order_",  "")
        family_  = art.get("family_", "")
        status   = art.get("taxonomicStatus", "unverified")
        rank     = art.get("taxonRank", "species")
        iucn     = art.get("iucnStatus", "")
        wid      = art.get("wormsID", "")

        lines = [
            f"# *{sp}*",
            "",
            f"| **Family:** {family_} | **Order:** {order_} | **Phylum:** {phylum} |",
            f"|:---|:---|:---|",
            f"| **Status:** {status} | **Rank:** {rank} | **Authority:** {auth} |",
            "",
        ]
        if iucn:
            lines.append(f"> **IUCN Status:** {iucn}")
            lines.append("")
        if wid:
            lines.append(
                f"[🔗 WoRMS AphiaID {wid}](https://www.marinespecies.org/aphia.php?p=taxdetails&id={wid})"
            )
            lines.append("")

        secs = art.get("sections", {})
        for key, heading in [
            ("lead",                 "## Overview"),
            ("taxonomy_phylogeny",   "## Taxonomy & Phylogeny"),
            ("morphology",           "## Anatomy & Morphology"),
            ("distribution_habitat", "## Distribution & Habitat"),
            ("ecology_behaviour",    "## Ecology & Behaviour"),
            ("conservation",         "## Conservation"),
            ("specimen_records",     "## Specimen Records"),
        ]:
            txt = secs.get(key, "")
            if txt:
                lines.extend([heading, "", txt, ""])

        # Occurrence summary
        occ_pts = art.get("occurrence_points", [])
        if occ_pts:
            lines.append("## Documented Occurrences")
            lines.append("")
            for pt in occ_pts[:20]:
                loc  = pt.get("locality", "—")
                ot   = pt.get("occurrenceType", "?")
                src  = str(pt.get("source", ""))[:60]
                dep  = pt.get("depth_m")
                dep_s = f" · {dep} m depth" if dep else ""
                lines.append(f"- **{loc}** ({ot}{dep_s}) — _{src}_")
            lines.append("")

        # Provenance
        provs = art.get("provenance", [])
        if provs:
            lines.append("## References")
            for i, p in enumerate(provs, 1):
                enhanced = " ✨" if p.get("enhanced") else ""
                lines.append(f"{i}. {p.get('citation','')} ({p.get('date','')[:10]}){enhanced}")

        return "\n".join(lines)

    # ── Streamlit UI ──────────────────────────────────────────────────────────

    def render_streamlit_tab(
        self,
        provider:  str = "",
        model_sel: str = "",
        api_key:   str = "",
        ollama_url:str = "http://localhost:11434",
        meta_db:   str = "",
        call_llm_fn: Optional[Callable] = None,
    ):
        """
        Full Tab 7 replacement — single unified wiki UI.
        Call inside  `with tabs[6]:` in biotrace_v53.py.
        """
        if not _ST:
            return

        inject_css_streamlit(self.css_path)

        st.subheader("📖 Wiki — Living Knowledge Base")
        st.caption("Wikipedia-style · LLM-enhanced · Git-versioned · Ever-evolving")

        idx  = self.index_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Articles", idx["total_articles"])
        c2.metric("Species",  idx["by_section"].get("species",  0))
        c3.metric("Localities",idx["by_section"].get("locality", 0))
        ver_count = self._count_versions()
        c4.metric("Saved Versions", ver_count)

        st.divider()

        sp_list = self.list_species()
        if not sp_list:
            st.info("No wiki articles yet. Run an extraction to populate.")
            return

        # Strip authority suffixes for display
        def _strip_auth(name: str) -> str:
            return re.sub(
                r"\s+[A-Z][A-Za-z\-'']+(?:\s+(?:and|&|et)\s+[A-Z][A-Za-z\-'']+)?"
                r"[,.]?\s*\d{4}.*$", "", name
            ).strip()

        display_map = {_strip_auth(s): s for s in sp_list}
        selected_display = st.selectbox(
            "View Species Article:", sorted(display_map.keys()),
            key="wiki_unified_sp_sel",
        )
        selected_sp = display_map.get(selected_display, selected_display)

        if not selected_sp:
            return

        art = self.get_species_article(selected_sp) or {}

        # ── View mode tabs ─────────────────────────────────────────────────
        view_tab, raw_tab, ver_tab = st.tabs(
            ["📄 Wiki Page", "🗂️ Raw Article", "📜 Version History"]
        )

        with view_tab:
            # Render full HTML wiki page in iframe-like component
            html_page = self.render_unified_page(selected_sp)
            st.components.v1.html(html_page, height=800, scrolling=True)

            # Map (native Streamlit, outside iframe)
            import pandas as pd
            pts = [
                {"lat": p["latitude"], "lon": p["longitude"],
                 "name": p.get("locality", "")}
                for p in art.get("occurrence_points", [])
                if p.get("latitude") and p.get("longitude")
            ]
            if pts:
                st.markdown("#### 🗺️ Occurrence Map")
                mdf = pd.DataFrame(pts)
                st.map(mdf[["lat","lon"]], zoom=4)

        with raw_tab:
            st.json(art, expanded=False)

            # LLM enhancement panel
            if call_llm_fn:
                st.markdown("#### 🤖 LLM Wiki Enhancement")
                enhance_text = st.text_area(
                    "Paste new PDF chunk / text to enhance this article:",
                    height=180,
                    key="wiki_enhance_text",
                )
                enhance_cite = st.text_input(
                    "Citation for this chunk:", key="wiki_enhance_cite"
                )
                if st.button("✨ Enhance Article", key="wiki_enhance_btn"):
                    if enhance_text.strip():
                        with st.spinner("Wiki Architect is enhancing…"):
                            try:
                                self._enhance_with_llm(
                                    selected_sp, enhance_text, enhance_cite, call_llm_fn
                                )
                                st.success("Article enhanced and versioned ✅")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Enhancement failed: {exc}")
                    else:
                        st.warning("Paste some text first.")

        with ver_tab:
            versions = self.list_versions("species", selected_sp)
            if not versions:
                st.info("No previous versions — article will be versioned after next update.")
            else:
                st.write(f"**{len(versions)} saved versions** (newest first):")
                import pandas as pd
                ver_df = pd.DataFrame(versions)
                st.dataframe(ver_df, use_container_width=True, hide_index=True)

                rollback_ver = st.number_input(
                    "Rollback to version:", min_value=1,
                    max_value=max(v["version"] for v in versions),
                    step=1, key="wiki_rollback_ver",
                )
                if st.button("⏪ Rollback", key="wiki_rollback_btn"):
                    ok = self.rollback("species", selected_sp, rollback_ver)
                    if ok:
                        st.success(f"Rolled back to v{rollback_ver} ✅")
                        st.rerun()
                    else:
                        st.error("Rollback failed — version not found.")

        st.divider()

        # ── Locality checklist ─────────────────────────────────────────────
        with st.expander("📍 Locality Species Checklist"):
            loc_list = self.list_localities()
            if loc_list:
                sel_loc = st.selectbox("Locality:", loc_list, key="wiki_loc_unified")
                if sel_loc:
                    loc_art = self._read("locality", self._slug(sel_loc)) or {}
                    sps = loc_art.get("species_checklist", [])
                    st.write(f"**{len(sps)} species at {sel_loc}:**")
                    cols = st.columns(2)
                    for i, sp in enumerate(sps):
                        cols[i % 2].markdown(f"• *{_strip_auth(sp)}*")
                    lat = loc_art.get("decimalLatitude")
                    lon = loc_art.get("decimalLongitude")
                    if lat and lon:
                        import pandas as pd
                        st.map(pd.DataFrame([{"lat": lat, "lon": lon}]), zoom=7)
            else:
                st.info("No locality articles yet.")

    def _count_versions(self) -> int:
        try:
            con = sqlite3.connect(self.db_path)
            n   = con.execute("SELECT COUNT(*) FROM wiki_versions").fetchone()[0]
            con.close()
            return n
        except Exception:
            return 0

    # ── Backward-compatibility shims ──────────────────────────────────────────

    def _load_article(self, section: str, slug: str) -> Optional[dict]:
        """Shim for old code that calls wiki._load_article()."""
        return self._read(section, slug)

    def list_species_articles(self) -> list[str]:
        return self.list_species()

    @property
    def _slug_fn(self):
        return self._slug
