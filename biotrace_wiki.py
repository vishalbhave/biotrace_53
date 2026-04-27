"""
biotrace_wiki.py  —  BioTrace v5.1
────────────────────────────────────────────────────────────────────────────
LLM-Wiki Engine for Marine Biodiversity Knowledge Management.

v5.1 enhancements:
  • GNV-enriched species profiles (vernacularNames, full taxonomy ranks,
    colID, gbifID, eolID, classificationPath, outlink)
  • render_species_markdown includes vernacular names table + external links
  • update_species_article now merges gnv_profile from GNVEnrichedVerifier

Implements Andrej Karpathy's "LLM-Wiki" vision (April 2026):
  "Instead of RAG on raw documents, AI agents build, structure, and maintain
   a persistent wiki from raw documents."

This module treats every paper as a document that WRITES to a wiki, not just
one that is QUERIED. Over time the wiki becomes the authoritative structured
knowledge base — richer than any individual paper.

Wiki structure (hierarchical):
  /wiki/
    species/<canonical_name>.json   — species article with all known facts
    locality/<slugified_name>.json  — locality article with species checklists
    habitat/<type>.json             — habitat article with ecological context
    taxonomy/<family>.json          — family-level taxonomic overview
    papers/<citation_hash>.json     — paper summary + extracted records
    index.json                      — master index of all articles

Each wiki article:
  • Has a "facts" section (structured data — confidence-weighted)
  • Has a "narrative" section (LLM-generated plain English summary)
  • Has a "provenance" section (which papers contributed)
  • Has a "last_updated" timestamp and "version" counter
  • Can be queried semantically via BioTraceMemoryBank.recall()

Integration with BioTrace pipeline:
  wiki = BioTraceWiki("biodiversity_data/wiki")
  wiki.update_from_occurrences(occurrences, citation, llm_fn)
  article = wiki.get_species_article("Acanthurus triostegus")
  md      = wiki.render_species_markdown("Acanthurus triostegus")

Usage as GraphRAG context:
  context = wiki.build_wiki_context(query, top_k=5)
  # → injects article summaries into LLM prompt
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("biotrace.wiki")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", str(text).lower().strip())[:80]


def _cite_hash(citation: str) -> str:
    return hashlib.sha1(citation.encode()).hexdigest()[:12]


def _now() -> str:
    return datetime.utcnow().isoformat()


# ─────────────────────────────────────────────────────────────────────────────
#  ARTICLE TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────
def _blank_species_article(name: str) -> dict:
    return {
        "type":       "species",
        "title":      name,
        "canonical":  name,
        "version":    0,
        "created_at": _now(),
        "last_updated": _now(),
        "facts": {
            "recorded_names":   [],
            "worms_id":         "",
            "itis_id":          "",
            "col_id":           "",
            "gbif_id":          "",
            "eol_id":           "",
            "outlink":          "",
            "taxon_rank":       "species",
            "taxonomic_status": "unverified",
            "kingdom":          "",
            "phylum":           "",
            "subphylum":        "",
            "class_":           "",
            "subclass":         "",
            "superorder":       "",
            "order_":           "",
            "suborder":         "",
            "superfamily":      "",
            "family_":          "",
            "subfamily":        "",
            "tribe":            "",
            "genus_":           "",
            "classification_path": "",
            "name_according_to": "",
            "vernacular_names":  [],   # [{name, language}]
            "habitats":         [],
            "localities":       [],
            "depth_range_m":    [],
            "sampling_methods": [],
            "occurrence_types": [],
        },
        "narrative": "",
        "provenance": [],          # list of {citation, date, n_records}
        "occurrences": [],         # list of condensed occurrence dicts
    }


def _blank_locality_article(name: str) -> dict:
    return {
        "type":         "locality",
        "title":        name,
        "slug":         _slug(name),
        "version":      0,
        "created_at":   _now(),
        "last_updated": _now(),
        "facts": {
            "latitude":   None,
            "longitude":  None,
            "state":      "",
            "country":    "India",
            "habitats":   [],
            "species_checklist": [],
        },
        "narrative": "",
        "provenance": [],
    }


def _blank_habitat_article(name: str) -> dict:
    return {
        "type":         "habitat",
        "title":        name,
        "slug":         _slug(name),
        "version":      0,
        "created_at":   _now(),
        "last_updated": _now(),
        "facts": {
            "species_checklist": [],
            "families":         [],
            "localities":       [],
        },
        "narrative": "",
        "provenance": [],
    }


def _blank_paper_article(citation: str) -> dict:
    return {
        "type":         "paper",
        "title":        citation,
        "cite_hash":    _cite_hash(citation),
        "version":      0,
        "created_at":   _now(),
        "last_updated": _now(),
        "facts": {
            "species_recorded": [],
            "localities_studied": [],
            "habitats_covered":   [],
            "n_records":          0,
            "occurrence_types":   [],
        },
        "narrative": "",
        "abstract_summary": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────
class BioTraceWiki:
    """
    LLM-Wiki engine: builds and maintains a persistent structured wiki
    from marine biodiversity occurrence records.

    File layout:
        wiki_root/
          index.json
          species/<slug>.json
          locality/<slug>.json
          habitat/<slug>.json
          taxonomy/<slug>.json
          papers/<cite_hash>.json
    """

    SECTIONS = ("species", "locality", "habitat", "taxonomy", "papers")

    def __init__(self, wiki_root: str = "biodiversity_data/wiki"):
        self.root = Path(wiki_root)
        self._init_dirs()
        logger.info("[Wiki] Initialised at %s", self.root)

    def _init_dirs(self):
        for section in self.SECTIONS:
            (self.root / section).mkdir(parents=True, exist_ok=True)
        index_path = self.root / "index.json"
        if not index_path.exists():
            self._write_json(index_path, {
                "created_at": _now(),
                "sections": {s: [] for s in self.SECTIONS},
                "total_articles": 0,
            })

    # ── JSON I/O ──────────────────────────────────────────────────────────────
    def _write_json(self, path: Path, data: dict):
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _read_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _article_path(self, section: str, slug: str) -> Path:
        return self.root / section / f"{slug}.json"

    def _load_article(self, section: str, slug: str) -> Optional[dict]:
        p = self._article_path(section, slug)
        return self._read_json(p) if p.exists() else None

    def _save_article(self, section: str, slug: str, article: dict):
        article["version"]      = article.get("version", 0) + 1
        article["last_updated"] = _now()
        self._write_json(self._article_path(section, slug), article)
        self._update_index(section, slug, article.get("title", slug))

    def _update_index(self, section: str, slug: str, title: str):
        index_path = self.root / "index.json"
        idx = self._read_json(index_path)
        if slug not in idx.get("sections", {}).get(section, []):
            idx.setdefault("sections", {}).setdefault(section, []).append(slug)
            idx["total_articles"] = sum(len(v) for v in idx["sections"].values())
        self._write_json(index_path, idx)

    # ── Species article ───────────────────────────────────────────────────────
    def get_species_article(self, name: str) -> Optional[dict]:
        return self._load_article("species", _slug(name))

    def update_species_article(
        self,
        occ: dict,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> dict:
        """
        Update (or create) the wiki article for a species from one occurrence.
        """
        sp_name = (
            occ.get("validName") or occ.get("recordedName") or
            occ.get("Valid Name") or occ.get("Recorded Name","")
        ).strip()
        if not sp_name:
            return {}

        slug    = _slug(sp_name)
        article = self._load_article("species", slug) or _blank_species_article(sp_name)

        f = article["facts"]

        # Merge recorded names
        rec_name = (occ.get("recordedName") or occ.get("Recorded Name","")).strip()
        if rec_name and rec_name not in f["recorded_names"]:
            f["recorded_names"].append(rec_name)

        # Authority fields (WoRMS wins)
        for field, key in [
            ("worms_id","wormsID"), ("itis_id","itisID"),
            ("col_id","colID"), ("gbif_id","gbifID"), ("eol_id","eolID"),
            ("outlink","outlink"),
            ("phylum","phylum"), ("class_","class_"),
            ("order_","order_"), ("family_","family_"),
            ("kingdom","kingdom"), ("subclass","subclass"),
            ("superorder","superorder"), ("suborder","suborder"),
            ("superfamily","superfamily"), ("subfamily","subfamily"),
            ("tribe","tribe"), ("genus_","genus_"),
            ("taxon_rank","taxonRank"), ("taxonomic_status","taxonomicStatus"),
            ("name_according_to","nameAccordingTo"),
            ("classification_path","classificationPath"),
        ]:
            new_val = (occ.get(key) or "").strip()
            if new_val:
                cur = f.get(field,"")
                if not cur or occ.get("nameAccordingTo","").upper() in ("WORMS","WORLD REGISTER OF MARINE SPECIES"):
                    f[field] = new_val

        # Vernacular names from GNV
        new_vn   = occ.get("vernacularNames") or []
        exist_vn = f.get("vernacular_names", [])
        exist_set= {v.get("name","").lower() for v in exist_vn}
        for vn in new_vn:
            if isinstance(vn, dict) and vn.get("name","").lower() not in exist_set:
                exist_vn.append(vn)
                exist_set.add(vn.get("name","").lower())
        f["vernacular_names"] = exist_vn[:20]

        # Locality
        loc = str(occ.get("verbatimLocality") or "").strip()
        if loc and loc not in f["localities"]:
            f["localities"].append(loc)

        # Habitat
        hab = str(occ.get("Habitat") or occ.get("habitat","")).strip()
        if hab and hab not in f["habitats"]:
            f["habitats"].append(hab)

        # Depth
        sampling = occ.get("Sampling Event") or occ.get("samplingEvent") or {}
        if isinstance(sampling, str):
            try:
                sampling = json.loads(sampling)
            except Exception:
                sampling = {}
        if isinstance(sampling, dict):
            depth = str(sampling.get("depth_m","")).strip()
            if depth and depth not in f["depth_range_m"]:
                f["depth_range_m"].append(depth)
            method = str(sampling.get("method","")).strip()
            if method and method not in f["sampling_methods"]:
                f["sampling_methods"].append(method)

        # Occurrence type
        occ_t = str(occ.get("occurrenceType") or occ.get("occurrence_type","")).strip()
        if occ_t and occ_t not in f["occurrence_types"]:
            f["occurrence_types"].append(occ_t)

        # Provenance
        citation = str(occ.get("Source Citation") or occ.get("sourceCitation","?")).strip()
        prov = article.get("provenance", [])
        prov_cites = [p.get("citation","") for p in prov]
        if citation not in prov_cites:
            prov.append({"citation": citation, "date": _now()})
        article["provenance"] = prov

        # Condensed occurrence for article record
        article.setdefault("occurrences", []).append({
            "locality": loc,
            "habitat":  hab,
            "date":     sampling.get("date","") if isinstance(sampling,dict) else "",
            "type":     occ_t,
            "evidence": str(occ.get("Raw Text Evidence") or "")[:300],
            "source":   citation[:100],
        })
        # Keep only latest 50 occurrences per article
        article["occurrences"] = article["occurrences"][-50:]

        # LLM narrative update (only if narrative is blank or stale)
        if llm_fn and (not article.get("narrative") or article.get("version",0) % 5 == 0):
            article["narrative"] = self._generate_species_narrative(article, llm_fn)

        self._save_article("species", slug, article)
        return article

    def _generate_species_narrative(self, article: dict, llm_fn: Callable) -> str:
        f    = article["facts"]
        name = article["title"]
        sp_prompt = (
            f"Write a concise 3-sentence scientific wiki summary for {name} "
            f"({f.get('family_','')}, {f.get('phylum','')}) "
            f"based on these observations:\n"
            f"Localities: {', '.join(f.get('localities',[])[:5])}\n"
            f"Habitats: {', '.join(f.get('habitats',[])[:4])}\n"
            f"Depths: {', '.join(f.get('depth_range_m',[])[:4])} m\n"
            f"Sources: {len(article.get('provenance',[]))} paper(s)\n"
            f"WoRMS ID: {f.get('worms_id','')}\n\n"
            f"Write in third person, scientific style. No headers."
        )
        try:
            return llm_fn(sp_prompt)[:1000]
        except Exception as exc:
            logger.debug("[Wiki] Narrative LLM error: %s", exc)
            return ""

    # ── Locality article ──────────────────────────────────────────────────────
    def update_locality_article(self, occ: dict) -> dict:
        loc = str(occ.get("verbatimLocality") or "").strip()
        if not loc:
            return {}
        slug    = _slug(loc)
        article = self._load_article("locality", slug) or _blank_locality_article(loc)

        f = article["facts"]
        lat = occ.get("decimalLatitude")
        lon = occ.get("decimalLongitude")
        if lat is not None: f["latitude"]  = lat
        if lon is not None: f["longitude"] = lon

        hab = str(occ.get("Habitat") or occ.get("habitat","")).strip()
        if hab and hab not in f["habitats"]:
            f["habitats"].append(hab)

        sp_name = (
            occ.get("validName") or occ.get("recordedName") or ""
        ).strip()
        if sp_name and sp_name not in f["species_checklist"]:
            f["species_checklist"].append(sp_name)
        f["species_checklist"] = sorted(set(f["species_checklist"]))

        citation = str(occ.get("Source Citation") or "?").strip()
        prov = article.get("provenance", [])
        if not any(p.get("citation","") == citation for p in prov):
            prov.append({"citation": citation, "date": _now()})
        article["provenance"] = prov

        self._save_article("locality", slug, article)
        return article

    # ── Locality Coord ───────────────────────────────────────────────────────
    def update_locality_coords(
        self,
        locality_name: str,
        lat: float,
        lon: float,
    ) -> bool:
        """
        Update geocoordinates for a locality article by name.
        Thin wrapper around update_locality_article() so postprocessing
        can call it without constructing a full occurrence dict.
        Called from biotrace_postprocessing.sync_wiki_coordinates().
        """
        slug    = _slug(locality_name)
        article = self._load_article("locality", slug) \
                or _blank_locality_article(locality_name)
        article["facts"]["latitude"]  = lat
        article["facts"]["longitude"] = lon
        article["last_updated"] = _now()
        self._save_article("locality", slug, article)
        self._update_index("locality", slug, locality_name)
        return True
    # ── Habitat article ───────────────────────────────────────────────────────
    def update_habitat_article(self, occ: dict) -> dict:
        hab = str(occ.get("Habitat") or occ.get("habitat","")).strip()
        if not hab:
            return {}
        slug    = _slug(hab)
        article = self._load_article("habitat", slug) or _blank_habitat_article(hab)

        f = article["facts"]
        sp_name = (occ.get("validName") or occ.get("recordedName","")).strip()
        if sp_name and sp_name not in f["species_checklist"]:
            f["species_checklist"].append(sp_name)

        family = str(occ.get("family_","")).strip()
        if family and family not in f["families"]:
            f["families"].append(family)

        loc = str(occ.get("verbatimLocality","")).strip()
        if loc and loc not in f["localities"]:
            f["localities"].append(loc)

        self._save_article("habitat", slug, article)
        return article

    # ── Paper article ─────────────────────────────────────────────────────────
    def update_paper_article(
        self,
        occurrences: list[dict],
        citation: str,
        llm_fn: Optional[Callable] = None,
    ) -> dict:
        ch      = _cite_hash(citation)
        article = self._load_article("papers", ch) or _blank_paper_article(citation)

        f = article["facts"]
        all_sp  = list({(o.get("validName") or o.get("recordedName","")).strip() for o in occurrences if isinstance(o,dict)})
        all_loc = list({str(o.get("verbatimLocality","")).strip() for o in occurrences if isinstance(o,dict)})
        all_hab = list({str(o.get("Habitat") or o.get("habitat","")).strip() for o in occurrences if isinstance(o,dict)})
        occ_types = list({str(o.get("occurrenceType","")).strip() for o in occurrences if isinstance(o,dict)})

        f["species_recorded"]   = sorted(set(f.get("species_recorded",[])   + all_sp))
        f["localities_studied"] = sorted(set(f.get("localities_studied",[]) + all_loc))
        f["habitats_covered"]   = sorted(set(f.get("habitats_covered",[])   + all_hab))
        f["n_records"]          = len(occurrences)
        f["occurrence_types"]   = sorted(set(f.get("occurrence_types",[])   + occ_types))

        if llm_fn and not article.get("abstract_summary"):
            prompt = (
                f"Write a 2-sentence scientific abstract for a paper titled:\n"
                f"'{citation}'\n\n"
                f"It reports {len(all_sp)} species from {len(all_loc)} localities "
                f"covering habitats: {', '.join(all_hab[:4])}. "
                f"Species include: {', '.join(all_sp[:8])}."
            )
            try:
                article["abstract_summary"] = llm_fn(prompt)[:600]
            except Exception:
                pass

        self._save_article("papers", ch, article)
        return article

    # ── Batch update ──────────────────────────────────────────────────────────
    def update_from_occurrences(
        self,
        occurrences: list[dict],
        citation: str = "Unknown",
        llm_fn: Optional[Callable] = None,
        update_narratives: bool = False,
    ) -> dict[str, int]:
        """
        Process a full list of occurrences → update species, locality, habitat,
        and paper articles. Returns counts of articles updated per section.
        """
        counts = {s: 0 for s in self.SECTIONS}

        for occ in occurrences:
            if not isinstance(occ, dict):
                continue
            if not occ.get("Source Citation"):
                occ["Source Citation"] = citation

            _llm = llm_fn if update_narratives else None

            self.update_species_article(occ, _llm)
            counts["species"] += 1

            self.update_locality_article(occ)
            counts["locality"] += 1

            self.update_habitat_article(occ)
            counts["habitat"] += 1

        self.update_paper_article(occurrences, citation, llm_fn)
        counts["papers"] += 1

        logger.info("[Wiki] Updated: %s", counts)
        return counts

    # ── Retrieval for LLM context ─────────────────────────────────────────────
    def build_wiki_context(self, query: str, top_k: int = 5) -> str:
        """
        Build a rich wiki context block for LLM prompt injection.
        Searches species, locality, habitat articles for query relevance.
        """
        tokens = re.findall(r"[A-Za-z]{4,}", query)
        lines  = ["=== WIKI CONTEXT ==="]

        # Search species articles
        sp_matches = []
        sp_dir = self.root / "species"
        for fp in sp_dir.glob("*.json"):
            art = self._read_json(fp)
            if not art:
                continue
            title = art.get("title","")
            if any(t.lower() in title.lower() or title.lower() in t.lower()
                   for t in tokens):
                sp_matches.append(art)

        if not sp_matches:
            # Fall back to all species articles sorted by recency
            all_arts = []
            for fp in sorted(sp_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:top_k]:
                art = self._read_json(fp)
                if art:
                    all_arts.append(art)
            sp_matches = all_arts

        for art in sp_matches[:top_k]:
            f = art.get("facts", {})
            lines.append(
                f"\n[SPECIES] {art.get('title','?')} "
                f"({f.get('family_','')}, {f.get('phylum','')})"
            )
            if art.get("narrative"):
                lines.append(f"  Summary: {art['narrative'][:300]}")
            locs = f.get("localities", [])[:3]
            habs = f.get("habitats", [])[:3]
            if locs:
                lines.append(f"  Localities: {', '.join(locs)}")
            if habs:
                lines.append(f"  Habitats: {', '.join(habs)}")
            wid = f.get("worms_id","")
            if wid:
                lines.append(f"  WoRMS: https://www.marinespecies.org/aphia.php?p=taxdetails&id={wid}")
            prov = art.get("provenance", [])
            if prov:
                lines.append(f"  Sources: {len(prov)} paper(s)")

        # Locality matches
        loc_dir = self.root / "locality"
        for fp in loc_dir.glob("*.json"):
            art = self._read_json(fp)
            if not art:
                continue
            title = art.get("title","")
            if any(t.lower() in title.lower() for t in tokens):
                f = art.get("facts",{})
                sp_list = f.get("species_checklist",[])
                lines.append(
                    f"\n[LOCALITY] {title} "
                    f"({len(sp_list)} species recorded)"
                )
                if sp_list:
                    lines.append(f"  Checklist: {', '.join(sp_list[:8])}")
                lat, lon = f.get("latitude"), f.get("longitude")
                if lat and lon:
                    lines.append(f"  Coords: {lat:.4f}, {lon:.4f}")
                break

        return "\n".join(lines) or "=== WIKI === (no articles yet)"

    # ── Markdown rendering ────────────────────────────────────────────────────
    def render_species_markdown(self, name: str) -> str:
        art = self.get_species_article(name)
        if not art:
            return f"# {name}\n\n_No wiki article found._"

        f = art.get("facts", {})
        lines = [
            f"# {art['title']}",
            "",
            f"**Family:** {f.get('family_','-')}  |  "
            f"**Order:** {f.get('order_','-')}  |  "
            f"**Phylum:** {f.get('phylum','-')}",
        ]
        # Full taxonomy line
        tax_parts = []
        for rank_label, fkey in [
            ("Kingdom",f.get("kingdom","")), ("Phylum",f.get("phylum","")),
            ("Class",f.get("class_","")),   ("Order",f.get("order_","")),
            ("Family",f.get("family_","")), ("Genus",f.get("genus_","")),
        ]:
            if fkey:
                tax_parts.append(f"{rank_label}: *{fkey}*")
        if tax_parts:
            lines += ["", "**Full taxonomy:** " + " › ".join(tax_parts)]

        # External database links
        ext_links = []
        wid = f.get("worms_id","")
        if wid:
            ext_links.append(f"[WoRMS AphiaID {wid}](https://www.marinespecies.org/aphia.php?p=taxdetails&id={wid})")
        if f.get("itis_id"):
            ext_links.append(f"[ITIS TSN {f['itis_id']}](https://www.itis.gov/servlet/SingleRpt/SingleRpt?search_topic=TSN&search_value={f['itis_id']})")
        if f.get("gbif_id"):
            ext_links.append(f"[GBIF {f['gbif_id']}](https://www.gbif.org/species/{f['gbif_id']})")
        if f.get("col_id"):
            ext_links.append(f"[Catalogue of Life](https://www.catalogueoflife.org/data/taxon/{f['col_id']})")
        if f.get("eol_id"):
            ext_links.append(f"[Encyclopedia of Life](https://eol.org/pages/{f['eol_id']})")
        if ext_links:
            lines += ["", "**External databases:** " + " · ".join(ext_links)]

        lines += [
            "",
            f"**Status:** {f.get('taxonomic_status','-')}  |  "
            f"**Rank:** {f.get('taxon_rank','-')}  |  "
            f"**Authority:** {f.get('name_according_to','-')}",
        ]

        # Vernacular names table
        vn = f.get("vernacular_names", [])
        if vn:
            lines += ["", "## Common Names"]
            lines.append("| Name | Language |")
            lines.append("|------|----------|")
            for v in vn[:10]:
                lines.append(f"| {v.get('name','-')} | {v.get('language','-')} |")

        lines += ["", "## Summary", art.get("narrative","_No narrative generated yet._")]

        lines += ["", "## Localities"]
        for loc in sorted(set(f.get("localities",[])))[:15]:
            lines.append(f"- {loc}")

        lines += ["", "## Habitats"]
        for hab in sorted(set(f.get("habitats",[])))[:10]:
            lines.append(f"- {hab}")

        lines += ["", "## Sampling"]
        depths  = f.get("depth_range_m",[])
        methods = f.get("sampling_methods",[])
        if depths:
            lines.append(f"- **Depths recorded:** {', '.join(depths)} m")
        if methods:
            lines.append(f"- **Methods:** {', '.join(methods)}")

        lines += ["", "## Provenance"]
        for p in art.get("provenance",[]):
            lines.append(f"- {p.get('citation','?')} ({p.get('date','')[:10]})")

        lines += [
            "",
            f"_Wiki version {art.get('version',1)} · Last updated {art.get('last_updated','')[:10]}_"
        ]
        return "\n".join(lines)

    # ── Index / stats ─────────────────────────────────────────────────────────
    def index_stats(self) -> dict:
        idx = self._read_json(self.root / "index.json")
        return {
            "total_articles": idx.get("total_articles", 0),
            "by_section": {
                s: len(idx.get("sections", {}).get(s, []))
                for s in self.SECTIONS
            },
            "wiki_root": str(self.root),
        }

    def list_species(self) -> list[str]:
        return sorted(
            self._read_json(fp).get("title","")
            for fp in (self.root / "species").glob("*.json")
        )

    def list_localities(self) -> list[str]:
        return sorted(
            self._read_json(fp).get("title","")
            for fp in (self.root / "locality").glob("*.json")
        )
