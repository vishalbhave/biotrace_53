"""
biotrace_docling_wiki_bridge.py  —  BioTrace v5.6
═══════════════════════════════════════════════════════════════════════════════════════════
Inline docling → Wiki article enrichment + wiki-relation nodes for KG filtering.

DESIGN GOALS
────────────
1. No separate wiki-agent run required by default.
   Docling already segments PDFs into typed sections (Methods, Results, Tables,
   Taxonomy, References …).  This bridge wires those sections directly to the
   UnifiedWiki updater so species articles are enriched *during* the normal
   extraction pipeline — zero extra user steps.

2. Wiki-agent relations become KG filter nodes.
   When the OllamaWikiAgent runs (e.g. for a high-priority species), the
   semantic relations it returns (e.g. "Cassiopea andromeda – FOUND_AT – Narara reef")
   are ingested into the KG as typed WikiRelation nodes and exposed as sidebar
   filter chips in every Streamlit tab that shows species data.

3. No duplication of docling PDF parsing.
   biotrace_v5.py already converts PDFs via docling's DocumentConverter.
   This module accepts the *already-parsed* DoclingDocument (or its markdown
   export) — it never calls docling a second time.

PUBLIC API
──────────
    from biotrace_docling_wiki_bridge import DoclingWikiBridge

    bridge = DoclingWikiBridge(
        wiki_root   = "biodiversity_data/wiki",
        kg_db_path  = "biodiversity_data/knowledge_graph.db",
        run_wiki_agent = False,   # True only for priority species (opt-in)
    )

    # Called once per extracted document (after docling parse)
    bridge.process_document(
        doc_sections   = sections_dict,   # {section_role: text, ...}
        occurrences    = occurrence_list, # list[dict] from BioTrace extractor
        citation       = "Author (Year) Journal",
        species_names  = ["Cassiopea andromeda", ...],
    )

    # In Streamlit sidebar — call to render filter chips
    selected_filters = bridge.render_relation_filter_sidebar()
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("biotrace.docling_wiki_bridge")

# ─── Pydantic models for wiki relations ──────────────────────────────────────
try:
    from pydantic import BaseModel, Field as PField
    _PYDANTIC_OK = True
except ImportError:
    _PYDANTIC_OK = False
    BaseModel = object  # type: ignore
    def PField(*args, **kwargs): return None  # type: ignore


if _PYDANTIC_OK:
    class WikiRelation(BaseModel):
        """A single semantic relation extracted from a wiki section or agent run."""
        subject:        str         # e.g. "Cassiopea andromeda"
        predicate:      str         # e.g. "FOUND_AT" | "HAS_HABITAT" | "CO_OCCURS_WITH"
        object_:        str = PField(..., alias="object")  # e.g. "Narara reef"
        source:         str = ""    # citation string
        confidence:     float = 1.0
        origin:         str = "docling"  # "docling" | "wiki_agent" | "rule"

        model_config = {"populate_by_name": True}

else:
    @dataclass
    class WikiRelation:  # type: ignore
        subject: str; predicate: str; object_: str
        source: str = ""; confidence: float = 1.0; origin: str = "docling"


# ─── Rule-based relation extractor from section text ─────────────────────────

_BINOMIAL_RE  = re.compile(r"\b([A-Z][a-z]{2,})\s+([a-z]{3,}(?:\s+(?:cf\.|aff\.|sp\.)?\S*)?)\b")
_FOUND_AT_RE  = re.compile(
    r"(?:recorded?|collected?|found?|observed?|reported?|documented?)"
    r"\s+(?:from|at|in|near)\s+([A-Z][A-Za-z\s,]{3,40})",
    re.IGNORECASE,
)
_HABITAT_RE   = re.compile(
    r"(?:inhabit[s]?|associat\w+\s+with|occur[s]?\s+in)\s+([a-z][a-z\s\-]{3,35})",
    re.IGNORECASE,
)
_DEPTH_RE     = re.compile(
    r"(?:at\s+depth[s]?\s+of|between)\s+([\d.]+)\s*(?:–|-)\s*([\d.]+)\s*m",
    re.IGNORECASE,
)
_COOCCUR_RE   = re.compile(
    r"(?:alongside|together\s+with|associated\s+with)\s+([A-Z][a-z]{2,}\s+[a-z]{3,})",
    re.IGNORECASE,
)


def _extract_relations_from_text(
    text:     str,
    species:  list[str],
    citation: str,
) -> list["WikiRelation"]:
    """
    Rule-based relation extraction from a section of text.
    Returns WikiRelation objects for FOUND_AT, HAS_HABITAT, CO_OCCURS_WITH, HAS_DEPTH.
    """
    relations: list[WikiRelation] = []
    if not text.strip():
        return relations

    # For each sentence, try to associate relations with species mentioned in it
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sent in sentences:
        sent_species = [sp for sp in species if sp.lower() in sent.lower()]
        if not sent_species:
            # Try to detect a new binomial in this sentence
            for m in _BINOMIAL_RE.finditer(sent):
                candidate = m.group(0)
                if candidate not in sent_species:
                    sent_species.append(candidate)

        if not sent_species:
            continue

        # FOUND_AT
        for m in _FOUND_AT_RE.finditer(sent):
            loc = m.group(1).strip().rstrip(".,;")
            for sp in sent_species:
                relations.append(WikiRelation(
                    subject=sp, predicate="FOUND_AT", **{"object": loc},
                    source=citation, confidence=0.8, origin="rule"))

        # HAS_HABITAT
        for m in _HABITAT_RE.finditer(sent):
            hab = m.group(1).strip().rstrip(".,;")
            for sp in sent_species:
                relations.append(WikiRelation(
                    subject=sp, predicate="HAS_HABITAT", **{"object": hab},
                    source=citation, confidence=0.7, origin="rule"))

        # HAS_DEPTH
        for m in _DEPTH_RE.finditer(sent):
            depth_str = f"{m.group(1)}–{m.group(2)} m"
            for sp in sent_species:
                relations.append(WikiRelation(
                    subject=sp, predicate="HAS_DEPTH", **{"object": depth_str},
                    source=citation, confidence=0.75, origin="rule"))

        # CO_OCCURS_WITH
        for m in _COOCCUR_RE.finditer(sent):
            co_sp = m.group(1).strip()
            for sp in sent_species:
                if sp != co_sp:
                    relations.append(WikiRelation(
                        subject=sp, predicate="CO_OCCURS_WITH", **{"object": co_sp},
                        source=citation, confidence=0.65, origin="rule"))

    return relations


# ─── KG relation node ingestor ────────────────────────────────────────────────

def _ingest_relations_to_kg(
    kg_db_path: str,
    relations:  list["WikiRelation"],
) -> int:
    """
    Ingest WikiRelation objects into the KG as typed nodes + edges.

    Node types added:
      WikiRelation  — the relation triple (as a reification node)
    Edge types added:
      REL_FOUND_AT, REL_HAS_HABITAT, REL_CO_OCCURS_WITH, REL_HAS_DEPTH
    (prefixed with REL_ to distinguish from primary occurrence edges)

    Returns the number of relations successfully ingested.
    """
    if not kg_db_path or not relations:
        return 0
    count = 0
    try:
        from biotrace_knowledge_graph import BioTraceKnowledgeGraph, _node_id
        kg = BioTraceKnowledgeGraph(kg_db_path)

        for rel in relations:
            try:
                subj_nid = _node_id("Species", rel.subject)
                obj_label = rel.object_
                # Determine object node type from predicate
                obj_type = {
                    "FOUND_AT":       "Locality",
                    "HAS_HABITAT":    "Habitat",
                    "CO_OCCURS_WITH": "Species",
                    "HAS_DEPTH":      "DepthRange",
                }.get(rel.predicate, "WikiConcept")

                obj_nid = kg._upsert_node(obj_type, obj_label, {
                    "from_wiki_relation": True,
                    "confidence": rel.confidence,
                    "source": rel.source,
                })
                # Ensure subject species node exists
                if not kg._G.has_node(subj_nid):
                    kg._upsert_node("Species", rel.subject, {})

                kg._upsert_edge(
                    subj_nid,
                    f"REL_{rel.predicate}",
                    obj_nid,
                    weight=rel.confidence,
                    props={"source": rel.source, "origin": rel.origin,
                           "confidence": rel.confidence},
                )
                count += 1
            except Exception as exc:
                logger.debug("[DoclingWiki] relation ingest skip: %s", exc)

        kg._conn.commit()
        try:
            kg.detect_communities()
        except Exception:
            pass
        kg.close()
        logger.info("[DoclingWiki] Ingested %d wiki relations into KG.", count)
    except Exception as exc:
        logger.warning("[DoclingWiki] KG ingest failed: %s", exc)
    return count


# ─── Wiki inline updater (from docling sections) ─────────────────────────────

def _update_wiki_from_sections(
    wiki_root:  str,
    species:    str,
    sections:   dict[str, str],
    occurrence: dict,
    citation:   str,
) -> bool:
    """
    Update a species wiki article from docling-identified section text.
    Avoids calling OllamaWikiAgent unless explicitly requested.

    Sections dict keys expected (same as ScientificPaperChunker roles):
      ABSTRACT, METHODS, RESULTS, DISCUSSION, TAXONOMY, TABLES, OTHER
    """
    try:
        from Unified_Wiki_Module import UnifiedWiki
        import datetime as _dt

        wiki    = UnifiedWiki(wiki_root)
        sp_slug = wiki._slugify(species)
        sp_fp   = wiki.root / "species" / f"{sp_slug}.json"
        art     = wiki._read_json(sp_fp) or {
            "species_name":  species,
            "citations":     [],
            "occurrences":   [],
            "sections":      {},
            "last_updated":  _dt.datetime.now().isoformat(),
        }

        # Merge section text
        existing_sections = art.get("sections", {})
        for role, text in sections.items():
            if not text.strip():
                continue
            existing = existing_sections.get(role, "")
            if text.strip() not in existing:
                # Append with citation attribution
                existing_sections[role] = (
                    existing.rstrip() + f"\n\n[{citation}]\n" + text
                ).strip()
        art["sections"] = existing_sections

        # Merge occurrence
        existing_occs = art.get("occurrences", [])
        occ_key = (occurrence.get("verbatimLocality", ""),
                   occurrence.get("sourceCitation", ""))
        if not any(
            (o.get("verbatimLocality", ""), o.get("sourceCitation", "")) == occ_key
            for o in existing_occs
        ):
            existing_occs.append(occurrence)
        art["occurrences"] = existing_occs

        # Merge citation
        if citation and citation not in art.get("citations", []):
            art.setdefault("citations", []).append(citation)

        art["last_updated"] = _dt.datetime.now().isoformat()
        wiki._save_json(sp_fp, art)

        # Update locality article
        loc = occurrence.get("verbatimLocality", "")
        if loc:
            wiki.update_locality_article(loc, species, occurrence, citation)

        wiki._rebuild_index()
        return True
    except Exception as exc:
        logger.warning("[DoclingWiki] wiki update failed for %s: %s", species, exc)
        return False


# ─── Main bridge class ────────────────────────────────────────────────────────

class DoclingWikiBridge:
    """
    Wire docling-parsed sections directly into Wiki + KG without running a
    separate OllamaWikiAgent pass.

    Parameters
    ──────────
    wiki_root       Root directory for UnifiedWiki.
    kg_db_path      BioTraceKnowledgeGraph SQLite path.
    run_wiki_agent  If True, OllamaWikiAgent is called for species that have
                    a Taxonomy or Discussion section (opt-in enrichment).
    wiki_agent_model  Ollama model for optional agent calls (default qwen2.5:14b).
    min_confidence  Minimum confidence for rule-based relation inclusion (0–1).
    """

    def __init__(
        self,
        wiki_root:        str = "biodiversity_data/wiki",
        kg_db_path:       str = "",
        run_wiki_agent:   bool = False,
        wiki_agent_model: str = "qwen2.5:14b",
        min_confidence:   float = 0.6,
    ):
        self.wiki_root        = wiki_root
        self.kg_db_path       = kg_db_path
        self.run_wiki_agent   = run_wiki_agent
        self.wiki_agent_model = wiki_agent_model
        self.min_confidence   = min_confidence
        self._relation_store: list[WikiRelation] = []  # in-session store for filter UI

    def process_document(
        self,
        doc_sections:   dict[str, str],
        occurrences:    list[dict],
        citation:       str,
        species_names:  list[str] | None = None,
        log_cb:         Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Process a docling-parsed document's sections.

        Parameters
        ──────────
        doc_sections    {section_role: text} dict from ScientificPaperChunker
                        or docling section extractor.
        occurrences     List of occurrence dicts extracted by BioTrace.
        citation        Formatted citation string.
        species_names   Override list of species (auto-detected from occurrences if None).
        log_cb          Optional logging callback.

        Returns summary dict with wiki_updated, relations_ingested, agent_runs.
        """
        def _log(msg: str):
            logger.info(msg)
            if log_cb: log_cb(msg)

        if species_names is None:
            species_names = list({
                o.get("validName") or o.get("recordedName", "")
                for o in occurrences
                if o.get("validName") or o.get("recordedName")
            })

        _log(f"[DoclingWiki] Processing {len(species_names)} species, "
             f"{len(doc_sections)} sections, citation='{citation[:60]}'")

        # ── Step 1: Extract relations from all sections ───────────────────────
        all_relations: list[WikiRelation] = []
        for role, text in doc_sections.items():
            if not text.strip():
                continue
            rels = _extract_relations_from_text(text, species_names, citation)
            rels = [r for r in rels if r.confidence >= self.min_confidence]
            all_relations.extend(rels)
            _log(f"[DoclingWiki] {role}: {len(rels)} relations extracted")

        # ── Step 2: Ingest relations into KG as filter nodes ─────────────────
        ingested = 0
        if self.kg_db_path and all_relations:
            ingested = _ingest_relations_to_kg(self.kg_db_path, all_relations)
            _log(f"[DoclingWiki] {ingested} relations ingested into KG")

        # Store for filter UI
        self._relation_store.extend(all_relations)

        # ── Step 3: Update Wiki from docling sections (no agent needed) ───────
        wiki_updates = 0
        for species in species_names:
            sp_occs = [o for o in occurrences
                       if (o.get("validName") or o.get("recordedName", "")) == species]
            occ = sp_occs[0] if sp_occs else {}

            # Filter sections relevant to this species
            sp_sections = {}
            for role, text in doc_sections.items():
                if species.lower() in text.lower() or role in ("METHODS", "ABSTRACT"):
                    sp_sections[role] = text

            if sp_sections and self.wiki_root:
                ok = _update_wiki_from_sections(
                    self.wiki_root, species, sp_sections, occ, citation)
                if ok:
                    wiki_updates += 1

        _log(f"[DoclingWiki] Wiki updated for {wiki_updates}/{len(species_names)} species")

        # ── Step 4: Optional OllamaWikiAgent for Taxonomy/Discussion sections ─
        agent_runs = 0
        if self.run_wiki_agent:
            try:
                from biotrace_wiki_agent import OllamaWikiAgent
                from Unified_Wiki_Module import UnifiedWiki
                wiki  = UnifiedWiki(self.wiki_root)
                agent = OllamaWikiAgent(wiki=wiki, model=self.wiki_agent_model)

                # Only run agent for species with Taxonomy or Discussion sections
                agent_text = "\n\n".join(
                    v for k, v in doc_sections.items()
                    if k in ("TAXONOMY", "DISCUSSION", "RESULTS") and v.strip()
                )
                if agent_text:
                    for species in species_names:
                        try:
                            result = agent.orchestrate(
                                species_name=species,
                                chunk_text=agent_text[:8000],
                                citation=citation,
                            )
                            # Agent relations → KG
                            if self.kg_db_path and result.sections:
                                agent_rels = _extract_relations_from_text(
                                    result.sections.distribution_habitat + " "
                                    + result.sections.ecology_behaviour,
                                    species_names, citation)
                                if agent_rels:
                                    for r in agent_rels:
                                        r.origin = "wiki_agent"
                                    _ingest_relations_to_kg(self.kg_db_path, agent_rels)
                                    self._relation_store.extend(agent_rels)
                            agent_runs += 1
                            _log(f"[DoclingWiki] Agent enriched: {species}")
                        except Exception as ae:
                            _log(f"[DoclingWiki] Agent failed for {species}: {ae}")
            except ImportError:
                _log("[DoclingWiki] OllamaWikiAgent not available — skipping agent pass")

        return {
            "wiki_updated":       wiki_updates,
            "relations_extracted": len(all_relations),
            "relations_ingested":  ingested,
            "agent_runs":          agent_runs,
        }

    # ── Filter node helpers ───────────────────────────────────────────────────

    def get_filter_nodes(self) -> dict[str, list[str]]:
        """
        Return all unique relation objects grouped by predicate.
        Used to populate Streamlit sidebar filter chips.

        Example return:
            {
              "FOUND_AT":       ["Narara reef", "Gulf of Kutch", ...],
              "HAS_HABITAT":    ["coral reef", "seagrass bed", ...],
              "CO_OCCURS_WITH": ["Aurelia aurita", ...],
              "HAS_DEPTH":      ["0.5–3 m", ...],
            }
        """
        nodes: dict[str, set[str]] = {}
        for rel in self._relation_store:
            nodes.setdefault(rel.predicate, set()).add(rel.object_)
        return {k: sorted(v) for k, v in nodes.items()}

    def get_filter_nodes_from_kg(self) -> dict[str, list[str]]:
        """
        Read persisted WikiRelation filter nodes from KG (survives restarts).
        Falls back to in-memory store if KG unavailable.
        """
        if not self.kg_db_path:
            return self.get_filter_nodes()
        try:
            from biotrace_knowledge_graph import BioTraceKnowledgeGraph
            kg   = BioTraceKnowledgeGraph(self.kg_db_path)
            rows = kg._conn.execute(
                """SELECT rel_type, label FROM kg_edges e
                   JOIN kg_nodes n ON e.tgt_id=n.node_id
                   WHERE e.rel_type LIKE 'REL_%'
                   GROUP BY e.rel_type, n.label"""
            ).fetchall()
            kg.close()
            result: dict[str, list[str]] = {}
            for rel_type, label in rows:
                pred = rel_type.replace("REL_", "")
                result.setdefault(pred, [])
                if label not in result[pred]:
                    result[pred].append(label)
            return {k: sorted(v) for k, v in result.items()}
        except Exception as exc:
            logger.warning("[DoclingWiki] filter_nodes_from_kg: %s", exc)
            return self.get_filter_nodes()

    def apply_filters(
        self,
        occurrences: list[dict],
        selected_filters: dict[str, list[str]],
    ) -> list[dict]:
        """
        Filter occurrence list by selected relation filter nodes.

        Parameters
        ──────────
        occurrences     Full occurrence list.
        selected_filters {predicate: [selected_values]} from sidebar widget.

        Returns filtered occurrence list.
        """
        if not selected_filters:
            return occurrences

        filtered = []
        for occ in occurrences:
            match = True
            for pred, values in selected_filters.items():
                if not values:
                    continue
                if pred == "FOUND_AT":
                    loc = occ.get("verbatimLocality", "")
                    if not any(v.lower() in loc.lower() for v in values):
                        match = False; break
                elif pred == "HAS_HABITAT":
                    hab = occ.get("habitat", "") or occ.get("habitat_", "")
                    if not any(v.lower() in hab.lower() for v in values):
                        match = False; break
                # CO_OCCURS_WITH and HAS_DEPTH not directly in occurrence dict;
                # skip rather than exclude
            if match:
                filtered.append(occ)
        return filtered

    def render_relation_filter_sidebar(
        self,
        use_kg:        bool = True,
        key_prefix:    str  = "wiki_rel",
    ) -> dict[str, list[str]]:
        """
        Render Streamlit sidebar multiselect widgets for wiki-relation filter nodes.
        Returns selected_filters dict to pass to apply_filters().

        Place this in any tab that renders species occurrence data:

            from biotrace_docling_wiki_bridge import DoclingWikiBridge
            bridge = DoclingWikiBridge(wiki_root=WIKI_ROOT, kg_db_path=KG_DB_PATH)
            selected = bridge.render_relation_filter_sidebar()
            filtered_occs = bridge.apply_filters(occurrences, selected)
        """
        import streamlit as st

        nodes = self.get_filter_nodes_from_kg() if use_kg else self.get_filter_nodes()
        if not nodes:
            return {}

        selected: dict[str, list[str]] = {}
        with st.sidebar.expander("🔗 Wiki Relation Filters", expanded=False):
            st.caption("Filter occurrences by semantically extracted relations.")
            pred_labels = {
                "FOUND_AT":       "📍 Found at locality",
                "HAS_HABITAT":    "🌿 Habitat",
                "CO_OCCURS_WITH": "🐟 Co-occurs with",
                "HAS_DEPTH":      "🌊 Depth range",
            }
            for pred, values in nodes.items():
                label = pred_labels.get(pred, pred)
                sel = st.multiselect(label, values, key=f"{key_prefix}_{pred}")
                if sel:
                    selected[pred] = sel

        # Show active filter summary
        if selected:
            total_filters = sum(len(v) for v in selected.values())
            st.sidebar.info(f"🔍 {total_filters} wiki-relation filter(s) active")

        return selected


# ─── Section extractor from docling output ────────────────────────────────────

def extract_sections_from_docling(
    docling_doc: Any,
    fallback_markdown: str = "",
) -> dict[str, str]:
    """
    Extract typed sections from a docling DocumentResult or its markdown export.

    Compatible with both docling's native DoclingDocument and the markdown
    string that biotrace_v5.py already generates via docling.

    Returns {section_role: combined_text} matching ScientificPaperChunker roles.
    """
    sections: dict[str, str] = {}

    # ── Attempt native docling section access ─────────────────────────────────
    try:
        # docling >= 1.x: doc.export_to_dict() has 'body' with section elements
        doc_dict = docling_doc.export_to_dict() if hasattr(docling_doc, "export_to_dict") else {}
        for elem in doc_dict.get("body", []):
            label = (elem.get("label") or "").lower()
            text  = elem.get("text") or elem.get("content") or ""
            role  = _map_docling_label_to_role(label)
            if role and text:
                sections[role] = sections.get(role, "") + "\n\n" + text
        if sections:
            return {k: v.strip() for k, v in sections.items()}
    except Exception:
        pass

    # ── Fallback: parse the markdown string produced by biotrace_v5.py ────────
    if fallback_markdown:
        from biotrace_scientific_chunker import ScientificPaperChunker, classify_section
        sc = ScientificPaperChunker()
        raw_sections = sc._split_sections(fallback_markdown)
        for sec in raw_sections:
            role = sec["role"]
            sections[role] = sections.get(role, "") + "\n\n" + sec["text"]
        return {k: v.strip() for k, v in sections.items()}

    return sections


def _map_docling_label_to_role(label: str) -> Optional[str]:
    """Map a docling element label to a ScientificPaperChunker section role."""
    label = label.lower()
    mapping = {
        "abstract":     "ABSTRACT",
        "introduction": "INTRODUCTION",
        "method":       "METHODS",
        "material":     "METHODS",
        "result":       "RESULTS",
        "discussion":   "DISCUSSION",
        "conclusion":   "DISCUSSION",
        "taxonomy":     "TAXONOMY",
        "table":        "TABLES",
        "figure":       "OTHER",
        "reference":    "OTHER",
    }
    for key, role in mapping.items():
        if key in label:
            return role
    return None