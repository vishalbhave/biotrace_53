"""
biotrace_knowledge_graph.py  —  BioTrace v5.0
────────────────────────────────────────────────────────────────────────────
GraphRAG-style Knowledge Graph for marine biodiversity occurrence data.

Inspired by Microsoft GraphRAG (2024) and Karpathy's LLM-Wiki vision (Apr 2026):
  • Entities   : Species, Locality, Habitat, Family, Order, Phylum, Paper
  • Relations  : FOUND_AT, BELONGS_TO, CITED_BY, CO_OCCURS_WITH, SIMILAR_TO
  • Community  : Louvain-style community detection for thematic clustering
  • RAG Bridge : Graph-aware context retrieval for LLM prompts
  • Persistence: SQLite-backed adjacency store (no GPU / vector DB required)

Key differences from plain RAG:
  ✓ Multi-hop reasoning  —  "which families co-occur in mangroves?"
  ✓ Relationship context —  edges carry habitat, date, depth, source
  ✓ Structural summaries —  community summaries fed to LLM as context
  ✓ Zero external deps   —  NetworkX + SQLite only

Darwin Core–aligned field names throughout.

Usage:
    kg = BioTraceKnowledgeGraph("biodiversity_data/knowledge_graph.db")
    kg.ingest_occurrences(occurrences)          # list[dict] from BioTrace extractor
    results = kg.graph_rag_query("What species were found in coral reefs?", llm_fn)
    kg.export_pyvis_html("kg_viz.html")
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Optional

import networkx as nx

logger = logging.getLogger("biotrace.knowledge_graph")

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
NODE_COLORS = {
    "Species":   "#2E86AB",   # marine blue
    "Locality":  "#A23B72",   # deep purple
    "Habitat":   "#F18F01",   # amber
    "Family":    "#C73E1D",   # coral red
    "Order":     "#3B1F2B",   # dark maroon
    "Phylum":    "#44BBA4",   # teal
    "Paper":     "#6B4226",   # brown
}

EDGE_STYLES = {
    "FOUND_AT":       {"color": "#2E86AB", "width": 2},
    "BELONGS_TO":     {"color": "#C73E1D", "width": 1},
    "CITED_BY":       {"color": "#6B4226", "width": 1},
    "CO_OCCURS_WITH": {"color": "#44BBA4", "width": 3},
    "SIMILAR_TO":     {"color": "#F18F01", "width": 1, "dashes": True},
}

_MIN_COOCCUR = 2   # minimum co-occurrence count to create a CO_OCCURS_WITH edge


# ─────────────────────────────────────────────────────────────────────────────
#  SCHEMA SQL
# ─────────────────────────────────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS kg_nodes (
    node_id     TEXT PRIMARY KEY,
    node_type   TEXT NOT NULL,
    label       TEXT NOT NULL,
    properties  TEXT,          -- JSON blob
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS kg_edges (
    edge_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    src_id      TEXT NOT NULL,
    rel_type    TEXT NOT NULL,
    tgt_id      TEXT NOT NULL,
    weight      REAL DEFAULT 1.0,
    properties  TEXT,          -- JSON blob (habitat, depth, date, source)
    created_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(src_id, rel_type, tgt_id)
);
CREATE TABLE IF NOT EXISTS kg_communities (
    community_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    community_key   TEXT UNIQUE,
    node_ids        TEXT,      -- JSON list
    summary         TEXT,
    keywords        TEXT,      -- JSON list
    created_at      TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_edges_src  ON kg_edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_tgt  ON kg_edges(tgt_id);
CREATE INDEX IF NOT EXISTS idx_edges_rel  ON kg_edges(rel_type);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON kg_nodes(node_type);
"""


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _slugify(text: str) -> str:
    """Create a deterministic node ID from arbitrary text."""
    return re.sub(r"[^a-z0-9_]", "_", str(text).lower().strip())[:80]


def _node_id(node_type: str, label: str) -> str:
    return f"{node_type.lower()}::{_slugify(label)}"


def _now() -> str:
    return datetime.utcnow().isoformat()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────
class BioTraceKnowledgeGraph:
    """
    Graph-augmented RAG knowledge store for BioTrace biodiversity occurrences.

    Architecture:
        SQLite  ──►  NetworkX in-memory graph  ──►  GraphRAG context builder
                          │
                          └──►  PyVis / Plotly visualisation
    """

    def __init__(self, db_path: str = "biodiversity_data/knowledge_graph.db"):
        self.db_path = db_path
        self._G: nx.MultiDiGraph = nx.MultiDiGraph()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()
        self._load_graph_from_db()
        logger.info(
            "[KG] Loaded: %d nodes, %d edges",
            self._G.number_of_nodes(),
            self._G.number_of_edges(),
        )

    # ── DB bootstrap ──────────────────────────────────────────────────────────
    def _init_db(self):
        import os
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_DDL)
        self._conn.commit()

    def _load_graph_from_db(self):
        """Rebuild in-memory NetworkX graph from SQLite."""
        self._G.clear()
        for row in self._conn.execute(
            "SELECT node_id, node_type, label, properties FROM kg_nodes"
        ):
            nid, ntype, label, props_json = row
            props = json.loads(props_json or "{}")
            self._G.add_node(
                nid,
                node_type=ntype,
                label=label,
                color=NODE_COLORS.get(ntype, "#888888"),
                **props,
            )

        for row in self._conn.execute(
            "SELECT src_id, rel_type, tgt_id, weight, properties FROM kg_edges"
        ):
            src, rel, tgt, wt, props_json = row
            props = json.loads(props_json or "{}")
            self._G.add_edge(src, tgt, rel_type=rel, weight=wt, **props)

    # ── Node / Edge persistence ───────────────────────────────────────────────
    def _upsert_node(self, node_type: str, label: str, props: dict | None = None) -> str:
        nid = _node_id(node_type, label)
        props_json = json.dumps(props or {})
        self._conn.execute(
            """INSERT INTO kg_nodes(node_id, node_type, label, properties, updated_at)
               VALUES(?,?,?,?,?)
               ON CONFLICT(node_id) DO UPDATE SET
                 properties=excluded.properties, updated_at=excluded.updated_at""",
            (nid, node_type, label, props_json, _now()),
        )
        if not self._G.has_node(nid):
            self._G.add_node(
                nid,
                node_type=node_type,
                label=label,
                color=NODE_COLORS.get(node_type, "#888888"),
                **(props or {}),
            )
        return nid

    def _upsert_edge(
        self,
        src_id: str,
        rel_type: str,
        tgt_id: str,
        weight: float = 1.0,
        props: dict | None = None,
    ):
        props_json = json.dumps(props or {})
        try:
            self._conn.execute(
                """INSERT INTO kg_edges(src_id, rel_type, tgt_id, weight, properties)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(src_id, rel_type, tgt_id) DO UPDATE SET
                     weight=weight+1, properties=excluded.properties""",
                (src_id, rel_type, tgt_id, weight, props_json),
            )
        except sqlite3.IntegrityError:
            pass
        if not self._G.has_edge(src_id, tgt_id):
            self._G.add_edge(
                src_id, tgt_id, rel_type=rel_type, weight=weight, **(props or {})
            )

    # ── Core ingestion ────────────────────────────────────────────────────────
    def ingest_occurrences(self, occurrences: list[dict]) -> int:
        """
        Build / update the knowledge graph from a list of occurrence dicts
        (output of BioTrace extraction + species_verifier).

        Returns the number of NEW nodes created.
        """
        before = self._G.number_of_nodes()

        # Group by locality for CO_OCCURS_WITH edge generation
        locality_species: dict[str, list[str]] = defaultdict(list)

        for occ in occurrences:
            if not isinstance(occ, dict):
                continue

            # ── Species node ─────────────────────────────────────────────────
            sp_name = (
                occ.get("validName")
                or occ.get("recordedName")
                or occ.get("Recorded Name", "")
            ).strip()
            if not sp_name:
                continue

            worms_id = occ.get("wormsID", "")
            sp_props = {
                "wormsID":        worms_id,
                "taxonRank":      occ.get("taxonRank", "species"),
                "taxonomicStatus": occ.get("taxonomicStatus", ""),
                "matchScore":     occ.get("matchScore", 0),
                "phylum":         occ.get("phylum", ""),
                "class_":         occ.get("class_", ""),
                "order_":         occ.get("order_", ""),
                "family_":        occ.get("family_", ""),
            }
            sp_id = self._upsert_node("Species", sp_name, sp_props)

            # ── Taxonomy chain  Species → Family → Order → Phylum ────────────
            family_name = (occ.get("family_") or "").strip()
            order_name  = (occ.get("order_")  or "").strip()
            phylum_name = (occ.get("phylum")  or "").strip()

            if family_name:
                fam_id = self._upsert_node("Family", family_name)
                self._upsert_edge(sp_id, "BELONGS_TO", fam_id)

                if order_name:
                    ord_id = self._upsert_node("Order", order_name)
                    self._upsert_edge(fam_id, "BELONGS_TO", ord_id)

                    if phylum_name:
                        phy_id = self._upsert_node("Phylum", phylum_name)
                        self._upsert_edge(ord_id, "BELONGS_TO", phy_id)

            # ── Locality node ─────────────────────────────────────────────────
            locality_raw = (
                occ.get("verbatimLocality")
                or occ.get("locality", {}).get("site_name", "")
                if isinstance(occ.get("locality"), dict)
                else occ.get("verbatimLocality", "")
            )
            locality = str(locality_raw or "Unknown Locality").strip()[:120]

            lat = occ.get("decimalLatitude")
            lon = occ.get("decimalLongitude")
            loc_id = self._upsert_node(
                "Locality",
                locality,
                {
                    "decimalLatitude":  lat,
                    "decimalLongitude": lon,
                    "geocodingSource":  occ.get("geocodingSource", ""),
                },
            )

            # ── Habitat node ──────────────────────────────────────────────────
            habitat = str(occ.get("Habitat") or occ.get("habitat", "") or "").strip()
            if habitat:
                hab_id = self._upsert_node("Habitat", habitat)
                self._upsert_edge(loc_id, "FOUND_AT", hab_id)  # locality is IN habitat

            # ── FOUND_AT edge with sampling context ───────────────────────────
            sampling = occ.get("Sampling Event") or occ.get("samplingEvent") or {}
            if isinstance(sampling, str):
                try:
                    sampling = json.loads(sampling)
                except Exception:
                    sampling = {"raw": sampling}
            edge_props = {
                "habitat":         habitat,
                "date":            sampling.get("date", "") if isinstance(sampling, dict) else "",
                "depth_m":         sampling.get("depth_m", "") if isinstance(sampling, dict) else "",
                "method":          sampling.get("method", "") if isinstance(sampling, dict) else "",
                "occurrenceType":  occ.get("occurrenceType", occ.get("occurrence_type", "")),
            }
            self._upsert_edge(sp_id, "FOUND_AT", loc_id, props=edge_props)

            # ── Paper / Source node ───────────────────────────────────────────
            citation = (
                occ.get("Source Citation")
                or occ.get("sourceCitation", "Unknown Source")
            ).strip()[:200]
            paper_id = self._upsert_node("Paper", citation)
            self._upsert_edge(sp_id, "CITED_BY", paper_id)

            locality_species[loc_id].append(sp_id)

        # ── CO_OCCURS_WITH edges ──────────────────────────────────────────────
        for loc_id, sp_ids in locality_species.items():
            unique_sps = list(set(sp_ids))
            if len(unique_sps) < _MIN_COOCCUR:
                continue
            for i, a in enumerate(unique_sps):
                for b in unique_sps[i + 1:]:
                    self._upsert_edge(a, "CO_OCCURS_WITH", b, props={"locality": loc_id})
                    self._upsert_edge(b, "CO_OCCURS_WITH", a, props={"locality": loc_id})

        self._conn.commit()
        added = self._G.number_of_nodes() - before
        logger.info(
            "[KG] Ingested %d occurrences → +%d nodes (total %d)",
            len(occurrences),
            added,
            self._G.number_of_nodes(),
        )
        return added

    # ── Graph statistics ──────────────────────────────────────────────────────
    def stats(self) -> dict:
        G = self._G
        type_counts = defaultdict(int)
        for _, d in G.nodes(data=True):
            type_counts[d.get("node_type", "?")] += 1

        rel_counts = defaultdict(int)
        for _, _, d in G.edges(data=True):
            rel_counts[d.get("rel_type", "?")] += 1

        top_species = sorted(
            [(n, d.get("degree", G.degree(n)))
             for n, d in G.nodes(data=True)
             if d.get("node_type") == "Species"],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return {
            "total_nodes":    G.number_of_nodes(),
            "total_edges":    G.number_of_edges(),
            "node_types":     dict(type_counts),
            "edge_types":     dict(rel_counts),
            "top_species_by_degree": [
                (self._G.nodes[n].get("label", n), deg)
                for n, deg in top_species
            ],
            "density":       nx.density(nx.Graph(G)),
        }

    # ── Community detection ───────────────────────────────────────────────────
    def detect_communities(self) -> dict[int, list[str]]:
        """
        Louvain-style greedy community detection on the undirected projection.
        Returns {community_id: [node_labels]}.
        """
        UG = nx.Graph(self._G)  # collapse to undirected
        if UG.number_of_nodes() == 0:
            return {}
        try:
            comms = nx.community.greedy_modularity_communities(UG, weight="weight")
        except Exception as exc:
            logger.warning("[KG] Community detection failed: %s", exc)
            return {}

        result = {}
        self._conn.execute("DELETE FROM kg_communities")
        for i, comm in enumerate(comms):
            labels = [self._G.nodes[n].get("label", n) for n in comm if self._G.has_node(n)]
            types  = [self._G.nodes[n].get("node_type", "") for n in comm if self._G.has_node(n)]
            sp_labels = [
                labels[j] for j, t in enumerate(types) if t == "Species"
            ]
            keywords = sp_labels[:8]
            key = f"comm_{i}"
            self._conn.execute(
                """INSERT OR REPLACE INTO kg_communities
                   (community_key, node_ids, keywords)
                   VALUES(?,?,?)""",
                (key, json.dumps(list(comm)), json.dumps(keywords)),
            )
            result[i] = labels
        self._conn.commit()
        logger.info("[KG] %d communities detected", len(result))
        return result

    # ── GraphRAG context builder ──────────────────────────────────────────────
    def build_rag_context(self, query: str, top_k: int = 8) -> str:
        """
        Build a structured context string for an LLM prompt using graph
        neighbourhood rather than plain vector similarity.

        Strategy:
          1. Extract candidate entity names from the query (simple regex).
          2. Find those nodes in the graph.
          3. Return their 2-hop neighbourhood as structured text.
          4. Append community summary if available.
        """
        # Simple lexical match — no embeddings needed
        tokens = re.findall(r"[A-Z][a-z]+ ?[a-z]*", query)
        candidate_ids: list[str] = []

        for tok in tokens:
            slug = _slugify(tok)
            for nid in self._G.nodes():
                node_label = self._G.nodes[nid].get("label", "").lower()
                if slug in node_label or tok.lower() in node_label:
                    candidate_ids.append(nid)

        if not candidate_ids:
            # Fall back to highest-degree species nodes
            candidate_ids = sorted(
                [n for n, d in self._G.nodes(data=True) if d.get("node_type") == "Species"],
                key=lambda n: self._G.degree(n),
                reverse=True,
            )[:top_k]

        # Build subgraph context
        lines = ["=== KNOWLEDGE GRAPH CONTEXT ==="]
        seen: set = set()
        for nid in candidate_ids[:top_k]:
            if nid in seen:
                continue
            seen.add(nid)
            nd = self._G.nodes.get(nid, {})
            lines.append(
                f"\n[{nd.get('node_type','?')}] {nd.get('label', nid)}"
            )

            # Outgoing edges
            for _, tgt, ed in self._G.out_edges(nid, data=True):
                tgt_label = self._G.nodes.get(tgt, {}).get("label", tgt)
                rel = ed.get("rel_type", "→")
                hab = ed.get("habitat", "")
                depth = ed.get("depth_m", "")
                detail = " | ".join(filter(None, [hab, f"{depth}m" if depth else ""]))
                lines.append(
                    f"  —{rel}→ {tgt_label}" + (f" [{detail}]" if detail else "")
                )

        # Community context
        rows = self._conn.execute(
            "SELECT keywords, summary FROM kg_communities ORDER BY community_id LIMIT 5"
        ).fetchall()
        if rows:
            lines.append("\n=== COMMUNITY CLUSTERS ===")
            for keywords_json, summary in rows:
                kws = json.loads(keywords_json or "[]")
                if kws:
                    lines.append(f"  Cluster: {', '.join(kws[:6])}")
                if summary:
                    lines.append(f"  Summary: {summary}")

        return "\n".join(lines)

    def graph_rag_query(
        self,
        query: str,
        llm_fn: Callable[[str], str],
        top_k: int = 8,
    ) -> str:
        """
        Full GraphRAG pipeline:
          build_rag_context → inject into prompt → call LLM → return response.

        llm_fn: callable that accepts a prompt string and returns a response string.
        """
        context = self.build_rag_context(query, top_k=top_k)
        prompt = (
            f"You are a marine biodiversity expert. Use the knowledge graph context "
            f"below to answer the question accurately. Do not hallucinate species "
            f"or localities not in the context.\n\n"
            f"{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"Answer with taxonomic precision, citing species names and localities."
        )
        try:
            return llm_fn(prompt)
        except Exception as exc:
            logger.error("[KG] GraphRAG LLM call failed: %s", exc)
            return f"[GraphRAG Error] {exc}\n\nContext preview:\n{context[:500]}"

    # ── Neighbourhood queries ─────────────────────────────────────────────────
    def get_species_at_locality(self, locality_name: str) -> list[str]:
        """Return species names found at a given locality (fuzzy match)."""
        from rapidfuzz import process as rf_process

        locality_nodes = {
            nid: d.get("label", "")
            for nid, d in self._G.nodes(data=True)
            if d.get("node_type") == "Locality"
        }
        if not locality_nodes:
            return []

        match = rf_process.extractOne(
            locality_name,
            locality_nodes,
            score_cutoff=70,
        )
        if not match:
            return []

        _, _, loc_id = match
        species = []
        for src, tgt, d in self._G.in_edges(loc_id, data=True):
            if d.get("rel_type") == "FOUND_AT":
                nd = self._G.nodes.get(src, {})
                if nd.get("node_type") == "Species":
                    species.append(nd.get("label", src))
        return sorted(set(species))

    def get_co_occurring_species(self, species_name: str, min_weight: int = 1) -> list[str]:
        """Return species that co-occur with the given species."""
        from rapidfuzz import process as rf_process

        sp_nodes = {
            nid: d.get("label", "")
            for nid, d in self._G.nodes(data=True)
            if d.get("node_type") == "Species"
        }
        match = rf_process.extractOne(species_name, sp_nodes, score_cutoff=80)
        if not match:
            return []

        _, _, sp_id = match
        cooccurring = []
        for _, tgt, d in self._G.out_edges(sp_id, data=True):
            if d.get("rel_type") == "CO_OCCURS_WITH":
                w = self._conn.execute(
                    "SELECT weight FROM kg_edges WHERE src_id=? AND tgt_id=? AND rel_type='CO_OCCURS_WITH'",
                    (sp_id, tgt),
                ).fetchone()
                if w and w[0] >= min_weight:
                    tnd = self._G.nodes.get(tgt, {})
                    cooccurring.append((tnd.get("label", tgt), w[0]))
        return [sp for sp, _ in sorted(cooccurring, key=lambda x: -x[1])]

    def get_family_species_list(self, family_name: str) -> list[str]:
        """Return all species belonging to a taxonomic family."""
        fam_id = _node_id("Family", family_name)
        if not self._G.has_node(fam_id):
            return []
        species = []
        for src, tgt, d in self._G.in_edges(fam_id, data=True):
            if d.get("rel_type") == "BELONGS_TO":
                nd = self._G.nodes.get(src, {})
                if nd.get("node_type") == "Species":
                    species.append(nd.get("label", src))
        return sorted(set(species))

    # ── Visualisation ─────────────────────────────────────────────────────────
    def export_pyvis_html(
        self,
        output_path: str = "kg_visualization.html",
        max_nodes: int = 200,
        filter_types: list[str] | None = None,
    ) -> str:
        """
        Export an interactive PyVis HTML visualization of the knowledge graph.
        Filters to top max_nodes by degree to keep it readable.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            logger.error("[KG] pyvis not installed — pip install pyvis")
            return ""

        # Subgraph selection
        filter_types = filter_types or list(NODE_COLORS.keys())
        candidate_nodes = [
            n for n, d in self._G.nodes(data=True)
            if d.get("node_type") in filter_types
        ]
        top_nodes = sorted(
            candidate_nodes,
            key=lambda n: self._G.degree(n),
            reverse=True,
        )[:max_nodes]

        SG = self._G.subgraph(top_nodes).copy()

        net = Network(
            height="800px",
            width="100%",
            bgcolor="#1a1a2e",
            font_color="#e0e0e0",
            directed=True,
        )
        net.set_options("""
        {
          "physics": { "enabled": true, "barnesHut": { "gravitationalConstant": -3000 } },
          "edges": { "smooth": { "type": "curvedCW", "roundness": 0.2 } }
        }
        """)

        for nid, nd in SG.nodes(data=True):
            ntype = nd.get("node_type", "?")
            label = nd.get("label", nid)[:40]
            color = NODE_COLORS.get(ntype, "#888888")
            size  = {
                "Species": 15, "Locality": 20, "Habitat": 18,
                "Family": 12, "Order": 10, "Phylum": 8, "Paper": 8,
            }.get(ntype, 12)
            worms = nd.get("wormsID", "")
            title = f"<b>{label}</b> [{ntype}]"
            if worms:
                title += f'<br><a href="https://www.marinespecies.org/aphia.php?p=taxdetails&id={worms}" target="_blank">WoRMS #{worms}</a>'
            net.add_node(
                nid, label=label, color=color, size=size,
                title=title, font={"size": 10},
            )

        for src, tgt, ed in SG.edges(data=True):
            if src not in top_nodes or tgt not in top_nodes:
                continue
            rel   = ed.get("rel_type", "→")
            style = EDGE_STYLES.get(rel, {})
            net.add_edge(
                src, tgt,
                title=rel,
                color=style.get("color", "#aaaaaa"),
                width=style.get("width", 1),
                dashes=style.get("dashes", False),
                arrows="to",
            )

        net.save_graph(output_path)
        logger.info("[KG] PyVis HTML saved → %s", output_path)
        return output_path

    def export_graphml(self, output_path: str = "kg_export.graphml") -> str:
        """Export to GraphML for Gephi / Cytoscape."""
        nx.write_graphml(self._G, output_path)
        logger.info("[KG] GraphML saved → %s", output_path)
        return output_path

    def to_plotly_figure(self, max_nodes: int = 150):
        """Return a Plotly figure of the graph (no external HTML file needed)."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        top_nodes = sorted(
            self._G.nodes(),
            key=lambda n: self._G.degree(n),
            reverse=True,
        )[:max_nodes]
        SG = self._G.subgraph(top_nodes)

        pos = nx.spring_layout(SG, seed=42, k=0.8)

        edge_x, edge_y = [], []
        for src, tgt in SG.edges():
            x0, y0 = pos.get(src, (0, 0))
            x1, y1 = pos.get(tgt, (0, 0))
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edges_trace = go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.5, color="#aaaaaa"),
            hoverinfo="none",
        )

        traces = [edges_trace]
        for ntype, color in NODE_COLORS.items():
            type_nodes = [n for n in SG.nodes() if SG.nodes[n].get("node_type") == ntype]
            if not type_nodes:
                continue
            nx_vals = [pos.get(n, (0, 0))[0] for n in type_nodes]
            ny_vals = [pos.get(n, (0, 0))[1] for n in type_nodes]
            labels  = [SG.nodes[n].get("label", n)[:30] for n in type_nodes]
            traces.append(go.Scatter(
                x=nx_vals, y=ny_vals, mode="markers+text",
                marker=dict(size=10, color=color),
                text=labels,
                textposition="top center",
                textfont=dict(size=8),
                name=ntype,
                hovertemplate="%{text}<extra></extra>",
            ))

        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title="BioTrace Knowledge Graph",
                showlegend=True,
                hovermode="closest",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#1a1a2e",
                font=dict(color="white"),
            ),
        )
        return fig

    def close(self):
        if self._conn:
            self._conn.close()
