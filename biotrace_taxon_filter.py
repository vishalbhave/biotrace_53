"""
biotrace_taxon_filter.py  —  BioTrace v5.6  Dynamic Taxonomic Filter Widget
═══════════════════════════════════════════════════════════════════════════════
Provides a hierarchical, dynamically-populated multi-select filter panel
that reads distinct taxonomic values directly from the occurrence database
and refreshes wiki species data accordingly.

Inspired by the Streamlit Dynamic Multi-Select Filters pattern:
  https://discuss.streamlit.io/t/new-component-dynamic-multi-select-filters/49595

Design
──────
• Reads kingdom / phylum / class_ / order_ / family_ from occurrences_v4
• Each level narrows the options of the next (true hierarchical cascade)
• Selected filters returned as a dict for caller to apply to any DataFrame
• Optional: fetch matching wiki articles dynamically
• All species supported (not just marine)

Public API
──────────
    from biotrace_taxon_filter import TaxonFilterWidget

    widget = TaxonFilterWidget(meta_db_path="biodiversity_data/metadata_v5.db")

    # In Streamlit sidebar or main area:
    filters = widget.render()
    # filters = {"phylum": ["Mollusca"], "family_": ["Chromodorididae"], ...}

    # Apply to a DataFrame:
    filtered_df = widget.apply_filters(df, filters)

    # Get matching species names for wiki lookup:
    species_list = widget.get_filtered_species(filters)
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Optional

logger = logging.getLogger("biotrace.taxon_filter")


def _resolve_table(conn: sqlite3.Connection) -> str:
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    for t in ("occurrences_v4", "occurrences"):
        if t in names:
            return t
    raise sqlite3.OperationalError("No occurrence table found.")


class TaxonFilterWidget:
    """
    Hierarchical taxonomic multi-select filter panel for Streamlit.

    Cascade order:  kingdom → phylum → class_ → order_ → family_ → species
    Each level is filtered by all selections above it.
    """

    LEVELS = [
        ("kingdom",  "🌍 Kingdom"),
        ("phylum",   "🔬 Phylum"),
        ("class_",   "📦 Class"),
        ("order_",   "📋 Order"),
        ("family_",  "🏷️ Family"),
    ]

    def __init__(self, meta_db_path: str):
        self.db_path = meta_db_path

    # ── Data retrieval ────────────────────────────────────────────────────────

    def _get_distinct(
        self, col: str, where_clause: str = "", params: tuple = ()
    ) -> list[str]:
        """Return distinct non-null, non-empty values for a column."""
        try:
            conn = sqlite3.connect(self.db_path)
            table = _resolve_table(conn)
            sql = (
                f"SELECT DISTINCT {col} FROM {table} "
                f"WHERE {col} IS NOT NULL AND trim({col}) != ''"
            )
            if where_clause:
                sql += f" AND ({where_clause})"
            sql += f" ORDER BY {col} COLLATE NOCASE"
            rows = conn.execute(sql, params).fetchall()
            conn.close()
            return [r[0] for r in rows if r[0]]
        except Exception as exc:
            logger.debug("[TaxonFilter] _get_distinct %s: %s", col, exc)
            return []

    def _build_where(self, selections: dict[str, list[str]]) -> tuple[str, tuple]:
        """Build SQL WHERE fragment + params from current selections."""
        clauses, params = [], []
        for col, vals in selections.items():
            if vals:
                placeholders = ",".join("?" * len(vals))
                clauses.append(f"{col} IN ({placeholders})")
                params.extend(vals)
        return (" AND ".join(clauses), tuple(params))

    # ── Species query ─────────────────────────────────────────────────────────

    def get_filtered_species(self, selections: dict[str, list[str]]) -> list[str]:
        """Return distinct validName values matching current filter selections."""
        try:
            where, params = self._build_where(selections)
            conn = sqlite3.connect(self.db_path)
            table = _resolve_table(conn)
            sql = (
                f"SELECT DISTINCT validName FROM {table} "
                f"WHERE validName IS NOT NULL AND trim(validName) != ''"
            )
            if where:
                sql += f" AND {where}"
            sql += " ORDER BY validName COLLATE NOCASE"
            rows = conn.execute(sql, params).fetchall()
            conn.close()
            return [r[0] for r in rows if r[0]]
        except Exception as exc:
            logger.warning("[TaxonFilter] get_filtered_species: %s", exc)
            return []

    def get_filtered_occurrences(self, selections: dict[str, list[str]]) -> list[dict]:
        """Return full occurrence rows matching filter selections."""
        try:
            where, params = self._build_where(selections)
            conn = sqlite3.connect(self.db_path)
            table = _resolve_table(conn)
            sql = (
                f"SELECT id, validName, recordedName, verbatimLocality, "
                f"decimalLatitude, decimalLongitude, occurrenceType, "
                f"phylum, class_, order_, family_, sourceCitation, "
                f"geocodingSource, taxonomicStatus FROM {table}"
            )
            if where:
                sql += f" WHERE {where}"
            sql += " ORDER BY validName COLLATE NOCASE LIMIT 5000"
            cur = conn.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.close()
            return rows
        except Exception as exc:
            logger.warning("[TaxonFilter] get_filtered_occurrences: %s", exc)
            return []

    # ── DataFrame filter ──────────────────────────────────────────────────────

    def apply_filters(self, df, selections: dict[str, list[str]]):
        """Apply selections to an in-memory pandas DataFrame."""
        try:
            import pandas as pd
            mask = pd.Series([True] * len(df), index=df.index)
            for col, vals in selections.items():
                if vals and col in df.columns:
                    mask &= df[col].isin(vals)
            return df[mask]
        except Exception as exc:
            logger.warning("[TaxonFilter] apply_filters: %s", exc)
            return df

    # ── Streamlit render ──────────────────────────────────────────────────────

    def render(
        self,
        container=None,
        show_species_count: bool = True,
        show_record_count:  bool = True,
        key_prefix:         str  = "txf",
    ) -> dict[str, list[str]]:
        """
        Render hierarchical multi-select filters.

        Returns
        -------
        dict  {db_column: [selected_values, ...], ...}
              Only non-empty selections are included.

        Parameters
        ----------
        container       : Streamlit container (sidebar, column, expander) or None for main area
        show_species_count : show count of matched species below filters
        show_record_count  : show count of matched occurrence records
        key_prefix      : unique key prefix to avoid widget key collisions
        """
        import streamlit as st

        ctx = container or st

        ctx.markdown("### 🔬 Taxonomic Filters")
        ctx.caption(
            "Filters cascade: selecting a Phylum restricts Class options, "
            "and so on. Leave blank to show all."
        )

        selections: dict[str, list[str]] = {}

        for col, label in self.LEVELS:
            # Build options from current upstream selections
            where, params = self._build_where(selections)
            opts = self._get_distinct(col, where, params)

            if not opts:
                continue   # skip levels with no data

            chosen = ctx.multiselect(
                label,
                options=opts,
                default=[],
                key=f"{key_prefix}_{col}",
                help=f"Filter by {label.split()[-1]}. "
                     f"{len(opts)} option(s) available with current upstream selection.",
            )
            if chosen:
                selections[col] = chosen

        # Species-level text search as bonus filter
        sp_search = ctx.text_input(
            "🔍 Species name search",
            key=f"{key_prefix}_sp_search",
            placeholder="e.g. Cassiopea, Acropora sp.",
        )

        # ── Summary metrics ───────────────────────────────────────────────────
        if show_species_count or show_record_count:
            species = self.get_filtered_species(selections)
            if sp_search.strip():
                q = sp_search.strip().lower()
                species = [s for s in species if q in s.lower()]

            if show_species_count:
                ctx.metric("Species matched", len(species))
            if show_record_count:
                recs = self.get_filtered_occurrences(selections)
                if sp_search.strip():
                    q = sp_search.strip().lower()
                    recs = [r for r in recs if q in (r.get("validName") or "").lower()]
                ctx.metric("Occurrence records", len(recs))

        # Attach species search to selections for callers
        if sp_search.strip():
            selections["_sp_search"] = [sp_search.strip()]

        return selections

    def render_sidebar(self, key_prefix: str = "txf_sb") -> dict[str, list[str]]:
        """Convenience: render inside st.sidebar."""
        import streamlit as st
        with st.sidebar:
            return self.render(
                container=st.sidebar,
                key_prefix=key_prefix,
            )

    def render_expander(
        self, label: str = "🔬 Filter by Taxonomy", key_prefix: str = "txf_exp"
    ) -> dict[str, list[str]]:
        """Convenience: render inside a collapsible expander."""
        import streamlit as st
        with st.expander(label, expanded=False):
            return self.render(key_prefix=key_prefix)


# ─────────────────────────────────────────────────────────────────────────────
#  Wiki-aware filtered species list helper
# ─────────────────────────────────────────────────────────────────────────────

def get_wiki_species_for_filter(
    filter_widget: TaxonFilterWidget,
    selections:    dict[str, list[str]],
    wiki,          # BioTraceWikiUnified instance
) -> list[str]:
    """
    Return the intersection of:
      - species matching the taxonomic filter (from occurrence DB)
      - species that have a wiki article
    Both sources needed: DB may have records without wiki, wiki may have
    articles without DB records.
    """
    db_species  = set(filter_widget.get_filtered_species(selections))
    wiki_species = set(wiki.list_species())

    sp_search = selections.get("_sp_search", [""])[0].lower()

    if db_species:
        matched = db_species & wiki_species if wiki_species else db_species
    else:
        matched = wiki_species

    if sp_search:
        matched = {s for s in matched if sp_search in s.lower()}

    return sorted(matched)
