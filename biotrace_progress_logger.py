"""
biotrace_progress_logger.py  —  BioTrace v5.4
────────────────────────────────────────────────────────────────────────────
Rich progress reporting for species detection + deduplication.

PROBLEM:
  The original log_cb() function emits flat strings like:
    "✅ [Results] 12 records"
    "✅ [Dedup] Removed 3 duplicate entries (stages 1+2)"

  The user cannot see WHICH species were found, which were deduplicated,
  or why a particular locality was rejected. This black-box logging makes
  debugging extraction quality very difficult.

THIS MODULE ADDS:

  1. SpeciesProgressTracker
     Tracks each species through the pipeline stages:
       DETECTED → VALIDATED → DEDUPED → GEOCODED → SAVED
     Maintains per-species counts, locality assignments, and dedup reasons.

  2. render_species_progress_panel(tracker)
     Streamlit widget showing live detection table with colour-coded status.

  3. BioTraceLogger
     Drop-in replacement for log_cb that also feeds the tracker and
     produces structured log output visible in extraction log expander.

INTEGRATION (in biotrace_v5.py extract tab):

    from biotrace_progress_logger import BioTraceLogger, render_species_progress_panel

    logger_inst = BioTraceLogger(log_container, schema_errors)
    progress_panel = st.empty()

    occurrences = extract_occurrences(..., log_cb=logger_inst)

    with progress_panel.container():
        render_species_progress_panel(logger_inst.tracker)
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("biotrace.progress")

# ─────────────────────────────────────────────────────────────────────────────
#  Species pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

class Stage(str, Enum):
    DETECTED   = "detected"    # found by thinker / NER
    EXTRACTED  = "extracted"   # structured JSON record produced
    FILTERED   = "filtered"    # passed post-parse filters
    DEDUPED    = "deduplicated" # survived dedup pipeline
    GEOCODED   = "geocoded"    # has lat/lon
    SAVED      = "saved"       # written to SQLite

# Colour mapping for Streamlit metric display
_STAGE_COLOUR = {
    Stage.DETECTED:  "#5DCAA5",   # teal
    Stage.EXTRACTED: "#378ADD",   # blue
    Stage.FILTERED:  "#EF9F27",   # amber
    Stage.DEDUPED:   "#7F77DD",   # purple
    Stage.GEOCODED:  "#1D9E75",   # dark teal
    Stage.SAVED:     "#085041",   # dark green
}

_STAGE_ICON = {
    Stage.DETECTED:  "🔍",
    Stage.EXTRACTED: "📄",
    Stage.FILTERED:  "🔬",
    Stage.DEDUPED:   "🔗",
    Stage.GEOCODED:  "📍",
    Stage.SAVED:     "💾",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Per-species tracking record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpeciesRecord:
    name:             str
    recorded_names:   set[str]    = field(default_factory=set)
    localities:       list[str]   = field(default_factory=list)
    occurrence_types: list[str]   = field(default_factory=list)
    stage:            Stage       = Stage.DETECTED
    dedup_reason:     str         = ""    # why it was merged/removed in dedup
    lat:              Optional[float] = None
    lon:              Optional[float] = None
    db_id:            Optional[int]   = None
    chunk_sources:    list[str]   = field(default_factory=list)

    def advance(self, new_stage: Stage) -> None:
        """Advance to the next pipeline stage."""
        self.stage = new_stage

    def add_locality(self, loc: str) -> None:
        if loc and loc not in self.localities:
            self.localities.append(loc)

    def summary(self) -> str:
        locs = ", ".join(self.localities[:3])
        locs_extra = f" (+{len(self.localities)-3} more)" if len(self.localities) > 3 else ""
        return f"{self.name} | {self.stage.value} | locs: {locs}{locs_extra}"


# ─────────────────────────────────────────────────────────────────────────────
#  Tracker
# ─────────────────────────────────────────────────────────────────────────────

class SpeciesProgressTracker:
    """
    Tracks species through all BioTrace pipeline stages.

    Updated by BioTraceLogger as events arrive. Can be queried at any
    point to render a progress table.
    """

    def __init__(self):
        self._species: dict[str, SpeciesRecord] = {}
        self._discarded: list[tuple[str, str]] = []   # (name, reason)
        self._start_time = time.time()
        self.chunk_count     = 0
        self.error_count     = 0
        self.records_raw     = 0
        self.records_deduped = 0
        self.records_geocoded= 0
        self.records_saved   = 0

    def _key(self, name: str) -> str:
        return name.strip().lower()

    def on_detected(self, names: list[str], chunk_section: str = "") -> None:
        """Called when thinker / NER detects species names in a chunk."""
        for name in names:
            k = self._key(name)
            if k not in self._species:
                self._species[k] = SpeciesRecord(name=name)
            self._species[k].chunk_sources.append(chunk_section)

    def on_extracted(self, records: list[dict], chunk_section: str = "") -> None:
        """Called after LLM structured extraction. One call per chunk."""
        self.records_raw += len(records)
        self.chunk_count += 1
        for rec in records:
            if not isinstance(rec, dict):
                continue
            name = str(rec.get("validName") or rec.get("recordedName") or
                       rec.get("Recorded Name", "")).strip()
            if not name:
                continue
            k = self._key(name)
            if k not in self._species:
                self._species[k] = SpeciesRecord(name=name)
            sp = self._species[k]
            sp.recorded_names.add(str(rec.get("recordedName") or rec.get("Recorded Name", "")))
            sp.advance(Stage.EXTRACTED)
            loc = str(rec.get("verbatimLocality", "")).strip()
            if loc and loc.lower() not in ("not reported", "unknown", ""):
                sp.add_locality(loc)
            occ_type = str(rec.get("occurrenceType", "")).strip()
            if occ_type:
                sp.occurrence_types.append(occ_type)

    def on_filtered(
        self,
        passed: list[dict],
        discarded_ls: list[dict],
        discarded_loc: list[dict],
    ) -> None:
        """Called after post-parse filters."""
        for rec in discarded_ls:
            name = str(rec.get("recordedName") or rec.get("Recorded Name", "")).strip()
            if name:
                self._discarded.append((name, "life-stage term"))
                k = self._key(name)
                self._species.pop(k, None)

        for rec in discarded_loc:
            name = str(rec.get("recordedName") or rec.get("Recorded Name", "")).strip()
            if name:
                self._discarded.append((name, "morphology/habitat locality"))

        for rec in passed:
            name = str(rec.get("validName") or rec.get("recordedName") or
                       rec.get("Recorded Name", "")).strip()
            k = self._key(name)
            if k in self._species:
                self._species[k].advance(Stage.FILTERED)

    def on_deduped(self, final: list[dict], removed: list[dict]) -> None:
        """Called after dedup_occurrences + suppress_regional_duplicates."""
        self.records_deduped = len(final)
        for rec in removed:
            name = str(rec.get("validName") or rec.get("recordedName", "")).strip()
            if name:
                self._discarded.append((name, "duplicate / regional suppression"))

        for rec in final:
            name = str(rec.get("validName") or rec.get("recordedName", "")).strip()
            k = self._key(name)
            if k in self._species:
                self._species[k].advance(Stage.DEDUPED)

    def on_geocoded(self, records: list[dict]) -> None:
        """Called after geocoding cascade."""
        for rec in records:
            if not isinstance(rec, dict):
                continue
            name = str(rec.get("validName") or rec.get("recordedName", "")).strip()
            k = self._key(name)
            if k in self._species:
                lat = rec.get("decimalLatitude")
                lon = rec.get("decimalLongitude")
                if lat is not None and lon is not None:
                    self._species[k].lat = float(lat)
                    self._species[k].lon = float(lon)
                    self._species[k].advance(Stage.GEOCODED)
                    self.records_geocoded += 1

    def on_saved(self, n_saved: int) -> None:
        self.records_saved = n_saved
        for sp in self._species.values():
            if sp.stage == Stage.GEOCODED or sp.stage == Stage.DEDUPED:
                sp.advance(Stage.SAVED)

    def on_error(self, section: str, msg: str) -> None:
        self.error_count += 1
        logger.warning("[tracker] chunk error in %s: %s", section, msg)

    # ── Query helpers ────────────────────────────────────────────────────────

    def unique_species(self) -> list[SpeciesRecord]:
        return sorted(self._species.values(), key=lambda s: s.name)

    def by_stage(self, stage: Stage) -> list[SpeciesRecord]:
        return [s for s in self._species.values() if s.stage == stage]

    def stage_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for sp in self._species.values():
            counts[sp.stage.value] += 1
        return dict(counts)

    def elapsed(self) -> str:
        s = int(time.time() - self._start_time)
        return f"{s//60}m {s%60}s" if s >= 60 else f"{s}s"

    def summary_log(self) -> str:
        sc = self.stage_counts()
        return (
            f"[Progress] unique={len(self._species)} | "
            f"extracted={sc.get('extracted',0)} | "
            f"deduped={sc.get('deduplicated',0)} | "
            f"geocoded={sc.get('geocoded',0)} | "
            f"saved={self.records_saved} | "
            f"discarded={len(self._discarded)} | "
            f"elapsed={self.elapsed()}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Drop-in log_cb replacement
# ─────────────────────────────────────────────────────────────────────────────

class BioTraceLogger:
    """
    Drop-in replacement for the inline log_cb closure in biotrace_v5.py.

    Usage:
        bl = BioTraceLogger(log_container, schema_errors_list)
        occurrences = extract_occurrences(..., log_cb=bl)
        render_species_progress_panel(bl.tracker)
    """

    def __init__(self, log_container=None, schema_errors: list | None = None):
        self.tracker       = SpeciesProgressTracker()
        self._container    = log_container
        self._errors       = schema_errors if schema_errors is not None else []
        self._logs: list[str] = []

    def __call__(self, msg: str, lvl: str = "ok") -> None:
        """Callable interface matching original log_cb signature."""
        self._logs.append(f"[{lvl.upper()}] {msg}")
        if lvl == "warn":
            self._errors.append(msg)
        if self._container is not None:
            try:
                import streamlit as st
                icon = {"ok": "✅", "warn": "⚠️", "error": "❌"}.get(lvl, "ℹ️")
                with self._container:
                    st.write(f"{icon} {msg}")
            except Exception:
                pass

        # Feed tracker from structured log messages
        self._parse_event(msg, lvl)

    def _parse_event(self, msg: str, lvl: str) -> None:
        import re
        msg = msg.strip()

        # "[BiodiViz NER] 18 organisms" / "[Thinker] N names pre-identified"
        m = re.search(r"\[(BiodiViz NER|Thinker[^\]]*)\]\s+(\d+)", msg)
        if m:
            n = int(m.group(2))
            # Synthetic keys so unique count is correct; real names arrive via log_extraction_result
            section = self.tracker.chunk_count
            self.tracker.on_detected(
                [f"__candidate_{section}_{i}" for i in range(n)],
                chunk_section=m.group(1),
            )
            return

        # "  [Family: Aplysiidae Aplysia oculifera…] 1 records"
        m = re.search(r"\[(.+?)\]\s+(\d+) records?$", msg)
        if m:
            self.tracker.chunk_count += 1
            self.tracker.records_raw += int(m.group(2))
            return

        # "[Dedup] Removed N duplicates" / "[Dedup/Stage3] Suppressed N"
        m = re.search(r"\[Dedup[^\]]*\].*?(?:Removed|Suppressed)\s+(\d+)", msg)
        if m:
            n = int(m.group(1))
            self.tracker.records_deduped = max(
                0, self.tracker.records_raw - n
            )
            return

        # "[DB] N records saved"
        m = re.search(r"\[DB\]\s+(\d+) records? saved", msg)
        if m:
            self.tracker.on_saved(int(m.group(1)))
            return

        # "[LocalityNER] N/M records have coordinates" / "[Geocoding] Processing N"
        m = re.search(r"\[(?:LocalityNER|Geocoding)[^\]]*\].*?(\d+)(?:/(\d+))? records?", msg)
        if m:
            self.tracker.records_geocoded = int(m.group(1))
            return

    def log_extraction_result(
        self,
        section: str,
        records: list[dict],
        species_detected: list[str] | None = None,
    ) -> None:
        """Call after each chunk's extraction result is available."""
        if species_detected:
            self.tracker.on_detected(species_detected, chunk_section=section)
        self.tracker.on_extracted(records, chunk_section=section)

    def log_filter_result(
        self,
        passed: list[dict],
        discarded_ls: list[dict],
        discarded_loc: list[dict],
    ) -> None:
        self.tracker.on_filtered(passed, discarded_ls, discarded_loc)

    def log_dedup_result(self, final: list[dict], removed: list[dict]) -> None:
        self.tracker.on_deduped(final, removed)

    def log_geocoded(self, records: list[dict]) -> None:
        self.tracker.on_geocoded(records)

    def log_saved(self, n: int) -> None:
        self.tracker.on_saved(n)
        self(self.tracker.summary_log())

    @property
    def logs(self) -> list[str]:
        return self._logs


# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_species_progress_panel(tracker: SpeciesProgressTracker) -> None:
    """
    Render a live species progress table in Streamlit.
    Call inside an st.empty().container() for live updates.

    Shows:
      • Pipeline stage counts (metric cards)
      • Per-species table: name | recorded_as | localities | stage | coords
      • Discarded records list
    """
    try:
        import streamlit as st
        import pandas as pd
    except ImportError:
        return

    st.markdown("#### Species Detection Progress")

    # Stage counts row
    sc = tracker.stage_counts()
    stage_order = [Stage.DETECTED, Stage.EXTRACTED, Stage.FILTERED,
                   Stage.DEDUPED, Stage.GEOCODED, Stage.SAVED]
    cols = st.columns(len(stage_order))
    for col, stage in zip(cols, stage_order):
        col.metric(
            label=f"{_STAGE_ICON[stage]} {stage.value.title()}",
            value=sc.get(stage.value, 0),
        )

    # Pipeline summary
    elapsed = tracker.elapsed()
    st.caption(
        f"Chunks processed: {tracker.chunk_count} | "
        f"Errors: {tracker.error_count} | "
        f"Elapsed: {elapsed}"
    )

    # Species table
    species_list = tracker.unique_species()
    if species_list:
        rows = []
        for sp in species_list:
            has_coords = sp.lat is not None and sp.lon is not None
            rows.append({
                "Species": sp.name,
                "Recorded as": " / ".join(list(sp.recorded_names)[:2]) or sp.name,
                "Localities": "; ".join(sp.localities[:3]),
                "Occ. types": ", ".join(set(sp.occurrence_types))[:20],
                "Stage": f"{_STAGE_ICON[sp.stage]} {sp.stage.value}",
                "Coords": f"{sp.lat:.4f}, {sp.lon:.4f}" if has_coords else "—",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=min(400, 60 + 35 * len(rows)))
    else:
        st.info("No species detected yet.")

    # Discarded records
    if tracker._discarded:
        with st.expander(f"🗑️ {len(tracker._discarded)} discarded records"):
            for name, reason in tracker._discarded:
                st.write(f"• **{name}** — *{reason}*")


def render_dedup_audit_log(tracker: SpeciesProgressTracker) -> None:
    """
    Shows a clean audit table of which records were kept vs merged vs suppressed.
    Useful for debugging dedup quality.
    """
    try:
        import streamlit as st
        import pandas as pd
    except ImportError:
        return

    st.markdown("#### Deduplication Audit")
    col1, col2, col3 = st.columns(3)
    col1.metric("Raw records",      tracker.records_raw)
    col2.metric("After dedup",      tracker.records_deduped)
    col3.metric("Removed",
                tracker.records_raw - tracker.records_deduped
                if tracker.records_deduped else 0)

    if tracker._discarded:
        rows = [
            {"Name": n, "Reason": r}
            for n, r in tracker._discarded
        ]
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            height=min(300, 60 + 35 * len(rows)),
        )
