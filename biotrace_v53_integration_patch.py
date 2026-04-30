# ════════════════════════════════════════════════════════════════════════════
#  BioTrace v5.5 — Integration Patch for biotrace_v53.py
#  Unified Wiki Module + CSS + Agent + Chunking Cleanup
# ════════════════════════════════════════════════════════════════════════════
#
#  HOW TO APPLY
#  ─────────────
#  Each block below shows:
#    [LINE RANGE]  — exact lines in biotrace_v53.py to locate
#    ACTION        — REPLACE  / ADD AFTER  / REMOVE
#    OLD CODE      — the exact text currently in the file
#    NEW CODE      — what to put instead
#
#  No other lines are touched.  This is a surgical patch — no full rewrites.
# ════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
#  PATCH 1  — Replace wiki import block
#  Lines 148-155  (the "_WIKI_AVAILABLE / BioTraceWiki" block)
#  ACTION: REPLACE
# ─────────────────────────────────────────────────────────────────────────────
# OLD (lines 148-155):
"""
_WIKI_AVAILABLE = False
BioTraceWiki = None
try:
    from biotrace_wiki import BioTraceWiki
    _WIKI_AVAILABLE = True
    logger.info("[v5] Wiki loaded")
except ImportError:
    logger.warning("[v5] biotrace_wiki.py not found")
"""

# NEW — drop both old wiki classes, import unified replacement:
"""
_WIKI_AVAILABLE = False
BioTraceWikiUnified = None
try:
    from biotrace_wiki_unified import BioTraceWikiUnified, inject_css_streamlit
    _WIKI_AVAILABLE = True
    logger.info("[v5.5] BioTraceWikiUnified loaded (versioned, LLM-enhanced)")
except ImportError:
    logger.warning("[v5.5] biotrace_wiki_unified.py not found — Wiki tab disabled")

# Optional: Ollama Wiki Architect Agent
_WIKI_AGENT_AVAILABLE = False
OllamaWikiAgent = None
try:
    from biotrace_wiki_agent import OllamaWikiAgent
    _WIKI_AGENT_AVAILABLE = True
    logger.info("[v5.5] OllamaWikiAgent loaded")
except ImportError:
    logger.info("[v5.5] biotrace_wiki_agent.py not found — agent panel disabled")
"""
# ─────────────────────────────────────────────────────────────────────────────
#  END PATCH 1
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  PATCH 2  — Remove "Hierarchical late-chunking" checkbox from Chunking expander
#  Lines 2218-2233  (inside the "🧩 Chunking & Extraction Options" expander)
#  ACTION: REMOVE lines 2222-2226  (use_hierarchical checkbox only)
# ─────────────────────────────────────────────────────────────────────────────
# OLD (lines 2222-2226):
"""
            use_hierarchical = st.checkbox(
                "Hierarchical late-chunking (v5.3 — recommended)",
                value=_HIER_CHUNKER_AVAILABLE,
                help="3-level (section→paragraph→sentence) solves species/locality separation",
            )
"""
# NEW: DELETE these 5 lines entirely.
#      The HierarchicalChunker is still used internally as Priority 2 in
#      extract_occurrences() — the user simply no longer needs a separate
#      checkbox for it since ScientificPaperChunker (Priority 1) supersedes it.
#      The variable use_hierarchical is read in extract_occurrences() at the
#      Priority 2 branch.  Replace its source with a derived constant:

# ADD AFTER the expander opening (line 2220, after `col_h1, col_h2 = st.columns(2)`):
"""
        # v5.5: Hierarchical chunking is always-on internally (Priority-2 fallback).
        # The user checkbox has been removed to simplify the UI.
        use_hierarchical = _HIER_CHUNKER_AVAILABLE   # always True when module present
"""
# ─────────────────────────────────────────────────────────────────────────────
#  END PATCH 2
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  PATCH 3  — Replace get_wiki() cache function
#  Lines 1849-1857
#  ACTION: REPLACE
# ─────────────────────────────────────────────────────────────────────────────
# OLD (lines 1849-1857):
"""
@st.cache_resource
def get_wiki() -> "BioTraceWiki | None":
    if not _WIKI_AVAILABLE:
        return None
    try:
        return BioTraceWiki(WIKI_ROOT)
    except Exception as exc:
        logger.error("[v5] Wiki init: %s", exc)
        return None
"""

# NEW:
"""
@st.cache_resource
def get_wiki() -> "BioTraceWikiUnified | None":
    \"\"\"Returns the singleton BioTraceWikiUnified instance (versioned SQLite store).\"\"\"
    if not _WIKI_AVAILABLE:
        return None
    try:
        css_path = os.path.join(os.path.dirname(__file__), "biotrace_wiki.css")
        return BioTraceWikiUnified(
            root_dir = WIKI_ROOT,
            css_path = css_path if os.path.exists(css_path) else None,
        )
    except Exception as exc:
        logger.error("[v5.5] Wiki init: %s", exc)
        return None


@st.cache_resource
def get_wiki_agent() -> "OllamaWikiAgent | None":
    \"\"\"Returns an OllamaWikiAgent bound to the shared wiki store.\"\"\"
    if not _WIKI_AGENT_AVAILABLE:
        return None
    wiki = get_wiki()
    if not wiki:
        return None
    # model / URL come from sidebar — read from session_state with defaults
    model    = st.session_state.get("ollama_model_sel", "qwen2.5:14b")
    base_url = st.session_state.get("ollama_url",       "http://localhost:11434")
    try:
        return OllamaWikiAgent(wiki=wiki, model=model, base_url=base_url)
    except Exception as exc:
        logger.error("[v5.5] WikiAgent init: %s", exc)
        return None
"""
# ─────────────────────────────────────────────────────────────────────────────
#  END PATCH 3
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  PATCH 4  — Replace ingest_into_v5_systems Wiki call to pass chunk_text
#  Lines 1904-1915  (the "# Wiki" block inside ingest_into_v5_systems)
#  ACTION: REPLACE
# ─────────────────────────────────────────────────────────────────────────────
# OLD (lines 1904-1915):
"""
    # Wiki
    wiki = get_wiki() if use_wiki else None  # FIX 2: respect toggle
    if wiki:
        try:
            counts = wiki.update_from_occurrences(
                occurrences, citation=citation,
                llm_fn=llm_fn,
                update_narratives=update_wiki_narratives,
            )
            log_cb(f"[Wiki] Updated: {counts}")
        except Exception as exc:
            log_cb(f"[Wiki] Update error: {exc}", "warn")
"""

# NEW — adds chunk_text parameter so the unified wiki can fire targeted LLM passes:
"""
    # Wiki (BioTraceWikiUnified — versioned, CSS-styled, LLM-enhanced)
    wiki = get_wiki() if use_wiki else None
    if wiki:
        try:
            counts = wiki.update_from_occurrences(
                occurrences,
                citation           = citation,
                llm_fn             = llm_fn,
                update_narratives  = update_wiki_narratives,
                chunk_text         = chunk_text,    # NEW: enables targeted section update
            )
            log_cb(f"[Wiki] Updated: {counts}")
        except Exception as exc:
            log_cb(f"[Wiki] Update error: {exc}", "warn")
"""

# NOTE: The function signature of ingest_into_v5_systems must also accept chunk_text.
#       ADD this parameter to the function definition at line 1860:
# OLD (line 1860):
"""
def ingest_into_v5_systems(
    occurrences: list[dict],
    citation: str,
    session_id: str,
    log_cb,
    provider: str = "",
    model_sel: str = "",
    api_key: str  = "",
    ollama_base_url: str = "http://localhost:11434",
    update_wiki_narratives: bool = False,
    use_kg: bool = True,
    use_mb: bool = True,
    use_wiki: bool = True,
):
"""
# NEW — add chunk_text: str = "" parameter:
"""
def ingest_into_v5_systems(
    occurrences: list[dict],
    citation: str,
    session_id: str,
    log_cb,
    provider: str = "",
    model_sel: str = "",
    api_key: str  = "",
    ollama_base_url: str = "http://localhost:11434",
    update_wiki_narratives: bool = False,
    chunk_text: str = "",       # NEW v5.5 — passed to wiki for LLM section updates
    use_kg: bool = True,
    use_mb: bool = True,
    use_wiki: bool = True,
):
"""
# ─────────────────────────────────────────────────────────────────────────────
#  END PATCH 4
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  PATCH 5  — Replace entire Tab 7 (lines 3063-3279) with unified wiki tab
#  ACTION: REPLACE everything between:
#    "# ═══ TAB 7 — WIKI ═══"  (line 3063)
#  and the next tab boundary or end of file section
# ─────────────────────────────────────────────────────────────────────────────
# OLD (lines 3063-3279): the entire old Tab 7 block
# NEW — drop the old code; insert the following verbatim:

TAB7_NEW_CODE = '''
# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 7 — UNIFIED WIKI  (v5.5)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    # Inject wiki CSS once per session
    inject_css_streamlit()

    wiki = get_wiki()
    if not wiki:
        st.warning(
            "📖 Wiki unavailable — place **biotrace_wiki_unified.py** and "
            "**biotrace_wiki.css** alongside this file and restart."
        )
    else:
        # ── Build the LLM callable (used by manual enhance + auto-narrate) ──
        def _wiki_llm_fn(prompt: str) -> str:
            return call_llm(
                prompt, provider, model_sel, api_key, ollama_url
            )

        # ── Delegate all UI to BioTraceWikiUnified.render_streamlit_tab() ──
        wiki.render_streamlit_tab(
            provider      = provider,
            model_sel     = model_sel,
            api_key       = api_key,
            ollama_url    = ollama_url,
            meta_db       = META_DB_PATH,
            call_llm_fn   = _wiki_llm_fn,
        )

        # ── Optional: Ollama Wiki Architect Agent panel ──────────────────
        agent = get_wiki_agent()
        if agent and _WIKI_AGENT_AVAILABLE:
            st.divider()
            with st.expander("🤖 Ollama Wiki Architect Agent (optional / agentic)", expanded=False):
                agent.render_agent_panel()
'''
# ─────────────────────────────────────────────────────────────────────────────
#  END PATCH 5
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  PATCH 6  — Add css injection call right after st.set_page_config()
#  Line 1935 (after the set_page_config block, before existing st.markdown CSS)
#  ACTION: ADD AFTER line 1939 (`layout="wide",`)
# ─────────────────────────────────────────────────────────────────────────────
# ADD these 4 lines immediately after `st.set_page_config(...)`:
"""
# v5.5: Inject wiki CSS once at startup so it's available in any tab that
# renders wiki HTML via st.components.v1.html().
try:
    inject_css_streamlit()
except Exception:
    pass
"""
# ─────────────────────────────────────────────────────────────────────────────
#  END PATCH 6
# ─────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
#  COMPLETE MODIFIED FUNCTIONS — copy-paste ready
# ════════════════════════════════════════════════════════════════════════════

# ── PATCH 3 COMPLETE (get_wiki replacement) ──────────────────────────────────

def get_wiki_REPLACEMENT():
    """
    Copy this block verbatim to replace lines 1849-1857 in biotrace_v53.py.
    """
    pass


GET_WIKI_CODE = '''
@st.cache_resource
def get_wiki() -> "BioTraceWikiUnified | None":
    """Returns the singleton BioTraceWikiUnified instance (versioned SQLite store)."""
    if not _WIKI_AVAILABLE:
        return None
    try:
        css_path = os.path.join(os.path.dirname(__file__), "biotrace_wiki.css")
        return BioTraceWikiUnified(
            root_dir = WIKI_ROOT,
            css_path = css_path if os.path.exists(css_path) else None,
        )
    except Exception as exc:
        logger.error("[v5.5] Wiki init: %s", exc)
        return None


@st.cache_resource
def get_wiki_agent() -> "OllamaWikiAgent | None":
    """Returns an OllamaWikiAgent bound to the shared wiki store."""
    if not _WIKI_AGENT_AVAILABLE:
        return None
    wiki = get_wiki()
    if not wiki:
        return None
    model    = st.session_state.get("ollama_model_sel", "qwen2.5:14b")
    base_url = st.session_state.get("ollama_url",       "http://localhost:11434")
    try:
        return OllamaWikiAgent(wiki=wiki, model=model, base_url=base_url)
    except Exception as exc:
        logger.error("[v5.5] WikiAgent init: %s", exc)
        return None
'''

# ── PATCH 4 COMPLETE (ingest_into_v5_systems replacement) ───────────────────

INGEST_FN_CODE = '''
def ingest_into_v5_systems(
    occurrences: list[dict],
    citation: str,
    session_id: str,
    log_cb,
    provider: str = "",
    model_sel: str = "",
    api_key: str  = "",
    ollama_base_url: str = "http://localhost:11434",
    update_wiki_narratives: bool = False,
    chunk_text: str = "",       # v5.5 — passed to wiki for LLM section updates
    use_kg: bool = True,
    use_mb: bool = True,
    use_wiki: bool = True,
):
    """Push verified/geocoded occurrences into KG + Memory Bank + Wiki."""
    llm_fn = None
    if update_wiki_narratives:
        def llm_fn(prompt: str) -> str:
            return call_llm(prompt, provider, model_sel, api_key, ollama_base_url)

    # Knowledge Graph
    kg = get_knowledge_graph() if use_kg else None
    if kg:
        try:
            added = kg.ingest_occurrences(occurrences)
            log_cb(f"[KG] +{added} nodes. Total: {kg.stats()[\'total_nodes\']}")
        except Exception as exc:
            log_cb(f"[KG] Ingest error: {exc}", "warn")

    # Memory Bank
    mb = get_memory_bank() if use_mb else None
    if mb:
        try:
            r = mb.store_occurrences(
                occurrences, session_id=session_id,
                session_title=citation, source_file=session_id,
            )
            log_cb(
                f"[MemoryBank] inserted={r[\'inserted\']} merged={r[\'merged\']} "
                f"conflicts={r[\'conflicts\']}"
            )
        except Exception as exc:
            log_cb(f"[MemoryBank] Store error: {exc}", "warn")

    # Wiki — BioTraceWikiUnified (versioned, CSS-styled, LLM-enhanced)
    wiki = get_wiki() if use_wiki else None
    if wiki:
        try:
            counts = wiki.update_from_occurrences(
                occurrences,
                citation           = citation,
                llm_fn             = llm_fn,
                update_narratives  = update_wiki_narratives,
                chunk_text         = chunk_text,
            )
            log_cb(f"[Wiki] Updated: {counts}")
        except Exception as exc:
            log_cb(f"[Wiki] Update error: {exc}", "warn")
'''


# ════════════════════════════════════════════════════════════════════════════
#  SUMMARY OF ALL CHANGES
# ════════════════════════════════════════════════════════════════════════════
CHANGES = """
PATCH 1  Lines 148-155   — Wiki import: BioTraceWiki → BioTraceWikiUnified
                            + OllamaWikiAgent optional import
PATCH 2  Lines 2222-2226 — Remove "Hierarchical late-chunking" checkbox UI;
                            keep internal use_hierarchical as derived constant
PATCH 3  Lines 1849-1857 — get_wiki() returns BioTraceWikiUnified;
                            add get_wiki_agent() factory
PATCH 4  Lines 1860-1873 — ingest_into_v5_systems: add chunk_text param;
          Lines 1904-1915   update wiki call to pass chunk_text
PATCH 5  Lines 3063-3279 — Replace entire old Tab 7 body with
                            wiki.render_streamlit_tab() + agent panel
PATCH 6  Line 1939+      — inject_css_streamlit() after set_page_config()

Files added to project directory:
  biotrace_wiki_unified.py   — drop-in BioTraceWiki replacement
  biotrace_wiki.css          — Wikipedia-dark CSS stylesheet
  biotrace_wiki_agent.py     — optional Ollama multi-agent pipeline
"""

if __name__ == "__main__":
    print(CHANGES)
