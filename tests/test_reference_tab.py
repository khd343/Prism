import ast
import html
import os

from ui.ui_components import _RAW_GLOSSARY
from ui.ui_reference import render_reference

_REF = os.path.join(os.path.dirname(__file__), "..", "ui", "ui_reference.py")


def test_renders_every_glossary_term():
    """Coverage guard: an empty query renders ALL 173 terms (escaping-aware)."""
    out = render_reference(_RAW_GLOSSARY, "")
    missing = [t for t in _RAW_GLOSSARY if html.escape(t) not in out]
    assert not missing, f"Reference dropped {len(missing)} terms: {missing[:5]}"


def test_query_filters_to_strict_subset():
    """A distinctive query narrows to its term and excludes unrelated ones."""
    out = render_reference(_RAW_GLOSSARY, "Dilution Vampire")
    assert html.escape("Dilution Vampire") in out
    assert html.escape("ROCE Current") not in out          # unrelated term filtered out
    assert out != render_reference(_RAW_GLOSSARY, "")        # strictly fewer than all


def test_no_match_is_graceful():
    """A query that matches nothing returns a friendly message, not broken/empty HTML."""
    out = render_reference(_RAW_GLOSSARY, "zzzznotarealterm")
    assert "No terms match" in out


def test_special_chars_are_escaped():
    """Definitions contain '<' — they must be HTML-escaped, not injected raw."""
    out = render_reference({"X": "Nano (<₹100 Cr)"}, "")
    assert "&lt;₹100" in out and "(<₹100" not in out


def test_render_is_pure_no_streamlit_calls():
    """Structural purity: ui_reference.py makes ZERO st.* calls (widgets live in app.py)."""
    tree = ast.parse(open(_REF, encoding="utf-8").read(), filename=_REF)
    st_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "st"
    ]
    assert not st_calls, "ui_reference.py must be pure HTML — no st.* calls"
    assert isinstance(render_reference({"Foo": "bar"}, ""), str)


# ── Framework registry ON SCREEN (2026-08-28) ────────────────────────────────────────────────
def test_frameworks_are_searchable_on_screen():
    """Before 2026-08-28 the Markdown export documented all 37 frameworks while on-screen search
    could find only 6 — the answer to "what is Dhandho Asymmetry?" lived in the download but not
    the search box. The on-screen section renders from the SAME _FW_META-derived dict the export
    consumes, so the two can never disagree."""
    from ui.ui_reference import render_frameworks
    from ui.ui_tearsheet import _FW_META
    fw = {name: {"emoji": meta[1], "name": name, "desc": meta[2]} for name, meta in _FW_META.items()}
    html_all = render_frameworks(fw, "")
    for name in ("Dhandho Asymmetry", "QGLP", "100-Bagger"):
        assert name in html_all, f"{name} missing from the on-screen framework registry"
    assert render_frameworks(fw, "dhandho") != ""
    assert render_frameworks(fw, "zzz-no-such-framework") == ""


def test_app_renders_the_framework_registry():
    """Updated 2026-08-29 for the two-mode layout: the registry renders through _show_frameworks,
    which must be wired into BOTH modes (see test_reference_two_mode_layout for the full contract)."""
    import io as _io, os
    src = _io.open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8").read()
    assert "render_frameworks(_fw_md, q)" in src, (
        "app.py no longer renders the framework registry on screen — the export/search asymmetry is back"
    )
    assert "Framework Registry" in src


# ── Wealth Creation Studies corpus (2026-08-29) ──────────────────────────────────────────────
def test_wcs_studies_entries_are_complete_and_verified_shape():
    """The honesty contract: an entry ships only after a COMPLETE read of the study, and every
    entry carries the full schema (a study summarized without its 'In PRISM' mapping, or with an
    empty summary, is a half-entry that must not ship). Entries stay in study order."""
    from ui.ui_reference_data import WCS_STUDIES
    assert 1 <= len(WCS_STUDIES) <= 30
    for s in WCS_STUDIES:
        for key in ("study", "years", "pub", "theme", "says", "prism"):
            assert len(str(s.get(key, "")).strip()) > 0, f"{s.get('study')} missing {key}"
        assert len(s["says"]) >= 200, f"{s['study']}: summary under the depth floor — not 'best explained'"
        assert len(s["prism"]) >= 80, f"{s['study']}: the engine mapping is too thin"
    nums = [int(str(s["study"]).split()[0].rstrip("stndrh")) for s in WCS_STUDIES]
    assert nums == sorted(nums), "studies must stay in study-number order"


def test_wcs_studies_are_searchable_on_screen():
    """The corpus obeys the same search grammar as the other four: token-AND over every field,
    empty string when nothing matches, everything when the query is empty."""
    from ui.ui_reference import render_wcs_studies
    from ui.ui_reference_data import WCS_STUDIES
    html_all = render_wcs_studies(WCS_STUDIES, "")
    for s in WCS_STUDIES:
        assert s["study"] in html_all
    assert render_wcs_studies(WCS_STUDIES, "margin of safety") != ""
    assert render_wcs_studies(WCS_STUDIES, "zzz-no-such-study") == ""


def test_wcs_studies_ride_into_the_markdown_download():
    """One generator, one source: the download emits the SAME list the screen renders — the
    export/search asymmetry that bit the frameworks (2026-08-28) cannot recur here."""
    from ui.ui_reference import build_reference_markdown
    from ui.ui_reference_data import WCS_STUDIES
    md = build_reference_markdown({"T": "d"}, {"C": [("l", "e" * 45)]}, {}, studies=WCS_STUDIES)
    assert "## Wealth Creation Studies" in md
    for s in WCS_STUDIES:
        assert s["study"] in md and s["theme"] in md


def test_app_renders_the_wcs_studies_corpus():
    """Updated 2026-08-29 for the two-mode layout: the corpus renders through _show_wcs."""
    import io as _io, os
    src = _io.open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8").read()
    assert "render_wcs_studies(WCS_STUDIES, q)" in src, (
        "app.py no longer renders the Wealth Creation Studies on screen"
    )
    assert "studies=WCS_STUDIES" in src, "the download no longer carries the studies corpus"


# ── Two-mode Reference layout (2026-08-29) ───────────────────────────────────────────────────
def test_reference_two_mode_layout():
    """THE TWO-MODE CONTRACT. BROWSE (empty query) = five inner tabs, one corpus each — the tab
    was five stacked corpora (~70KB) and scrolling was the app's worst UX. SEARCH (any query) =
    the tab bar disappears and results come UNIFIED from all five corpora — one box searches
    everything, so a user never has to know which section holds the answer (the export/search
    asymmetry class, frameworks 2026-08-28, must not recur mode-wise). Every corpus must be wired
    into BOTH branches: dropping one from either mode turns this red. The sixth, 🎯 Scans (2026-10-14),
    is the scan guide (ui/scan_guide.py, tests/test_scan_guide.py)."""
    import io as _io, os
    src = _io.open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8").read()
    helpers = ("_show_concepts", "_show_glossary", "_show_flags", "_show_frameworks", "_show_wcs",
               "_show_scans")

    # SEARCH MODE: between the query conditional and the browse branch, all six filtered by _ref_q.
    search_branch = src[src.index("if _ref_q.strip():"):src.index("_ref_tabs = st.tabs")]
    for h in helpers:
        assert f"{h}(_ref_q)" in search_branch, f"search mode lost {h} — unified search broken"
    assert "Searching across all six sections" in search_branch

    # BROWSE MODE: six inner tabs with the exact corpus labels, all six rendered unfiltered.
    browse = src[src.index("_ref_tabs = st.tabs"):]
    for label in ("🏷️ Labels & Verdicts", "📖 Glossary", "🚩 Red Flags",
                  "🏛️ Frameworks", "📚 WCS Studies", "🎯 Scans"):
        assert label in browse, f"browse mode lost its {label} tab"
    for h in helpers:
        assert f'{h}("")' in browse, f"browse mode lost {h} — a corpus vanished from its tab"

    # The inner-tab variable must NOT be _mp_tabs (would confuse the Market Pulse tab extractor).
    assert "_ref_tabs" in src and "_mp_tabs = st.tabs" not in browse


# ── Framework double-documentation: DRIFT-PROOFING (2026-08-30 final-build sign-off) ─────────
# The 37 frameworks are documented in TWO corpora ON PURPOSE — they do different jobs:
#   _FW_META[name][2]        = the terse GATE SPEC (what it screens on) — feeds the tearsheet
#                              pills, the Reference registry and the Markdown download.
#   CONCEPT_REFERENCE[...]   = the plain-language MEANING (what it tells an investor).
# Two texts for two jobs is legitimate; two texts that DRIFT is not. The 2026-08-30 sign-off
# measured median word-overlap of 0.12 and found FIVE pairs naming mutually exclusive criteria
# — Diamond's registry copy described a deep-value screen (engine: Mukherjea three-lens),
# Quality Compounder's cited ROCE≥20/PAT-CAGR (engine: NFAT>4 + FCF-yield), Peaceful Investing
# borrowed another framework's NFAT signal — plus a "17th WCS" attribution for 100x that the
# study audit had already refuted (it is the 19th). Nothing caught any of it. These pins do.
_METRIC_VOCAB = ["roce", "roe", "pat", "fcf", "cfo", "d/e", "debt", "peg", "p/e", "interest cover",
                 "icr", "nfat", "asset turnover", "payout", "dividend", "rsi", "52w", "promoter",
                 "pledge", "earnings yield", "g-sec", "cagr", "volatility", "market cap", "mcap",
                 "working capital", "accrual", "yield"]


def _metrics_named(text):
    t = " " + str(text).lower() + " "
    return {m for m in _METRIC_VOCAB if m in t}


def _norm_fw(name):
    """Framework identity, emoji-insensitive. Concept labels carry the pill emoji by design
    ('🐘 100x Candidate'); the registry keys do not ('100x Candidate'). Comparing raw strings
    pairs NOTHING and every check below passes vacuously — which is exactly what happened on
    the first draft of these pins, and is why this helper exists."""
    import re
    return " ".join(re.sub(r"[^A-Za-z0-9&/→ -]", " ", str(name)).lower().split())


def _framework_pairs():
    """(name, concept_text, registry_text) for every framework documented in both corpora."""
    from ui.ui_reference_data import CONCEPT_REFERENCE
    from ui.ui_tearsheet import _FW_META
    concept = {_norm_fw(l): e for c in CONCEPT_REFERENCE if "Framework" in c
               for l, e in CONCEPT_REFERENCE[c]}
    pairs = [(n, concept[_norm_fw(n)], meta[2]) for n, meta in _FW_META.items()
             if _norm_fw(n) in concept]
    assert len(pairs) >= 30, (
        f"only {len(pairs)} framework pairs matched — the pairing key broke and every drift "
        f"check below would pass vacuously"
    )
    return pairs


def test_framework_names_are_identical_in_both_corpora():
    """Name parity (emoji aside — the concept label carries the pill glyph by design) is what
    makes the two copies checkable at all, and what lets a reader who saw a pill find the same
    framework in the reference. A name in one corpus only is a silent orphan."""
    from ui.ui_reference_data import CONCEPT_REFERENCE
    from ui.ui_tearsheet import _FW_META
    concept = {_norm_fw(l) for c in CONCEPT_REFERENCE if "Framework" in c
               for l, _ in CONCEPT_REFERENCE[c]}
    registry = {_norm_fw(n) for n in _FW_META}
    assert concept == registry, (
        f"framework names differ between corpora — only in concepts: {sorted(concept - registry)}; "
        f"only in registry: {sorted(registry - concept)}"
    )


def test_registry_gate_specs_match_the_engine_gates():
    """THE TRUTH PIN. _FW_META's description is the terse GATE SPEC shown on the tearsheet pill,
    the Reference registry and the Markdown download — three surfaces asserting to a user what
    the engine screens on. Each entry below was verified line-by-line against its gate in
    core/scoring_engine.py during the 2026-08-30 final-build sign-off, which found FOUR of them
    describing a screen the engine does not run (Diamond as a deep-value earnings-yield test,
    Quality Compounder citing ROCE≥20/PAT-CAGR instead of its NFAT and FCF-yield gates, Long
    Game citing PAT-CAGR/volatility instead of ICR≥5 and FCF/PAT≥60, Peaceful Investing
    borrowing Quality Compounder's NFAT signal). MUST/MUST-NOT below are the signals the gate
    actually tests — change them only after re-reading the gate."""
    from ui.ui_tearsheet import _FW_META
    # framework -> (tokens the gate DOES test — at least one must appear,
    #               tokens the gate does NOT test — none may appear)
    VERIFIED = {
        "Diamond":            (["d/e", "cfo/pat", "fcf/cfo", "forensic"], ["earnings yield", "g-sec"]),
        "Long Game Quality":  (["interest cover", "icr", "fcf/pat"],      ["volatility"]),
        "Quality Compounder": (["nfat", "fcf yield"],                     ["pat cagr"]),
        "Peaceful Investing": (["pillar", "profit stability", "debt fortress", "self-funded"], ["nfat"]),
    }
    bad = []
    for fw, (must_any, must_not) in sorted(VERIFIED.items()):
        d = _FW_META[fw][2].lower()
        if not any(t in d for t in must_any):
            bad.append(f"{fw}: names none of the signals its gate tests {must_any}")
        for t in must_not:
            if t in d:
                bad.append(f"{fw}: claims {t!r}, which its engine gate does not test")
    assert not bad, "registry gate specs contradicting the engine: " + " | ".join(bad)


def test_every_framework_is_documented_in_all_three_corpora():
    """Structural completeness across the three framework texts, each with a different job:
    _FW_META = terse gate spec (pills/registry/download) · _FW_IDEA = the plain-language idea
    ('?' tooltip) · CONCEPT_REFERENCE = the Reference-tab meaning. Two texts for two jobs is
    legitimate; a framework MISSING from one of them is a hole a reader falls through."""
    from ui.ui_reference_data import CONCEPT_REFERENCE
    from ui.ui_tearsheet import _FW_META, _FW_IDEA
    concept = {_norm_fw(l) for c in CONCEPT_REFERENCE if "Framework" in c
               for l, _ in CONCEPT_REFERENCE[c]}
    idea = {_norm_fw(n) for n in _FW_IDEA}
    registry = {_norm_fw(n) for n in _FW_META}
    assert registry == idea == concept, (
        f"missing from concepts: {sorted(registry - concept)}; "
        f"missing from ideas: {sorted(registry - idea)}; "
        f"orphans not in the registry: {sorted((concept | idea) - registry)}"
    )


def test_no_refuted_study_attribution_survives_in_the_registry():
    """Provenance corrected in the engine must be corrected in the UI. The 100x screen belongs
    to the 19th Wealth Creation Study; the 17th is Economic Moat, and 'Mouse to Elephant'
    appears zero times in the source study (established by the 19th-study audit). A stale
    citation on a pill is a claim the product makes to a user."""
    from ui.ui_tearsheet import _FW_META
    desc = _FW_META["100x Candidate"][2]
    assert "17th" not in desc, "the refuted 17th-WCS attribution is back on the 100x pill"
    assert "19th" in desc, "the 100x description no longer records its real (19th WCS) lineage"


def test_no_entry_promises_an_outcome():
    """A reference explains; it never promises. This system's credibility rests on not
    overclaiming, and after final sign-off nothing else polices the tone."""
    import re
    from ui.ui_reference_data import CONCEPT_REFERENCE
    from ui.ui_tearsheet import _FW_META
    promise = re.compile("(will (?:outperform|rise|deliver|beat|double)|"
                         "guaranteed returns?|never fails|is a buy|should buy|"
                         "sure[- ]?shot returns?)", re.I)
    bad = [f"{l}: {e[:70]}" for cat in CONCEPT_REFERENCE.values() for l, e in cat if promise.search(e)]
    bad += [f"{n} (registry): {m[2][:70]}" for n, m in _FW_META.items() if promise.search(m[2])]
    assert not bad, "promise-language in the reference: " + " | ".join(bad)
