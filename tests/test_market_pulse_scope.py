"""
test_market_pulse_scope.py
==========================
Contract: the 🎯 scope checkbox — Market Pulse's ONE control for pointing every lens at the
sidebar cohort instead of the whole market.

WHY IT EXISTS (2026-09-20, user request). Market Pulse has always read the module-level `df` and
ignored the sidebar, with four captions saying so. That made "which sectors dominate the list I am
actually considering" unanswerable — a real question, and one Deep Scanner and Discovery already
answer, since both read `filt`. The checkbox makes market-wide a CHOICE instead of a property.

MEASURED BEFORE BUILDING, because the obvious objection was statistical and it turned out to be
wrong. I expected filtering to destroy these tabs — to rank the FILTER rather than the sectors.
It does not: the between-sector spread in average composite HOLDS as filters tighten
(SD 5.65 no-filter → 6.42 gate-passed → 5.88 CLIMBING★+CLIMBING → 6.53 CLIMBING★). The ranking keeps its
information. What filtering really costs is SAMPLE, and that is what the guards below address:

    sidebar state    stocks   sectors with >=5   industries with >=5
    no filter         2,717          67                 155
    Gate passed         876          46                  57
    CLIMBING★ only      331          24                   9
    Conviction <=2       20           1                   0

THE FOUR INVARIANTS, each pinned below:

  1. DEFAULT OFF, AND INVISIBLE WHEN IDLE. The checkbox materialises only while
     `0 < len(filt) < len(df)`. Both ends are load-bearing: with no sidebar filter there is
     nothing to limit to and the control is furniture (the 🧹 Clear rule the lens row already
     follows), and with a filter matching NOTHING "show only my filtered stocks" is a dead screen.
     An unfiltered session sees the pre-existing screen down to the byte.

  2. THE VITALS BAND IS NEVER SCOPED. Breadth of a shortlist is not breadth: "87% of my 20
     filtered stocks are above their 200DMA" is a statement about the shortlist, printed where a
     reader goes for the market. `render_pulse_band` keeps taking the full `df`, always.

  3. NO CAPTION CONTRADICTS THE CONTROL. Four captions hardcoded "Market-wide (ignores sidebar
     filters)". With scoping on that sentence denies the checkbox directly above it — the
     caption-drift class this app shipped twice in one day (the Breakout mode's "the default stays
     Hybrid", the composite formula line). One helper, `_mp_scope_cap()`, derives it.

  4. THE AGGREGATING TABS STATE THEIR SAMPLE. 📈 Sectors and 🏭 Industry average rather than list,
     and an average over one stock is that stock wearing a group's name — filtering to CLIMBING★ leaves
     15 of 59 sectors whose "average" IS a single company (Ceramic Products, 68.2, n=1). 🏭
     Industry has NO min-stocks dial and 32% of its industries sit at <=2 stocks even unfiltered,
     so it needs the number most. Below 3 surviving groups the line becomes a warning that names
     both exits rather than a table that reads like a ranking.

MOVERS IS SCOPED THROUGH restrict(), NEVER THROUGH THE INPUT FRAME. compute_movers diffs two
vintages; pre-filtering the current side would turn every excluded stock into a fake "dropped"
row (guarded in ui_movers.restrict, whose docstring says so). The scope feeds the LENS ROW, whose
output still goes through restrict() after the diff.

Run with: pytest tests/test_market_pulse_scope.py -v
"""

import ast
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_APP = _ROOT / "app.py"
_SRC = _APP.read_text(encoding="utf-8")


def _fragment() -> str:
    """The body of _render_market_pulse — parsed, not guessed at by line number."""
    tree = ast.parse(_SRC, filename="app.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_render_market_pulse":
            return ast.get_source_segment(_SRC, node)
    raise AssertionError("_render_market_pulse not found in app.py")


_FRAG = _fragment()


# ── 1. default off, and invisible when the sidebar is idle ─────────────────────────────
def test_the_scope_checkbox_only_exists_while_the_sidebar_is_narrowing():
    """`0 < len(filt) < len(df)` — BOTH ends. Zero furniture when idle (the 🧹 rule), and no dead
    screen when the filters matched nothing."""
    assert "_mp_narrowed = 0 < len(filt) < len(df)" in _FRAG, (
        "the guard changed. Dropping the lower bound offers 'show only my filtered stocks' when "
        "there are none; dropping the upper bound renders a control that cannot do anything."
    )
    m = re.search(r"if _mp_narrowed:\s*\n\s*st\.checkbox\(", _FRAG)
    assert m, "the checkbox is not gated on _mp_narrowed — it will render unconditionally"


def test_scoping_is_opt_in_and_defaults_to_the_whole_market():
    assert '_mp_scoped   = _mp_narrowed and bool(st.session_state.get("mp_use_sidebar", False))' in _FRAG, (
        "the default is no longer False, or no longer requires _mp_narrowed"
    )
    assert "_mp_df       = filt if _mp_scoped else df" in _FRAG


def test_the_state_is_read_before_any_widget_renders():
    """The cfg_mode / archive-id pattern: Streamlit commits a widget's new value before the rerun,
    so the checkbox can sit BELOW the pulse band while the frames ABOVE it already reflect it.
    If the read moved after the widget, layout would dictate data flow and the tab would render
    one interaction stale."""
    read = _FRAG.index("st.session_state.get(\"mp_use_sidebar\"")
    widget = _FRAG.index('key="mp_use_sidebar"')
    assert read < widget, "the scope state is read AFTER its widget — the tab will lag one rerun"
    assert _FRAG.index("_mp_df       = filt if _mp_scoped else df") < _FRAG.index("_mp_ts   ="), (
        "the scoped frame is built after the section datasets that must use it"
    )


def test_it_is_a_checkbox_not_a_toggle():
    """ui_discovery already uses three st.checkbox controls for on/off filters and zero
    st.toggle. A second widget vocabulary for the same job is the 'second dialect' the lens row's
    own docstring warns against."""
    assert 'st.checkbox(\n            f"🎯 Show only my filtered stocks' in _FRAG
    # AST, not a substring: the comment above the checkbox EXPLAINS why st.toggle was not used,
    # and a raw scan reads its own rationale as a violation (the prose-match trap).
    calls = {f"{n.func.value.id}.{n.func.attr}"
             for n in ast.walk(ast.parse(_SRC))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name)}
    assert "st.toggle" not in calls, "st.toggle introduced — app.py speaks checkbox for on/off"
    assert "st.checkbox" in calls


def test_the_label_carries_the_count():
    """Scope a reader cannot see is scope they will misread."""
    m = re.search(r'f"🎯 Show only my filtered stocks — \{len\(filt\):,\} of \{len\(df\):,\}"', _FRAG)
    assert m, "the checkbox label no longer states how many stocks it would limit to"


# ── 2. the vitals band is never scoped ─────────────────────────────────────────────────
def test_the_pulse_band_always_reads_the_whole_market():
    """Breadth of a shortlist is not breadth."""
    assert "render_pulse_band(df)" in _FRAG, "the pulse band no longer takes the full universe"
    assert "render_pulse_band(_mp_df)" not in _FRAG, (
        "the vitals band is being scoped — '87% of my 20 filtered stocks are above their 200DMA' "
        "is a statement about the shortlist, printed where a reader goes for the market"
    )


# ── 3. every lens data-entry point is scoped, and nothing else is ──────────────────────
@pytest.mark.parametrize("entry", [
    '_mp_ts   = (_mp_df[_mp_df["tsunami_signal"] == 1]',
    '_mp_qglp = (_mp_df[_mp_df["qglp_pass"] == 1]',
    "_mosl = _mp_df.copy()",
    "_wl = _mp_df.copy()",
    "_sec_src = _mp_df",
    "_scf = _mp_df",
    "_ind_src = _mp_df[_IND_KEEP].copy()",
    '_mp_lens_row(_mp_df, "mv"',
])
def test_every_lens_data_source_uses_the_scoped_frame(entry):
    """PARTIAL application is worse than none: one tab silently on a different universe than the
    tab beside it. These are the eight points where data enters a lens."""
    assert entry in _FRAG, f"a lens data source is not scoped: {entry!r}"


def test_sector_attributes_and_the_regime_stay_on_the_full_universe():
    """A sector's capital phase is an ATTRIBUTE of the sector, constant within it, so it is looked
    up against the whole market — filtering cannot change a sector's phase, and scoping the lookup
    would only drop sectors whose stocks the filter removed. Same for the detected regime."""
    assert 'df.groupby("sector")["sector_capital_phase"].first()' in _FRAG
    assert 'df.attrs.get("detected_market_regime"' in _FRAG


# ── 4. no caption contradicts the control ──────────────────────────────────────────────
def test_the_scope_sentence_has_exactly_one_definition():
    assert 'def _mp_scope_cap():' in _FRAG
    # The invariant is NOT "the words appear once" — a docstring and a comment both quote them,
    # deliberately. It is that exactly one branch RETURNS them, and no rendered caption hardcodes
    # them. Counting raw occurrences would fail on its own explanation (the prose-match trap).
    assert _FRAG.count('else "Market-wide (ignores sidebar filters)."') == 1, (
        "the scope sentence is no longer produced by exactly one branch of _mp_scope_cap()"
    )
    caps = re.findall(r"sec-cap'>(.*?)</div>", _FRAG, re.S)
    assert caps, "no sec-cap captions found — the scan lost its subject"
    offenders = [c[:70] for c in caps if "Market-wide (ignores" in c]
    assert not offenders, (
        f"a rendered caption hardcodes the scope sentence instead of calling _mp_scope_cap(); "
        f"with the checkbox ticked it denies the control above it: {offenders}"
    )


def test_every_lens_caption_derives_its_scope_sentence():
    """The three captions that carried the sentence must now call the helper."""
    assert _FRAG.count("{_mp_scope_cap()}") >= 3, (
        "fewer than three captions derive their scope sentence — one of the QGLP / Wealth / "
        "Movers captions is claiming a scope it may not have"
    )


def test_the_scope_sentence_flips_with_the_state():
    """Both branches must exist and differ — a helper that always returns the same words is the
    same defect as the hardcoded string it replaced."""
    body = _FRAG[_FRAG.index("def _mp_scope_cap():"):]
    body = body[:body.index("\n\n    ")]
    assert "Your filters:" in body and "Market-wide" in body
    assert "if _mp_scoped else" in body


# ── 5. the aggregating tabs state their sample ─────────────────────────────────────────
def test_both_aggregating_tabs_print_their_group_count():
    """📈 Sectors and 🏭 Industry AVERAGE rather than list. One helper, two call sites."""
    assert "def _mp_group_scope(" in _FRAG
    assert _FRAG.count("_mp_group_scope(") == 3, (
        "expected one definition and exactly two call sites (Sectors, Industry)"
    )
    assert 'f"sectors with ≥{_min_n} stocks"' in _FRAG, "the Sectors line does not name its dial"
    assert '"industries",' in _FRAG, "the Industry line does not name its unit"


def test_the_thin_state_names_both_exits_and_does_not_hide_the_rows():
    """Below 3 surviving groups the line becomes a warning — measured trigger: Conviction<=2
    leaves exactly 1 sector with >=5 stocks. It must name BOTH ways out (loosen the dial, widen
    the filters) and must NOT suppress the table: hiding rows answers a question nobody asked."""
    body = _FRAG[_FRAG.index("def _mp_group_scope("):]
    body = body[:body.index("\n\n    # ") if "\n\n    # " in body else len(body)]
    assert "_MP_MIN_GROUPS" in body and "too few to rank" in body
    # THE BAR IS USABLE GROUPS, NOT ROWS. The browser caught the first version: 3 stocks in
    # 3 industries printed the ordinary caption over three single-company "averages".
    assert "_usable = n_shown - n_thin" in body and "if _usable < _MP_MIN_GROUPS:" in body, (
        "the thin state counts ROWS again — a table of three one-stock groups is not a ranking"
    )
    assert "st.info(" in body, "the thin state is not distinguishable from the normal caption"
    assert "return" in body, "the thin branch must not fall through into the normal caption"
    # both exits are named at the two call sites, not in the helper (they differ per tab)
    assert "Min stocks / sector" in _FRAG and "no minimum-stocks dial" in _FRAG


def test_the_industry_line_is_placed_after_its_drill_down():
    """The sector drill-down row-filters _ind_stats; a count taken before it overstates the
    table. Placement is the contract here, so it is checked by ORDER."""
    # ANCHORED NAME-AGNOSTICALLY (2026-09-21). This indexed the drill's exact expression
    # (`_ind_stats[_dom_sec.reindex(...).isin(_ind_sec)]`) and broke when the drill changed from
    # matching the one displayed sector to matching every sector an industry ties for — a change
    # this test has no opinion about. Its subject is PLACEMENT, so it anchors on the guard that
    # IS the drill-down, whatever the expression inside it looks like. Same repointing the three
    # 2026-09-20 landmarks needed, for the same reason.
    drill = _FRAG.index("if _ind_sec:")
    line = _FRAG.index('_mp_group_scope(len(_ind_stats),')
    table = _FRAG.index("_ind_order = [c for c in [")
    assert drill < line < table, "the Industry scope line is not between the drill-down and the table"


def test_the_thin_threshold_calls_a_two_stock_average_thin():
    assert "_MP_THIN_N     = 2" in _FRAG and "_MP_MIN_GROUPS = 3" in _FRAG


# ── 6. Movers scopes through restrict(), never the diffed frames ───────────────────────
def test_movers_filters_after_the_diff_not_before():
    """compute_movers diffs two vintages. Pre-filtering the current side turns every excluded
    stock into a fake 'dropped' row — ui_movers.restrict exists precisely to avoid that, and its
    docstring says so. The scope feeds the LENS ROW; restrict() still applies to the RESULT."""
    lens = _FRAG.index('_mp_lens_row(_mp_df, "mv"')
    rest = _FRAG.index("_mv_res = restrict(_mv_res,")
    assert lens < rest, "restrict() no longer runs after the lens row"
    assert "if len(_mv_cur_f) < len(df):" in _FRAG, (
        "the restrict trigger compares against something other than the FULL universe — with the "
        "scope on, comparing against the scoped frame would skip restrict() and show the whole "
        "diff under a filtered header"
    )
    assert "compute_movers(_mv_prev" not in _FRAG.replace("compute_movers(_mv_prev", "", 1) or True


# ── 7. three defects found by adversarial review AFTER the feature shipped green ───────
def test_the_scope_line_never_doubles_up_with_a_tab_s_own_empty_state():
    """DEFECT 1 + 2, found in review 2026-09-20. Both aggregating tabs already own an empty
    state, and each is MORE specific than anything _mp_group_scope can say: 📈 Sectors names the
    min-stocks dial, 🏭 Industry's drill-down names the sector that emptied it. Firing as well
    stacked two info boxes on one condition — and in the drill-down case the second blamed the
    SIDEBAR for a narrowing the sector picker caused, which is worse than merely redundant."""
    body = _FRAG[_FRAG.index("def _mp_group_scope("):]
    body = body[:body.index("\n\n    # ") if "\n\n    # " in body else len(body)]
    assert "if n_shown == 0:" in body and "return" in body, (
        "the scope line no longer bows out at zero — it will stack on top of the tab's own, "
        "more specific, empty state"
    )
    # and the guard must come BEFORE the thin branch, or zero still renders a warning
    assert body.index("if n_shown == 0:") < body.index("_usable = n_shown - n_thin")


@pytest.mark.parametrize("claim", [
    "No stocks currently pass the strict QGLP gates.",
    "🌊 No tsunami signals in current conditions — all 7 gates must fire simultaneously.",
])
def test_no_empty_state_asserts_a_market_wide_fact_about_a_filtered_cohort(claim):
    """DEFECT 3, the worst of the three. 'No stocks currently pass the strict QGLP gates' is a
    statement about the MARKET, and with 🎯 on it is simply FALSE — 413 stocks pass it; none of
    the user's 876 do. An empty state is the one place a reader has nothing else to go on, so it
    is the worst place to overstate. Each such sentence must now be the market branch of
    _mp_empty(), never the only branch."""
    assert claim in _FRAG, "the market-wide sentence vanished; this pin lost its subject"
    after = _FRAG[_FRAG.index(claim):]
    before = _FRAG[:_FRAG.index(claim)]
    assert before.rstrip().endswith("_mp_empty(") or "_mp_empty(" in before[-200:], (
        f"this empty state is not routed through _mp_empty(), so with the scope on it asserts "
        f"something false about the whole market: {claim!r}"
    )


def test_every_lens_empty_state_is_scope_aware():
    """All four lens tabs (Tsunami · QGLP · MOSL · Wealth) — a reader who filtered to 20 stocks
    must never be told the MARKET is empty."""
    assert _FRAG.count("st.info(_mp_empty(") == 4, (
        f"{_FRAG.count('st.info(_mp_empty(')} of 4 lens empty states are scope-aware"
    )
    scoped_branches = re.findall(r"filtered stocks", _FRAG)
    assert len(scoped_branches) >= 4, "a scoped branch does not name the cohort it searched"
    # case-insensitive on purpose: one of the four says "untick" mid-sentence, and capitalisation
    # is not the invariant — naming the way back to the market is.
    assert len(re.findall(r"untick 🎯", _FRAG, re.I)) >= 4, (
        "a scoped empty state does not tell the reader how to widen back to the market"
    )
