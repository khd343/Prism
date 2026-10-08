"""
test_deep_scanner_views.py
==========================
Contract for the Deep Scanner's column views, rebuilt 2026-10-05 after measuring them on the live
app at a 1536px laptop width (the user's screen):

  - in 5 of 6 views the rows were ordered by Score while no Score column was on screen, so a view's
    own columns read in a random-looking order (Quality: 86, 80, 88, 85, 92 ...) — the same
    "rank-jumble" that removed Discovery's sort pills on 2026-08-24;
  - 5 of 6 views overflowed the screen (Quality by 574px, Technical 366, Core 335, Valuation and
    Forensic 115) and the stock name scrolled away with them;
  - only Core showed Rank or Score, so switching views lost where a stock stands.

WHAT IS PINNED
  1. Every view leads with Rank · Stock (unpinned since 2026-10-06: Streamlit draws pinned columns
     faded, and every view fits the screen, so they stay in view anyway). Rank IS the Score order (the engine re-derives it from the
     post-penalty composite — pinned by test_rank_tracks_post_penalty_composite), so the default
     sort reads in every view without a Score bar in each.
  2. Whatever the table is sorted by is on screen: a view that lacks the sorted column gets it right
     after Stock (ui_scanner.ds_view_columns), and a view that has it is left EXACTLY as curated —
     so the pinned pairs (a verdict beside its number, the ROCE trio, P/E beside the yields) hold.
  3. Measured repeats stay out of their views: Forensic Score (an exact copy of the red-flag count,
     rank correlation -1.00), Piotroski and CFO/PAT in Quality (both live in Forensic), Smart Money
     in Core (it lives in Technical), the 52-week-high distance in Technical (-0.95 with Breakout).
  4. Every score bar has an explicit width, and the three view switchers (Deep Scanner, Market Pulse
     Sectors and Industry) are segmented controls — the widget for 'pick one view'.

  5. EVERY VIEW FITS a 1536px laptop screen with the sidebar open, pinned by ARITHMETIC: every column
     has a set width (app.py _DS_W) and, measured in the browser on all six views, the grid's content
     width was exactly those widths plus the 44px row selector, inside a grid re-measured at 1,064px on
     2026-10-08 (the first measurement read 1,074 and allowed 10px too many). Getting there took cuts the user
     approved after seeing the measurements: E.Yield (P/E inverted, -1.00) and MCap (size, not value;
     still shown when sorted by and still exported) left Valuation; VSTOP (identical to Stop Δ > 0 on
     100% of stocks), Tsunami (3 of 2,717 stocks, and it has its own tab), Price (in Valuation) and
     Smart Money (in Ownership) left Technical; Quality's parts read as numbers beside its bar, and
     Governance — a separate composite leg built from ownership signals — moved to Ownership.

  6. ONE HOME PER COLUMN (2026-10-08). No column shows in two of the five detail views (Promoter and
     Pledged sat in Forensic AND Ownership). Core is the overview and repeats only Improvement and
     Momentum — the pair the Sectors and Industry Core views show, in their order. Conviction left
     Core (the Score cut into five bands, an exact function of it) and so did Moat·Growth (the user's
     choice: its 'Wealth Creator' words read as a second wealth verdict beside the Wealth column).
     Forensic gained Schilit pass — a tick, with the glossary's tip, whose words are checked against
     the engine — and Evidence from Core.

Run with: pytest tests/test_deep_scanner_views.py -v
"""

import ast
import io as _io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from ui.ui_scanner import ds_view_columns

SRC = _io.open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8").read()
TREE = ast.parse(SRC)


def _dict(name):
    node = next(n.value for n in ast.walk(TREE) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return dict(zip([k.value for k in node.keys], node.values))


VIEWS = {k: [e.value for e in v.elts] for k, v in _dict("_DS_VIEWS").items()}
SORTS = {k: v.elts[0].value for k, v in _dict("_DS_SORTS").items()}
WITNESS = {"composite_score": "rank", "result_age_days": "result_when"}


def _tab_position(tree, label):
    """The tab's position in app.py's own st.tabs([...]) list, found by its LABEL — so a tab inserted
    before it (🎯 Scans became tab 2 on 2026-10-06) cannot leave this test reading the wrong body."""
    call = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "tabs")
    return [e.value for e in call.args[0].elts].index(label)


def _ds_block():
    """The Deep Scanner's `with tabs[i]:` node (i found by its label) — label checks are scoped to it."""
    for n in ast.walk(TREE):
        if isinstance(n, ast.With):
            for item in n.items:
                c = item.context_expr
                if (isinstance(c, ast.Subscript) and isinstance(c.value, ast.Name) and c.value.id == "tabs"
                        and isinstance(c.slice, ast.Constant)
                        and c.slice.value == _tab_position(TREE, "🔍 Deep Scanner")):
                    return n
    raise AssertionError("Deep Scanner block not found")


# ── 1. The spine ────────────────────────────────────────────────────────────────────────────────
def test_every_view_leads_with_rank_and_stock():
    assert len(VIEWS) >= 5
    for view, cols in VIEWS.items():
        assert cols[:2] == ["rank", "name"], f"{view} leads with {cols[:2]}, not Rank · Stock"


def test_rank_and_stock_are_configured_once_and_never_faded():
    """Repointed 2026-10-06: the spine was PINNED, and Streamlit draws pinned columns faded, so Rank and
    Stock read grey (tests/test_no_faded_columns.py). Every view fits the screen, so they stay in view
    unpinned; what this pins now is that they are configured ONCE and never pinned again."""
    spine = {k: v.value for k, v in _dict("_DS_SPINE").items()}
    assert spine == {"rank": "Rank", "name": "Stock"}
    block = _ds_block()
    loop = [n for n in ast.walk(block) if isinstance(n, ast.For) and ast.unparse(n.iter) == "_DS_SPINE.items()"]
    assert len(loop) == 1, "the spine's columns must be configured from _DS_SPINE"
    calls = [c for c in ast.walk(loop[0]) if isinstance(c, ast.Call)
             and ast.unparse(c.func).startswith("st.column_config.")]
    assert len(calls) == 2 and not any(k.arg == "pinned" for c in calls for k in c.keywords), (
        "Rank and Stock must not be pinned: Streamlit draws pinned columns faded, the name turns grey")
    # No other label map may configure them again: a later loop would overwrite this config.
    others = [ast.unparse(d)[:60] for d in ast.walk(block) if isinstance(d, ast.Dict) and d.keys
              and any(isinstance(k, ast.Constant) and k.value in ("rank", "name") for k in d.keys)
              and not all(isinstance(v, ast.List) for v in d.values)]
    assert len(others) == 1, f"rank / name configured in more than one label map: {others}"


# ── 2. The sorted column is on screen ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("sort", sorted(SORTS))
def test_the_sorted_column_is_on_screen_in_every_view(sort):
    col = SORTS[sort]
    witness = WITNESS.get(col, col)
    for view, curated in VIEWS.items():
        avail = set(curated) | {col, witness}
        cols = ds_view_columns(curated, col, avail)
        assert witness in cols, f"{view} sorted by {sort}: {witness} is not on screen"
        assert len(cols) == len(set(cols)), f"{view}: a column shows twice"
        if witness in curated:
            assert cols == curated, f"{view}: a view that carries {witness} must stay exactly as curated"
        else:
            assert cols[:3] == ["rank", "name", witness], f"{view}: {witness} must sit right after Stock"
            assert [c for c in cols if c != witness] == curated, f"{view}: nothing else may move"


def test_the_score_sort_reads_from_rank_without_adding_a_column():
    curated = ["rank", "name", "quality_score"]
    assert ds_view_columns(curated, "composite_score", set(curated) | {"composite_score"}) == curated


def test_without_rank_the_score_itself_is_shown():
    curated = ["name", "quality_score"]
    got = ds_view_columns(curated, "composite_score", {"name", "quality_score", "composite_score"})
    assert got == ["name", "composite_score", "quality_score"]


def test_only_available_columns_are_shown():
    got = ds_view_columns(["rank", "name", "gone", "peg"], "pe", {"rank", "name", "peg"})
    assert got == ["rank", "name", "peg"], "a missing column is dropped, and an absent sort column is not invented"


def test_the_app_builds_its_columns_through_the_shared_rule():
    hits = [n for n in ast.walk(_ds_block()) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_view_cols" for t in n.targets)]
    assert len(hits) == 1 and ast.unparse(hits[0].value).startswith("ds_view_columns("), (
        "the Deep Scanner must pick its columns through ui_scanner.ds_view_columns")


# ── 3. Measured repeats and the approved cuts stay out ──────────────────────────────────────────
def test_repeats_and_approved_cuts_stay_out_of_their_views():
    out = {"🔬 Forensic": ["forensic_score"],
           "📊 Quality": ["piotroski_fscore", "cfo_to_pat", "governance_bonus"],
           # Conviction is the Score cut into bands; Moat·Growth left on the user's choice (2026-10-08)
           "🏆 Core": ["smart_money_flow", "conviction_tier", "moat_growth_quad"],
           "💰 Valuation": ["earnings_yield", "market_cap"],
           "📈 Technical": ["dist_52wh", "vstop_green", "tsunami_signal", "close_price", "smart_money_flow"]}
    for view, cols in out.items():
        for c in cols:
            assert c not in VIEWS[view], f"{c} is back in {view} — see the docstring for why it left"
    # …and what moved still lives where it belongs
    assert {"piotroski_fscore", "cfo_to_pat"} <= set(VIEWS["🔬 Forensic"])
    assert {"governance_bonus", "smart_money_flow"} <= set(VIEWS["👥 Ownership"])
    assert {"promoter_holdings", "pledged_percentage"} <= set(VIEWS["👥 Ownership"])
    assert {"schilit_pass", "data_coverage_pct"} <= set(VIEWS["🔬 Forensic"])
    assert "close_price" in VIEWS["💰 Valuation"]
    assert "market_cap" in SORTS.values(), "MCap left the Valuation view; its sort must stay so it can still be shown"


def _bar_map():
    loop = next(n for n in ast.walk(_ds_block()) if isinstance(n, ast.For)
                and ast.unparse(n.target) == "(_sc, _sl)")
    return {k.value for k in loop.iter.func.value.keys}


def test_quality_reads_as_its_bar_with_the_parts_as_numbers():
    bars, fmt = _bar_map(), _dict("_num_fmt")
    assert "quality_score" in bars and "composite_score" in bars
    for c in ("moat_score", "growth_score", "improvement_score", "cash_score", "governance_bonus"):
        assert c not in bars and c in fmt, f"{c} must read as a number — five more bars cost a laptop screen"


def test_the_export_carries_every_sort_column():
    exp = [n for n in ast.walk(_ds_block()) if isinstance(n, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == "_export_cols" for t in n.targets)]
    assert len(exp) == 1 and "_DS_SORTS.values()" in ast.unparse(exp[0].value), (
        "MCap left the views — without the sort columns the CSV export would lose it")


# ── 5. Every view fits a laptop screen ──────────────────────────────────────────────────────────
# Re-measured 2026-10-08 at a 1536px window, sidebar open at its default 300px (stored widths cleared
# first), from the grid's OWN column state (glide-data-grid props): the grid is 1,064px wide and the
# row-selection column is 44px. Streamlit gives every column grow:1, so a narrower view is stretched
# to fill the grid, and a view scrolls sideways only when its set widths + 44 pass 1,064. Technical
# needed 1,074 and Ownership 1,069 — 10px and 5px of sideways scroll that the 2026-10-05 budget
# (a 1,074px grid, 10px wider than today's) allowed. (The grid's scroll-area readings vary between
# renders; its column state does not, so measure that.)
_GRID_PX, _ROW_SELECTOR_PX = 1064, 44
W = {k: v.value for k, v in _dict("_DS_W").items()}


def test_every_column_a_view_can_show_has_a_set_width():
    can_show = {c for cols in VIEWS.values() for c in cols} | {WITNESS.get(c, c) for c in SORTS.values()}
    missing = sorted(c for c in can_show if not isinstance(W.get(c), int) or W[c] <= 0)
    assert not missing, f"columns with no set width (Streamlit would auto-size them): {missing}"


def test_every_view_fits_a_laptop_screen():
    for view, cols in VIEWS.items():
        need = sum(W[c] for c in cols) + _ROW_SELECTOR_PX
        assert need <= _GRID_PX, f"{view} needs {need}px: {need - _GRID_PX}px off a 1536px laptop screen"


# ── 4. Widths and the switcher ──────────────────────────────────────────────────────────────────
def test_every_score_bar_has_an_explicit_width():
    bars = [c for c in ast.walk(_ds_block()) if isinstance(c, ast.Call)
            and ast.unparse(c.func) == "st.column_config.ProgressColumn"]
    assert bars, "no score bars found in the Deep Scanner"
    assert all(any(k.arg == "width" for k in c.keywords) for c in bars), (
        "a score bar without a width takes Streamlit's ~150px and pushes the view off a laptop screen")


@pytest.mark.parametrize("key", ["ds_view", "mp_sec_view", "mp_ind_view"])
def test_the_view_switchers_are_segmented_controls(key):
    calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call)
             and any(k.arg == "key" and isinstance(k.value, ast.Constant) and k.value.value == key
                     for k in n.keywords)]
    assert len(calls) == 1, f"expected one widget keyed {key!r}, found {len(calls)}"
    assert ast.unparse(calls[0].func) == "st.segmented_control", (
        f"{key}: a view switcher is a segmented control — pills read as tags or filters")


# ── 6. One home per column (2026-10-08) ─────────────────────────────────────────────────────────
CORE = "🏆 Core"
DETAIL = [v for v in VIEWS if v != CORE]
SPINE = {"rank", "name"}


def test_no_column_has_two_homes_among_the_detail_views():
    """Promoter and Pledged sat in 🔬 Forensic AND 👥 Ownership. One home each: a reader who looks
    for a column finds it in one place, and a view's width goes to columns it alone shows."""
    homes = {}
    for v in DETAIL:
        for c in VIEWS[v]:
            if c not in SPINE:
                homes.setdefault(c, []).append(v)
    two = {c: vs for c, vs in sorted(homes.items()) if len(vs) > 1}
    assert not two, f"columns shown in two detail views: {two}"


def _group_core(name):
    """The column list of a Market Pulse table's 🏆 Core view, read from app.py (`_sec_order`, `_ind_order`)."""
    node = next(n.value for n in ast.walk(TREE) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return [c.value for c in ast.walk(node) if isinstance(c, ast.Constant) and isinstance(c.value, str)]


@pytest.mark.parametrize("group_core", ["_sec_order", "_ind_order"])
def test_core_repeats_only_the_pair_the_group_core_views_show(group_core):
    """Core is the overview: of the detail views' columns it repeats Improvement and Momentum and
    nothing else — side by side, in the order the Sectors and Industry Core views use for them."""
    detail = {c for v in DETAIL for c in VIEWS[v]} - SPINE
    shared = [c for c in VIEWS[CORE] if c in detail]
    twin = {"avg_improvement": "improvement_score", "avg_momentum": "momentum_score"}
    group_pair = [twin[c] for c in _group_core(group_core) if c in twin]
    assert len(group_pair) == 2, f"{group_core} no longer shows the Improvement / Momentum pair"
    assert shared == group_pair, f"Core repeats {shared}; the {group_core} pair reads {group_pair}"
    i = VIEWS[CORE].index(shared[0])
    assert VIEWS[CORE][i + 1] == shared[1], "Improvement and Momentum must sit side by side"


def test_schilit_pass_reads_as_a_tick_with_the_glossarys_own_tip():
    from ui.ui_components import _RAW_GLOSSARY
    from ui.ui_scanner import _SCANNER_HEADER_TIPS
    loop = next(n for n in ast.walk(_ds_block()) if isinstance(n, ast.For)
                and ast.unparse(n.target) == "(_bc, _lbl)")
    boxes = {k.value for k in loop.iter.func.value.keys}
    assert {"gate_pass", "schilit_pass"} <= boxes, "a pass / fail reads as a tick, like the Gate"
    assert _SCANNER_HEADER_TIPS.get("schilit_pass") == _RAW_GLOSSARY["Schilit Pass"], (
        "the tip must be the glossary's own entry, read by key")


def test_the_schilit_tip_states_the_engines_rule():
    """'At most 2 of its 4' is what the engine does: rows that fire 0, 1, 2, 3 and 4 of the four
    checkers pass, pass, pass, fail, fail. If the rule moves, the tip's words must move with it."""
    import numpy as np
    import pandas as pd
    from core.forensic_engine import compute_schilit_forensic_score
    from ui.ui_components import _RAW_GLOSSARY
    k = np.arange(5)
    f = pd.DataFrame({
        "accruals_warning":    (k >= 1).astype(int),                                     # 1 · earnings (EMS)
        "cash_machine_label":  np.where(k >= 2, "📄 Paper Profits", "✅ Cash Machine"),  # 2 · cash flow (CFS)
        "high_cash_high_debt": (k >= 3).astype(int),                                     # 3 · leverage (KMS)
        "days_receivable":     np.where(k >= 4, 120.0, 30.0),                            # 4 · bloat (KMS)
    })
    out = compute_schilit_forensic_score(f)
    fired = out[["schilit_ems_flag", "schilit_cfs_flag", "schilit_kms_lev_flag",
                 "schilit_kms_bloat_flag"]].sum(axis=1).tolist()
    assert fired == [0, 1, 2, 3, 4], f"the fixture must fire exactly 0 to 4 checkers, fired {fired}"
    assert out["schilit_pass"].tolist() == [1, 1, 1, 0, 0]
    assert "at most 2 of its 4" in _RAW_GLOSSARY["Schilit Pass"]
