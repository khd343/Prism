"""
test_market_pulse_views.py
==========================
Contract for the column VIEWS on Market Pulse → 📈 Sectors and 🏭 Industry (2026-10-05):
🏆 Core · 📊 Quality · 💰 Valuation · 🔬 Forensic · 📈 Technical · 👥 Ownership — the Deep Scanner's
views, so there is one vocabulary in the app.

WHY. Both tables had outgrown the screen: at a 1536px viewport Sectors carried 1,292px of content in
995px and Industry 1,567px, so Momentum, Valuation and the dominant sector sat behind a sideways
scroll. Views show five or six columns at a time, and they make room for what a sector table could
not show at all — breadth above the 200-day average, the median P/E, the average red-flag count.

WHAT IS PINNED, each a way this breaks silently:
  1. HOW a group is summarised (ui/ui_group_views.py): 0-100 scores by MEAN, raw multiples by
     MEDIAN (one P/E of 900 must not move a sector), yes/no conditions as a SHARE counted only among
     stocks with the data (an unknown is not a "no" — INCLUDING the unknowns the engine records as 0:
     no 200-day average yet, no cash-flow figure; the first draft's toy wrote them as NaN, which the
     engine never does, so it passed while 180 recent listings were counted "below"), and the "few
     red flags" line read from the forensic penalty tiers, never retyped.
  2. Both tables receive the measures from the SAME filtered stocks as their other columns — and
     🏭 Industry's fixed column list carries every input (that list crashed this tab once already).
  3. Every view leads with Count and the tab's ranking column, so the order always makes sense:
     switching views changes the COLUMNS, never the RANKING. Industry keeps its sector last.
  4. Every column on screen has a header — none renders as a raw snake_case name.
  5. The view is a display choice, not a filter: 🧹 Clear never resets it.
  6. On live data no two columns in one view move together (rank correlation < 0.85) and none is
     flat — the test that chose them (2026-10-05) dropped median ROCE (0.90 with Quality), earnings
     yield (-0.99 with P/E), % at the penalty floor (0.90 with average red flags), RS score (0.96
     with Momentum), breakout score (0.90 with distance from the high), the median 52-week RS
     (0.85 with % above the 200-day average — found by THIS file's live test, which the first
     measurement never paired), dividend yield and pledge (median 0 almost everywhere).
  7. 👥 OWNERSHIP (2026-10-05), in all three places: each group measure on a toy where every column
     carries an unknown; net breadth = % raised − % cut among companies WITH a reading; the note that
     shareholding is filed quarterly shows with the view and only with it; per stock, each stake sits
     beside its change, changes read in percentage points, and the tips are the glossary's own.
  8. ONE HOME (2026-10-08): a group column sits in the view whose Deep Scanner twin shows the same
     stock-level column — Piotroski 7+ had sat under 📊 Quality while the F-Score lives in the Deep
     Scanner's 🔬 Forensic view.

Run with: pytest tests/test_market_pulse_views.py -v
"""

import ast
import contextlib
import io as _io
import os
import re
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

from ui.ui_group_views import (MEASURES, OWNERSHIP, OWNERSHIP_GROUP_NOTE, OWNERSHIP_NOTE, SOURCE_COLUMNS,
                                VIEW_MEASURES, VIEWS, _per_stock, group_measures)

_APP = os.path.join(os.path.dirname(__file__), "..", "app.py")
SRC = _io.open(_APP, encoding="utf-8").read()
TREE = ast.parse(SRC)
CORE = "🏆 Core"
_TABLE_AVGS = {"avg_quality", "avg_improvement", "avg_momentum", "avg_valuation"}


# ── 1. How a group is summarised ────────────────────────────────────────────────────────────────
def _toy():
    return pd.DataFrame({
        "sector":           ["A", "A", "A", "B", "B"],
        "pe":               [10.0, 20.0, 900.0, -5.0, 30.0],
        # The ENGINE's encoding: above_sma200 = close > sma_200d, so a stock with no 200-day average
        # yet (row 3) carries 0 — never NaN. The share must leave it out, not count it below.
        "above_sma200":     [1.0, 0.0, 0.0, 1.0, 1.0],
        "close_price":      [100.0, 80.0, 100.0, 50.0, 60.0],
        "sma_200d":         [90.0, 90.0, np.nan, 40.0, 50.0],
        "weinstein_stage":  ["📈 Stage 2 Advancing", "❔ Unknown", "📉 Stage 4 Declining", np.nan,
                             "📈 Stage 2 Advancing"],
        "red_flag_count":   [0.0, 2.0, 3.0, 6.0, 1.0],
    })


def test_a_raw_multiple_is_summarised_by_its_median():
    g = group_measures(_toy(), "sector")
    assert g.loc["A", "grp_pe"] == 20.0, "one P/E of 900 must not move the group (the mean is 310)"


def test_a_loss_maker_has_no_pe():
    assert group_measures(_toy(), "sector").loc["B", "grp_pe"] == 30.0


def test_a_share_counts_only_stocks_with_the_data():
    g = group_measures(_toy(), "sector")
    assert g.loc["A", "grp_above_200dma"] == 50.0, (
        "1 of the 2 stocks WITH a 200-day average — the one without carries the engine's 0 and must "
        "be left out, not counted below (33% would mean it was)")
    assert g.loc["B", "grp_above_200dma"] == 100.0


def test_a_stock_the_engine_could_not_judge_has_no_accrual_reading():
    """rf_high_accruals is 0 — not blank — when profit, cash flow or assets are missing (the engine's
    guard), so each missing input must take the stock out of the share. One per row, so dropping any
    one leg of the guard moves the share. The last row is the case the first draft got wrong: this
    year's assets blank, last year's present — the engine falls back and JUDGES it, so it counts."""
    f = pd.DataFrame({
        "sector":              ["A"] * 6,
        "rf_high_accruals":    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "pat_cf_year":         [10.0, 10.0, np.nan, 10.0, 10.0, 10.0],
        "operating_cash_flow": [2.0, 9.0, 9.0, np.nan, 9.0, 9.0],
        "total_assets":        [100.0, 100.0, 100.0, 100.0, np.nan, np.nan],
        "total_assets_1yb":    [100.0, 100.0, 100.0, 100.0, np.nan, 100.0],
    })
    assert np.isclose(group_measures(f, "sector").loc["A", "grp_high_accruals"], 100 / 3), (
        "1 of the 3 stocks the engine judged — 16.7% counts the unjudged as 'no', 25% drops one leg "
        "of the guard, 50% ignores the engine's fallback to last year's assets")


def test_an_unknown_stage_is_left_out_of_the_share():
    g = group_measures(_toy(), "sector")
    assert g.loc["A", "grp_stage2"] == 50.0 and g.loc["B", "grp_stage2"] == 100.0


def test_the_few_flags_line_comes_from_the_penalty_tiers():
    from config import FORENSIC_PENALTY_TIERS
    watch = next(t["max_flags"] for t in FORENSIC_PENALTY_TIERS if t["multiplier"] == 0.90)
    flags = _toy()["red_flag_count"]
    want_a = 100.0 * (flags[:3] <= watch).mean()
    assert np.isclose(group_measures(_toy(), "sector").loc["A", "grp_few_flags"], want_a)
    assert str(watch) in MEASURES["grp_few_flags"]["header"], "the header must state the tier's line"


def test_summarising_never_mutates_its_input():
    f = _toy(); before = f.copy()
    group_measures(f, "sector")
    pd.testing.assert_frame_equal(f, before)


def test_every_view_column_is_defined():
    for view, cols in VIEW_MEASURES.items():
        for c in cols:
            assert c in MEASURES or c in _TABLE_AVGS, f"{view}: {c} is neither a measure nor a table average"


def test_every_measure_reads_a_declared_source_column():
    for k, m in MEASURES.items():
        for c in (m["source"], *m.get("needs", ())):
            assert c in SOURCE_COLUMNS, f"{k} reads {c}, which SOURCE_COLUMNS does not list"


def test_the_red_flag_total_is_read_from_config():
    """'out of 28' retyped would go stale the day a 29th check ships (FORENSIC_MAX_FLAGS is bumped
    by hand when one does) — the tooltip reads the constant."""
    from config import FORENSIC_MAX_FLAGS
    assert f"out of {FORENSIC_MAX_FLAGS} checks" in MEASURES["grp_red_flags"]["help"]
    src = _io.open(os.path.join(os.path.dirname(__file__), "..", "ui", "ui_group_views.py"),
                   encoding="utf-8").read()
    entry = next(v for k, v in zip(*(lambda d: (d.keys, d.values))(next(
        n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "MEASURES" for t in n.targets))))
        if isinstance(k, ast.Constant) and k.value == "grp_red_flags")
    help_kw = next(kw.value for kw in entry.keywords if kw.arg == "help")
    assert "FORENSIC_MAX_FLAGS" in ast.unparse(help_kw), "the red-flag total is retyped, not read"


def test_the_views_are_the_deep_scanner_views():
    """One vocabulary in the app: same names, same order, as the Deep Scanner's Column View."""
    ds = next(n.value for n in ast.walk(TREE) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "_DS_VIEWS" for t in n.targets))
    assert tuple(k.value for k in ds.keys) == VIEWS
    assert VIEWS[0] == CORE and set(VIEW_MEASURES) == set(VIEWS[1:])


# ── 2-5. The two tables ─────────────────────────────────────────────────────────────────────────
TABLES = {"sectors": dict(key="sector", src="_sec_src", stats="_sec_stats", core="_sec_order",
                          show="_sec_show", view_var="_sec_view", view_key="mp_sec_view", defaults="_SEC_DEFAULTS",
                          lead=["stocks", "pct_qualify"], tail=[], name_hdr="Sector"),
          "industry": dict(key="industry", src="_ind_src", stats="_ind_stats", core="_ind_order",
                           show="_ind_show", view_var="_ind_view", view_key="mp_ind_view", defaults="_IND_DEFAULTS",
                           lead=["stocks", "delta_vs_sector"], tail=["dom_sector"], name_hdr="Industry")}
_BLOCKS = {"sectors": ("# ══ Sectors ══", "# ══ Industry ══"), "industry": ("# ══ Industry ══", "# ══ Movers ══")}


def _lines(table):
    a, b = _BLOCKS[table]
    return SRC[:SRC.index(a)].count("\n") + 1, SRC[:SRC.index(b)].count("\n") + 1


def _in_block(node, table):
    lo, hi = _lines(table)
    return lo <= node.lineno <= hi


def _assigned_in(table, name):
    hits = [n for n in ast.walk(TREE) if isinstance(n, ast.Assign) and _in_block(n, table)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(hits) == 1, f"{table}: expected ONE assignment to {name}, found {len(hits)}"
    return hits[0]


def _imported(name):
    for n in ast.walk(TREE):
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if (a.asname or a.name) == name:
                    return getattr(__import__(n.module, fromlist=[a.name]), a.name)
    raise AssertionError(f"{name} is neither assigned nor imported in app.py")


def _run(node, ns):
    expr = ast.fix_missing_locations(ast.Expression(body=node))
    return eval(compile(expr, "<app.py>", "eval"), {"pd": pd, "np": np, **ns})


def _view_fn():
    """Lift the shared view helper out of the Market Pulse fragment and make it callable."""
    lines = SRC.splitlines()
    i = next(j for j, l in enumerate(lines) if l.strip().startswith("def _mp_view_columns("))
    ind = len(lines[i]) - len(lines[i].lstrip()); body = [lines[i]]
    for nxt in lines[i + 1:]:
        if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= ind:
            break
        body.append(nxt)
    ns = {"pd": pd, "np": np, "VIEW_MEASURES": VIEW_MEASURES}
    exec(compile(textwrap.dedent("\n".join(body)), "<_mp_view_columns>", "exec"), ns)
    return ns["_mp_view_columns"]


def _all_columns(table):
    t = TABLES[table]
    core = [c.value for c in ast.walk(_assigned_in(table, t["core"]).value)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)]
    return pd.DataFrame(columns=sorted(set(core) | set(MEASURES) | _TABLE_AVGS | set(t["lead"] + t["tail"])))


def _shown(table, view):
    """What the table ACTUALLY shows for a view: the app's own `_sec_show = _mp_view_columns(...)`
    call, evaluated with the app's own arguments. (A first draft passed the expected lead/tail in
    itself, so a wrong argument in app.py was invisible — two mutations proved it, 2026-10-05.)"""
    t = TABLES[table]
    stats = _all_columns(table)
    core = _run(_assigned_in(table, t["core"]).value, {t["stats"]: stats})
    call = _assigned_in(table, t["show"]).value
    return _run(call, {"_mp_view_columns": _view_fn(), t["view_var"]: view, t["core"]: core,
                       t["stats"]: stats}), core


@pytest.mark.parametrize("table", sorted(TABLES))
def test_both_tables_join_the_measures_of_their_own_filtered_stocks(table):
    t = TABLES[table]
    calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call) and _in_block(n, table)
             and getattr(n.func, "id", "") == "group_measures"]
    assert len(calls) == 1, f"{table}: expected one group_measures call, found {len(calls)}"
    c = calls[0]
    assert ast.unparse(c.args[0]) == t["src"] and c.args[1].value == t["key"], (
        f"{table}: measures must come from {t['src']} grouped by {t['key']} — the frame every filter "
        f"and the 🎯 scope already narrowed — found {ast.unparse(c)}")
    joins = [n for n in ast.walk(TREE) if isinstance(n, ast.Assign) and _in_block(n, table)
             and ast.unparse(n.value) == f"{t['stats']}.join({ast.unparse(c)})"]
    assert len(joins) == 1, f"{table}: the measures are computed but not joined onto {t['stats']}"


def test_the_industry_column_list_carries_every_measure_input():
    """🏭 Industry projects its stocks to a fixed list before averaging — improvement_score was missing
    from it once and the tab died with a KeyError. Every measure's input must survive the projection."""
    a = _assigned_in("industry", "_IND_KEEP")
    ns = {"_mp_df": pd.DataFrame(columns=sorted(set(SOURCE_COLUMNS) | {"industry", "sector", "name"}))}
    bound = {x.id for comp in ast.walk(a.value) if isinstance(comp, ast.comprehension)
             for x in ast.walk(comp.target) if isinstance(x, ast.Name)}
    for name in sorted({x.id for x in ast.walk(a.value) if isinstance(x, ast.Name)}
                       - {"_mp_df", "dict", "list"} - bound):
        ns[name] = _imported(name)
    keep = _run(a.value, ns)
    missing = sorted(set(SOURCE_COLUMNS) - set(keep))
    assert not missing, f"_IND_KEEP drops measure inputs: {missing}"
    assert len(keep) == len(set(keep)), "_IND_KEEP lists a column twice — the projection would duplicate it"


@pytest.mark.parametrize("table", sorted(TABLES))
@pytest.mark.parametrize("view", VIEWS)
def test_every_view_leads_with_count_and_the_ranking_column(table, view):
    t = TABLES[table]
    cols, core = _shown(table, view)
    # Core keeps the order the tab chose for it (Industry: Count, % Qualify, Score, then Δ — pinned
    # <= 3 by test_industry_tab); every view must still show Count FIRST and the ranking column.
    assert cols[0] == "stocks" and t["lead"][1] in cols, f"{table}/{view}: no Count first / no ranking column"
    if view != CORE:
        assert cols[:2] == t["lead"], f"{table}/{view} leads with {cols[:2]}, not Count + the ranking column"
    if t["tail"]:
        assert cols[-1] == t["tail"][-1], f"{table}/{view}: the wide sector column must come last"
    if view == CORE:
        assert cols == core, f"{table}: Core must be the tab's own column list"
    else:
        # A table may skip a measure that earned its place on the other table only — read from the
        # app's own skip= argument and pinned in test_near_52w_high_is_on_sectors_only.
        want = [m for m in VIEW_MEASURES[view] if m not in _skip(table)]
        assert cols[len(t["lead"]):len(cols) - len(t["tail"])] == want


def _skip(table):
    """The measures a table declares it skips: the skip= argument of its own _mp_view_columns call."""
    call = _assigned_in(table, TABLES[table]["show"]).value
    kw = [k for k in call.keywords if k.arg == "skip"]
    return set(ast.literal_eval(kw[0].value)) if kw else set()


def test_near_52w_high_is_on_sectors_only():
    """📈 Near 52w high (2026-10-05) passed the pre-declared bar for sectors (+0.11 to +0.31 beyond
    Momentum in all four windows) and failed it for industries in one window (-0.01). So Sectors'
    Technical view shows it and Industry's skips it — and Industry skips nothing else."""
    tech = "📈 Technical"
    assert "grp_near_high" in VIEW_MEASURES[tech]
    sec, _ = _shown("sectors", tech)
    ind, _ = _shown("industry", tech)
    assert "grp_near_high" in sec, "Sectors' Technical view must show Near 52w high"
    assert "grp_near_high" not in ind, "industry breadth failed the bar — December re-tests it first"
    assert _skip("industry") == {"grp_near_high"} and _skip("sectors") == set(), (
        "only the Industry table skips, and only this measure")
    assert sec.index("grp_near_high") == sec.index("grp_from_52w_high") - 1, (
        "Near 52w high reads beside Below 52w high: the share near the highs, then the typical distance")


def test_near_52w_high_counts_the_share_within_5pct_among_stocks_with_a_high():
    """Within 5% counts (5.0 is in, 5.01 is out), and a stock with no 52-week high is left out, never
    counted as not near: an unknown is not a no."""
    f = pd.DataFrame({"sector": ["A"] * 5 + ["B"], "dist_52wh": [0.0, 5.0, 5.01, 30.0, np.nan, np.nan]})
    g = group_measures(f, "sector")
    assert g.loc["A", "grp_near_high"] == 50.0, "2 of the 4 stocks with a 52-week high are within 5%"
    assert pd.isna(g.loc["B", "grp_near_high"]), "a group with no reading has no share, not 0%"


@pytest.mark.parametrize("table", sorted(TABLES))
def test_the_table_shows_the_selected_views_columns(table):
    t = TABLES[table]
    show = _assigned_in(table, t["show"])
    assert ast.unparse(show.value).startswith("_mp_view_columns("), f"{table}: {t['show']} is not the view's list"
    frames = [n for n in ast.walk(TREE) if isinstance(n, ast.Call) and _in_block(n, table)
              and getattr(n.func, "attr", "") == "dataframe" and n.args
              and ast.unparse(n.args[0]) == f"{t['stats']}[{t['show']}].reset_index()"]
    assert len(frames) == 1, f"{table}: the table does not display {t['show']}"


def _cfg(table):
    t = TABLES[table]
    for n in ast.walk(TREE):
        if (isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "dataframe" and n.args
                and ast.unparse(n.args[0]) == f"{t['stats']}[{t['show']}].reset_index()"):
            return next(k.value for k in n.keywords if k.arg == "column_config")
    raise AssertionError(f"{table}: table not found")


@pytest.mark.parametrize("table", sorted(TABLES))
def test_every_column_in_every_view_has_a_header(table):
    t = TABLES[table]
    cfg = _cfg(table)
    literal = {k.value for k in cfg.keys if isinstance(k, ast.Constant)}
    spread = [ast.unparse(v) for k, v in zip(cfg.keys, cfg.values) if k is None]
    assert spread == ["_MP_VIEW_CFG"], f"{table}: the measures' headers are not merged in ({spread})"
    configured = literal | set(MEASURES)
    for view in VIEWS:
        cols = _shown(table, view)[0] + [t["key"]]
        raw = [c for c in cols if c not in configured]
        assert not raw, f"{table}/{view}: no header for {raw} — they would render as raw names"


def test_the_measure_headers_are_built_once_from_the_registry():
    hits = [n for n in ast.walk(TREE) if isinstance(n, ast.Assign)
            and any(isinstance(x, ast.Name) and x.id == "_MP_VIEW_CFG" for x in n.targets)]
    assert len(hits) == 1, "the measure headers must be built ONCE and shared by both tables"
    assert "MEASURES" in ast.unparse(hits[0].value), "the headers must come from the MEASURES registry"


@pytest.mark.parametrize("table", sorted(TABLES))
def test_the_name_column_is_never_faded(table):
    """Repointed 2026-10-06: the name column WAS pinned so a sideways scroll kept it, and Streamlit draws
    every pinned column faded, so the sector / industry name read grey (tests/test_no_faded_columns.py).
    Every view fits a 1536px screen, so the name stays in view unpinned; this pins the header and that
    the name and Count columns are never pinned again."""
    t = TABLES[table]
    cfg = dict(zip([k.value if isinstance(k, ast.Constant) else None for k in _cfg(table).keys], _cfg(table).values))
    v = cfg[t["key"]]
    assert ast.unparse(v.args[0]) == repr(t["name_hdr"]), f"{table}: name header changed"
    for col in (t["key"], "stocks"):
        assert not any(k.arg == "pinned" for k in cfg[col].keywords), (
            f"{table}: {col} is pinned again, and Streamlit draws pinned columns faded (grey)")


@pytest.mark.parametrize("table", sorted(TABLES))
def test_the_view_is_a_display_choice_not_a_filter(table):
    """Same as the Deep Scanner's Column View: 🧹 Clear resets FILTERS. Resetting the view as well
    would make the Clear button appear the moment someone picks a view, as if a filter were set."""
    t = TABLES[table]
    # A segmented control since 2026-10-05 (tests/test_deep_scanner_views.py): the widget for
    # 'pick one view' — pills read as tags or filters.
    switch = [n for n in ast.walk(TREE) if isinstance(n, ast.Call) and _in_block(n, table)
              and ast.unparse(n.func) == "st.segmented_control"]
    assert len(switch) == 1, f"{table}: expected one view switcher, found {len(switch)}"
    kw = {k.arg: k.value for k in switch[0].keywords}
    assert ast.unparse(kw["key"]) == repr(t["view_key"]) and ast.unparse(kw["default"]) == repr(CORE)
    assert ast.unparse(switch[0].args[1]) == "list(VIEWS)", f"{table}: the switcher must offer the shared VIEWS"
    d = _assigned_in(table, t["defaults"]).value
    assert t["view_key"] not in {k.value for k in d.keys}, f"{table}: 🧹 Clear would reset the view"


@pytest.mark.parametrize("table", sorted(TABLES))
def test_switching_view_never_changes_the_ranking(table):
    names = set(MEASURES) | {TABLES[table]["view_key"], "_sec_view", "_ind_view"}
    for n in ast.walk(TREE):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "sort_values" and _in_block(n, table):
            used = {c.value for c in ast.walk(n) if isinstance(c, ast.Constant)} | \
                   {c.id for c in ast.walk(n) if isinstance(c, ast.Name)}
            assert not (used & names), f"{table}: a sort reads the view or a view measure: {used & names}"


# ── 6. Live data: every column earns its place ──────────────────────────────────────────────────
@pytest.fixture(scope="module")
def live():
    from core import run_scoring_pipeline
    from data_engine import coerce_numeric_columns, compute_derived_signals, load_all_csvs, merge_datasets
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


def _groups(live, level, floor):
    d = live.copy()
    d[level] = d[level].astype(str).str.strip()
    g = group_measures(d, level)
    for c, src in [("avg_quality", "quality_score"), ("avg_improvement", "improvement_score"),
                   ("avg_momentum", "momentum_score"), ("avg_valuation", "valuation_score")]:
        g[c] = d.groupby(level)[src].mean()
    n = d.groupby(level).size()
    return g[n.reindex(g.index) >= floor]


@pytest.mark.parametrize("level,floor", [("sector", 5), ("industry", 3)])
def test_no_two_columns_in_a_view_move_together(live, level, floor):
    g = _groups(live, level, floor)
    assert len(g) >= 30
    bad = []
    for view, cols in VIEW_MEASURES.items():
        r = g[list(cols)].rank().corr()
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                if abs(r.loc[a, b]) >= 0.85:
                    bad.append(f"{view}: {a} ~ {b} ({r.loc[a, b]:+.2f})")
    assert not bad, "columns that tell the same story — keep one: " + "; ".join(bad)


def test_every_measure_varies_across_sectors(live):
    g = _groups(live, "sector", 5)
    flat = [k for k in MEASURES if g[k].quantile(0.9) - g[k].quantile(0.1) <= 0]
    assert not flat, f"flat across sectors (a column that says the same thing on every row): {flat}"


def test_every_source_column_exists_in_the_live_frame(live):
    missing = sorted(set(SOURCE_COLUMNS) - set(live.columns))
    assert not missing, f"measures read columns the engine no longer produces: {missing}"


# Shares that mirror an engine 0/1 FLAG behind a needs-mask (above_sma200, rf_high_accruals). The Strong
# qtr share (2026-10-05) is not one: its value IS the engine's own 1 / 0 / NaN definition, so there is no
# flag column to mirror, and tests/test_strong_quarter.py pins its judgeable share instead.
_JUDGED = sorted(k for k, m in MEASURES.items() if m["stat"] == "share" and m.get("needs")
                 and k != "grp_strong_quarter")


@pytest.mark.parametrize("key", _JUDGED)
def test_a_share_never_leaves_out_a_stock_the_engine_flagged(live, key):
    """The `needs` mask removes what the engine COULD NOT judge — never a stock it did flag, and
    never a large part of the universe (a mask that leans on a sparse column would quietly turn a
    group share into a share of a few stocks)."""
    assert len(_JUDGED) >= 2, "the measures that need inputs are gone — the test has lost its teeth"
    v = _per_stock(live, MEASURES[key])
    flagged = pd.to_numeric(live[MEASURES[key]["source"]], errors="coerce") == 1
    dropped = flagged & v.isna()
    assert not dropped.any(), f"{key} leaves out {int(dropped.sum())} stocks the engine DID flag"
    assert v.notna().mean() >= 0.85, f"{key} judges only {v.notna().mean():.0%} of the universe"


# ── 7. 👥 Ownership ─────────────────────────────────────────────────────────────────────────────
def _own():
    """One group of five companies. Every column carries one unknown, so 'an unknown is not a no'
    is tested on each measure, and the changes carry a raise, a cut and a flat quarter."""
    return pd.DataFrame({
        "sector":             ["A"] * 5,
        "promoter_holdings":  [10.0, 60.0, 70.0, 65.0, np.nan],
        "pledged_percentage": [0.0, 4.5, np.nan, 0.0, 0.0],
        "fii_holdings":       [2.0, 10.0, 1.0, 0.0, 3.0],
        "dii_holdings":       [3.0, np.nan, 1.0, 0.0, 4.0],
        "change_fii_lq":      [0.5, 0.1, -0.2, 0.0, np.nan],
        "change_dii_lq":      [-0.3, -0.1, np.nan, 0.0, 0.2],
    })


def test_promoter_holding_is_summarised_by_its_median():
    assert group_measures(_own(), "sector").loc["A", "grp_promoter"] == 62.5, (
        "median of 10 / 60 / 70 / 65 — the mean is 51.25")


def test_pledged_counts_any_pledge_among_companies_with_data():
    assert group_measures(_own(), "sector").loc["A", "grp_pledged"] == 25.0, (
        "1 of the 4 companies with pledge data — 20 would count the unknown as 'no pledge'")


def test_institutions_need_both_the_fii_and_the_dii_stake():
    assert group_measures(_own(), "sector").loc["A", "grp_institutions"] == 3.5, (
        "median of 5 / 2 / 0 / 7 — a company with no DII figure has no combined stake; 5.0 would "
        "mean its FII stake alone was read as the institutions' whole holding")


def test_net_breadth_is_raised_minus_cut_among_companies_with_a_reading():
    g = group_measures(_own(), "sector")
    assert g.loc["A", "grp_fii_net"] == 25.0, (
        "(2 raised - 1 cut) / 4 with a reading: 20 counts the blank as flat, 33 drops the flat "
        "quarter, 50 is the raised share alone, 0.25 forgot the percentage")
    assert g.loc["A", "grp_dii_net"] == -25.0


def test_the_ownership_view_carries_the_agreed_five():
    assert VIEW_MEASURES[OWNERSHIP] == ("grp_promoter", "grp_pledged", "grp_institutions",
                                        "grp_fii_net", "grp_dii_net")


def test_every_number_measure_declares_its_width():
    missing = sorted(k for k, m in MEASURES.items() if not m["bar"] and "width" not in m)
    assert not missing, f"number columns without a width raise a KeyError on render: {missing}"


_NOTE_SITES = {"deep scanner": ("ds_view", "OWNERSHIP_NOTE"),
               "sectors": ("_sec_view", "OWNERSHIP_GROUP_NOTE"),
               "industry": ("_ind_view", "OWNERSHIP_GROUP_NOTE")}


@pytest.mark.parametrize("site", sorted(_NOTE_SITES))
def test_the_ownership_view_says_its_data_is_quarterly(site):
    """Shareholding is filed quarterly, so the view says so beside the table — never presented as
    this week's trading. Exactly one `if <view> == OWNERSHIP:` per site whose whole body is the
    caption: it shows with the view and only with it (`if False and ...` is not that shape)."""
    var, note = _NOTE_SITES[site]
    hits = [n for n in ast.walk(TREE) if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
            and ast.unparse(n.test) == f"{var} == OWNERSHIP"]
    assert len(hits) == 1, f"{site}: expected one `if {var} == OWNERSHIP:`, found {len(hits)}"
    body = [ast.unparse(x) for x in hits[0].body]
    assert body == [f"st.caption({note})"], f"{site}: the Ownership branch shows {body}"


def test_the_notes_say_what_the_numbers_are():
    assert "filed quarterly" in OWNERSHIP_NOTE and "latest filed quarter" in OWNERSHIP_NOTE
    assert OWNERSHIP_GROUP_NOTE.startswith(OWNERSHIP_NOTE), "the tables extend the shared note, never fork it"
    assert "companies, not rupees" in OWNERSHIP_GROUP_NOTE
    for k in ("grp_fii_net", "grp_dii_net"):
        assert "companies, not rupees" in MEASURES[k]["help"] and "−100" in MEASURES[k]["help"], k


def _ds_dict(name):
    node = next(n.value for n in ast.walk(TREE) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return dict(zip([k.value for k in node.keys], node.values))


def test_the_deep_scanner_shows_each_stake_beside_its_change():
    v = [e.value for e in _ds_dict("_DS_VIEWS")[OWNERSHIP].elts]
    # Rank · Stock lead every Deep Scanner view since 2026-10-05 (tests/test_deep_scanner_views.py).
    assert v[:2] == ["rank", "name"] and "pledged_percentage" in v and "smart_money_flow" in v
    for level, move in [("promoter_holdings", "change_promoter_1y"), ("fii_holdings", "change_fii_lq"),
                        ("dii_holdings", "change_dii_lq")]:
        assert v.index(move) == v.index(level) + 1, f"{move} must sit directly after {level}"


def test_ownership_changes_read_in_percentage_points():
    cfg = {k: (v.elts[0].value, v.elts[1].value) for k, v in _ds_dict("_num_fmt").items()}
    for c in ("change_promoter_1y", "change_fii_lq", "change_dii_lq"):
        label, f = cfg[c]
        assert f.endswith("pp") and "Δ" in label, f"{c} is a change in percentage points: {label!r} {f!r}"
    for c in ("fii_holdings", "dii_holdings"):
        assert cfg[c][1].endswith("%%"), f"{c} is a stake in %"


def test_the_ownership_tips_are_the_glossarys_own():
    from ui.ui_components import _RAW_GLOSSARY
    from ui.ui_scanner import _SCANNER_HEADER_TIPS
    for col, key in [("change_promoter_1y", "Promoter 1Y Δ"), ("fii_holdings", "FII %"),
                     ("change_fii_lq", "FII Chg"), ("dii_holdings", "DII %"), ("change_dii_lq", "DII Chg")]:
        assert _SCANNER_HEADER_TIPS.get(col) == _RAW_GLOSSARY[key], f"{col} must read the glossary's {key!r}"


# ── 8. One home (2026-10-08) ────────────────────────────────────────────────────────────────────
_AVG_SOURCE = {"avg_quality": "quality_score", "avg_improvement": "improvement_score",
               "avg_momentum": "momentum_score", "avg_valuation": "valuation_score"}


def test_a_group_column_sits_in_the_view_that_shows_its_stock_column():
    """Piotroski 7+ sat under 📊 Quality while the Deep Scanner shows the F-Score in 🔬 Forensic: the
    same reading under two names of home. Core is the overview, so only the five detail views count,
    and a measure with no stock-level twin on screen (EV/EBITDA, above the 200-DMA) has no rule."""
    ds = {k: [e.value for e in v.elts] for k, v in _ds_dict("_DS_VIEWS").items() if k != CORE}
    checked = 0
    for view, cols in VIEW_MEASURES.items():
        for c in cols:
            src = _AVG_SOURCE[c] if c in _AVG_SOURCE else MEASURES[c]["source"]
            homes = sorted(v for v, dcols in ds.items() if src in dcols)
            if len(homes) == 1:
                checked += 1
                assert homes == [view], (
                    f"{c} sits in {view}; the Deep Scanner shows {src} in {homes[0]}")
    assert checked >= 15, f"only {checked} measures have a Deep Scanner twin — the test has lost its teeth"
