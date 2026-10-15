"""
test_wealth_tier_display.py
===========================
Contract for WHERE the wealth tier is displayed — the five-surface rollout of 2026-08-28.

THE PLACEMENT ARCHITECTURE (each surface earns its spot; the omissions are deliberate):

    1. Tearsheet hero        a labeled 💹 pill beside the Soundness band — completes the
                             three-layer sentence where a stock is actually READ. Unlocked by
                             the vocabulary rename: "FLAWED · 💹 RECOVERING" is two labeled layers
                             disagreeing openly, where the pre-2026-08-27 "AVOID · BUY★" was the
                             cf_triangle contradiction.
    2. Matrix & WCS EP strip a fifth card — the strip's four siblings ARE the tier's raw
                             inputs, and the normalized EP%/Vel% appear on NO other per-stock
                             surface.
    3. All Data grid         cells in the MOSL Wealth Creation section (the tier IS that
                             formula operationalized) — the tab's caption promises EVERY
                             decision-grade signal, so absence broke its own contract.
    4. Discovery cards       a chip beside the Soundness chip — filter-result coherence: if
                             you filtered Wealth = RECOVERING, the cards show it.
    5. Deep Scanner          a Core-view column beside Soundness.

    NOT in Tsunami/QGLP tables (width limits — the clipping lesson); NOT twice on any one
    screen (the verdict_strength lesson); N/A and missing render NOTHING (no pill over no
    data); and the ⚠ rides with the tier on every surface, never blended into it.

Run with: pytest tests/test_wealth_tier_display.py -v
"""

import contextlib
import io as _io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

import ui.ui_tearsheet as T
from core.verdict_engine import WEALTH_TIERS

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_APP = os.path.join(_ROOT, "app.py")
_COMPONENTS = os.path.join(_ROOT, "ui", "ui_components.py")


@pytest.fixture(scope="module")
def live():
    from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                             merge_datasets)
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


def _render(fn, row):
    out = []

    class _Rec:
        def markdown(self, *a, **k):
            if a:
                out.append(str(a[0]))

        def __getattr__(self, _n):
            return lambda *a, **k: None

    real = T.st
    try:
        T.st = _Rec()
        fn(row)
    finally:
        T.st = real
    return " ".join(out)


# -- 1. The EP-strip card (rendered, with teeth) ---------------------------------------------
def test_ep_strip_card_renders_tier_and_normalized_inputs(live):
    row = live[live["wealth_tier"] == "CLIMBING★"].iloc[0].copy()
    html = _render(T.render_ep_power_curve_module, row)
    assert "Wealth Tier" in html, "the fifth card is gone from the EP strip"
    assert "CLIMBING★" in html
    seg = html[html.index("Wealth Tier"):][:1600]   # the ~500-char tooltip sits between label and value
    assert re.search(r"EP% [+\-][\d.]+", seg), "the normalized EP% vanished from the sub-line"
    assert re.search(r"Vel% [+\-][\d.]+", seg), "the normalized Vel% vanished from the sub-line"


def test_ep_strip_card_carries_the_warn_marker(live):
    row = live[(live["wealth_tier"] == "CLIMBING★") & (live["wealth_warn"] == 1)]
    if row.empty:
        pytest.skip("no warned CLIMBING★ on live data")
    html = _render(T.render_ep_power_curve_module, row.iloc[0])
    seg = html[html.index("Wealth Tier"):][:1600]
    assert "⚠" in seg, "the ⚠ no longer rides with the tier on the EP-strip card"


def test_ep_strip_card_missing_inputs_render_na_not_a_tier(live):
    row = live.iloc[0].copy()
    row["wealth_tier"] = ""
    row["wealth_ep_pct"] = np.nan
    row["wealth_vel_pct"] = np.nan
    html = _render(T.render_ep_power_curve_module, row)
    seg = html[html.index("Wealth Tier"):][:1600]
    assert "N/A" in seg, "an absent tier must read N/A on the card, never a fabricated tier"


# -- 2. The All Data cells (rendered) --------------------------------------------------------
def test_all_data_grid_carries_the_wealth_cells(live):
    row = live[live["wealth_tier"] == "CLIMBING★"].iloc[0]
    html = _render(T.render_raw_signals, row)
    for label in ("Wealth Tier", "Wealth EP%", "Wealth Vel%"):
        assert label in html, (
            f"{label!r} missing from the All Data grid — the tab's caption promises every "
            f"decision-grade signal, and the tier is one"
        )


def test_all_data_wealth_cells_are_searchable_by_meaning(live):
    """The grid's search matches label + glossary; 'cost of equity' must find the EP% cell."""
    from ui.ui_components import _RAW_GLOSSARY
    for label in ("Wealth Tier", "Wealth EP%", "Wealth Vel%"):
        assert label in _RAW_GLOSSARY, f"{label!r} has no glossary entry — its '?' chip is dead"
    assert "cost of equity" in _RAW_GLOSSARY["Wealth EP%"].lower()


# -- 3. The hero pill (structural — the hero lives in app.py's main body) --------------------
@pytest.fixture(scope="module")
def app_src():
    return _io.open(_APP, encoding="utf-8").read()


def test_hero_builds_the_wealth_pill(app_src):
    i = app_src.index("_wealth_pill = ")
    block = app_src[i - 1200:i + 800]
    assert '"wealth_tier"' in block, "the hero pill no longer reads wealth_tier"
    assert '"wealth_warn"' in block, "the hero pill dropped the ⚠ marker"
    assert "💹" in block, "the pill lost its layer label — an unlabeled tier beside the Soundness band is ambiguous"


def test_hero_pill_is_joined_into_the_band(app_src):
    assert "{_wealth_pill}" in app_src, (
        "the wealth pill is built but never rendered — a control that exists and does nothing"
    )


def test_hero_pill_skips_na(app_src):
    """No pill over no data: N/A must not be a pill colour key."""
    i = app_src.index("_WT_PILL_CLR = {")
    block = app_src[i:app_src.index("}", i) + 1]
    assert '"N/A"' not in block, "N/A grew a pill — unverifiable must render nothing"
    for t in WEALTH_TIERS:
        assert f'"{t}"' in block, f"tier {t} lost its pill colour"


def test_every_surface_colours_every_tier_the_same_way():
    """The hero pill, the Discovery chip and the EP-strip card each key a colour by tier word. One of
    them still keyed on a retired word would paint that tier in no colour at all — silently."""
    comp = _io.open(_COMPONENTS, encoding="utf-8").read()
    i = comp.index("_wt_c = {")
    chip = comp[i:comp.index("}", i) + 1]
    tear = _io.open(os.path.join(_ROOT, "ui", "ui_tearsheet.py"), encoding="utf-8").read()
    j = tear.index("_wt_val = str(stock.get(\"wealth_tier\"")
    card = tear[j:tear.index("_wt_sub", j)]
    for t in WEALTH_TIERS:
        assert f'"{t}"' in chip, f"the Discovery chip has no colour for {t}"
        assert f'"{t}"' in card, f"the EP-strip card has no colour for {t}"
    for old in ("BUY★", "WATCH★", '"BUY"', '"WATCH"', '"AVOID"'):
        assert old not in chip + card, f"a retired word still keys a colour: {old}"


# -- 4. The Discovery card chip + scanner column (structural) --------------------------------
def test_discovery_cards_carry_the_wealth_chip():
    src = _io.open(_COMPONENTS, encoding="utf-8").read()
    i = src.index("_wealth_chip = ")
    block = src[i - 900:i + 600]
    assert '"wealth_tier"' in block and '"wealth_warn"' in block
    assert "{_verdict_chip}{_wealth_chip}" in src, (
        "the wealth chip is not rendered beside the Soundness chip"
    )


def test_scanner_core_view_shows_both_verdicts(app_src):
    i = app_src.index('"🏆 Core":')
    block = app_src[i:app_src.index("]", i) + 1]
    assert '"wealth_tier"' in block, "the Core scanner view lost the wealth tier column"
    assert '"verdict_direction"' in block, "the Core view lost Soundness — both lenses belong there"
    assert '"wealth_tier": "Wealth"' in app_src, "the scanner header map lost the Wealth label"


# -- 5. The deliberate omission --------------------------------------------------------------
def test_tsunami_and_qglp_tables_stay_out(app_src):
    """Width limits (the clipping lesson): the SOUND×CLIMBING★ cross-read lives in the Wealth tab.
    If someone adds the column to these tables, it must be a conscious change against this."""
    for var in ("_ts_cols", "_q_cols"):
        i = app_src.index(var + " = [c for c in [")
        block = app_src[i:app_src.index("]", i + 20) + 1]
        assert "wealth_tier" not in block, (
            f"{var} grew a wealth_tier column — these tables are at width limits and the "
            f"cross-read has a dedicated home (💹 Wealth tab)"
        )


# -- 6. Every table that shows the tier fits its longest word (2026-10-15) -------------------
# Measured in the app at 1536px: the grids draw in 400 14px "Source Sans" with 8px of padding a
# side, and RECOVERING, the widest tier word, is 77px of text, so a tier column needs 93px. The old
# widest, WATCH★, needed exactly the Core view's 70. A "small" column (~75px) would clip RECOVERING.
_TIER_PX_NEEDED = 93


def _app_dict(name):
    import ast
    tree = ast.parse(_io.open(_APP, encoding="utf-8").read())
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and getattr(n.targets[0], "id", None) == name:
            return ast.literal_eval(n.value)
    raise AssertionError(f"{name} not found in app.py")


def test_every_table_that_shows_the_tier_fits_its_longest_word(app_src):
    assert max(len(w) for w in WEALTH_TIERS) <= len("RECOVERING"), (
        "a tier word longer than RECOVERING: re-measure its width in the grid font before shipping it")
    core = _app_dict("_DS_W")["wealth_tier"]
    assert core >= _TIER_PX_NEEDED, f"the Deep Scanner's Wealth column ({core}px) clips RECOVERING"
    i = app_src.index('"wealth_tier":       st.column_config.TextColumn("Tier"')
    assert 'width=_DS_W["wealth_tier"]' in app_src[i:i + 120], "the Wealth tab's Tier column is not sized for RECOVERING"
    import ui.ui_movers as M
    assert M.TIER_COL_PX == core, "the Movers tables and the Deep Scanner size the tier differently"


def test_the_movers_tables_size_both_tier_columns(monkeypatch):
    """Behavioural: a Movers table puts the tier, and the tier it came from, at TIER_COL_PX — not the
    "small" every other text column gets (which clipped RECOVERING)."""
    import ui.ui_movers as M
    seen = {}
    monkeypatch.setattr(M.st, "dataframe", lambda data, **kw: seen.update(kw.get("column_config") or {}))
    frame = pd.DataFrame({"company_id": ["A"], "name": ["Alpha"], "sector": ["S"],
                          "wealth_tier_prev": ["UNEVEN"], "wealth_tier": ["RECOVERING"], "steps": [1]})
    M._table(frame)
    for col in ("wealth_tier", "wealth_tier_prev"):
        assert seen[col]["width"] == M.TIER_COL_PX, f"{col} is not sized for RECOVERING: {seen[col]}"
    assert seen["sector"]["width"] == "small", "the fixture no longer exercises the ordinary text path"
