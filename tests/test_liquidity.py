"""
test_liquidity.py — one liquidity number on every surface, and never in the score (2026-10-09).

Measured that day on the four saved snapshots (June → October 2026): 42% of stocks trade under ₹1 Cr a
day, and so do 16 of the top 50 by rank (#1, Infinium Pharmachem, ₹0.28 Cr). Liquidity tracks market
cap (rank correlation 0.88), yet a Small-cap-and-above filter still keeps 308 illiquid stocks and drops
85 liquid micro caps, so Market Cap cannot stand in for it. The Breakout score predicted returns among
liquid stocks (₹1 Cr/day or more) in all four windows, about half as strongly as among illiquid ones;
liquid and illiquid stocks did about as well on median. So liquidity is a BUYABILITY check: shown and
filterable, never a score input.

ONE definition — core.data_engine.traded_value_cr, which the engine materialises as `traded_value_cr` —
and ONE vocabulary (ui/ui_liquidity.py) for the Scans floor, the sidebar filter, the Deep Scanner
column and the Tear-Sheet line.

Run with: pytest tests/test_liquidity.py -v
"""
import ast
import contextlib
import io
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def _src(rel):
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


# ── 1. The definition ─────────────────────────────────────────────────────────────────────────────
def test_the_definition_is_twenty_day_volume_times_price_in_crore():
    from core.data_engine import traded_value_cr
    df = pd.DataFrame({"vol_sma_20d": [1e5, 1e6, np.nan, 2e5, 0.0],
                       "close_price": [1000.0, 10.0, 50.0, np.nan, 100.0]})
    got = traded_value_cr(df)
    assert np.allclose(got.iloc[[0, 1, 4]], [10.0, 1.0, 0.0])
    assert got.iloc[[2, 3]].isna().all(), "a missing volume or price is unknown, never zero"


def test_a_frame_without_the_inputs_reads_unknown_not_zero():
    from core.data_engine import traded_value_cr
    got = traded_value_cr(pd.DataFrame({"name": ["A", "B"]}))
    assert got.isna().all() and len(got) == 2


@pytest.fixture(scope="module")
def raw():
    from core import fetch_and_clean_data
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return fetch_and_clean_data("local")
    except Exception as exc:                                   # the CSVs are gitignored
        pytest.skip(f"local data not available: {exc}")


def test_the_engine_materialises_the_same_number(raw):
    from core.data_engine import traded_value_cr
    assert "traded_value_cr" in raw.columns
    assert np.allclose(raw["traded_value_cr"], traded_value_cr(raw), equal_nan=True)
    assert raw["traded_value_cr"].notna().mean() >= 0.95, "liquidity known for too few stocks"


# ── 2. One definition, one vocabulary ─────────────────────────────────────────────────────────────
def test_every_surface_reads_one_definition_and_one_vocabulary():
    from core import data_engine as de
    from ui import ui_liquidity as L
    from ui import ui_scans as S
    assert S.traded_value_cr is de.traded_value_cr, "the Scans floor and record must use the engine's number"
    assert S.LIQUIDITY_STEPS is L.LIQUIDITY_STEPS and S.ORDER_SHARE == L.ORDER_SHARE
    assert S.order_size_lakh is L.order_size_lakh
    disc = _src("ui/ui_discovery.py")
    assert re.search(r"_narrow\(_cf,\s*traded_value_cr\(_cf\)\s*>=\s*sel_liq", disc), (
        "the sidebar floor must narrow through the choke point with the engine's definition")
    assert '"traded_value_cr"' in _src("ui/ui_tearsheet.py"), "the Tear-Sheet must read the engine's column"


def test_liquidity_never_reaches_a_score_gate_flag_or_verdict():
    """AST: no scoring, forensic or verdict code names the column; in the data engine it appears only
    in its definition and its one assignment."""
    for rel in ("core/scoring_engine.py", "core/forensic_engine.py", "core/verdict_engine.py"):
        tree = ast.parse(_src(rel))
        hits = [n.lineno for n in ast.walk(tree)
                if (isinstance(n, ast.Constant) and n.value == "traded_value_cr")
                or (isinstance(n, ast.Name) and n.id == "traded_value_cr")]
        assert not hits, f"{rel} reads liquidity at lines {hits} — it is a buyability check, not a score input"
    de_src = _src("core/data_engine.py")
    assert de_src.count('df["traded_value_cr"]') == 1, "the engine must only write the column, never read it"


def test_scrambling_liquidity_moves_no_score(raw):
    """Behavioural: reverse the column across stocks before scoring — nothing scored may move."""
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(io.StringIO()):
        a = run_scoring_pipeline(raw.copy()).set_index("company_id")
        mixed = raw.copy()
        mixed["traded_value_cr"] = mixed["traded_value_cr"].to_numpy()[::-1]
        b = run_scoring_pipeline(mixed).set_index("company_id")
    b = b.loc[a.index]
    for col in ("composite_score", "rank", "gate_pass", "conviction_tier", "red_flag_count",
                "wealth_tier", "verdict_direction", "frameworks_passed"):
        same = (a[col] == b[col]) | (a[col].isna() & b[col].isna())
        assert same.all(), f"{col} moved on {int((~same).sum())} stocks when only liquidity changed"


# ── 3. The words ──────────────────────────────────────────────────────────────────────────────────
def test_step_labels_and_order_sizes():
    from ui.ui_liquidity import LIQUIDITY_STEPS, order_size_text, per_day_text, step_label
    assert [step_label(s) for s in LIQUIDITY_STEPS] == ["Any", "₹0.5 Cr/day", "₹1 Cr/day", "₹2 Cr/day",
                                                        "₹5 Cr/day"]
    assert order_size_text(0.28) == "₹1.4 lakh"
    assert order_size_text(2.0) == "₹10 lakh"
    assert order_size_text(50.0) == "₹2.5 Cr"
    assert order_size_text(0.01) == "₹5,000"
    assert per_day_text(0.28) == "₹0.28 Cr/day"
    assert per_day_text(17.98) == "₹18.0 Cr/day"
    assert per_day_text(1234.4) == "₹1,234 Cr/day"
    # 46 stocks trade under ₹0.005 Cr a day (2026-10-03): never print a '₹0.00' that reads as no trading
    assert per_day_text(0.004) == "under ₹0.01 Cr/day"
    assert per_day_text(99.96) == "₹100 Cr/day", "a rounding boundary must not print '₹100.0'"
    assert per_day_text(0.996) == "₹1.0 Cr/day"


def test_the_tear_sheet_line_says_no_trades_and_nothing_for_unknown():
    from ui.ui_liquidity import trade_line
    assert trade_line(0.0) == "no trades in the last 20 days"
    assert trade_line(0.28) == "trades ₹0.28 Cr/day · orders up to about ₹1.4 lakh"
    for unknown in (float("nan"), None, "n/a", -1.0):
        assert trade_line(unknown) == "", f"{unknown!r} must say nothing, not a made-up figure"


def test_the_glossary_explains_the_number_and_what_to_do_with_it():
    from ui.ui_components import _RAW_GLOSSARY
    from ui.ui_liquidity import ORDER_SHARE
    from ui.ui_scanner import _SCANNER_HEADER_TIPS
    text = _RAW_GLOSSARY["Liquidity"]
    for must in ("20 days", "₹ crore", f"{ORDER_SHARE:.0%}", "not a score", "ASM/GSM"):
        assert must in text, f"the Liquidity glossary entry lost {must!r}"
    assert _SCANNER_HEADER_TIPS["traded_value_cr"] == text, "the Deep Scanner header must read the glossary"


# ── 4. The sidebar floor, driven through the real widget machinery ────────────────────────────────
def _liquidity_app():
    import pandas as _pd
    import streamlit as _st
    from ui.ui_discovery import render_discovery_sidebar
    df = _pd.DataFrame({
        "company_id": ["A", "B", "C", "D"],
        "name": ["A", "B", "C", "D"],
        "market_category": ["Small Cap"] * 4,
        "sector": ["P"] * 4,
        "industry": ["P-ind"] * 4,
        # the columns the sidebar always reads (the same set as test_discovery_filters' _widening_app)
        "wealth_tier": ["BUY★"] * 4,
        "conviction_tier": [1, 1, 1, 1],
        "piotroski_fscore": [5, 5, 5, 5],
        "quality_score": [50.0] * 4,
        "composite_score": [50.0] * 4,
        "red_flag_count": [0, 1, 0, 0],                 # a range for the Max-red-flags slider
        # A: ₹10 Cr/day · B: ₹0.1 Cr/day · C: volume unknown · D: exactly ₹1 Cr/day
        "vol_sma_20d": [1e5, 1e3, float("nan"), 1e4],
        "close_price": [1000.0, 1000.0, 1000.0, 1000.0],
    })
    filt = render_discovery_sidebar(df)
    _st.text(f"KEPT={sorted(filt['name'].tolist())}")


@pytest.mark.parametrize("floor,expected", [
    (None, ["A", "B", "C", "D"]),     # Any = no floor: unknown liquidity stays
    (1.0, ["A", "D"]),                # the floor is inclusive; unknown liquidity is not passed
    (0.5, ["A", "D"]),
    (5.0, ["A"]),
    ("₹1 Cr/day", ["A", "D"]),        # a round-tripped LABEL still filters (the 2026-09-21 class)
], ids=["any", "1cr", "0.5cr", "5cr", "label"])
def test_the_sidebar_floor_keeps_what_trades_enough(floor, expected):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_liquidity_app)
    at.session_state["sb_liq"] = floor
    at.run(timeout=30)
    assert not at.exception, at.exception
    kept = next(t.value for t in at.text if t.value.startswith("KEPT="))
    assert kept == f"KEPT={expected}"


def test_the_sidebar_floor_is_a_registered_filter_with_a_readable_chip():
    from ui.ui_discovery import _CHIP_META, _compute_active_chips
    assert ("sb_liq", "Liquidity", "liq") in _CHIP_META
    chips = dict(_compute_active_chips({"sb_liq": 1.0}, rf_max=28))
    assert chips["sb_liq"] == "Liquidity: ≥₹1 Cr/day"
    assert "sb_liq" not in dict(_compute_active_chips({"sb_liq": None}, rf_max=28)), "Any is not a filter"
    disc = _src("ui/ui_discovery.py")
    grp = next(l for l in disc.splitlines() if '_grp("🏢 Universe"' in l)
    assert '"sb_liq"' in grp, "the Universe group badge must count the floor"
    # placed directly after Market Category: size, then whether you can buy it in your size
    assert disc.index('"Market Category"') < disc.index('key="sb_liq"') < disc.index('"Sector", _sector_opts')


# ── 5. The Deep Scanner column ────────────────────────────────────────────────────────────────────
def _app_tree():
    return ast.parse(_src("app.py"))


def _dict_assigned(name):
    for node in ast.walk(_app_tree()):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise AssertionError(f"{name} not found in app.py")


def test_core_shows_liquidity_right_after_market_cap():
    views = ast.literal_eval(_dict_assigned("_DS_VIEWS"))
    core = views["🏆 Core"]
    assert core.index("traded_value_cr") == core.index("market_category") + 1
    assert sum("traded_value_cr" in cols for cols in views.values()) == 1, "one home per column"


def test_the_column_wears_the_one_liquidity_header_and_shows_small_values():
    from ui.ui_liquidity import HEADER
    fmt = ast.literal_eval(_dict_assigned("_num_fmt"))
    label, spec = fmt["traded_value_cr"]
    assert label == HEADER, "the Deep Scanner and the Scans table must head this number the same way"
    assert spec == "%.2f", "a stock trading ₹4 lakh a day must not read 0.0"
    assert "HEADER" in _src("ui/ui_scans.py"), "the Scans table must take its header from the vocabulary"


# ── 6. The Tear-Sheet line ────────────────────────────────────────────────────────────────────────
def _hero_text(row):
    import ui.ui_tearsheet as T
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
        T.render_stock_hero(row)
    finally:
        T.st = real
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", " ".join(out)))


def test_the_tear_sheet_says_how_much_trades_and_what_order_fits():
    row = pd.Series({"name": "X", "market_cap": 500.0, "close_price": 100.0, "traded_value_cr": 0.28})
    text = _hero_text(row)
    assert "trades ₹0.28 Cr/day" in text and "orders up to about ₹1.4 lakh" in text, text


def test_the_tear_sheet_says_nothing_when_liquidity_is_unknown():
    row = pd.Series({"name": "X", "market_cap": 500.0, "close_price": 100.0, "traded_value_cr": np.nan})
    assert "Cr/day" not in _hero_text(row)
