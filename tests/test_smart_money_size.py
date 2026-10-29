"""Contract: the Smart Money label sees the SIZE of an institutional exit, and the tear-sheet states the move as a
fact (2026-10-29). Display only: no score, rank, framework, verdict, scan recipe or feed reads either.

THE GAP. `smart_money_flow` read only the SIGN of change_fii_lq / change_dii_lq, so a -0.01 move and a -10.4 move
counted the same. Measured on the live sheet (28 Oct 2026, 2,692 stocks): 43 stocks had a quarterly FII or DII exit
of 5 points or more, and they read 18 Distribution, 16 Neutral and 9 Moderate Interest -- the last misleading.
EPack Prefab: FII 11.1% -> 0.71% (-10.4) in one quarter.

THE RULE. A quarterly exit of config.SMART_MONEY_BIG_MOVE_PP points or more, by FII OR DII, reads Distribution,
evaluated FIRST, whatever the volume or the price. NaN is not <= -5, so a missing change never triggers it (no
fillna: unverifiable is not passed). Entries and vqs_score are untouched.

THE CHIP. One quiet line per group whose |quarterly change| >= the same constant, e.g. "FII -10.4 pts . 11.1% ->
0.7%" with a real minus sign: before = now - change, shown only when both are known and before is a real share.
"""
import contextlib
import io
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

import config as C
import ui.ui_tearsheet as ts
from data_engine import (compute_derived_signals, COMMON_COLS, RATIO_COLS, INCOME_COLS, BALANCE_COLS,
                         CASHFLOW_COLS, SHAREHOLDING_COLS, TECHNICAL_COLS)

DIST, MOD, STRONG, ELITE, NEUTRAL = ("❌ Distribution", "✅ Moderate Interest", "🎯 Strong Accumulation",
                                     "🌊💎 Elite Accumulation", "⚪ Neutral")
_MAPPED = sorted(set().union(*(m.values() for m in (COMMON_COLS, RATIO_COLS, INCOME_COLS, BALANCE_COLS,
                                                    CASHFLOW_COLS, SHAREHOLDING_COLS, TECHNICAL_COLS))))
# A stock the OLD rule reads as Moderate Interest: no volume figure (liquidity 10), one group adding (+10),
# relative strength aligned (+20), beating the market over 3 months (+10) -> vqs 50, CRS 50D >= 0.
_MODERATE = {"change_fii_lq": -0.1, "change_dii_lq": 0.5, "crs_50d": 10.0, "crs_26w": 10.0, "crs_52w": 10.0,
             "ret_vs_n500_3m": 5.0}


def _labels(*rows) -> pd.DataFrame:
    """compute_derived_signals over one stock per dict -- every mapped column NaN unless given (CLAUDE.md §6)."""
    n = len(rows)
    base = pd.DataFrame({"company_id": [f"NSE:T{i}" for i in range(n)], "name": [f"Test {i}" for i in range(n)],
                         "sector": "Chemicals", "industry": "Specialty Chemicals", "market_category": "Mid Cap",
                         "market_cap": 5000.0, "close_price": 100.0})
    missing = [c for c in _MAPPED if c not in base.columns]
    df = pd.concat([base, pd.DataFrame(np.nan, index=base.index, columns=missing)], axis=1)
    for i, r in enumerate(rows):
        for k, v in r.items():
            df.loc[i, k] = v
    with contextlib.redirect_stdout(io.StringIO()):
        return compute_derived_signals(df)


# ── 1. the label ───────────────────────────────────────────────────────────────────────────────────────────────
def test_the_constant_is_five_points():
    assert C.SMART_MONEY_BIG_MOVE_PP == 5.0


def test_a_big_exit_reads_distribution_whatever_the_volume_or_price():
    """Heavy volume (5x the 20-day average), a rising price and DII buying: the old rule read Strong Accumulation.
    A 6-point FII exit in the same quarter is distribution, not interest."""
    heavy = {**_MODERATE, "volume": 5e5, "vol_sma_20d": 1e5, "change_dii_lq": 1.0}
    out = _labels({**heavy, "change_fii_lq": -0.1}, {**heavy, "change_fii_lq": -6.0})
    assert out.loc[0, "smart_money_flow"] == STRONG, "the fixture no longer reaches Strong under the old tiers"
    assert out.loc[1, "smart_money_flow"] == DIST
    assert out.loc[0, "vqs_score"] == out.loc[1, "vqs_score"] == 90, "vqs_score must not change"


def test_minus_4_9_is_unchanged_and_minus_5_reads_distribution():
    out = _labels({**_MODERATE, "change_fii_lq": -4.9}, {**_MODERATE, "change_fii_lq": -5.0})
    assert out.loc[0, "smart_money_flow"] == MOD
    assert out.loc[1, "smart_money_flow"] == DIST, "the rule is <= -5, so exactly -5 counts"


def test_a_missing_change_never_triggers_it():
    out = _labels({**_MODERATE, "change_fii_lq": np.nan, "change_dii_lq": np.nan},
                  {**_MODERATE, "change_fii_lq": np.nan})
    assert out.loc[0, "smart_money_flow"] == MOD, "NaN is not <= -5"
    assert out.loc[1, "smart_money_flow"] == MOD


def test_an_fii_only_exit_and_a_dii_only_exit_each_read_distribution():
    out = _labels({**_MODERATE, "change_fii_lq": -8.0, "change_dii_lq": 2.0},
                  {**_MODERATE, "change_fii_lq": 2.0, "change_dii_lq": -8.0})
    assert list(out["smart_money_flow"]) == [DIST, DIST]


def test_an_epack_like_row_reads_distribution():
    """EPack Prefab, 28 Oct 2026: FII 11.1% -> 0.71% (-10.4), DII -0.3, CRS 50D -4.0."""
    out = _labels({"fii_holdings": 0.71, "change_fii_lq": -10.4, "change_dii_lq": -0.3, "crs_50d": -4.0})
    assert out.loc[0, "smart_money_flow"] == DIST


def test_big_entries_are_labelled_as_before():
    """Only exits are sized. Both groups adding 10 points on heavy volume stays Elite Accumulation."""
    out = _labels({**_MODERATE, "volume": 5e5, "vol_sma_20d": 1e5, "change_fii_lq": 10.0, "change_dii_lq": 10.0})
    assert out.loc[0, "smart_money_flow"] == ELITE


def test_only_the_five_labels_are_ever_written():
    out = _labels(_MODERATE, {**_MODERATE, "change_fii_lq": -9.0}, {"change_fii_lq": np.nan},
                  {**_MODERATE, "volume": 5e5, "vol_sma_20d": 1e5, "change_fii_lq": 3.0, "change_dii_lq": 3.0})
    assert set(out["smart_money_flow"]) <= {DIST, MOD, STRONG, ELITE, NEUTRAL}


# ── 2. the fact chip ──────────────────────────────────────────────────────────────────────────────────────────
def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def _stock(**kw) -> pd.Series:
    return pd.Series({"fii_holdings": np.nan, "change_fii_lq": np.nan, "dii_holdings": np.nan,
                      "change_dii_lq": np.nan, **kw})


def test_the_chip_states_an_exit_in_red():
    html = ts.holder_moves_html(_stock(fii_holdings=0.71, change_fii_lq=-10.4))
    assert "FII −10.4 pts · 11.1% → 0.7%" in _text(html)
    assert re.search(rf'color:{re.escape(C.COLORS["red"])}[^>]*>−10\.4<', html), "only the figure, in red"
    assert "11.1% → 0.7%" in _text(html) and C.COLORS["green"] not in html
    styles = re.findall(r'style="([^"]*)"', html)
    assert styles and not any("−" in s for s in styles), "the real minus sign belongs to the figure, not the CSS"


def test_the_chip_states_an_entry_in_green():
    html = ts.holder_moves_html(_stock(dii_holdings=15.0, change_dii_lq=12.0))
    assert "DII +12.0 pts · 3.0% → 15.0%" in _text(html)
    assert re.search(rf'color:{re.escape(C.COLORS["green"])}[^>]*>\+12\.0<', html)


def test_both_groups_get_a_line_each():
    html = ts.holder_moves_html(_stock(fii_holdings=2.0, change_fii_lq=-6.0, dii_holdings=20.0, change_dii_lq=7.5))
    txt = _text(html)
    assert "FII −6.0 pts · 8.0% → 2.0%" in txt and "DII +7.5 pts · 12.5% → 20.0%" in txt


def test_a_narrow_cell_breaks_the_chip_at_the_dot_never_inside_the_move():
    """The All Data cell is ~143 px wide, so the line wraps there. It must break at the "·": the 29 Oct screenshot
    showed "11.1% →" on one line and "0.7%" on the next."""
    html = ts.holder_moves_html(_stock(fii_holdings=0.71, change_fii_lq=-10.4))
    nowrap = '<span style="white-space:nowrap;">'
    assert re.search(re.escape(nowrap) + r"FII <span[^>]*>−10\.4</span> pts ·</span>", html)
    assert nowrap + "11.1% → 0.7%" in html


def test_a_move_below_the_threshold_shows_no_chip():
    assert ts.holder_moves_html(_stock(fii_holdings=3.0, change_fii_lq=-4.9)) == ""
    assert ts.holder_moves_html(_stock(dii_holdings=9.0, change_dii_lq=4.9)) == ""


@pytest.mark.parametrize("now,chg", [(np.nan, -10.4), (0.71, np.nan), (None, -10.4), (0.71, None),
                                     ("", -10.4), ("abc", -10.4), (math.inf, -10.4), (0.71, -math.inf)])
def test_a_missing_or_unreadable_value_shows_no_chip(now, chg):
    assert ts.holder_moves_html(_stock(fii_holdings=now, change_fii_lq=chg)) == ""


@pytest.mark.parametrize("now,chg", [(2.0, 5.0), (98.0, -5.0), (0.0, 6.0), (100.0, -6.0)])
def test_an_impossible_previous_share_shows_no_chip(now, chg):
    """before = now - change must be a real share: 2.0 - 5.0 = -3.0 and 98.0 + 5.0 = 103.0 are not."""
    before = now - chg
    html = ts.holder_moves_html(_stock(fii_holdings=now, change_fii_lq=chg))
    assert (html == "") == (not 0.0 <= before <= 100.0)


def test_the_chip_never_prints_nan_or_an_action_word():
    values = [np.nan, None, -12.0, -5.0, -4.99, 0.0, 5.0, 30.0, math.inf]
    for now in (np.nan, 0.5, 40.0, 99.5):
        for chg in values:
            html = ts.holder_moves_html(_stock(fii_holdings=now, change_fii_lq=chg, dii_holdings=now,
                                              change_dii_lq=chg))
            assert "nan" not in html.lower() and "inf" not in _text(html).lower()
    tip = ts.HOLDER_MOVE_TIP.lower()
    assert not re.search(r"\b(buy|sell|avoid|hold|accumulate|watch)\b", tip), "no action words in the tooltip"
    assert "1–4 months" in ts.HOLDER_MOVE_TIP and "quarter" in tip


@pytest.mark.slow
def test_the_chip_over_every_row_of_a_live_shaped_frame():
    """The local sheet, every row: no error, never 'nan', and the chip fires somewhere (it is not dead)."""
    from core.data_engine import fetch_and_clean_data
    with contextlib.redirect_stdout(io.StringIO()):
        df = fetch_and_clean_data("local")
    fired = 0
    for i in range(len(df)):
        html = ts.holder_moves_html(df.iloc[i])
        assert "nan" not in html.lower()
        fired += bool(html)
    assert fired > 0, "no stock in the local sheet shows a chip -- check the inputs before trusting it"


# ── 3. where the chip shows, and the words that explain it ────────────────────────────────────────────────────
class _StubST:
    def __init__(self):
        self.out = []

    def markdown(self, x=None, **k):
        self.out.append(str(x))

    def __getattr__(self, name):
        return lambda *a, **k: None


def _render(fn, stock, *args):
    stub, real = _StubST(), ts.st
    ts.st = stub
    try:
        fn(stock, *args)
    finally:
        ts.st = real
    return " ".join(stub.out)


def _smart_money_cell(html: str) -> str:
    start = html.index('<div class="ts-raw-lbl">Smart Money')
    end = html.find(ts._CELL_OPEN, start)
    return html[start:end if end >= 0 else len(html)]


def test_the_smart_money_cell_carries_the_chip_and_an_ordinary_one_does_not():
    big = _stock(smart_money_flow=DIST, fii_holdings=0.71, change_fii_lq=-10.4, dii_holdings=3.0,
                 change_dii_lq=-0.3)
    cell = _smart_money_cell(_render(ts.render_raw_signals, big))
    assert "FII −10.4 pts · 11.1% → 0.7%" in _text(cell)
    plain = _stock(smart_money_flow=NEUTRAL, fii_holdings=4.0, change_fii_lq=-0.2, dii_holdings=3.0,
                   change_dii_lq=0.1)
    assert " pts · " not in _text(_smart_money_cell(_render(ts.render_raw_signals, plain)))


def test_the_ownership_card_states_the_move_under_its_holdings_row():
    row = {"promoter_holdings": 60.0, "pledged_percentage": 0.0, "change_promoter_lq": 0.0,
           "fii_holdings": 0.71, "dii_holdings": 3.0, "change_fii_lq": -10.4, "change_dii_lq": -0.3,
           "smart_money_flow": DIST}
    txt = _text(_render(ts.render_financial_insights, pd.Series(row)))
    i = txt.index("FII + DII Holdings")
    assert "FII −10.4 pts · 11.1% → 0.7%" in txt[i:i + 300]
    calm = {**row, "fii_holdings": 4.0, "change_fii_lq": -0.2, "smart_money_flow": NEUTRAL}
    assert " pts · " not in _text(_render(ts.render_financial_insights, pd.Series(calm)))


def test_the_glossary_states_the_five_point_rule():
    from ui.ui_components import _RAW_GLOSSARY
    text = _RAW_GLOSSARY["Smart Money"]
    assert f"{C.SMART_MONEY_BIG_MOVE_PP:g} points" in text and "Distribution" in text


def test_the_reference_entry_states_the_same_rule():
    """ui_reference_data is pure data (no imports), so the number is written there by hand and pinned here."""
    from ui.ui_reference_data import CONCEPT_REFERENCE
    entries = [d for sec in CONCEPT_REFERENCE.values() for t, d in sec if t == DIST]
    assert entries and all(f"{C.SMART_MONEY_BIG_MOVE_PP:g} points" in d for d in entries)
