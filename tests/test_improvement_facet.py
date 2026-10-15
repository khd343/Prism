"""Contract: the IMPROVEMENT facet of quality_score — is the business getting BETTER, over the LAST YEAR?

WHY IT EXISTS (2026-10-03). quality_score was built almost entirely from LEVELS and long histories
(10-year ROCE/ROE medians, 5-year margin medians, 5/10-year growth). A study of the next three years'
winners said the level carried almost nothing and the recent CHANGE carried the signal. Measured on a
one-off StockScans export (Returns 1Y/3Y CAGR, verified against the snapshot and known prices),
traits known at the START of each window:

    trait at the start               3Y Oct23->26   1Y Oct25->26 (falling market)
    profit growth, 2y before            +0.173          +0.186
    margin widening, 2y before          +0.138          +0.190
    margin LEVEL at start               +0.015          +0.129
    P/E at start                        -0.003            -

It held in small, mid AND large companies, and was never negative in any cyclicality tier (deep
cyclicals ~0 over 3 years — the Lynch peak-margin trap was looked for and not found). Whole-market
ranking was kept on measurement: within-sector/industry ranking lost signal without fixing deep cyclicals.

THE DESIGN: LAST-YEAR CHANGES, EQUAL THIRDS (revised the same day it first shipped).
The first version (commit 3671312) used profit growth over TWO years + the 1-year margin change. A
follow-up asked which of OPM / GPM / NPM / PAT / Revenue / EBITDA the facet needs, and the answer was a
single rule — RECENCY: for every measure, the last year's change beat the 2-year change in both long
windows (and a 2-year margin span beat a 4-year one, +0.190 vs +0.120). Head-to-head, rank-IC:

    design                                        3Y      1Y (falling)   short 66d   short 28d
    shipped: profit 2y + margin                +0.169      +0.205       +0.065      +0.110
    THIS:    profit 1y + revenue 1y + margin 1y +0.223      +0.271       +0.093      +0.147

Inside quality at 20%, in EXACT engine form: default mode +0.146/+0.101 -> +0.148/+0.104, Fundamental
+0.121/+0.015 -> +0.127/+0.022. Quintiles rise as a clean staircase (3Y: -0.4, +1.9, +3.8, +9.3, +16.8 %/yr).
(The long windows could only test the 1-year MARGIN leg via NPM — no OPM 2 or 4 years back exists; OPM
is used live because NPM is redundant here, see below.)

WHAT IS DELIBERATELY NOT IN IT:
  * NPM change — NPM = PAT / revenue, so with profit growth AND revenue growth both in, the NPM change is
    already determined by them (NPM_now / NPM_1yb = (1+profit growth) / (1+revenue growth)).
  * EBITDA growth — EBITDA ~ revenue x OPM, so its growth is revenue growth + margin change, both in
    (overlap +0.75); and only ebitda / ebitda_1yb exist, so it cannot be tested on a long window.
  * GPM change — no full-year gross margin exists at all (quarterly only, and only since 2026-09-18).
  * ROCE change — the vendor half-updated `roce` to the new fiscal year (changed on 61% of rows June ->
    September, rolled into roce_1yb on 1.3%) while roce_1yb / roce_2yb stayed frozen, so any ROCE delta
    today compares mismatched years (CLAUDE.md §5 cross-year basis rule). Rule: add it after the balance
    sheet rolls ONLY if it adds >= +0.01 on top of this facet (today +0.003 / +0.001).
  * A turnaround bonus — a loss year has no growth RATE (forward edge measured -0.2 / +0.2pp), so it is
    no evidence, not a reward.

20% is a prior, not a fit: 30% scored slightly higher and choosing it would be fitting two windows.
The six level facets are scaled by 0.8 each, so levels stay in — margin LEVEL still helped in the
falling year (+0.129).
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config as C
from core import scoring_engine as SE
from core.data_engine import compute_derived_signals
from tests.test_data_quality_fixes import _frame

_ROOT = Path(__file__).resolve().parent.parent
_INPUTS = {"pat", "pat_1yb", "revenue", "revenue_1yb", "opm", "opm_1yb"}

# the six level facets as they stood BEFORE improvement joined (2026-10-03)
_PRE = {"moat": 0.22, "growth": 0.22, "cash": 0.20, "margin": 0.13, "balance_sheet": 0.13, "valuation": 0.10}


# ── 1. Weights ──────────────────────────────────────────────────────────────────────────────
def test_improvement_is_one_fifth_of_quality_and_levels_are_scaled_not_dropped():
    W = C.QUALITY_WEIGHTS
    assert W["improvement"] == pytest.approx(0.20)
    for k, v in _PRE.items():
        assert W[k] == pytest.approx(v * 0.80), f"{k}: expected {v} x 0.8, got {W[k]}"
    assert set(W) == set(_PRE) | {"improvement"}
    assert sum(W.values()) == pytest.approx(1.0, abs=1e-9)


# ── 2. Inputs: the six last-year P&L columns, nothing else ────────────────────────────────────
def _fn_strings(name):
    src = (_ROOT / "core" / "scoring_engine.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name)
    return {c.value for c in ast.walk(fn) if isinstance(c, ast.Constant) and isinstance(c.value, str)}


def test_improvement_reads_exactly_the_six_last_year_pnl_columns():
    # The reads live in _improvement_legs since 2026-10-15 (shared with the 🚀 recipe's evidence check); the
    # score itself reads no column of its own, so these six are still everything the facet sees.
    cols = {s for s in _fn_strings("_improvement_legs") if s.isidentifier()}
    assert cols == _INPUTS, cols
    assert not {s for s in _fn_strings("_compute_improvement_score") if s.isidentifier()}
    # the excluded measures, named so a well-meaning addition has to defeat a test first
    assert not {c for c in cols if c.startswith(("roce", "npm", "ebitda", "gpm", "asset", "debt"))}
    assert not {c for c in cols if c.endswith(("_2yb", "_3yb", "_5yb"))}, "the facet is LAST-YEAR changes"


def _imp(**cols):
    return SE._compute_improvement_score(pd.DataFrame(cols)).tolist()


_FLAT = dict(pat=[100.0] * 3, pat_1yb=[100.0] * 3, revenue=[1000.0] * 3, revenue_1yb=[1000.0] * 3,
             opm=[12.0] * 3, opm_1yb=[12.0] * 3)


@pytest.mark.parametrize("leg,now", [("pat", [130.0, 100.0, 80.0]),
                                     ("revenue", [1300.0, 1000.0, 800.0]),
                                     ("opm", [15.0, 12.0, 9.0])])
def test_each_third_moves_the_score_in_the_right_direction(leg, now):
    s = _imp(**{**_FLAT, leg: now})
    assert s[0] > s[1] > s[2], f"{leg}: {s}"
    assert all(0.0 <= v <= 100.0 for v in s)


def test_the_three_legs_carry_equal_weight():
    # one stock leads on exactly one leg each; with equal thirds their scores tie
    s = _imp(pat=[130.0, 100.0, 100.0, 90.0], pat_1yb=[100.0] * 4,
             revenue=[1000.0, 1300.0, 1000.0, 900.0], revenue_1yb=[1000.0] * 4,
             opm=[12.0, 12.0, 15.0, 11.0], opm_1yb=[12.0] * 4)
    assert s[0] == pytest.approx(s[1]) == pytest.approx(s[2])
    assert s[3] < s[0]


def test_a_loss_or_no_sales_year_is_not_growth_evidence():
    # profit: loss->profit, profit->loss and loss in BOTH years (-20 vs -40 is a POSITIVE ratio — the
    # row that catches a missing profitability guard; sign-mixed rows go NaN by accident).
    # revenue: a zero base. Margin evidence alone decides for every one of these rows.
    base = dict(opm=[10.0] * 5, opm_1yb=[10.0] * 5)
    a = _imp(pat=[50.0, -20.0, -20.0, 120.0, 120.0], pat_1yb=[-10.0, 30.0, -40.0, 100.0, 100.0],
             revenue=[500.0, 500.0, 500.0, 600.0, 600.0], revenue_1yb=[500.0, 500.0, 500.0, 0.0, 500.0], **base)
    b = _imp(pat=[np.nan, np.nan, np.nan, 120.0, 120.0], pat_1yb=[np.nan, np.nan, np.nan, 100.0, 100.0],
             revenue=[500.0, 500.0, 500.0, np.nan, 600.0], revenue_1yb=[500.0, 500.0, 500.0, np.nan, 500.0], **base)
    assert a == b


def test_no_evidence_is_neutral_not_a_verdict():
    s = _imp(pat=[np.nan, 120.0], pat_1yb=[np.nan, 100.0], revenue=[np.nan, 1100.0],
             revenue_1yb=[np.nan, 1000.0], opm=[np.nan, 12.0], opm_1yb=[np.nan, 10.0])
    assert s[0] == 50.0


# ── 3. Wiring: the facet really is 20% of quality_score ───────────────────────────────────────
def test_quality_score_carries_exactly_one_fifth_of_improvement(monkeypatch):
    n = 6
    df = compute_derived_signals(_frame(
        n, pat=[90.0, 100.0, 110.0, 120.0, 130.0, 140.0], pat_1yb=[100.0] * n,
        revenue=[950.0, 1000.0, 1050.0, 1100.0, 1150.0, 1200.0], revenue_1yb=[1000.0] * n,
        opm=[10.0, 11.0, 12.0, 13.0, 14.0, 15.0], opm_1yb=[12.0] * n))
    on = SE.compute_quality_score(df)
    monkeypatch.setitem(C.QUALITY_WEIGHTS, "improvement", 0.0)
    off = SE.compute_quality_score(df)
    assert on["improvement_score"].nunique() > 1          # the inputs discriminate
    assert np.allclose(on["quality_score"] - off["quality_score"], 0.20 * on["improvement_score"])


# ── 4. Every vintage can compute it (Movers re-scores archived copies with this engine) ───────
def test_every_snapshot_carries_the_inputs():
    snaps = sorted((_ROOT / "Other Resources" / "snapshots").glob("prism_snapshot_*.csv"))
    if not snaps:
        pytest.skip("snapshots are local-only")
    for p in snaps:
        head = set(pd.read_csv(p, nrows=0).columns)
        assert _INPUTS <= head, (p.name, sorted(_INPUTS - head))


# ── 5. Surfaces: the facet is visible wherever the facets are shown ───────────────────────────
def test_config_card_counts_its_layers_from_the_weights():
    src = (_ROOT / "app.py").read_text(encoding="utf-8")
    assert "Quality Sub-Weights · 6 Layers" not in src
    assert "len(QUALITY_WEIGHTS)" in src


def test_improvement_appears_in_the_facet_rows_radar_and_quality_view():
    app = (_ROOT / "app.py").read_text(encoding="utf-8")
    comp = (_ROOT / "ui" / "ui_components.py").read_text(encoding="utf-8")
    assert '_qfrow("🚀 Improvement", "improvement_score")' in app
    i = app.index('"📊 Quality":')
    assert "improvement_score" in app[i:app.index("]", i)]
    radar = comp[comp.index("def render_radar_chart"):]
    radar = radar[:radar.index("fig = go.Figure()")]
    assert "improvement_score" in radar


def test_the_glossary_describes_the_shipped_design():
    from ui.ui_components import _RAW_GLOSSARY
    text = _RAW_GLOSSARY["Improvement Score"].lower()
    assert "revenue" in text and "last year" in text and "2 years" not in text, text
