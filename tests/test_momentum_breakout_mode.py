"""
test_momentum_breakout_mode.py
==============================
Contract: the "MomentumBreakout" Analysis Mode — momentum and breakout in HALVES, quality 0% in
the blend — is THE DEFAULT since 2026-10-15 (selectable since 2026-10-10).

THE EVIDENCE, measured in exactly the engine's form (governance 15%, framework boosts, the
governance shield and the forensic multiplier all kept) on the four forward windows PRISM holds.
Rank-IC on all stocks / on stocks trading >= Rs 1 Cr a day at the window's start:

    window                         typical stock   Breakout 33/33/33     Momentum + Breakout
    06-17 -> 08-22 (66 days)            +0.2%       +0.147 / +0.132       +0.143 / +0.137
    08-22 -> 09-19 (28 days)            -2.0%       +0.091 / +0.072       +0.135 / +0.134
    06-29 -> 10-09 (102 days, archive)  -6.2%       +0.110 / +0.073       +0.133 / +0.111
    06-17 -> 10-03 (108 days)           -3.6%       +0.128 / +0.126       +0.157 / +0.172

It ties the default once and beats it three times; on the buyable stocks it wins all four.
Quality + Breakout (no momentum), measured the same day, trailed the default in three of the
four and is NOT a mode.

WHY IT WAS NOT THE DEFAULT AT FIRST (2026-10-10): the two long windows overlap almost entirely
(June -> October), three of the four were falling markets — where price strength protects best —
and the longest is ~3.5 months while PRISM's aim is years. December's 6-month window was to decide
(docs/december-2026-plan.md rule 26, written before these results).

2026-10-15: THE USER MADE IT THE DEFAULT, ahead of December. Those caveats still stand, so
December's window now REVIEWS it: if equal thirds beats it there on BOTH columns, quality earned
its third back. tools/validate.py rebuilds every mode on every snapshot from the snapshot's own
`scored_mode`, so that comparison is the same whichever mode is the default.

WHAT "QUALITY 0%" DOES NOT MEAN: the forensic penalty, the governance shield, the hard gates and
the framework boosts all still apply — the mode removes quality from the BLEND only, which is
exactly the form that was measured. The description says so.

Run with: pytest tests/test_momentum_breakout_mode.py -v
"""

import contextlib
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

import config as C
from scoring_engine import compute_composite_score

_ROOT = Path(__file__).resolve().parent.parent
_MODE = "MomentumBreakout"


# ── 1. the mode exists, is what it says, and is the default ─────────────────────────────
def test_the_mode_is_momentum_and_breakout_in_halves_with_no_quality_leg():
    m = C.ANALYSIS_MODES[_MODE]
    assert (m["fundamental_w"], m["momentum_w"], m["breakout_w"]) == (0.0, 0.5, 0.5)
    assert m["label"].strip() and m["description"].strip()
    assert "allowed_profiles" not in m


def test_it_is_the_default_and_its_caption_says_so():
    """Promoted 2026-10-15 by the user, ahead of December's review (rule 26), which now reviews it."""
    assert C.DEFAULT_ANALYSIS_MODE == _MODE, "the user made Momentum + Breakout the default on 2026-10-15"
    assert "default" in C.ANALYSIS_MODES[_MODE]["description"].lower(), (
        "the caption under the selector must say the mode is the default it is")


def test_the_description_says_the_penalties_still_apply():
    """'Quality 0%' would read as 'fundamentals ignored'. They are not: the forensic penalty and the
    hard gates still act on the score, and that is the form the measurement used."""
    d = C.ANALYSIS_MODES[_MODE]["description"].lower()
    assert "quality" in d and "penalt" in d


# ── 2. the engine: quality does not move the blend, the legs are exact halves ───────────
def _frame(n=6):
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "quality_score":    rng.uniform(10, 90, n),
        "momentum_score":   rng.uniform(10, 90, n),
        "breakout_score":   rng.uniform(0, 100, n),
        "governance_bonus": rng.uniform(0, 60, n),
        "pb_ratio":         rng.uniform(0.5, 8, n),
        "roe":              rng.uniform(5, 30, n),
        "rev_gr_5y":        rng.uniform(0, 30, n),
        "roce_med_5y":      rng.uniform(5, 30, n),
        "market_cap":       rng.uniform(500, 50000, n),
    })


def _score(f):
    m = C.ANALYSIS_MODES[_MODE]
    return compute_composite_score(f, m["fundamental_w"], m["momentum_w"], breakout_w=m["breakout_w"])


def test_the_blend_is_exact_halves_of_momentum_and_breakout():
    gov = C.COMPOSITE_WEIGHTS["governance"]
    f = _frame()
    want = ((f["momentum_score"] + f["breakout_score"]) / 2) * (1 - gov) + f["governance_bonus"] * gov
    assert np.allclose(_score(f)["composite_score"], want.clip(0, 100))


def test_quality_does_not_move_the_blend():
    f, g = _frame(), _frame()
    g["quality_score"] = 100.0 - g["quality_score"]
    assert np.allclose(_score(f)["composite_score"], _score(g)["composite_score"])


# ── 3. live: a real re-rank against equal thirds, nothing upstream moves ───────────────
@pytest.fixture(scope="module")
def live_pair():
    """(equal thirds, this mode) on the local data. Thirds is NAMED, not taken as the default: since
    2026-10-15 the default IS this mode, and a pair of one mode would compare nothing."""
    from core import fetch_and_clean_data, run_scoring_pipeline
    with contextlib.redirect_stdout(io.StringIO()):
        clean = fetch_and_clean_data("local")
        base = run_scoring_pipeline(clean.copy(), analysis_mode="Breakout")
        mode = run_scoring_pipeline(clean.copy(), analysis_mode=_MODE)
    assert mode.attrs.get("analysis_mode") == _MODE, "the engine fell back to another mode"
    b = base.set_index("company_id")
    m = mode.set_index("company_id").reindex(b.index)
    return b, m


def test_the_mode_scores_every_stock_and_genuinely_reranks(live_pair):
    """Measured 2026-10-10 on the local 09-18 data: rank corr with equal thirds (then the default)
    0.941, top-50 overlap 31/50, top-tier counts 4/20/172 -> 2/47/255."""
    base, mode = live_pair
    c_b = pd.to_numeric(base["composite_score"], errors="coerce")
    c_m = pd.to_numeric(mode["composite_score"], errors="coerce")
    assert c_m.isna().sum() == 0 and c_m.between(0, 100).all()
    rho = c_b.rank().corr(c_m.rank())
    assert 0.80 < rho < 0.99, f"rank-corr default vs Momentum + Breakout {rho:+.3f}"
    top = len(set(c_b.nlargest(50).index) & set(c_m.nlargest(50).index))
    assert 15 <= top <= 45, f"top-50 overlap {top}/50 (31/50 when it shipped)"


def test_the_mode_changes_the_blend_and_nothing_upstream(live_pair):
    base, mode = live_pair
    for c in ("quality_score", "momentum_score", "breakout_score", "governance_bonus",
              "forensic_multiplier", "gate_pass", "red_flag_count"):
        a, b = pd.to_numeric(base[c], errors="coerce"), pd.to_numeric(mode[c], errors="coerce")
        assert np.allclose(a, b, equal_nan=True), f"{c} differs between modes — the mode leaked upstream"


# ── 4. surfaces: every place that lists the modes lists this one ───────────────────────
def test_the_config_tab_names_the_mode():
    src = (_ROOT / "app.py").read_text(encoding="utf-8")
    i = src.index("_mode_icon = ")
    icons = src[i:src.index("}", i)]                  # the icon dict alone, not the names below it
    assert f'"{_MODE}": "' in icons, "no icon for the mode in _mode_icon"
    j = src.index("_mode_name = ")
    assert f'"{_MODE}": "Momentum + Breakout"' in src[j:src.index("}", j)], (
        "the weights row would print the raw key 'MomentumBreakout'")
    i = src.index('"Analysis Mode", options=')
    assert "Momentum + Breakout 0/50/50" in src[i:i + 900], "the selector help does not list the mode"
