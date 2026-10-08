"""
test_regime_detection.py
========================
Contract tests for the market-regime subsystem: core/scoring_engine.detect_market_regime (breadth
consensus) and the regime weight factory config.get_adaptive_weights.

Phase-1 audit found this subsystem entirely UNPINNED and carrying a semantic-truth bug: breadth was
measured with `(s > 0).mean()` over ALL rows, so NaN inputs (~10% of CRS) were silently counted as
"not bull", biasing every vote bearish. These tests pin the 2-of-3 consensus, the NaN-EXCLUDED
breadth (the fix), abstention on missing data, and the weight-cascade invariants (sum==1, no
negative weight) that nothing previously guarded.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from core.scoring_engine import detect_market_regime
from config import MASTER_PROFILES, REGIME_ADJUSTMENTS, get_adaptive_weights


def _frac(p, n=100):
    """n-length series: exactly round(p*n) positive (+1.0), the rest negative (-1.0)."""
    k = round(p * n)
    return pd.Series([1.0] * k + [-1.0] * (n - k))


def _px(p, n=100):
    """A (close, sma) pair giving a close>sma fraction of p."""
    k = round(p * n)
    return pd.Series([110.0] * k + [90.0] * (n - k)), pd.Series([100.0] * n)


def _frame(crs50=None, crs26=None, close=None, sma200=None):
    """Build a regime input frame from only the provided signals (omitted = absent column).
    All provided series must share one length (a single market snapshot)."""
    data = {}
    if crs50 is not None:
        data["crs_50d"] = pd.Series(crs50).reset_index(drop=True)
    if crs26 is not None:
        data["crs_26w"] = pd.Series(crs26).reset_index(drop=True)
    if close is not None:
        data["close_price"] = pd.Series(close).reset_index(drop=True)
    if sma200 is not None:
        data["sma_200d"] = pd.Series(sma200).reset_index(drop=True)
    return pd.DataFrame(data)


# ── consensus truth table (full data, no NaN) ────────────────────────────────

def test_all_three_bullish_returns_bull():
    c, m = _px(0.70)
    assert detect_market_regime(_frame(_frac(0.70), _frac(0.70), c, m)) == "BULL"


def test_all_three_bearish_returns_bear():
    c, m = _px(0.30)
    assert detect_market_regime(_frame(_frac(0.30), _frac(0.30), c, m)) == "BEAR"


def test_two_bull_one_bear_returns_bull():
    c, m = _px(0.30)   # 200D bearish, but crs_50d + crs_26w bullish -> 2-of-3 -> BULL
    assert detect_market_regime(_frame(_frac(0.70), _frac(0.70), c, m)) == "BULL"


def test_one_bull_one_bear_one_neutral_returns_sideways():
    c, m = _px(0.30)   # 200D bear ; crs50 bull ; crs26 neutral -> no 2-of-3 -> SIDEWAYS
    assert detect_market_regime(_frame(_frac(0.70), _frac(0.50), c, m)) == "SIDEWAYS"


def test_single_signal_cannot_reach_consensus():
    assert detect_market_regime(_frame(crs50=_frac(0.70))) == "SIDEWAYS"


# ── NaN-bias: breadth must be measured over non-NaN rows ONLY ─────────────────

def test_nan_bias_excluded_from_breadth():
    """RED on the old code: 6/10 and 5/10 INCLUDING NaN read as neutral -> SIDEWAYS. EXCLUDING NaN
    the valid stocks are 100% bullish -> 2 bull votes -> BULL. NaN must never count as 'not bull'."""
    crs50 = [1.0] * 6 + [np.nan] * 4   # excl-NaN: 6/6 = 1.00 bull ; incl-NaN: 6/10 = 0.60 neutral
    crs26 = [1.0] * 5 + [np.nan] * 5   # excl-NaN: 5/5 = 1.00 bull ; incl-NaN: 5/10 = 0.50 neutral
    assert detect_market_regime(_frame(crs50, crs26)) == "BULL"


def test_all_nan_inputs_abstain_to_sideways():
    nan10 = [np.nan] * 10
    assert detect_market_regime(_frame(nan10, nan10, nan10, [100.0] * 10)) == "SIDEWAYS"


# ── 200D vote computed from RAW close/SMA, NaN-excluded (not the biased column) ─

def test_200d_vote_from_raw_close_sma_nan_excluded():
    close = pd.Series([110.0] * 60 + [np.nan] * 40)   # 60 valid, all above SMA -> bull
    sma = pd.Series([100.0] * 60 + [np.nan] * 40)
    # crs_50d bullish too -> 2 bull votes -> BULL
    assert detect_market_regime(_frame(crs50=_frac(0.70), close=close, sma200=sma)) == "BULL"


def test_200d_all_nan_abstains():
    nan100 = [np.nan] * 100
    # only crs_50d votes (bull); 200D abstains, crs_26w absent -> 1 vote -> SIDEWAYS
    assert detect_market_regime(_frame(crs50=_frac(0.70), close=nan100, sma200=nan100)) == "SIDEWAYS"


# ── weight factory invariants (previously unguarded) ─────────────────────────

# THE PROFILE AXIS COLLAPSED 2026-09-20 (§6 stale-test updating, not deletion). These two cases
# were parametrized over profile x regime = 8 x 3; seven of the eight profiles were removed with
# the selector that chose between them, so the surviving axis is the REGIME. What replaces the
# lost breadth is depth on that axis: the GATE cascade below was never asserted anywhere —
# get_adaptive_weights layers roce/growth/peg deltas and clamps them, and nothing recomputed the
# arithmetic independently, so a broken cascade showed up only as a silent shift in qglp_pass.
@pytest.mark.parametrize("regime", ["BULL", "SIDEWAYS", "BEAR"])
def test_adaptive_weights_sum_to_one_and_nonnegative(regime):
    w = get_adaptive_weights("Balanced", regime)
    four = [w["quality_w"], w["growth_w"], w["longevity_w"], w["price_w"]]
    assert abs(sum(four) - 1.0) < 1e-9, f"{regime} weights sum to {sum(four)}"
    assert min(four) >= 0.0, f"{regime} has a negative weight: {four}"


@pytest.mark.parametrize("gate,floor", [("roce_gate", 5.0), ("growth_gate", 0.0), ("peg_gate", 0.5)])
@pytest.mark.parametrize("regime", ["BULL", "SIDEWAYS", "BEAR"])
def test_the_gate_cascade_is_book_plus_delta_clamped(regime, gate, floor):
    """Recompute the cascade from its two sources and demand the factory agree.

    This is the leg qglp_pass actually hangs on (325 SIDEWAYS / 309 BULL / 249 BEAR, measured
    2026-08-30). Asserting only "the number is sane" would pass on a dropped delta; recomputing
    it independently from MASTER_PROFILES + REGIME_ADJUSTMENTS will not.
    """
    base = MASTER_PROFILES["Balanced"][gate]
    delta = REGIME_ADJUSTMENTS[regime][f"{gate}_delta"]
    expected = max(floor, base + delta)
    got = get_adaptive_weights("Balanced", regime)[gate]
    assert got == pytest.approx(expected), (
        f"{regime}/{gate}: factory says {got}, book {base} + delta {delta} clamped at {floor} "
        f"says {expected} — the regime cascade has drifted from its own inputs"
    )


def test_unknown_regime_falls_back_to_sideways():
    for profile in sorted(MASTER_PROFILES.keys()):
        w = get_adaptive_weights(profile, "NOT_A_REGIME")
        four = [w["quality_w"], w["growth_w"], w["longevity_w"], w["price_w"]]
        assert abs(sum(four) - 1.0) < 1e-9 and min(four) >= 0.0
        sw = get_adaptive_weights(profile, "SIDEWAYS")
        assert (w["quality_w"], w["growth_w"], w["longevity_w"], w["price_w"]) == \
               (sw["quality_w"], sw["growth_w"], sw["longevity_w"], sw["price_w"])


# ── The load-bearing argument (2026-08-30 audit) ─────────────────────────────────────────────
# The ENTIRE regime -> weights/gates modulation hangs on run_full_scoring passing the ADAPTIVE
# dict into compute_qglp_score. A one-word revert to the raw profile would silently disconnect
# regime from QGLP (weights AND the roce/growth/peg gates) while every other test stayed green —
# measured live 2026-08-30: qglp_pass moves 325 (SIDEWAYS) -> 309 (BULL) -> 249 (BEAR) only
# because of this argument. One structural pin + one behavioral pin, so neither a syntax revert
# nor a flow-preserving refactor can break it invisibly.

def test_qglp_receives_the_adaptive_dict_not_the_raw_profile():
    import io as _io, os
    src = _io.open(os.path.join(os.path.dirname(__file__), "..", "core", "scoring_engine.py"),
                   encoding="utf-8").read()
    assert "compute_qglp_score(df, profile=adaptive)" in src, (
        "run_full_scoring no longer passes the regime-adjusted ADAPTIVE dict into QGLP — "
        "regime would silently stop moving QGLP weights and gates"
    )


import functools


@functools.lru_cache(maxsize=2)
def _forced_regime_frames(drives: bool = False):
    """The full pipeline under forced SIDEWAYS / BEAR / BULL (patched at the module global
    run_full_scoring resolves at call time; restored in finally), with config.REGIME_DRIVES_SCORING
    as given (patched on scoring_engine, which imported it). Cached per switch position."""
    import contextlib, io as _io
    import core.scoring_engine as se
    from core import fetch_and_clean_data, run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        raw = fetch_and_clean_data("local")
    _orig, _orig_drives = se.detect_market_regime, se.REGIME_DRIVES_SCORING
    frames = {}
    try:
        se.REGIME_DRIVES_SCORING = drives
        for regime in ("SIDEWAYS", "BEAR", "BULL"):
            se.detect_market_regime = lambda d, v=regime: v
            with contextlib.redirect_stdout(_io.StringIO()):
                frames[regime] = run_scoring_pipeline(raw.copy())
    finally:
        se.detect_market_regime, se.REGIME_DRIVES_SCORING = _orig, _orig_drives
    return frames


def test_the_detected_regime_does_not_reach_the_scores_while_switched_off():
    """2026-10-08: the detector stood half a point from BULL while the Nifty 500 was in a Stage-4
    decline, and its adjustments were never tested — so scoring uses SIDEWAYS until December decides.
    A forced BULL or BEAR must leave every score exactly as SIDEWAYS leaves it."""
    from config import REGIME_DRIVES_SCORING
    assert REGIME_DRIVES_SCORING is False, "December's review decides when the regime reaches the scores"
    frames = _forced_regime_frames(False)
    base = frames["SIDEWAYS"].set_index("company_id")
    for regime in ("BEAR", "BULL"):
        other = frames[regime].set_index("company_id").reindex(base.index)
        for col in ("composite_score", "momentum_score", "rank", "qglp_pass", "conviction_tier",
                    "fw_can_slim", "gate_pass", "wealth_tier"):
            if col in base.columns:
                same = (base[col].astype(str) == other[col].astype(str)).all()
                assert same, f"forced {regime} moved {col} — the detected regime reached the scores"


def test_with_the_switch_on_bear_still_tightens_qglp_pass_on_live_data():
    """The machinery stays alive for December: switched on, BEAR raises the QGLP bars (config
    REGIME_ADJUSTMENTS), so a forced-BEAR run passes STRICTLY fewer stocks than SIDEWAYS. The numbers
    judge, not the syntax."""
    frames = _forced_regime_frames(True)
    side, bear = int(frames["SIDEWAYS"]["qglp_pass"].sum()), int(frames["BEAR"]["qglp_pass"].sum())
    assert side > 0, "qglp_pass fires on nothing — the gate died"
    assert bear < side, (
        f"forced BEAR passed {bear} vs SIDEWAYS {side} — regime no longer reaches the QGLP gates"
    )


def test_regime_dual_write_agrees():
    """detect_market_regime is stored twice (df.attrs primary + _detected_market_regime column
    fallback). The two must never disagree — a consumer picking the 'wrong' channel must get the
    same answer. The SCORING regime is stored the same way, and it is SIDEWAYS while switched off."""
    for drives in (False, True):
        for regime, frame in sorted(_forced_regime_frames(drives).items()):
            assert frame.attrs.get("detected_market_regime") == regime
            assert frame["_detected_market_regime"].iloc[0] == regime
            want = regime if drives else "SIDEWAYS"
            assert frame.attrs.get("scoring_regime") == want and frame["_scoring_regime"].iloc[0] == want
