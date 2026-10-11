"""
test_validate_buyable.py — December judges every signal where you can buy, too (2026-10-10).

Measured 2026-10-09: the Breakout composite's edge among stocks trading ₹1 Cr a day or more was about
half its edge among the rest (+0.063 to +0.132 vs +0.131 to +0.176). An edge in stocks you cannot buy in
size is worth far less than its headline, so tools/validate.py now prints, beside every signal, the same
rank-IC on the stocks that were buyable AT THE START of the window (the engine's own traded_value_cr,
the ₹1 Cr step the app's Liquidity controls use) — fixed now, before December's data exists.

Momentum + Breakout, first registered here as a price-only candidate, became the FIFTH Analysis Mode
the same day (selectable, not the default; tests/test_momentum_breakout_mode.py), so the harness now
measures it as a mode, rebuilt in exactly the engine's form like every mode (the 2026-09-19 lesson: a
bare blend flatters every candidate). The momentum-only and breakout-only candidates were dropped: the
user asked for the two blends only, and Quality + Breakout trailed the default.

Run with: pytest tests/test_validate_buyable.py -v
"""
import contextlib
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if not (_ROOT / "tools" / "validate.py").exists():
    pytest.skip("tools/ is gitignored; the harness is local-only", allow_module_level=True)
sys.path.insert(0, str(_ROOT / "tools"))
sys.path.insert(0, str(_ROOT))
import validate as V  # noqa: E402

from config import COMPOSITE_WEIGHTS  # noqa: E402
from core.data_engine import traded_value_cr  # noqa: E402
from ui.ui_liquidity import LIQUIDITY_STEPS  # noqa: E402


def test_momentum_breakout_is_measured_as_a_mode_for_december():
    for name in ("composite_mode_momentumbreakout", "composite_mode_breakout", "breakout_score"):
        assert name in V._DEFAULT_SIGNALS, f"{name} is not measured on every window"
    assert not hasattr(V, "_PRICE_ONLY"), "the side list of candidates came back beside the modes"
    assert not any(s.startswith("composite_cand_") for s in V._DEFAULT_SIGNALS)


def _hybrid_snapshot(n=6):
    rng = np.random.default_rng(3)
    Q, M, B, G = (rng.uniform(10, 90, n) for _ in range(4))
    boost, F, GM = rng.uniform(0, 5, n), rng.uniform(0.6, 1.0, n), rng.uniform(0.8, 1.0, n)
    gov = COMPOSITE_WEIGHTS["governance"]
    stored = (((0.7 * Q + 0.3 * M) * (1 - gov) + G * gov) + boost) * F * GM
    return pd.DataFrame({"quality_score": Q, "momentum_score": M, "breakout_score": B, "governance_bonus": G,
                         "forensic_multiplier": F, "governance_risk_multiplier": GM,
                         "composite_score": stored}), (M, B, G, boost, F, GM, gov)


def test_momentum_breakout_is_rebuilt_in_the_engines_exact_form():
    """Governance share, framework boosts and BOTH multiplicative tails kept — only the blend changes."""
    snap, (M, B, G, boost, F, GM, gov) = _hybrid_snapshot()
    out = V._add_candidates(snap)
    want = np.clip(((0.5 * M + 0.5 * B) * (1 - gov) + G * gov + boost) * F * GM, 0, 100)
    assert np.allclose(out["composite_mode_momentumbreakout"], want, atol=1e-9)


def test_buyable_means_the_engines_number_at_the_start_and_unknown_is_not_buyable():
    assert V._BUYABLE_FLOOR_CR in LIQUIDITY_STEPS, "the floor must be a step the app's Liquidity controls offer"
    early = pd.DataFrame({"vol_sma_20d": [1e5, 1e3, np.nan, 1e4], "close_price": [1000.0] * 4})
    assert V._buyable(early).tolist() == (traded_value_cr(early) >= V._BUYABLE_FLOOR_CR).tolist() == [
        True, False, False, True]


def _window(n=400):
    """A signal that predicts returns ONLY among the illiquid half: all-stocks IC is strong, buyable ~0."""
    rng = np.random.default_rng(4)
    liquid = np.arange(n) % 2 == 0
    sig = rng.uniform(0, 100, n)
    ret = np.where(liquid, rng.normal(0, 0.1, n), sig / 100 - 0.5 + rng.normal(0, 0.02, n))
    early = pd.DataFrame({"company_id": [f"S{i}" for i in range(n)], "close_price": 100.0,
                          "vol_sma_20d": np.where(liquid, 1e6, 1e3),          # ₹10 Cr vs ₹0.01 Cr a day
                          "momentum_score": sig})
    late = early[["company_id"]].assign(close_price=100.0 * (1 + ret))
    return early, late


def test_every_window_prints_the_buyable_reading_beside_the_headline():
    early, late = _window()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        V._evaluate_pair(early, late, "2026-06-01", "2026-07-15", ["momentum_score"])
    text = out.getvalue()
    assert "IC ≥₹1Cr" in text, "the table has no buyable column"
    row = next(l for l in text.splitlines() if l.strip().startswith("momentum_score"))
    nums = re.findall(r"[+-]\d\.\d{3}", row)
    ic_all, ic_buyable = float(nums[0]), float(nums[1])
    assert ic_all > 0.4 and abs(ic_buyable) < 0.2, (
        f"the buyable column must be the same test on the buyable stocks only: {row}")
    assert re.search(r"buyable = .*₹1 Cr", text), "the table must say what 'buyable' means"
