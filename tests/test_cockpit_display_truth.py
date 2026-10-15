"""
test_cockpit_display_truth.py
=============================
Contracts for the Matrix & WCS tab's display truth — three bugs found by reading the rendered
tab for EPack Prefab Technologies on 2026-08-25, all of the same family: a card stating
something the engine never computed.

1. HARDCODED STOP DISTANCE. The stop-loss card's subtitle was the literal string
   "-7-8% Active Perimeter Shield" — a number presented as measurement but never derived. True
   for 95 of 2,117 stocks (4.5%); the real distance spans -20%..+27% (p10-p90). EPack rendered
   "-7-8%" while its stop sat 25.5% BELOW the price, with the card immediately to its right
   correctly reading "+34.2% vs stop" — one row contradicting itself.

2. FABRICATED ZEROS. `value_creation_velocity` (96.0% populated) and `expectations_gap` (68.4%)
   were read with a 0.0 default, so 669 stocks were shown a confident "+0.00%" for a number
   nobody had. A data hole is not a measurement (CLAUDE.md §5).

3. EP LABEL. "✅ EP Positive" carried a green check on 285 of its 299 rows whose economic profit
   was SHRINKING (the other 14 have no prior-year figure; zero are flat or rising) — rendered
   directly beside "EP VELOCITY ₹-12 Cr Descending ↓". Renamed "➖ EP Positive, Not Rising".
   The taxonomy was always sound: "🚀 Hockey Stick" is 0-for-533 on negative velocity.

Run with: pytest tests/test_cockpit_display_truth.py -v
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pytest

import ui.ui_tearsheet as T
from config import COLORS


class _Recorder:
    """Minimal Streamlit stand-in — captures every string the renderer emits."""

    def __init__(self):
        self.out = []

    def _cap(self, *a, **k):
        if a:
            self.out.append(str(a[0]))

    markdown = write = success = info = warning = error = caption = _cap

    def __getattr__(self, _name):
        return lambda *a, **k: None


def _render(**over):
    """Render the sizing cockpit for one synthetic stock and return everything it emitted."""
    base = {
        "close_price": 268.50, "vstop_value": 200.01, "dist_to_vstop": 34.24,
        "optimal_portfolio_weight_pct": 3.92, "rupee_capital_allocation": 39202.80,
        "value_creation_velocity": 6.27, "expectations_gap": -0.94,
        "sepa_vcp_dryup": 0, "sepa_profile_score": 1, "sepa_pass": 0,
    }
    base.update(over)
    rec, real = _Recorder(), T.st
    try:
        T.st = rec
        T.render_valuation_inversion_and_sizing_cockpit(pd.Series(base))
    finally:
        T.st = real
    return " ".join(rec.out)


# ── 1. The stop-loss distance must be COMPUTED ──────────────────────────────────────
def test_stop_distance_is_computed_not_hardcoded():
    html = _render()
    assert "-7-8" not in html, "the hardcoded '-7-8%' stop-loss subtitle is back"
    m = re.search(r"price sits ([\d.]+)% above it", html)
    assert m, "the stop card no longer states a computed distance"
    assert abs(float(m.group(1)) - 34.24) < 0.1, f"wrong distance: {m.group(1)}"


def test_stop_card_and_price_card_quote_the_SAME_number():
    """One gap, one number. The first version of this subtitle recomputed against the price
    ((vstop-close)/close) while the price card uses the engine's dist_to_vstop, so Avanti Feeds
    showed "33.9% ABOVE price" beside "-25.3% vs stop" — two numbers for one relationship.
    The stop card must mirror dist_to_vstop, never re-derive a second denominator."""
    for dist in (34.24, -25.31, 7.0):
        html = _render(close_price=861.40, vstop_value=1153.36 if dist < 0 else 700.0,
                       dist_to_vstop=dist)
        m = re.search(r"price sits ([\d.]+)% (above|below) it", html)
        assert m, f"no gap stated for dist={dist}"
        assert abs(float(m.group(1)) - abs(dist)) < 0.05, (
            f"stop card says {m.group(1)}% but the engine's dist_to_vstop is {dist}"
        )


def test_a_stop_above_the_price_is_not_called_a_stop_loss():
    """Avanti Feeds: price Rs 861.40, volatility stop Rs 1,153.36. A stop-loss is where you exit
    to CAP a loss, so it must sit below the position — one 34% above would trigger instantly.
    A volatility stop flips to the short side on a breakdown, and 876 stocks (41.4%) sit there.
    The number is real (it is the reclaim level); calling it a stop-loss is not."""
    html = _render(close_price=861.40, vstop_value=1153.36, dist_to_vstop=-25.31)
    assert "Trend Broken" in html, "a stop above the price must not keep the stop-loss label"
    assert "Stop-Loss Level" not in html, "still labelled a stop-loss while sitting above price"
    assert "no long stop to set" in html
    # and the healthy case keeps the real label
    ok = _render(close_price=268.50, vstop_value=200.01, dist_to_vstop=34.24)
    assert "Stop-Loss Level" in ok and "Trend Broken" not in ok


@pytest.mark.parametrize("over", [
    {"close_price": np.nan}, {"vstop_value": np.nan}, {"close_price": 0.0},
])
def test_stop_subtitle_states_nothing_when_inputs_are_missing(over):
    html = _render(**over)
    assert "below price" not in html and "ABOVE price" not in html


# ── 2. Missing inputs must render N/A, never a fabricated zero ──────────────────────
@pytest.mark.parametrize("col", ["value_creation_velocity", "expectations_gap"])
def test_absent_structural_metric_renders_na_not_zero(col):
    html = _render(**{col: np.nan})
    assert "N/A" in html, f"{col} missing → the card must say N/A"
    assert "+0.00%" not in html, f"{col} missing → a fabricated '+0.00%' was rendered"


def test_present_structural_metrics_render_their_values():
    html = _render(value_creation_velocity=6.27, expectations_gap=-0.94)
    assert "+6.27%" in html and "-0.94%" in html


def test_expectations_gap_polarity_matches_the_engine_definition():
    """data_engine ~L3308: positive = priced above what it can deliver; negative = safety."""
    good = _render(expectations_gap=-5.0)
    bad  = _render(expectations_gap=+5.0)
    assert "margin of safety" in good
    assert "priced above what it can deliver" in bad
    assert COLORS["green"] in good


def test_structural_metrics_render_as_cards_not_bare_text():
    """They were three unstyled st.write/st.markdown lines under a card grid."""
    html = _render()
    for label in ("Value Creation Velocity", "Expectations Gap", "Consolidation Base"):
        assert label in html
    # a card carries the cockpit's border/background chrome; bare st.write does not
    assert html.count("border-radius:10px") >= 6


# ── 4. VCV honesty states (2026-08-28: found walking the live CLIMBING★ cohort) ──────────
def _vcv_card(html):
    """The VCV card's own html: label → next card's label. The value's color style sits AFTER
    the label div, and the next card's value color sits after ITS label — so this window holds
    exactly one card's verdict colors (plus neutral chrome)."""
    i = html.index("Value Creation Velocity")
    return html[i:html.index("Expectations Gap", i)]


def test_full_payout_zero_is_gold_not_condemned():
    """Castrol India: ROCE 75.8, DPR 135 → RR clips to 0 → VCV 0.00. Eleven live stocks with a
    real above-hurdle spread (median ROCE 48.4) rendered a RED zero beside a green Moat axis —
    a full distributor is not a value destroyer."""
    seg = _vcv_card(_render(value_creation_velocity=0.0, reinvestment_rate=0.0,
                            dividend_payout_ratio=135.36))
    assert "distributes all earnings" in seg and "not retained" in seg
    assert COLORS["red"] not in seg, "a full-payout zero still wears the below-hurdle red"
    assert COLORS["gold"] in seg


def test_assumed_retention_is_named_not_presented_as_measured():
    """DPR missing (58.8% of the universe) → RR is fillna-fabricated at 1.0, so the number is
    really ROCE − CoE. The sub-line must say the assumption, never claim a measured rate."""
    seg = _vcv_card(_render(value_creation_velocity=13.34))   # fixture has no DPR key
    assert "assumed full retention — payout unreported" in seg
    assert "Reinvestment rate × capital spread" not in seg, (
        "an assumed RR is still presented as measured")


def test_measured_rr_keeps_the_measured_subline():
    seg = _vcv_card(_render(value_creation_velocity=13.05, dividend_payout_ratio=10.21,
                            reinvestment_rate=0.8979))
    assert "Reinvestment rate × capital spread" in seg
    assert "assumed full retention" not in seg and "distributes all earnings" not in seg
    assert COLORS["green"] in seg


def test_zero_is_a_third_state_not_below_hurdle_red():
    """The engine's own definition: 'Negative = compounding capital below its hurdle rate.'
    Zero is not negative — a spread exactly at the hurdle must render muted, never red."""
    seg = _vcv_card(_render(value_creation_velocity=0.0, dividend_payout_ratio=40.0,
                            reinvestment_rate=0.6))
    assert COLORS["red"] not in seg
    assert COLORS["gold"] not in seg, "an at-hurdle zero is not the full-payout state"


def test_negative_velocity_keeps_the_red():
    seg = _vcv_card(_render(value_creation_velocity=-3.2, dividend_payout_ratio=40.0,
                            reinvestment_rate=0.6))
    assert COLORS["red"] in seg, "genuine below-hurdle compounding lost its red"


# ── 5. A ruled-out position is a known zero, not missing data (2026-08-28) ──────────
def test_ruled_out_position_says_zero_shares_not_dash():
    """Sarda: price below its volatility stop → the sizing rule computes weight 0 — a KNOWN
    answer. The tile showed the em-dash, this codebase's honest-blank glyph for 'unknown',
    dressing a decision as a data hole (the VCV full-payout defect class)."""
    html = _render(close_price=531.15, vstop_value=589.54, dist_to_vstop=-9.9,
                   rupee_capital_allocation=0.0, optimal_portfolio_weight_pct=0.0)
    i = html.index("Executable at 10L Base")
    seg = html[i:i + 700]
    assert "0 shares" in seg, "a computed-zero position still renders the unknown dash"
    assert "no entry" in seg


def test_missing_allocation_keeps_the_honest_dash():
    """The reverse must also hold: when the engine truly has no allocation figure, the dash
    stays — a data hole must never be dressed as a confident zero."""
    html = _render(rupee_capital_allocation=np.nan)
    i = html.index("Executable at 10L Base")
    seg = html[i:i + 700]
    assert "0 shares" not in seg
    assert "—" in seg


def test_positive_allocation_still_prints_the_share_count():
    html = _render(rupee_capital_allocation=39202.80, close_price=268.50)
    i = html.index("Executable at 10L Base")
    seg = html[i:i + 700]
    assert "146 shares" in seg     # 39202.80 // 268.50 = 146


# ── 3. The EP power-curve label must not praise a shrinking company ─────────────────
def _live():
    import contextlib, io as _io
    from core.data_engine import (load_all_csvs, merge_datasets, coerce_numeric_columns,
                                  compute_derived_signals)
    with contextlib.redirect_stdout(_io.StringIO()):
        return compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local"))))


@pytest.fixture(scope="module")
def live():
    return _live()


def test_no_ep_label_reads_positive_while_economic_profit_shrinks(live):
    vel = live["economic_profit_velocity"]
    for label, grp in live.groupby("ep_power_curve"):
        if not label or "✅" not in str(label):
            continue
        shrinking = int((vel[grp.index] < 0).sum())
        assert shrinking == 0, (
            f"{label!r} carries a ✅ but {shrinking} of its {len(grp)} rows have SHRINKING "
            f"economic profit — the label praises what the velocity card contradicts"
        )


def test_discovery_filter_order_matches_the_engine_labels(live):
    """_EPPC_ORDER is a hand-kept list; drift silently drops an option from the filter."""
    import ast
    src = open(os.path.join(os.path.dirname(__file__), "..", "ui", "ui_discovery.py"),
               encoding="utf-8").read()
    order = next(
        (ast.literal_eval(n.value) for n in ast.walk(ast.parse(src))
         if isinstance(n, ast.Assign) and any(
             isinstance(t, ast.Name) and t.id == "_EPPC_ORDER" for t in n.targets)),
        None)
    assert order, "_EPPC_ORDER not found in ui_discovery.py"
    emitted = {str(x) for x in live["ep_power_curve"].unique() if str(x).strip()}
    assert emitted <= set(order), (
        f"engine emits labels the Discovery filter cannot show: {sorted(emitted - set(order))}"
    )
