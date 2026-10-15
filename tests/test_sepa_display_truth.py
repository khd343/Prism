"""Contract: the SEPA panel and the Reference say what the engine checks (2026-10-15).

Measured on the live sheet (14 Oct data, 2,695 stocks) before the change:
  - the T box claimed "50D > 150D > 200D all rising (C1–C5)": of 1,081 stocks passing it, 125 lacked that
    stack and 290 passed on 4 of its 5 checks — and no slope is in the data;
  - the Reference called the volume contraction (VCP) part of SEPA: 38 of 58 certified stocks had none;
  - 9 stocks showed six green boxes and "Criteria Not Met" with nothing saying why;
  - ↩️ Pullback said "on volume dry-up" for a 10-day-below-50-day rule while "dry-up" everywhere else is
    the 70% line: 77 of 300 Pullback stocks read ⚪ Normal Volume on the ⏳ Consolidation Base card.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

import ui.ui_tearsheet as T
from config import SEPA_MIN_MARKET_CAP_CR, TREND_TEMPLATE_MAX_BELOW_HIGH
from core.data_engine import VCP_DRYUP_SHARE
from test_cockpit_display_truth import _Recorder
from ui.ui_reference_data import CONCEPT_REFERENCE

_SIX = ("sepa_trend_template", "sepa_adx_confirmed", "sepa_low_base",
        "sepa_rs_confirmed", "sepa_earnings_fuel", "sepa_institutional")


def _panel(**over) -> str:
    """render_sepa_radar for one stock — six green pillars and both gates met unless overridden."""
    base = {**{k: 1 for k in _SIX}, "sepa_vcp_dryup": 0, "sepa_score": 6, "sepa_pass": 1,
            "sepa_above_50d": 1, "price_vs_50d_pct": 2.1,
            "sepa_near_high": 1, "sepa_mcap_ok": 1, "dist_52wh": 12.0, "market_cap": 1234.0,
            "vcp_volume_ratio": 0.9}
    base.update(over)
    rec, real = _Recorder(), T.st
    try:
        T.st = rec
        T.render_sepa_radar(pd.Series(base))
    finally:
        T.st = real
    return " ".join(rec.out)


def _seen(html: str) -> str:
    """The panel as read on screen: tags gone, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


# ── The T box ─────────────────────────────────────────────────────────────────────────────────────
def test_the_t_box_names_the_five_checks_the_engine_counts():
    """sepa_trend_template = d45_trend_structure >= 4: price above the 200-day, price above the 30-week,
    30-week above 200-day, 50-day above both, price above its stop."""
    seen = _seen(_panel())
    t_box = seen[seen.index("Trend Template"):seen.index("ADX Confirmed")]
    assert "4 of 5" in t_box
    for check in ("30-week", "200-day", "50-day", "stop"):
        assert check in t_box, f"the T box does not name the {check} check"


# ── The "Also required" line ──────────────────────────────────────────────────────────────────────
def test_also_required_states_each_gate_with_the_stocks_own_number():
    seen = _seen(_panel(sepa_pass=0, sepa_mcap_ok=0, market_cap=312.0))
    assert "SEPA Criteria Not Met" in seen
    assert "✓ price above its 50-day average (2% above)" in seen
    assert "✓ within 25% of its 52-week high (12% below)" in seen
    assert "✗ market cap ₹500 Cr+ (₹312 Cr)" in seen


def test_a_price_below_its_50_day_is_named_as_the_reason():
    """Criterion 5 (2026-10-15): 15 of 58 certified stocks on the 14 Oct sheet sat 0.3-4.0% below their
    50-day — six green boxes, and now the line says which rule they miss."""
    seen = _seen(_panel(sepa_pass=0, sepa_above_50d=0, price_vs_50d_pct=-3.9))
    assert "SEPA Criteria Not Met" in seen
    assert "✗ price above its 50-day average (4% below)" in seen


def test_also_required_reads_the_engines_flags_never_the_inputs():
    """Pure display: each flag says no while its input looks fine — the panel shows the flag's verdict
    for every gate (a mutation that re-judged the 50-day number in the UI survived until this covered it)."""
    seen = _seen(_panel(sepa_pass=0, sepa_above_50d=0, price_vs_50d_pct=2.1, sepa_near_high=0, dist_52wh=10.0,
                        sepa_mcap_ok=0, market_cap=1234.0))
    assert "✗ price above its 50-day average" in seen
    assert "✗ within 25% of its 52-week high" in seen
    assert "✗ market cap ₹500 Cr+" in seen


def test_a_printed_number_never_contradicts_its_mark():
    seen = _seen(_panel(sepa_pass=0, sepa_near_high=0, dist_52wh=25.3, sepa_mcap_ok=0, market_cap=499.6))
    assert "(25.3% below)" in seen, "25.3% missed the 25% line, so it never prints as 25%"
    assert "(₹499.6 Cr)" in seen, "₹499.6 Cr missed the ₹500 Cr floor, so it never prints as ₹500"
    seen = _seen(_panel(dist_52wh=0.2, market_cap=123456.0))
    assert "(at its high)" in seen and "(₹123,456 Cr)" in seen
    assert "(0.04% above)" in _seen(_panel(price_vs_50d_pct=0.04)), "a ✓ never prints as 0% above"
    assert "(0.04% below)" in _seen(_panel(sepa_pass=0, sepa_above_50d=0, price_vs_50d_pct=-0.04))
    # Above vs below IS criterion 5, so the sign must survive the rounding: 0.3% below never reads as
    # "at its average" (rounding -0.3 to 0 keeps "not above" but loses "below").
    assert "(0.3% below)" in _seen(_panel(sepa_pass=0, sepa_above_50d=0, price_vs_50d_pct=-0.3))


def test_missing_numbers_are_named_not_guessed():
    seen = _seen(_panel(sepa_pass=0, sepa_near_high=0, dist_52wh=np.nan, sepa_mcap_ok=0, market_cap=np.nan,
                        sepa_above_50d=0, price_vs_50d_pct=np.nan))
    assert seen.count("(not reported)") == 3


def test_the_line_quotes_the_engines_own_lines():
    seen = _seen(_panel())
    assert f"within {TREND_TEMPLATE_MAX_BELOW_HIGH}% of its 52-week high" in seen
    assert f"market cap ₹{SEPA_MIN_MARKET_CAP_CR:,} Cr+" in seen


def test_a_frame_scored_before_the_flags_existed_shows_no_line():
    """A cached frame without the flags must not show a ✗ for a gate it never measured."""
    base = {**{k: 1 for k in _SIX}, "sepa_vcp_dryup": 0, "sepa_score": 6, "sepa_pass": 1}
    rec, real = _Recorder(), T.st
    try:
        T.st = rec
        T.render_sepa_radar(pd.Series(base))
    finally:
        T.st = real
    assert "Also required" not in " ".join(rec.out)
    assert "Also required" in _panel()


# ── The Reference and the framework card ──────────────────────────────────────────────────────────
def _ref(category: str, label: str) -> str:
    return dict(CONCEPT_REFERENCE[category])[label]


def test_the_reference_sepa_line_states_what_certification_checks():
    txt = _ref("⚡ Frameworks — Momentum & Growth", "⚡ SEPA Momentum")
    assert f"{TREND_TEMPLATE_MAX_BELOW_HIGH}%" in txt and f"₹{SEPA_MIN_MARKET_CAP_CR:,} Cr" in txt
    assert "above its 50-day average" in txt, "criterion 5 is part of certification (2026-10-15)"
    assert f"{VCP_DRYUP_SHARE:.0%}" in txt and "not required" in txt
    assert "rising" not in txt, "no moving-average slope is in the data"


def test_the_sepa_card_calls_the_volume_contraction_a_bonus():
    assert "not required" in T._FW_IDEA["SEPA Momentum"]


def test_dry_up_means_one_thing_across_the_reference():
    """"Dry-up" is the 70% line (VCP_DRYUP_SHARE) wherever the Reference uses it; a looser volume rule gets
    its own words. Only the ⏳ Consolidation Base family, which defines the term, is exempt."""
    for category, rows in CONCEPT_REFERENCE.items():
        if category == "⏳ Consolidation Base":
            continue
        for label, text in rows:
            if re.search(r"dry-up|dried up|dries up|drying up", text, re.I):
                assert f"{VCP_DRYUP_SHARE:.0%}" in text, f"{label}: says dry-up without the 70% line"


def test_the_pullback_states_its_own_volume_rule():
    """trend_pullback asks for vol_sma_10d < vol_sma_50d — lighter volume, not the 70% dry-up."""
    assert "10-day average below the 50-day" in _ref("↩️ Trend Modifier", "↩️ Pullback")
