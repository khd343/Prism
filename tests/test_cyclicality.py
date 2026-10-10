"""Contract: the cyclicality columns — max_earnings_drawdown_5y + cyclicality_tier.

⚠️ SCOPE OF THE BYTE-IDENTICAL TEST BELOW (clarified 2026-08-24). It poisons these columns AFTER
fetch_and_clean_data has run, so it proves only that NOTHING DOWNSTREAM OF THE DATA STAGE reads
them. It does NOT prove the columns are score-free: within the data stage, cyclicality_tier caps
fair_pe_qglp at 18.0 for 'Deep Cyclical / Commodity' (data_engine ~L3672), and that flows into
pe_discount_to_quality → the valuation axis → composite_score. Measured 2026-08-24: re-tiering ONE
industry AT SOURCE moved 28 stocks' composite_score (max 0.28 pts), spilling to sector peers via the
sector-relative rank. Re-tier only behind /census + /verify.

Two columns shipped into compute_derived_signals (core/data_engine.py); no consumer downstream of
the data stage scores them (pinned by the byte-identical composite test below):

  max_earnings_drawdown_5y — deepest TIME-ORDERED peak-to-trough fall in annual PAT over the 6
      available years (current + 5 back). A monotone compounder scores ~0; a commodity that
      collapsed at the trough scores high; >1.0 = the trough went negative. NaN when <4 of 6 present.
  cyclicality_tier(_code) — the a-priori business TYPE from the committed core/cyclicality_map.py
      (industry → sector fallback → "F"). The FINER, industry-level DISPLAY sibling of the
      SECTOR-level cyclical_peak_trap (which DOES feed scoring). Prior(tier) vs realized(drawdown)
      disagreement is itself the signal.

The COMMITTED map (not a runtime CSV read) is what makes this deploy-safe. The coverage guard is
SPLIT (mirroring the project's committed-test / live-ritual pattern): the synthetic miss-detection
below is committed (passes in a code-only CI clone with no data); the LIVE count-of-unmapped==0
tripwire lives in tools/verify.py, where the gitignored real CSV exists.
"""
import sys
import os
import io
import contextlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

from data_engine import (compute_derived_signals, COMMON_COLS, RATIO_COLS, INCOME_COLS,
                          BALANCE_COLS, CASHFLOW_COLS, SHAREHOLDING_COLS, TECHNICAL_COLS)
from core.cyclicality_map import INDUSTRY_TIER, SECTOR_TIER_FALLBACK, TIER_LABELS

_ALL_MAPPED_COLS = set()
for _m in (COMMON_COLS, RATIO_COLS, INCOME_COLS, BALANCE_COLS,
           CASHFLOW_COLS, SHAREHOLDING_COLS, TECHNICAL_COLS):
    _ALL_MAPPED_COLS.update(_m.values())

# Real CSV data is gitignored (code-only repo). The slow display-only test reads it — guard it
# PER-TEST (not module-level) so the 5 synthetic tests still run in a code-only CI clone (the
# committed half of the coverage split). Mirrors test_ui_smoke.py / test_output_consistency.py.
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Other Resources", "CSV Data")


def _frame(n: int = 2, **overrides) -> pd.DataFrame:
    """Full-NaN mapped frame (the §6 synthetic pattern) with per-row overrides. sorted() pins the
    fill order so the fixture's block layout is seed-independent (determinism mandate, §5)."""
    base = {"company_id": [f"NSE:T{i}" for i in range(n)], "name": [f"T{i}" for i in range(n)]}
    for k, v in overrides.items():
        base[k] = list(v) if isinstance(v, (list, np.ndarray)) else [v] * n
    df = pd.DataFrame(base)
    missing = sorted(c for c in _ALL_MAPPED_COLS if c not in df.columns)
    if missing:
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing)], axis=1)
    return df


# ── map integrity ────────────────────────────────────────────────────────────
def test_map_integrity():
    """Every tier code in either map ∈ {A..F}, has a label, and both dicts are key-sorted (§5)."""
    codes = set(INDUSTRY_TIER.values()) | set(SECTOR_TIER_FALLBACK.values())
    assert codes <= set("ABCDEF"), f"tier code outside A-F: {sorted(codes - set('ABCDEF'))}"
    assert codes <= set(TIER_LABELS), f"tier code without a label: {sorted(codes - set(TIER_LABELS))}"
    assert list(INDUSTRY_TIER) == sorted(INDUSTRY_TIER), "INDUSTRY_TIER must be key-sorted (determinism)"
    assert list(SECTOR_TIER_FALLBACK) == sorted(SECTOR_TIER_FALLBACK), "SECTOR_TIER_FALLBACK must be key-sorted"


# ── vendor-known industries that no live stock carries YET (2026-09-19) ────────────
# WHY THESE EXIST, so nobody prunes them as dead entries. The generator builds INDUSTRY_TIER from
# the LIVE universe, so an industry could only ever be tiered AFTER a stock already carried it --
# meaning the first stock in a new industry scored against a sector fallback (or the runtime
# .fillna("F")) until somebody noticed tools/verify.py tripping and re-ran the generator. The
# generator now UNIONS the vendor's own market-scan industry list into its input, which closes that
# one-regeneration lag; classify() is a pure function of the NAME and needs no stock to run.
#
# On the day the union landed, 13 of the vendor's 238 names had zero live stocks -- twelve
# lenders/insurers/AMCs and Music Licensing. That is not random: this universe contains NO BANKS AT
# ALL (HDFC Bank, ICICI, SBI, Axis, Kotak, Bajaj Finance all absent), a deliberate exclusion,
# because a lender has no OPM, no inventory, no CCC and a meaningless D/E.
#
# WHAT THIS DOES *NOT* CLAIM: tiering a bank does not make PRISM able to score one. It means only
# that IF a lender ever enters the universe, its cyclicality tier is correct on arrival instead of
# one regeneration later. Verified at the time: adding all 13 re-tiered ZERO existing industries,
# left SECTOR_TIER_FALLBACK byte-identical, and moved composite_score / rank / gate_pass /
# conviction_tier / fair_pe_qglp on exactly 0 of 2,717 rows -- which is guaranteed by construction,
# since no stock carries any of these names.
_VENDOR_ZERO_STOCK_TIERS = {
    "Banks - PSU": "E",
    "Banks - Private": "E",
    "Banks - Small Finance": "E",
    "Conglomerate Backed NBFC": "E",
    "Finance & Investments - CV Finance": "E",
    "Finance & Investments - Gold Loan": "E",
    "Finance & Investments - MSME Lending": "E",
    "Finance & Investments - Microfinance": "E",
    "Finance - AMC": "E",
    "Finance - Insurance": "E",
    "Finance - Investment Bankers": "E",
    "Finance - Non Life Insurance": "E",
    "Music Licensing": "B",
}


def test_vendor_known_industries_are_tiered_before_their_first_stock_arrives():
    """These carry zero live stocks BY DESIGN and must survive every regeneration. If this fails,
    the generator's VENDOR_INDUSTRY_LIST union was dropped and the one-regen lag is back -- the
    next new industry will score against a sector fallback with nothing saying so."""
    missing = sorted(k for k in _VENDOR_ZERO_STOCK_TIERS if k not in INDUSTRY_TIER)
    assert not missing, (
        f"{len(missing)} vendor-known industries vanished from the map: {missing}. The generator "
        f"unions the vendor's market-scan list into its live input; that union has been removed or "
        f"the map was regenerated by an older copy of the generator."
    )
    wrong = {k: (INDUSTRY_TIER[k], want) for k, want in _VENDOR_ZERO_STOCK_TIERS.items()
             if INDUSTRY_TIER[k] != want}
    assert not wrong, f"tier changed for a vendor-known industry (got, expected): {wrong}"


def test_music_licensing_is_tiered_like_its_media_peers():
    """Music Licensing takes classify()'s DEFAULT tier (B) rather than a bespoke rule -- and that
    default is only defensible because its peers agree. Pinned so a future 'improvement' that
    special-cases one zero-stock industry has to justify breaking the family."""
    assert INDUSTRY_TIER["Music Licensing"] == "B"
    for peer in ("Entertainment & Media", "Printing/Publishing/Stationery"):
        assert INDUSTRY_TIER[peer] == "B", f"{peer} moved off B; re-judge Music Licensing with it"


# ── the vendor's October 2026 names (2026-10-10) ─────────────────────────────────────────────────
# Early October the vendor regrouped and renamed its industries. core/taxonomy.py translates the plain
# relabels back; the 25 it keeps as written (regroupings, new categories) fell to their SECTOR's tier —
# 7.9% of the live stocks, found by the daily data check on its first run. The map now carries them,
# each tier read from its FAMILY in the map (classify() gained five name patterns for the new spellings:
# "it services", "railway", "learning", "food ingredient", "biotechnology"; re-tiering zero of the 384
# existing names). Measured on the live sheet: 7 stocks' tier moved, composite / rank / fair P/E / gate /
# verdict / wealth on 0 of 2,712 — no name is tier A, the only tier that caps the fair P/E.
_VENDOR_OCTOBER_2026 = {
    "Airport Restaurants and Lounges": "B", "Auto Parts Distribution": "B", "Biotechnology": "C",
    "Cryogenic Equipment": "B", "Cutting Tools and Machine Tools": "B", "Engineering and Infrastructure": "B",
    "FMCG and Soap Ingredients": "C", "General Electric Equipment": "B", "IT Services": "D",
    "Insurance Distribution": "E", "Machine Tools": "B", "Medical Equipment and Diagnostic Technology": "C",
    "Natural Extracts and Food Ingredients": "C", "Pharma Distribution": "C", "Pharma Formulations": "C",
    "Pharma Gelatin Capsules": "C", "Pharma IV Fluids": "C", "Pharma Integrated": "C",
    "Publishing and Learning Services": "D", "Railway Equipment": "D", "Restaurants": "B",
    "Secure Printing and Payment Cards": "B", "Solar Power and EPC": "D", "Technology Distribution": "B",
    "Telecommunications Equipment and Network EPC": "B",
}


def test_the_vendors_october_names_are_tiered_by_their_family():
    wrong = {k: (INDUSTRY_TIER.get(k), want) for k, want in _VENDOR_OCTOBER_2026.items()
             if INDUSTRY_TIER.get(k) != want}
    assert not wrong, f"missing or re-tiered (got, expected): {wrong}"
    # each beside the family it was judged against: break the family, re-judge the name
    for new, peer in [("IT Services", "IT - Software"), ("Railway Equipment", "Railways"),
                      ("Natural Extracts and Food Ingredients", "Food - Processing - Others"),
                      ("Publishing and Learning Services", "Computer Education"),
                      ("Telecommunications Equipment and Network EPC", "Telecommunications - Equipment"),
                      ("Machine Tools", "Machine Tools - Others"), ("Solar Power and EPC", "Solar EPC"),
                      ("Restaurants", "Quick Service Restaurant - QSR"), ("Pharma Distribution", "Pharmacy Distribution")]:
        assert INDUSTRY_TIER[new] == INDUSTRY_TIER[peer], f"{new} no longer matches its family ({peer})"
    assert not {k for k in _VENDOR_OCTOBER_2026 if INDUSTRY_TIER[k] == "A"}, (
        "a vendor relabel must not newly cap a fair P/E")
    assert SECTOR_TIER_FALLBACK.get("Insurance") == "E" and SECTOR_TIER_FALLBACK.get("Alcoholic Beverages") == "C"


def test_the_map_never_drops_a_name_an_older_copy_carries():
    """The archived copies Movers re-scores carry the names of their own day (2026-09-18 and before):
    a name that left the live sheet must keep its tier, or an old copy's stocks change tier in a diff."""
    for old_name, tier in [("IT - Software", "D"), ("Pharma - Formulators", "C"), ("Hotels", "B"),
                           ("Pharmaceuticals Bulk Drugs & Formulation", "C"), ("Pharmacy Distribution", "C")]:
        assert INDUSTRY_TIER.get(old_name) == tier, old_name
    assert "Telecom-Handsets/Mobile" in SECTOR_TIER_FALLBACK, "a sector only older copies carry was dropped"
    assert len(INDUSTRY_TIER) >= 409, f"the map shrank to {len(INDUSTRY_TIER)} — a regeneration dropped names"


def test_every_lender_like_industry_is_tier_E():
    """A RULE-level pin, not a list of names: anything whose name says bank / NBFC / insurance /
    microfinance / gold loan / MSME lending / AMC belongs in Financials. Catches a mis-tier in a
    future regeneration that a name-by-name list would miss, and it fires on industries that do not
    exist yet. Tokens are chosen to avoid the 'invest' trap -- Infra/Real Estate Investment Trusts
    are legitimately C, not E."""
    tokens = ("bank", "nbfc", "insurance", "microfinance", "gold loan", "msme lending", " amc", "- amc")
    offenders = {k: v for k, v in INDUSTRY_TIER.items()
                 if any(t in k.lower() for t in tokens) and v != "E"}
    assert not offenders, f"lender-like industries not tiered E (Financials): {offenders}"
    matched = [k for k in INDUSTRY_TIER if any(t in k.lower() for t in tokens)]
    assert len(matched) >= 14, (
        f"only {len(matched)} lender-like industries matched (14 at the 2026-09-19 union) -- the "
        f"token list or the map shrank, so this test may be passing vacuously"
    )


# ── synthetic unmapped-industry detection (the committed half of the coverage split) ──
def test_unmapped_industry_is_detectable_and_falls_to_F():
    """An industry+sector absent from both maps must (a) be detectable as a miss and (b) resolve to
    the 'F' catch-all — never crash. (The live count-of-misses==0 tripwire lives in tools/verify.py.)"""
    assert "__UNMAPPED__" not in INDUSTRY_TIER and "__UNMAPPED__" not in SECTOR_TIER_FALLBACK
    out = compute_derived_signals(_frame(n=1, industry="__UNMAPPED__", sector="__UNMAPPED__"))
    assert out["cyclicality_tier_code"].iloc[0] == "F"
    assert out["cyclicality_tier"].iloc[0] == TIER_LABELS["F"]


def test_known_industry_and_sector_fallback_resolve():
    """A mapped industry resolves to its committed tier; a NEW industry with a KNOWN sector uses the
    sector fallback (not 'F') — the chain industry → sector → 'F'."""
    ind = next(iter(INDUSTRY_TIER))               # any mapped industry (sorted-first)
    sec = next(iter(SECTOR_TIER_FALLBACK))        # any mapped sector
    out = compute_derived_signals(_frame(
        n=2,
        industry=[ind, "__NEW_INDUSTRY__"],
        sector=["__UNKNOWN_SECTOR__", sec],
    ))
    assert out["cyclicality_tier_code"].iloc[0] == INDUSTRY_TIER[ind]          # industry wins
    assert out["cyclicality_tier_code"].iloc[1] == SECTOR_TIER_FALLBACK[sec]   # sector fallback


# ── drawdown semantics ───────────────────────────────────────────────────────
def test_drawdown_vshape_high_compounder_zero_shorthistory_nan():
    """V-shape (collapse-then-recover) scores deep; monotone compounder ~0; <4 of 6 years → NaN.
    Columns are newest→oldest (pat, pat_1yb … pat_5yb); the engine reverses to oldest→newest."""
    out = compute_derived_signals(_frame(
        n=3,
        pat    =[100.0, 100.0, 50.0],   # row0 V-shape (recovered) | row1 compounder | row2 short
        pat_1yb=[ 20.0,  80.0, 40.0],
        pat_2yb=[100.0,  60.0, np.nan],
        pat_3yb=[100.0,  40.0, np.nan],
        pat_4yb=[100.0,  20.0, np.nan],
        pat_5yb=[100.0,  10.0, np.nan],
    ))
    dd = out["max_earnings_drawdown_5y"]
    # row0 oldest→newest 100,100,100,100,20,100 → peak 100, trough 20 → dd 0.8
    assert dd.iloc[0] > 0.7, f"V-shape should show deep drawdown, got {dd.iloc[0]}"
    # row1 oldest→newest 10,20,40,60,80,100 monotone up → dd ~0
    assert dd.iloc[1] < 0.05, f"compounder should be ~0, got {dd.iloc[1]}"
    # row2 only 2 of 6 present → NaN
    assert pd.isna(dd.iloc[2]), f"short history (<4 yrs) must be NaN, got {dd.iloc[2]}"


def test_drawdown_negative_trough_exceeds_one():
    """A trough that goes negative yields drawdown > 1.0 — (peak − neg_trough)/peak."""
    out = compute_derived_signals(_frame(
        n=1, pat=[50.0], pat_1yb=[-30.0], pat_2yb=[80.0],
        pat_3yb=[100.0], pat_4yb=[90.0], pat_5yb=[70.0],
    ))
    # oldest→newest 70,90,100,80,-30,50 → peak 100, trough -30 → (100-(-30))/100 = 1.3
    assert out["max_earnings_drawdown_5y"].iloc[0] > 1.0


def test_drawdown_caps_tiny_running_peak_blowup():
    """A small EARLY running peak before a later large loss makes the raw ratio explode on a near-zero
    denominator; the column caps at 3.0 so the data — and any UI/mean — never sees the 55100% tail."""
    out = compute_derived_signals(_frame(
        n=1, pat=[6.0], pat_1yb=[5.0], pat_2yb=[4.0], pat_3yb=[3.0], pat_4yb=[-11.0], pat_5yb=[2.0],
    ))
    # oldest→newest 2,-11,3,4,5,6: running peak before -11 is 2 → raw (2-(-11))/2 = 6.5, capped to 3.0
    assert out["max_earnings_drawdown_5y"].iloc[0] == 3.0


# ── the KEY invariant: display-only, zero scoring leakage ─────────────────────
@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(_DATA_DIR),
                    reason="local CSV data absent (code-only checkout) — needs real data")
def test_display_only_composite_byte_identical():
    """Corrupting the two cyclicality columns must NOT change composite_score — proving they feed no
    score (mirrors the verdict-engine display-only contract). compute_derived_signals runs in the
    DATA stage (fetch_and_clean_data), so run_scoring_pipeline never recomputes them: poisoning the
    clean frame's columns persists through scoring, and an identical composite proves no dependency."""
    from core import run_scoring_pipeline
    from core.data_engine import fetch_and_clean_data
    with contextlib.redirect_stdout(io.StringIO()):
        clean = fetch_and_clean_data("local")
        baseline = run_scoring_pipeline(clean.copy(), "Hybrid", "Balanced")
        poisoned_in = clean.copy()
        poisoned_in["max_earnings_drawdown_5y"] = 999.0
        poisoned_in["cyclicality_tier_code"] = "Z"
        poisoned_in["cyclicality_tier"] = "GARBAGE"
        poisoned = run_scoring_pipeline(poisoned_in, "Hybrid", "Balanced")
    pd.testing.assert_series_equal(
        baseline["composite_score"], poisoned["composite_score"], check_names=False)
