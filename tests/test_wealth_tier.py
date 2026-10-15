"""
test_wealth_tier.py
===================
Contract for the wealth-tier grammar (core/verdict_engine.py — wealth_ep_pct, wealth_vel_pct,
wealth_tier, wealth_warn).

WHAT IT IS. A second, deliberately PRICE-BLIND verdict answering a different question from
verdict_direction: not "is this business good and safe now?" but "is it becoming more valuable?"
Three clocks only:

    A  EP%  = economic_profit / reserves × 100 = ROE − cost of equity    (earning above hurdle)
    B  Vel% = EP velocity / reserves × 100 ≥ 0.5                         (materially improving)
    C  tau  = moat_tau ≥ +0.25                                            (5Y margin spine)
    fading: tau ≤ −0.25 caps everything at UNEVEN

    CLIMBING★ A·B·C · CLIMBING A·B(flat) · RECOVERING ¬A·B·C (confirmed turnaround) · UNEVEN the
    clocks disagree · STALLED nothing improving · N/A any input unverifiable

THE WORDS (renamed 2026-10-15 from BUY★ / BUY / WATCH★ / WATCH / AVOID — the user's call, after the
live sheet showed what the action words said: BUY on 518 stocks, 85% of BUY★ FLAWED on Soundness,
AVOID on Colgate at EP% +71.7). The tier measures which way value creation is moving and never sees
the price, so it speaks DIRECTION words. Same rules, same thresholds, the same stocks in every tier —
pinned below by replaying the pre-rename grammar through the map.

PROVENANCE — the grammar was stress-tested BEFORE being built, on ~250 names across three
dry-runs (2026-08-27): the user's watchlists (mostly engine-AVOIDs, where it promoted hidden
improvers like Sarda), the engine's complete 18-stock BUY list (which it split 6/1/0/5/5/1,
demoting decaying legends like Vedant Fashions and Gulf Oil), and the engine's complete 126-stock
WATCH tier (split 36/12/0/43/27/8 — separating Nestle, still compounding, from Colgate at EP%
+70.4 but decaying). (Those BUY / WATCH / AVOID are the ENGINE VERDICT's words of the time, now
SOUND / MIXED / FLAWED.) The normalization matters: absolute EP is size-biased (it ranked Sarda's
+₹251 Cr above far larger PERCENTAGE improvers); EP/reserves is scale-free and equals ROE − CoE
EXACTLY (one basis — EP is built on reserves. The original build divided by net_worth, a
market_cap ÷ P/B figure on a DIFFERENT equity definition: corr 0.980 but ratio 0.004–14.0 in
the tails, GKW displaying EP% −77.7 against a true −12.1. Fixed 2026-08-28; 8 tiers moved,
none into or out of CLIMBING/CLIMBING★).

THE FOUR RULES THESE TESTS DEFEND:
  1. ⚠ NEVER ALTERS THE TIER — the clocks cannot see forensics (16 of 26 CLIMBING★ on one live list
     carried 8+ flags), so risk rides BESIDE the tier, unblended.
  2. STRICT N/A — any missing input yields N/A, never a tier: unverifiable is not passed, and
     equally never condemned. (The Lohia ruling: EP% +24.6 with tau missing stays N/A.)
  3. DISPLAYED-PRECISION CLASSIFICATION — the grammar rounds to what the UI prints (1dp for %,
     2dp for tau) before comparing, so two stocks printing the same number tier the same.
  4. THE FADING CAP BEATS EVERYTHING — improvement without a margin spine stays on probation.

Run with: pytest tests/test_wealth_tier.py -v
"""

import contextlib
import inspect
import io as _io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

from core.verdict_engine import (WEALTH_NA, WEALTH_TAU_CONF, WEALTH_TAU_FADE, WEALTH_TIER_RENAMED,
                                 WEALTH_TIERS, WEALTH_VEL_MIN, WEALTH_WARN_FLAGS, compute_verdict)

TIERS = [*WEALTH_TIERS, WEALTH_NA]
OLD_ACTION_WORDS = ("BUY★", "BUY", "WATCH★", "WATCH", "AVOID")

_DEFAULTS = {
    "conviction_tier": 1, "composite_score": 90.0,
    "forensic_score": 90.0, "schilit_pass": 1, "red_flag_count": 1,
    "corporate_class": "🏆 GREAT",
    "quality_score": 90.0, "growth_score": 80.0,
    "expected_excess_return": 20.0, "pe": 20.0, "fair_pe_qglp": 30.0,
    "buy_zone_label": "Accumulate", "governance_risk_multiplier": 1.0,
    "data_coverage_pct": 90.0,
    # wealth inputs — defaults make a clean CLIMBING★; each test overrides what it probes.
    # reserves (not net_worth) is the denominator: the SAME equity base EP is built on.
    "reserves": 1000.0, "economic_profit": 100.0,
    "economic_profit_velocity": 50.0, "moat_tau": 1.0,
}


def _tier(**overrides):
    row = {**_DEFAULTS, **overrides}
    out = compute_verdict(pd.DataFrame([row]))
    return out["wealth_tier"].iloc[0]


def _warn(**overrides):
    row = {**_DEFAULTS, **overrides}
    return int(compute_verdict(pd.DataFrame([row]))["wealth_warn"].iloc[0])


# ── 0. The words (2026-10-15) ───────────────────────────────────────────────────────────────
def test_the_ladder_is_five_direction_words_strongest_first():
    assert WEALTH_TIERS == ("CLIMBING★", "CLIMBING", "RECOVERING", "UNEVEN", "STALLED")
    assert WEALTH_NA == "N/A" and WEALTH_NA not in WEALTH_TIERS, "N/A is off the ladder, never a tier"


def test_the_old_action_words_map_one_to_one_onto_the_new():
    """Snapshots and exports written before 2026-10-15 carry the old words; tools/validate.py reads them
    through this map. One-to-one, in ladder order — a merged or crossed pair would mislabel history."""
    assert WEALTH_TIER_RENAMED == {"BUY★": "CLIMBING★", "BUY": "CLIMBING", "WATCH★": "RECOVERING",
                                   "WATCH": "UNEVEN", "AVOID": "STALLED"}
    assert tuple(WEALTH_TIER_RENAMED) == OLD_ACTION_WORDS
    assert tuple(WEALTH_TIER_RENAMED.values()) == WEALTH_TIERS


def _pre_rename_tier(frame: pd.DataFrame) -> np.ndarray:
    """The grammar exactly as it shipped before 2026-10-15, old words and all (a frozen oracle; the
    thresholds are read from the engine, so this pins the WORDS and the rule order, not calibration)."""
    eq = frame["reserves"].to_numpy(float)
    ok = np.isfinite(eq) & (eq > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ep = np.round(np.where(ok, frame["economic_profit"].to_numpy(float) / eq * 100.0, np.nan), 1)
        vel = np.round(np.where(ok, frame["economic_profit_velocity"].to_numpy(float) / eq * 100.0, np.nan), 1)
        tau = np.round(frame["moat_tau"].to_numpy(float), 2)
        known = np.isfinite(ep) & np.isfinite(vel) & np.isfinite(tau)
        A, B = known & (ep > 0.0), known & (vel >= WEALTH_VEL_MIN)
        C, F = known & (tau >= WEALTH_TAU_CONF), known & (tau <= WEALTH_TAU_FADE)
    return np.select([~known, F & B, F, A & B & C, A & B, B & C, (C & ~B) | (B & ~A)],
                     ["N/A", "WATCH", "AVOID", "BUY★", "BUY", "WATCH★", "WATCH"], default="AVOID")


def test_the_rename_moved_no_stock():
    """THE CONTRACT OF THE RENAME: words only. Over every corner of the grammar (each clock on, off,
    at its bar and printing-to-zero; tau fading, flat and confirming; a missing input), today's tier
    is exactly the pre-rename tier read through WEALTH_TIER_RENAMED."""
    ep, vel = [-30.0, 0.4, 100.0], [-10.0, 0.0, 0.4, 4.6, 50.0]
    tau, res = [-0.83, -0.25, 0.0, 0.24, 0.25, 1.0, np.nan], [1000.0, np.nan]
    grid = pd.DataFrame([{**_DEFAULTS, "economic_profit": e, "economic_profit_velocity": v, "moat_tau": t,
                          "reserves": r} for e in ep for v in vel for t in tau for r in res])
    now = compute_verdict(grid)["wealth_tier"].to_numpy()
    then = pd.Series(_pre_rename_tier(grid)).replace(WEALTH_TIER_RENAMED).to_numpy()
    assert len(grid) == 210 and (now == then).all(), [(a, b) for a, b in zip(then, now) if a != b][:5]
    assert set(now) == set(TIERS), f"the grid no longer reaches every tier: {sorted(set(now))}"


def test_the_tier_never_speaks_an_action_word():
    grid = pd.DataFrame([{**_DEFAULTS, "economic_profit": e, "economic_profit_velocity": v, "moat_tau": t}
                         for e in (-30.0, 100.0) for v in (-10.0, 50.0) for t in (-0.83, 0.0, 1.0)])
    assert not set(compute_verdict(grid)["wealth_tier"]) & set(OLD_ACTION_WORDS)


# ── 1. The grammar, tier by tier ────────────────────────────────────────────────────────────
def test_all_three_clocks_is_climbing_star():
    assert _tier() == "CLIMBING★"                  # EP% +10, Vel% +5, tau +1.00


def test_earning_and_improving_with_flat_tau_is_climbing():
    assert _tier(moat_tau=0.0) == "CLIMBING"


def test_confirmed_turnaround_is_recovering():
    """EP still below the hurdle, but improving with a margin spine — the Entero/Shilpa shelf."""
    assert _tier(economic_profit=-30.0) == "RECOVERING"


def test_tau_only_is_uneven():
    """The Igarashi case: margin spine, nothing else."""
    assert _tier(economic_profit=-30.0, economic_profit_velocity=-10.0) == "UNEVEN"


def test_good_business_with_momentum_broken_is_uneven():
    """The Jagsonpal case: earning + spine, velocity negative."""
    assert _tier(economic_profit_velocity=-20.0) == "UNEVEN"


def test_velocity_only_is_uneven():
    """Improving, not earning, no spine — one clock."""
    assert _tier(economic_profit=-30.0, moat_tau=0.0) == "UNEVEN"


def test_level_without_change_is_stalled():
    """The Vedant disease: earning fine, nothing improving, no spine."""
    assert _tier(economic_profit_velocity=0.0, moat_tau=0.0) == "STALLED"


def test_nothing_is_stalled():
    assert _tier(economic_profit=-30.0, economic_profit_velocity=-10.0, moat_tau=0.0) == "STALLED"


# ── 2. The fading cap beats everything ──────────────────────────────────────────────────────
def test_fading_caps_a_full_house_at_uneven():
    """Even A·B, margins collapsing → UNEVEN, never a CLIMBING tier (the Varroc rule)."""
    assert _tier(moat_tau=-0.83) == "UNEVEN"


def test_fading_without_improvement_is_stalled():
    """The Thermax/Colgate case: earning (even hugely), fading, momentum dead."""
    assert _tier(economic_profit=700.0, economic_profit_velocity=-60.0, moat_tau=-0.83) == "STALLED"


# ── 3. Strict N/A — unverifiable is neither passed nor condemned ────────────────────────────
@pytest.mark.parametrize("hole", [
    {"reserves": np.nan}, {"reserves": 0.0}, {"reserves": -50.0},
    {"economic_profit": np.nan}, {"economic_profit_velocity": np.nan}, {"moat_tau": np.nan},
])
def test_any_missing_input_is_na(hole):
    assert _tier(**hole) == "N/A"


def test_the_lohia_ruling_strong_numbers_with_missing_tau_stay_na():
    """EP% +24.6, Vel% +10.9, tau absent → N/A, deliberately. The row still shows the numbers;
    the TIER refuses to certify on absent evidence."""
    assert _tier(reserves=1000.0, economic_profit=246.0,
                 economic_profit_velocity=109.0, moat_tau=np.nan) == "N/A"


# ── 4. Displayed-precision classification ───────────────────────────────────────────────────
def test_two_stocks_printing_the_same_velocity_tier_the_same():
    """Vel% 0.04 prints +0.0 → below the 0.5 bar; 0.46 prints +0.5 → at it. The comparison runs
    on the printed value, so what you read is what was tiered."""
    below = _tier(economic_profit_velocity=0.4, moat_tau=1.0)     # Vel% = 0.04 → prints 0.0
    at    = _tier(economic_profit_velocity=4.6, moat_tau=1.0)     # Vel% = 0.46 → prints 0.5
    assert below == "UNEVEN"     # A + C, momentum below the bar
    assert at == "CLIMBING★"


def test_ep_that_prints_zero_is_not_earning():
    """EP% 0.04 prints +0.0 — the reader sees zero, so A must be False."""
    assert _tier(economic_profit=0.4) == "RECOVERING"   # ¬A · B · C


# ── 5. ⚠ rides beside the tier, never inside it ────────────────────────────────────────────
def test_warn_fires_on_flags_and_on_schilit():
    assert _warn(red_flag_count=WEALTH_WARN_FLAGS) == 1
    assert _warn(red_flag_count=WEALTH_WARN_FLAGS - 1) == 0
    assert _warn(schilit_pass=0) == 1


def test_warn_never_changes_the_tier():
    """The Krishana rule: 12 flags is a ⚠, not a demotion — blending would hide the tension."""
    assert _tier(red_flag_count=12) == _tier(red_flag_count=0) == "CLIMBING★"
    assert _warn(red_flag_count=12) == 1


# ── 6. Display-only, vectorized, and price-blind ────────────────────────────────────────────
def test_wealth_block_mutates_no_scoring_column():
    frame = pd.DataFrame([_DEFAULTS])
    out = compute_verdict(frame)
    for c in ("composite_score", "conviction_tier", "quality_score", "growth_score"):
        assert float(out[c].iloc[0]) == float(frame[c].iloc[0]), f"{c} was mutated"


def test_wealth_block_is_vectorized():
    """The engine's vectorization mandate — no .apply / iterrows anywhere in compute_verdict."""
    src = inspect.getsource(compute_verdict)
    assert ".apply(" not in src, "compute_verdict grew a row-wise .apply"
    assert "iterrows" not in src, "compute_verdict grew a row loop"


def test_grammar_reads_no_price_column():
    """PRICE-BLIND BY DESIGN: the tier must be computable from the four wealth inputs alone.
    Behavioural proof — removing every price/valuation input must not change the tier."""
    row = {k: v for k, v in _DEFAULTS.items() if k not in ("pe", "fair_pe_qglp", "buy_zone_label")}
    assert compute_verdict(pd.DataFrame([row]))["wealth_tier"].iloc[0] == "CLIMBING★"


# ── 7. Liveness on the real universe ────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def live():
    from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                             merge_datasets)
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


def test_every_tier_is_alive(live):
    counts = live["wealth_tier"].value_counts()
    for t in TIERS:
        assert counts.get(t, 0) > 0, f"tier {t} fires on zero stocks — dead by the liveness rule"
    assert set(counts.index) <= set(TIERS), f"a word off the ladder: {sorted(set(counts.index) - set(TIERS))}"


def test_calibration_bands(live):
    """Shares measured at build time: CLIMBING★ 12% · CLIMBING 5% · RECOVERING 9% · UNEVEN 32% ·
    STALLED 34% · N/A 8%. Wide bands so a data refresh does not cry wolf; a collapse into one tier fails."""
    share = live["wealth_tier"].value_counts(normalize=True)
    assert 0.04 < share.get("CLIMBING★", 0) < 0.30, f"CLIMBING★ at {share.get('CLIMBING★', 0):.1%}"
    assert share.get("STALLED", 0) < 0.60, "STALLED is swallowing the universe — the engine-verdict disease"
    assert share.get("N/A", 0) < 0.20, "N/A too large — an input's coverage collapsed"


def test_ep_pct_is_exactly_roe_minus_coe(live):
    """THE ONE-BASIS CONTRACT (2026-08-28). EP = reserves × (ROE − CoE) / 100 in data_engine, and
    this engine divides by the SAME reserves base — so EP% ≡ ROE − CoE must hold EXACTLY (fp
    tolerance), for every row where it is defined. This is the drift alarm: if data_engine ever
    changes EP's equity base, or this divisor ever changes, the identity breaks and this fails.
    The original build divided by net_worth (market_cap ÷ P/B — a different equity definition)
    and this identity held only approximately (ratio 0.004–14.0 in the tails)."""
    from config import COST_OF_EQUITY
    known = live["wealth_ep_pct"].notna()
    assert known.sum() > 1500, "EP% coverage collapsed — the identity check has nothing to bite"
    np.testing.assert_allclose(
        live.loc[known, "wealth_ep_pct"].to_numpy(),
        (live.loc[known, "roe"] - COST_OF_EQUITY).to_numpy(),
        rtol=0, atol=1e-6,
        err_msg="wealth_ep_pct no longer equals ROE − CoE — the EP numerator and the wealth "
                "denominator have drifted onto different equity bases",
    )


def test_tiers_mean_what_they_claim_on_live_data(live):
    """Every CLIMBING★ must actually satisfy its own definition — the grammar and the data agree."""
    b = live[live["wealth_tier"] == "CLIMBING★"]
    assert (np.round(b["wealth_ep_pct"], 1) > 0).all()
    assert (np.round(b["wealth_vel_pct"], 1) >= WEALTH_VEL_MIN).all()
    assert (np.round(b["moat_tau"], 2) >= WEALTH_TAU_CONF).all()
    na = live[live["wealth_tier"] == "N/A"]
    assert (na["wealth_ep_pct"].isna() | na["wealth_vel_pct"].isna() | na["moat_tau"].isna()).all()


def _validate_harness():
    """tools/validate.py — the local harness December's test runs on (plan rule 2). tools/ is gitignored."""
    root = os.path.join(os.path.dirname(__file__), "..", "tools")
    if not os.path.exists(os.path.join(root, "validate.py")):
        pytest.skip("tools/ is gitignored; the harness is local-only")
    sys.path.insert(0, root)
    with contextlib.redirect_stdout(_io.StringIO()):
        import validate
    return validate


def test_old_snapshots_are_read_in_todays_words(tmp_path, monkeypatch):
    """Snapshots saved before 2026-10-15 carry BUY★…AVOID. Read raw, December's wealth arms would match
    none of them and the 09-19 and 10-03 vintages would drop out of the label's first real test. The
    verdict column keeps ITS OWN older map (BUY → SOUND): one column's words never cross into the other."""
    V = _validate_harness()
    pd.DataFrame({"company_id": list("abcdef"), "wealth_tier": [*OLD_ACTION_WORDS, "N/A"],
                  "verdict_direction": ["BUY", "WATCH", "AVOID", "SOUND", "MIXED", "FLAWED"]}).to_csv(
        tmp_path / "prism_snapshot_2026-10-03.csv", index=False)
    monkeypatch.setattr(V, "_SNAP_DIR", tmp_path)
    (_date, snap), = V._load_snapshots()
    assert list(snap["wealth_tier"].iloc[:5]) == list(WEALTH_TIERS)
    assert list(snap["verdict_direction"]) == ["SOUND", "MIXED", "FLAWED", "SOUND", "MIXED", "FLAWED"]


def test_decembers_wealth_arms_are_todays_ladder():
    V = _validate_harness()
    df = pd.DataFrame({"wealth_tier": list(WEALTH_TIERS)})
    arms = dict(V._registered_cohorts(df))["LABEL wealth_tier — does it separate returns?"]
    assert [name for name, _ in arms] == list(WEALTH_TIERS)
    assert all(int(mask.sum()) == 1 for _, mask in arms), "an arm matched a word other than its own"


def test_the_two_verdicts_are_genuinely_different_lenses(live):
    """The point of the whole build: the wealth tier must NOT collapse into verdict_direction.
    Measured at build time, the engine's 18 BUYs split across four wealth tiers."""
    cross = pd.crosstab(live["verdict_direction"], live["wealth_tier"])
    assert (cross.astype(bool).sum(axis=1) >= 2).all(), (
        "every engine verdict maps to a single wealth tier — the lenses have collapsed into one"
    )
