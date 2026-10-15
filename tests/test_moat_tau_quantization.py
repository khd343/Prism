"""
test_moat_tau_quantization.py
=============================
Contract: `moat_tau` is QUANTIZED, and the wealth tier's C/FADE gates depend on the attainable
value set — which is a function of HOW MANY OPM LEVEL COLUMNS the ladder holds.

HISTORY (2026-09-11 label-threshold audit). The ladder was FOUR points — and one of them was
`opm_med_5y`, a 5-year MEDIAN standing in for the missing 5-years-back level. Four points is 6
pairs, so tau held only the thirteen multiples of 1/6; 0.25 was unreachable, the gate that READ
0.25 MEANT `tau >= 1/3`, and 692 stocks (25.5%) sat one OPM observation from flipping C or FADE.
That audit said the coarseness must NOT be "fixed" — because moat_tau already consumed every OPM
level column that existed. That was the only reason.

2026-09-19: the vendor's `OPM 3 Years Back` and `OPM 5 Years Back` arrived, both verified to be
what their names claim (lag decay vs `opm` 0.951 → 0.716 → 0.577; each correlates best with the
PAT/revenue-derived NPM of its OWN year; ≤0.1% identical to any level sibling). The ladder is
now FIVE true levels, oldest first: opm_5yb → opm_3yb → opm_1yb → opm → opm_latest_q. The median
left the ladder — a median is centred ~2.5 years back and is not a point in time.

WHAT CHANGED, MEASURED (company_id-aligned A/B on the 2,717-row vintage):
  * attainable values 13 → 27: multiples of 1/10 on the full ladder (79.8% of rows) and of 1/6
    on rows lacking one level (11.9%; they keep exactly min_pairs = 6). 0.25 is STILL unreachable
    (2dp: 0.20 / 0.30 / 0.17 / 0.33), so `>= 0.25` now means `>= 0.3` on tenths and `>= 1/3` on
    sixths — pinned below, because the constant's meaning is a function of this list.
  * the fragility is FINER, not gone: the step (0.1) still exceeds the gap to the cut (0.05), so
    one pair flip still crosses it; but a single OPM observation now touches 4 of 10 pairs (40%)
    where it touched 3 of 6 (50%). 604 stocks (24.3%) sit one step from a gate — 267 at +0.2,
    246 at −0.2, a handful at ±0.3 and the sixths.
  * the new ladder SPANS FIVE YEARS and the old one did not. Against a Kendall tau over the
    PAT/revenue-derived NPM of the same four years (columns the OPM ladder never touches) the
    new tau reads ρ +0.563 where the old read +0.353.

TWO CORRECTIONS TO THE ABOVE, MADE THE SAME DAY. Both are recorded rather than quietly edited,
because the first draft of this file argued the change was justified on evidence it did not have.

  1. THE WITNESS IS SPAN-MATCHING, NOT AN ACCURACY TEST, and the first draft used it as one.
     Re-run against NPM witnesses of three different spans, EACH LADDER WINS ITS OWN SPAN:
         witness 5Y (FY21,23,25,26):  OLD +0.353   NEW +0.563   A +0.315   -> NEW
         witness 3Y (FY23,25,26):     OLD +0.492   NEW +0.509   A +0.546   -> A
         witness 2Y (FY25,26):        OLD +0.447   NEW +0.315   A +0.440   -> OLD
     (A = [opm_3yb, opm_1yb, opm, opm_latest_q], the 4-rung variant.) So the witness cannot say
     which ladder is "more accurate" in general -- it says which SPAN each one measures. That
     turns the question into "which span is specified?", which has a written answer: the tile
     reads "Margin Trend (5Y Tau)" and verdict_engine's leg C is "the 5-year margin spine".
     Five years is the spec, and only this ladder delivers it -- the old ladder's oldest rung
     was a MEDIAN centred ~FY2024, so three of its four rungs sat inside ~18 months.

  2. A FORWARD TEST IS POSSIBLE AFTER ALL, AND THIS LADDER LOSES IT. The first draft said no
     forward test existed because the columns are in no earlier snapshot. That was wrong for one
     window: opm_3yb/opm_5yb are FIXED FISCAL-YEAR levels, and the fiscal year did NOT roll
     between the 2026-08-22 and 2026-09-19 snapshots (roll test 1.1%; every shared level matches
     92.8-99.6%), so today's values are what the August vintage carried and can be backfilled.
     Validated: recomputing the OLD ladder on that snapshot reproduces its stored moat_tau on
     100.0% of rows. Result, forward to 2026-09-19 (n=1818 common, corp actions excluded):
         OLD +0.1266   NEW +0.1106   A +0.1400
     P(NEW beats OLD) = 10.7%, bootstrap 90% CI [-0.0373, +0.0052]. Where the two ladders
     disagree on direction the OLD one is right (-4.57% vs a -2.08% universe). Window 1 CANNOT
     be reconstructed: the fiscal year rolled between June and August (June's opm = September's
     opm_1yb on 80.8%), so June's "3 years back" is FY2022, which exists in no column.

WHY IT SHIPS ANYWAY, stated so the next reader can overturn it with evidence rather than taste:
  (a) moat_tau's DIRECT consumer is wealth_tier leg C, which is price-blind BY DESIGN and
      documented as "a description, not a recommendation" -- describing the 5-year margin trend
      IS its job, and the tile's label claims exactly that span.
  (b) the only return-predictive consumer is trajectory_score -> win_rate_proxy -> Kelly, and it
      averages moat_tau with roce_tau/revenue_tau/pat_tau, so the IC cost arrives DILUTED 4:1
      while the span gain is undiluted.
  (c) one 28-day window with a CI straddling zero does not meet this repo's own bar (same sign
      in BOTH windows), and window 1 is unavailable.
DECEMBER'S RULE, PRE-DECLARED: if the new ladder loses again on a second window, the fix is NOT
to revert this measurement -- it is to give trajectory_score a shorter-span margin tau and leave
the 5Y one where the label says 5Y. Ladder A is the candidate; it won here but it was the best
of seven variants on a single window, which is a selection effect, not a result.

SETTLED BY THE SAME SWEEP, in BOTH windows: THE QUARTER RUNG IS LOAD-BEARING. Dropping
opm_latest_q costs -0.0251 (W1) and -0.0246 (W2) -- and W1 uses only columns native to the June
snapshot, so that half needs no backfill and no fiscal-year assumption. docs/known-issues.md had
this open as "test whether dropping opm_latest_q improves trajectory's predictive value"; the
answer is no, and the question is closed.
  * 568 wealth tiers (20.9%) were re-measured on the switch — CLIMBING★ 336 → 331 (50 in / 46 out),
    N/A 369 → 441: the 154 stocks that LOST a tau are shallow-history firms (91.6% lack a 5-year
    PAT vs 17.3% of the universe; a third listed under two years ago) — a company without five
    years of margins gets no five-year margin trend. Unverifiable is not passed, nor condemned.

Run with: pytest tests/test_moat_tau_quantization.py -v
"""

import contextlib
import io as _io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

_ENGINE = os.path.join(os.path.dirname(__file__), "..", "core", "data_engine.py")

# The ordered OPM level ladder moat_tau is built from, OLDEST FIRST. Changing this list changes the
# step size, and therefore changes what WEALTH_TAU_CONF means -- which is the whole point of this file.
TAU_COLS = ["opm_5yb", "opm_3yb", "opm_1yb", "opm", "opm_latest_q"]
_EPS = 1e-9


@pytest.fixture(scope="module")
def live():
    from core import fetch_and_clean_data
    with contextlib.redirect_stdout(_io.StringIO()):
        return fetch_and_clean_data("local")


def _tau(live):
    return pd.to_numeric(live["moat_tau"], errors="coerce")


# -- 1. The ladder, pinned at the source ------------------------------------------------
def test_moat_tau_is_built_from_exactly_these_five_levels_oldest_first():
    """Order is load-bearing (newer − older is the sign), and length is load-bearing (the quantum
    is 1/pairs). Both are the contract."""
    src = _io.open(_ENGINE, encoding="utf-8").read()
    i = src.index('df["moat_tau"] = _kendall_tau_cols(')
    call = src[i:src.index(")", src.index("[", i))]
    found = [c.strip().strip('"]') for c in call[call.rindex("[") + 1:].split(",") if c.strip()]
    assert found == TAU_COLS, (
        f"the moat_tau ladder is {found}, expected {TAU_COLS}. A different length changes the "
        f"quantum and silently redefines WEALTH_TAU_CONF; a different order flips signs. Re-derive "
        f"every number in this file before changing it."
    )
    assert "opm_med_5y" not in call, "the 5Y median is a stand-in, not a point in time — it left the ladder on 2026-09-19"


# -- 2. The attainable value set ---------------------------------------------------------
def test_tau_takes_only_tenths_and_sixths(live):
    """5 columns → 10 pairs on the full ladder; one missing level leaves exactly min_pairs = 6
    (the four remaining columns); two missing leaves 3 < 6 → NaN. So every value is k/10 or k/6."""
    vals = sorted(_tau(live).dropna().unique())
    bad = [v for v in vals if not (np.isclose(v * 10, round(v * 10), atol=_EPS)
                                   or np.isclose(v * 6, round(v * 6), atol=_EPS))]
    assert not bad, f"values that are neither tenths nor sixths: {bad}"
    assert 20 <= len(vals) <= 31, f"{len(vals)} distinct values (13 before the ladder grew; 31 is the ceiling)"


def test_most_rows_sit_on_the_full_ten_pair_ladder(live):
    full = live[TAU_COLS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    assert full.mean() >= 0.70, f"only {full.mean():.1%} of rows carry all five levels"
    t = _tau(live)
    assert t.isna().mean() < 0.12, f"moat_tau is {t.isna().mean():.1%} NaN (8.4% on 2026-09-19)"


# -- 3. What the gate actually means ------------------------------------------------------
def test_the_0_25_gate_is_unreachable_and_means_0_3_on_tenths_and_one_third_on_sixths(live):
    from verdict_engine import WEALTH_TAU_CONF, WEALTH_TAU_FADE
    t = _tau(live)
    assert not np.isclose(t, WEALTH_TAU_CONF, atol=_EPS).any(), "0.25 became attainable — re-derive"
    assert not np.isclose(t, WEALTH_TAU_FADE, atol=_EPS).any()
    # 0.3 (tenths) and 1/3 (sixths) both clear the cut; 0.2 and 1/6 both miss it — one rule.
    assert ((t >= WEALTH_TAU_CONF) == (t >= 0.3 - _EPS)).all()
    assert ((t <= WEALTH_TAU_FADE) == (t <= -0.3 + _EPS)).all()


def test_the_quantum_still_exceeds_the_distance_to_the_gate(live):
    """The fragility survives in a finer form: one step is 0.1 while the gap from 0.2 to the 0.25
    cut is 0.05. A single pair flip still crosses the gate — only fewer pairs now depend on any one
    observation (4 of 10, not 3 of 6)."""
    t = _tau(live).dropna()
    full_vals = sorted(v for v in t.unique() if np.isclose(v * 10, round(v * 10), atol=_EPS))
    step = float(min(np.diff(full_vals)))
    assert np.isclose(step, 0.1), f"the full-ladder quantum is {step}, expected 0.1"
    from verdict_engine import WEALTH_TAU_CONF
    below = max(v for v in full_vals if v < WEALTH_TAU_CONF)
    assert step > (WEALTH_TAU_CONF - below)
    n = len(TAU_COLS)
    assert (n - 1) / (n * (n - 1) // 2) == pytest.approx(0.4), "one observation touches 4 of 10 pairs"


# -- 4. The exposure, so December has a baseline -----------------------------------------
def test_the_one_step_population_is_still_about_a_quarter_of_the_universe(live):
    """Stocks one pair flip from C or FADE: ±0.2 / ±0.3 on the full ladder, ±1/6 / ±1/3 on the
    six-pair rows. 604 stocks, 24.3%, on 2026-09-19 (692 / 25.5% under the four-point ladder)."""
    tv = _tau(live).to_numpy(dtype=float)
    one_step = np.zeros(len(tv), dtype=bool)
    for q in (0.2, 0.3, -0.2, -0.3, 1 / 6, 1 / 3, -1 / 6, -1 / 3):
        one_step |= np.isclose(tv, q, rtol=0, atol=_EPS)
    share = one_step.sum() / np.isfinite(tv).sum()
    assert 0.15 < share < 0.35, (
        f"{share:.1%} of rated stocks are one step from flipping C or FADE (24.3% at the switch). "
        f"A large move means the OPM distribution shifted -- re-measure before trusting the churn "
        f"analysis in docs/known-issues.md"
    )


def test_tau_is_not_dead_and_does_not_pile_into_one_value(live):
    t = _tau(live)
    top = t.value_counts(normalize=True).iloc[0]
    assert top < 0.40, f"a single tau value holds {top:.1%} of the universe"


# -- 5. The ladder measures the SPAN its label claims -------------------------------------
#    NOT "the ladder is a better ruler" -- that is what the first draft of this file asserted and
#    it does not follow from these numbers (see correction 1 in the module docstring: the witness
#    is span-matching, and the forward test actually favours the ladder this one replaced).
def _npm_witness(live, spans):
    """Kendall tau over the PAT/revenue-derived NPM of the given fiscal years -- source columns
    the OPM ladder never reads, so it is an independent reading of margin DIRECTION. `spans` is
    the ordered oldest->newest list of column suffixes."""
    def _num(c):
        return pd.to_numeric(live[c], errors="coerce")
    pts = []
    for s in spans:
        rev, pat = _num(f"revenue{s}"), _num(f"pat{s}")
        pts.append(pd.Series(np.where(rev > 0, pat / rev, np.nan), index=live.index))
    n = len(pts)
    net = pd.Series(0.0, index=live.index)
    valid = pd.Series(0, index=live.index)
    for i in range(n):
        for j in range(i + 1, n):
            both = pts[i].notna() & pts[j].notna()
            net = net + np.where(both, np.sign(pts[j] - pts[i]), 0.0)
            valid = valid + both.astype(int)
    min_pairs = (n * (n - 1) // 2) // 2 + 1
    return pd.Series(np.where(valid >= min_pairs, net / valid.replace(0, np.nan), np.nan), index=live.index)


def _rho(a, b):
    m = a.notna() & b.notna()
    return float(a[m].rank().corr(b[m].rank())), int(m.sum())    # Spearman = Pearson on ranks (no scipy)


def test_the_ladder_tracks_a_five_year_margin_witness(live):
    """FLOOR, not a superiority claim: the ladder must still track an independent five-year
    margin trend. The median-stand-in ladder scores +0.353 here, so this floor is what fails if
    anyone puts opm_med_5y back in place of the true oldest level."""
    rho, n = _rho(_npm_witness(live, ("_5yb", "_3yb", "_1yb", "")), _tau(live))
    assert n > 1500
    assert rho >= 0.45, (
        f"moat_tau vs the 5Y NPM witness: ρ {rho:+.3f} (this ladder measured +0.563; the median "
        f"stand-in it replaced, +0.353)"
    )


def test_the_ladder_is_a_five_year_instrument_not_a_recent_trend_one(live):
    """THE SPAN PIN, and the one that carries the argument for this ladder over its rivals.

    The tile reads "Margin Trend (5Y Tau)" and verdict_engine's leg C is "the 5-year margin
    spine", so the column is REQUIRED to be a five-year instrument. A ladder that tracks a
    two-year witness better than a five-year one is a recent-trend measure wearing a 5Y label,
    whatever its other merits. Measured 2026-09-19, this is the assertion that separates the
    three candidates -- it is TRUE only for the shipped ladder:
        shipped [5yb,3yb,1yb,opm,q]  5Y +0.563 > 2Y +0.315   PASS
        old     [med,1yb,opm,q]      5Y +0.353 < 2Y +0.447   FAIL (a recent-trend measure)
        A       [3yb,1yb,opm,q]      5Y +0.315 < 2Y +0.440   FAIL (a ~3.5-year instrument)
    So reverting to either rival breaks this test, and it breaks for the RIGHT reason -- the
    label would no longer describe the column. If leg C is ever respecified to a shorter span,
    change the tile, the verdict_engine comment and this test together, in one edit."""
    t = _tau(live)
    rho_5y, n5 = _rho(_npm_witness(live, ("_5yb", "_3yb", "_1yb", "")), t)
    rho_2y, n2 = _rho(_npm_witness(live, ("_1yb", "")), t)
    assert n5 > 1500 and n2 > 1500
    assert rho_5y > rho_2y, (
        f"moat_tau tracks the 2-year margin witness ({rho_2y:+.3f}) at least as well as the "
        f"5-year one ({rho_5y:+.3f}) -- it is no longer the five-year instrument its tile and "
        f"verdict_engine leg C both claim. Either restore a five-year span or respecify leg C."
    )
