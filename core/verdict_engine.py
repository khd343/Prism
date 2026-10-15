"""
Verdict Engine — display-only decision-synthesis layer.
=======================================================
Fuses the engine's already-computed sub-verdicts (conviction_tier, mauboussin_ev_verdict,
corporate_class, buy_zone_label) and the six quality/risk axes into ONE coherent top-line
verdict: a Direction, a Conviction strength, a Confidence, a plain-English narrative, and the
single most important risk.

PIPELINE POSITION — STEP 4 (after apply_forensic_penalty in app.get_scored_data).
  Reads the POST-penalty composite_score / conviction_tier (apply_forensic_penalty re-derives the
  tier from the penalized score, so these are consistent here — verified 2026-06-14).

INVARIANTS:
  * ZERO new scoring. Every output column is a label/string DERIVED from existing columns.
    composite_score and all scoring columns are read-only here — never reassigned.
  * Fully vectorized (np.select / np.where) — no .apply, no row loops. get_scored_data is
    UNcached (<0.5s/dropdown), so this must stay O(rows) array ops.
  * Defensive: every input via _col() so a missing column degrades gracefully (never KeyErrors).
  * Asymmetric vetoes (consistent with the engine's forensic-multiplier philosophy): a verdict can
    only be capped DOWNWARD by a hard risk, never upgraded. The forensic multiplier already lowered
    the tier, so the 🚨 veto is a SAFETY CAP for the rare high-score-despite-sharp-practices case —
    not a double penalty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Wealth-tier grammar constants (see the wealth_* block at the end of compute_verdict) ──────
# Derived and stress-tested 2026-08-27 on the user's watchlists, the engine's complete BUY list
# (18 stocks) and its complete WATCH tier (126) — ~250 names across three dry-runs.
WEALTH_VEL_MIN   = 0.5     # Vel% at/above this = excess return MATERIALLY improving (B)
WEALTH_TAU_CONF  = 0.25    # tau at/above this = margin spine confirms (C)
WEALTH_TAU_FADE  = -0.25   # tau at/below this = margins fading — caps the tier at UNEVEN
WEALTH_WARN_FLAGS = 8      # red flags at/above this raise the ⚠ marker (engine veto sits at 10)

# The ladder's words, strongest first; "N/A" (an input is missing) sits off the ladder, never a tier.
# RENAMED 2026-10-15 from BUY★ / BUY / WATCH★ / WATCH / AVOID, the user's call after the live sheet
# showed what the action words said: BUY on 518 stocks (19% of the market), 85% of BUY★ FLAWED on
# Soundness, AVOID on Colgate at EP% +71.7. The tier measures which WAY value creation is moving and
# never sees the price, so it speaks DIRECTION words. Same rules, same thresholds, the same stocks in
# every tier (tests/test_wealth_tier.py replays the old grammar through the map). Every word was checked
# against every label on screen: RISING nests inside the EP Power Curve's "Not Rising", IMPROVING is
# that curve's own label, TURNAROUND is ROE Turnaround's and Lynch's, BUILDING is in industry names.
WEALTH_TIERS = ("CLIMBING★", "CLIMBING", "RECOVERING", "UNEVEN", "STALLED")
WEALTH_NA = "N/A"
# Old word → today's, for snapshots and exports written before the rename (tools/validate.py reads
# them through it). One-to-one in ladder order — the tier sets are unchanged, only their names.
WEALTH_TIER_RENAMED = dict(zip(("BUY★", "BUY", "WATCH★", "WATCH", "AVOID"), WEALTH_TIERS))


def compute_verdict(df: pd.DataFrame) -> pd.DataFrame:
    """Materialize the verdict_* display columns. Pure synthesis of existing signals."""
    df = df.copy()
    idx = df.index

    def _col(name: str, default):
        """Defensive column access — absent column → constant Series (never KeyError)."""
        return df[name] if name in df.columns else pd.Series(default, index=idx)

    # ── Inputs (all verified present 2026-06-14; _col() keeps it crash-proof regardless) ──
    tier    = _col("conviction_tier", 5).fillna(5).astype(float)
    rflags  = _col("red_flag_count", 0).fillna(0).astype(float)
    cclass  = _col("corporate_class", "").astype(str)
    cov     = _col("data_coverage_pct", 100.0).fillna(0.0)
    q       = _col("quality_score", np.nan)      # overall quality (narrative only — see note)
    g       = _col("growth_score", np.nan)
    eer     = _col("expected_excess_return", np.nan)
    pe      = _col("pe", np.nan)
    fair_pe = _col("fair_pe_qglp", np.nan)
    bz      = _col("buy_zone_label", "").astype(str)
    govmult = _col("governance_risk_multiplier", 1.0).fillna(1.0)
    # Orthogonal scorecard axes — the engine's INDEPENDENT sub-scores, NOT the quality_score
    # composite (which already contains growth 22% + valuation 10% → showing Quality+Growth+Value
    # would triple-count). Polarity verified 2026-06-14: all high=good (valuation_score corr w/ PE
    # is −0.14 → high = cheap = good).
    moat    = _col("moat_score", np.nan)
    valn    = _col("valuation_score", np.nan)
    bal     = _col("balance_sheet_score", np.nan)
    # Evidence honesty (2026-08-23): the rank helpers neutral-fill missing inputs to 50, so a
    # stock with ZERO moat/growth signals still carries a real-looking score. Mask the axis to
    # NaN when its evidence count (from scoring_engine, same list the scorer iterates) is zero —
    # _band_num then renders the pill as ⚪ N/A instead of a fabricated 🟡 50. `== 0` is False
    # for NaN, so frames WITHOUT the count columns (old snapshots, minimal fixtures) keep the
    # pre-2026-08-23 behavior — the mask can only default OFF, never ON.
    moat = moat.mask(_col("moat_signals_available", np.nan) == 0)
    g    = g.mask(_col("growth_signals_available", np.nan) == 0)
    valn = valn.mask(_col("valuation_signals_available", np.nan) == 0)
    bal  = bal.mask(_col("balance_signals_available", np.nan) == 0)

    # ── Hard-risk veto masks (cap downward only) ──
    # CALIBRATED 2026-06-14: forensic_label is "🚨" for 98.6% of the universe (only 29 "Clean") →
    # useless as a veto. red_flag_count median is ~4-5 → "several flags" is the NORM. So the AVOID
    # veto uses genuinely SELECTIVE severe signals; the forensic multiplier already discounted
    # moderate-flag stocks into lower tiers (the 32 Tier-1-2 stocks are already clean by construction).
    fscore        = _col("forensic_score", 100.0).fillna(100.0)
    schilit_fail  = _col("schilit_pass", 1).fillna(1).astype(float) == 0   # Schilit checker hard-fail
    forensic_veto = (fscore < 50.0) | (rflags >= 10)        # severe accounting floor / extreme flags
    gruesome_veto = cclass.str.contains("GRUESOME", case=False)
    gov_veto      = govmult < 0.85               # governance risk multiplier materially below 1.0
    timing_poor   = bz.str.contains("Below|Avoid|Overextend|Stop", case=False)

    # ── Direction: base from post-penalty tier, then asymmetric caps ──
    # VOCABULARY RENAMED 2026-08-27: BUY/WATCH/AVOID → SOUND/MIXED/FLAWED. Two reasons:
    # (1) COLLISION — the wealth tier (below) then spoke action words (BUY★/BUY/WATCH★/WATCH/
    #     AVOID), so the same words meant two different things in adjacent columns: the
    #     cf_triangle "Perfect — Buy Zone" defect class, rebuilt at vocabulary level.
    # (2) HONESTY — with 18 BUYs against 93% AVOID this was never a buy call; it is a
    #     qualification gate (quality + forensics + valuation + timing), so it now speaks
    #     CONDITION words. (Since 2026-10-15 the wealth tier speaks DIRECTION words — see
    #     WEALTH_TIERS — so no layer issues an action word.)
    # Alternatives audited and rejected: PASS/FAIL & QUALIFIED (conflate with gate_pass /
    # "% Qualify"), STRONG/WEAK (verdict_strength's and a tier label's words), NEUTRAL
    # (smart_money_flow + capital phase), UNSOUND (SOUND ⊂ UNSOUND substring nesting — the bug
    # class hit four times on 2026-08-27 alone). Old snapshots keep the old strings;
    # tools/validate.py maps them on load.
    base = np.select([tier <= 2, tier == 3], ["SOUND", "MIXED"], default="FLAWED")
    is_buy = pd.Series(base == "SOUND", index=idx)   # the vetoes cap the affirmative call
    _soft  = (schilit_fail | gov_veto | timing_poor) & is_buy   # soft downgrade BUY → WATCH
    direction = np.where(
        forensic_veto | gruesome_veto, "FLAWED",
        np.where(_soft, "MIXED", base),
    )
    df["verdict_direction"] = direction

    df["verdict_emoji"] = np.select(
        [direction == "SOUND", direction == "MIXED"], ["🟢", "🟡"], default="🔴"
    )

    # ── Conviction strength (from tier) ──
    # DISPLAY-RETIRED 2026-08-27: a measured 1:1 map of conviction_tier (zero added
    # information), it was shown in the hero beside the tier badge and the score — the same
    # fact three times. The column REMAINS (snapshot-schema stability; exports) but no UI
    # surface renders it. If you are about to display it, display the tier instead.
    df["verdict_strength"] = pd.Series(tier, index=idx).map(
        {1.0: "HIGH CONVICTION", 2.0: "STRONG", 3.0: "EMERGING", 4.0: "SPECULATIVE", 5.0: "WEAK"}
    ).fillna("WEAK")

    # ── Confidence (kept SEPARATE from the score — a high score on thin data is flagged) ──
    df["verdict_confidence"] = np.select(
        [cov >= 80, cov >= 60, cov >= 40], ["High", "Medium", "Low"], default="Very Low"
    )

    # ── Six axis pills (compact display strings) ──
    def _band_num(s: pd.Series, hi: float, mid: float) -> np.ndarray:
        v = s.fillna(-1.0)
        return np.select([v < 0, v >= hi, v >= mid], ["⚪", "🟢", "🟡"], default="🔴")

    def _pill_num(label: str, s: pd.Series, hi: float, mid: float):
        # Vectorized (2026-08-28): the original zipped a Python list-comprehension over rows +
        # a lambda .map — the one row loop in a file whose own docstring says "no row loops".
        # .round(0) and "{:.0f}" share IEEE half-even rounding, so the strings are identical.
        dot = _band_num(s, hi, mid)
        val = s.round(0).fillna(0).astype(int).astype(str).where(s.notna(), "N/A")
        return label + " " + pd.Series(dot, index=idx) + " " + val

    # 4 numeric axes (0-100 sub-scores, 🟢≥60 🟡≥40) + 2 categorical (Governance/Forensics —
    # cleaner than a compressed bonus score). Orthogonal: Moat·Growth·Valuation·Balance·Govern·Forensics.
    df["verdict_axis_moat"]      = _pill_num("Moat",      moat, 60, 40)
    df["verdict_axis_growth"]    = _pill_num("Growth",    g,    60, 40)
    df["verdict_axis_valuation"] = _pill_num("Valuation", valn, 60, 40)
    df["verdict_axis_balance"]   = _pill_num("Balance",   bal,  55, 38)
    # Zero flags on Very-Low coverage (the existing <40 confidence band — no new threshold) means
    # the checks could not RUN, not that they passed: "unverifiable is not passed" (§5). Fired
    # flags ARE evidence, so Flagged/Watch outrank Unverified; only the Clean claim needs coverage.
    _forensics_unverified = (rflags == 0) & (cov < 40.0)
    df["verdict_axis_forensics"] = np.where(
        forensic_veto | schilit_fail, "Forensics 🔴 Flagged",
        np.where(rflags >= 5, "Forensics 🟡 Watch",
        np.where(_forensics_unverified, "Forensics ⚪ Unverified", "Forensics 🟢 Clean")),
    )
    # Governance needs no evidence mask (measured 2026-08-23): zero live rows have all four
    # promoter columns missing — the Shareholdings sheet populates them universally. Re-measure
    # if that sheet's coverage ever changes.
    df["verdict_axis_governance"] = np.select(
        [govmult < 0.85, govmult < 1.0], ["Govern 🔴 Risk", "Govern 🟡 Caution"],
        default="Govern 🟢 Safe",
    )

    # ── Single most important risk (one line, only the top one) ──
    df["verdict_top_risk"] = np.select(
        [forensic_veto, gruesome_veto, schilit_fail, gov_veto, timing_poor, cov < 50],
        [
            "🚨 Severe forensic / accounting-quality flags",
            "💀 Value-destroying capital allocation",
            "🕵️ Schilit forensic checker flags",
            "⚠️ Governance risk (pledge/dilution)",
            "⏳ Poor entry timing — wait for a base",
            "🔍 Thin data — verdict tentative",
        ],
        default="",
    )

    # ── Narrative: deterministic np.select over decision archetypes ──
    # Unambiguous helper signals (no dependence on any score's polarity sign convention):
    cheap  = (pe.notna() & fair_pe.notna() & (pe < fair_pe))                       # price below quality-fair PE
    pricey = (pe.notna() & fair_pe.notna() & (pe > fair_pe * 1.30))               # >30% above fair PE
    q_hi   = q.fillna(0) >= 65
    g_hi   = g.fillna(0) >= 60
    clean  = ~(forensic_veto | gruesome_veto | schilit_fail)
    great  = cclass.str.contains("GREAT", case=False)
    eer_pos = eer.fillna(0) >= 5.0
    # DIRECTION-AWARE so the narrative can NEVER contradict the direction. The positive archetypes
    # describe a good FUNDAMENTAL setup (quality/growth/cheap), but the direction reads the post-
    # penalty composite — so a cheap-and-clean stock whose composite still lands in AVOID (momentum /
    # risk-adjusted quality dragging it — a classic value trap, e.g. Sarda Energy: q71 g66 cheap but
    # composite 42 / Stage-4) must NOT read "high-conviction core holding". AVOID is matched FIRST so
    # it can never fall through to a buy-toned line; the strongest BUY phrasing is reserved for BUYs.
    dir_avoid = pd.Series(direction == "FLAWED", index=idx)
    dir_buy   = pd.Series(direction == "SOUND",  index=idx)

    df["verdict_narrative"] = np.select(
        [
            forensic_veto,
            gruesome_veto,
            schilit_fail & (tier <= 2),
            # AVOID matched up-front (after the hard-reason vetoes) → never a buy-toned narrative
            dir_avoid & q_hi & (cheap | eer_pos),
            dir_avoid,
            q_hi & g_hi & pricey & clean,
            dir_buy & q_hi & g_hi & (cheap | eer_pos) & clean,   # strongest phrasing: BUY only
            q_hi & g_hi & (cheap | eer_pos) & clean,             # same setup, non-BUY → softer
            q_hi & g_hi & clean,
            (cheap | eer_pos) & ~q_hi & clean,
            q_hi & ~g_hi & clean,
            great & clean,
            cov < 50,
        ],
        [
            "Severe forensic / accounting-quality flags override the thesis — avoid until the accounts are clean.",
            "Capital allocation destroys value (Gruesome) — avoid regardless of how cheap it looks.",
            "Quality screens well, but Schilit forensic checkers flag it — verify the accounts before buying.",
            "Cheap with decent quality on the screens — but the overall score is weak enough to land FLAWED; treat it as a value-trap watch, not a buy.",
            "Overall score lands FLAWED — the weak axes outweigh the strengths; pass until they improve.",
            "Elite franchise, but priced for perfection — wonderful business, demanding price; wait for a pullback.",
            "Elite compounder at a reasonable price — high-conviction core holding.",
            "A high-quality compounder trading at a fair price — quality and value aligned.",
            "Strong quality and growth, fairly valued — a solid compounder.",
            "Statistically cheap, but quality is thin — verify it isn't a value trap before buying.",
            "Quality intact but growth is decelerating — watch the next few prints before adding.",
            "Disciplined capital allocator (Great) — durable compounding, watch the entry price.",
            "Thin data coverage — treat this verdict as tentative until more inputs report.",
        ],
        default="No decisive edge on any axis right now — pass and revisit on a re-rate or result.",
    )

    # ══ WEALTH TIER — the change-lens verdict (2026-08-27) ═══════════════════════════════════
    # Answers a DIFFERENT question from verdict_direction. The direction asks "is this business
    # good and safe now?" (level + forensics + valuation). The wealth tier asks "is it becoming
    # more valuable?" — three clocks, nothing else:
    #     A  wealth_ep_pct  > 0                earning above the cost of equity
    #        (EP/reserves = ROE − CoE, EXACTLY: the MOSL Wealth Creation studies' own recurring
    #         engine — ROCE expansion + earnings growth — re-derived independently and normalized
    #         so a ₹190 Cr business and a ₹2,025 Cr one are comparable. Absolute EP is size-biased:
    #         it ranked Sarda's +₹251 Cr improvement above far larger PERCENTAGE improvers.)
    #     B  wealth_vel_pct >= WEALTH_VEL_MIN  excess return materially improving this year
    #     C  moat_tau       >= WEALTH_TAU_CONF the 5-year margin spine confirms it
    #     fading: tau <= WEALTH_TAU_FADE caps everything at UNEVEN — improvement without a margin
    #             spine stays on probation (Varroc: Vel +7.5 but tau −0.83).
    #
    #     CLIMBING★  A·B·C   CLIMBING  A·B(tau flat)   RECOVERING  ¬A·B·C (the confirmed turnaround)
    #     UNEVEN  the clocks disagree   STALLED  nothing improving   N/A  unverifiable
    #     (BUY★ / BUY / WATCH★ / WATCH / AVOID until 2026-10-15 — WEALTH_TIER_RENAMED.)
    #
    # PRICE-BLIND BY DESIGN and the UI must say so: CLIMBING★ describes the business — its value
    # creation is rising and confirmed — never the price. A compressed DESCRIPTION, never a
    # recommendation; the decision stays with the user.
    #
    # ⚠ (wealth_warn) IS DELIBERATELY NOT BLENDED INTO THE TIER. These three clocks cannot see
    # forensics — measured: 16 of 26 CLIMBING★ on one live watchlist carried 8+ red flags — so the
    # risk marker rides beside the tier and never alters it. Blending would hide exactly the
    # tension the reader needs to see (the cf_triangle lesson).
    #
    # STRICT N/A RULING: any missing input (reserves ≤ 0/absent, EP, velocity or tau NaN) yields
    # N/A, never a tier — "unverifiable is not passed" (§5), and equally never condemned. This
    # buries e.g. Lohia Corp (EP% +24.6, Vel% +10.9, tau missing) as N/A; accepted deliberately —
    # the row still shows the strong numbers.
    #
    # CLASSIFIED ON DISPLAYED PRECISION (the _shown rule, tests/test_display_threshold_
    # consistency.py): the UI prints EP%/Vel% at 1dp and tau at 2dp, so the grammar rounds to
    # those BEFORE comparing — two stocks printing the same number must land in the same tier.
    # ONE-BASIS DENOMINATOR (2026-08-28; was net_worth). economic_profit is built on RESERVES
    # (data_engine's _ep_equity = reserves.where(> 0)); net_worth is market_cap ÷ P/B — a
    # DIFFERENT equity definition, with a silent reserves fallback on 34 rows. Mixing the two
    # made the "EP% = ROE − CoE" identity only approximate: median shrink ×0.962 but ratio
    # 0.004–14.0 in the tails (GKW displayed EP% −77.7 against a true −12.1; Swan Defence was
    # 256pp off), and 7 stocks failed the Vel% bar on the shrink alone. Dividing by the SAME
    # base EP is built on makes wealth_ep_pct ≡ ROE − CoE exact by construction — pinned by
    # test_ep_pct_is_exactly_roe_minus_coe, which also catches any future drift between this
    # divisor and data_engine's EP equity base. Migration measured: 8 tiers changed, none into or out
    # of CLIMBING/CLIMBING★.
    _w_eq  = _col("reserves", np.nan).astype(float).to_numpy()
    _w_ep  = _col("economic_profit", np.nan).astype(float).to_numpy()
    _w_vel = _col("economic_profit_velocity", np.nan).astype(float).to_numpy()
    _w_tau = _col("moat_tau", np.nan).astype(float).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        _w_eq_ok  = np.isfinite(_w_eq) & (_w_eq > 0)
        _ep_pct   = np.where(_w_eq_ok, _w_ep / _w_eq * 100.0, np.nan)
        _vel_pct  = np.where(_w_eq_ok, _w_vel / _w_eq * 100.0, np.nan)
        _ep_s, _vel_s, _tau_s = np.round(_ep_pct, 1), np.round(_vel_pct, 1), np.round(_w_tau, 2)
        _known = np.isfinite(_ep_s) & np.isfinite(_vel_s) & np.isfinite(_tau_s)
        _A    = _known & (_ep_s > 0.0)
        _B    = _known & (_vel_s >= WEALTH_VEL_MIN)
        _C    = _known & (_tau_s >= WEALTH_TAU_CONF)
        _FADE = _known & (_tau_s <= WEALTH_TAU_FADE)
    df["wealth_ep_pct"]  = _ep_pct
    df["wealth_vel_pct"] = _vel_pct
    # Condition ORDER is load-bearing: N/A first (unverifiable outranks everything), then the
    # fading cap (it overrides even A·B·C — C and FADE are mutually exclusive, but A·B·fading
    # must land on UNEVEN/STALLED, not CLIMBING), then the tiers from strongest down.
    df["wealth_tier"] = np.select(
        [~_known,
         _FADE & _B,
         _FADE,
         _A & _B & _C,
         _A & _B,
         _B & _C,
         (_C & ~_B) | (_B & ~_A)],
        [WEALTH_NA, "UNEVEN", "STALLED", "CLIMBING★", "CLIMBING", "RECOVERING", "UNEVEN"],
        default="STALLED",
    )
    df["wealth_warn"] = ((rflags >= WEALTH_WARN_FLAGS) | schilit_fail).astype(int)

    return df
