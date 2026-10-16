"""
test_breakout_mode.py
=====================
Contract: the "Breakout" Analysis Mode — quality, momentum and breakout in EQUAL THIRDS — was
THE DEFAULT (config.DEFAULT_ANALYSIS_MODE) from 2026-09-19 to 2026-10-15, when the user made
Momentum + Breakout the default (tests/test_momentum_breakout_mode.py); every other mode is
byte-identical to before the third leg existed.

HOW IT GOT HERE (2026-09-19, all in one day). The user proposed blending the composite from
quality + momentum + breakout. Measured on both forward windows in EXACTLY the form the engine
computes (governance 15%, every framework boost, the governance shield and the forensic multiplier
all kept — NOT the bare rank blend, which flatters it):

    form                                   W1 06-17→08-22   W2 08-22→09-19
    Hybrid composite (current default)          +0.134           +0.034
    Q+M+B, engine form                          +0.148           +0.094      <- this mode
    Q+B,   engine form                          +0.148           +0.066
    Q+M,   engine form                          +0.139           +0.065
    quality_score alone                         +0.116           -0.025

It beats Hybrid in both windows and beats Q+M and Q+B; Q+B is 0.975-correlated with it.

PROMOTED TO DEFAULT THE SAME DAY (2026-09-19), which reversed the paragraph that stood here. The
original text argued it should wait for December because 28- and 66-day windows favour price
strength by construction. Two things changed that: (a) the 94-DAY window (06-17 -> 09-19) was
added — the longest we hold, 3x the short one — and the theory that the best quality weight RISES
with horizon got no support from it (its top-5 weightings carry 0% quality, same as at 28 days),
so the argument for waiting was a prior with no data behind it; and (b) a 66-point sweep of the
whole Q/M/B simplex showed Hybrid ranks #56/#60/#60 of 66 across the three windows — not merely
beaten but near the BOTTOM, with 14 of 21 quality-led weightings beating it in all three. The
default was indefensible on every horizon that can be measured.

WHAT IS KNOWINGLY GIVEN UP, since it is a hedge and not a measurement: breakout_score is already
40% of momentum_score, so the default now puts ~2/3 of the composite on price strength. The
remaining third keeps the fundamental work in the ranking against the horizon nobody can see yet
(the grid optimum wants 0% quality on ALL THREE windows — following it would not be PRISM).
DECEMBER IS A REVIEW, NOT A CORONATION: on the first 6-month window, a quality-led weighting that
beats thirds moves the default back. tools/validate.py reconstructs EVERY mode on EVERY snapshot
(composite_mode_*), so that comparison is apples-to-apples whatever mode each was taken under.

Run with: pytest tests/test_breakout_mode.py -v
"""

import contextlib
import io
import os
import re
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
_APP = _ROOT / "app.py"


# ── 1. the mode exists and is well-formed ───────────────────────────────────────────────
def test_breakout_mode_is_equal_thirds_and_well_formed():
    m = C.ANALYSIS_MODES["Breakout"]
    for k in ("fundamental_w", "momentum_w", "breakout_w"):
        assert abs(m[k] - 1 / 3) < 1e-12, f"{k} is {m[k]}, expected exactly one third"
    assert m["label"].strip() and m["description"].strip()
    # CAUGHT BY THE BROWSER, NOT BY A TEST: this caption renders under the selector and still
    # said "the default stays Hybrid until December's longer window agrees" AFTER the mode had
    # become the default — a sentence that contradicted the control directly above it.
    _d = m["description"].lower()
    assert "default stays hybrid" not in _d, "the caption contradicts the live default"
    assert ("default" in _d) == (C.DEFAULT_ANALYSIS_MODE == "Breakout"), (
        "the caption must claim default status exactly when the mode actually holds it"
    )
    # The two allowed_profiles assertions that stood here went with the key on 2026-09-20 —
    # no mode carries a profile list any more, because the QGLP screen is a constant. That the
    # key is gone EVERYWHERE is pinned once, in test_qglp_profile_fixed.py, rather than per mode.
    assert "allowed_profiles" not in m, "the Breakout mode must not reintroduce a profile list"


def test_only_the_declared_modes_carry_a_third_leg():
    """The other modes are unchanged: no breakout_w key at all (not 0.0 — the leg must not exist
    there, so nothing can quietly grow one). Repointed 2026-10-10, not loosened: the fifth mode,
    Momentum + Breakout (tests/test_momentum_breakout_mode.py), carries the leg by design, and
    the set is still exact, so any OTHER mode growing one fails here."""
    carriers = sorted(n for n, m in C.ANALYSIS_MODES.items() if "breakout_w" in m)
    assert carriers == ["Breakout", "MomentumBreakout"], f"third-leg carriers: {carriers}"


def test_the_default_mode_is_momentum_breakout_and_lives_in_exactly_one_place():
    """THE DIVERGENCE GUARD, and the reason this is a constant rather than four literals: before
    2026-09-19 the app carried "Hybrid" and so did run_scoring_pipeline, run_full_scoring,
    /census and /verify — independently. Changing the app alone would have left every snapshot,
    census and verify run scored under a mode the user no longer sees, with nothing failing.
    The default was Breakout from 2026-09-19 and is Momentum + Breakout since 2026-10-15, the
    user's decision (tests/test_momentum_breakout_mode.py)."""
    import inspect
    import core
    from scoring_engine import run_full_scoring
    assert C.DEFAULT_ANALYSIS_MODE == "MomentumBreakout"
    assert C.DEFAULT_ANALYSIS_MODE in C.ANALYSIS_MODES, "the default names a mode that does not exist"
    for fn in (core.run_scoring_pipeline, run_full_scoring):
        got = inspect.signature(fn).parameters["analysis_mode"].default
        assert got == C.DEFAULT_ANALYSIS_MODE, (
            f"{fn.__name__} defaults to {got!r}, not the config constant — the app and the engine "
            f"can now score under different modes and nothing will say so"
        )
    src = _APP.read_text(encoding="utf-8")
    assert 'st.session_state.setdefault("cfg_mode", DEFAULT_ANALYSIS_MODE)' in src
    assert 'setdefault("cfg_mode", "' not in src, "app.py re-hardcoded a default mode literal"


def test_no_tool_scores_under_a_hardcoded_mode():
    """/census and /verify exist to describe the LIVE engine. If they pin a mode literal they
    describe a different one, and their fire rates stop matching what the user sees."""
    for rel in ("tools/census.py", "tools/verify.py"):
        p = _ROOT / rel
        if not p.exists():
            continue                                   # tools/ is gitignored; local-only
        src = p.read_text(encoding="utf-8")
        assert 'run_full_scoring(m, "Hybrid"' not in src, f"{rel} scores under a hardcoded mode"
        assert "DEFAULT_ANALYSIS_MODE" in src, f"{rel} does not read the default from config"
        # AND it must actually RESOLVE. The first version of this test only grepped for the
        # name, so it passed while census.py raised NameError on every run — the import was
        # never added. Compile and walk the imports instead of trusting a substring.
        import ast as _ast
        tree = _ast.parse(src, filename=rel)
        imported = {a.name for n in _ast.walk(tree) if isinstance(n, _ast.ImportFrom)
                    for a in n.names} | {a.name for n in _ast.walk(tree)
                                         if isinstance(n, _ast.Import) for a in n.names}
        assert "DEFAULT_ANALYSIS_MODE" in imported, (
            f"{rel} uses DEFAULT_ANALYSIS_MODE without importing it — it will raise NameError "
            f"the moment it runs, and a substring check cannot see that"
        )


def test_hybrid_weights_are_untouched_because_history_is_reconstructed_from_them():
    """Every snapshot taken before 2026-09-19 was scored under Hybrid and carries no scored_mode
    stamp, so tools/validate.py backs its framework boosts out using THESE numbers. Re-tuning
    Hybrid in place would silently corrupt the reconstruction of all three existing windows —
    the comparison December depends on. Add a new mode instead."""
    h = C.ANALYSIS_MODES["Hybrid"]
    assert (h["fundamental_w"], h["momentum_w"]) == (0.70, 0.30)
    assert "breakout_w" not in h


# ── 2. the engine: inert leg by default, exact blend when asked ────────────────────────
def _frame(n=6, with_breakout=True):
    rng = np.random.default_rng(7)
    f = pd.DataFrame({
        "quality_score":    rng.uniform(10, 90, n),
        "momentum_score":   rng.uniform(10, 90, n),
        "governance_bonus": rng.uniform(0, 60, n),
        "pb_ratio":         rng.uniform(0.5, 8, n),
        "roe":              rng.uniform(5, 30, n),
        "rev_gr_5y":        rng.uniform(0, 30, n),
        "roce_med_5y":      rng.uniform(5, 30, n),
        "market_cap":       rng.uniform(500, 50000, n),
    })
    if with_breakout:
        f["breakout_score"] = rng.uniform(0, 100, n)
    return f


def test_default_modes_are_byte_identical_and_never_touch_breakout_score():
    """breakout_w = 0 must SKIP the leg, not multiply by zero: a frame with NO breakout_score
    column (every pre-existing synthetic test) must still score, and a frame WITH one must give
    the identical composite whether the column is present or absent."""
    gov = C.COMPOSITE_WEIGHTS["governance"]
    with_b = compute_composite_score(_frame(with_breakout=True), 0.70, 0.30)
    without = compute_composite_score(_frame(with_breakout=False), 0.70, 0.30)
    assert np.allclose(with_b["composite_score"], without["composite_score"])
    f = _frame(with_breakout=True)
    out = compute_composite_score(f, 0.70, 0.30)
    expected = (f["quality_score"] * 0.70 + f["momentum_score"] * 0.30) * (1 - gov) + f["governance_bonus"] * gov
    # no framework flag fires on this frame, so the only post-blend operation is the clip
    assert np.allclose(out["composite_score"], expected.clip(0, 100))


def test_breakout_mode_blends_the_three_legs_in_equal_thirds():
    gov = C.COMPOSITE_WEIGHTS["governance"]
    f = _frame(with_breakout=True)
    out = compute_composite_score(f, 1 / 3, 1 / 3, breakout_w=1 / 3)
    expected = ((f["quality_score"] + f["momentum_score"] + f["breakout_score"]) / 3) * (1 - gov) \
               + f["governance_bonus"] * gov
    assert np.allclose(out["composite_score"], expected.clip(0, 100))
    # and it is genuinely different from Hybrid on the same frame
    hyb = compute_composite_score(_frame(with_breakout=True), 0.70, 0.30)["composite_score"]
    assert not np.allclose(out["composite_score"], hyb)


def test_a_positive_breakout_weight_requires_the_column():
    """No silent fallback: asking for the leg without the column is a programming error, not a
    NaN composite or a quiet 50 — the engine always has breakout_score by the time it blends."""
    with pytest.raises(KeyError):
        compute_composite_score(_frame(with_breakout=False), 1 / 3, 1 / 3, breakout_w=1 / 3)


# ── 3. live: the mode re-ranks, stays NaN-free, and the default is untouched ────────────
@pytest.fixture(scope="module")
def live_pair():
    from core import fetch_and_clean_data, run_scoring_pipeline
    with contextlib.redirect_stdout(io.StringIO()):
        clean = fetch_and_clean_data("local")
        hyb = run_scoring_pipeline(clean.copy(), analysis_mode="Hybrid")
        brk = run_scoring_pipeline(clean.copy(), analysis_mode="Breakout")
    h = hyb.set_index("company_id")                       # set_index returns a NEW frame —
    b = brk.set_index("company_id").reindex(h.index)      # align on the indexed copy, not `hyb`
    assert b["quality_score"].notna().sum() == h["quality_score"].notna().sum(), "alignment lost rows"
    return h, b


def test_breakout_mode_scores_every_stock_and_genuinely_reranks(live_pair):
    hyb, brk = live_pair
    c_h = pd.to_numeric(hyb["composite_score"], errors="coerce")
    c_b = pd.to_numeric(brk["composite_score"], errors="coerce")
    assert c_b.isna().sum() == 0, "the Breakout composite has NaN rows"
    assert c_b.between(0, 100).all()
    rho = c_h.rank().corr(c_b.rank())
    assert 0.50 < rho < 0.97, f"rank-corr Hybrid vs Breakout {rho:+.3f}: either not a real re-rank or a different engine"
    top_h, top_b = set(c_h.nlargest(50).index), set(c_b.nlargest(50).index)
    assert 15 <= len(top_h & top_b) <= 48, f"top-50 overlap {len(top_h & top_b)}/50 (35/50 at the switch)"


def test_breakout_mode_leaves_everything_but_the_composite_chain_untouched(live_pair):
    """The mode changes the BLEND, nothing upstream of it: quality, momentum, breakout, governance
    and the forensic multiplier must be identical between the two runs, or the mode is doing
    something other than what its name says."""
    hyb, brk = live_pair
    for c in ("quality_score", "momentum_score", "breakout_score", "governance_bonus",
              "forensic_multiplier", "gate_pass", "red_flag_count"):
        a, b = pd.to_numeric(hyb[c], errors="coerce"), pd.to_numeric(brk[c], errors="coerce")
        assert np.allclose(a, b, equal_nan=True), f"{c} differs between modes — the mode leaked upstream"


# ── 4. surfaces: the third leg is visible where the other two are ──────────────────────
def test_config_tab_shows_the_breakout_weight_only_for_modes_that_carry_it():
    src = _APP.read_text(encoding="utf-8")
    i = src.index("_mode_rows = ")
    block = src[i:i + 1400]
    assert "breakout_w" in block and "Breakout" in block, "the weights row does not print the third leg"
    assert 'if _v.get("breakout_w", 0.0) > 0.0 else ""' in block, (
        "the Breakout cell must be conditional — printing 0% on Hybrid claims a leg it does not have"
    )
    assert '"Breakout": "⚡"' in src, "no icon for the new mode in _mode_icon"
    assert "Breakout 33/33/33" in src, "the selector help does not mention the new mode"
    # caught by the browser, not the tests: the FORMULA line above the rows still read
    # "Quality × F + Momentum × M + Governance" — incomplete for the mode that has a third leg
    assert "Breakout × B" in src, "the composite formula line does not name the third leg"
    assert "F, M and B (B only in the modes that carry it)" in src


def test_validate_reconstructs_every_mode_from_whatever_mode_a_snapshot_used():
    v = (_ROOT / "tools" / "validate.py")
    if not v.exists():
        pytest.skip("tools/ is gitignored; the harness is local-only")
    src = v.read_text(encoding="utf-8")
    for mode in C.ANALYSIS_MODES:
        assert f'"composite_mode_{mode.lower()}"' in src, f"{mode} is not reported by the harness"
    assert 'src_name = "Hybrid"' in src, "unstamped (pre-2026-09-19) snapshots must read as Hybrid"
    assert "round-trip failed" in src, "the harness lost its reconstruction self-check"
    assert not re.search(r"0\.595|0\.255|0\.70 \*|0\.30 \*", src), "mode weights retyped — read them from config"


def test_the_harness_divides_out_BOTH_multiplicative_tails():
    """A DEFECT I SHIPPED AND THEN MEASURED OUT, pinned so it cannot come back. The engine finishes
    the composite with TWO multiplicative steps after the additive framework boosts — the
    governance risk shield, then the forensic penalty. The first version of _add_candidates divided
    out only the forensic one, which folds the shield into "boosts"; those then fail to rescale
    when re-applied to a different blend, leaving a residual of exactly
    (blend_new - blend_src) * (1 - gov_mult). Measured reconstructing Hybrid from a Breakout-scored
    frame: 698 rows off by >0.01 and 100 by >1pt, of which 98 carried gov_mult < 1 — which is what
    identified it. (My first explanation, 100-cap clipping, was wrong: zero rows were at the cap.)
    After dividing out both: 2 rows, and those are genuine intermediate clips."""
    v = (_ROOT / "tools" / "validate.py")
    if not v.exists():
        pytest.skip("tools/ is gitignored; the harness is local-only")
    src = v.read_text(encoding="utf-8")
    assert "governance_risk_multiplier" in src, (
        "the harness divides out only the forensic multiplier — the governance shield is "
        "multiplicative too and will not rescale across modes"
    )
    assert "tail = F * GM" in src and "/ tail" in src, "both tails must be divided out together"


def _synthetic_snapshot(mode_name):
    """A tiny frame scored EXACTLY as the engine would under `mode_name`, plus the two
    multiplicative tails, stamped with that mode. No live data — the arithmetic is the contract."""
    m = C.ANALYSIS_MODES[mode_name]
    gov = C.COMPOSITE_WEIGHTS["governance"]; scale = 1.0 - gov
    f = pd.DataFrame({
        "company_id": ["A", "B", "C", "D"],
        "quality_score":  [80.0, 20.0, 55.0, 10.0],
        "momentum_score": [10.0, 90.0, 40.0, 70.0],
        "breakout_score": [30.0, 70.0, 50.0, 95.0],
        "governance_bonus": [40.0, 10.0, 60.0, 0.0],
        "forensic_multiplier": [1.0, 0.75, 0.50, 1.0],
        "governance_risk_multiplier": [1.0, 0.82, 1.0, 0.70],   # the tail that broke the first version
        "scored_mode": [mode_name] * 4,
    })
    blend = ((f["quality_score"] * m["fundamental_w"] + f["momentum_score"] * m["momentum_w"]
              + f["breakout_score"] * m.get("breakout_w", 0.0)) * scale + f["governance_bonus"] * gov)
    boosts = pd.Series([0.0, 15.0, 5.0, 3.0])                    # framework boosts, additive
    f["composite_score"] = ((blend + boosts)
                            * f["governance_risk_multiplier"] * f["forensic_multiplier"]).clip(0, 100)
    return f


@pytest.mark.parametrize("mode_name", ["Hybrid", "Breakout", "Technical"])
def test_the_harness_reconstructs_from_the_STAMPED_mode_not_an_assumed_one(mode_name):
    """BEHAVIOURAL, and it replaced a source scan that a mutation walked straight through: the
    first version asserted the string '"scored_mode" in early.columns' appeared in validate.py,
    which `if False and "scored_mode" in early.columns:` satisfies perfectly. Existence is not
    reachability — this drives the real function instead.

    The round-trip is the discriminator: back-solving the boosts with the SOURCE mode's weights
    and re-applying them with the same weights must return the stored composite EXACTLY. Get the
    source mode wrong and it cannot, because the blend it subtracted was never the one used."""
    v = (_ROOT / "tools" / "validate.py")
    if not v.exists():
        pytest.skip("tools/ is gitignored; the harness is local-only")
    sys.path.insert(0, str(_ROOT / "tools"))
    import validate as V
    out = V._add_candidates(_synthetic_snapshot(mode_name))
    rt = pd.to_numeric(out[f"composite_mode_{mode_name.lower()}"], errors="coerce")
    stored = pd.to_numeric(out["composite_score"], errors="coerce")
    assert np.allclose(rt, stored, atol=1e-9), (
        f"reconstructing the SOURCE mode ({mode_name}) did not return the stored composite — the "
        f"harness ignored the scored_mode stamp, so every snapshot taken under a non-Hybrid "
        f"default will be mis-compared. stored={stored.tolist()} rebuilt={rt.tolist()}"
    )
    for other in C.ANALYSIS_MODES:
        assert f"composite_mode_{other.lower()}" in out.columns, f"{other} not reconstructed"


def test_an_unstamped_snapshot_is_read_as_hybrid():
    """Every snapshot taken before 2026-09-19 predates the stamp and WAS Hybrid."""
    v = (_ROOT / "tools" / "validate.py")
    if not v.exists():
        pytest.skip("tools/ is gitignored; the harness is local-only")
    sys.path.insert(0, str(_ROOT / "tools"))
    import validate as V
    f = _synthetic_snapshot("Hybrid").drop(columns=["scored_mode"])
    out = V._add_candidates(f)
    assert np.allclose(pd.to_numeric(out["composite_mode_hybrid"], errors="coerce"),
                       pd.to_numeric(out["composite_score"], errors="coerce"), atol=1e-9)


def test_the_snapshot_records_which_mode_scored_it():
    """A snapshot without its mode is an orphan: the harness has to back framework boosts out of
    the stored composite, and that arithmetic needs the weights it was actually built from."""
    from ui.ui_export import stamp_snapshot
    df = pd.DataFrame({"name": ["A"], "composite_score": [50.0]})
    out = stamp_snapshot(df, "2026-09-18", "local")
    assert list(out.columns[:5]) == ["snapshot_vintage", "snapshot_source", "engine_version",
                                     "scored_at", "scored_mode"]
    assert out["scored_mode"].iloc[0] == C.DEFAULT_ANALYSIS_MODE, "unstamped frame must fall back to the CURRENT default"
    assert stamp_snapshot(df, None, "local", analysis_mode="Hybrid")["scored_mode"].iloc[0] == "Hybrid"
    marked = df.copy(); marked.attrs["analysis_mode"] = "Technical"
    assert stamp_snapshot(marked, None, "local")["scored_mode"].iloc[0] == "Technical", (
        "stamp_snapshot ignores df.attrs, so a frame scored under a non-default mode is mislabelled"
    )


def test_the_pipeline_records_the_mode_it_scored_under(live_pair):
    hyb, brk = live_pair
    assert hyb.attrs.get("analysis_mode") == "Hybrid"
    assert brk.attrs.get("analysis_mode") == "Breakout"
