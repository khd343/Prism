"""
test_movers.py
==============
Contract for ui/ui_movers.py — the two-vintage diff behind the 🔁 Movers tab.

THE THREE RULES, each pinned:
  1. Same engine on both sides is the CALLER's job (app.py re-scores the raw archived copy);
     this module must add zero scoring and mutate nothing — pinned by identity checks.
  2. Never a fabricated delta: a stock on one side only is NEW/DROPPED, a NaN on either side
     drops that row from that section, N/A on the wealth ladder is UNVERIFIABLE (never an
     upgrade or downgrade), a negative result age is a scheduled result, not a fresh one.
  3. Deterministic: shuffled input order yields byte-identical output; every sort is
     tie-broken by name.

Plus the cross-language parity that keeps the live side's quarter label equal to the label the
Apps Script archiver writes into the index — the five cases Prism.gs::testQuarterLabel pins,
and three more on the boundaries.

Run with: pytest tests/test_movers.py -v
"""
import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from ui.ui_movers import (JOIN_KEY, QUARTER_END_MONTHS, SOUND_LADDER, WEALTH_LADDER, compute_movers,
                          default_vintage, fy_quarter, quarter_end_vintages, quarter_label, usable_vintages)

_APP = os.path.join(os.path.dirname(__file__), "..", "app.py")
_MOV = os.path.join(os.path.dirname(__file__), "..", "ui", "ui_movers.py")


def _v(*rows):
    """A synthetic scored vintage. Each row: (company_id, name, wealth, sound, tier, rank,
    score, gate, tsunami, flags, age). Sector/cap constant — they are identity, not signal."""
    cols = ["company_id", "name", "wealth_tier", "verdict_direction", "conviction_tier", "rank",
            "composite_score", "gate_pass", "tsunami_signal", "red_flag_count", "result_age_days"]
    df = pd.DataFrame(list(rows), columns=cols)
    df["sector"] = "Chemicals"
    df["market_category"] = "Mid Cap"
    return df


PREV = _v(
    ("A", "Alpha",   "WATCH★", "MIXED",  3, 400, 41.0, 0, 0, 3,  80),   # graduates to BUY★
    ("B", "Beta",    "BUY",    "SOUND",  2, 120, 62.0, 1, 0, 2,  30),   # slides to AVOID, loses gate
    ("C", "Gamma",   "N/A",    "FLAWED", 5, 900, 20.0, 0, 0, 9,  10),   # N/A -> BUY: unverifiable
    ("D", "Delta",   "BUY★",   "SOUND",  1,   5, 88.0, 1, 1, 0,  40),   # unchanged
    ("E", "Epsilon", "WATCH",  "MIXED",  4, 700, 35.0, 0, 0, 5, -20),   # dropped from universe
    ("F", "Zeta",    "AVOID",  "FLAWED", 5, 950, 18.0, 0, 0, 7,  60),   # flags fall, result upcoming
    ("G", "Eta",     "WATCH",  "MIXED",  4, 650, np.nan, 0, 0, 4, 90),  # NaN score prev side
)
CUR = _v(
    ("A", "Alpha",   "BUY★",   "SOUND",  1, 180, 70.0, 1, 1, 3,   5),   # up 2 rungs, gate new, tsunami new, fresh
    ("B", "Beta",    "AVOID",  "FLAWED", 5, 800, 25.0, 0, 0, 7,  40),   # down 3 rungs, gate lost, flags +5
    ("C", "Gamma",   "BUY",    "MIXED",  3, 300, 50.0, 0, 0, 9,  20),
    ("D", "Delta",   "BUY★",   "SOUND",  1,   5, 88.0, 1, 1, 0,  50),   # unchanged (age just grew)
    ("F", "Zeta",    "AVOID",  "FLAWED", 5, 940, 19.0, 0, 0, 4, -10),   # flags -3; age negative = not fresh
    ("G", "Eta",     "WATCH",  "MIXED",  4, 640, 36.0, 0, 0, 4,  95),
    ("H", "Theta",   "WATCH★", "MIXED",  3, 250, 55.0, 0, 0, 1,   3),   # new to universe
)


@pytest.fixture(scope="module")
def res():
    return compute_movers(PREV, CUR)


# ── 1. Ladder moves ──────────────────────────────────────────────────────────
def test_wealth_upgrade_counts_rungs_and_downgrade_is_negative(res):
    up, dn = res["wealth_up"], res["wealth_down"]
    assert list(up["name"]) == ["Alpha"] and int(up["steps"].iloc[0]) == 2      # WATCH★ -> BUY★
    assert list(dn["name"]) == ["Beta"] and int(dn["steps"].iloc[0]) == -3      # BUY -> AVOID
    assert list(up.columns[-2:]) == ["wealth_tier", "steps"] or "steps" in up.columns


def test_na_on_the_wealth_ladder_is_unverifiable_never_an_upgrade(res):
    """Gamma went N/A -> BUY. An input came back; the engine did NOT judge it better."""
    assert list(res["wealth_unverifiable"]["name"]) == ["Gamma"]
    assert "Gamma" not in set(res["wealth_up"]["name"]) | set(res["wealth_down"]["name"])


def test_unchanged_stocks_appear_in_no_move_section(res):
    for k in ("wealth_up", "wealth_down", "wealth_unverifiable", "sound_up", "sound_down",
              "gate_new", "gate_lost", "tsunami_new", "flags", "fresh"):
        assert "Delta" not in set(res[k]["name"]), f"Delta (unchanged) leaked into {k}"


def test_soundness_flips_both_directions(res):
    assert list(res["sound_up"]["name"]) == ["Alpha", "Gamma"] or set(res["sound_up"]["name"]) == {"Alpha", "Gamma"}
    assert list(res["sound_down"]["name"]) == ["Beta"]


# ── 2. One-sided stocks ──────────────────────────────────────────────────────
def test_new_and_dropped_carry_no_deltas(res):
    assert list(res["new"]["name"]) == ["Theta"]
    assert list(res["dropped"]["name"]) == ["Epsilon"]
    for k in ("wealth_up", "wealth_down", "rank", "flags", "fresh", "gate_new"):
        names = set(res[k]["name"])
        assert "Theta" not in names and "Epsilon" not in names, f"a one-sided stock has a delta in {k}"
    assert "rank_delta" not in res["new"].columns and "rank_delta" not in res["dropped"].columns
    assert res["n_both"] == 6          # A B C D F G — E dropped, H new


# ── 3. Rank / flags / gates / fresh ──────────────────────────────────────────
def test_rank_delta_is_positive_for_a_climb_and_climbers_lead(res):
    r = res["rank"]
    assert int(r.loc[r["name"] == "Alpha", "rank_delta"].iloc[0]) == 220
    assert int(r.loc[r["name"] == "Beta", "rank_delta"].iloc[0]) == -680
    assert list(r["name"])[0] == "Gamma" or int(r["rank_delta"].iloc[0]) == r["rank_delta"].max()
    assert r["rank_delta"].is_monotonic_decreasing


def test_nan_on_either_side_yields_no_delta_not_a_zero(res):
    """Eta has no previous composite: its composite_delta must be NaN, never 0 (it still has a
    rank delta because rank exists on both sides)."""
    r = res["rank"]
    eta = r[r["name"] == "Eta"]
    assert len(eta) == 1 and int(eta["rank_delta"].iloc[0]) == 10
    assert np.isnan(eta["composite_delta"].iloc[0])


def test_flag_changes_rises_first_falls_after(res):
    fl = res["flags"]
    assert list(fl["name"]) == ["Beta", "Zeta"]
    assert list(fl["flag_delta"].astype(int)) == [5, -3]


def test_gate_and_tsunami_transitions(res):
    assert list(res["gate_new"]["name"]) == ["Alpha"]
    assert list(res["gate_lost"]["name"]) == ["Beta"]
    assert list(res["tsunami_new"]["name"]) == ["Alpha"]


def test_fresh_result_means_the_age_reset_and_is_not_negative(res):
    """Alpha 80 -> 5 landed. Delta 40 -> 50 just aged. Zeta 60 -> -10 is SCHEDULED, not fresh.
    Gamma 10 -> 20 aged."""
    assert list(res["fresh"]["name"]) == ["Alpha"]


def test_a_scheduled_result_that_has_since_been_declared_is_fresh():
    """The vendor's days-from-result is dual-signed: negative age = a SCHEDULED result not yet
    declared, positive = days since the last one. A stock that was scheduled in the previous
    vintage and carries a positive age now has reported in between — fresh. Found live: the
    June-29 copy held a scheduled date for most of the universe (Q1 results announced for
    July/August), and a plain `cur < prev` rule counted only 193 of ~2,000 as fresh."""
    p = _v(("X", "X", "BUY", "SOUND", 2, 10, 50.0, 1, 0, 1, -15),   # scheduled in the previous vintage
           ("Y", "Y", "BUY", "SOUND", 2, 11, 50.0, 1, 0, 1, -15))   # still scheduled now
    c = _v(("X", "X", "BUY", "SOUND", 2, 10, 50.0, 1, 0, 1,  20),   # declared since
           ("Y", "Y", "BUY", "SOUND", 2, 11, 50.0, 1, 0, 1,  -3))
    assert list(compute_movers(p, c)["fresh"]["name"]) == ["X"]


def test_a_scheduled_result_that_has_since_been_declared_is_fresh():
    """The vendor's days-from-result is dual-signed: negative age = a SCHEDULED result not yet
    declared, positive = days since the last one. A stock that was scheduled in the previous
    vintage and carries a positive age now has reported in between — fresh. Found live: the
    June-29 copy held a scheduled date for most of the universe (Q1 results announced for
    July/August), and a plain `cur < prev` rule counted only 193 of ~2,000 as fresh."""
    p = _v(("X", "X", "BUY", "SOUND", 2, 10, 50.0, 1, 0, 1, -15),   # scheduled in the previous vintage
           ("Y", "Y", "BUY", "SOUND", 2, 11, 50.0, 1, 0, 1, -15))   # still scheduled now
    c = _v(("X", "X", "BUY", "SOUND", 2, 10, 50.0, 1, 0, 1,  20),   # declared since
           ("Y", "Y", "BUY", "SOUND", 2, 11, 50.0, 1, 0, 1,  -3))
    assert list(compute_movers(p, c)["fresh"]["name"]) == ["X"]


def test_churn_rates_are_the_share_of_comparable_stocks_that_changed(res):
    """THE HEADLINE. Measured live on the first quarter-over-quarter run: 47% of wealth tiers and
    80% of red-flag counts changed — a fact about the ENGINE's label stability that no other
    surface can see. Denominator = stocks with the column present on BOTH sides."""
    ch = res["churn"]
    assert ch["wealth_tier"] == pytest.approx(3 / 6)        # A, B, C changed (C via N/A) of A B C D F G
    assert ch["verdict_direction"] == pytest.approx(3 / 6)  # A, B, C
    assert ch["red_flag_count"] == pytest.approx(2 / 6)     # B +5, Zeta -3
    assert ch["gate_pass"] == pytest.approx(2 / 6)          # A gained, B lost
    assert set(ch) == {"wealth_tier", "verdict_direction", "red_flag_count", "gate_pass"}


def test_churn_denominator_excludes_rows_missing_on_either_side():
    p = _v(("X", "X", "BUY", "SOUND", 2, 10, 50.0, 1, 0, np.nan, 5),
           ("Y", "Y", "BUY", "SOUND", 2, 11, 50.0, 1, 0, 2, 5))
    c = _v(("X", "X", "BUY", "SOUND", 2, 10, 50.0, 1, 0, 4, 5),
           ("Y", "Y", "BUY", "SOUND", 2, 11, 50.0, 1, 0, 5, 5))
    ch = compute_movers(p, c)["churn"]
    assert ch["red_flag_count"] == pytest.approx(1.0), "X has no previous count — only Y is comparable, and Y changed"
    assert ch["wealth_tier"] == 0.0


def test_material_head_section_names_each_reason_once_and_caps(res):
    """'What matters': into/out of BUY★ (ladder moves only), crossed the gate, new Tsunami, the
    top-N rank jumps, |Δ flags| ≥ min. One row per stock, reasons joined in a fixed order, sorted
    by how many reasons a stock has, then by rank."""
    from ui.ui_movers import material
    m = material(res, top_rank=2, min_flags=3)
    why = dict(zip(m["name"], m["why"]))
    # top_rank=2 -> Beta (−680) and Gamma (+600); Alpha's +220 is third and NOT a reason
    assert why["Alpha"] == "→ BUY★ · gate ✓ · 🌊 new"
    assert why["Beta"] == "gate ✗ · rank −680 · flags +5"
    assert why["Gamma"] == "rank +600", "Gamma: only a rank jump (N/A→BUY is unverifiable, not 'into BUY★')"
    assert why["Zeta"] == "flags −3"
    assert "Delta" not in why and "Eta" not in why
    assert list(m["name"])[:2] == ["Alpha", "Beta"], "most reasons first, then by rank"
    assert len(material(res, top_rank=2, min_flags=3, cap=1)) == 1
    for col in ("name", "sector", "why", "wealth_tier", "rank", "rank_delta"):
        assert col in m.columns


def test_material_out_of_buy_star_is_a_ladder_move_not_unverifiable():
    from ui.ui_movers import material
    p = _v(("X", "X", "BUY★", "SOUND", 1, 10, 80.0, 1, 0, 1, 5),
           ("Y", "Y", "BUY★", "SOUND", 1, 11, 80.0, 1, 0, 1, 5))
    c = _v(("X", "X", "WATCH", "SOUND", 1, 10, 80.0, 1, 0, 1, 5),     # real downgrade
           ("Y", "Y", "N/A",   "SOUND", 1, 11, 80.0, 1, 0, 1, 5))     # input went missing
    m = material(compute_movers(p, c), top_rank=0, min_flags=99)
    assert dict(zip(m["name"], m["why"])) == {"X": "BUY★ →"}


def test_material_into_buy_star_means_the_top_rung_not_any_upgrade():
    """An upgrade that stops short of BUY★ (WATCH → WATCH★) is a wealth-tier move for the full
    section, but it is NOT 'what matters' — a mutation run found the head section firing
    '→ BUY★' for every upgrade because the fixture's only upgrade happened to reach the top."""
    from ui.ui_movers import material
    p = _v(("X", "X", "WATCH",  "MIXED", 3, 10, 50.0, 1, 0, 1, 5),
           ("Y", "Y", "BUY",    "MIXED", 3, 11, 50.0, 1, 0, 1, 5))
    c = _v(("X", "X", "WATCH★", "MIXED", 3, 10, 50.0, 1, 0, 1, 5),     # up one rung, not to the top
           ("Y", "Y", "BUY★",   "MIXED", 3, 11, 50.0, 1, 0, 1, 5))     # into BUY★
    res = compute_movers(p, c)
    assert set(res["wealth_up"]["name"]) == {"X", "Y"}
    assert dict(zip(*[material(res, top_rank=0, min_flags=99)[k] for k in ("name", "why")])) == {"Y": "→ BUY★"}


def test_fresh_with_the_vintage_gap_means_reported_since_the_previous_vintage():
    """THE RULE THAT SURVIVES QUARTERLY VINTAGES. Both sides of a quarter-over-quarter comparison
    sit ~20-40 days after a filing deadline, so 'age reset' (cur < prev) cannot see a result that
    landed in between when the two ages happen to be similar — live it counted 877 of ~2,000.
    Given the gap in days between the vintages, fresh = the result date is AFTER the previous
    vintage: 0 ≤ cur_age < gap. The previous side's age is irrelevant."""
    p = _v(("A", "A", "BUY", "SOUND", 2, 1, 50.0, 1, 0, 1, 20),
           ("B", "B", "BUY", "SOUND", 2, 2, 50.0, 1, 0, 1, 20),
           ("C", "C", "BUY", "SOUND", 2, 3, 50.0, 1, 0, 1, 90),
           ("D", "D", "BUY", "SOUND", 2, 4, 50.0, 1, 0, 1, -5))
    c = _v(("A", "A", "BUY", "SOUND", 2, 1, 50.0, 1, 0, 1, 23),   # 23 < 66: reported since (age reset would MISS it)
           ("B", "B", "BUY", "SOUND", 2, 2, 50.0, 1, 0, 1, 66),   # exactly the gap: on the previous vintage day, not since
           ("C", "C", "BUY", "SOUND", 2, 3, 50.0, 1, 0, 1, 70),   # 70 > 66: before the previous vintage (age reset would WRONGLY count it)
           ("D", "D", "BUY", "SOUND", 2, 4, 50.0, 1, 0, 1, -3))   # still scheduled
    assert list(compute_movers(p, c, days_between=66)["fresh"]["name"]) == ["A"]
    # without the gap the age-reset rule is the only evidence available
    assert list(compute_movers(p, c)["fresh"]["name"]) == ["C"]


def test_flag_transition_treats_nan_as_not_a_move():
    p = _v(("X", "X", "BUY", "SOUND", 2, 10, 50.0, np.nan, np.nan, 1, 5))
    c = _v(("X", "X", "BUY", "SOUND", 2, 10, 50.0, 1, 1, 1, 6))
    r = compute_movers(p, c)
    assert r["gate_new"].empty and r["tsunami_new"].empty, "NaN -> 1 is not evidence of a transition"


# ── 4. Purity, determinism, guards ───────────────────────────────────────────
def test_inputs_are_not_mutated():
    p, c = PREV.copy(deep=True), CUR.copy(deep=True)
    compute_movers(p, c)
    pd.testing.assert_frame_equal(p, PREV)
    pd.testing.assert_frame_equal(c, CUR)


def test_shuffled_input_order_yields_identical_output(res):
    rng = np.random.default_rng(7)
    p = PREV.iloc[rng.permutation(len(PREV))].reset_index(drop=True)
    c = CUR.iloc[rng.permutation(len(CUR))].reset_index(drop=True)
    other = compute_movers(p, c)
    for k, v in res.items():
        if isinstance(v, pd.DataFrame):
            pd.testing.assert_frame_equal(v, other[k], check_like=False)
    assert other["counts"] == res["counts"]


def test_duplicate_or_missing_join_key_fails_loud():
    dup = pd.concat([PREV, PREV.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        compute_movers(dup, CUR)
    with pytest.raises(ValueError, match=JOIN_KEY):
        compute_movers(PREV.drop(columns=[JOIN_KEY]), CUR)


def test_empty_vintages_yield_empty_sections_not_errors():
    r = compute_movers(PREV.iloc[0:0], CUR.iloc[0:0])
    assert r["n_both"] == 0 and all(v == 0 for v in r["counts"].values())


def test_a_column_missing_from_one_side_yields_an_empty_section_not_a_crash():
    r = compute_movers(PREV.drop(columns=["tsunami_signal", "result_age_days"]), CUR)
    assert r["tsunami_new"].empty and r["fresh"].empty
    assert not r["wealth_up"].empty, "the other sections must still compute"


def test_no_scoring_logic_and_no_score_column_is_invented(res):
    """Every column in every section either exists in the inputs (or its _prev twin) or is one
    of the four declared deltas — the module may describe change, never score it."""
    allowed = set(CUR.columns) | {f"{c}_prev" for c in CUR.columns} | {"rank_delta", "composite_delta",
                                                                       "tier_delta", "flag_delta", "steps"}
    for k, v in res.items():
        if isinstance(v, pd.DataFrame):
            extra = sorted(set(v.columns) - allowed)
            assert not extra, f"section {k} invented columns {extra}"


def test_restrict_filters_every_section_by_current_id_but_never_dropped(res):
    """The lens row narrows the CURRENT side. Applied after the diff so a filtered-out stock can
    never masquerade as 'dropped'; `dropped` itself has no current side and stays whole."""
    from ui.ui_movers import restrict
    r = restrict(res, ["A", "H"])
    assert list(r["wealth_up"]["name"]) == ["Alpha"]
    assert r["wealth_down"].empty and r["flags"].empty          # Beta and Zeta are outside the lens
    assert list(r["new"]["name"]) == ["Theta"]
    assert list(r["dropped"]["name"]) == ["Epsilon"], "dropped must not be filtered — it has no current side"
    assert r["counts"]["wealth_up"] == 1 and r["counts"]["rank"] == 1
    assert res["counts"]["rank"] == 6, "restrict must not mutate the original result"
    # SCOPE HONESTY (found in the browser): with the lens on Steel the header read "2,101 stocks"
    # over tables showing 69 names, and the universe churn rate sat unlabelled beside them.
    assert r["n_both"] == 1, f"the header's stock count must follow the lens, got {r['n_both']}"
    assert res["n_both"] == 6, "restrict must not mutate the original count"
    assert r["restricted"] is True and not res.get("restricted"), "the restriction is not declared"
    assert r["churn"] == res["churn"], "churn cannot be recomputed from sections — it must stay universe-wide"
    from ui.ui_movers import _churn_bars
    assert "whole universe" in _churn_bars(r["churn"], True), "a filtered page hides the churn's real scope"
    assert "whole universe" not in _churn_bars(res["churn"], False)


def test_material_direction_is_up_down_or_mixed(res):
    """↑ every reason improves · ↓ every reason deteriorates · ↕ mixed. Alpha: → BUY★, gate ✓,
    🌊 new — all good. Beta: gate ✗, rank −680, flags +5 — all bad. Zeta: flags −3 — good."""
    from ui.ui_movers import material
    m = material(res, top_rank=2, min_flags=3)
    d = dict(zip(m["name"], m["direction"]))
    assert d["Alpha"] == "↑" and d["Beta"] == "↓" and d["Zeta"] == "↑" and d["Gamma"] == "↑"
    assert list(m.columns)[1] == "direction", "direction must be the first visible column"
    mixed = _v(("M", "M", "WATCH★", "SOUND", 2, 10, 50.0, 1, 0, 1, 5))
    mixed_c = _v(("M", "M", "BUY★", "SOUND", 2, 10, 50.0, 0, 0, 1, 5))    # into BUY★ but lost the gate
    mm = material(compute_movers(mixed, mixed_c), top_rank=0, min_flags=99)
    assert list(mm["direction"]) == ["↕"] and mm["why"].iloc[0] == "→ BUY★ · gate ✗"


def test_material_reason_filter_applies_before_the_cap_and_keeps_all_reasons(res):
    """A chip selects STOCKS, not words: a kept stock shows every reason it has. And the filter
    runs on the full set — filtering the visible 40 would miss a match ranked 41st."""
    from ui.ui_movers import material
    only_gate_lost = material(res, top_rank=2, min_flags=3, reasons=["gate ✗"])
    assert list(only_gate_lost["name"]) == ["Beta"]
    assert only_gate_lost["why"].iloc[0] == "gate ✗ · rank −680 · flags +5", "the kept stock lost its other reasons"
    flags = material(res, top_rank=2, min_flags=3, reasons=["flags"])
    assert set(flags["name"]) == {"Beta", "Zeta"}, "the numeric token must cover every magnitude and sign"
    assert material(res, top_rank=2, min_flags=3, reasons=["🌊 new"])["name"].tolist() == ["Alpha"]
    assert material(res, top_rank=2, min_flags=3, reasons=["BUY★ →"]).empty
    # before-the-cap: cap=1 with a reason that only the LAST-ranked material stock has must still find it
    assert list(material(res, top_rank=2, min_flags=3, cap=1, reasons=["flags"])["name"]) == ["Beta"]
    assert len(material(res, top_rank=2, min_flags=3, cap=1)) == 1
    assert len(material(res, top_rank=2, min_flags=3, cap=None)) == 4, "cap=None must return everything"


def test_cap_none_returns_every_material_row_beyond_forty():
    """The download depends on cap=None meaning EVERYTHING. The six-stock fixture cannot tell
    cap=None from head(40) (a mutation run proved it), so this builds 60 gate-crossers: the table
    shows 40, the download must carry 60."""
    from ui.ui_movers import material
    rows_p = [(f"S{i:02d}", f"Stock{i:02d}", "WATCH", "MIXED", 3, 100 + i, 40.0, 0, 0, 2, 30) for i in range(60)]
    rows_c = [(f"S{i:02d}", f"Stock{i:02d}", "WATCH", "MIXED", 3, 100 + i, 40.0, 1, 0, 2, 30) for i in range(60)]
    res60 = compute_movers(_v(*rows_p), _v(*rows_c))
    assert len(material(res60, top_rank=0, min_flags=99)) == 40, "the default cap must still be 40"
    assert len(material(res60, top_rank=0, min_flags=99, cap=None)) == 60, "cap=None must return every row"
    assert len(material(res60, top_rank=0, min_flags=99, cap=None, reasons=["gate ✓"])) == 60
    assert material(res60, top_rank=0, min_flags=99, cap=None, reasons=["gate ✗"]).empty


def test_flag_threshold_is_the_quarters_top_decile_floored_at_three():
    """MEASURED June→Sep: median |Δ flags| 2, 80% of stocks changed, so a fixed ≥3 admitted 553
    stocks (the bulk); the 90th percentile was 5 (181). The rule: top decile of non-zero |Δ|,
    never below the floor, and the floor alone when too few changed for a percentile to mean
    anything. A fixed number is wrong in both directions."""
    from ui.ui_movers import flag_threshold, material
    # small sample -> floor only (the six-stock fixture: 2 changed, so 3)
    assert flag_threshold(compute_movers(PREV, CUR)) == 3
    # noisy quarter: 100 changed, deltas 1..10 with a long tail of 1s and 2s -> 90th pct = 5 -> 5
    deltas = [1] * 50 + [2] * 30 + [3] * 8 + [4] * 2 + [5] * 5 + [6] * 3 + [8] * 2
    rows_p = [(f"S{i:03d}", f"Stock{i:03d}", "WATCH", "MIXED", 3, 100 + i, 40.0, 0, 0, 5, 30) for i in range(100)]
    rows_c = [(f"S{i:03d}", f"Stock{i:03d}", "WATCH", "MIXED", 3, 100 + i, 40.0, 0, 0, 5 + d, 30) for i, d in enumerate(deltas)]
    noisy = compute_movers(_v(*rows_p), _v(*rows_c))
    assert flag_threshold(noisy) == 5
    m = material(noisy, top_rank=0)
    assert len(m) == 10 and m["why"].str.startswith("flags +").all(), "only |Δ| ≥ 5 (the tail) is material in a noisy quarter"
    # quiet quarter: 100 changed but every |Δ| is 1 or 2 -> percentile 2 < floor -> floor holds (3), nothing material
    rows_q = [(f"S{i:03d}", f"Stock{i:03d}", "WATCH", "MIXED", 3, 100 + i, 40.0, 0, 0, 5 + (1 if i % 2 else 2), 30) for i in range(100)]
    quiet = compute_movers(_v(*rows_p), _v(*rows_q))
    assert flag_threshold(quiet) == 3 and material(quiet, top_rank=0).empty
    # the header states the effective value — never a hardcoded 3
    src = open(_MOV, encoding="utf-8").read()
    star = src[src.index('_section("⭐ What matters'):src.index("picked = _table(mat")]
    assert "{flag_threshold(res)}" in star and "|Δ flags| ≥ 3" not in star, "the star note hardcodes the threshold"
    tag = src[src.index("def _tag_reasons("):src.index("def reason_counts(")]
    assert "flag_threshold(res, floor=min_flags)" in tag, "_tag_reasons does not use the calibrated threshold"


def test_reason_counts_match_material_and_use_canonical_order(res):
    from ui.ui_movers import REASON_TOKENS, material, reason_counts
    rc = reason_counts(res, top_rank=2, min_flags=3)
    assert rc == {"→ BUY★": 1, "gate ✓": 1, "gate ✗": 1, "🌊 new": 1, "rank": 2, "flags": 2}
    assert list(rc) == [k for k in REASON_TOKENS if k in rc], "counts must follow the canonical token order"
    for tok, n in rc.items():
        assert len(material(res, top_rank=2, min_flags=3, reasons=[tok])) == n, f"{tok}: chip count ≠ filtered rows"
    assert reason_counts(compute_movers(PREV.iloc[0:0], CUR.iloc[0:0])) == {}


def test_star_table_is_uncapped_in_material_but_shown_capped_with_a_download():
    """The 40 on screen are stated; the download carries the whole material set."""
    src = open(_MOV, encoding="utf-8").read()
    i = src.index("mat = material(res")
    # end at the RENDER's own `r, fl = res[...]` line — `_tag_reasons` has an earlier
    # `up, dn, r, fl = res[...]` that a naive index() would hit, yielding an empty slice
    star = src[i:src.index("r, fl = res", i)]
    assert 'material(res, reasons=meta.get("reasons"), cap=None)' in star, "the star must compute UNCAPPED"
    assert "shown=min(len(mat), 40)" in star and "picked = _table(mat, limit=40" in star
    assert "st.download_button(" in star and "_to_csv_bytes(mat" in star, "no download of the full set"
    assert "prism_movers_" in star, "the download is not named by the two vintages"
    assert "st.session_state" not in src, "ui_movers must stay stateless — download_button writes none"


# ── 5. Ladders mirror the app ────────────────────────────────────────────────
def test_wealth_ladder_matches_the_apps_wt_order_minus_unverifiable():
    src = open(_APP, encoding="utf-8").read()
    node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
                and any(getattr(t, "id", "") == "_WT_ORDER" for t in n.targets))
    app_order = [e.value for e in node.value.elts]
    assert app_order[:-1] == WEALTH_LADDER and app_order[-1] == "N/A", (
        "the Movers ladder drifted from app.py's _WT_ORDER — one ladder, two surfaces")
    assert SOUND_LADDER == ["SOUND", "MIXED", "FLAWED"]


# ── 6. Quarter-label parity with the archiver (Prism.gs::_fyQuarter) ─────────
@pytest.mark.parametrize("iso,label", [
    ("2026-06-05", "FY26Q4"), ("2026-08-28", "FY27Q1"), ("2026-12-01", "FY27Q2"),
    ("2027-03-02", "FY27Q3"), ("2026-08-10", "FY26Q4 (off-cycle)"),           # the script's 5
    ("2026-06-30", "FY26Q4"), ("2026-09-24", "FY27Q1"), ("2027-01-31", "FY27Q2 (off-cycle)"),
    ("2026-10-31", "FY27Q1 (off-cycle)"), ("2027-05-31", "FY27Q4"),
])
def test_fy_quarter_matches_the_archiver(iso, label):
    assert fy_quarter(iso) == label


# ── 7. Index handling ────────────────────────────────────────────────────────
def _index(*rows):
    return pd.DataFrame(list(rows), columns=["vintage_date", "fy_quarter", "spreadsheet_id", "url",
                                             "archived_at", "rows_technicals", "status"])


def test_usable_vintages_keeps_ok_rows_newest_first_as_text_dates():
    idx = _index(
        ("2026-09-03", "FY27Q1", "id1", "u", "t", 2116, "superseded by 2026-09-10"),
        (pd.Timestamp("2026-09-24"), "FY27Q1", "id2", "u", "t", 2116, "ok"),
        ("2026-10-29", "FY27Q1 (off-cycle)", "id3", "u", "t", 2116, "ok"),
        ("2026-08-01", "x", "", "u", "t", "", "failed: tab missing"),
    )
    ok = usable_vintages(idx)
    assert list(ok["vintage_date"]) == ["2026-10-29", "2026-09-24"]
    assert ok["vintage_date"].map(type).eq(str).all()
    assert list(ok.columns) == ["vintage_date", "fy_quarter", "spreadsheet_id"]


def test_default_vintage_prefers_the_most_recent_on_cycle_quarter():
    ok = usable_vintages(_index(
        ("2026-10-29", "FY27Q1 (off-cycle)", "id3", "u", "t", 1, "ok"),
        ("2026-09-24", "FY27Q1", "id2", "u", "t", 1, "ok"),
        ("2026-06-30", "FY26Q4", "id1", "u", "t", 1, "ok"),
    ))
    assert default_vintage(ok)["spreadsheet_id"] == "id2"
    only_off = usable_vintages(_index(("2026-10-29", "FY27Q1 (off-cycle)", "id3", "u", "t", 1, "ok")))
    assert default_vintage(only_off)["spreadsheet_id"] == "id3"
    assert default_vintage(ok.iloc[0:0]) is None


# ── 7b. One copy per quarter: the calendar quarter-end (2026-10-07) ─────────────────────────
def test_quarter_end_vintages_keep_the_newest_copy_of_each_quarter_end_month():
    ok = usable_vintages(_index(
        ("2026-06-15", "FY26Q4", "jun15", "u", "t", 1, "ok"),
        ("2026-06-30", "FY26Q4", "jun30", "u", "t", 1, "ok"),
        ("2026-07-31", "FY26Q4 (off-cycle)", "jul31", "u", "t", 1, "ok"),
        ("2026-08-31", "FY27Q1", "aug31", "u", "t", 1, "ok"),
        ("2026-09-19", "FY27Q1", "sep19", "u", "t", 1, "ok"),
        ("2026-09-30", "FY27Q1", "sep30", "u", "t", 1, "ok"),
        ("2026-10-06", "FY27Q1 (off-cycle)", "oct06", "u", "t", 1, "ok"),
        ("2026-12-31", "FY27Q2", "dec31", "u", "t", 1, "ok"),
    ))
    q = quarter_end_vintages(ok)
    assert q["spreadsheet_id"].tolist() == ["dec31", "sep30", "jun30"]
    assert list(q.columns) == ["vintage_date", "fy_quarter", "spreadsheet_id"]
    assert default_vintage(q)["spreadsheet_id"] == "dec31", "the default is the latest quarter"


def test_no_quarter_end_copy_yet_means_an_empty_list_for_the_caller_to_fall_back_from():
    ok = usable_vintages(_index(("2026-10-06", "FY27Q1 (off-cycle)", "oct06", "u", "t", 1, "ok")))
    assert quarter_end_vintages(ok).empty
    assert quarter_end_vintages(ok.iloc[0:0]).empty


def test_every_day_of_a_quarter_end_month_is_a_clean_on_cycle_quarter():
    """The claim the picker rests on, checked against the archiver's own rule for every day of five
    years: a quarter-end month is always after that quarter's deadline and before the next season,
    and holds a fixed results quarter — Jun Q4, Sep Q1, Dec Q2, Mar Q3."""
    expect = {3: "Q3", 6: "Q4", 9: "Q1", 12: "Q2"}
    assert set(QUARTER_END_MONTHS) == set(expect)
    for day in pd.date_range("2026-01-01", "2030-12-31"):
        if day.month in QUARTER_END_MONTHS:
            label = fy_quarter(day.strftime("%Y-%m-%d"))
            assert "off-cycle" not in label, (day, label)
            assert label.endswith(expect[day.month]), (day, label)


def test_quarter_label_names_the_results_the_copy_holds():
    assert quarter_label("2026-06-30") == "Jan–Mar + annual results (FY26 Q4) · 30 Jun 2026"
    assert quarter_label("2026-09-30") == "Apr–Jun results (FY27 Q1) · 30 Sep 2026"
    assert quarter_label("2026-12-31") == "Jul–Sep results (FY27 Q2) · 31 Dec 2026"
    assert quarter_label("2027-03-31") == "Oct–Dec results (FY27 Q3) · 31 Mar 2027"


def test_the_tab_lists_quarter_ends_from_an_index_but_never_filters_a_pasted_copy():
    """app.py: the index path narrows to quarter_end_vintages (falling back to every copy until one
    exists); a single pasted copy is compared as is — the way to reach any other copy."""
    import ast
    src = open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8").read()
    i = src.index("_mv_ok = _load_archive_index(_mv_id)")
    j = src.index("except ValueError:", i)
    index_path = src[i:j]
    assert "quarter_end_vintages(_mv_ok)" in index_path and "if not _mv_qe.empty:" in index_path
    copy_path = src[j:src.index("except Exception as _e:", j)]
    assert "quarter_end_vintages" not in copy_path
    assert "format_func=lambda i: _mv_opt.get(i, i)" in src
    assert '_mv_ok["vintage_date"].map(quarter_label)' in src
    ast.parse(src)


def test_a_non_index_sheet_fails_loud():
    with pytest.raises(ValueError, match="not a PRISM Archive Index"):
        usable_vintages(pd.DataFrame({"Ratio": [1], "ROCE": [2]}))


# ── 8. Rendering: stateless, honest headers, no crash ─────────────────────────
def test_render_uses_no_raw_column_names_as_headers():
    """Every column a section can show has a display header, and none of them is the raw
    snake_case name (the app-wide vocabulary rule)."""
    import re
    src = open(_MOV, encoding="utf-8").read()
    hdr = re.search(r"_HDR = \{(.*?)\n\}", src, re.S).group(1)
    pairs = re.findall(r'"([a-z_]+)": (?:"([^"]*)"|None)', hdr)
    assert len(pairs) >= 20, "the header map lost entries"
    for col, label in pairs:
        if label:
            assert label != col and not (label.islower() and "_" in label), f"{col} shown raw as {label!r}"
    for delta in ("rank_delta", "composite_delta", "flag_delta", "steps", "tier_delta"):
        assert f'"{delta}":' in hdr, f"{delta} has no header"


def _movers_app():
    import pandas as _pd
    import numpy as _np
    import streamlit as _st
    from ui.ui_movers import compute_movers, render_movers
    cols = ["company_id", "name", "sector", "market_category", "wealth_tier", "verdict_direction",
            "conviction_tier", "rank", "composite_score", "gate_pass", "tsunami_signal",
            "red_flag_count", "result_age_days"]
    p = _pd.DataFrame([["A", "Alpha", "S", "Mid Cap", "WATCH★", "MIXED", 3, 400, 41.0, 0, 0, 3, 80],
                       ["B", "Beta", "S", "Mid Cap", "BUY", "SOUND", 2, 120, 62.0, 1, 0, 2, 30]], columns=cols)
    c = _pd.DataFrame([["A", "Alpha", "S", "Mid Cap", "BUY★", "SOUND", 1, 180, 70.0, 1, 1, 3, 5],
                       ["B", "Beta", "S", "Mid Cap", "AVOID", "FLAWED", 5, 800, 25.0, 0, 0, 7, 40]], columns=cols)
    res = compute_movers(p, c)
    picked = render_movers(res, {"prev_vintage": "2026-09-24", "cur_vintage": "2026-12-29",
                                 "prev_label": "FY27Q1", "cur_label": "FY27Q2", "engine": "abc1234",
                                 "prev_regime": "SIDEWAYS", "cur_regime": "BULL", "mode": "Hybrid", "profile": "Balanced"})
    _st.text(f"UP={list(res['wealth_up']['name'])}")
    _st.text(f"PICKED={picked}")


def test_render_movers_renders_every_section_without_exception():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_movers_app)
    at.run(timeout=30)
    assert not at.exception, f"render raised: {at.exception}"
    html = " ".join(m.value for m in at.markdown)
    for needle in ("FY27Q1 → FY27Q2", "regime changed", "Churn — share of stocks whose label changed",
                   "wealth tier", "width:100%", "What matters", "Upgrades", "Climbers",
                   "Red flags — rises", "Fresh results", "New to the universe"):
        assert needle in html, f"section/header {needle!r} missing from the render"


def test_header_is_a_stat_row_with_churn_bars_and_the_explainer_in_a_tooltip():
    """THE FOLD. Measured live: seven controls and ~280 words sat above the first data row, and
    ⭐ What matters started ~900px down on a 732px viewport. The header is now a compact stat
    row, churn is four BARS (read in a glance, not a sentence), the same-engine explainer lives
    in an ⓘ tooltip rather than permanent prose, and the compare time folds into the header."""
    from streamlit.testing.v1 import AppTest
    src = open(_MOV, encoding="utf-8").read()
    hdr = src[src.index("def render_movers("):src.index('_section("⭐ What matters')]
    # source: the explainer text is bound to `tip` and emitted ONLY through a title= attribute
    assert 'tip = ("Both sides were scored by the same engine' in hdr, "the explainer text moved or changed"
    assert "title='{tip}'" in hdr, "the same-engine explainer must be a tooltip (title=), not visible prose"
    assert hdr.count("{tip}") == 1, "the explainer is emitted somewhere besides the tooltip"
    assert "Both sides scored by the" not in hdr, "the old paragraph form is back"
    assert "_churn_bars(" in hdr and "def _churn_bars(" in src, "churn is not rendered as bars"
    bars = src[src.index("def _churn_bars("):src.index("def render_movers(")]
    assert "width:{pct:.0f}%" in bars and "height:6px" in bars, "no bar element per churn label"
    assert "(whole universe" in bars, "the restricted-scope note left the churn block"
    assert 'meta.get("elapsed")' in hdr and "compared in" in hdr, "the compare time does not fold into the header"
    # and it renders: the tooltip is in the DOM as an attribute, the bars carry the percentages
    at = AppTest.from_function(_movers_app)
    at.run(timeout=30)
    html = " ".join(m.value for m in at.markdown)
    assert "title='Both sides were scored by the same engine" in html
    assert html.count("height:6px;background:") >= 4, "fewer than four churn bars rendered"
    out = [t.value for t in at.text]
    assert "UP=['Alpha']" in out
    assert "PICKED=None" in out, "with no row clicked the module must return None, never a default pick"


def test_the_evidence_sections_live_behind_five_sub_tabs():
    """THE LAYOUT FIX. Rendered flat, the sixteen sections made a 7,820px page — 10.7 screens,
    thirteen tables at their height cap, and ⭐ What matters (the whole point of the materiality
    pass) as one of sixteen equals. The star stays open; the fifteen evidence sections go behind
    five st.tabs — the language Market Pulse already speaks, and the one container that holds its
    selection across a rerun on Streamlit 1.54 (st.expander has no persistent open state)."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_movers_app)
    at.run(timeout=30)
    labels = [t.label for t in at.tabs]
    assert len(labels) == 5, f"expected exactly five evidence sub-tabs, got {labels}"
    for i, head in enumerate(["💹 Wealth", "🧭 Soundness", "📈 Rank", "🚦 Gates & flags", "🆕 Universe"]):
        assert labels[i].startswith(head + " · "), f"tab {i} is {labels[i]!r}, expected '{head} · <count>'"
    # ⭐ What matters must NOT be inside a tab — it is the page
    src = open(_MOV, encoding="utf-8").read()
    i, j = src.index('_section("⭐ What matters'), src.index("_ev = st.tabs(")
    assert i < j, "What matters was pushed inside/below the evidence tabs — it is the page, not evidence"


def test_truncation_is_stated_never_silent():
    """A section header that says 1,992 above a table showing 40 promises what it cannot deliver —
    the same quiet mislead as a mislabelled unit. _block always states 'showing N of M'."""
    src = open(_MOV, encoding="utf-8").read()
    i = src.index("def _section(")
    body = src[i:src.index("def _block(")]
    assert "showing {shown} of {n:,}" in body, "the truncation disclosure is gone from the section header"
    blk = src[src.index("def _block("):src.index("def _churn_bars(")]
    assert "shown=min(len(df), limit)" in blk, "_block no longer tells the header how much is shown"
    star = src[src.index('_section("⭐ What matters'):src.index("picked = _table(")]
    assert "shown=min(len(mat), 40)" in star, "the star section hides its own truncation"
    # every evidence section goes through _block, so none can drift into a bare _section+_table
    ev = src[src.index("_ev = st.tabs("):]
    assert "_table(" not in ev, "an evidence section bypasses _block — its truncation would be silent"
    assert ev.count("_block(") == 15, f"expected 15 evidence sections via _block, found {ev.count('_block(')}"


def test_noise_columns_are_hidden_and_label_prev_twins_are_kept():
    """Identity, the thing that moved, the delta, where it stands. Market cap repeated in sixteen
    tables is noise; a raw 0/1 gate column is noise in sections defined BY that transition; a
    numeric `_prev` twin is redundant beside its delta. The LABEL twins stay — 'was AVOID, now
    BUY★' is the entire content of a ladder move."""
    from ui.ui_movers import _HDR
    for hidden in ("market_category", "gate_pass", "tsunami_signal", "rank_prev",
                   "composite_score_prev", "conviction_tier_prev", "red_flag_count_prev", JOIN_KEY):
        assert _HDR[hidden] is None, f"{hidden} is visible again — noise in every table"
    assert _HDR["wealth_tier_prev"] == "was" and _HDR["verdict_direction_prev"] == "was"
    assert _HDR["wealth_tier"] == "Wealth tier" and _HDR["verdict_direction"] == "Soundness"


def test_what_matters_is_click_to_tearsheet_and_the_module_stays_stateless():
    """The loop this tab was missing: it named the candidates and could not pass them on, while
    Tsunami, QGLP and Discovery all hand a clicked row to the Tear-Sheet. Selection is on ⭐ What
    matters ONLY — one obvious affordance, and sixteen competing selection widgets would make
    'which pick wins' ambiguous. The module returns the name; app.py owns the session_state."""
    src = open(_MOV, encoding="utf-8").read()
    assert src.count('on_select="rerun"') == 1, "selection must exist on exactly one table"
    assert 'selection_mode="single-row"' in src
    assert "st.session_state" not in src, "ui_movers must never touch session_state — app.py owns it"
    assert "return picked" in src, "render_movers no longer returns the clicked stock"
    star = src[src.index('_section("⭐ What matters'):src.index("_ev = st.tabs(")]
    assert "select=True" in star, "the star table is not selectable"
    assert "click a row to load it into the tear-sheet" in star.lower(), (
        "the star section does not tell the reader the rows are clickable")
    assert "click The Tear-Sheet tab" in star, "no confirmation banner after a pick"
    # The non-selectable branch must return None. Every _block discards it today, so a phantom
    # pick there changes nothing observable (a mutation run confirmed it is an equivalent mutant)
    # — which is exactly why the contract is pinned here rather than left to be discovered by the
    # first caller that reads the value.
    nosel = src[src.index("    if not select:"):src.index("    sel = st.dataframe(")]
    assert "return None" in nosel, "the non-selectable branch must return None, never a phantom pick"
