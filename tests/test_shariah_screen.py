"""
test_shariah_screen.py
======================
Contract for the Shariah screen (core/shariah.py, 2026-10-16): every stock gets shariah_status =
Shariah / Non-Shariah / Review / Doubtful with a one-line reason, from six rules — the KMI-30 / Meezan method
with the operator's thresholds (debt 33%, illiquid assets 20%), every number and list in config.py.

  Shariah      passes every rule PRISM's data can check
  Non-Shariah  a prohibited business, or a PROVEN fail on a rule
  Review       nothing visible fails, but a rule the sheet cannot prove (investments, other income) or a
               business awaiting a scholar's ruling is open — the operator's check settles it (the operator's
               fourth label, 2026-10-16: measured, 236 of 1,484 in the size window, Infosys / HCL / Wipro among them)
  Doubtful     a figure is MISSING — PRISM cannot screen the stock fully

Each rule likewise reads pass / fail / review / unknown. A fail beats a missing figure, and a missing figure beats
a review item. Kite will later buy only `Shariah`, sell on `Non-Shariah` and NEVER sell on `Review` or `Doubtful`,
so a missing figure must never read as a pass (CLAUDE.md §5 "Unverifiable is not passed") nor as a fail.

The operator's answers of 2026-10-16 (measured on the 15 Oct sheet before they were given):
  * the 'Aerospace & Defence' SECTOR goes to review — 18 of its 20 stocks outside the defence industry
    describe themselves as defence suppliers (warships, missile propellants, "defence-focused");
  * a company-level prohibited list, each entry with its business-activity reason (Delta Corp: casinos,
    filed by the vendor under "Miscellaneous");
  * Rule 6's receivables use the BALANCE SHEET's own revenue year: the vendor's balance sheet trails its
    P&L by a year on 2,235 of 2,695 stocks, and its working-capital ratios use the balance sheet's year
    (inventory turnover x inventory reproduces year-back revenue on 372 of 402). Where the year cannot be
    told, receivables count as unknown (0), so a pass then needs no receivables at all.

Zero scoring impact: a full-pipeline A/B on real data keeps every pre-existing column byte-identical.

Run with: pytest tests/test_shariah_screen.py -v
"""

import ast
import contextlib
import copy
import io
import itertools
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)

import config
from config import (SHARIAH_ALLOWED_NAME_MATCHES, SHARIAH_PROHIBITED_COMPANIES, SHARIAH_PROHIBITED_INDUSTRIES,
                    SHARIAH_REVIEW_INDUSTRIES, SHARIAH_REVIEW_SECTORS, SHARIAH_RULE_STATES, SHARIAH_RULES,
                    SHARIAH_STATUSES)
from core.cyclicality_map import INDUSTRY_TIER, SECTOR_TIER_FALLBACK
from core.shariah import SHARIAH_COLUMNS, SHARIAH_PASS_REASON, compute_shariah

_DATA_DIR = os.path.join(_ROOT, "Other Resources", "CSV Data")
_KEYWORDS = ("alcohol", "tobacco", "casino", "gaming", "betting", "lottery", "hotel", "resort", "restaurant",
             "media", "entertainment", "music", "film", "broadcast", "defence", "weapon", "brew", "distill",
             "liquor", "wine")

# A stock that passes all six rules, with room to spare on each:
#   other income 0% of revenue · debt 10% · illiquid 50% · NLA 400 vs market cap 2,000 ·
#   cash + unexplained assets 20% (receivables 1,000 x 73 / 365 = 200)
_CLEAN = dict(company_id="NSE:TEST", name="Test Co", industry="Speciality Chemicals", sector="Chemicals",
              cyclicality_tier_code="A", total_assets=1000.0, total_liabilities=1000.0, debt=100.0,
              fixed_assets=500.0, cwip=0.0, inventory=100.0, cash_equivalents=50.0, market_cap=2000.0,
              price_to_book=2.0, revenue=1000.0, revenue_1yb=1000.0, ebitda=200.0, ebit=180.0,
              days_receivable=73.0, cf_year_lag=0.0)


def _screen(**over) -> pd.Series:
    return compute_shariah(pd.DataFrame([{**_CLEAN, **over}])).iloc[0]


def _screen_many(rows) -> pd.DataFrame:
    return compute_shariah(pd.DataFrame([{**_CLEAN, **r} for r in rows]))


# ── 0. The shape of the output ───────────────────────────────────────────────────────────────────────
def test_a_clean_stock_is_shariah_with_the_pass_sentence():
    s = _screen()
    assert s["shariah_status"] == "Shariah"
    assert s["shariah_reason"] == SHARIAH_PASS_REASON
    assert SHARIAH_PASS_REASON == ("Passes every rule PRISM can check (interest income is not in the data "
                                   "— purify)")
    assert [s[f"shariah_{k}"] for k in SHARIAH_RULES] == ["pass"] * 6


def test_it_only_adds_columns_and_adds_exactly_the_declared_ones():
    df = pd.DataFrame([_CLEAN, {**_CLEAN, "company_id": "NSE:T2", "debt": np.nan}])
    before = df.copy()
    out = compute_shariah(df)
    assert list(out.columns[: len(before.columns)]) == list(before.columns)
    pd.testing.assert_frame_equal(out[before.columns], before)
    assert set(out.columns) - set(before.columns) == set(SHARIAH_COLUMNS)
    assert len(SHARIAH_COLUMNS) == len(set(SHARIAH_COLUMNS))


def test_the_vocabularies_are_closed():
    rows = [{}, {"debt": 900.0}, {"fixed_assets": np.nan}, {"industry": "Hotels"}, {"total_assets": 0.0},
            {"price_to_book": np.nan}, {"ebit": 900.0}, {"industry": None}]
    out = _screen_many(rows)
    assert set(out["shariah_status"]) <= set(SHARIAH_STATUSES)
    assert set(out["shariah_status"]) == set(SHARIAH_STATUSES), "a status never fired in this sweep"
    for k in SHARIAH_RULES:
        assert set(out[f"shariah_{k}"]) <= set(SHARIAH_RULE_STATES), k
    assert out["shariah_reason"].str.len().gt(0).all()
    assert SHARIAH_STATUSES == ("Shariah", "Non-Shariah", "Review", "Doubtful")
    assert SHARIAH_RULE_STATES == ("pass", "fail", "review", "unknown")


def test_a_missing_input_column_is_unknown_never_a_pass():
    """A synthetic or half-loaded frame without a column: every rule that reads it is unknown."""
    df = pd.DataFrame([{k: v for k, v in _CLEAN.items() if k not in ("debt", "fixed_assets")}])
    s = compute_shariah(df).iloc[0]
    assert s["shariah_debt"] == "unknown" and s["shariah_illiquid"] == "unknown"
    assert s["shariah_status"] == "Doubtful"


# ── 1. Rule 1 — Business ─────────────────────────────────────────────────────────────────────────────
def test_rule1_financials_by_the_tier_code_are_non_shariah():
    s = _screen(industry="Banks - Private", sector="Banks", cyclicality_tier_code="E")
    assert (s["shariah_business"], s["shariah_status"]) == ("fail", "Non-Shariah")
    assert s["shariah_reason"] == "prohibited business: financial services (Banks - Private)"


def test_rule1_each_prohibited_industry_is_non_shariah():
    out = _screen_many([{"industry": i} for i in sorted(SHARIAH_PROHIBITED_INDUSTRIES)])
    assert out["shariah_business"].eq("fail").all()
    assert out["shariah_status"].eq("Non-Shariah").all()
    assert (out["shariah_reason"] == "prohibited business: " + out["industry"]).all()


def test_rule1_each_review_industry_awaits_a_scholars_ruling():
    out = _screen_many([{"industry": i} for i in sorted(SHARIAH_REVIEW_INDUSTRIES)])
    assert out["shariah_business"].eq("review").all()
    assert out["shariah_status"].eq("Review").all()
    assert (out["shariah_reason"] == out["industry"] + " — awaiting a scholar's ruling").all()


def test_rule1_the_review_sector_awaits_a_ruling_and_its_prohibited_industry_still_fails():
    s = _screen(industry="Industrial Explosives", sector="Aerospace & Defence")
    assert (s["shariah_business"], s["shariah_status"]) == ("review", "Review")
    assert s["shariah_reason"] == "Aerospace & Defence sector — awaiting a scholar's ruling"
    s = _screen(industry="Aerospace & Defence - Equipments", sector="Aerospace & Defence")
    assert (s["shariah_business"], s["shariah_status"]) == ("fail", "Non-Shariah"), "a fail beats a review"


def test_rule1_a_listed_company_is_non_shariah_with_its_written_reason():
    cid, why = next(iter(sorted(SHARIAH_PROHIBITED_COMPANIES.items())))
    s = _screen(company_id=cid, industry="Miscellaneous", sector="Miscellaneous")
    assert (s["shariah_business"], s["shariah_status"]) == ("fail", "Non-Shariah")
    assert s["shariah_reason"] == f"prohibited business: {why}"


def test_the_company_list_holds_delta_corp_and_every_entry_gives_a_reason():
    assert "NSE:DELTACORP" in SHARIAH_PROHIBITED_COMPANIES
    for cid, why in SHARIAH_PROHIBITED_COMPANIES.items():
        assert re.fullmatch(r"(NSE|BSE):\S+", cid), cid
        assert isinstance(why, str) and len(why) >= 10, f"{cid}: an entry needs its business-activity reason"


def test_rule1_a_blank_industry_is_doubtful():
    for blank in (None, np.nan, "", "  "):
        s = _screen(industry=blank)
        assert (s["shariah_business"], s["shariah_status"]) == ("unknown", "Doubtful"), repr(blank)
        assert s["shariah_reason"] == "no industry in the data"


def test_rule1_an_ordinary_industry_passes():
    s = _screen()
    assert s["shariah_business"] == "pass"
    assert s["shariah_business_note"] == "Speciality Chemicals — not on PRISM's prohibited lists"


# ── Rule 1 through the real call site (compute_derived_signals sets the tier code first) ──────────────
@pytest.fixture(scope="module")
def mapped_cols():
    from core.data_engine import (BALANCE_COLS, CASHFLOW_COLS, COMMON_COLS, INCOME_COLS, RATIO_COLS,
                                  SHAREHOLDING_COLS, TECHNICAL_COLS)
    cols = set()
    for m in (COMMON_COLS, RATIO_COLS, INCOME_COLS, BALANCE_COLS, CASHFLOW_COLS, SHAREHOLDING_COLS,
              TECHNICAL_COLS):
        cols.update(m.values())
    return sorted(cols)


def _derived(mapped_cols, industries, sectors):
    """compute_derived_signals on a frame that materialises EVERY mapped column as NaN (CLAUDE.md §6)."""
    from core.data_engine import compute_derived_signals
    n = len(industries)
    df = pd.DataFrame({"company_id": [f"NSE:T{i}" for i in range(n)], "name": [f"T{i}" for i in range(n)],
                       "industry": industries, "sector": sectors, "market_cap": [1000.0] * n})
    missing = [c for c in mapped_cols if c not in df.columns]
    df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing)], axis=1)
    with contextlib.redirect_stdout(io.StringIO()):
        return compute_derived_signals(df)


def test_every_tier_e_industry_is_non_shariah_through_the_call_site(mapped_cols):
    e_ind = sorted(k for k, v in INDUSTRY_TIER.items() if v == "E")
    assert len(e_ind) >= 20, "the map's financial tier shrank — re-measure Rule 1"
    out = _derived(mapped_cols, e_ind, ["Finance"] * len(e_ind))
    assert set(SHARIAH_COLUMNS) <= set(out.columns), "the call site no longer adds the screen"
    assert out["shariah_status"].eq("Non-Shariah").all(), out.loc[out["shariah_status"].ne("Non-Shariah"), "industry"]


def test_an_unmapped_industry_under_a_financial_sector_is_non_shariah(mapped_cols):
    fin = sorted(k for k, v in SECTOR_TIER_FALLBACK.items() if v == "E")
    out = _derived(mapped_cols, ["A Brand New Lending Business"] * len(fin), fin)
    assert out["shariah_status"].eq("Non-Shariah").all()


def test_review_prohibited_and_blank_through_the_call_site(mapped_cols):
    """With every figure blank, nothing but Rule 1 can fail — so the prohibited are Non-Shariah. The review
    industries still await a ruling at Rule 1, but a missing figure beats a review item: they read Doubtful, the
    reason naming the first missing figure. A blank industry is itself a missing figure (Rule 1's reason)."""
    rev = sorted(SHARIAH_REVIEW_INDUSTRIES)
    pro = sorted(SHARIAH_PROHIBITED_INDUSTRIES)
    out = _derived(mapped_cols, rev + pro + [np.nan], ["Hotels & Restaurants"] * (len(rev) + len(pro))
                   + ["Chemicals"])
    assert out["shariah_business"].iloc[: len(rev)].eq("review").all()
    assert out["shariah_status"].iloc[: len(rev)].eq("Doubtful").all()
    assert out["shariah_reason"].iloc[: len(rev)].str.startswith("other income cannot be estimated").all()
    assert out["shariah_status"].iloc[len(rev): -1].eq("Non-Shariah").all()
    assert (out["shariah_status"].iloc[-1], out["shariah_reason"].iloc[-1]) == ("Doubtful",
                                                                                "no industry in the data")


# ── 2. Rule 2 — impermissible income ≤ 5% of revenue (only its LEAST can be seen) ──────────────────────
def test_rule2_at_exactly_the_limit_passes_with_the_caveat_always_shown():
    s = _screen(ebit=250.0, ebitda=200.0)                                  # 50 / 1,000 = 5%
    assert s["shariah_income"] == "pass" and s["shariah_income_pct"] == pytest.approx(5.0)
    assert "interest income is not in the data; purify" in s["shariah_income_note"]
    assert "interest income is not in the data; purify" in _screen()["shariah_income_note"]


def test_rule2_above_the_limit_is_for_review_never_non_shariah():
    s = _screen(ebit=251.0, ebitda=200.0)                                  # 5.1%
    assert (s["shariah_income"], s["shariah_status"]) == ("review", "Review")
    assert s["shariah_reason"] == "other income is at least 5.10% of revenue — may be interest (limit 5%)"
    for ebit in (300.0, 900.0, 5000.0):
        s = _screen(ebit=ebit, ebitda=200.0)
        assert s["shariah_income"] != "fail" and s["shariah_status"] == "Review", ebit


def test_rule2_cannot_be_estimated_without_revenue_ebit_or_ebitda():
    for over in ({"revenue": 0.0}, {"revenue": -5.0}, {"revenue": np.nan}, {"ebit": np.nan},
                 {"ebitda": np.nan}):
        s = _screen(**over)
        assert s["shariah_income"] == "unknown", over
        assert np.isnan(s["shariah_income_pct"]), over


def test_rule2_reads_only_the_least_other_income():
    """EBIT includes other income and EBITDA does not, so max(EBIT - EBITDA, 0) is the LEAST it can be."""
    assert _screen(ebit=150.0, ebitda=200.0)["shariah_income_pct"] == 0.0     # EBIT below EBITDA: 0, never < 0
    assert _screen(ebit=230.0, ebitda=200.0)["shariah_income_pct"] == pytest.approx(3.0)


# ── 3. Rule 3 — interest-bearing debt ≤ 33% of total assets ─────────────────────────────────────────
def test_rule3_edges():
    assert _screen(debt=330.0)["shariah_debt"] == "pass"                     # exactly 33%
    s = _screen(debt=331.0)
    assert (s["shariah_debt"], s["shariah_status"]) == ("fail", "Non-Shariah")
    assert s["shariah_reason"] == "debt is 33.10% of assets (limit 33%)"
    s = _screen(debt=410.0)
    assert s["shariah_reason"] == "debt is 41% of assets (limit 33%)"       # the brief's own example
    assert _screen(debt=0.0)["shariah_debt"] == "pass"


def test_rule3_missing_figures_are_unknown():
    s = _screen(debt=np.nan)
    assert (s["shariah_debt"], s["shariah_status"]) == ("unknown", "Doubtful")
    assert s["shariah_reason"] == "no debt figure in the data"
    s = _screen(debt=-5.0)
    assert s["shariah_debt"] == "unknown", "a negative debt is not a measurement"
    assert s["shariah_debt_note"] == "the debt figure in the data is negative (a vendor error)"
    for ta in (0.0, -10.0, np.nan):
        s = _screen(total_assets=ta)
        assert s["shariah_debt"] == "unknown", ta
        assert s["shariah_reason"] == "no total-assets figure in the data", ta


def test_a_negative_balance_sheet_figure_is_not_a_measurement():
    """Debt was negative on 2 live rows (15 Oct sheet): a vendor error, read as missing, never as a pass."""
    assert _screen(fixed_assets=-500.0)["shariah_illiquid"] == "unknown"
    assert _screen(fixed_assets=150.0, cwip=-10.0)["shariah_illiquid"] == "unknown"
    s = _screen(inventory=-5000.0, cash_equivalents=-5000.0)
    assert s["shariah_nla"] == "pass" and s["shariah_investments_pct"] == pytest.approx(30.0)


# ── 4. Rule 4 — illiquid assets (fixed assets + CWIP) ≥ 20% of total assets ──────────────────────────
def test_rule4_edges():
    assert _screen(fixed_assets=200.0)["shariah_illiquid"] == "pass"          # exactly 20%
    assert _screen(fixed_assets=150.0, cwip=50.0)["shariah_illiquid"] == "pass"   # CWIP counts
    s = _screen(fixed_assets=199.0)
    assert (s["shariah_illiquid"], s["shariah_status"]) == ("fail", "Non-Shariah")
    assert s["shariah_reason"] == "illiquid assets are 19.90% of assets (needs at least 20%)"


def test_rule4_missing_cwip_below_the_limit_is_doubtful_never_non_shariah():
    s = _screen(fixed_assets=150.0, cwip=np.nan)
    assert (s["shariah_illiquid"], s["shariah_status"]) == ("unknown", "Doubtful")
    assert s["shariah_illiquid_note"] == ("fixed assets are 15% of assets and there is no CWIP figure "
                                          "(needs at least 20%)")


def test_rule4_fixed_assets_alone_at_the_limit_pass_without_cwip():
    s = _screen(fixed_assets=250.0, cwip=np.nan)
    assert s["shariah_illiquid"] == "pass"
    assert s["shariah_illiquid_note"] == "illiquid assets are at least 25% of assets (needs at least 20%)"


def test_rule4_no_fixed_assets_figure_is_doubtful():
    s = _screen(fixed_assets=np.nan)
    assert (s["shariah_illiquid"], s["shariah_status"]) == ("unknown", "Doubtful")
    assert s["shariah_reason"] == "no fixed-assets figure in the data"        # the brief's own example


# ── 5. Rule 5 — net liquid assets ≤ market cap ───────────────────────────────────────────────────────
def test_rule5_pb_of_one_or_more_always_passes():
    """NLA = net worth − (FA + CWIP + inventory) ≤ net worth = market cap / P/B ≤ market cap."""
    rows = [{"price_to_book": pb, "market_cap": mc, "fixed_assets": fa, "cwip": cw, "inventory": inv}
            for pb, mc, fa, cw, inv in itertools.product((1.0, 1.01, 3.0, 40.0), (5.0, 2000.0, 1e6),
                                                          (0.0, 300.0, np.nan), (0.0, np.nan), (0.0, 50.0, np.nan))]
    out = _screen_many(rows)
    assert out["shariah_nla"].eq("pass").all()


def test_rule5_total_liabilities_is_never_read():
    """The vendor's 'Total Liabilities' is the balance-sheet TOTAL (= total assets on every row)."""
    a = compute_shariah(pd.DataFrame([{**_CLEAN, "total_liabilities": np.nan}]))
    b = compute_shariah(pd.DataFrame([{**_CLEAN, "total_liabilities": 1e15}]))
    pd.testing.assert_frame_equal(a[list(SHARIAH_COLUMNS)], b[list(SHARIAH_COLUMNS)])
    src = open(os.path.join(_ROOT, "core", "shariah.py"), encoding="utf-8").read()
    assert "total_liabilities" not in src


def test_rule5_proven_fail_needs_every_piece():
    # net worth 100 / 0.2 = 500; NLA = 500 − 200 = 300 > market cap 100
    base = dict(market_cap=100.0, price_to_book=0.2, fixed_assets=200.0, cwip=0.0, inventory=0.0)
    s = _screen(**base)
    assert (s["shariah_nla"], s["shariah_status"]) == ("fail", "Non-Shariah")
    assert s["shariah_reason"] == "net liquid assets are 300% of market cap (limit 100%)"
    for gap in ("cwip", "inventory"):
        s = _screen(**{**base, gap: np.nan})
        assert s["shariah_nla"] == "unknown", gap
    # inventory is illiquid for this rule: 500 − 200 − 250 = 50, inside the market cap of 100
    # (found by the mutation check, 2026-10-16 — every case above held no inventory)
    assert _screen(**{**base, "inventory": 250.0})["shariah_nla"] == "pass"


def test_rule5_needs_a_positive_net_worth():
    for over in ({"price_to_book": np.nan}, {"price_to_book": 0.0}, {"price_to_book": -2.0},
                 {"market_cap": 0.0}, {"market_cap": np.nan}):
        assert _screen(**over)["shariah_nla"] == "unknown", over


# ── 6. Rule 6 — non-compliant investments < 33% of total assets (provable pass, never a fail) ─────────
def test_rule6_never_non_shariah():
    # cash 700 of 1,000: cash + unexplained assets 70%; every other rule passes — the investments are for review
    s = _screen(fixed_assets=300.0, inventory=0.0, days_receivable=0.0, cash_equivalents=700.0)
    assert (s["shariah_investments"], s["shariah_status"]) == ("review", "Review")
    assert s["shariah_reason"] == ("investments are not in the data; cash + unexplained assets are up to "
                                   "70% of assets (limit 33%)")
    sweep = _screen_many([{"cash_equivalents": c, "fixed_assets": f} for c in (0.0, 500.0, 990.0)
                          for f in (200.0, 600.0)])
    assert not sweep["shariah_investments"].eq("fail").any()


def test_rule6_at_exactly_the_limit_is_for_review():
    s = _screen(inventory=70.0, days_receivable=36.5)        # 1,000 − 500 − 70 − 100 = 330 → 33%
    assert s["shariah_investments"] == "review"
    assert _screen(inventory=80.0, days_receivable=36.5)["shariah_investments"] == "pass"   # 32%


def test_rule6_without_total_assets_is_unknown():
    """No total assets is a MISSING figure, not a review item."""
    for ta in (0.0, np.nan):
        assert _screen(total_assets=ta)["shariah_investments"] == "unknown"


def test_rule6_receivables_use_the_balance_sheets_own_revenue_year():
    """Revenue 2,000 now, 1,000 a year back, 73 days receivable. Same-year balance sheet: receivables 400,
    cash + unexplained 20% → pass. A balance sheet a year behind: receivables 200 → 40% → unknown."""
    base = dict(revenue=2000.0, revenue_1yb=1000.0, fixed_assets=300.0, cash_equivalents=0.0, ebit=150.0)
    assert _screen(**base, cf_year_lag=0.0)["shariah_investments_pct"] == pytest.approx(20.0)
    s = _screen(**base, cf_year_lag=1.0)
    assert s["shariah_investments_pct"] == pytest.approx(40.0) and s["shariah_investments"] == "review"


def test_rule6_an_unknown_year_counts_receivables_as_zero():
    base = dict(revenue=2000.0, revenue_1yb=1000.0, fixed_assets=300.0, cash_equivalents=0.0, ebit=150.0)
    assert _screen(**base, cf_year_lag=np.nan)["shariah_investments_pct"] == pytest.approx(60.0)
    # a pass that needs no receivables still passes with the year unknown
    assert _screen(cf_year_lag=np.nan, fixed_assets=700.0)["shariah_investments"] == "pass"
    # a revenue that is not positive, or no days figure, leaves receivables unknown too
    for over in ({"revenue_1yb": -10.0, "cf_year_lag": 1.0}, {"days_receivable": np.nan}):
        assert _screen(**{**base, **over})["shariah_investments_pct"] > 40.0, over


# ── Status: precedence and reasons ───────────────────────────────────────────────────────────────────
def test_a_proven_fail_beats_a_missing_figure():
    s = _screen(debt=500.0, fixed_assets=np.nan, price_to_book=np.nan)
    assert s["shariah_status"] == "Non-Shariah"
    assert s["shariah_reason"] == "debt is 50% of assets (limit 33%)"


def test_the_non_shariah_reason_is_the_first_fail_in_the_order_1_3_4_5():
    assert _screen(industry="Hotels", debt=500.0)["shariah_reason"] == "prohibited business: Hotels"
    assert _screen(debt=500.0, fixed_assets=100.0)["shariah_reason"].startswith("debt is ")
    s = _screen(fixed_assets=100.0, market_cap=100.0, price_to_book=0.2)          # 4 and 5 both fail
    assert s["shariah_reason"].startswith("illiquid assets are ")


def test_a_missing_figure_beats_a_review_item_and_each_status_names_its_own_first_rule():
    """Doubtful's reason is the first MISSING figure in the order 1-6; Review's is the first review item."""
    s = _screen(ebit=900.0, fixed_assets=np.nan)                  # 2 for review, 4 missing
    assert (s["shariah_status"], s["shariah_reason"]) == ("Doubtful", "no fixed-assets figure in the data")
    s = _screen(industry="Restaurants", cash_equivalents=900.0)   # 1 and 6 for review
    assert (s["shariah_status"], s["shariah_reason"]) == ("Review", "Restaurants — awaiting a scholar's ruling")
    s = _screen(ebit=900.0, cash_equivalents=900.0)               # 2 and 6 for review: Rule 2 first
    assert s["shariah_status"] == "Review" and s["shariah_reason"].startswith("other income is at least")
    s = _screen(fixed_assets=np.nan, cash_equivalents=900.0)      # 4 missing, 6 for review
    assert (s["shariah_status"], s["shariah_reason"]) == ("Doubtful", "no fixed-assets figure in the data")
    s = _screen(industry=None, cash_equivalents=900.0)            # a blank industry is a missing figure
    assert (s["shariah_status"], s["shariah_reason"]) == ("Doubtful", "no industry in the data")


def test_the_reason_is_always_the_deciding_rules_own_note():
    rows = [{}, {"debt": 500.0}, {"fixed_assets": np.nan}, {"industry": "Hotels"}, {"ebit": 900.0},
            {"cash_equivalents": 900.0}, {"market_cap": 100.0, "price_to_book": 0.2}, {"industry": "Restaurants"}]
    out = _screen_many(rows)
    notes = out[[f"shariah_{k}_note" for k in SHARIAH_RULES]]
    for i, r in out.iterrows():
        if r["shariah_status"] == "Shariah":
            assert r["shariah_reason"] == SHARIAH_PASS_REASON
        else:
            assert r["shariah_reason"] in set(notes.loc[i]), i


# ── The operator's rulings: figures read from the annual report (the operator's choice, 2026-10-16) ───
# PRISM applies its own rules to the operator's figure exactly as to the sheet's, so one rulebook covers every
# stock. A ruling is tied to the balance sheet it was read against (the sheet's total assets at the time):
# when that changes, the ruling lapses and the stock reads at best Review — "re-check".
_RULED = "NSE:TEST"
_SOURCE = "Annual report FY26, notes 6 and 12"


def _rule(monkeypatch, total_assets=1000.0, **figures):
    monkeypatch.setitem(config.SHARIAH_RULINGS, _RULED,
                        {"checked": "2026-10-20", "total_assets": total_assets, "source": _SOURCE, **figures})


_YOURS = f" — your figure ({_SOURCE}, checked 2026-10-20)"


def test_a_ruled_figure_settles_a_rule_the_sheet_cannot(monkeypatch):
    assert _screen(cash_equivalents=900.0)["shariah_status"] == "Review"            # Rule 6: up to 90%
    _rule(monkeypatch, investments_pct=12.0)
    s = _screen(cash_equivalents=900.0)
    assert (s["shariah_investments"], s["shariah_status"]) == ("pass", "Shariah")
    assert s["shariah_investments_pct"] == pytest.approx(12.0)
    assert s["shariah_investments_note"] == "non-compliant investments are 12% of assets (limit 33%)" + _YOURS
    assert s["shariah_ruling"] == f"your figures: {_SOURCE}, checked 2026-10-20"


def test_a_ruled_figure_can_prove_a_fail(monkeypatch):
    _rule(monkeypatch, investments_pct=45.0)
    s = _screen()
    assert (s["shariah_investments"], s["shariah_status"]) == ("fail", "Non-Shariah")
    assert s["shariah_reason"] == "non-compliant investments are 45% of assets (limit 33%)" + _YOURS
    _rule(monkeypatch, income_pct=9.0)
    s = _screen()
    assert (s["shariah_income"], s["shariah_status"]) == ("fail", "Non-Shariah")
    assert s["shariah_reason"] == "impermissible income is 9% of revenue (limit 5%)" + _YOURS
    # the limits hold at their edges for a ruled figure too (found by the mutation check, 2026-10-16)
    _rule(monkeypatch, investments_pct=33.0)
    assert _screen()["shariah_investments"] == "fail", "Rule 6 needs BELOW 33%; exactly 33% is not"
    _rule(monkeypatch, investments_pct=32.9, income_pct=5.0)
    s = _screen()
    assert (s["shariah_investments"], s["shariah_income"]) == ("pass", "pass"), "Rule 2's 5% is 'at most'"


def test_a_ruled_figure_replaces_the_sheets_figure_for_its_rule(monkeypatch):
    """The vendor's Debt can include leases and interest-free loans; the annual report says what bears interest."""
    assert _screen(debt=340.0)["shariah_status"] == "Non-Shariah"
    _rule(monkeypatch, debt_pct=25.0)
    s = _screen(debt=340.0)
    assert (s["shariah_debt"], s["shariah_status"]) == ("pass", "Shariah")
    assert s["shariah_debt_note"] == "debt is 25% of assets (limit 33%)" + _YOURS
    _rule(monkeypatch, illiquid_pct=15.0)
    assert _screen()["shariah_illiquid"] == "fail", "a figure decides both ways"


def test_the_non_shariah_reason_reads_rules_in_order_once_figures_can_fail(monkeypatch):
    _rule(monkeypatch, income_pct=9.0)
    assert _screen(debt=500.0)["shariah_reason"].startswith("impermissible income is 9%")


def test_a_business_ruling_decides_a_review_business_and_can_only_tighten_a_listed_one(monkeypatch):
    _rule(monkeypatch, business="pass")
    s = _screen(industry="Quick Service Restaurant - QSR")
    assert (s["shariah_business"], s["shariah_status"]) == ("pass", "Shariah")
    assert s["shariah_business_note"] == "Quick Service Restaurant - QSR — your scholar's ruling: permissible" + _YOURS
    assert _screen(industry="Hotels")["shariah_status"] == "Non-Shariah", "a prohibited industry stays prohibited"
    _rule(monkeypatch, business="fail")
    s = _screen()
    assert (s["shariah_business"], s["shariah_status"]) == ("fail", "Non-Shariah")
    assert s["shariah_reason"] == "Speciality Chemicals — your scholar's ruling: prohibited" + _YOURS
    # a "pass" ruling on a business the sheet already passes decides nothing: the sheet's own note stands
    # (found by the mutation check, 2026-10-16 — the guard on `sheet_unknown` had no test)
    _rule(monkeypatch, business="pass")
    assert _screen()["shariah_business_note"] == "Speciality Chemicals — not on PRISM's prohibited lists"


def test_a_ruling_lapses_when_the_balance_sheet_changes(monkeypatch):
    _rule(monkeypatch, total_assets=900.0, investments_pct=12.0)                     # read against 900
    s = _screen()                                                                    # the sheet now says 1,000
    assert (s["shariah_status"], s["shariah_investments"]) == ("Review", "pass"), "the figure is ignored"
    assert s["shariah_reason"] == ("your figures of 2026-10-20 were read against an older balance sheet — "
                                   "re-check")
    assert s["shariah_ruling"] == s["shariah_reason"]
    s = _screen(debt=500.0)
    assert (s["shariah_status"], s["shariah_reason"]) == ("Non-Shariah", "debt is 50% of assets (limit 33%)")
    s = _screen(fixed_assets=np.nan)                         # a missing figure still beats the re-check
    assert (s["shariah_status"], s["shariah_reason"]) == ("Doubtful", "no fixed-assets figure in the data")
    # beside a review item the re-check is still the reason: re-reading the report settles both
    # (found by the mutation check, 2026-10-16 — nothing pinned this order)
    s = _screen(cash_equivalents=900.0)
    assert (s["shariah_status"], s["shariah_investments"]) == ("Review", "review")
    assert s["shariah_reason"].endswith("older balance sheet — re-check")


def test_a_ruling_read_against_a_blank_total_holds_until_a_total_appears(monkeypatch):
    _rule(monkeypatch, total_assets=None, debt_pct=10.0, illiquid_pct=40.0, investments_pct=10.0)
    s = _screen(total_assets=np.nan)
    assert s["shariah_status"] == "Shariah" and s["shariah_ruling"].startswith("your figures:")
    assert _screen()["shariah_reason"].endswith("older balance sheet — re-check")


def test_no_ruling_means_an_empty_ruling_column():
    assert _screen()["shariah_ruling"] == ""
    assert _screen(company_id="NSE:SOMEONE_ELSE")["shariah_ruling"] == ""


def test_every_ruling_is_well_formed():
    """Each entry: an exchange-prefixed company id; the day it was checked (not in the future); the sheet's total
    assets it was read against (a positive ₹ Cr figure, or None when the sheet had none); where the figures come
    from; and at least one ruling — a figure in percent (0-100) or a business verdict."""
    from datetime import date
    figures = {f"{k}_pct" for k in ("income", "debt", "illiquid", "investments")}
    for cid, r in config.SHARIAH_RULINGS.items():
        assert re.fullmatch(r"(NSE|BSE):\S+", cid), cid
        assert {"checked", "total_assets", "source"} <= set(r) <= {"checked", "total_assets", "source", "business"} | figures, cid
        assert date.fromisoformat(r["checked"]) <= date.today(), cid
        assert r["total_assets"] is None or (isinstance(r["total_assets"], (int, float)) and r["total_assets"] > 0), cid
        assert len(r["source"].strip()) >= 10, f"{cid}: say where the figures come from"
        assert set(r) & (figures | {"business"}), f"{cid}: a ruling needs a figure or a business verdict"
        assert all(0.0 <= r[f] <= 100.0 for f in figures & set(r)), f"{cid}: figures are percent, 0-100"
        assert r.get("business", "pass") in ("pass", "fail"), cid


# ── What to check: every rule PRISM cannot decide says what to look up, with the rupee limit ────────────
def test_every_open_rule_says_what_to_check_with_the_rupee_limit():
    """For review (investments, other income, a business ruling) and for a missing figure alike."""
    s = _screen(cash_equivalents=900.0, fixed_assets=np.nan, ebit=900.0, industry="Restaurants")
    assert s["shariah_investments_check"] == ("interest-bearing investments and deposits (cash included) must be "
                                              "below ₹330 Cr (33% of ₹1,000 Cr total assets)")
    assert s["shariah_illiquid_check"] == "fixed assets + CWIP must be at least ₹200 Cr (20% of ₹1,000 Cr total assets)"
    assert s["shariah_income_check"] == ("interest and other impermissible income must be at most ₹50 Cr "
                                         "(5% of ₹1,000 Cr revenue)")
    assert s["shariah_business_check"] == "your scholar's ruling on this business"
    assert _screen(industry=None)["shariah_business_check"] == "what the company's business is (no industry in the data)"
    assert _screen(debt=np.nan)["shariah_debt_check"] == ("interest-bearing debt must be at most ₹330 Cr "
                                                          "(33% of ₹1,000 Cr total assets)")


def test_a_rule_that_is_decided_has_nothing_to_check():
    s = _screen(debt=500.0)
    assert [s[f"shariah_{k}_check"] for k in SHARIAH_RULES] == [""] * 6


def test_a_check_without_a_base_figure_still_names_the_limit():
    s = _screen(total_assets=np.nan)
    assert s["shariah_debt_check"] == "interest-bearing debt must be at most 33% of total assets (no total-assets figure in the data)"
    assert _screen(revenue=np.nan)["shariah_income_check"] == ("interest and other impermissible income must be at "
                                                               "most 5% of revenue (no revenue figure in the data)")


def test_a_fact_is_its_note_without_the_limit_in_brackets():
    """The Tear-Sheet card shows each rule's limit in its own column; the note's bracketed limit is dropped there,
    and only there — a ruling's '(source, checked …)' stays."""
    from core.shariah import shariah_fact
    assert shariah_fact("debt is 41% of assets (limit 33%)") == "debt is 41% of assets"
    assert shariah_fact("illiquid assets are 18% of assets (needs at least 20%)") == "illiquid assets are 18% of assets"
    assert shariah_fact("other income is at least 3% of revenue (limit 5%) — interest income is not in the data; "
                        "purify") == "other income is at least 3% of revenue — interest income is not in the data; purify"
    yours = "debt is 25% of assets (limit 33%) — your figure (Annual report FY26, checked 2026-10-20)"
    assert shariah_fact(yours) == "debt is 25% of assets — your figure (Annual report FY26, checked 2026-10-20)"
    assert shariah_fact("no fixed-assets figure in the data") == "no fixed-assets figure in the data"
    assert shariah_fact(None) == "" and shariah_fact(np.nan) == ""


# ── Lists: integrity and the classification guard ─────────────────────────────────────────────────────
def test_every_listed_industry_exists_in_the_map():
    """A typo would silently match nothing."""
    for name in sorted(SHARIAH_PROHIBITED_INDUSTRIES | SHARIAH_REVIEW_INDUSTRIES | SHARIAH_ALLOWED_NAME_MATCHES):
        assert name in INDUSTRY_TIER, f"{name!r} is not an industry the map knows"


def test_every_review_sector_exists_in_prisms_sector_vocabulary():
    from core.taxonomy import VOCABULARY
    for name in sorted(SHARIAH_REVIEW_SECTORS):
        assert name in VOCABULARY["sector"], name


def test_the_lists_are_disjoint():
    lists = (SHARIAH_PROHIBITED_INDUSTRIES, SHARIAH_REVIEW_INDUSTRIES, SHARIAH_ALLOWED_NAME_MATCHES)
    for a, b in itertools.combinations(lists, 2):
        assert not (a & b), sorted(a & b)


def test_every_map_industry_with_a_flagged_word_is_classified_exactly_once():
    """A regenerated map cannot add a brewery unnoticed: every industry whose name holds one of these
    words must sit in exactly one of PROHIBITED, REVIEW or the explicit ALLOWED list."""
    pat = re.compile("|".join(_KEYWORDS), re.I)
    hits = sorted(i for i in INDUSTRY_TIER if pat.search(i))
    assert hits, "the keyword guard matched nothing — the map changed shape"
    for name in hits:
        n = sum(name in s for s in (SHARIAH_PROHIBITED_INDUSTRIES, SHARIAH_REVIEW_INDUSTRIES,
                                    SHARIAH_ALLOWED_NAME_MATCHES))
        assert n == 1, f"{name!r} holds a flagged word and sits in {n} lists — classify it"


def test_the_lists_are_the_operators_final_lists():
    assert SHARIAH_PROHIBITED_INDUSTRIES == frozenset([
        "Alcoholic Beverages", "Cigarettes & Tobacco Products", "Hotels", "Hotels - Resorts", "Resorts",
        "Entertainment & Media", "Music Licensing", "Aerospace & Defence - Equipments"])
    assert SHARIAH_REVIEW_INDUSTRIES == frozenset([
        "Restaurants", "Quick Service Restaurant - QSR", "Airport Restaurants and Lounges", "Amusement Parks",
        "Chemicals - Organic - Alcohol Based", "Real Estate Investment Trusts", "Infrastructure Investment Trusts",
        "Infra/Real Estate Investment Trust"])
    assert SHARIAH_REVIEW_SECTORS == frozenset(["Aerospace & Defence"])
    assert SHARIAH_ALLOWED_NAME_MATCHES == frozenset([
        "Packaging - Films", "Plastics - Sheets/Films", "Detergents/Intermediates", "Dyes - Intermediate"])


# ── One source: config.py ─────────────────────────────────────────────────────────────────────────────
def test_the_thresholds_are_the_operators():
    assert {k: r["limit"] for k, r in SHARIAH_RULES.items()} == {
        "business": None, "income": 0.05, "debt": 0.33, "illiquid": 0.20, "nla": None, "investments": 0.33}
    assert [r["n"] for r in SHARIAH_RULES.values()] == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("rule,limit,over,before,after", [
    ("debt", 0.50, {"debt": 400.0}, "fail", "pass"),
    ("illiquid", 0.60, {}, "pass", "fail"),
    ("income", 0.01, {"ebit": 230.0}, "pass", "review"),
    ("investments", 0.10, {}, "pass", "review"),
])
def test_patching_a_config_threshold_moves_the_result(monkeypatch, rule, limit, over, before, after):
    assert _screen(**over)[f"shariah_{rule}"] == before
    monkeypatch.setitem(SHARIAH_RULES[rule], "limit", limit)
    assert _screen(**over)[f"shariah_{rule}"] == after


def test_the_engine_holds_configs_own_objects_not_copies(monkeypatch):
    import core.shariah as sh
    for name in ("SHARIAH_RULES", "SHARIAH_PROHIBITED_INDUSTRIES", "SHARIAH_REVIEW_INDUSTRIES",
                 "SHARIAH_REVIEW_SECTORS", "SHARIAH_PROHIBITED_COMPANIES", "SHARIAH_STATUSES", "SHARIAH_RULINGS"):
        assert getattr(sh, name) is getattr(config, name), name
    monkeypatch.setattr(sh, "SHARIAH_PROHIBITED_INDUSTRIES", frozenset(["Speciality Chemicals"]))
    assert _screen()["shariah_status"] == "Non-Shariah"


# ── Static guards ─────────────────────────────────────────────────────────────────────────────────────
def test_core_shariah_is_vectorized():
    """No row loops, no lambdas, no apply(axis=1) (CLAUDE.md §5 vectorization mandate)."""
    tree = ast.parse(open(os.path.join(_ROOT, "core", "shariah.py"), encoding="utf-8").read())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Lambda, ast.For, ast.While, ast.AsyncFor)):
            bad.append(f"{type(node).__name__} at line {node.lineno}")
        if isinstance(node, ast.Attribute) and node.attr in ("iterrows", "itertuples", "iteritems"):
            bad.append(f".{node.attr} at line {node.lineno}")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "apply"):
            bad.append(f".apply at line {node.lineno}")
    assert not bad, bad


def test_no_scorer_gate_or_framework_reads_the_screen():
    """Display and filter only: no module in core/ but the screen itself names a shariah_ column."""
    offenders = []
    for f in sorted(os.listdir(os.path.join(_ROOT, "core"))):
        if not f.endswith(".py") or f == "shariah.py":
            continue
        tree = ast.parse(open(os.path.join(_ROOT, "core", f), encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "shariah_" in node.value:
                offenders.append(f"{f}:{node.lineno} {node.value!r}")
    assert not offenders, offenders


def test_no_scan_recipe_reaches_the_screen():
    """Kite refuses a list whose recipe fingerprint changes; no recipe, or anything it reaches, reads it."""
    from ui.ui_scans import SCANS, _dependencies, _recipe_functions
    for scan in SCANS:
        deps = _dependencies([*_recipe_functions(scan), *scan.get("inputs", ())], scan.get("values"),
                             scan.get("skip", ()))
        assert not [k for k, v in deps.items() if "shariah" in k.lower() or "shariah" in v.lower()], scan["key"]


def test_readers_match_the_status_by_equality_never_contains():
    """'Shariah' sits inside 'Non-Shariah': a substring filter would return both."""
    files = [os.path.join(_ROOT, "app.py")] + [os.path.join(_ROOT, "ui", f)
                                               for f in sorted(os.listdir(os.path.join(_ROOT, "ui")))
                                               if f.endswith(".py")]
    bad = []
    for f in files:
        for n, line in enumerate(open(f, encoding="utf-8"), 1):
            if "shariah" in line.lower() and re.search(r"\.contains\(|\.startswith\(|\.endswith\(|\.find\(", line):
                bad.append(f"{os.path.basename(f)}:{n}")
    assert not bad, bad


# ── Zero scoring impact, on real data ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def ab():
    """The full pipeline on the local workbook, with the screen and with its call made a no-op."""
    if not os.path.isdir(_DATA_DIR):
        pytest.skip("local CSV data absent (code-only checkout) — needs real data")
    import core.data_engine as de
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(io.StringIO()):
        datasets = de.load_all_csvs("local")
        with_screen = run_scoring_pipeline(de._clean_from(copy.deepcopy(datasets)))
        real = de.compute_shariah
        de.compute_shariah = lambda df: df
        try:
            without = run_scoring_pipeline(de._clean_from(copy.deepcopy(datasets)))
        finally:
            de.compute_shariah = real
    return with_screen, without


def test_every_pre_existing_column_is_byte_identical(ab):
    a, b = ab
    assert set(a.columns) - set(b.columns) == set(SHARIAH_COLUMNS)
    assert not set(b.columns) - set(a.columns)
    assert a.index.equals(b.index)
    moved = []
    for c in b.columns:
        try:
            pd.testing.assert_series_equal(a[c], b[c], check_exact=True)
        except AssertionError:
            moved.append(c)
    assert not moved, f"the screen moved pre-existing columns: {moved[:10]}"
    assert a.attrs == b.attrs


def test_every_scan_list_is_identical(ab):
    from ui.ui_scans import SCANS
    a, b = ab
    for scan in SCANS:
        pd.testing.assert_series_equal(scan["mask"](a), scan["mask"](b), check_names=False)


def test_every_label_is_alive_on_real_data(ab):
    a, _ = ab
    share = a["shariah_status"].value_counts(normalize=True)
    assert set(share.index) == set(SHARIAH_STATUSES)
    assert all(0.05 < share[s] < 0.70 for s in SHARIAH_STATUSES), share.to_dict()


def test_no_status_word_collides_with_another_layers_labels(ab):
    """NO WORD IN TWO LAYERS (tests/test_verdict_vocabulary.py), for the screen's words."""
    from tests.test_verdict_vocabulary import _label_columns
    a, _ = ab
    cols = [c for c in _label_columns(a) if not c.startswith("shariah_")]
    clashes = []
    for c in cols:
        for v in sorted(set(a[c].dropna().astype(str))):
            low = v.lower()
            for w in ("shariah", "doubtful", "review"):
                if w in low:
                    clashes.append(f"{w} ~ {c}={v!r}")
    assert not clashes, clashes
