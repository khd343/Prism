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
    describe themselves as defence suppliers (warships, missile propellants, "defence-focused"). The operator's
    ruling of 2026-10-17 then cleared the sector and its "Aerospace & Defence - Equipments" industry: they are
    screened by the six rules like any business (15 stocks Review -> Shariah on the 16 Oct sheet);
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


def test_rule1_aerospace_and_defence_is_screened_like_any_business_and_its_prohibited_industry_still_fails():
    """The operator's ruling, 2026-10-17: the sector and its Equipments industry no longer await a ruling."""
    for industry in ("Industrial Explosives", "Aerospace & Defence - Equipments"):
        s = _screen(industry=industry, sector="Aerospace & Defence")
        assert (s["shariah_business"], s["shariah_status"]) == ("pass", "Shariah"), industry
        assert s["shariah_reason"] == SHARIAH_PASS_REASON
    s = _screen(industry="Alcoholic Beverages", sector="Aerospace & Defence")
    assert (s["shariah_business"], s["shariah_status"]) == ("fail", "Non-Shariah"), "a prohibited industry fails"


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
    """Market cap 100: the balance-sheet ceiling (1,000 − 600 = 400) cannot settle it, so only net worth can."""
    for over in ({"price_to_book": np.nan}, {"price_to_book": 0.0}, {"price_to_book": -2.0},
                 {"market_cap": 0.0}, {"market_cap": np.nan}):
        assert _screen(**{"market_cap": 100.0, **over})["shariah_nla"] == "unknown", over


def test_rule5_a_net_worth_above_total_assets_is_no_reading():
    """Net worth = total assets − liabilities, so it can never exceed total assets. A P/B implying more is a vendor
    error (EPack Prefab, 15 Oct: P/B 0.34 → ₹7,133 Cr on a ₹1,427 Cr balance sheet) and proves nothing — the screen
    once made it a 266% fail."""
    s = _screen(market_cap=100.0, price_to_book=0.01)                  # net worth 10,000 on total assets 1,000
    assert (s["shariah_nla"], s["shariah_status"]) == ("unknown", "Doubtful")
    assert s["shariah_nla_note"] == "the P/B in the data implies a net worth above total assets — a vendor error"
    assert s["shariah_nla_check"] == "the book value: the P/B in the data implies a net worth above total assets"
    assert _screen(market_cap=100.0, price_to_book=0.1)["shariah_nla"] == "fail", "exactly total assets is a reading"


def test_rule5_the_balance_sheet_ceiling_passes_without_any_pb():
    """NLA can never exceed total assets − (fixed assets + CWIP + inventory): liabilities are never below zero. At or
    under the market cap that ceiling passes Rule 5 on the balance sheet alone, whatever the P/B."""
    s = _screen(price_to_book=np.nan)                                   # 1,000 − 600 = 400 against a cap of 2,000
    assert s["shariah_nla"] == "pass" and s["shariah_nla_pct"] == pytest.approx(20.0)
    assert s["shariah_nla_note"] == ("net liquid assets are at most 20% of market cap — total assets less fixed "
                                     "assets, CWIP and inventory (limit 100%)")
    epack = dict(market_cap=2425.11, price_to_book=0.34, total_assets=1426.90, fixed_assets=357.07, cwip=60.35,
                 inventory=265.75)                                       # the 15 Oct sheet's own figures
    s = _screen(**epack)
    assert s["shariah_nla"] == "pass" and s["shariah_nla_pct"] == pytest.approx(743.73 / 2425.11 * 100)
    assert _screen(price_to_book=np.nan, inventory=np.nan)["shariah_nla"] == "pass", "a missing piece counts as 0"
    assert _screen(price_to_book=np.nan, market_cap=399.0)["shariah_nla"] == "unknown", "the ceiling never fails"


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


# ── Your lists: what you say in config is final — until the data passes all six rules (the operator, 2026-10-16) ──
# "If I said it is Shariah then Shariah, if not Shariah then Non-Shariah — whatever I said in config, that's it.
# When a new quarter comes in, if everything qualifies give Shariah; if not, let the statement be what I said."
# "If in config it is Non-Shariah but after a new result all six rules pass, give it Shariah."
# One rule, read after the six: a stock passing all six on the latest data is Shariah; otherwise your list decides
# (SHARIAH_APPROVED Shariah, SHARIAH_REJECTED Non-Shariah, SHARIAH_REVIEW_COMPANIES Review); otherwise the data's
# own label. Ticker only, no dates. A haram BUSINESS the data cannot see is not a list but Rule 1 itself
# (SHARIAH_PROHIBITED_COMPANIES): a new quarter cannot change a business, so it never clears.
_ME = "NSE:TEST"


def _list(monkeypatch, name):
    import core.shariah as sh
    monkeypatch.setattr(sh, name, frozenset([_ME]))


def test_your_shariah_list_settles_what_prisms_data_cannot_show(monkeypatch):
    assert _screen(cash_equivalents=900.0)["shariah_status"] == "Review"            # Rule 6: up to 90%
    _list(monkeypatch, "SHARIAH_APPROVED")
    s = _screen(cash_equivalents=900.0, ebit=900.0)
    assert (s["shariah_status"], s["shariah_reason"]) == ("Shariah", "on your Shariah list")
    assert s["shariah_your_list"] == "on your Shariah list"
    assert s["shariah_investments"] == "review", "the rule still says what the data shows"


def test_your_shariah_list_settles_a_missing_figure_too(monkeypatch):
    assert _screen(fixed_assets=np.nan)["shariah_status"] == "Doubtful"
    _list(monkeypatch, "SHARIAH_APPROVED")
    s = _screen(fixed_assets=np.nan)
    assert (s["shariah_status"], s["shariah_illiquid"]) == ("Shariah", "unknown")


def test_your_shariah_list_is_final_and_a_fail_in_the_data_is_never_silent(monkeypatch):
    _list(monkeypatch, "SHARIAH_APPROVED")
    s = _screen(debt=500.0)
    assert s["shariah_status"] == "Shariah", "what you say in config is final"
    assert s["shariah_reason"] == "on your Shariah list — though the data shows: debt is 50% of assets (limit 33%)"
    assert s["shariah_your_list"] == s["shariah_reason"]
    assert s["shariah_debt"] == "fail", "the card still shows the failing rule"
    assert _screen(industry="Hotels")["shariah_status"] == "Shariah"


def test_when_a_new_quarter_passes_all_six_the_reason_says_so(monkeypatch):
    _list(monkeypatch, "SHARIAH_APPROVED")
    s = _screen()
    assert (s["shariah_status"], s["shariah_reason"]) == ("Shariah", SHARIAH_PASS_REASON)
    assert s["shariah_your_list"] == "on your Shariah list"


def test_your_shariah_list_clears_a_business_awaiting_a_ruling(monkeypatch):
    _list(monkeypatch, "SHARIAH_APPROVED")
    for industry in ("Quick Service Restaurant - QSR", "Restaurants"):
        s = _screen(industry=industry)
        assert (s["shariah_business"], s["shariah_status"]) == ("review", "Shariah"), industry


def test_your_non_shariah_list_holds_until_the_data_passes_all_six(monkeypatch):
    _list(monkeypatch, "SHARIAH_REJECTED")
    for over in ({"cash_equivalents": 900.0}, {"fixed_assets": np.nan}, {"debt": 500.0}):
        s = _screen(**over)
        assert (s["shariah_status"], s["shariah_reason"]) == ("Non-Shariah", "on your Non-Shariah list"), over
    s = _screen()                                           # a new result passes every rule
    assert (s["shariah_status"], s["shariah_reason"]) == ("Shariah", SHARIAH_PASS_REASON)
    assert s["shariah_your_list"] == "on your Non-Shariah list — the latest data passes all six rules"


def test_your_review_list_holds_until_the_data_passes_all_six(monkeypatch):
    _list(monkeypatch, "SHARIAH_REVIEW_COMPANIES")
    for over in ({"cash_equivalents": 900.0}, {"fixed_assets": np.nan}, {"debt": 500.0}):
        s = _screen(**over)
        assert (s["shariah_status"], s["shariah_reason"]) == ("Review", "on your review list"), over
    s = _screen()
    assert (s["shariah_status"], s["shariah_reason"]) == ("Shariah", SHARIAH_PASS_REASON)
    assert s["shariah_your_list"] == "on your review list — the latest data passes all six rules"
    assert _screen(company_id="NSE:SOMEONE_ELSE", debt=500.0)["shariah_status"] == "Non-Shariah"


def test_a_haram_business_on_rule_1s_company_list_never_clears():
    """Delta Corp's figures pass every financial rule; its casino is Rule 1 itself, so no new quarter clears it."""
    cid, why = next(iter(sorted(SHARIAH_PROHIBITED_COMPANIES.items())))
    s = _screen(company_id=cid, industry="Miscellaneous", sector="Miscellaneous")
    assert (s["shariah_status"], s["shariah_reason"], s["shariah_your_list"]) == (
        "Non-Shariah", f"prohibited business: {why}", "")


def test_your_lists_never_expire():
    """No dates: nothing in the screen reads the clock, so a company stays on your list until you remove it."""
    src = open(os.path.join(_ROOT, "core", "shariah.py"), encoding="utf-8").read()
    for clock in ("today(", "now(", "datetime", "Timestamp", "DateOffset", "to_datetime"):
        assert clock not in src, clock


def test_no_list_means_an_empty_your_list_column():
    assert _screen()["shariah_your_list"] == ""
    assert _screen(company_id="NSE:SOMEONE_ELSE")["shariah_your_list"] == ""


def test_every_list_entry_is_well_formed():
    """Company ids, exchange-prefixed ("NSE:SYMBOL", or "BSE:code"); a company sits in at most one list."""
    yours = (config.SHARIAH_APPROVED, config.SHARIAH_REJECTED, config.SHARIAH_REVIEW_COMPANIES)
    for cid in sorted(frozenset().union(*yours)):
        assert re.fullmatch(r"(NSE|BSE):\S+", cid), cid
    for a, b in itertools.combinations([set(x) for x in yours] + [set(SHARIAH_PROHIBITED_COMPANIES)], 2):
        assert not (a & b), f"a company in two lists: {sorted(a & b)}"


# ── Rules you checked yourself: that one rule settled, every other rule still applies (the operator, 2026-10-19) ──
# "Only because of Rule 6 — if in future an issue comes in another rule, PRISM should tell based on that." Rule 6
# (non-compliant investments) and Rule 2 (impermissible income) are the two rules the data cannot prove; the operator
# reads the real figure (their app, the annual report) and lists the ticker. That rule then passes, and the five
# others run as ever, so a fail, a missing figure or another open rule in any later quarter still sets the label.
def _checked(monkeypatch, name):
    import core.shariah as sh
    monkeypatch.setattr(sh, name, frozenset([_ME]))


def test_a_rule_6_check_settles_rule_6_and_nothing_else(monkeypatch):
    assert _screen(cash_equivalents=900.0)["shariah_status"] == "Review"            # Rule 6: up to 90%
    _checked(monkeypatch, "SHARIAH_INVESTMENTS_CLEARED")
    s = _screen(cash_equivalents=900.0)
    assert (s["shariah_investments"], s["shariah_status"]) == ("pass", "Shariah")
    assert s["shariah_investments_note"] == ("non-compliant investments checked by you: below 33% of assets (the data "
                                             "shows only that cash + unexplained assets are up to 90%)")
    assert (s["shariah_investments_check"], s["shariah_your_list"]) == ("", ""), "settled, and not one of your lists"
    assert _screen(cash_equivalents=900.0, debt=500.0)["shariah_status"] == "Non-Shariah", "a later fail decides"
    assert _screen(cash_equivalents=900.0, fixed_assets=np.nan)["shariah_status"] == "Doubtful"
    s = _screen(cash_equivalents=900.0, ebit=900.0)                                    # Rule 2 is still open
    assert (s["shariah_status"], s["shariah_reason"]) == (
        "Review", "other income is at least 70% of revenue — may be interest (limit 5%)")


def test_a_rule_2_check_settles_rule_2_and_nothing_else(monkeypatch):
    assert _screen(ebit=900.0)["shariah_status"] == "Review"                        # other income at least 70%
    _checked(monkeypatch, "SHARIAH_INCOME_CLEARED")
    s = _screen(ebit=900.0)
    assert (s["shariah_income"], s["shariah_status"]) == ("pass", "Shariah")
    assert s["shariah_income_note"] == ("impermissible income checked by you: at most 5% of revenue (the data shows "
                                        "other income of at least 70%) — interest income is not in the data; purify")
    assert _screen(ebit=900.0, cash_equivalents=900.0)["shariah_status"] == "Review", "Rule 6 is still open"
    assert _screen(ebit=900.0, debt=500.0)["shariah_status"] == "Non-Shariah"


def test_a_checked_rule_holds_without_its_figure(monkeypatch):
    _checked(monkeypatch, "SHARIAH_INCOME_CLEARED")
    s = _screen(ebitda=np.nan)
    assert s["shariah_income"] == "pass"
    assert "at most 5% of revenue (the data cannot estimate other income)" in s["shariah_income_note"]
    _checked(monkeypatch, "SHARIAH_INVESTMENTS_CLEARED")
    s = _screen(total_assets=np.nan)
    assert s["shariah_investments"] == "pass"
    assert s["shariah_investments_note"].endswith("(no total-assets figure in the data)")
    assert s["shariah_status"] == "Doubtful", "the missing total assets still leave Rules 3 and 4 unread"


def test_every_checked_rule_entry_is_well_formed():
    """Company ids, exchange-prefixed ("NSE:SYMBOL", or "BSE:code"). A company may sit in both, or in one of your
    lists too: a check settles one rule, a list decides the label."""
    for cid in sorted(config.SHARIAH_INVESTMENTS_CLEARED | config.SHARIAH_INCOME_CLEARED):
        assert re.fullmatch(r"(NSE|BSE):\S+", cid), cid


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
    and only there — any other bracket stays."""
    from core.shariah import shariah_fact
    assert shariah_fact("debt is 41% of assets (limit 33%)") == "debt is 41% of assets"
    assert shariah_fact("illiquid assets are 18% of assets (needs at least 20%)") == "illiquid assets are 18% of assets"
    assert shariah_fact("other income is at least 3% of revenue (limit 5%) — interest income is not in the data; "
                        "purify") == "other income is at least 3% of revenue — interest income is not in the data; purify"
    no_est = "other income cannot be estimated (no revenue, EBIT or EBITDA figure)"
    assert shariah_fact(no_est) == no_est
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
    # The operator's web edit of 2026-10-17 (cbfa906): defence equipment moved to review, industrial alcohol
    # cleared from review (and so named in the allowed list below).
    assert SHARIAH_PROHIBITED_INDUSTRIES == frozenset([
        "Alcoholic Beverages", "Cigarettes & Tobacco Products", "Hotels", "Hotels - Resorts", "Resorts",
        "Entertainment & Media", "Music Licensing"])
    # The operator's ruling of 2026-10-17: defence equipment and the Aerospace & Defence sector no longer await one.
    assert SHARIAH_REVIEW_INDUSTRIES == frozenset([
        "Restaurants", "Quick Service Restaurant - QSR", "Airport Restaurants and Lounges", "Amusement Parks",
        "Real Estate Investment Trusts", "Infrastructure Investment Trusts", "Infra/Real Estate Investment Trust"])
    assert SHARIAH_REVIEW_SECTORS == frozenset()
    assert SHARIAH_ALLOWED_NAME_MATCHES == frozenset([
        "Packaging - Films", "Plastics - Sheets/Films", "Detergents/Intermediates", "Dyes - Intermediate",
        "Chemicals - Organic - Alcohol Based", "Aerospace & Defence - Equipments"])


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
                 "SHARIAH_REVIEW_SECTORS", "SHARIAH_PROHIBITED_COMPANIES", "SHARIAH_STATUSES", "SHARIAH_APPROVED",
                 "SHARIAH_REJECTED", "SHARIAH_REVIEW_COMPANIES", "SHARIAH_INVESTMENTS_CLEARED",
                 "SHARIAH_INCOME_CLEARED"):
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
