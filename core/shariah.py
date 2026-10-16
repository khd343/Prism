"""
Shariah screen (2026-10-16) — a LABEL and a FILTER, never a score.

Every stock gets shariah_status = Shariah / Non-Shariah / Review / Doubtful and a one-line reason, from the six
rules of the KMI-30 / Meezan method (Pakistan Stock Exchange) with the operator's thresholds (config.SHARIAH_RULES).
Each rule returns pass / fail / review / unknown, and a missing figure is never a pass and never a fail (CLAUDE.md
§5, "Unverifiable is not passed"):

    Non-Shariah  a rule FAILS — the reason is the first failing rule in the order 1-6
    Doubtful     no fail, but a figure is MISSING — the reason is the first missing figure in the order 1-6
    Review       no fail, nothing missing, but a rule only the operator's check can settle is open (the sheet
                 cannot prove it, or the business awaits a scholar's ruling), or a ruling lapsed — the reason is
                 the lapse, else the first such rule in the order 1-6 (the operator's fourth label, 2026-10-16)
    Shariah      every rule passes

    1 Business                    fail: the map's financial tier (E), a prohibited industry, a listed company;
                                  review: a review industry or sector; unknown: no industry
    2 Impermissible income ≤ 5%   of revenue — the sheet shows only the LEAST other income: above the limit it
                                  is for review ("may be interest"), never a fail
    3 Interest-bearing debt ≤ 33% of total assets
    4 Illiquid assets ≥ 20%       fixed assets + CWIP, of total assets
    5 Net liquid assets           at most the market cap
    6 Investments < 33%           of total assets — the sheet bounds them from above: over the bound they are
                                  for review, never a fail

What the vendor's sheet can and cannot show (measured on the 15 Oct 2026 sheet, 2,695 stocks):
  * no interest-income line. EBIT includes other income and EBITDA does not, so max(EBIT − EBITDA, 0) is the
    least other income can be (Rule 2).
  * no investments line. Total assets less what the sheet explains (fixed assets, CWIP, inventory, estimated
    receivables), or cash if larger, is the most investments plus cash can be (Rule 6).
  * the vendor's "Total Liabilities" is the balance-sheet TOTAL — equal to total assets on every row — so it
    is never read; net worth is market cap ÷ P/B, only when both are positive (Rule 5).
  * Fixed Assets include goodwill and intangibles, which the method counts as illiquid. A BLANK figure means
    missing, not zero: 100 of the 179 blanks in the ₹500–20,000 Cr window show positive depreciation, and
    the published balance sheets of Kennametal, Castrol and Khazanchi carry fixed assets behind a blank.
  * the balance sheet trails the P&L by a fiscal year on 2,235 stocks in October (cf_year_lag), and the
    vendor's working-capital ratios use the balance sheet's year (inventory turnover × inventory reproduces
    the year-back revenue on 372 of 402). So Rule 6's receivables take the BALANCE SHEET's revenue year;
    where the year cannot be told they count as unknown (0), and a pass then needs no receivables at all.
    (data_engine's *_cf_year columns read an unknown year as the current one; for a certificate that errs
    lenient, so this rule does not.)
  * a negative balance-sheet figure (debt was negative on 2 rows) is a vendor error: read as missing.

What the sheet cannot show, the operator can (config.SHARIAH_RULINGS, the operator's design of 2026-10-16):
figures read in the annual report replace the sheet's for their rule and are judged by the SAME rules, so they
can prove a pass or a fail. Each ruling is tied to the balance sheet it was read against — the sheet's total
assets that day — and lapses, reading at best Review ("re-check"), when that changes. Every rule left undecided says what to look
up, with its rupee limit (shariah_*_check).

Called once from compute_derived_signals, right after cyclicality_tier_code. It adds its columns in one
block and changes none; no scorer, gate, framework or scan recipe reads them (tests/test_shariah_screen.py).
"""
import re

import numpy as np
import pandas as pd

from config import (SHARIAH_PROHIBITED_COMPANIES, SHARIAH_PROHIBITED_INDUSTRIES, SHARIAH_REVIEW_INDUSTRIES,
                    SHARIAH_REVIEW_SECTORS, SHARIAH_RULES, SHARIAH_RULINGS, SHARIAH_STATUSES)

SHARIAH_PASS_REASON = "Passes every rule PRISM can check (interest income is not in the data — purify)"
_PURIFY = "interest income is not in the data; purify"
_RULING = " — awaiting a scholar's ruling"
_NO_TA = "no total-assets figure in the data"
_NUMBERED = ("income", "debt", "illiquid", "nla", "investments")     # the rules with a figure to show

SHARIAH_COLUMNS = (("shariah_status", "shariah_reason", "shariah_ruling")
                   + tuple(f"shariah_{k}" for k in SHARIAH_RULES)
                   + tuple(f"shariah_{k}_pct" for k in _NUMBERED)
                   + tuple(f"shariah_{k}_note" for k in SHARIAH_RULES)
                   + tuple(f"shariah_{k}_check" for k in SHARIAH_RULES))


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """The column as numbers; all-NaN when absent, so a missing input decides nothing."""
    return (pd.to_numeric(df[col], errors="coerce") if col in df.columns
            else pd.Series(np.nan, index=df.index, dtype="float64"))


def _amount(df: pd.DataFrame, col: str) -> pd.Series:
    """A balance-sheet amount, which cannot be negative: a negative figure is read as missing."""
    s = _num(df, col)
    return s.where(s >= 0.0)


def _text(df: pd.DataFrame, col: str) -> pd.Series:
    """The column as stripped text; "" when blank or absent."""
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype=object)
    return df[col].where(df[col].notna(), "").astype(str).str.strip()


def _share(num: pd.Series, den: pd.Series) -> pd.Series:
    """num ÷ den where den > 0, else NaN (the denominator guard)."""
    return pd.Series(np.where(den > 0.0, num / den.where(den > 0.0), np.nan), index=den.index)


def _pct(share: pd.Series, limit: float) -> pd.Series:
    """A share as percent text: whole numbers, or two decimals within half a point of the rule's limit, so a
    figure just past the limit never prints as the limit itself (33.04% is not "33%")."""
    v = np.nan_to_num(share.to_numpy(dtype="float64") * 100.0)
    near = np.abs(v - limit * 100.0) < 0.5
    return pd.Series(np.where(near, np.char.mod("%.2f", v), np.char.mod("%.0f", v)), index=share.index)


def _limit(limit: float) -> str:
    return f"{limit * 100:.0f}%"


def _cr(amount: pd.Series) -> pd.Series:
    """₹ crore as text, thousands grouped ("₹1,264 Cr"); "" where missing."""
    return amount.map("₹{:,.0f} Cr".format).where(amount.notna(), "")


_LIMIT_IN_NOTE = re.compile(r"\s*\((?:limit|needs at least)\b[^)]*\)")


def shariah_fact(note) -> str:
    """A rule's note without its bracketed limit ("debt is 41% of assets (limit 33%)" -> "debt is 41% of assets"),
    for the Tear-Sheet card, which shows the limit in its own column. Kept beside the notes' format."""
    return _LIMIT_IN_NOTE.sub("", note) if isinstance(note, str) else ""


def _ruled(cid: pd.Series, key: str) -> pd.Series:
    """One field of the operator's rulings, per stock (NaN where the stock has no ruling or no such field)."""
    return cid.map({k: r.get(key) for k, r in SHARIAH_RULINGS.items()})


def compute_shariah(df: pd.DataFrame) -> pd.DataFrame:
    """df with the SHARIAH_COLUMNS added in one block; every existing column untouched."""
    idx = df.index
    lim = {k: r["limit"] for k, r in SHARIAH_RULES.items()}          # read per call: config is the one source
    ind, sec, cid = _text(df, "industry"), _text(df, "sector"), _text(df, "company_id")
    ta, debt = _num(df, "total_assets"), _amount(df, "debt")
    fa, cwip, inv, cash = (_amount(df, c) for c in ("fixed_assets", "cwip", "inventory", "cash_equivalents"))
    mc, pb = _num(df, "market_cap"), _num(df, "price_to_book")
    rev, rev_1yb, dso, lag = (_num(df, c) for c in ("revenue", "revenue_1yb", "days_receivable", "cf_year_lag"))
    ebit, ebitda = _num(df, "ebit"), _num(df, "ebitda")
    has_ta = ta > 0.0

    # ── The operator's rulings: live only while the sheet shows the balance sheet they were read against ──
    has_ruling = cid.isin(list(SHARIAH_RULINGS))
    read_ta = pd.to_numeric(_ruled(cid, "total_assets"), errors="coerce")
    current = has_ruling & ((read_ta.isna() & ta.isna()) | (ta - read_ta).abs().le(0.01))
    lapsed = has_ruling & ~current
    source, checked = _ruled(cid, "source").fillna("").astype(str), _ruled(cid, "checked").fillna("").astype(str)
    yours = " — your figure (" + source + ", checked " + checked + ")"
    fig = {k: (pd.to_numeric(_ruled(cid, f"{k}_pct"), errors="coerce") / 100.0).where(current)
           for k in ("income", "debt", "illiquid", "investments")}
    verdict = _ruled(cid, "business").where(current)

    # ── 1 Business ──
    listed_why = cid.map(SHARIAH_PROHIBITED_COMPANIES)
    listed = listed_why.notna()
    prohibited = ind.isin(SHARIAH_PROHIBITED_INDUSTRIES)
    financial = _text(df, "cyclicality_tier_code").eq("E")
    review_ind, review_sec, blank = ind.isin(SHARIAH_REVIEW_INDUSTRIES), sec.isin(SHARIAH_REVIEW_SECTORS), ind.eq("")
    sheet_fail = listed | prohibited | financial
    sheet_unknown = ~sheet_fail & (review_ind | review_sec | blank)
    ruled_fail, ruled_pass = ~sheet_fail & verdict.eq("fail"), sheet_unknown & verdict.eq("pass")
    business = np.select([sheet_fail | ruled_fail, ruled_pass, review_ind | review_sec, blank],
                         ["fail", "pass", "review", "unknown"], "pass")
    named = ind.where(~blank, "this business")
    business_note = np.select(
        [listed, prohibited, financial, ruled_fail, ruled_pass, review_ind, review_sec, blank],
        ["prohibited business: " + listed_why.fillna(""),
         "prohibited business: " + ind,
         "prohibited business: financial services (" + ind.where(~blank, sec) + ")",
         named + " — your scholar's ruling: prohibited" + yours,
         named + " — your scholar's ruling: permissible" + yours,
         ind + _RULING,
         sec + " sector" + _RULING,
         pd.Series("no industry in the data", index=idx)],
        ind + " — not on PRISM's prohibited lists")

    # ── 2 Impermissible income: the sheet shows the least other income; a ruled figure is the real one ──
    seen = _share((ebit - ebitda).clip(lower=0.0), rev)
    income_share = fig["income"].where(fig["income"].notna(), seen)
    income = np.select([fig["income"] > lim["income"], fig["income"].notna(), seen <= lim["income"],
                        seen > lim["income"]], ["fail", "pass", "pass", "review"], "unknown")
    inc_txt, inc_lim = _pct(income_share, lim["income"]), _limit(lim["income"])
    income_note = np.select(
        [fig["income"].notna(), seen.isna(), seen > lim["income"], seen <= 0.0],
        ["impermissible income is " + inc_txt + "% of revenue (limit " + inc_lim + ")" + yours,
         pd.Series("other income cannot be estimated (no revenue, EBIT or EBITDA figure)", index=idx),
         "other income is at least " + inc_txt + "% of revenue — may be interest (limit " + inc_lim + ")",
         pd.Series(f"no other income visible in the data (limit {inc_lim}) — {_PURIFY}", index=idx)],
        "other income is at least " + inc_txt + "% of revenue (limit " + inc_lim + ") — " + _PURIFY)

    # ── 3 Interest-bearing debt ──
    debt_share = fig["debt"].where(fig["debt"].notna(), _share(debt, ta))
    debt_state = np.select([debt_share.isna(), debt_share > lim["debt"]], ["unknown", "fail"], "pass")
    debt_txt = "debt is " + _pct(debt_share, lim["debt"]) + "% of assets (limit " + _limit(lim["debt"]) + ")"
    debt_note = np.select([fig["debt"].notna(), ~has_ta, _num(df, "debt") < 0.0, debt_share.isna()],
                          [debt_txt + yours, pd.Series(_NO_TA, index=idx),
                           pd.Series("the debt figure in the data is negative (a vendor error)", index=idx),
                           pd.Series("no debt figure in the data", index=idx)], debt_txt)

    # ── 4 Illiquid assets: fixed assets + CWIP; CWIP can only add ──
    fa_share, ill_share = _share(fa, ta), _share(fa + cwip, ta)        # ill_share is NaN without CWIP
    ill_fig = fig["illiquid"]
    illiquid = np.select(
        [ill_fig >= lim["illiquid"], ill_fig.notna(), fa_share.isna(), fa_share >= lim["illiquid"],
         ill_share >= lim["illiquid"], ill_share.notna()],
        ["pass", "fail", "unknown", "pass", "pass", "fail"], "unknown")
    need = " of assets (needs at least " + _limit(lim["illiquid"]) + ")"
    fa_txt = _pct(fa_share, lim["illiquid"])
    illiquid_shown = ill_fig.where(ill_fig.notna(), ill_share.where(ill_share.notna(), fa_share))
    illiquid_note = np.select(
        [ill_fig.notna(), ~has_ta, fa_share.isna(), ill_share.isna() & (fa_share >= lim["illiquid"]),
         ill_share.isna()],
        ["illiquid assets are " + _pct(ill_fig, lim["illiquid"]) + "%" + need + yours,
         pd.Series(_NO_TA, index=idx), pd.Series("no fixed-assets figure in the data", index=idx),
         "illiquid assets are at least " + fa_txt + "%" + need,
         "fixed assets are " + fa_txt + "% of assets and there is no CWIP figure (needs at least "
         + _limit(lim["illiquid"]) + ")"],
        "illiquid assets are " + _pct(ill_share, lim["illiquid"]) + "%" + need)

    # ── 5 Net liquid assets = net worth − (fixed assets + CWIP + inventory), at most the market cap ──
    has_nw = (mc > 0.0) & (pb > 0.0)
    nla_max = (mc / pb).where(has_nw) - (fa.fillna(0.0) + cwip.fillna(0.0) + inv.fillna(0.0))
    known = fa.notna() & cwip.notna() & inv.notna()
    nla_share = _share(nla_max, mc)
    nla = np.select([~has_nw, nla_max <= mc, known], ["unknown", "pass", "fail"], "unknown")
    nla_txt = _pct(nla_share, 1.0)
    nla_note = np.select(
        [~has_nw, nla_max <= 0.0, known, nla_max <= mc],
        [pd.Series("net worth cannot be read (no positive P/B or market cap in the data)", index=idx),
         pd.Series("net liquid assets are not positive (limit 100% of market cap)", index=idx),
         "net liquid assets are " + nla_txt + "% of market cap (limit 100%)",
         "net liquid assets are up to " + nla_txt + "% of market cap (limit 100%)"],
        "net liquid assets are up to " + nla_txt + "% of market cap and a figure is missing (limit 100%)")

    # ── 6 Investments: what the sheet cannot explain, or cash if larger — the most they can be ──
    rev_bs = rev_1yb.where(lag.eq(1.0), rev.where(lag.eq(0.0)))       # the balance sheet's own revenue year
    receivables = (rev_bs * dso / 365.0).where((rev_bs > 0.0) & (dso >= 0.0))
    unexplained = ta - (fa.fillna(0.0) + cwip.fillna(0.0) + inv.fillna(0.0) + receivables.fillna(0.0))
    bound_share = _share(pd.Series(np.fmax(unexplained, cash.fillna(0.0)), index=idx), ta)
    inv_fig = fig["investments"]
    inv_share = inv_fig.where(inv_fig.notna(), bound_share)
    investments = np.select([inv_fig >= lim["investments"], inv_fig.notna(), bound_share < lim["investments"],
                             has_ta], ["fail", "pass", "pass", "review"], "unknown")
    inv_lim = _limit(lim["investments"])
    bound = ("cash + unexplained assets are up to " + _pct(bound_share, lim["investments"]) + "% of assets (limit "
             + inv_lim + ")")
    investments_note = np.select(
        [inv_fig.notna(), ~has_ta, bound_share < lim["investments"]],
        ["non-compliant investments are " + _pct(inv_fig, lim["investments"]) + "% of assets (limit " + inv_lim + ")"
         + yours, pd.Series(_NO_TA, index=idx), bound],
        "investments are not in the data; " + bound)

    states = dict(business=business, income=income, debt=debt_state, illiquid=illiquid, nla=nla,
                  investments=investments)
    notes = dict(business=business_note, income=income_note, debt=debt_note, illiquid=illiquid_note,
                 nla=nla_note, investments=investments_note)

    # ── What to check, for every rule left undecided: the figure to look up and its rupee limit ──
    ta_cr, rev_cr = _cr(ta.where(has_ta)), _cr(rev.where(rev > 0.0))
    checks = dict(
        business=np.where(blank, "what the company's business is (no industry in the data)",
                          "your scholar's ruling on this business"),
        income=np.where(rev > 0.0,
                        "interest and other impermissible income must be at most " + _cr(rev * lim["income"])
                        + " (" + inc_lim + " of " + rev_cr + " revenue)",
                        f"interest and other impermissible income must be at most {inc_lim} of revenue "
                        f"(no revenue figure in the data)"),
        debt=np.where(has_ta,
                      "interest-bearing debt must be at most " + _cr(ta * lim["debt"]) + " ("
                      + _limit(lim["debt"]) + " of " + ta_cr + " total assets)",
                      f"interest-bearing debt must be at most {_limit(lim['debt'])} of total assets ({_NO_TA})"),
        illiquid=np.where(has_ta,
                          "fixed assets + CWIP must be at least " + _cr(ta * lim["illiquid"]) + " ("
                          + _limit(lim["illiquid"]) + " of " + ta_cr + " total assets)",
                          f"fixed assets + CWIP must be at least {_limit(lim['illiquid'])} of total assets ({_NO_TA})"),
        nla=np.where(has_nw,
                     "net worth − (fixed assets + CWIP + inventory) must be at most the market cap, " + _cr(mc),
                     "the book value: P/B is missing or not positive in the data"),
        investments=np.where(has_ta,
                             "interest-bearing investments and deposits (cash included) must be below "
                             + _cr(ta * lim["investments"]) + " (" + inv_lim + " of " + ta_cr + " total assets)",
                             f"interest-bearing investments and deposits (cash included) must be below {inv_lim} "
                             f"of total assets ({_NO_TA})"))

    # ── Status: a proven fail beats a missing figure, which beats a review item or a lapsed ruling ──
    fails = np.column_stack([states[k] == "fail" for k in SHARIAH_RULES])
    unknowns = np.column_stack([states[k] == "unknown" for k in SHARIAH_RULES])
    reviews = np.column_stack([states[k] == "review" for k in SHARIAH_RULES])
    note_matrix = np.column_stack([notes[k] for k in SHARIAH_RULES])
    rows = np.arange(len(idx))
    first_fail = note_matrix[rows, fails.argmax(axis=1)]
    first_unknown = note_matrix[rows, unknowns.argmax(axis=1)]
    first_review = note_matrix[rows, reviews.argmax(axis=1)]
    any_fail, any_unknown, any_review = fails.any(axis=1), unknowns.any(axis=1), reviews.any(axis=1)
    shariah, non_shariah, for_review, doubtful = SHARIAH_STATUSES
    lapse_note = "your figures of " + checked + " were read against an older balance sheet — re-check"

    cols = {"shariah_status": np.select([any_fail, any_unknown, lapsed | any_review],
                                        [non_shariah, doubtful, for_review], shariah),
            "shariah_reason": np.select([any_fail, any_unknown, lapsed, any_review],
                                        [first_fail, first_unknown, lapse_note, first_review], SHARIAH_PASS_REASON),
            "shariah_ruling": np.select([lapsed, current], [lapse_note, "your figures: " + source + ", checked "
                                                            + checked], "")}
    cols.update({f"shariah_{k}": states[k] for k in SHARIAH_RULES})
    cols.update(shariah_income_pct=income_share * 100.0, shariah_debt_pct=debt_share * 100.0,
                shariah_illiquid_pct=illiquid_shown * 100.0, shariah_nla_pct=nla_share * 100.0,
                shariah_investments_pct=inv_share * 100.0)
    cols.update({f"shariah_{k}_note": notes[k] for k in SHARIAH_RULES})
    cols.update({f"shariah_{k}_check": np.where((states[k] == "review") | (states[k] == "unknown"), checks[k], "")
                 for k in SHARIAH_RULES})
    added = pd.DataFrame(cols, index=idx)[list(SHARIAH_COLUMNS)]
    text_cols = [c for c in SHARIAH_COLUMNS if not c.endswith("_pct")]
    added[text_cols] = added[text_cols].astype(object)
    return pd.concat([df, added], axis=1)
