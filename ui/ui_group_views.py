"""
ui_group_views.py — the measures behind the Market Pulse 📈 Sectors and 🏭 Industry column views.

ONE DEFINITION FOR BOTH TABS (2026-10-05): what each column measures, how a GROUP of stocks is
summarised, and the header and tooltip it carries. Pure pandas, no Streamlit, so it is tested
directly (tests/test_market_pulse_views.py).

HOW A GROUP IS SUMMARISED
  mean    PRISM's own 0-100 scores — an average of percentile scores is well behaved.
  median  raw multiples and distances (P/E, EV/EBITDA, FCF yield, distance from the 52-week high):
          one P/E of 900 drags an average anywhere; the middle stock is the honest summary.
  net     ownership changes (the FII / DII stake, latest filed quarter): the % of the group's companies
          where the stake rose minus the % where it fell, among companies with a reading (−100 to
          +100). An average or median of the changes themselves is flat — most companies do not move
          in a quarter (FII 31% / DII 36% flat) — so who moved, and which way, is what reads.
  share   yes/no conditions (above the 200-day average, few red flags): the % of the group's stocks
          for which it holds, counted ONLY among stocks that have the data — an unknown is not a "no".
          The engine records some unknowns as 0 — a stock with no 200-day average yet is "not above"
          it, one with no cash-flow figure "not flagged" for accruals — so those measures name the
          inputs they NEED, and a stock missing one is left out. Measured 2026-10-05: counting the 180
          stocks with no 200-day average as "below" read Engineering Heavy General 45% above its
          200-DMA when 83% of the stocks that have one are (23 industries moved 10 points or more).
          Composite verdicts — the red-flag count, Schilit pass, the F-Score — are the engine's own.

WHY THESE COLUMNS. Measured on 2026-10-05 across 67 sectors (>= 5 stocks) and 252 industries (>= 3):
no two columns in one view rank-correlate at 0.85 or more, and none is flat. Dropped on that test:
median ROCE (0.90 with Quality), earnings yield (-0.99 with P/E — it is 1/P/E), % at the forensic
penalty floor (0.90 with the average red-flag count), RS score (0.96 with Momentum), breakout score
(0.90 with the distance from the 52-week high), the median 52-week relative strength against the
Nifty 500 (0.85 with % above the 200-day average, which overlaps less with the rest of the view),
and dividend yield / pledge (median 0 almost everywhere). Momentum lives in Core, so 📈 Technical carries breadth (% above the 200-day average,
0.89 with Momentum) instead of repeating it.

👥 OWNERSHIP (2026-10-05), measured on the four saved snapshots: inside the view no pair passes 0.50
(median promoter vs median FII+DII), and against everything else on screen the closest is FII net vs
average Momentum at 0.52. DISPLAY ONLY — the forward evidence is weak: across four windows, sectors
with more pledging returned less (−0.09 to −0.15) and less institutionally owned sectors more (−0.10
to −0.32, mostly surviving a size control); FII net was positive, but about a third of it is Momentum.
Pledge is a SHARE here (any pledge), not the median, which is 0 almost everywhere. Insider trading
(10% coverage) is left out.

🔥 STRONG QTR (2026-10-05) is on 📈 Sectors Core only. Sector earnings breadth ranked sector returns at
+0.27 to +0.39 and added +0.15 to +0.31 BEYOND Momentum in two independent results seasons — it passed
the pre-declared bar (>= +0.05 in both). Industry breadth FAILED it (-0.02 / +0.03 in Jan-Mar): small
groups are too noisy, so the Industry table does not show it until December re-tests.

📈 NEAR 52W HIGH (2026-10-05) is on 📈 Sectors' Technical view only. The share of a sector's stocks within
5% of their 52-week high added +0.11 to +0.31 BEYOND Momentum in all four windows, and kept an edge
beyond Strong qtr and beyond Below 52w high. Industry breadth failed one window (-0.01), so the
Industry table skips it until December re-tests.
"""
import numpy as np
import pandas as pd

from config import FORENSIC_MAX_FLAGS, FORENSIC_PENALTY_TIERS
from core.scoring_engine import strong_quarter

OWNERSHIP = "👥 Ownership"
VIEWS = ("🏆 Core", "📊 Quality", "💰 Valuation", "🔬 Forensic", "📈 Technical", OWNERSHIP)

# Shown beside the 👥 Ownership view wherever it appears. Shareholding is filed quarterly, so the
# view must never read as this week's trading. The tables extend the shared sentence, never fork it.
OWNERSHIP_NOTE = (
    "👥 Shareholding is filed quarterly, about three weeks after each quarter ends, so these figures "
    "stand at each company's latest filed quarter — not this week's trading. FII = foreign "
    "institutions · DII = domestic institutions (mutual funds, insurers).")
OWNERSHIP_GROUP_NOTE = OWNERSHIP_NOTE + (
    " **Net** = % of the group's companies where the stake rose minus % where it fell (−100 to +100): "
    "it counts companies, not rupees.")

# "Few red flags" is the forensic penalty's own line: the most flags that still cost only ×0.90.
_FEW_FLAGS = next(t["max_flags"] for t in FORENSIC_PENALTY_TIERS if t["multiplier"] == 0.90)


def _share(cond, known):
    """1.0 / 0.0 where the condition can be judged, NaN where it cannot (so a mean is a share of
    the stocks WITH data)."""
    return cond.astype(float).where(known)


def _net(change):
    """+1 where the stake rose, −1 where it fell, 0 where it held, NaN with no reading — so a mean is
    the % raised minus the % cut among companies WITH a reading."""
    return np.sign(change)


# key -> source column, statistic, header, number format, bar (0-100 progress bar), tooltip, value;
# `needs` (optional) names the further inputs `value` reads — for a share, the inputs without which
# the engine could not judge the stock;
# NUMBER columns carry their own width (sized to the header) — bars share the app's one bar width,
# so every view of both tables fits a 1536px laptop screen (measured in the browser, 2026-10-05).
MEASURES = {
    # 📊 Quality
    "grp_roce_rising": dict(
        source="roce_expansion", stat="share", header="ROCE rising", fmt="%.0f%%", bar=True,
        help="Share of the group's stocks whose return on capital is rising: the 3-year median ROCE "
             "is above the 10-year median. Counted among stocks with the history to measure it.",
        value=lambda f: _share(f["roce_expansion"] > 0, f["roce_expansion"].notna())),
    "grp_opm": dict(
        source="opm", stat="median", header="Median OPM", fmt="%.1f%%", bar=False, width=105,
        help="Operating margin of the group's middle stock, latest full year.",
        value=lambda f: f["opm"]),
    # 💰 Valuation
    "grp_pe": dict(
        source="pe", stat="median", header="Median P/E", fmt="%.1f", bar=False, width=105,
        help="Price to earnings of the group's middle stock, among stocks with positive earnings — a "
             "loss-maker has no meaningful P/E. A median, so one extreme multiple cannot move it.",
        value=lambda f: f["pe"].where(f["pe"] > 0)),
    "grp_ev_ebitda": dict(
        source="ev_ebitda", stat="median", header="Median EV/EBITDA", fmt="%.1f", bar=False, width=130,
        help="Enterprise value to EBITDA of the group's middle stock, among stocks with positive "
             "EBITDA. Unlike P/E it counts debt, so it compares leveraged and debt-free groups fairly.",
        value=lambda f: f["ev_ebitda"].where(f["ev_ebitda"] > 0)),
    "grp_fcf_yield": dict(
        source="fcf_yield", stat="median", header="Median FCF yield", fmt="%.1f%%", bar=False, width=130,
        help="Free cash flow as a % of market value for the group's middle stock. Negative means the "
             "typical stock in the group is burning cash.",
        value=lambda f: f["fcf_yield"]),
    "grp_pe_vs_history": dict(
        source="d32_pe_vs_median", stat="median", header="P/E vs history", fmt="%+.0f%%", bar=False, width=120,
        help="How far each stock's P/E sits from its own 5-year median P/E, for the group's middle "
             "stock. Negative = the group trades cheaper than it usually does.",
        value=lambda f: f["d32_pe_vs_median"]),
    # 🔬 Forensic
    "grp_red_flags": dict(
        source="red_flag_count", stat="mean", header="Avg red flags", fmt="%.1f", bar=False, width=115,
        help=f"Average number of forensic red flags per stock, out of {FORENSIC_MAX_FLAGS} checks.",
        value=lambda f: f["red_flag_count"]),
    "grp_few_flags": dict(
        source="red_flag_count", stat="share", header=f"≤{_FEW_FLAGS} red flags", fmt="%.0f%%", bar=True,
        help=f"Share of the group's stocks with at most {_FEW_FLAGS} red flags — the ones the forensic "
             f"penalty barely touches (×0.90, or none at all).",
        value=lambda f: _share(f["red_flag_count"] <= _FEW_FLAGS, f["red_flag_count"].notna())),
    "grp_piotroski_strong": dict(
        source="piotroski_fscore", stat="share", header="Piotroski 7+", fmt="%.0f%%", bar=True,
        help="Share of the group's stocks with a Piotroski F-Score of 7 or more out of 9 — healthy, "
             "improving books.",
        value=lambda f: _share(f["piotroski_fscore"] >= 7, f["piotroski_fscore"].notna())),
    "grp_schilit_pass": dict(
        source="schilit_pass", stat="share", header="Schilit pass", fmt="%.0f%%", bar=True,
        help="Share passing the Schilit forensic screen: at most 2 of its 4 accounting-shenanigan "
             "checks fired.",
        value=lambda f: _share(f["schilit_pass"] == 1, f["schilit_pass"].notna())),
    "grp_high_accruals": dict(
        source="rf_high_accruals", stat="share", header="High accruals", fmt="%.0f%%", bar=True,
        needs=("pat_cf_year", "operating_cash_flow", "total_assets", "total_assets_1yb"),
        help="Share whose profit runs well ahead of its cash: accruals above 5% of assets, with profit "
             "and cash taken from the same year. Counted among stocks with profit, cash flow and assets "
             "on record.",
        # The engine's own guard (forensic_engine, rf_high_accruals): profit and cash flow of one year,
        # and assets above 0 this year OR last (it uses whichever exists, in full) — else it records 0.
        value=lambda f: _share(f["rf_high_accruals"] == 1,
                               f["pat_cf_year"].notna() & f["operating_cash_flow"].notna()
                               & ((f["total_assets"] > 0) | (f["total_assets_1yb"] > 0)))),
    # 📈 Technical
    "grp_above_200dma": dict(
        source="above_sma200", stat="share", header="Above 200-DMA", fmt="%.0f%%", bar=True,
        needs=("close_price", "sma_200d"),
        help="Share of the group's stocks trading above their 200-day moving average — the classic "
             "breadth reading of how much of a group is in an uptrend. Counted among stocks with 200 "
             "days of history: a recent listing is not 'below' an average it does not have yet.",
        # The engine sets above_sma200 = close > sma_200d, so a missing average reads 0, never blank.
        value=lambda f: _share(f["above_sma200"] == 1, f["close_price"].notna() & f["sma_200d"].notna())),
    "grp_stage2": dict(
        source="weinstein_stage", stat="share", header="Stage 2", fmt="%.0f%%", bar=True,
        help="Share in Weinstein Stage 2 — the advancing phase of a stock's cycle — among stocks whose "
             "stage is known.",
        value=lambda f: _share(f["weinstein_stage"].astype(str).str.contains("Stage 2", regex=False),
                               f["weinstein_stage"].notna()
                               & ~f["weinstein_stage"].astype(str).str.contains("Unknown", regex=False))),
    # 📈 Sectors only (2026-10-05, from ValuePickr's 52-week / all-time-high thread: look for the sectors
    # where stocks are making new highs) — app.py's Industry table skips it.
    "grp_near_high": dict(
        source="dist_52wh", stat="share", header="Near 52w high", fmt="%.0f%%", bar=True,
        help="Share of the sector's stocks trading within 5% of their 52-week high — how much of the "
             "sector is leading. Read it beside Below 52w high, which shows where the typical stock "
             "sits: the two disagree when a few leaders run while the rest lag.",
        value=lambda f: _share(f["dist_52wh"] <= 5, f["dist_52wh"].notna())),
    "grp_from_52w_high": dict(
        source="dist_52wh", stat="median", header="Below 52w high", fmt="%.0f%%", bar=False, width=120,
        help="How far below its 52-week high the group's middle stock trades. Small = the group is "
             "close to its highs.",
        value=lambda f: f["dist_52wh"]),
    # 👥 Ownership — as of each company's latest filed quarter (OWNERSHIP_NOTE)
    "grp_promoter": dict(
        source="promoter_holdings", stat="median", header="Median promoter", fmt="%.0f%%", bar=False,
        width=130,
        help="Promoter holding of the group's middle company — how much of it the founders / "
             "controlling owners hold.",
        value=lambda f: f["promoter_holdings"]),
    "grp_pledged": dict(
        source="pledged_percentage", stat="share", header="Pledged", fmt="%.0f%%", bar=True,
        help="Share of the group's companies whose promoters have pledged any of their shares as loan "
             "collateral — a pledge can force a sale when the price falls. Counted among companies "
             "with pledge data.",
        value=lambda f: _share(f["pledged_percentage"] > 0, f["pledged_percentage"].notna())),
    "grp_institutions": dict(
        source="fii_holdings", needs=("dii_holdings",), stat="median", header="Median FII+DII",
        fmt="%.1f%%", bar=False, width=125,
        help="Combined stake of foreign (FII) and domestic (DII) institutions in the group's middle "
             "company. Low = little institutional ownership; high = widely held. Counted among "
             "companies with both figures.",
        value=lambda f: f["fii_holdings"] + f["dii_holdings"]),
    "grp_fii_net": dict(
        source="change_fii_lq", stat="net", header="FII net qtr", fmt="%+.0f", bar=False, width=105,
        help="Latest filed quarter: % of the group's companies where foreign institutions (FIIs) "
             "raised their stake minus % where they cut it, among companies with a reading (−100 to "
             "+100). +30 = FIIs added to 30 more companies in every 100 than they trimmed. It counts "
             "companies, not rupees.",
        value=lambda f: _net(f["change_fii_lq"])),
    "grp_dii_net": dict(
        source="change_dii_lq", stat="net", header="DII net qtr", fmt="%+.0f", bar=False, width=105,
        help="Latest filed quarter: % of the group's companies where domestic institutions (DIIs — "
             "mutual funds, insurers) raised their stake minus % where they cut it, among companies "
             "with a reading (−100 to +100). It counts companies, not rupees.",
        value=lambda f: _net(f["change_dii_lq"])),
    # 📈 Sectors Core only (2026-10-05) — earnings-season breadth; strong_quarter is the engine's own
    # definition (1 / 0 / NaN), so the mean IS the share among companies that can be judged.
    "grp_strong_quarter": dict(
        source="rev_lq", needs=("rev_pyq", "pat_lq", "pat_pyq", "ebitda_lq", "ebitda_pyq"),
        stat="share", header="Strong qtr", fmt="%.0f%%", bar=True,
        help="Share of the sector's companies whose latest reported quarter was strong against the "
             "same quarter last year: revenue up more than 15%, profit up more than 20%, EBITDA margin "
             "wider. Counted among companies that can be judged (profitable, with revenue, a year ago). "
             "Ishmohit Arora (SOIC) reads a sector where most players are reporting strong earnings "
             "growth as one starting to do well. It moves through each results season.",
        value=lambda f: strong_quarter(f)),
}

# Each view's columns after the tab's Count and ranking column. The avg_* entries are each table's
# own averages; everything else comes from MEASURES. 🏆 Core is each tab's own column list.
VIEW_MEASURES = {
    "📊 Quality":   ("avg_quality", "avg_improvement", "grp_roce_rising", "grp_opm"),
    "💰 Valuation": ("avg_valuation", "grp_pe", "grp_ev_ebitda", "grp_fcf_yield", "grp_pe_vs_history"),
    # The same order as the Deep Scanner's 🔬 Forensic view: the flags, then Piotroski, then Schilit.
    "🔬 Forensic":  ("grp_red_flags", "grp_few_flags", "grp_piotroski_strong", "grp_schilit_pass",
                     "grp_high_accruals"),
    "📈 Technical": ("grp_above_200dma", "grp_stage2", "grp_near_high", "grp_from_52w_high"),
    OWNERSHIP:      ("grp_promoter", "grp_pledged", "grp_institutions", "grp_fii_net", "grp_dii_net"),
}

def _inputs(m):
    """Every column a measure reads: its source, then anything it needs to judge a stock."""
    return (m["source"], *m.get("needs", ()))


# Every input the measures read — 🏭 Industry projects its stocks to a fixed column list first.
SOURCE_COLUMNS = tuple(sorted({c for m in MEASURES.values() for c in _inputs(m)}))


def _per_stock(frame: pd.DataFrame, m: dict) -> pd.Series:
    """One measure's value per stock: 1 / 0 / NaN for a share, the number for a mean or median."""
    src = frame[list(_inputs(m))].copy()
    for c in src.columns:
        if c != "weinstein_stage":
            src[c] = pd.to_numeric(src[c], errors="coerce")
    return m["value"](src).astype(float)


def group_measures(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """One row per group (index = the `key` values), one column per MEASURES entry.

    Vectorised: each measure's per-stock value is built once, then ONE groupby takes the means
    (scores, shares and net breadth) and one the medians. Shares come back as percentages (0-100),
    net breadth as -100 to +100. A measure
    whose inputs are absent is left out rather than invented. The input is never modified.
    """
    present = {k: m for k, m in MEASURES.items() if all(c in frame.columns for c in _inputs(m))}
    vals = pd.DataFrame(index=frame.index)
    for k, m in present.items():
        vals[k] = _per_stock(frame, m)
    vals[key] = frame[key].values
    means = [k for k, m in present.items() if m["stat"] in ("mean", "share", "net")]
    medians = [k for k, m in present.items() if m["stat"] == "median"]
    out = pd.concat([vals.groupby(key)[means].mean(), vals.groupby(key)[medians].median()], axis=1)
    pct = [k for k, m in present.items() if m["stat"] in ("share", "net")]
    out[pct] = out[pct] * 100.0
    return out[[k for k in MEASURES if k in out.columns]]
