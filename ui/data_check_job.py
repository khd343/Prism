"""
🩺 The daily data check (2026-10-10). Run by GitHub Actions (.github/workflows/data-check.yml):

    python -m ui.data_check_job            # needs SHEET_ID (the live data sheet) in the environment

The vendor changed its data four times in one quarter without a word, and PRISM kept running on it each
time: sector and industry names re-spelled (7 Oct — about 50 name lookups fell back silently), insider
trading turned from text into numbers (the signal dead for months), the P&L rolled a year ahead of the
balance sheet and cash flow, and the download endpoints moved. Each was found late and by accident, and
the local /verify cannot see the live sheet. This job loads the live sheet through PRISM's own loader
and engine — one download, exactly the app's path — and checks what each of those breaks would have
shown. A failure fails the run, and GitHub emails the repository owner the same morning.

Every limit below was set from the live sheet and the local data measured on 2026-10-10 (the numbers are
beside each one), wide enough that a normal day never trips it. The first measurement already found a
real break: 25 industry names the cyclicality map does not know, 7.9% of the live stocks.
"""
import contextlib
import io
import os
import sys
from dataclasses import dataclass

import pandas as pd

from config import BENCHMARK_INDICES
from core import data_engine as de
from core import run_scoring_pipeline
from core.cyclicality_map import INDUSTRY_TIER, SECTOR_TIER_FALLBACK
from core.sheet_meta import describe
from ui.scan_records_job import sheet_id_from_secret

TABS = ("ratio", "income", "balance", "cashflow", "shareholding", "technical")
MIN_TAB_ROWS = 1_500                    # 2,719 rows in each tab on 2026-10-10
UNIVERSE = (1_500, 4_000)               # 2,712 stocks (2,107 in June)
# Key columns and the share of stocks that must carry them (live 2026-10-10: the core ones 100%, the
# rest 88-100%; P/E 88% because the vendor leaves it blank for every loss-maker).
CORE_COLUMNS = ["close_price", "market_cap", "vol_sma_20d", "sector", "industry", "revenue", "pat",
                "equity_shares", "eps"]
OTHER_COLUMNS = ["opm", "roce", "operating_cash_flow", "total_assets", "promoter_holdings", "pe",
                 "debt_to_equity", "rev_lq", "pat_lq", "listed_days"]
CORE_FLOOR, OTHER_FLOOR = 0.95, 0.80
TEXT_COLUMNS = {"company_id", "name", "market_category", "eligibility", "industry", "sector",
                "insider_trading"}             # the loader keeps these as text (coerce_numeric_columns)
MISSING_WORDS = {"", "null", "nan", "none", "-", "na", "n/a"}
MAX_TEXT_SHARE = 0.05                   # 0.0% of filled cells in any number column on 2026-10-10
MIN_INSIDER_NUMBERS = 0.95              # 100% of 252 filled cells are numbers
UNMAPPED_FAIL = 0.02                    # 0.0% locally (2026-09-18); 7.9% live on 2026-10-10 — a real break
MAX_LAG_UNDECIDED = 0.10                # 2.2% live, 2.3% locally
SHARES_TOLERANCE = 0.05                 # shares × price vs market cap: a 1-3% gap is price timing (184
MIN_SHARES_MATCH = 0.95                 # stocks live, median 1.1%); a basis change moves it 10%+
MIN_SCORE_STD, MIN_DISTINCT = 5.0, 100  # composite std 13.6, 562 distinct values live
GATE_PASS = (0.05, 0.80)                # 31.3% live, 32.2% locally
MAX_FRAMEWORKS_NONE = 0.90              # 14.3% live (the /verify invariant)
MAX_SESSIONS_BEHIND = 1                 # the job runs before the evening data, so 0 is normal


@dataclass
class Result:
    name: str
    status: str                         # "ok" · "warn" · "fail"
    detail: str


def _ok(cond: bool, name: str, detail: str, warn: bool = False) -> Result:
    return Result(name, "ok" if cond else ("warn" if warn else "fail"), detail)


# ── The checks (pure; tests/test_data_check_job.py) ────────────────────────────────────────────────
def check_tabs(datasets: dict) -> list:
    rows = {t: len(datasets[t]) for t in TABS if t in datasets}
    missing = [t for t in TABS if t not in datasets]
    short = [t for t, n in rows.items() if n < MIN_TAB_ROWS]
    return [_ok(not missing and not short, "Six tabs",
                f"missing {missing}" if missing else (f"too few rows {short}" if short else
                                                      f"{min(rows.values()):,} rows each or more"))]


def check_universe(stocks: pd.DataFrame) -> list:
    n = len(stocks)
    return [_ok(UNIVERSE[0] <= n <= UNIVERSE[1], "Universe", f"{n:,} stocks (expected {UNIVERSE[0]:,}-{UNIVERSE[1]:,})")]


def check_benchmarks(bench: pd.DataFrame) -> list:
    have = {} if bench is None or bench.empty else dict(zip(bench["company_id"].astype(str),
                                                            pd.to_numeric(bench["close_price"], errors="coerce")))
    bad = [BENCHMARK_INDICES[i] for i in BENCHMARK_INDICES if not (have.get(i, 0) > 0)]
    return [_ok(not bad, "Index rows", f"missing or no price: {bad}" if bad else f"all {len(BENCHMARK_INDICES)} with a price")]


def check_coverage(stocks: pd.DataFrame) -> list:
    problems = []
    for cols, floor in ((CORE_COLUMNS, CORE_FLOOR), (OTHER_COLUMNS, OTHER_FLOOR)):
        for c in cols:
            if c not in stocks.columns:
                problems.append(f"{c} absent")
            elif stocks[c].notna().mean() < floor:
                problems.append(f"{c} {stocks[c].notna().mean():.0%} < {floor:.0%}")
    return [_ok(not problems, "Key columns", "; ".join(problems) or
                f"{len(CORE_COLUMNS) + len(OTHER_COLUMNS)} columns present and filled")]


def _filled(s: pd.Series) -> pd.Series:
    return s.notna() & ~s.astype(str).str.strip().str.lower().isin(MISSING_WORDS)


def check_types(datasets: dict) -> list:
    """On the RAW tabs, before the loader turns text into blanks: a number column that suddenly holds
    words loses its signal silently (the insider-trading break)."""
    worst, insider = [], []
    for tab in sorted(datasets):
        df = datasets[tab]
        for c in df.columns:
            if c == "insider_trading":
                insider.append(df[c])
            if c in TEXT_COLUMNS:
                continue
            filled = _filled(df[c])
            if filled.any():
                share = (filled & pd.to_numeric(df[c], errors="coerce").isna()).sum() / filled.sum()
                if share > MAX_TEXT_SHARE:
                    worst.append(f"{tab}.{c} {share:.0%} text")
    out = [_ok(not worst, "Number columns", "; ".join(worst) or "every number column holds numbers")]
    if insider:
        s = pd.concat(insider)
        filled = _filled(s)
        share = pd.to_numeric(s[filled], errors="coerce").notna().mean() if filled.any() else 1.0
        out.append(_ok(share >= MIN_INSIDER_NUMBERS, "Insider trading is a number",
                       f"{share:.0%} of {int(filled.sum())} filled cells are numbers"))
    return out


def check_taxonomy(stocks: pd.DataFrame) -> list:
    """After the loader's own translation (core/taxonomy.py): an industry the cyclicality map does not
    know falls back to its sector's tier — fine for one new industry, a break when names were re-spelled."""
    ind = stocks["industry"].dropna().astype(str)
    unknown = sorted(set(ind) - set(INDUSTRY_TIER))
    share = ind.isin(unknown).mean() if len(ind) else 0.0
    status = "ok" if not unknown else ("warn" if share <= UNMAPPED_FAIL else "fail")
    out = [Result("Industry names", status,
                  f"{len(unknown)} unmapped ({share:.1%} of stocks): {unknown[:8]}" if unknown else
                  f"all {ind.nunique()} industries mapped")]
    sec = stocks["sector"].dropna().astype(str)
    odd = sorted(set(sec) - set(SECTOR_TIER_FALLBACK))
    out.append(_ok(not odd, "Sector names", f"{len(odd)} without a fallback tier: {odd[:8]}" if odd
                   else f"all {sec.nunique()} sectors known", warn=True))
    return out


def check_identities(stocks: pd.DataFrame) -> list:
    """The vendor's own arithmetic. Shares × price = market cap (the live count IS market cap ÷ price);
    and the cash-flow year can be told from the vendor's own ratios (cash_flow_year_lag)."""
    eq, px, mc = (pd.to_numeric(stocks[c], errors="coerce") for c in ("equity_shares", "close_price", "market_cap"))
    both = eq.notna() & px.notna() & mc.notna() & (mc > 0)
    match = ((eq * px / 1e7 / mc - 1).abs() <= SHARES_TOLERANCE)[both].mean() if both.any() else 0.0
    undecided = de.cash_flow_year_lag(stocks).isna().mean()
    return [_ok(match >= MIN_SHARES_MATCH, "Shares × price = market cap",
                f"{match:.1%} of {int(both.sum()):,} within {SHARES_TOLERANCE:.0%}"),
            _ok(undecided <= MAX_LAG_UNDECIDED, "Cash-flow year decided", f"{undecided:.1%} undecided")]


def check_engine(scored: pd.DataFrame) -> list:
    c = pd.to_numeric(scored["composite_score"], errors="coerce")
    flat = [col for col in ("composite_score", "quality_score", "momentum_score")
            if pd.to_numeric(scored[col], errors="coerce").round(1).nunique() < MIN_DISTINCT]
    ranks = sorted(pd.to_numeric(scored["rank"], errors="coerce").dropna().astype(int))
    gate = pd.to_numeric(scored["gate_pass"], errors="coerce").mean()
    none = scored["frameworks_passed"].astype(str).isin(["None", "", "nan"]).mean()
    return [
        _ok(c.notna().all() and c.between(0, 100).all(), "Scores in range",
            f"{int(c.isna().sum())} missing · {c.min():.1f}–{c.max():.1f}"),
        _ok(not flat and c.std() >= MIN_SCORE_STD, "Scores not flat",
            f"flat: {flat}" if flat else f"std {c.std():.1f}, {c.round(1).nunique()} distinct"),
        _ok(ranks == list(range(1, len(scored) + 1)), "Ranks 1..N", f"{len(ranks):,} ranks for {len(scored):,} stocks"),
        _ok(GATE_PASS[0] <= gate <= GATE_PASS[1], "Gate pass rate", f"{gate:.1%}"),
        _ok(none < MAX_FRAMEWORKS_NONE, "Frameworks populating", f"{none:.1%} pass none"),
    ]


def check_freshness(fresh) -> list:
    behind = getattr(fresh, "sessions_behind", None)
    ok = bool(getattr(fresh, "is_known", False)) and behind is not None and behind <= MAX_SESSIONS_BEHIND
    return [_ok(ok, "Sheet is current", f"{getattr(fresh, 'label', '?')} · {behind} session(s) behind")]


# ── The run ──────────────────────────────────────────────────────────────────────────────────────
_ICON = {"ok": "✅", "warn": "⚠️", "fail": "❌"}


def _secret_shape(raw: str, sheet_id: str) -> str:
    """The secret's shape, never its value — enough to tell a quoted id, a pasted line or an address apart."""
    return (f"secret: {len(raw.strip())} chars{', quoted' if any(q in raw for q in chr(34) + chr(39)) else ''}"
            f"{', a full line' if '=' in raw else ''}{', an address' if '/' in raw else ''}"
            f" -> sheet id of {len(sheet_id)} chars")


def main() -> int:
    raw = os.environ.get("SHEET_ID", "")
    sheet_id = sheet_id_from_secret(raw)
    print(_secret_shape(raw, sheet_id))
    if not sheet_id:
        raise SystemExit("SHEET_ID holds no spreadsheet id — paste the live data sheet's id (the part of its "
                         "address between /d/ and /edit).")
    # ONE download, the app's own path (core.load_market = load_all_csvs -> _clean_from + benchmark_table),
    # keeping the raw tabs for the type check before the loader turns text into blanks.
    # The engine's progress lines are silenced so the log reads as the checks; an error still surfaces.
    with contextlib.redirect_stdout(io.StringIO()):
        datasets = de.load_all_csvs("sheet", None, sheet_id=sheet_id)
        stocks, bench = de._clean_from(datasets), de.benchmark_table(datasets)
        scored = run_scoring_pipeline(stocks.copy())
    results = (check_freshness(describe(sheet_id, "sheet")) + check_tabs(datasets) + check_universe(stocks)
               + check_benchmarks(bench) + check_coverage(stocks) + check_types(datasets)
               + check_taxonomy(stocks) + check_identities(stocks) + check_engine(scored))
    lines = [f"{_ICON[r.status]} {r.name}: {r.detail}" for r in results]
    print("\n" + "\n".join(lines))
    failed = [r for r in results if r.status == "fail"]
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("## 🩺 PRISM data check\n\n| | Check | Reading |\n|---|---|---|\n")
            f.write("".join(f"| {_ICON[r.status]} | {r.name} | {r.detail.replace('|', '/')} |\n" for r in results))
    print(f"\n{len(failed)} failed, {sum(r.status == 'warn' for r in results)} warning(s), "
          f"{sum(r.status == 'ok' for r in results)} passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
