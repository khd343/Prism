"""
test_data_check_job.py — the daily check on the live sheet (2026-10-10).

The vendor changed its data four times in one quarter without a word, and PRISM kept running on it each
time: sector and industry names re-spelled (7 Oct, ~50 lookups silently falling back), insider trading
turned from text into numbers (the signal dead for months), the P&L rolled a year ahead of the balance
sheet and the cash flow, and the download endpoints moved. Each was found late and by accident; the local
/verify cannot see the live sheet. ui/data_check_job.py loads the live sheet daily through PRISM's own
loader and engine (.github/workflows/data-check.yml) and fails — GitHub emails the owner — on a break.
Every limit was set from the live sheet and the local data measured that day, and the first measurement
already found one: 25 industry names (7.9% of stocks) the cyclicality map does not know.

Run with: pytest tests/test_data_check_job.py -v
"""
import contextlib
import io
import os
import re
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import BENCHMARK_INDICES  # noqa: E402
from core.cyclicality_map import INDUSTRY_TIER  # noqa: E402
from ui import data_check_job as D  # noqa: E402


def _status(results, name):
    hits = [r for r in results if r.name == name]
    assert len(hits) == 1, f"expected one {name!r} result, got {[r.name for r in results]}"
    return hits[0].status


# ── each check, on the break it exists to catch ───────────────────────────────────────────────────
def test_tabs_and_universe_bounds():
    tabs = {t: pd.DataFrame({"company_id": range(2000)}) for t in D.TABS}
    assert _status(D.check_tabs(tabs), "Six tabs") == "ok"
    assert _status(D.check_tabs({**tabs, "cashflow": pd.DataFrame({"company_id": range(40)})}), "Six tabs") == "fail"
    assert _status(D.check_tabs({t: v for t, v in tabs.items() if t != "technical"}), "Six tabs") == "fail"
    assert _status(D.check_universe(pd.DataFrame({"name": range(2700)})), "Universe") == "ok"
    assert _status(D.check_universe(pd.DataFrame({"name": range(300)})), "Universe") == "fail"


def _bench(ids=None, price=100.0):
    ids = list(BENCHMARK_INDICES) if ids is None else ids
    return pd.DataFrame({"company_id": ids, "close_price": [price] * len(ids)})


def test_every_benchmark_index_must_be_there_with_a_price():
    assert _status(D.check_benchmarks(_bench()), "Index rows") == "ok"
    assert _status(D.check_benchmarks(_bench(list(BENCHMARK_INDICES)[1:])), "Index rows") == "fail"
    assert _status(D.check_benchmarks(_bench(price=float("nan"))), "Index rows") == "fail"


def _stocks(n=200, **over):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({c: rng.uniform(1, 100, n) for c in D.CORE_COLUMNS + D.OTHER_COLUMNS})
    df["sector"] = "Chemicals"
    df["industry"] = next(iter(INDUSTRY_TIER))
    for k, v in over.items():
        df[k] = v
    return df


def test_a_missing_or_emptied_key_column_fails():
    assert _status(D.check_coverage(_stocks()), "Key columns") == "ok"
    assert _status(D.check_coverage(_stocks().drop(columns=["close_price"])), "Key columns") == "fail"
    thin = _stocks(); thin.loc[thin.index[:30], "market_cap"] = np.nan          # 85% < the 95% core floor
    assert _status(D.check_coverage(thin), "Key columns") == "fail"
    ok_thin = _stocks(); ok_thin.loc[ok_thin.index[:30], "pe"] = np.nan          # 85% >= the 80% floor
    assert _status(D.check_coverage(ok_thin), "Key columns") == "ok"


def test_text_where_numbers_belong_fails_and_names_the_column():
    good = {"ratio": pd.DataFrame({"company_id": ["a", "b"], "roce": [10.0, "null"], "insider_trading": [5.0, None]})}
    assert _status(D.check_types(good), "Number columns") == "ok"
    bad = {"ratio": pd.DataFrame({"company_id": list("abcd"), "roce": ["10", "high", "low", "mid"]})}
    r = [x for x in D.check_types(bad) if x.name == "Number columns"][0]
    assert r.status == "fail" and "roce" in r.detail
    # the insider-trading change, exactly: the vendor's number came back as words
    words = {"shareholding": pd.DataFrame({"company_id": list("abc"), "insider_trading": ["Bought", "Sold", "Bought"]})}
    assert _status(D.check_types(words), "Insider trading is a number") == "fail"


def test_unmapped_industry_names_warn_when_few_and_fail_when_many():
    known = next(iter(INDUSTRY_TIER))
    assert _status(D.check_taxonomy(_stocks(industry=known)), "Industry names") == "ok"
    few = _stocks(); few.loc[few.index[:1], "industry"] = "Brand New Industry"          # 0.5% of stocks
    assert _status(D.check_taxonomy(few), "Industry names") == "warn"
    many = _stocks(); many.loc[many.index[:30], "industry"] = "Re-spelled Industry"      # 15%
    r = [x for x in D.check_taxonomy(many) if x.name == "Industry names"][0]
    assert r.status == "fail" and "Re-spelled Industry" in r.detail


def test_the_vendors_own_arithmetic(monkeypatch):
    n = 200
    good = pd.DataFrame({"equity_shares": [1e7] * n, "close_price": [100.0] * n, "market_cap": [100.0] * n})
    monkeypatch.setattr(D.de, "cash_flow_year_lag", lambda df: pd.Series(1.0, index=df.index))
    assert _status(D.check_identities(good), "Shares × price = market cap") == "ok"
    assert _status(D.check_identities(good), "Cash-flow year decided") == "ok"
    timing = good.copy(); timing["market_cap"] = 102.0                                    # 2%: a price-timing gap
    assert _status(D.check_identities(timing), "Shares × price = market cap") == "ok"
    rebased = good.copy(); rebased.loc[rebased.index[:40], "equity_shares"] = 1e8         # 20% on a new basis
    assert _status(D.check_identities(rebased), "Shares × price = market cap") == "fail"
    monkeypatch.setattr(D.de, "cash_flow_year_lag", lambda df: pd.Series(np.nan, index=df.index))
    assert _status(D.check_identities(good), "Cash-flow year decided") == "fail"


def _scored(n=500, flat=False):
    rng = np.random.default_rng(2)
    v = (lambda: np.full(n, 50.0)) if flat else (lambda: rng.uniform(5, 95, n))
    return pd.DataFrame({"composite_score": v(), "quality_score": v(), "momentum_score": v(),
                         "rank": rng.permutation(n) + 1, "gate_pass": (rng.uniform(size=n) < 0.3).astype(int),
                         "frameworks_passed": np.where(rng.uniform(size=n) < 0.15, "None", "QGLP")})


def test_the_engine_scores_sensibly_and_never_flat():
    assert all(r.status == "ok" for r in D.check_engine(_scored()))
    # the 2026-06-12 loader bug flattened every score (Growth=50 / Momentum=26): caught here
    assert any(r.status == "fail" for r in D.check_engine(_scored(flat=True)))
    nan = _scored(); nan.loc[0, "composite_score"] = np.nan
    assert any(r.status == "fail" for r in D.check_engine(nan))
    dup = _scored(); dup.loc[0, "rank"] = dup.loc[1, "rank"]
    assert any(r.status == "fail" for r in D.check_engine(dup))


def test_a_stale_sheet_fails():
    fresh = SimpleNamespace(is_known=True, data_date=date(2026, 10, 9), sessions_behind=0, label="Fri, 09 Oct")
    assert _status(D.check_freshness(fresh), "Sheet is current") == "ok"
    assert _status(D.check_freshness(SimpleNamespace(**{**vars(fresh), "sessions_behind": 1})), "Sheet is current") == "ok"
    assert _status(D.check_freshness(SimpleNamespace(**{**vars(fresh), "sessions_behind": 2})), "Sheet is current") == "fail"
    assert _status(D.check_freshness(SimpleNamespace(is_known=False, data_date=None, sessions_behind=None, label="?")),
                   "Sheet is current") == "fail"


# ── the run ───────────────────────────────────────────────────────────────────────────────────────
def _fake_run(monkeypatch, stocks, bench, scored, fresh):
    monkeypatch.setattr(D.de, "load_all_csvs", lambda *a, **k: {t: pd.DataFrame({"company_id": range(2000)}) for t in D.TABS})
    monkeypatch.setattr(D.de, "_clean_from", lambda ds: stocks)
    monkeypatch.setattr(D.de, "benchmark_table", lambda ds: bench)
    monkeypatch.setattr(D, "run_scoring_pipeline", lambda df: scored)
    monkeypatch.setattr(D, "describe", lambda sid, src: fresh)
    monkeypatch.setattr(D.de, "cash_flow_year_lag", lambda df: pd.Series(1.0, index=df.index))


_SECRET = "1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789abcd"


def test_main_passes_on_good_data_fails_on_a_break_and_never_prints_the_id(monkeypatch, tmp_path):
    good = _stocks(2600, equity_shares=1e7, close_price=100.0, market_cap=100.0)
    fresh = SimpleNamespace(is_known=True, data_date=date(2026, 10, 9), sessions_behind=0, label="Fri, 09 Oct")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("SHEET_ID", f'SHEET_ID = "{_SECRET}"')
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    _fake_run(monkeypatch, good, _bench(), _scored(2600), fresh)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert D.main() == 0
    assert _SECRET not in out.getvalue() and _SECRET not in summary.read_text(encoding="utf-8")
    assert "| ✅ |" in summary.read_text(encoding="utf-8"), "the run's table goes to the Actions summary"
    _fake_run(monkeypatch, good, _bench(list(BENCHMARK_INDICES)[2:]), _scored(2600), fresh)
    with contextlib.redirect_stdout(io.StringIO()):
        assert D.main() == 1, "a missing index row must fail the run"


def test_main_refuses_a_missing_secret(monkeypatch):
    monkeypatch.delenv("SHEET_ID", raising=False)
    with pytest.raises(SystemExit):
        D.main()


def test_the_job_loads_exactly_what_the_app_loads():
    """One download, the app's own path: load_all_csvs -> _clean_from + benchmark_table is load_market."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ds = D.de.load_all_csvs("local")
            stocks, bench = D.de.load_market("local")
    except Exception as exc:                                     # the CSVs are gitignored
        pytest.skip(f"local data not available: {exc}")
    with contextlib.redirect_stdout(io.StringIO()):
        pd.testing.assert_frame_equal(D.de._clean_from(ds), stocks)


def test_the_limits_pass_the_real_local_data():
    """Calibration guard: the real data must not fail a check that is not about a real break (freshness and
    the index rows are left out: the local drop is weeks old and predates the index rows)."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ds = D.de.load_all_csvs("local")
            stocks = D.de._clean_from(ds)
            scored = D.run_scoring_pipeline(stocks.copy())
    except Exception as exc:
        pytest.skip(f"local data not available: {exc}")
    results = (D.check_tabs(ds) + D.check_universe(stocks) + D.check_coverage(stocks) + D.check_types(ds)
               + D.check_taxonomy(stocks) + D.check_identities(stocks) + D.check_engine(scored))
    failed = [f"{r.name}: {r.detail}" for r in results if r.status == "fail"]
    assert not failed, failed


# ── the workflow ──────────────────────────────────────────────────────────────────────────────────
_WF = (ROOT / ".github" / "workflows" / "data-check.yml").read_text(encoding="utf-8")


def test_the_workflow_runs_daily_and_on_demand_read_only():
    assert re.search(r'cron:\s*"\d+ \d+ \* \* \*"', _WF), "a daily schedule"
    assert "workflow_dispatch:" in _WF
    assert "contents: read" in _WF and "contents: write" not in _WF, "the check never writes to the repo"


def test_the_workflow_runs_the_job_with_the_sheet_secret():
    assert "SHEET_ID: ${{ secrets.SHEET_ID }}" in _WF
    assert "python -m ui.data_check_job" in _WF
