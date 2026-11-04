"""
test_index_closes.py
====================
The market box's numbers come from NSE's own daily index file (2026-11-04). Display only.

WHY. The vendor's history for Nifty500 Shariah was measured wrong on 3 Nov 2026 against NSE's files (its 200-day
average 7,835.0 against NSE's 6,824.3), so the box read Stage 4 where NSE's numbers give Stage 3; and the operator
asked for the 200-week line Kite's shadow book now gates on (addendum 37, rule 8: NIFTY MIDSML 400 against the simple
average of its last 200 weekly closes, a week's close = its last session's, the current week with its latest close).

Pinned here: NSE's file is read strictly (its own date, its own table, the declared indices only); a close once kept
is never rewritten; the weekly line is Kite's to the decimal (MidSmall 400 on 4 Nov: 17,628.6); an average needs its
whole count or it is not given; the stage is the one rule every stock uses; the nightly job asks only for the days it
lacks, never turns red on a holiday, and writes only to its own branch.
"""
import ast
import csv
import io
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import core.data_engine as de  # noqa: E402
import core.index_closes as X  # noqa: E402
import ui.index_closes_job as J  # noqa: E402
from config import BENCHMARK_INDICES  # noqa: E402

_HEAD = ("Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,"
         "Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield")


def _nse(day, closes=None, written=None):
    """NSE's ind_close_all file for `day`, in the shape of all 205 files read on 4 Nov 2026."""
    closes = closes if closes is not None else {"Nifty 500": "22769.20", "Nifty500 Shariah": "6796.30",
                                                "Nifty 50": "25597.65"}
    d = written or day.strftime("%d-%m-%Y")
    return _HEAD + "\n" + "".join(f"{n},{d},1,1,1,{c},0,0,0,0,0,0,0\n" for n, c in closes.items())


D = date(2026, 11, 4)


# ── 1. NSE's file, read strictly ─────────────────────────────────────────────────────────────────────────────────
def test_the_declared_indices_are_kept_by_vendor_id_with_nses_own_text():
    assert X.parse_nse_file(_nse(D), D) == {"NSE:CNX500": "22769.20", "NSE:NIFTY500SH": "6796.30"}


def test_names_match_whatever_the_case_and_spacing():
    assert X.parse_nse_file(_nse(D, {"NIFTY  500": "22769.20"}), D) == {"NSE:CNX500": "22769.20"}


def test_a_file_carrying_another_date_is_a_stale_copy_never_kept():
    assert X.parse_nse_file(_nse(D, written="03-11-2026"), D) is None
    assert X.parse_nse_file(_nse(D, written="2026-11-04"), D) is None


def test_nses_month_first_date_still_names_its_own_day():
    """Measured 4 Nov 2026 on 205 files: 195 write the date day-first, 9 read the same either way (day = month), and
    ONE writes it month-first — ind_close_all_06042023.csv says 04-06-2023 for 6 Apr 2023 (the Good Friday week's last
    session). Read day-first only, that real session would be thrown away as a stale copy."""
    day = date(2023, 4, 6)
    assert X.parse_nse_file(_nse(day, written="04-06-2023"), day) == {"NSE:CNX500": "22769.20",
                                                                     "NSE:NIFTY500SH": "6796.30"}
    assert X.parse_nse_file(_nse(day, written="06-04-2023"), day) is not None
    assert X.parse_nse_file(_nse(day, written="05-04-2023"), day) is None, "the day before, either way, is stale"


@pytest.mark.parametrize("text", ["<html><body>Access Denied</body></html>", _HEAD + "\n", "",
                                  _nse(D, {"Nifty 50": "25597.65"})])
def test_a_file_that_is_not_nses_index_table_stops_the_run(text):
    """Not the table (a block page), an empty table, nothing at all, or none of the declared indices: NSE changed its
    file, and the run says so instead of keeping nothing quietly."""
    with pytest.raises(X.NseFileError):
        X.parse_nse_file(text, D)


@pytest.mark.parametrize("close", ["-", "", "0", "0.00", "abc", "nan", "inf"])
def test_a_blank_or_unpriced_close_is_not_kept(close):
    assert "NSE:NIFTY500SH" not in X.parse_nse_file(_nse(D, {"Nifty 500": "22769.20", "Nifty500 Shariah": close}), D)


def test_a_bom_and_windows_line_endings_are_read():
    text = "﻿" + _nse(D).replace("\n", "\r\n")
    assert X.parse_nse_file(text, D)["NSE:CNX500"] == "22769.20"


def test_two_different_closes_for_one_index_stop_the_run():
    text = _nse(D) + f"Nifty 500,{D:%d-%m-%Y},1,1,1,22000.00,0,0,0,0,0,0,0\n"
    with pytest.raises(X.NseFileError):
        X.parse_nse_file(text, D)


# ── 2. A close, once kept, is the record ─────────────────────────────────────────────────────────────────────────
def test_a_kept_close_is_never_rewritten_and_the_rows_come_out_sorted():
    kept = [("2026-11-04", "NSE:CNX500", "22769.20")]
    new = [("2026-11-04", "NSE:CNX500", "99999.99"), ("2026-11-03", "NSE:CNX500", "22900.00"),
           ("2026-11-04", "NSE:NIFTY500SH", "6796.30")]
    got = X.merge(kept, new)
    assert got == [("2026-11-03", "NSE:CNX500", "22900.00"), ("2026-11-04", "NSE:CNX500", "22769.20"),
                   ("2026-11-04", "NSE:NIFTY500SH", "6796.30")]
    assert X.merge(got, new) == got, "merging the same rows again changes nothing"


def test_the_file_is_the_same_bytes_whatever_order_the_rows_arrive_in():
    rows = [("2026-11-04", "NSE:NIFTY500SH", "6796.30"), ("2026-11-03", "NSE:CNX500", "22900.00")]
    a, b = X.to_csv(rows), X.to_csv(list(reversed(rows)))
    assert a == b and "\r" not in a and a.startswith("date,company_id,close\n") and a.endswith("\n")
    assert X.from_csv(a) == sorted(rows)


@pytest.mark.parametrize("text", ["day,id,close\n2026-11-04,NSE:CNX500,1\n",
                                  "date,company_id,close\n2026-13-04,NSE:CNX500,1\n",
                                  "date,company_id,close\n2026-11-04,NSE:CNX500\n",
                                  "date,company_id,close\n2026-11-04,NSE:CNX500,-5\n"])
def test_a_damaged_history_file_is_refused_by_name(text):
    with pytest.raises(X.HistoryError):
        X.from_csv(text)


# ── 3. The box's numbers: Kite's weekly line, the one stage rule ─────────────────────────────────────────────────
def _frame(series):
    """{vendor id: [(date, close), ...]} -> the kept closes as the app reads them."""
    return pd.DataFrame([(d.isoformat(), cid, str(c)) for cid, pts in series.items() for d, c in pts],
                        columns=["date", "company_id", "close"])


def _weeks(n, first=date(2022, 1, 3), days=(0, 1, 2, 3, 4), close=lambda w, d: 100.0 + w):
    """n weeks of sessions from a Monday; close(week, weekday)."""
    return [(first + timedelta(weeks=w, days=d), close(w, d)) for w in range(n) for d in days]


def test_a_weeks_close_is_its_last_session_even_when_friday_is_a_holiday():
    pts = _weeks(200, close=lambda w, d: 100.0) + [(date(2022, 1, 3) + timedelta(weeks=200, days=d), c)
                                                   for d, c in ((0, 100.0), (1, 100.0), (2, 100.0), (3, 300.0))]
    v = X.box_view(_frame({"NSE:CNX500": pts})).set_index("company_id")
    # 201 weekly closes: 199 at 100 and the holiday week's Thursday 300 count in the last 200.
    assert v.loc["NSE:CNX500", "weeks"] == 201
    assert v.loc["NSE:CNX500", "sma_200w"] == pytest.approx((199 * 100.0 + 300.0) / 200)


def test_the_current_week_counts_with_its_latest_close():
    pts = _weeks(199, close=lambda w, d: 100.0) + [(date(2022, 1, 3) + timedelta(weeks=199), 500.0)]   # a Monday
    v = X.box_view(_frame({"NSE:CNX500": pts})).set_index("company_id")
    assert v.loc["NSE:CNX500", "weeks"] == 200
    assert v.loc["NSE:CNX500", "sma_200w"] == pytest.approx((199 * 100.0 + 500.0) / 200)
    assert v.loc["NSE:CNX500", "close"] == 500.0


def test_a_weekend_session_belongs_to_its_own_week():
    """A Muhurat session on a Sunday is that week's last session (Monday to Sunday), not the next week's first."""
    pts = _weeks(200, close=lambda w, d: 100.0) + [(date(2022, 1, 3) + timedelta(weeks=199, days=6), 300.0)]
    v = X.box_view(_frame({"NSE:CNX500": pts})).set_index("company_id")
    assert v.loc["NSE:CNX500", "weeks"] == 200
    assert v.loc["NSE:CNX500", "sma_200w"] == pytest.approx((199 * 100.0 + 300.0) / 200)


def test_an_average_is_given_only_on_its_whole_count():
    young = _weeks(199)                                   # 199 weeks: no 200-week average
    old = _weeks(200)
    v = X.box_view(_frame({"NSE:CNX500": old, "NSE:NIFTY500SH": young})).set_index("company_id")
    assert np.isnan(v.loc["NSE:NIFTY500SH", "sma_200w"]) and np.isnan(v.loc["NSE:NIFTY500SH", "vs_200w"])
    assert v.loc["NSE:CNX500", "sma_200w"] == pytest.approx(np.mean([100.0 + w for w in range(200)]))
    short = X.box_view(_frame({"NSE:CNX500": _weeks(29)})).set_index("company_id").loc["NSE:CNX500"]
    assert np.isnan(short["sma_30w"]), "29 weeks: no 30-week average"
    assert short["stage"] == "❔ Unknown"
    few = X.box_view(_frame({"NSE:CNX500": _weeks(39)})).set_index("company_id").loc["NSE:CNX500"]   # 195 sessions
    assert np.isnan(few["sma_200d"]) and not np.isnan(few["sma_30w"]), "195 sessions: no 200-day average"


def test_a_lone_session_before_a_long_gap_never_joins_an_average():
    """Measured on the seed (4 Nov 2026): the three young indices each appear ONCE in NSE's file of 12 Apr 2023, then
    not for 172-284 sessions until they launch. An average reaching back across that hole would mix two series, so
    the series starts again after a gap of more than one trading week; the seven older indices never miss a session."""
    market = _weeks(250)                                          # every session, 250 weeks
    young = [market[0]] + [p for p in market if p[0] >= date(2022, 1, 3) + timedelta(weeks=60)]
    v = X.box_view(_frame({"NSE:CNX500": market, "NSE:NIF500MO50": young})).set_index("company_id")
    assert v.loc["NSE:NIF500MO50", "weeks"] == 190, "the lone 2022 session is not a week of this series"
    assert v.loc["NSE:CNX500", "weeks"] == 250
    # 200 weeks after the hole the line exists, from those 200 weeks alone:
    long = [market[0]] + [p for p in _weeks(270) if p[0] >= date(2022, 1, 3) + timedelta(weeks=60)]
    v = X.box_view(_frame({"NSE:CNX500": _weeks(270), "NSE:NIF500MO50": long})).set_index("company_id")
    assert v.loc["NSE:NIF500MO50", "sma_200w"] == pytest.approx(np.mean([100.0 + w for w in range(70, 270)]))


def test_a_short_gap_is_bridged():
    """A day or two NSE's file misses an index does not restart its series."""
    market = _weeks(210)
    holey = [p for i, p in enumerate(market) if i not in (500, 501, 502)]        # three sessions missing
    v = X.box_view(_frame({"NSE:CNX500": market, "NSE:MIDSMAL400": holey})).set_index("company_id")
    assert v.loc["NSE:MIDSMAL400", "weeks"] == 210 and not np.isnan(v.loc["NSE:MIDSMAL400", "sma_200w"])


def test_the_200_day_and_30_week_averages_are_the_plain_means():
    pts = _weeks(60)                                       # 300 sessions, close 100+week
    v = X.box_view(_frame({"NSE:CNX500": pts})).set_index("company_id").loc["NSE:CNX500"]
    closes = [c for _, c in pts]
    assert v["sma_200d"] == pytest.approx(np.mean(closes[-200:]))
    assert v["sma_30w"] == pytest.approx(np.mean([100.0 + w for w in range(30, 60)]))


def test_vs_200w_is_the_close_against_its_200_week_average():
    pts = _weeks(199, close=lambda w, d: 100.0) + [(date(2022, 1, 3) + timedelta(weeks=199), 299.0)]
    v = X.box_view(_frame({"NSE:CNX500": pts})).set_index("company_id").loc["NSE:CNX500"]
    assert v["sma_200w"] == pytest.approx(100.995)
    assert v["vs_200w"] == pytest.approx((299.0 / 100.995 - 1) * 100)


def test_as_of_reads_the_market_on_the_sheets_own_date():
    pts = _weeks(250)
    full = X.box_view(_frame({"NSE:CNX500": pts})).set_index("company_id").loc["NSE:CNX500"]
    cut = X.box_view(_frame({"NSE:CNX500": pts}), as_of=date(2022, 1, 3) + timedelta(weeks=220, days=2))
    cut = cut.set_index("company_id").loc["NSE:CNX500"]
    assert cut["last_date"] == date(2022, 1, 3) + timedelta(weeks=220, days=2)
    assert cut["close"] == 100.0 + 220 and full["close"] == 100.0 + 249


def test_only_the_declared_indices_in_config_order():
    pts = _weeks(5)
    v = X.box_view(_frame({cid: pts for cid in reversed(list(BENCHMARK_INDICES))} | {"NSE:NIFTY": pts}))
    assert v["company_id"].tolist() == list(BENCHMARK_INDICES)
    assert v["name"].tolist() == list(BENCHMARK_INDICES.values())


def test_no_rows_give_an_empty_view_with_its_columns():
    for empty in (None, pd.DataFrame(columns=["date", "company_id", "close"])):
        v = X.box_view(empty)
        assert v.empty and {"company_id", "stage", "vs_200w", "last_date"} <= set(v.columns)


def test_the_stage_is_the_one_rule_every_stock_is_staged_with():
    pts = _weeks(60, close=lambda w, d: 100.0 + (w % 7) * 3 - d)
    v = X.box_view(_frame({"NSE:CNX500": pts})).set_index("company_id")
    want = de.weinstein_stage(v["close"], v["sma_30w"], v["sma_200d"])
    assert v["stage"].tolist() == list(want)
    tree = ast.parse((ROOT / "core" / "index_closes.py").read_text(encoding="utf-8"))
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "weinstein_stage" in calls and "np.select" not in calls, "the rule is called, never copied"


def test_midsmall_400_reads_kites_200_week_average_to_the_decimal():
    """NSE's own closes (tests/fixtures, from the seed; a special Saturday or Sunday session is its week's last): on
    3 Nov 2026 the line is Kite's gate figure, 17,626.9, to the decimal; on 4 Nov, with that day's close (20,251.7) in
    the current week, 17,628.7, and the close 14.9% above it."""
    with open(ROOT / "tests" / "fixtures" / "nse_midsmall400_week_closes.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.reader(line for line in fh if not line.startswith("#"))][1:]
    frame = _frame({"NSE:MIDSMAL400": [(date.fromisoformat(d), float(c)) for d, c in rows]})
    on_3 = X.box_view(frame, as_of=date(2026, 11, 3)).set_index("company_id").loc["NSE:MIDSMAL400"]
    assert round(on_3["sma_200w"], 1) == 17626.9, "Kite's gate figure for 3 Nov"
    on_4 = X.box_view(frame, as_of=date(2026, 11, 4)).set_index("company_id").loc["NSE:MIDSMAL400"]
    assert round(on_4["sma_200w"], 1) == 17628.7 and on_4["close"] == 20251.7 and on_4["weeks"] == 205
    assert round(on_4["vs_200w"], 1) == 14.9


def test_a_history_the_app_cannot_read_gives_an_empty_frame(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(requests, "get", boom)
    got = X.read_history("https://example.invalid/index_closes.csv")
    assert got.empty and list(got.columns) == ["date", "company_id", "close"]

    class Damaged:
        status_code, text = 200, "not,a,history\n1,2,3\n"
        def raise_for_status(self):
            pass
    monkeypatch.setattr(requests, "get", lambda *a, **k: Damaged())
    assert X.read_history("https://example.invalid/index_closes.csv").empty


# ── 4. The nightly job ───────────────────────────────────────────────────────────────────────────────────────────
def test_the_job_asks_only_for_the_days_it_lacks():
    start = date(2026, 10, 26)
    assert J.days_to_ask(set(), date(2026, 10, 28), start) == [date(2026, 10, 26), date(2026, 10, 27),
                                                               date(2026, 10, 28)]
    kept = {date(2026, 10, d) for d in (26, 27, 28, 29, 30)}
    got = J.days_to_ask(kept, date(2026, 11, 4), start)
    assert got == [date(2026, 10, 31), date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4)]
    # The look-back reaches J.LOOKBACK_DAYS before the newest kept day: a late file is asked for again.
    gap = {date(2026, 10, 26), date(2026, 10, 28)}
    assert date(2026, 10, 27) in J.days_to_ask(gap, date(2026, 10, 28), start)
    assert J.days_to_ask(kept, date(2026, 11, 4), start, backfill_from=date(2026, 10, 25))[0] == date(2026, 10, 25)


class _Fetch:
    """NSE as the job sees it: {day: file text}, anything else 404."""
    def __init__(self, files):
        self.files, self.asked = files, []

    def __call__(self, day):
        self.asked.append(day)
        return self.files.get(day)


def test_a_first_run_writes_the_history_its_readme_and_attributes(tmp_path):
    fetch = _Fetch({date(2026, 11, 3): _nse(date(2026, 11, 3)), date(2026, 11, 4): _nse(D)})
    changed, msg = J.run(tmp_path, date(2026, 11, 4), fetch, start=date(2026, 11, 2))
    assert changed and "2026-11-04" in msg
    assert X.from_csv((tmp_path / "index_closes.csv").read_text(encoding="utf-8")) == [
        ("2026-11-03", "NSE:CNX500", "22769.20"), ("2026-11-03", "NSE:NIFTY500SH", "6796.30"),
        ("2026-11-04", "NSE:CNX500", "22769.20"), ("2026-11-04", "NSE:NIFTY500SH", "6796.30")]
    assert (tmp_path / ".gitattributes").read_text(encoding="utf-8") == "* -text\n"
    assert "display only" in (tmp_path / "README.md").read_text(encoding="utf-8").lower()


def test_a_second_run_with_nothing_new_changes_nothing(tmp_path):
    files = {D: _nse(D)}
    J.run(tmp_path, D, _Fetch(files), start=D)
    before = (tmp_path / "index_closes.csv").read_bytes()
    changed, _ = J.run(tmp_path, D, _Fetch(files), start=D)
    assert not changed and (tmp_path / "index_closes.csv").read_bytes() == before


def test_a_holiday_or_a_file_not_out_yet_adds_nothing_and_is_not_an_error(tmp_path):
    changed, msg = J.run(tmp_path, D, _Fetch({}), start=D)
    assert not changed and "nothing new" in msg


def test_a_stale_copy_is_skipped_with_a_warning(tmp_path, capsys):
    J.run(tmp_path, D, _Fetch({D: _nse(D, written="03-11-2026")}), start=D)
    assert not (tmp_path / "index_closes.csv").exists() or "2026-11-04" not in (tmp_path / "index_closes.csv").read_text()
    assert "another date" in capsys.readouterr().out


def test_a_broken_nse_file_stops_the_run(tmp_path):
    with pytest.raises(X.NseFileError):
        J.run(tmp_path, D, _Fetch({D: "<html>Access Denied</html>"}), start=D)


def test_the_job_never_rewrites_a_kept_close(tmp_path):
    J.run(tmp_path, D, _Fetch({D: _nse(D)}), start=D)
    J.run(tmp_path, D, _Fetch({D: _nse(D, {"Nifty 500": "1.00"})}), start=D, backfill_from=D)
    assert ("2026-11-04", "NSE:CNX500", "22769.20") in X.from_csv((tmp_path / "index_closes.csv").read_text())


class _Resp:
    def __init__(self, code, text=""):
        self.status_code, self.text = code, text


class _Session:
    def __init__(self, codes):
        self.codes, self.urls = list(codes), []

    def get(self, url, **kw):
        self.urls.append(url)
        code = self.codes.pop(0)
        return _Resp(code, "table" if code == 200 else "")


def test_nse_is_asked_politely_and_a_404_means_no_file(monkeypatch):
    monkeypatch.setattr(J.time, "sleep", lambda s: None)
    s = _Session([404])
    assert J.fetch_day(D, s) is None and s.urls == [config.NSE_INDEX_FILE_URL.format(D)]
    assert J.fetch_day(D, _Session([500, 503, 200])) == "table", "a passing outage is retried"
    with pytest.raises(RuntimeError, match="403"):
        J.fetch_day(D, _Session([403, 403, 403]))


def test_main_tells_the_workflow_whether_to_commit(tmp_path, monkeypatch):
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setattr(J, "run", lambda *a, **k: (True, "index closes: 2026-11-04 · 10 indices · 1 new day"))
    monkeypatch.setattr(J, "_session", lambda: None)
    assert J.main(["--out", str(tmp_path / "closes"), "--backfill-from", ""]) == 0
    text = out.read_text(encoding="utf-8")
    assert "changed=true" in text and "message=index closes: 2026-11-04" in text


# ── 5. The workflow ──────────────────────────────────────────────────────────────────────────────────────────────
_WF = (ROOT / ".github" / "workflows" / "index-closes.yml")


def test_the_workflow_writes_only_its_own_branch_and_only_when_something_changed():
    y = _WF.read_text(encoding="utf-8")
    assert "contents: write" in y and "group: index-closes" in y and "cancel-in-progress: false" in y
    assert re.search(r"ref: index-closes\n\s+path: closes\n", y), "the history branch is checked out beside the code"
    assert "python -m ui.index_closes_job --out closes" in y
    pushes = re.findall(r"git push origin (\S+)", y)
    assert pushes == ["HEAD:index-closes"], pushes
    assert "if: steps.closes.outputs.changed == 'true'" in y
    assert re.search(r"timeout-minutes: \d+", y)


def test_the_backfill_input_reaches_the_script_only_through_env():
    y = _WF.read_text(encoding="utf-8")
    runs = re.findall(r"run: (.*)", y)
    assert not any("${{" in r for r in runs), "an input pasted into a script runs as shell beside the token"
    assert "BACKFILL_FROM: ${{ inputs.backfill_from }}" in y


def test_the_workflow_fires_after_the_evening_and_again_in_the_morning():
    y = _WF.read_text(encoding="utf-8")
    crons = re.findall(r'cron: "([^"]+)"', y)
    assert len(crons) == 2 and "workflow_dispatch:" in y, crons


def test_the_config_names_one_branch_and_its_raw_file():
    assert config.INDEX_CLOSES_URL.endswith(f"/{config.INDEX_CLOSES_BRANCH}/index_closes.csv")
    assert config.NSE_INDEX_FILE_URL.format(D).endswith("ind_close_all_04112026.csv")
    assert date.fromisoformat(config.INDEX_CLOSES_START) + timedelta(weeks=config.INDEX_WEEKS_LONG) < D
