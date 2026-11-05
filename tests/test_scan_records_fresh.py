"""
test_scan_records_fresh.py — the records of 🚀 and 🌬️ start fresh from 5 Nov 2026, and 🌱 keeps its study.

The operator, 2026-11-05: "Remove old record of scan record completely, and from today is the fresh." So the design-period
windows are gone from ui/scan_records.json (they stay in git's history, at b19d485), and no card, guide or download quotes
them. The same evening, on seeing 🌱's 3-year study gone with them: "PREVIOUS HOW IT WAS WRITTEN BEFORE RAM IN 🌱 Compounder
Candidates ... KEEP THAN ONLY". So 🌱 keeps its study, its "How the backtest was run", its own date (7 Oct) and its October
windows, exactly as they read at bee8641. Each scan's fingerprint, the 🆕 baseline, the hand checks' file and the job's alarms
all stay.

These tests hold what could break quietly with nobody around:
1. the file, and the weekly job on it, its Archive alarm above all;
2. which copies can start a window, and the Mondays that follow from that;
3. what a person sees before the first window, and that the first window reaches a card at all: the cards used to read an
   unseen window only beside a design-period record, so an empty one would have hidden the whole unseen record;
4. that nothing left in a file, or typed in the code, brings an old figure back.

The job rewrites the file, so a test of a card or of a later Monday reads the file's record with its unseen windows emptied.
"""
import html as _html
import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.test_scan_records_job import _later, _market  # noqa: E402
from tests.test_scans import _universe  # noqa: E402
from ui import scan_guide as G  # noqa: E402
from ui import scan_records_job as J  # noqa: E402
from ui import ui_scans as S  # noqa: E402

_SCANS = {s["key"]: s for s in S.SCANS}
_SHORT = ("playbook", "leaders")
_KEYS = (*_SHORT, "compounder")


def _file() -> dict:
    return json.loads(S.RECORDS_PATH.read_text(encoding="utf-8"))


def _fresh(key: str) -> dict:
    """The file's record for one scan as it stands before any window."""
    return dict(_file()["scans"][key], out_of_sample=[])


def _text(h: str) -> str:
    return " ".join(_html.unescape(re.sub(r"<[^>]+>", " ", h)).split())


# ── 1. The file, and the weekly job on it ─────────────────────────────────────────────────────────────────────────────
def test_the_file_holds_only_what_stays():
    rec = _file()
    assert set(rec) == {"version", "designed_on", "generated", "engine_version", "method", "scans", "members_date"}, (
        "latest_snapshot described the old record only")
    assert sorted(rec["scans"]) == sorted(_KEYS)
    for key, r in rec["scans"].items():
        study = {"study"} if key in S.STUDY_SCANS else set()
        assert set(r) <= {"fingerprint", "designed_on", "horizon_days", "windows", "out_of_sample", "members_at_latest", *study}, key
        assert r["windows"] == [], f"{key}: a design-period window is back"
        assert set(r) >= study, f"{key}: its study is gone"


def test_the_method_words_describe_only_the_record_that_is_left():
    method = _file()["method"].lower()
    for word in ("design period", "in-sample", "snapshot"):
        assert word not in method, word
    assert "month-end" in method and "designed_on" in method


def test_the_file_carries_the_date_every_scan_counts_from():
    rec = _file()
    assert rec["designed_on"] == S.DESIGNED_ON == "2026-11-05"
    comp = rec["scans"]["compounder"]
    assert comp["designed_on"] == S.scan_designed_on(_SCANS["compounder"]) and comp["horizon_days"] == 365


def test_the_compounder_keeps_its_own_date_and_its_study_on_file():
    """7 Oct, as the recipe's own comment says, and the study of bee8641 to the last number: its method is stamped with it."""
    comp = _file()["scans"]["compounder"]
    assert comp["designed_on"] == "2026-10-07" == S.scan_designed_on(_SCANS["compounder"]) != S.DESIGNED_ON
    assert S.scan_start_month(_SCANS["compounder"]) == 10
    assert G.study_stamp(comp["study"]) == "b49a4a17fde1607a" == G.GUIDES["compounder"]["backtest"]["written_for"]
    assert G.guide_stamp(_SCANS["compounder"]) == "63ba93225ce77da7" == G.GUIDES["compounder"]["written_for"]


def test_a_rewrite_by_the_job_leaves_the_study_to_the_byte():
    """The file on disk is what json.dumps writes, so a run that changes a baseline moves no byte of the study."""
    text = S.RECORDS_PATH.read_text(encoding="utf-8")
    assert json.dumps(json.loads(text), ensure_ascii=False, indent=1) + "\n" == text


def test_the_baseline_is_kept_so_its_stocks_are_not_new():
    rec = _file()
    date.fromisoformat(rec["members_date"])
    members = max((r["members_at_latest"] for r in rec["scans"].values()), key=len)
    assert members, "with no baseline every stock would read new"
    df = _universe().assign(company_id=lambda d: [members[0]] + [f"NSE:NOT{k}" for k in range(len(d) - 1)])
    t = S.scan_table(df, {"key": "all", "mask": lambda d: pd.Series(True, index=d.index)}, members_at_latest=members)
    new = dict(zip(t["company_id"], t["_new"]))
    assert new[members[0]] == "" and new["NSE:NOT0"] == "🆕"


def test_a_quiet_week_writes_nothing_on_the_file():
    """The file must say what the job would write, or the first quiet Monday commits a date with no measured change."""
    rec = _file()
    same = {k: {"out_of_sample": r["out_of_sample"], "members_at_latest": r["members_at_latest"]}
            for k, r in rec["scans"].items()}
    _, changed = J.merge(rec, same, {"generated": "2027-01-04", "engine_version": "zzz", "members_date": rec["members_date"]})
    assert not changed


def _job(monkeypatch, dates, today, frames=None, written=None, measured=None, fresh=True):
    """main() with the REAL scans, the REAL hand checks and the file's records (with `fresh`, its unseen windows emptied, as
    the job leaves them from January) on a fake archive. `frames`: {date: copy}, by default one synthetic market everywhere."""
    import ui.ui_movers as M
    base = _market()
    frames = frames or {}
    if fresh:
        records = _file()
        for r in records["scans"].values():
            r["out_of_sample"] = []
        monkeypatch.setattr(S, "load_records", lambda: records)
    monkeypatch.setattr(M, "load_index", lambda _id: pd.DataFrame(
        {"vintage_date": dates, "spreadsheet_id": [f"id-{d}" for d in dates]}))
    monkeypatch.setattr(J, "load_market", lambda source, files, sid: (
        frames.get(sid.removeprefix("id-"), base).copy(), pd.DataFrame(columns=["company_id", "close_price"])))
    monkeypatch.setattr(J, "run_scoring_pipeline", lambda df: df)
    monkeypatch.setattr(J, "copy_problem", lambda frame, day: "")       # synthetic copies: the input check has its own tests
    monkeypatch.setattr(J, "engine_version", lambda: "test-engine")
    monkeypatch.setattr(J, "_today", lambda: today)
    monkeypatch.setenv("ARCHIVE_INDEX_SHEET_ID", "x" * 44)
    written = {} if written is None else written
    measured = [] if measured is None else measured
    real = J.measure_window
    monkeypatch.setattr(J, "measure_window", lambda *a, **k: measured.append((a[2]["key"], a[3], a[4])) or real(*a, **k))
    monkeypatch.setattr(J, "write", lambda rec: written.update(rec))
    J.main()
    return written, measured


def test_the_archive_alarm_fires_on_the_committed_file(monkeypatch):
    """The real file, the real fingerprints and the real checks file: nothing before the alarm may stop the run first (a stale
    fingerprint or a missing checks file would, with other words), and it fires with nothing measured or written."""
    written, measured = {}, []
    with pytest.raises(SystemExit, match="stopped taking copies") as stop:
        _job(monkeypatch, ["2026-10-30", "2026-11-09"], date(2026, 11, 20), written=written, measured=measured, fresh=False)
    assert "2026-11-09, 11 days old" in str(stop.value)
    assert (written, measured) == ({}, [])


def test_an_archive_ten_days_old_is_no_alarm_on_the_committed_file(monkeypatch):
    written, _ = _job(monkeypatch, ["2026-10-30", "2026-11-09"], date(2026, 11, 19))
    assert written["members_date"] == "2026-10-30"


def test_a_stale_fingerprint_says_what_to_do_now():
    """The old stop sent a person to tools/scan_records.py, which would write the design-period windows back."""
    stale = {"scans": {s["key"]: {"fingerprint": "0" * 16} for s in S.SCANS}}
    with pytest.raises(SystemExit) as stop:
        J.check_fresh(stale)
    said = str(stop.value)
    assert "scan_records.py" not in said and "design-period" not in said
    assert S.recipe_fingerprint(S.SCANS[0]) in said and S.RECORDS_PATH.name in said


def test_a_missing_file_says_what_to_do_now(monkeypatch):
    monkeypatch.setattr(S, "load_records", lambda: {})
    monkeypatch.setenv("ARCHIVE_INDEX_SHEET_ID", "x" * 44)
    with pytest.raises(SystemExit) as stop:
        J.main()
    said = str(stop.value)
    assert "tools/scan_records.py" not in said and S.RECORDS_PATH.name in said and "git" in said


# ── 2. Which copies can start a window, and the Mondays that follow ───────────────────────────────────────────────────
_LATER_COPIES = ["2026-11-30", "2026-12-31", "2027-01-29", "2027-04-30", "2027-10-29", "2027-11-30", "2028-10-31", "2028-11-03"]


@pytest.mark.parametrize("october", ["2026-10-30", "2026-10-31"])
@pytest.mark.parametrize("key", _SHORT)
def test_no_copy_from_before_today_ever_starts_a_window(key, october):
    """October's month-end copy is dated by its last trading day or by the month's end: neither starts one on 🚀 or 🌬️."""
    windows = J.scan_windows(["2026-09-30", october, *_LATER_COPIES], _SCANS[key])
    assert windows, "the archive runs long enough to hold a window on every scan"
    assert min(a for a, _ in windows) > "2026-11-05", windows


@pytest.mark.parametrize("october", ["2026-10-30", "2026-10-31"])
def test_the_compounders_first_window_starts_on_the_october_copy(october):
    """Its own clock: one window a year from the October copy after its 7 Oct date, as it was before the fresh start."""
    windows = J.scan_windows(["2026-09-30", october, *_LATER_COPIES], _SCANS["compounder"])
    assert windows[0][0] == october and all(a[5:7] == "10" for a, _ in windows), windows


@pytest.mark.parametrize("key,first", [("playbook", "2026-11-30"), ("leaders", "2026-11-30"), ("compounder", "2026-10-30")])
def test_each_scans_first_window_starts_where_its_card_says(key, first):
    """🌱's one window a year starts each October (`start_month`), so its first starts a month before the short lists'."""
    windows = J.scan_windows(["2026-09-30", "2026-10-30", *_LATER_COPIES], _SCANS[key])
    assert windows[0][0] == first
    assert S._first_unseen_start(_SCANS[key]).strftime("%Y-%m") == first[:7]


def test_a_copy_dated_on_the_record_date_starts_no_window():
    """The code says `d > designed_on`, and "from today" means after it. It would matter only if November's kept copy were dated
    exactly 5 Nov, every later November ingest having failed. Found by a mutant no test caught, in the review of 5 Nov."""
    assert J.scan_windows(["2026-11-05", "2026-12-31", "2027-01-29"], _SCANS["playbook"]) == [("2026-12-31", "2027-01-29")]


# What each Monday's run sees on 🚀 and 🌬️: the month-end copies plus the current month's newest, and the windows it can have.
# 25 Dec 2026 is a Friday and an NSE holiday, so the newest copy on Monday 28 Dec is 24 days old: still short of the 25.
_BEFORE = ["2026-09-30", "2026-10-30"]
_MONDAYS = [("2026-11-09", [*_BEFORE, "2026-11-06"], []),
            ("2026-11-30", [*_BEFORE, "2026-11-27"], []),
            ("2026-12-07", [*_BEFORE, "2026-11-30", "2026-12-04"], []),
            ("2026-12-28", [*_BEFORE, "2026-11-30", "2026-12-24"], []),
            ("2027-01-04", [*_BEFORE, "2026-11-30", "2026-12-31", "2027-01-01"],
             [("2026-11-30", "2026-12-31"), ("2026-11-30", "2027-01-01")])]


@pytest.mark.parametrize("monday,dates,windows", _MONDAYS, ids=[m[0] for m in _MONDAYS])
@pytest.mark.parametrize("key", _SHORT)
def test_what_each_monday_can_measure(key, monday, dates, windows):
    assert J.scan_windows(dates, _SCANS[key]) == windows
    assert J.archive_problem(dates, date.fromisoformat(monday)) == ""


def test_the_next_monday_measures_nothing_and_moves_only_the_baseline(monkeypatch):
    written, measured = _job(monkeypatch, [*_BEFORE, "2026-11-06"], date(2026, 11, 9))
    assert measured == [], "the 30 Oct copy is the baseline: on 🚀 and 🌬️ never a start, and 🌱's window from it needs a year"
    assert written["members_date"] == "2026-10-30" and written["designed_on"] == "2026-11-05"
    on_file = _file()["scans"]
    for key in _KEYS:
        r = written["scans"][key]
        assert r["out_of_sample"] == [] and r["windows"] == [] and r.get("study") == on_file[key].get("study"), key
        assert r["members_at_latest"] == J.members(_market(), _SCANS[key]), key
    assert written["scans"]["compounder"]["study"]["n_match"] == 182, "🌱's study, untouched by the run"
    assert written["scans"]["compounder"]["designed_on"] == "2026-10-07", "and its own date"


@pytest.mark.parametrize("monday,newest", [(date(2026, 12, 7), ["2026-12-04"]), (date(2026, 12, 14), ["2026-12-11"]),
                                           (date(2026, 12, 21), ["2026-12-18"]), (date(2026, 12, 28), ["2026-12-24"])],
                         ids=["7 Dec", "14 Dec", "21 Dec", "28 Dec"])
def test_no_monday_of_december_can_go_red_on_a_fall(monkeypatch, monday, newest):
    """The alarm sheet's old date: on the old record the 31 Oct window was final by 7 Dec, so an unchecked fall stopped the run.
    Here every copy falls hard against 30 Nov, and no window exists to name it."""
    early = _market()
    copies = {"2026-11-30": early, newest[0]: _later(early)}
    written, measured = _job(monkeypatch, [*_BEFORE, "2026-11-30", *newest], monday, frames=copies)
    assert measured == [] and written["members_date"] == "2026-11-30"


def _first_month_frames(fall=False):
    early = _market()
    late = _later(early) if fall else early.assign(close_price=early["close_price"] * 1.1)
    return {"2026-11-30": early, "2026-12-31": late, "2027-01-01": late}


_JAN_4 = [*_BEFORE, "2026-11-30", "2026-12-31", "2027-01-01"]


def test_the_first_red_monday_is_the_first_of_2027(monkeypatch):
    """The first window is final the day it first exists, so a fall no check covers stops that very run."""
    written = {}
    with pytest.raises(SystemExit, match=r"playbook: 2026-11-30 → 2026-12-31 is about to be written final") as stop:
        _job(monkeypatch, _JAN_4, date(2027, 1, 4), frames=_first_month_frames(fall=True), written=written)
    assert written == {} and "scan_record_checks.json" in str(stop.value)


def test_the_first_windows_are_written_on_the_first_monday_of_2027_and_reach_the_cards(monkeypatch):
    """The whole path on the real scans: the job measures 30 Nov on, writes the month final and the newest copy running, 🌱
    measures nothing yet, and both cards then read those windows. Before this change a record with no design-period windows
    read "Not yet measured" on both cards, whatever the job had written."""
    written, measured = _job(monkeypatch, _JAN_4, date(2027, 1, 4), frames=_first_month_frames())
    assert measured == [(k, "2026-11-30", end) for k in _SHORT for end in ("2026-12-31", "2027-01-01")]
    assert written["scans"]["compounder"]["out_of_sample"] == []
    assert written["scans"]["compounder"]["study"] == _file()["scans"]["compounder"]["study"]
    for key in _SHORT:
        scan = _SCANS[key]
        month, running = written["scans"][key]["out_of_sample"]
        assert (month["end"], month["final"], month["engine"]) == ("2026-12-31", True, "test-engine"), key
        assert (running["end"], running["final"]) == ("2027-01-01", False), key
        assert month["fingerprint"] == S.recipe_fingerprint(scan) and "ram_pct" not in month["order"], key
        state, rec = S.record_state(written, scan)
        full, summ = _text(S.card_html(scan, 9, state, rec, True)), _text(S.card_summary_html(scan, 9, state, rec, False))
        assert state == "measured" and "NEVER SAW" in full and "so far" in full, key
        assert re.search(r"ahead in \d of 2 unseen windows", summ) and "1 of them still running" in summ, key
        assert "No record yet" not in full + summ and "Not yet measured" not in full + summ, key


def test_the_compounders_first_window_is_written_a_year_on_beside_its_study(monkeypatch):
    """The whole path on the real scans: 🌱's first window runs from the October copy to the first copy a year later, and the
    card then reads it under the study, which the run left as it was. The first copy 365+ days on from 30 Oct 2026 is 1 Nov 2027."""
    dates = [*_BEFORE, "2026-11-30", "2027-10-29", "2027-11-01"]
    early = _market()
    late = early.assign(close_price=early["close_price"] * 1.1)
    written, measured = _job(monkeypatch, dates, date(2027, 11, 2),
                             frames={"2026-10-30": early, "2027-10-29": late, "2027-11-01": late})
    assert [m for m in measured if m[0] == "compounder"] == [("compounder", "2026-10-30", "2027-11-01")]
    comp = written["scans"]["compounder"]
    assert comp["study"] == _file()["scans"]["compounder"]["study"] and comp["designed_on"] == "2026-10-07"
    [window] = comp["out_of_sample"]
    assert (window["start"], window["end"], window["final"]) == ("2026-10-30", "2027-11-01", False)
    state, rec = S.record_state(written, _SCANS["compounder"])
    full = _text(S.card_html(_SCANS["compounder"], 9, state, rec, True))
    assert state == "measured" and full.index("3-YEAR BACKTEST · IN-SAMPLE") < full.index("NEVER SAW") and "pending" not in full


def _month_chain(reading, through="2027-04-30"):
    months = [m for m in ("2026-10-30", "2026-11-30", "2026-12-31", "2027-01-29", "2027-02-26", "2027-03-31", "2027-04-30",
                          "2027-05-07") if m <= through]
    done = set(J.frozen(months))           # a window is final once its end copy's month is over, as keep_final writes it
    return [dict(start=a, end=b, final=b in done, **reading(a)) for a, b in J.scan_windows(months, _SCANS["playbook"])]


def test_the_revenue_led_check_reads_december_january_and_february():
    gaps = {"2026-11-30": 1.1, "2026-12-31": 1.2, "2027-01-29": 1.3, "2027-02-26": 1.4, "2027-03-31": 1.5}
    v = S.revenue_led_rule(_month_chain(lambda a: dict(revenue_led=dict(mean_gap_pp=gaps.get(a, 99.0), median_gap_pp=2.0))))
    assert v["state"] == "held" and v["mean_gaps"] == [1.1, 1.2, 1.3]


def test_the_order_checks_first_window_is_30_nov_and_its_fourth_ends_on_31_mar():
    ram = {"2026-11-30": 70, "2026-12-31": 72, "2027-01-29": 74, "2027-02-26": 76, "2027-03-31": 78}
    reading = lambda a: dict(order=dict(n=40, ram_pct=ram.get(a, 99), rank_pct=50))  # noqa: E731
    v = S.order_rule(_month_chain(reading))
    assert v["state"] == "held" and v["ram_avg"] == pytest.approx(73.0), "70, 72, 74 and 76: nothing from the 30 Oct copy"
    assert S.order_rule(_month_chain(reading, through="2027-03-31"))["counted"] == 3, (
        "the copy after 31 Mar is what finishes its month, so the fourth window counts from April")


# ── 3. What a person sees ─────────────────────────────────────────────────────────────────────────────────────────────
_FIRST = {"playbook": "30 Nov", "leaders": "30 Nov"}
_NEVER_ON_A_FRESH_CARD = ("IN-SAMPLE", "In-sample", "in-sample", "DESIGN-PERIOD", "BACKTEST", "backtest", "Not yet measured",
                          "NEVER SAW", "<table", "provisional", "Doubled", "better odds")


def _cards(scan, state, rec):
    return {"summary": S.card_summary_html(scan, 7, state, rec, False), "full": S.card_html(scan, 7, state, rec, True)}


@pytest.mark.parametrize("card", ["summary", "full"])
@pytest.mark.parametrize("key", _SHORT)
def test_a_card_with_no_record_says_so_plainly(key, card):
    state, rec = S.record_state({"scans": {key: _fresh(key)}}, _SCANS[key])
    assert state == "measured", "a current recipe with no window is not a stale or missing record"
    h = _cards(_SCANS[key], state, rec)[card]
    t = _text(h)
    assert f"No record yet. The first window starts on the {_FIRST[key]} copy." in t
    assert not re.search(r"\bnan\b", t, re.I) and "None" not in t, t
    for word in _NEVER_ON_A_FRESH_CARD:
        assert word not in h, word


def test_the_compounder_card_says_its_windows_run_a_year_and_start_each_october():
    state, rec = S.record_state({"scans": {"compounder": _fresh("compounder")}}, _SCANS["compounder"])
    assert "one 1-year window a year, each starting at the end of October" in _text(
        S.card_html(_SCANS["compounder"], 7, state, rec, True))
    state, rec = S.record_state({"scans": {"playbook": _fresh("playbook")}}, _SCANS["playbook"])
    assert "a year" not in _text(S.card_html(_SCANS["playbook"], 7, state, rec, True))


@pytest.mark.parametrize("card", ["summary", "full"])
def test_the_compounder_card_reads_its_study_and_never_no_record(card):
    state, rec = S.record_state({"scans": {"compounder": _fresh("compounder")}}, _SCANS["compounder"])
    t = _text(_cards(_SCANS["compounder"], state, rec)[card])
    assert "IN-SAMPLE" in t and "+20.4%/yr vs +4.4%/yr for the rest" in t and "31 Oct 2027" in t and "did not double" in t
    assert "No record yet" not in t and "Not yet measured" not in t


def test_a_compounder_file_that_lost_its_study_says_no_record_on_its_own_date():
    """Never a crash and never a blank: the card reads like the short lists', on 🌱's date."""
    rec = {k: v for k, v in _fresh("compounder").items() if k != "study"}
    state, got = S.record_state({"scans": {"compounder": rec}}, _SCANS["compounder"])
    for t in map(_text, _cards(_SCANS["compounder"], state, got).values()):
        assert "No record yet. The first window starts on the 31 Oct copy." in t and "IN-SAMPLE" not in t


@pytest.mark.parametrize("key", _KEYS)
def test_a_record_with_no_window_keeps_its_state_and_its_baseline(key):
    """app.py reads the 🆕 baseline only from a "measured" record, so a record with no window must stay measured."""
    rec = dict(_fresh(key), members_at_latest=["NSE:A", "NSE:B"])
    state, got = S.record_state({"scans": {key: rec}}, _SCANS[key])
    assert state == "measured" and got["members_at_latest"] == ["NSE:A", "NSE:B"] and got["out_of_sample"] == []


def _unseen_window(key, **kw):
    return dict(start="2026-11-30", end="2026-12-31", days=31, n=120, median_excess_pp=1.4, beat_pct=55, final=True,
                fingerprint=S.recipe_fingerprint(_SCANS[key]), **kw)


@pytest.mark.parametrize("key", _KEYS)
def test_the_first_unseen_window_reaches_both_cards_with_no_design_record(key):
    """The quiet failure this change could have shipped: the cards read an unseen window only beside a design-period record or
    a study, so with both gone the job's first window would have read "Not yet measured" for six months with nobody there."""
    scan = _SCANS[key]
    records = {"scans": {key: dict(_fresh(key), out_of_sample=[_unseen_window(key)])}}
    state, rec = S.record_state(records, scan)
    cards = _cards(scan, state, rec)
    full, summ = _text(cards["full"]), _text(cards["summary"])
    assert "NEVER SAW" in full and "+1.4" in full and "55% of matches beat the market" in full
    assert "ahead in 1 of 1 unseen window: +1.4 pts before costs" in summ
    for t in (full, summ):
        assert "No record yet" not in t and "Not yet measured" not in t


def test_the_card_does_not_call_the_fresh_date_the_day_the_recipe_was_built():
    """The recipes were chosen in October; 5 Nov is the day their record starts."""
    h = S.card_html(_SCANS["playbook"], 7, "measured", {"out_of_sample": [_unseen_window("playbook")]}, True)
    assert "WAS BUILT" not in h and "SINCE 05 NOV 2026" in h


def _kept_only() -> dict:
    """A record file holding only the fields that stay, every scan alike."""
    return {"version": S.RECORD_VERSION, "designed_on": S.DESIGNED_ON, "generated": "2026-11-05", "engine_version": "x",
            "method": "unseen", "members_date": "2026-10-30",
            "scans": {s["key"]: {"fingerprint": S.recipe_fingerprint(s), "windows": [], "out_of_sample": [],
                                 "members_at_latest": ["NSE:A"]} for s in S.SCANS}}


def test_a_file_with_only_the_kept_fields_renders_on_every_surface():
    records = _kept_only()
    assert S.members_label(records) == "30 Oct"
    for scan in S.SCANS:
        state, rec = S.record_state(records, scan)
        assert state == "measured"
        for h in (*_cards(scan, state, rec).values(), G.guide_html(scan, rec, state)):
            assert h and not re.search(r"\bnan\b", _text(h), re.I)
    assert "None has finished yet" in _text(G.reference_html(records)) and "None has finished yet" in G.guide_markdown(records)


# ── 4. Nothing brings an old figure back ──────────────────────────────────────────────────────────────────────────────
_OLD_WINDOW = dict(start="2026-06-17", end="2026-08-22", days=66, median_excess_pp=7.1, beat_pct=70, n=148, dropped_no_return=0,
                   dropped_corp_action=10, net=dict(min_cr=2.0, cost_pp=1.0, n=119, median_excess_pp=6.1, beat_pct=66))
_OLD_STUDY = dict(title="3-year backtest", match_cagr_pct=20.4, rest_cagr_pct=4.4, match_doubled_pct=40.7, rest_doubled_pct=20.8,
                  n_match=182, n_rest=1325, caveats="Survivors only", definition_note="margin leg", **{"from": "Oct 2023", "to": "Oct 2026"})
_LEFT_IN_A_FILE = ("+7.1", "+6.1", "+20.4", "+4.4", "40.7", "20.8", "3-YEAR", "3-year backtest", "Survivors only",
                   "IN-SAMPLE", "DESIGN-PERIOD", "Design period", "in-sample")


def _surfaces(scan, records) -> dict:
    """Every place a scan's record is read, its Reference block and download taken alone (the others' prose has its own words)."""
    state, got = S.record_state(records, scan)
    return {**_cards(scan, state, got), "guide": G.guide_html(scan, got, state),
            "reference": G.reference_html(records, scans=[scan]), "download": G.guide_markdown(records, scans=[scan])}


@pytest.mark.parametrize("key", _SHORT)
def test_a_record_left_in_a_file_reaches_no_surface(key):
    """A local tools/scan_records.py or a hand edit could write these back: no card, guide or download may read them."""
    rec = dict(_fresh(key), windows=[_OLD_WINDOW], study=_OLD_STUDY)
    records = {**_kept_only(), "scans": {**_kept_only()["scans"], key: rec}}
    for name, h in _surfaces(_SCANS[key], records).items():
        t = _text(h)
        for word in _LEFT_IN_A_FILE:
            assert word not in t, (name, word)


def test_a_design_window_left_in_the_compounders_record_reaches_no_surface():
    """🌱's study is read; a design-period window beside it never is."""
    key = "compounder"
    clean = {**_kept_only(), "scans": {**_kept_only()["scans"], key: _fresh(key)}}
    left = {**_kept_only(), "scans": {**_kept_only()["scans"], key: dict(_fresh(key), windows=[_OLD_WINDOW])}}
    assert _surfaces(_SCANS[key], left) == _surfaces(_SCANS[key], clean)


# The headline figures of 🚀's and 🌬️'s old record, those quoted beside it and the words that went with them. 🌱's are back.
_OLD_FIGURES = ("+7.1", "+10.7", "+3.3", "+3.1", "+3.9", "+11.2", "+3.4", "+4.7", "+29%", "38/33/73", "93/63/83", "+2.0 / +0.3",
                "62-70%", "58-67%")


@pytest.mark.parametrize("rel,figures", [("ui/ui_scans.py", _OLD_FIGURES), ("ui/scan_guide.py", _OLD_FIGURES),
                                          ("ui/scan_records_job.py", _OLD_FIGURES),
                                          ("app.py", ("+7.1", "+10.7", "+3.9", "+11.2", "+13.5", "+23.6", "+20.4"))],
                         ids=["ui_scans", "scan_guide", "job", "app"])
def test_no_source_quotes_an_old_figure(rel, figures):
    src = (ROOT / rel).read_text(encoding="utf-8")
    found = [f for f in figures if f in src]
    assert not found, (rel, found)


def test_no_screen_word_quotes_an_old_result():
    """🚀's and 🌬️'s Rank tips, as they read on the Scans tab, each still saying what it is for."""
    words = " ".join(S.table_config("30 Oct", _SCANS[key])["rank"]["help"] for key in _SHORT)
    for said in ("in-sample", "beat the market by", "points", "2026-10-06 test", "measured:"):
        assert said not in words, said
    assert "order check" in words and "Every match is the scan" in words


def test_the_compounders_tips_and_the_left_legend_keep_their_figures():
    """The Rank and Cycle tips and the Left 🌱 legend are 🌱's own, and read as they did before the fresh start."""
    tips = S.table_config("30 Oct", _SCANS["compounder"])
    assert "inside this scan the top of the Rank did not beat the rest in the 2026-10-06 test" in tips["rank"]["help"]
    cycle = tips["_cycle"]["help"]
    assert "beat the market by +24 to +54 points in each of three years" in cycle and "(in-sample)" in cycle
    assert "same year" in cycle and "A year-old improvement did not help" in cycle
    legend = S.left_legend()
    for key, (mark, label, why) in S.EXIT_REASONS.items():
        assert f"- {mark} **{label}** — {why}" in legend and why, key
    assert "measured:" in legend and "8 to 11 points" in legend


_GUIDE_FIRST = {"playbook": ("30 Nov 2026", "25+ days"), "leaders": ("30 Nov 2026", "25+ days")}


def _guide(key) -> str:
    return _text(G.guide_html(_SCANS[key], _fresh(key), "measured"))


@pytest.mark.parametrize("key", _SHORT)
def test_the_guide_says_when_the_record_starts_and_its_first_window(key):
    start, runs = _GUIDE_FIRST[key]
    t = _guide(key)
    assert f"Record starts: 5 Nov 2026 — its first window starts on the {start} copy." in t
    assert f"None has finished yet; the first starts on the {start} copy and runs {runs}." in t
    assert "windows that start on a month-end copy after 5 Nov 2026" in t


def test_the_compounders_guide_says_when_its_recipe_was_fixed_and_when_its_first_window_is_due():
    t = _guide("compounder")
    assert "Recipe fixed on: 7 Oct 2026 — every window that starts later is unseen." in t
    assert "None has finished yet; the first is due around 31 Oct 2027." in t
    assert "windows that start on a month-end copy after 7 Oct 2026" in t
    assert "Record starts" not in t and "copy and runs" not in t


@pytest.mark.parametrize("key", _SHORT)
def test_the_guide_has_no_old_record_left_in_it(key):
    t = _guide(key)
    for word in ("in-sample", "In-sample", "Design period", "design-period", "backtest", "Backtest", "Recipe fixed on",
                 "doubled", "compounded", "study"):
        assert word not in t, word


def test_how_every_scan_is_measured_has_no_old_record_left_in_it():
    t = " ".join(map(str, dict(G.general_sections())["How every scan is measured"]))
    for word in ("design-period", "+4.7", "On June–October 2026 data"):
        assert word not in t, word
    assert "after 5 Nov 2026 (🌱: after 7 Oct 2026) is out of sample" in t and '"still in the list" figures were measured on them' in t
    assert "**In-sample:** the months studied while a recipe was chosen." in t, "the word on 🌱's card"


def test_the_compounders_prose_keeps_its_recipe_and_its_backtest():
    g = G.GUIDES["compounder"]
    recipe, limits = " ".join(g["recipe"]), " ".join(g["limits"])
    assert "In the backtest the edge all but vanished without this condition" in recipe
    assert "part of the recipe, not an option" in recipe
    assert "**The backtest is in-sample.**" in limits and "It did well inside this list in the backtest" in limits
    assert "reading aid" in limits and g["backtest"]["items"]
