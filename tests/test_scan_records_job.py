"""
test_scan_records_job.py — the weekly job that keeps the 🎯 Scans records current (2026-10-06).

The job (ui/scan_records_job.py, run by .github/workflows/scan-records.yml) re-scores the PRISM
Archive's month-end copies with today's engine and records two things per scan: the UNSEEN windows
(starting after the recipes were chosen — the most trustworthy evidence a card can show) and the 🆕
baseline. These tests pin which windows count as unseen, that the job never rewrites the design-period
record or commits when nothing measured changed, that it refuses to mix records of two recipes, and
that the workflow is wired to do exactly this and nothing more.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui import scan_records_job as J  # noqa: E402
from ui import ui_scans as S  # noqa: E402

ARCHIVE = ["2026-06-29", "2026-09-30", "2026-10-31", "2026-11-28", "2026-12-04"]


# ── 1. Which windows are unseen ───────────────────────────────────────────────────────────────────
def test_only_finished_months_can_start_a_window():
    assert J.frozen(ARCHIVE) == ["2026-06-29", "2026-09-30", "2026-10-31", "2026-11-28"], (
        "the latest month's copy is replaced daily until the month ends, so it cannot be a start")


def test_unseen_windows_start_after_the_recipes_were_chosen():
    w = J.unseen_windows(ARCHIVE, designed_on="2026-10-06", min_days=25)
    assert w == [("2026-10-31", "2026-11-28"), ("2026-10-31", "2026-12-04")], w
    assert all(a > "2026-10-06" for a, _ in w), "a window starting before the design date is in-sample"


def test_a_window_shorter_than_the_minimum_is_not_evidence():
    # 11-28 -> 12-04 is 6 days: no window from 11-28 yet.
    assert all(a != "2026-11-28" for a, _ in J.unseen_windows(ARCHIVE, "2026-10-06", 25))


def test_today_there_is_no_unseen_window_yet():
    today = ["2026-06-29", "2026-09-30", "2026-10-05"]
    assert J.unseen_windows(today) == []
    assert J.baseline_date(today) == "2026-09-30", "🆕 compares against the latest finished month"


# ── 2. One measurement ────────────────────────────────────────────────────────────────────────────
def _frames(n=40):
    ids = [f"S{i:02d}" for i in range(n)]
    early = pd.DataFrame({"company_id": ids, "close_price": 100.0, "equity_shares": 10.0,
                          "flag": [1] * 30 + [0] * (n - 30)})
    late = early.copy()
    late["close_price"] = np.r_[np.full(30, 120.0), np.full(n - 30, 100.0)]
    late.loc[0, "equity_shares"] = 50.0                     # a split: excluded, not a -80% or +500%
    return early, late


_SCAN = {"key": "toy", "mask": lambda d: d["flag"] == 1}


def test_measure_window_reads_excess_beat_rate_and_size():
    # 30 flagged stocks rise 20% (one of them splits and must leave), 50 others stay flat: the market
    # median is 0, so every remaining match beats it by 20 points.
    early, late = _frames(80)
    early = pd.concat([early, early.iloc[[1]].assign(company_id="S99", flag=1)], ignore_index=True)
    late = pd.concat([late, late.iloc[[1]].assign(company_id="S99")], ignore_index=True)
    r = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")
    assert r["n"] == 30, "the split stock leaves the window; 29 + S99 remain"
    assert r["days"] == 28 and r["beat_pct"] == 100
    assert r["median_excess_pp"] == pytest.approx(20.0)


def test_a_window_with_too_few_matches_is_not_recorded():
    early, late = _frames(40)
    assert J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28") is None, "29 < MIN_N"


def test_measure_window_counts_what_the_headline_leaves_out():
    """2026-10-07: a match with no price at the end (delisted, suspended, missing) cannot be scored,
    so the headline is the survivors' record — the window says how many fell out, and how many were
    corporate actions, instead of letting them vanish."""
    early, late = _frames(80)
    late = late[~late["company_id"].isin(["S01", "S02"])]     # two matches with no price at the end
    r = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")
    assert r is None, "27 scored matches < MIN_N"
    early = pd.concat([early, early.iloc[[5] * 3].assign(company_id=["X1", "X2", "X3"])], ignore_index=True)
    late = pd.concat([late, late.iloc[[3] * 3].assign(company_id=["X1", "X2", "X3"])], ignore_index=True)
    r = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")
    assert r["n"] == 30 and r["dropped_no_return"] == 2 and r["dropped_corp_action"] == 1


def test_the_net_record_keeps_only_liquid_matches_and_charges_the_round_trip():
    early, late = _frames(80)
    early["vol_sma_20d"] = 100_000.0                          # x ₹100 = ₹1 Cr a day: under the floor
    early.loc[1:30, "vol_sma_20d"] = 1_000_000.0              # ₹10 Cr a day: S01..S29 + S30 (not a match)
    early = pd.concat([early, early.iloc[[1]].assign(company_id="S99")], ignore_index=True)
    late = pd.concat([late, late.iloc[[1]].assign(company_id="S99")], ignore_index=True)
    r = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")
    net = r["net"]
    assert (net["min_cr"], net["cost_pp"]) == (S.NET_MIN_CR, S.NET_COST_PP)
    assert net["n"] == 30, "S01..S29 and S99 clear ₹2 Cr; S00 split out; S30 is not a match"
    assert net["median_excess_pp"] == pytest.approx(r["median_excess_pp"] - S.NET_COST_PP)
    assert net["beat_pct"] == 100


def test_too_few_liquid_matches_give_a_net_count_and_no_reading():
    early, late = _frames(80)
    early = pd.concat([early, early.iloc[[1]].assign(company_id="S99")], ignore_index=True)
    late = pd.concat([late, late.iloc[[1]].assign(company_id="S99")], ignore_index=True)
    r = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")
    assert r["net"]["n"] == 0, "no traded value = unknown = does not clear the floor"
    assert "median_excess_pp" not in r["net"] and "beat_pct" not in r["net"]


# ── 3. Merging never loses the other half, and a quiet week writes nothing ─────────────────────────
def _existing():
    return {"version": S.RECORD_VERSION, "designed_on": S.DESIGNED_ON, "generated": "2026-10-06",
            "engine_version": "abc", "latest_snapshot": "2026-10-03",
            "scans": {"playbook": {"fingerprint": "f", "windows": [{"start": "2026-06-17"}],
                                   "out_of_sample": [], "members_at_latest": ["A"]}}}


def test_the_job_update_keeps_the_design_period_record():
    new, changed = J.merge(_existing(), {"playbook": {"out_of_sample": [{"start": "2026-10-31"}]}}, {})
    assert changed and new["scans"]["playbook"]["windows"] == [{"start": "2026-06-17"}]
    assert new["scans"]["playbook"]["fingerprint"] == "f"


def test_the_local_update_keeps_the_unseen_record():
    ex = _existing()
    ex["scans"]["playbook"]["out_of_sample"] = [{"start": "2026-10-31"}]
    new, _ = J.merge(ex, {"playbook": {"windows": [{"start": "2026-08-22"}], "fingerprint": "g"}}, {})
    assert new["scans"]["playbook"]["out_of_sample"] == [{"start": "2026-10-31"}]


def test_only_timestamps_changing_is_not_a_change():
    _, changed = J.merge(_existing(), {"playbook": {"members_at_latest": ["A"]}},
                         {"generated": "2026-12-07", "engine_version": "def"})
    assert not changed, "a quiet week must not commit"


def test_the_job_refuses_a_stale_design_record():
    stale = {"scans": {s["key"]: {"fingerprint": "0" * 16} for s in S.SCANS}}
    with pytest.raises(SystemExit):
        J.check_fresh(stale)
    J.check_fresh(S.load_records())          # the committed records match today's recipes


# ── 4. The committed file, as the job left it ─────────────────────────────────────────────────────
def test_the_committed_records_carry_the_baseline_and_the_unseen_section():
    rec = json.loads(S.RECORDS_PATH.read_text(encoding="utf-8"))
    assert rec["version"] == S.RECORD_VERSION and rec["designed_on"] == S.DESIGNED_ON
    assert rec["members_date"] <= max(rec["latest_snapshot"], rec["members_date"])
    for scan in S.SCANS:
        r = rec["scans"][scan["key"]]
        assert isinstance(r["out_of_sample"], list)
        for w in r["out_of_sample"]:
            assert w["start"] > S.scan_designed_on(scan) and w["days"] >= S.scan_horizon(scan)
            assert w["n"] >= J.MIN_N
            assert {"net", "dropped_no_return", "dropped_corp_action"} <= set(w), (
                "the job records what each unseen window leaves out")
        if "study" not in r:                   # 🌱's baseline arrives with the job's first run
            assert r["members_at_latest"], f"{scan['key']}: no 🆕 baseline"


# ── 4b. The secret is read however it was pasted ───────────────────────────────────────────────────
_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789_-abcde"


@pytest.mark.parametrize("pasted", [
    _ID,                                                              # the bare id
    f'"{_ID}"', f"'{_ID}'", f"  {_ID}\n",                             # quoted, or with whitespace
    f'ARCHIVE_INDEX_SHEET_ID = "{_ID}"',                              # the whole Streamlit line
    f"https://docs.google.com/spreadsheets/d/{_ID}/edit?gid=0#gid=0",  # the sheet's address
])
def test_the_archive_index_id_is_read_however_it_was_pasted(pasted):
    """Found on the first live run (2026-10-06): the secret reached Google as pasted and came back
    HTTP 400 Bad Request — a Streamlit secret is written with quotes, and pasting the same text into
    GitHub carries them along."""
    assert J.sheet_id_from_secret(pasted) == _ID


def test_a_secret_with_no_sheet_id_in_it_is_rejected():
    for bad in ("", "   ", "ARCHIVE_INDEX_SHEET_ID", "15YwCtTSHSW4"):
        assert J.sheet_id_from_secret(bad) == ""


# ── 5. The workflow does exactly this ─────────────────────────────────────────────────────────────
_WF = (ROOT / ".github" / "workflows" / "scan-records.yml").read_text(encoding="utf-8")


def test_the_workflow_runs_weekly_and_on_demand():
    assert "schedule:" in _WF and "cron:" in _WF and "workflow_dispatch:" in _WF


def test_the_workflow_runs_the_job_with_the_archive_secret():
    assert "python -m ui.scan_records_job" in _WF
    assert "ARCHIVE_INDEX_SHEET_ID: ${{ secrets.ARCHIVE_INDEX_SHEET_ID }}" in _WF
    assert "pip install -r requirements.txt" in _WF, "the job must run the pinned engine"


def test_the_workflow_commits_only_the_records_and_only_when_they_changed():
    assert "permissions:" in _WF and "contents: write" in _WF
    assert 'git diff --quiet -- ui/scan_records.json' in _WF
    adds = [line.strip() for line in _WF.splitlines() if line.strip().startswith("git add")]
    assert adds == ["git add ui/scan_records.json"], adds


# ── 6. The schedule survives GitHub's best-effort queue and a quiet repository (2026-10-07) ────────
def _crons(text):
    import re
    return re.findall(r'cron:\s*"([^"]+)"', text)


def test_the_job_has_a_catch_up_slot_off_the_busy_minutes():
    crons = _crons(_WF)
    assert len(crons) == 2, "a first slot and a catch-up: a dropped run must not cost a week"
    for c in crons:
        minute = int(c.split()[0])
        assert minute not in (0, 15, 30, 45), f"{c}: the busiest minutes are the ones GitHub drops"
        assert c.split()[4] == "1", f"{c}: Mondays"


def test_a_push_made_meanwhile_does_not_lose_the_records():
    assert "git pull --rebase" in _WF and "for attempt in 1 2 3" in _WF


_KA = (ROOT / ".github" / "workflows" / "keepalive.yml").read_text(encoding="utf-8")


def test_the_keepalive_commits_only_after_a_quiet_stretch_and_inside_60_days():
    """GitHub switches off scheduled workflows after 60 days without a commit. The keepalive writes
    only when the repository has been quiet for QUIET days, and runs often enough that the longest gap
    between runs plus QUIET stays under 60."""
    import re
    quiet = int(re.search(r'-lt (\d+) \]', _KA).group(1))
    (cron,) = _crons(_KA)
    days = [int(d) for d in cron.split()[2].split(",")]
    assert int(cron.split()[0]) not in (0, 15, 30, 45)
    longest_gap = max([b - a for a, b in zip(days, days[1:])] + [31 - days[-1] + days[0]])
    assert quiet - 1 + longest_gap < 60, (quiet, longest_gap)
    adds = [line.strip() for line in _KA.splitlines() if line.strip().startswith("git add")]
    assert adds == ["git add .keepalive"], "the keepalive touches its own file and nothing else"
    assert "workflow_dispatch:" in _KA and "contents: write" in _KA


# ── 7. Each scan on its own clock (🌱, 2026-10-07) ───────────────────────────────────────────────────
def test_the_compounder_is_only_ever_recorded_on_a_year_or_more():
    comp = next(s for s in S.SCANS if s["key"] == "compounder")
    play = next(s for s in S.SCANS if s["key"] == "playbook")
    dates = ["2026-09-30", "2026-10-31", "2026-11-30", "2027-04-30", "2027-10-29", "2027-10-31", "2027-11-05"]
    # 2026-09-30 is before the design date; only October starts a 🌱 window, and it runs to the first
    # copy a year later only — not also to the latest copy, which would overlap it.
    assert J.scan_windows(dates, comp) == [("2026-10-31", "2027-10-31")], (
        "one window a year: starts in October after its design date, ends at the first copy 365+ days on")
    assert all(J._days(a, b) >= 365 for a, b in J.scan_windows(dates, comp))
    assert ("2026-10-31", "2026-11-30") in J.scan_windows(dates, play), "the short lists keep their 25 days"
    assert J.scan_windows(dates[:4], comp) == [], "no year has passed yet: nothing to record"


def test_the_compounder_gets_one_window_a_year_never_overlapping():
    """2026-10-07: a window from every month-end copy would show ~12 bars by 2028 that overlap by 11
    months — one year of evidence that looks like twelve. 🌱 starts one window each October."""
    import pandas as _pd
    comp = next(s for s in S.SCANS if s["key"] == "compounder")
    months = [d.strftime("%Y-%m-%d") for d in _pd.date_range("2026-09-30", "2029-01-31", freq="ME")] + ["2029-02-04"]
    w = J.scan_windows(months, comp)
    assert w == [("2026-10-31", "2027-10-31"), ("2027-10-31", "2028-10-31")], w
    for (a, b), (c, _) in zip(w, w[1:]):
        assert b <= c, "a window ends before the next one starts"
    assert all(a[5:7] == "10" for a, _ in w), "every start is an October month-end"
    play = next(s for s in S.SCANS if s["key"] == "playbook")
    assert len(J.scan_windows(months, play)) > 20, "the short lists keep a window from every month-end"


def test_a_missing_october_copy_costs_that_year_not_a_wrong_window():
    comp = next(s for s in S.SCANS if s["key"] == "compounder")
    dates = ["2026-10-31", "2026-11-30", "2027-09-30", "2027-11-30", "2027-12-31", "2028-11-30", "2028-12-05"]
    assert J.scan_windows(dates, comp) == [("2026-10-31", "2027-11-30")], (
        "no October 2027 copy: no 2027 window, and no window starting in another month instead")
