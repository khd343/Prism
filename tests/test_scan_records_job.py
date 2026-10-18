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
from datetime import date
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


def test_a_fall_the_share_count_does_not_explain_is_named_for_a_hand_check():
    """The scan-records audit (2026-10-18): a demerger lowers the price while the share count stays, so
    forward_returns cannot flag it and the window books the holder's new shares as a loss — NSE's own list showed
    HEG -66.2% (7 Sep) and Triveni -33.6% (22 Jul), neither flagged, and nothing in PRISM's data tells one from a
    crash. So each window names its matches that fell S.FALL_CHECK_PCT or more (as shown, to one decimal) with the
    share count unmoved, worst first, to be checked against NSE's list. Not named: a split (already left out), a
    fall short of the bar, a stock outside the scan."""
    early, late = _frames(80)
    early = pd.concat([early, early.iloc[[1]].assign(company_id="S99")], ignore_index=True)
    late = pd.concat([late, late.iloc[[1]].assign(company_id="S99")], ignore_index=True)
    price = dict(S01=33.8, S02=70.0, S03=70.1, S40=40.0, S00=20.0)   # S00 splits x5: re-based, left out
    late["close_price"] = late["company_id"].map(price).fillna(late["close_price"])
    r = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")
    assert S.FALL_CHECK_PCT == 30.0
    assert r["falls"] == [["S01", -66.2], ["S02", -30.0]], "S03 fell 29.9%; S40 is not a match; S00 split"
    assert r["dropped_corp_action"] == 1 and r["n"] == 30, "named, never removed: the reading is unchanged"


def test_a_window_with_no_such_fall_names_none():
    early, late = _frames(80)
    early = pd.concat([early, early.iloc[[1]].assign(company_id="S99")], ignore_index=True)
    late = pd.concat([late, late.iloc[[1]].assign(company_id="S99")], ignore_index=True)
    assert J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")["falls"] == []


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


# The first Node-24 major of each action, read from each release's action.yml on 2026-10-08
# (checkout v4 and setup-python v5 still declared node20, which GitHub is retiring).
_NODE24_FROM = {"actions/checkout": 5, "actions/setup-python": 6}


# EVERY workflow in the folder (2026-10-10, when the daily data check joined): a hand-kept list let a
# new workflow skip the rule below without anything failing.
_ALL_WF = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


def test_the_workflow_list_is_read_from_the_folder():
    names = {p.name for p in _ALL_WF}
    assert {"scan-records.yml", "keepalive.yml", "data-check.yml"} <= names, names


@pytest.mark.parametrize("name,text", [(p.name, p.read_text(encoding="utf-8")) for p in _ALL_WF],
                         ids=[p.name for p in _ALL_WF])
def test_the_workflows_run_on_a_pinned_runner_and_current_actions(name, text):
    """A job that writes the out-of-sample record must not change under us (2026-10-08): the runner is
    pinned rather than ubuntu-latest (moved to Ubuntu 26 from 2026-10-19), and every action is on a
    release that runs on Node 24."""
    import re
    runners = re.findall(r"runs-on:\s*(\S+)", text)
    assert runners and all(re.fullmatch(r"ubuntu-\d\d\.\d\d", r) for r in runners), (name, runners)
    uses = re.findall(r"uses:\s*([\w./-]+)@v(\d+)", text)
    assert uses, f"{name}: no actions found — the pattern no longer reads this file"
    for action, major in uses:
        assert action in _NODE24_FROM, f"{name}: {action} is not checked for its Node runtime"
        assert int(major) >= _NODE24_FROM[action], f"{name}: {action}@v{major} runs on Node 20"


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


# ── 8. The out-of-sample record cannot see the future, and its arithmetic is checked twice (2026-10-07)
# Ideas from khd343/Stage's VALIDATION_PROTOCOL: §7 "prove future data cannot affect a past signal",
# §4 "check a critical calculation against an independent implementation".
def _market(n=300, seed=11):
    """A start snapshot every real recipe can match in (≥ MIN_N each): momentum and breakout move together,
    every quarter is strong, every stock is above its stop; profits grow faster than sales; 10 sectors."""
    rng = np.random.default_rng(seed)
    i = np.arange(n, dtype=float)
    return pd.DataFrame({
        "company_id": [f"NSE:F{k:03d}" for k in range(n)], "name": [f"F{k}" for k in range(n)],
        "sector": [f"Sec{k % 10}" for k in range(n)], "industry": [f"Ind{k % 20}" for k in range(n)],
        "rank": rng.permutation(n) + 1,
        "momentum_score": i, "breakout_score": i, "improvement_score": rng.uniform(0, 100, n),
        "dist_to_vstop": rng.uniform(1, 9, n), "dist_52wh": rng.uniform(0, 30, n),
        "rev_pyq": 100.0, "rev_lq": 130.0, "pat_pyq": 10.0, "pat_lq": 15.0, "ebitda_pyq": 15.0, "ebitda_lq": 26.0,
        "pat_1yb": 100.0, "pat": 100.0 * (1 + rng.uniform(0.0, 1.0, n)),
        "opm_1yb": 10.0, "opm": 10.0 + rng.uniform(-3, 3, n),
        "revenue_1yb": 100.0, "revenue": 112.0,
        "close_price": rng.uniform(50, 500, n), "equity_shares": 1e7,
        "vol_sma_20d": rng.uniform(1e4, 2e6, n),
    })


def _later(early, seed=12):
    late = early.copy()
    late["close_price"] = early["close_price"] * np.random.default_rng(seed).uniform(0.6, 1.6, len(early))
    return late


def _scrambled(late, seed=13):
    """Every column of the END snapshot permuted across stocks — except the identity, the price and the
    share count that the forward return legitimately reads — plus a column nothing should read."""
    rng = np.random.default_rng(seed)
    out = late.copy()
    for c in out.columns:
        if c not in ("company_id", "close_price", "equity_shares"):
            out[c] = rng.permutation(out[c].to_numpy())
    out["future_only_column"] = rng.uniform(size=len(out))
    return out


@pytest.mark.parametrize("key", [s["key"] for s in S.SCANS])
def test_a_scans_record_cannot_see_the_end_snapshot(key):
    """Scrambling everything the END snapshot holds — scores, fundamentals, sectors, traded value — must
    not change one number of the record: who is measured comes from the start snapshot alone."""
    scan = next(s for s in S.SCANS if s["key"] == key)
    early = _market(); late = _later(early)
    a = J.measure_window(early, late, scan, "2026-10-31", "2027-10-31")
    b = J.measure_window(early, _scrambled(late), scan, "2026-10-31", "2027-10-31")
    assert a is not None, f"{key}: the fixture must produce a measurable window"
    assert a == b, f"{key}: the record changed when only end-snapshot non-price data changed"


@pytest.mark.parametrize("key", [s["key"] for s in S.SCANS])
def test_the_recipe_and_the_liquidity_floor_only_ever_read_the_start_snapshot(key, monkeypatch):
    scan = dict(next(s for s in S.SCANS if s["key"] == key))
    early = _market(); late = _later(early)
    seen = []
    real_mask, real_tv = scan["mask"], S.traded_value_cr
    scan["mask"] = lambda df: (seen.append(("recipe", df is early)), real_mask(df))[1]
    monkeypatch.setattr(S, "traded_value_cr", lambda df: (seen.append(("liquidity", df is early)), real_tv(df))[1])
    J.measure_window(early, late, scan, "2026-10-31", "2027-10-31")
    assert seen and all(is_early for _, is_early in seen), seen


def test_end_prices_move_the_returns_never_the_membership():
    scan = next(s for s in S.SCANS if s["key"] == "compounder")
    early = _market(); late = _later(early)
    a = J.measure_window(early, late, scan, "2026-10-31", "2027-10-31")
    b = J.measure_window(early, _later(early, seed=99), scan, "2026-10-31", "2027-10-31")
    assert (a["n"], a["dropped_no_return"], a["dropped_corp_action"]) == (b["n"], b["dropped_no_return"], b["dropped_corp_action"])
    assert a["median_excess_pp"] != b["median_excess_pp"], "the fixture must actually move the returns"


def _reference_window(early, late, flagged):
    """measure_window's numbers recomputed by a plain loop, from the written definitions — not from
    measure_window or forward_returns: a return needs a positive price at both ends; a share count that
    moved by more than SHARE_TOL is a corporate action (no return); excess = return minus the median
    return of EVERY stock with one; the net reading keeps matches trading NET_MIN_CR a day or more,
    each charged NET_COST_PP."""
    import statistics
    from core.forward_returns import SHARE_TOL
    end = {r.company_id: r for r in late.itertuples()}
    ret, corp = {}, set()
    for r in early.itertuples():
        e = end.get(r.company_id)
        if e is None or not (r.close_price > 0 and e.close_price > 0):
            continue
        if r.equity_shares > 0 and abs(e.equity_shares / r.equity_shares - 1.0) > SHARE_TOL:
            corp.add(r.company_id); continue
        ret[r.company_id] = (e.close_price / r.close_price - 1.0) * 100
    med = statistics.median(ret.values())
    matches = [cid for cid in early["company_id"] if cid in flagged and cid in ret]
    exc = [ret[cid] - med for cid in matches]
    tv = dict(zip(early["company_id"], early["vol_sma_20d"] * early["close_price"] / 1e7))
    liquid = [ret[cid] - med - S.NET_COST_PP for cid in matches if tv[cid] >= S.NET_MIN_CR]
    net = dict(min_cr=S.NET_MIN_CR, cost_pp=S.NET_COST_PP, n=len(liquid))
    if len(liquid) >= J.MIN_N:
        net.update(median_excess_pp=round(statistics.median(liquid), 1),
                   beat_pct=int(round(100 * sum(x > 0 for x in liquid) / len(liquid))))
    return dict(median_excess_pp=round(statistics.median(exc), 1),
                beat_pct=int(round(100 * sum(x > 0 for x in exc) / len(exc))), n=len(matches),
                dropped_no_return=sum(1 for cid in flagged if cid not in ret and cid not in corp),
                dropped_corp_action=sum(1 for cid in flagged if cid in corp), net=net)


def _random_market(seed):
    """A random market with every awkward case: matches that vanish (no end row), a zero or missing end
    price, share counts that move (splits, issues) or wobble inside the tolerance, illiquid matches."""
    rng = np.random.default_rng(seed)
    n = 400
    early = pd.DataFrame({"company_id": [f"S{k:03d}" for k in range(n)],
                          "close_price": rng.uniform(10, 900, n), "equity_shares": rng.uniform(1e6, 1e8, n),
                          "vol_sma_20d": rng.uniform(1e3, 3e6, n), "flag": rng.random(n) < 0.4})
    late = early.copy()
    late["close_price"] = early["close_price"] * rng.uniform(0.3, 2.5, n)
    late.loc[rng.random(n) < 0.05, "close_price"] = 0.0                                  # no usable end price
    late.loc[rng.random(n) < 0.06, "equity_shares"] *= rng.choice([2.0, 10.0, 1.3])      # corporate actions
    late.loc[rng.random(n) < 0.06, "equity_shares"] *= 1.01                              # wobble, inside tolerance
    late = late.drop(index=late.index[rng.random(n) < 0.05])                               # vanished at the end
    return early, late


_RANDOM_SCAN = {"key": "toy", "mask": lambda d: d["flag"].astype(bool)}


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_measure_window_agrees_with_an_independent_recalculation(seed):
    """Random markets with every awkward case (see _random_market)."""
    early, late = _random_market(seed)
    scan = _RANDOM_SCAN
    got = J.measure_window(early, late, scan, "2026-10-31", "2027-10-31")
    want = _reference_window(early, late, set(early.loc[early["flag"], "company_id"]))
    assert {k: got[k] for k in want} == want


# ── 7. Against the index fund you could hold instead (2026-10-08, December plan rule 23) ──────────────
# Written before any result: each scan's liquid matches as ONE equal-weight portfolio, each window
# charged the NET_COST_PP round trip, against every declared index's plain price change less
# S.FUND_FEE_PCT_YR pro-rated to the window. The index rows reach the job beside each archive copy
# (core.load_market: one download), never as stocks.
from config import BENCHMARK_INDICES  # noqa: E402


def _bench(prices: dict) -> pd.DataFrame:
    return pd.DataFrame({"company_id": list(prices), "close_price": list(prices.values())})


def test_an_index_return_is_its_price_change_less_the_fund_fee_for_the_window():
    early = _bench({"NSE:CNX500": 100.0, "NSE:NIF500MO50": 200.0})
    late = _bench({"NSE:CNX500": 110.0, "NSE:NIF500MO50": 190.0})
    assert J.index_returns(early, late, 365) == {
        "NSE:CNX500": pytest.approx(10.0 - S.FUND_FEE_PCT_YR, abs=0.006),
        "NSE:NIF500MO50": pytest.approx(-5.0 - S.FUND_FEE_PCT_YR, abs=0.006)}
    assert J.index_returns(early, late, 28)["NSE:CNX500"] == pytest.approx(
        10.0 - S.FUND_FEE_PCT_YR * 28 / 365, abs=0.006), "the fee is charged for the window's days, not a year"


def test_an_index_missing_or_unpriced_at_either_end_is_left_out():
    early = _bench({"NSE:CNX500": 100.0, "NSE:SMALLCA250": 0.0, "NSE:NFMICRO250": 50.0,
                    "NSE:NOTANINDEX": 10.0})
    late = _bench({"NSE:CNX500": 101.0, "NSE:SMALLCA250": 90.0, "NSE:NOTANINDEX": 12.0})
    assert list(J.index_returns(early, late, 30)) == ["NSE:CNX500"], (
        "no price at the start, no row at the end, or not a declared index: no reading")
    assert J.index_returns(_bench({}), late, 30) == {} and J.index_returns(None, None, 30) == {}


def test_the_indices_are_recorded_in_the_declared_order():
    early = _bench({k: 100.0 for k in reversed(list(BENCHMARK_INDICES))})
    late = _bench({k: 103.0 for k in BENCHMARK_INDICES})
    assert list(J.index_returns(early, late, 30)) == list(BENCHMARK_INDICES)


def _liquid_frames():
    """30 liquid matches (S01..S29 and S99; S00 splits out), one of them up 300% and the rest up 20%:
    an equal-weight portfolio earns the MEAN, which a median-based record would hide."""
    early, late = _frames(80)
    early["vol_sma_20d"] = 0.0
    early.loc[early["flag"] == 1, "vol_sma_20d"] = 1_000_000.0          # x ₹100 = ₹10 Cr a day
    late.loc[late["company_id"] == "S05", "close_price"] = 400.0
    early = pd.concat([early, early.iloc[[1]].assign(company_id="S99")], ignore_index=True)
    late = pd.concat([late, late.iloc[[1]].assign(company_id="S99")], ignore_index=True)
    return early, late


_MEAN = (29 * 20.0 + 300.0) / 30          # the liquid matches' equal-weight return, %


def test_the_portfolio_is_the_equal_weight_mean_of_liquid_matches_after_the_round_trip():
    early, late = _liquid_frames()
    r = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28",
                         _bench({"NSE:CNX500": 100.0}), _bench({"NSE:CNX500": 102.0}))
    vs = r["vs_index"]
    assert (vs["fee_pct_yr"], vs["cost_pp"], vs["min_cr"]) == (S.FUND_FEE_PCT_YR, S.NET_COST_PP, S.NET_MIN_CR)
    assert vs["n"] == r["net"]["n"] == 30, "the same liquid matches as the net record"
    assert vs["portfolio_pp"] == pytest.approx(_MEAN - S.NET_COST_PP, abs=0.006), (
        "an equal-weight portfolio earns the mean (28.3 here) — the median would read 19.0")
    assert vs["indices"] == {"NSE:CNX500": pytest.approx(2.0 - S.FUND_FEE_PCT_YR * 28 / 365, abs=0.006)}


def test_an_illiquid_match_stays_out_of_the_portfolio():
    early, late = _liquid_frames()
    early = pd.concat([early, early.iloc[[2]].assign(company_id="S98", vol_sma_20d=1_000.0)], ignore_index=True)
    late = pd.concat([late, late.iloc[[2]].assign(company_id="S98", close_price=1_000.0)], ignore_index=True)
    vs = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28",
                          _bench({"NSE:CNX500": 100.0}), _bench({"NSE:CNX500": 102.0}))["vs_index"]
    assert vs["n"] == 30 and vs["portfolio_pp"] == pytest.approx(_MEAN - S.NET_COST_PP, abs=0.006), (
        "a ₹0.01 Cr-a-day match up 900% cannot be held in size: it is not in the portfolio")


def test_too_few_liquid_matches_carry_the_index_readings_but_no_portfolio():
    early, late = _frames(80)
    early = pd.concat([early, early.iloc[[1]].assign(company_id="S99")], ignore_index=True)
    late = pd.concat([late, late.iloc[[1]].assign(company_id="S99")], ignore_index=True)
    vs = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28",
                          _bench({"NSE:CNX500": 100.0}), _bench({"NSE:CNX500": 99.0}))["vs_index"]
    assert vs["n"] == 0 and "portfolio_pp" not in vs and list(vs["indices"]) == ["NSE:CNX500"]


def test_without_index_readings_at_both_ends_there_is_no_comparison():
    early, late = _liquid_frames()
    for be, bl in ((None, None), (_bench({}), _bench({})), (_bench({"NSE:CNX500": 100.0}), _bench({}))):
        r = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28", be, bl)
        assert "vs_index" not in r, "a window with no index reading at both ends compares with nothing"
    assert "vs_index" not in J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_index_comparison_agrees_with_an_independent_recalculation(seed):
    """The awkward random market again, with every declared index at random levels: the portfolio is the
    plain mean of the liquid matches' returns (same corporate-action rule as the headline) less the
    round trip, and each index its price change less the fee for the window's days."""
    from core.forward_returns import SHARE_TOL
    early, late = _random_market(seed)
    rng = np.random.default_rng(seed + 100)
    codes = list(BENCHMARK_INDICES)
    be = _bench(dict(zip(codes, rng.uniform(100, 30_000, len(codes)))))
    bl = be.assign(close_price=be["close_price"] * rng.uniform(0.7, 1.4, len(codes)))
    got = J.measure_window(early, late, _RANDOM_SCAN, "2026-10-31", "2027-02-14", be, bl)["vs_index"]
    end = {r.company_id: r for r in late.itertuples()}
    rets = []
    for r in early.itertuples():
        e = end.get(r.company_id)
        if not r.flag or e is None or not (r.close_price > 0 and e.close_price > 0):
            continue
        if r.equity_shares > 0 and abs(e.equity_shares / r.equity_shares - 1.0) > SHARE_TOL:
            continue
        if r.vol_sma_20d * r.close_price / 1e7 >= S.NET_MIN_CR:
            rets.append((e.close_price / r.close_price - 1.0) * 100)
    assert got["n"] == len(rets) >= J.MIN_N, "the fixture must give a measurable portfolio"
    assert got["portfolio_pp"] == pytest.approx(sum(rets) / len(rets) - S.NET_COST_PP, abs=0.0051)
    fee = S.FUND_FEE_PCT_YR * 106 / 365
    want = {c: (b / a - 1.0) * 100 - fee for c, a, b in zip(codes, be["close_price"], bl["close_price"])}
    assert got["indices"] == {c: pytest.approx(v, abs=0.0051) for c, v in want.items()}


# ── A finished window is written once (2026-10-12) ────────────────────────────────────────────────
# The job re-scores every copy with TODAY's engine. The recipe fingerprint guards the recipe, not the engine
# under it (the Playbook reads momentum/breakout scores the engine computes), so without this a later scoring
# change would silently rewrite October's unseen record. A window whose END copy is a finished month is FINAL:
# recorded once with the engine that measured it, never re-measured, never dropped.
DATES = ["2026-09-30", "2026-10-31", "2026-11-28", "2026-12-04"]


def test_a_final_window_is_kept_exactly_as_first_recorded():
    old = [{"start": "2026-10-31", "end": "2026-11-28", "median_excess_pp": 5.0, "final": True, "engine": "aaa"}]
    fresh = [{"start": "2026-10-31", "end": "2026-11-28", "median_excess_pp": 9.9},
             {"start": "2026-10-31", "end": "2026-12-04", "median_excess_pp": 1.0}]
    out = J.keep_final(old, fresh, DATES, "bbb")
    assert out == [old[0], {"start": "2026-10-31", "end": "2026-12-04", "median_excess_pp": 1.0, "final": False}]


def test_a_window_becomes_final_when_its_end_month_is_over_and_is_stamped_then():
    fresh = [{"start": "2026-10-31", "end": "2026-11-28", "median_excess_pp": 2.0}]
    assert J.keep_final([], fresh, DATES, "e1") == [{**fresh[0], "final": True, "engine": "e1"}]   # Nov is over
    assert J.keep_final([], fresh, DATES[:3], "e1") == [{**fresh[0], "final": False}]   # 28 Nov is the latest


def test_a_running_window_is_measured_again_every_week_and_never_stamped():
    """A stamp alone must not count as a change: an unstamped running window keeps quiet weeks commit-free."""
    old = [{"start": "2026-10-31", "end": "2026-12-04", "median_excess_pp": 1.0, "final": False}]
    fresh = [{"start": "2026-10-31", "end": "2026-12-04", "median_excess_pp": 3.0}]
    assert J.keep_final(old, fresh, DATES, "bbb") == [{**fresh[0], "final": False}]


def test_a_final_window_is_never_dropped():
    """A copy later missing from the archive must not erase evidence already recorded on it."""
    old = [{"start": "2026-10-31", "end": "2026-11-28", "median_excess_pp": 5.0, "final": True, "engine": "aaa"}]
    assert J.keep_final(old, [], DATES) == old


@pytest.mark.parametrize("on_file", [True, False], ids=["final-on-file", "first-run"])
def test_the_job_never_remeasures_a_final_window(monkeypatch, on_file):
    """main() end to end: a final window already on file survives a new engine untouched and is not measured
    again; on a first run it is measured once and stamped; the running window is measured, never stamped."""
    import ui.ui_movers as M
    early, late = _liquid_frames()
    stocks = {"2026-09-30": early, "2026-10-31": early, "2026-11-28": late, "2026-12-04": late}
    monkeypatch.setattr(J, "load_market", lambda source, files, sid: (stocks[sid.removeprefix("id-")].copy(),
                                                                       _bench({"NSE:NIF500MO50": 100.0})))
    monkeypatch.setattr(J, "run_scoring_pipeline", lambda df: df)
    monkeypatch.setattr(J, "engine_version", lambda: "new-engine")
    monkeypatch.setattr(M, "load_index", lambda _id: pd.DataFrame(
        {"vintage_date": DATES, "spreadsheet_id": [f"id-{d}" for d in DATES]}))
    toy = {"key": "toy", "emoji": "T", "name": "Toy", "mask": lambda d: d["flag"] == 1, "benchmark": "NSE:NIF500MO50"}
    monkeypatch.setattr(S, "SCANS", (toy,))
    monkeypatch.setattr(S, "recipe_fingerprint", lambda scan: "fp")
    monkeypatch.setattr(J, "_today", lambda: date(2026, 12, 7))           # the archive is current on that day
    first = {"start": "2026-10-31", "end": "2026-11-28", "days": 28, "median_excess_pp": 42.0, "beat_pct": 61,
             "n": 40, "fingerprint": "fp", "final": True, "engine": "old"}   # as the job writes it
    monkeypatch.setattr(S, "load_records", lambda: {"version": S.RECORD_VERSION, "scans": {
        "toy": {"fingerprint": "fp", "windows": [], "out_of_sample": [first] if on_file else []}}})
    measured = []
    real = J.measure_window
    monkeypatch.setattr(J, "measure_window", lambda *a, **k: measured.append((a[3], a[4])) or real(*a, **k))
    written = {}
    monkeypatch.setattr(J, "write", lambda rec: written.update(rec))
    monkeypatch.setenv("ARCHIVE_INDEX_SHEET_ID", "x" * 44)
    assert J.main() == 0
    oos = written["scans"]["toy"]["out_of_sample"]
    if on_file:
        assert oos[0] == first and measured == [("2026-10-31", "2026-12-04")]
    else:
        assert measured == [("2026-10-31", "2026-11-28"), ("2026-10-31", "2026-12-04")]
        assert (oos[0]["end"], oos[0]["final"], oos[0]["engine"]) == ("2026-11-28", True, "new-engine")
    assert (oos[1]["end"], oos[1]["final"], "engine" in oos[1]) == ("2026-12-04", False, False)


def test_the_job_reads_the_indices_beside_each_copy_and_records_the_comparison(monkeypatch):
    """main() end to end on fake archive copies: each copy is ONE load (stocks and index rows), only the
    stocks are scored, and every unseen window carries the comparison read from ITS OWN two copies."""
    import ui.ui_movers as M
    dates = ["2026-09-30", "2026-10-31", "2026-11-28", "2026-12-04"]
    early, late = _liquid_frames()
    stocks = {"2026-09-30": early, "2026-10-31": early, "2026-11-28": late, "2026-12-04": late}
    level = {"2026-09-30": 90.0, "2026-10-31": 100.0, "2026-11-28": 104.0, "2026-12-04": 108.0}
    loads = []

    def fake_load(source, files, sheet_id):
        d = sheet_id.removeprefix("id-")
        loads.append(d)
        return stocks[d].copy(), _bench({"NSE:NIF500MO50": level[d]})

    monkeypatch.setattr(J, "load_market", fake_load)
    monkeypatch.setattr(J, "run_scoring_pipeline", lambda df: df)
    monkeypatch.setattr(M, "load_index", lambda _id: pd.DataFrame(
        {"vintage_date": dates, "spreadsheet_id": [f"id-{d}" for d in dates]}))
    toy = {"key": "toy", "emoji": "T", "name": "Toy", "mask": lambda d: d["flag"] == 1,
           "benchmark": "NSE:NIF500MO50"}
    monkeypatch.setattr(S, "SCANS", (toy,))
    monkeypatch.setattr(S, "recipe_fingerprint", lambda scan: "fp")
    monkeypatch.setattr(J, "_today", lambda: date(2026, 12, 7))           # the archive is current on that day
    monkeypatch.setattr(S, "load_records", lambda: {"version": S.RECORD_VERSION,
                                                    "scans": {"toy": {"fingerprint": "fp", "windows": []}}})
    written = {}
    monkeypatch.setattr(J, "write", lambda rec: written.update(rec))
    monkeypatch.setenv("ARCHIVE_INDEX_SHEET_ID", "x" * 44)
    assert J.main() == 0
    assert len(loads) == len(set(loads)), f"a copy was downloaded twice: {loads}"
    oos = written["scans"]["toy"]["out_of_sample"]
    assert [(w["start"], w["end"]) for w in oos] == [("2026-10-31", "2026-11-28"), ("2026-10-31", "2026-12-04")]
    for w, lvl in zip(oos, (104.0, 108.0)):
        assert w["vs_index"]["indices"]["NSE:NIF500MO50"] == pytest.approx(
            (lvl / 100.0 - 1.0) * 100 - S.FUND_FEE_PCT_YR * w["days"] / 365, abs=0.006)
        assert w["vs_index"]["portfolio_pp"] == pytest.approx(_MEAN - S.NET_COST_PP, abs=0.006)


# ── 8. Every unseen window names the recipe it measured; a stalled archive is loud (2026-10-15) ────────
# The audit found that a recipe change would have kept the old recipe's FINAL windows (keep_final, by design)
# with nothing saying which recipe produced them, so a card would have counted two recipes as one record.
# Each window now carries the recipe fingerprint it was measured with, "final" is final for THAT recipe, and
# the cards read the current recipe's windows only (ui_scans.record_state). A new recipe's own unseen record
# may only start after the windows it replaces — its designer has seen them. And an archive that stops taking
# copies used to leave the job green every Monday while the record silently stopped growing.
def _job_on_fake_archive(monkeypatch, records, today=date(2026, 12, 7), fp="fp", designed_on="2026-10-06",
                         frames=None):
    """main() on a fake archive (DATES) with one toy scan; returns (written, windows measured). `frames` replaces
    the (month-end, later) copies."""
    import ui.ui_movers as M
    early, late = frames or _liquid_frames()
    stocks = {"2026-09-30": early, "2026-10-31": early, "2026-11-28": late, "2026-12-04": late}
    monkeypatch.setattr(J, "load_market", lambda source, files, sid: (stocks[sid.removeprefix("id-")].copy(),
                                                                       _bench({"NSE:NIF500MO50": 100.0})))
    monkeypatch.setattr(J, "run_scoring_pipeline", lambda df: df)
    monkeypatch.setattr(J, "engine_version", lambda: "new-engine")
    monkeypatch.setattr(J, "_today", lambda: today)
    monkeypatch.setattr(M, "load_index", lambda _id: pd.DataFrame(
        {"vintage_date": DATES, "spreadsheet_id": [f"id-{d}" for d in DATES]}))
    toy = {"key": "toy", "emoji": "T", "name": "Toy", "mask": lambda d: d["flag"] == 1,
           "benchmark": "NSE:NIF500MO50", "designed_on": designed_on}
    monkeypatch.setattr(S, "SCANS", (toy,))
    monkeypatch.setattr(S, "recipe_fingerprint", lambda scan: fp)
    monkeypatch.setattr(S, "load_records", lambda: records)
    measured, written = [], {}
    real = J.measure_window
    monkeypatch.setattr(J, "measure_window", lambda *a, **k: measured.append((a[3], a[4])) or real(*a, **k))
    monkeypatch.setattr(J, "write", lambda rec: written.update(rec))
    monkeypatch.setenv("ARCHIVE_INDEX_SHEET_ID", "x" * 44)
    J.main()
    return written, measured


def _records(oos, fp="fp"):
    return {"version": S.RECORD_VERSION, "scans": {"toy": {"fingerprint": fp, "windows": [], "out_of_sample": oos}}}


_OLD_FINAL = {"start": "2026-10-31", "end": "2026-11-28", "days": 28, "median_excess_pp": 5.0, "beat_pct": 60,
              "n": 40, "fingerprint": "old", "final": True, "engine": "aaa"}


def test_every_window_the_job_writes_names_the_recipe_it_measured(monkeypatch):
    written, _ = _job_on_fake_archive(monkeypatch, _records([]))
    oos = written["scans"]["toy"]["out_of_sample"]
    assert oos and all(w["fingerprint"] == "fp" for w in oos), oos


def test_a_final_window_is_final_for_its_own_recipe_only():
    fresh = [{"start": "2026-10-31", "end": "2026-11-28", "median_excess_pp": 9.9, "fingerprint": "new"}]
    out = J.keep_final([_OLD_FINAL], fresh, DATES, "bbb")
    assert _OLD_FINAL in out, "the old recipe's evidence is never dropped or rewritten"
    assert {**fresh[0], "final": True, "engine": "bbb"} in out, "the new recipe gets its own reading"


def test_a_changed_recipe_must_be_dated_after_the_windows_it_replaces(monkeypatch):
    """Its designer has seen those windows: a design date before their end would let the new recipe call
    data it was fitted on 'unseen'. The job stops (GitHub emails the owner) until the date moves."""
    # The design record was already re-measured for the new recipe (fp "new"); the old unseen window remains.
    with pytest.raises(SystemExit, match="design date"):
        _job_on_fake_archive(monkeypatch, _records([_OLD_FINAL], fp="new"), fp="new")


def test_a_changed_recipe_dated_after_them_keeps_the_old_windows_and_waits_for_its_own(monkeypatch):
    written, measured = _job_on_fake_archive(monkeypatch, _records([_OLD_FINAL], fp="new"), fp="new",
                                             designed_on="2026-11-30")
    assert measured == [], "no month-end copy after 30 Nov is finished yet, so nothing is unseen for the new recipe"
    assert written["scans"]["toy"]["out_of_sample"] == [_OLD_FINAL], "the old recipe's window stays on file"


def test_the_archive_is_current_up_to_ten_days_and_stalled_after():
    assert J.archive_problem(["2026-11-28", "2026-12-04"], date(2026, 12, 14)) == "", "10 days old is current"
    msg = J.archive_problem(["2026-11-28", "2026-12-04"], date(2026, 12, 15))
    assert "2026-12-04" in msg and "11 days" in msg and "stopped" in msg
    assert J.archive_problem([], date(2026, 12, 7)), "an index with no usable copy is a problem too"


def test_the_job_stops_loudly_when_the_archive_stalled(monkeypatch):
    with pytest.raises(SystemExit, match="stopped"):
        _job_on_fake_archive(monkeypatch, _records([]), today=date(2026, 12, 20))


# ── A window the data cannot measure stops the run (2026-10-18, the scan-records audit) ───────────────────────
# Before, measure_window's None (fewer than MIN_N matches with a return) dropped the window and the job stayed green:
# a renamed column or a half-saved copy would have stopped a scan's record with nobody told. A recipe that matched
# too few stocks in a real month (a crash can shrink 🚀) is no evidence and stays unrecorded, as designed — no alarm.
def test_a_recipe_matching_nothing_or_most_matches_unpriced_points_at_the_data():
    early, late = _frames(80)
    nothing = early.assign(flag=0)
    assert "matches no stock on the 2026-10-31 copy" in J.unmeasured_problem(nothing, late, _SCAN, "2026-10-31",
                                                                             "2026-11-28")
    half_gone = late[~late["company_id"].isin([f"S{i:02d}" for i in range(16)])]      # 14 of 30 matches priced
    msg = J.unmeasured_problem(early, half_gone, _SCAN, "2026-10-31", "2026-11-28")
    assert "only 14 of its 30 matches" in msg and "2026-11-28" in msg
    assert J.unmeasured_problem(early, late[~late["company_id"].isin(["S01"])], _SCAN, "2026-10-31", "2026-11-28") == "", (
        "29 of 30 priced: too few for evidence, and the data is fine")
    few = early.assign(flag=[1] * 20 + [0] * 60)
    assert J.unmeasured_problem(few, late, _SCAN, "2026-10-31", "2026-11-28") == "", "a real month with 20 matches"


def test_a_copy_the_recipe_finds_nothing_on_stops_the_run(monkeypatch):
    early, late = _liquid_frames()
    with pytest.raises(SystemExit, match="toy: its recipe matches no stock on the 2026-10-31 copy"):
        _job_on_fake_archive(monkeypatch, _records([]), frames=(early.assign(flag=0), late))


def test_a_copy_that_lost_most_prices_stops_the_run(monkeypatch):
    early, late = _liquid_frames()
    gone = late[~late["company_id"].isin([f"S{i:02d}" for i in range(20)])]
    with pytest.raises(SystemExit, match="toy: only 11 of its 31 matches"):
        _job_on_fake_archive(monkeypatch, _records([]), frames=(early, gone))


def test_a_baseline_copy_the_recipe_finds_nothing_on_stops_the_run(monkeypatch):
    early, late = _liquid_frames()
    with pytest.raises(SystemExit, match="toy: its recipe matches no stock on the 2026-11-28 copy"):
        _job_on_fake_archive(monkeypatch, _records([]), frames=(early, late.assign(flag=0)))


def test_a_month_with_too_few_matches_stays_unrecorded_with_no_alarm(monkeypatch):
    early, late = _liquid_frames()
    few = early.assign(flag=[1] * 20 + [0] * (len(early) - 20))
    written, measured = _job_on_fake_archive(monkeypatch, _records([]), frames=(few, late))
    assert measured and written["scans"]["toy"]["out_of_sample"] == [], "no evidence, not recorded, the run green"


def test_a_design_date_on_the_last_old_windows_end_is_enough(monkeypatch):
    """The boundary: dated ON the end of the last old window, every window it can start is after it — unseen."""
    written, measured = _job_on_fake_archive(monkeypatch, _records([_OLD_FINAL], fp="new"), fp="new",
                                             designed_on="2026-11-28")
    assert measured == [] and written["scans"]["toy"]["out_of_sample"] == [_OLD_FINAL]


# ── 7. The design-period record re-scores snapshots with today's engine (2026-10-15) ─────────────────
def test_a_snapshot_is_rescored_with_the_current_engine():
    """A stored score is whichever definition was live that day; the design-period record must use today's
    engine on the snapshot's raw columns, or a fixed engine is stamped with its fingerprint but measured old."""
    from core.data_engine import price_trend_inputs
    from core.scoring_engine import compute_momentum_score
    raw = pd.DataFrame({"close_price": [110.0], "vstop_value": [100.0], "last_vstop_change": [1.0],
                        "sma_200d": [100.0], "volume": [10.0], "vol_sma_20d": [10.0],
                        "breakout_window": [np.nan], "dist_52wh": [2.0],
                        "vstop_fresh": [0], "momentum_score": [0.0], "breakout_score": [0.0]})
    out = J.rescore_snapshot(raw)
    want = compute_momentum_score(raw.assign(**price_trend_inputs(raw)))
    assert out["vstop_fresh"].iloc[0] == 1
    assert out["momentum_score"].iloc[0] == want["momentum_score"].iloc[0] > 0
    assert out["breakout_score"].iloc[0] == want["breakout_score"].iloc[0] > 0


# ── 9. The Shariah slice (2026-10-17) ─────────────────────────────────────────────────────────────────
# The operator buys only Shariah stocks of Mega, Large, Mid and Small caps (the caps "Holding 20" already uses), so
# each window also reads the scan on exactly those — against the SAME slice of the market, never the whole one.
def _slice_frames(blocks, extra=0):
    """blocks: (count, shariah_status, market_category, in_scan, end_price) runs of stocks priced 100 at the start.
    `extra` adds that many plain non-scan Shariah Small caps that stay flat."""
    rows = []
    for count, status, cap, flag, end in blocks + ([(extra, "Shariah", "Small Cap", 0, 100.0)] if extra else []):
        rows += [dict(shariah_status=status, market_category=cap, flag=flag, end=end)] * count
    early = pd.DataFrame(rows)
    early["company_id"] = [f"C{i:03d}" for i in range(len(early))]
    early["close_price"], early["equity_shares"], early["vol_sma_20d"] = 100.0, 10.0, 1_000_000.0  # ₹10 Cr a day
    late = early[["company_id", "equity_shares"]].assign(close_price=early["end"])
    return early.drop(columns="end"), late


def test_the_shariah_slice_reads_only_shariah_matches_of_the_four_caps():
    """35 Shariah Small caps match, beside matches the slice must leave out: Shariah Micro and Nano caps (not caps
    the operator buys), a Non-Shariah Mid cap (matched by EQUALITY: "Shariah" sits inside "Non-Shariah") and the
    Review and Doubtful labels (never a pass). A Mega cap is in."""
    early, late = _slice_frames([(34, "Shariah", "Small Cap", 1, 130.0), (1, "Shariah", "Mega Cap", 1, 130.0),
                                 (5, "Shariah", "Micro Cap", 1, 130.0), (3, "Shariah", "Nano Cap", 1, 130.0),
                                 (6, "Non-Shariah", "Mid Cap", 1, 130.0), (2, "Review", "Small Cap", 1, 130.0),
                                 (2, "Doubtful", "Large Cap", 1, 130.0)], extra=41)
    late = late.iloc[:-1]                      # one plain Shariah stock has no price at the end: not in the pool
    r = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")
    cut = r["shariah"]
    assert cut["caps"] == list(S.HOLD_CAPS) == ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap"]
    assert cut["n"] == 35, "34 Small + 1 Mega; Micro, Nano, Non-Shariah, Review and Doubtful left out"
    assert cut["pool"] == 75, ("the market the slice is measured against: every Shariah stock of the four caps "
                               "WITH a return")
    assert r["n"] == 53, "the scan's own reading is unchanged: every match"


def test_the_slice_is_measured_against_the_same_slice_of_the_market():
    """A month that lifts Shariah stocks as a group must not read as scan skill. Shariah matches +30%, other Shariah
    stocks of the four caps +10%, everything else -20%: the slice's excess is 30 - 10 = 20 points, not 30 - (-20)."""
    early, late = _slice_frames([(35, "Shariah", "Small Cap", 1, 130.0), (40, "Shariah", "Mid Cap", 0, 110.0),
                                 (100, "Non-Shariah", "Small Cap", 0, 80.0)])
    r = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")
    assert r["shariah"]["median_excess_pp"] == pytest.approx(20.0)
    assert r["shariah"]["beat_pct"] == 100
    assert r["median_excess_pp"] == pytest.approx(50.0), "the whole-market reading stays as it was"


def test_too_few_shariah_matches_give_only_the_count():
    early, late = _slice_frames([(29, "Shariah", "Small Cap", 1, 130.0), (20, "Non-Shariah", "Small Cap", 1, 130.0)],
                                extra=30)
    cut = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")["shariah"]
    assert cut["n"] == 29
    assert not {"median_excess_pp", "beat_pct", "net", "hold"} & set(cut), "29 matches are not evidence"


def test_a_copy_without_the_shariah_label_records_no_slice():
    """No label, no reading: a slice on absent evidence would be invented (an archive copy is always re-scored with
    today's engine, which writes the label, so this is a guard, not a path the job takes)."""
    early, late = _slice_frames([(40, "Shariah", "Small Cap", 1, 130.0)], extra=40)
    r = J.measure_window(early.drop(columns="shariah_status"), late, _SCAN, "2026-10-31", "2026-11-28")
    assert r is not None and "shariah" not in r


def test_the_slice_net_reading_keeps_liquid_matches_and_charges_the_round_trip():
    early, late = _slice_frames([(35, "Shariah", "Small Cap", 1, 130.0), (40, "Shariah", "Mid Cap", 0, 110.0)])
    early.loc[:4, "vol_sma_20d"] = 10_000.0                    # x ₹100 = ₹0.1 Cr a day: five matches under the floor
    cut = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")["shariah"]
    net = cut["net"]
    assert (net["min_cr"], net["cost_pp"], net["n"]) == (S.NET_MIN_CR, S.NET_COST_PP, 30)
    assert net["median_excess_pp"] == pytest.approx(20.0 - S.NET_COST_PP)
    assert net["portfolio_pp"] == pytest.approx(30.0 - S.NET_COST_PP), "the equal-weight mean of liquid matches"


def test_the_slice_holds_twenty_from_its_own_matches():
    early, late = _slice_frames([(35, "Shariah", "Small Cap", 1, 130.0), (40, "Shariah", "Mid Cap", 0, 110.0),
                                 (30, "Non-Shariah", "Small Cap", 1, 160.0)])
    r = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")
    assert r["shariah"]["hold"]["n"] == 35 and r["hold"]["n"] == 65
    assert r["shariah"]["hold"]["median_pp"] == pytest.approx(30.0 - S.NET_COST_PP)


def test_the_slice_reads_only_stocks_kite_v2_can_buy_a_crore_a_day_or_more():
    """The slice is Kite v2's universe (2026-10-18, its §4.1: Shariah, the four caps, traded value of ₹1 Cr a day or
    more, `>=` as kite_v2/engine/select.py reads it), so the reading is about stocks the operator can buy. A match
    trading ₹0.99 Cr a day leaves the slice AND the market it is measured against; one at exactly ₹1.00 Cr stays.
    SME stocks cannot be told apart in PRISM's data (no lot size), and the reading says so."""
    early, late = _slice_frames([(35, "Shariah", "Small Cap", 1, 130.0), (5, "Shariah", "Small Cap", 1, 50.0),
                                 (40, "Shariah", "Mid Cap", 0, 110.0), (3, "Shariah", "Mid Cap", 0, 200.0)])
    early.loc[35:39, "vol_sma_20d"] = 99_000.0                 # x ₹100 = ₹0.99 Cr a day: five matches out
    early.loc[80:82, "vol_sma_20d"] = 99_000.0                 # three thin non-matches leave the market too
    early.loc[0, "vol_sma_20d"] = 100_000.0                    # ₹1.00 Cr a day exactly: in
    cut = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")["shariah"]
    assert (cut["min_cr"], cut["sme_excluded"]) == (S.SHARIAH_MIN_CR, False) and S.SHARIAH_MIN_CR == 1.0
    assert (cut["n"], cut["pool"]) == (35, 75), "35 matches and 40 Mid caps; the eight thin stocks are out"
    assert cut["median_excess_pp"] == pytest.approx(20.0), "30 - 10: the thin -50% matches never pull it down"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_slice_agrees_with_an_independent_recalculation(seed):
    rng = np.random.default_rng(seed)
    n = 800                     # 400 until the ₹1 Cr floor (2026-10-18) left some seeds' slices under MIN_N
    early = pd.DataFrame({
        "company_id": [f"R{i:03d}" for i in range(n)], "close_price": 100.0, "equity_shares": 10.0,
        "vol_sma_20d": rng.choice([10_000.0, 1_000_000.0], n),
        "flag": rng.integers(0, 2, n),
        "shariah_status": rng.choice(["Shariah", "Non-Shariah", "Review", "Doubtful"], n, p=[0.5, 0.3, 0.1, 0.1]),
        "market_category": rng.choice(["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Nano Cap"], n)})
    late = early[["company_id", "equity_shares"]].assign(close_price=100.0 * (1 + rng.normal(0.02, 0.15, n)))
    cut = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")["shariah"]
    ret = (late["close_price"] / 100.0 - 1) * 100
    pool = ((early["shariah_status"] == "Shariah") & early["market_category"].isin(
        ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap"])
            & (early["vol_sma_20d"] * early["close_price"] / 1e7 >= 1.0))     # Kite v2's ₹1 Cr a day (2026-10-18)
    sub = pool & (early["flag"] == 1)
    ex = ret - ret[pool].median()
    assert (cut["n"], cut["pool"]) == (int(sub.sum()), int(pool.sum()))
    assert cut["median_excess_pp"] == pytest.approx(round(float(ex[sub].median()), 1))
    assert cut["beat_pct"] == int(round(float((ex[sub] > 0).mean() * 100)))

