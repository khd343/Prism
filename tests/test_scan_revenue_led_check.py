"""
test_scan_revenue_led_check.py — inside 🚀, do the revenue-led matches beat the rest? (registered 2026-10-15,
BEFORE any unseen window has a result).

Where it came from: a Kite-side plan proposed trading "🚀 stocks that are also revenue-led" in Kite's window 2.
An in-sample pre-check (data to 12 Oct; 17 Jun -> 22 Aug, 22 Aug -> 16 Sep, 16 Sep -> 12 Oct) read +2.0 / +0.3
/ +6.0 points on the mean, revenue-led minus the rest of 🚀: 3 of 3, but the middle a tie. Kite's go-rule needs
the SAME design in two windows in a row, so the operator kept Kite frozen and chose to MEASURE the idea beside it.

So every unseen 🚀 window records one more reading (scan_records_job.measure_window, `revenue_led`): its scored
matches split by ui_scans.revenue_led, the test 🌱 uses, read from it and never re-typed; each arm's size, and
once both arms have S.REVENUE_LED_RULE['min_n'], each arm's mean and median excess over the window's market
median and the two gaps. The check (S.revenue_led_rule) HOLDS if, over the scan's first three separate finished
windows with both arms that large, the revenue-led matches beat the rest by at least one point on the mean AND
on the median in EVERY one; otherwise "did not hold", decided once. Three monthly windows because the archive
keeps month-end copies: 31 Oct -> 30 Nov -> 31 Dec -> 31 Jan cover the months of Kite's windows 1 and 2. It
decides nothing by itself: nothing reaches the list, the feed or Kite.

Run with: pytest tests/test_scan_revenue_led_check.py -v
"""
import html as _html
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui import scan_guide as G  # noqa: E402
from ui import scan_records_job as J  # noqa: E402
from ui import ui_scans as S  # noqa: E402


# ── 1. The rule is fixed before any result ──────────────────────────────────────────────────────────
def test_the_revenue_led_rule_is_declared_before_any_result():
    assert S.REVENUE_LED_RULE == dict(scans=("playbook",), windows=3, min_gap_pp=1.0, min_n=30)
    assert S.REVENUE_LED_RULE["min_n"] == J.MIN_N


# ── 2. One window's reading ─────────────────────────────────────────────────────────────────────────
def _market(n_rest=60):
    """40 revenue-led matches (sales +20%, profit +40%): 39 rise 10%, one 50% (mean 11, median 10); n_rest
    matches that are not (sales +5%) rise 5%; 150 flat non-matches, so the market median is exactly 0."""
    rows = [dict(company_id=f"L{i:03d}", flag=1, revenue=120.0, pat=14.0, ret=0.50 if i == 0 else 0.10)
            for i in range(40)]
    rows += [dict(company_id=f"R{i:03d}", flag=1, revenue=105.0, pat=14.0, ret=0.05) for i in range(n_rest)]
    rows += [dict(company_id=f"N{i:03d}", flag=0, revenue=100.0, pat=10.0, ret=0.0) for i in range(150)]
    d = pd.DataFrame(rows)
    early = d.drop(columns="ret").assign(revenue_1yb=100.0, pat_1yb=10.0, close_price=100.0, equity_shares=10.0)
    late = early[["company_id", "equity_shares"]].assign(close_price=100.0 * (1 + d["ret"]))
    return early, late


_PLAYBOOK = {"key": "playbook", "mask": lambda d: d["flag"] == 1}


def _reading(early, late, scan=_PLAYBOOK):
    return J.measure_window(early, late, scan, "2026-10-31", "2026-11-30").get("revenue_led")


def test_the_window_splits_its_matches_by_the_test_the_compounder_uses():
    r = _reading(*_market())
    assert (r["n_led"], r["n_rest"]) == (40, 60)
    assert r["led_mean_pp"] == pytest.approx(11.0) and r["led_median_pp"] == pytest.approx(10.0)
    assert r["rest_mean_pp"] == pytest.approx(5.0) and r["rest_median_pp"] == pytest.approx(5.0)
    assert r["mean_gap_pp"] == pytest.approx(6.0) and r["median_gap_pp"] == pytest.approx(5.0)


def test_the_split_reads_the_shared_test_never_a_copy(monkeypatch):
    monkeypatch.setattr(S, "revenue_led", lambda d: d["company_id"].str.startswith("R"))
    r = _reading(*_market())
    assert (r["n_led"], r["n_rest"]) == (60, 40), "the reading must follow ui_scans.revenue_led"


def test_the_split_leaves_out_what_the_headline_leaves_out():
    early, late = _market()
    late.loc[late["company_id"] == "L001", "equity_shares"] = 50.0     # a split: out of the window
    late = late[late["company_id"] != "L002"]                           # no price at the end: out
    assert _reading(early, late)["n_led"] == 38


def test_an_arm_under_the_minimum_is_counted_and_not_read():
    r = _reading(*_market(n_rest=29))
    assert (r["n_led"], r["n_rest"]) == (40, 29)
    assert "mean_gap_pp" not in r and "led_mean_pp" not in r


def test_only_the_scans_the_rule_names_carry_the_reading():
    early, late = _market()
    r = J.measure_window(early, late, {"key": "leaders", "mask": lambda d: d["flag"] == 1}, "2026-10-31",
                         "2026-11-30")
    assert "revenue_led" not in r


# ── 3. The check ────────────────────────────────────────────────────────────────────────────────────
_CHAIN = [("2026-10-31", "2026-11-30"), ("2026-11-30", "2026-12-31"), ("2026-12-31", "2027-01-29"),
          ("2027-01-29", "2027-02-26")]


def _w(span, mean_gap, median_gap, final=True, n_rest=60):
    rl = dict(n_led=40, n_rest=n_rest)
    if n_rest >= 30:
        rl.update(led_mean_pp=mean_gap + 2.0, rest_mean_pp=2.0, led_median_pp=median_gap + 1.0, rest_median_pp=1.0,
                  mean_gap_pp=mean_gap, median_gap_pp=median_gap)
    return dict(start=span[0], end=span[1], final=final, revenue_led=rl)


def test_the_check_waits_for_three_separate_finished_windows():
    ws = [_w(s, 3.0, 3.0) for s in _CHAIN[:2]]
    assert S.revenue_led_rule(ws) == dict(counted=2, state="pending")
    v = S.revenue_led_rule(ws + [_w(_CHAIN[2], 3.0, 3.0, final=False)])
    assert v["state"] == "pending" and v["counted"] == 2, "a window still running does not count"
    v = S.revenue_led_rule(ws + [_w(("2026-11-30", "2027-01-29"), 3.0, 3.0)])
    assert v["state"] == "pending" and v["counted"] == 2, "an overlapping window does not count"
    v = S.revenue_led_rule(ws + [_w(_CHAIN[2], None, None, n_rest=25)])
    assert v["state"] == "pending" and v["counted"] == 2, "an arm under 30: the next window counts instead"


def test_it_holds_only_when_every_window_clears_one_point_on_both_readings():
    held = S.revenue_led_rule([_w(s, a, b) for s, (a, b) in zip(_CHAIN, ((1.0, 1.0), (2.5, 1.4), (6.0, 5.1)))])
    assert held["state"] == "held", "exactly one point counts"
    assert held["mean_gaps"] == [1.0, 2.5, 6.0] and held["median_gaps"] == [1.0, 1.4, 5.1]
    for gaps in (((2.0, 2.0), (0.3, 1.5), (6.0, 5.0)),        # a tie on the mean (the in-sample middle window)
                 ((2.0, 2.0), (1.5, 0.4), (6.0, 5.0)),        # a tie on the middle stock
                 ((2.0, 2.0), (2.0, 2.0), (-3.0, -2.0))):     # worse in one window
        v = S.revenue_led_rule([_w(s, a, b) for s, (a, b) in zip(_CHAIN, gaps)])
        assert v["state"] == "did not hold", gaps


def test_only_the_first_three_separate_windows_decide():
    ws = [_w(s, 3.0, 3.0) for s in _CHAIN[:3]] + [_w(_CHAIN[3], -9.0, -9.0)]
    v = S.revenue_led_rule(ws)
    assert v["state"] == "held" and v["counted"] == 3, "a later window cannot reopen a decided check"


# ── 4. On the 🚀 card ────────────────────────────────────────────────────────────────────────────────
def _unseen(*windows):
    return [dict(start=w["start"], end=w["end"], days=30, n=120, median_excess_pp=1.0, beat_pct=55, final=True,
                 revenue_led=w["revenue_led"]) for w in windows]


def _record(ws):
    return {"windows": [dict(start="2026-06-17", end="2026-08-22", days=66, median_excess_pp=7.1, beat_pct=69,
                             n=150)], "out_of_sample": ws}


def _card(scan_key, ws):
    scan = next(s for s in S.SCANS if s["key"] == scan_key)
    return _html.unescape(S.card_html(scan, 235, "measured", _record(ws), selected=False, latest="03 Dec"))


def test_the_playbook_card_shows_the_split_and_where_the_check_stands():
    w = _w(_CHAIN[0], 3.8, 2.1)
    w["revenue_led"]["n_led"], w["revenue_led"]["n_rest"] = 52, 118
    h = _card("playbook", _unseen(w))
    for s in ("Revenue-led vs the rest", "+3.8", "+2.1", "52 vs 118", "Revenue-led check: 1 of 3"):
        assert s in h, s
    assert h.index("NEVER SAW") < h.index("Revenue-led vs the rest"), "it reads under the unseen record"
    assert "changes nothing in the list or in Kite" in h


def test_the_card_states_a_decided_check_in_words():
    for gaps, word in ((((2.0, 2.0), (2.5, 1.4), (6.0, 5.1)), "held"),
                       (((2.0, 2.0), (0.3, 1.5), (6.0, 5.0)), "did not hold")):
        h = _card("playbook", _unseen(*[_w(s, a, b) for s, (a, b) in zip(_CHAIN, gaps)]))
        assert f"Revenue-led check after 3 separate windows: {word}" in h, word


def test_an_arm_too_small_to_read_shows_a_dash():
    h = _card("playbook", _unseen(_w(_CHAIN[0], None, None, n_rest=25)))
    assert "Revenue-led vs the rest of the list: – points on the mean" in h and "40 vs 25" in h


def test_no_other_card_shows_it_and_none_before_a_window_carries_it():
    assert "Revenue-led" not in _card("leaders", _unseen(_w(_CHAIN[0], 3.0, 3.0)))
    bare = _unseen(_w(_CHAIN[0], 3.0, 3.0))
    bare[0].pop("revenue_led")
    assert "Revenue-led" not in _card("playbook", bare)


# ── 5. In the 🚀 guide, written from the rule ─────────────────────────────────────────────────────────
def _record_section(scan_key, rec):
    scan = next(s for s in S.SCANS if s["key"] == scan_key)
    return " ".join(dict(G.scan_sections(scan, rec, "measured"))["Its record"])


def test_the_playbook_guide_states_the_check_and_where_it_stands():
    text = _record_section("playbook", _record(_unseen(_w(_CHAIN[0], 3.0, 3.0))))
    assert "**Revenue-led check:**" in text and "at least 1 point" in text and "first 3 separate" in text
    assert "So far 1 of 3 windows are in." in text
    rec = _record(_unseen(*[_w(s, 3.0, 3.0) for s in _CHAIN[:3]]))
    assert "Decided: it held." in _record_section("playbook", rec)


def test_the_guide_follows_the_rule_not_a_copy(monkeypatch):
    monkeypatch.setattr(S, "REVENUE_LED_RULE", dict(S.REVENUE_LED_RULE, windows=4, min_gap_pp=2.0))
    text = _record_section("playbook", _record([]))
    assert "at least 2 points" in text and "first 4 separate" in text


def test_no_other_scan_guide_mentions_it():
    for key in ("leaders", "compounder"):
        assert "Revenue-led check" not in _record_section(key, _record([])), key
