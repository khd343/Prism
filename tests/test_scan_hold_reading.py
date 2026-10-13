"""
test_scan_hold_reading.py — "what if you held 20 of them?", and the rule that decides a scan's order
(2026-10-13, written BEFORE the first unseen window).

Measured first (docs/december-2026-plan.md rule 26 E-F): on windows that do not overlap, the order inside
a scan barely mattered — the top 25 by PRISM Rank sat at the 38th/33rd/73rd percentile of random picks
(Playbook) and 38/60/46 (Leading-Sector), the top 25 by momentum at 93/63/83 and 68/69/79 — while HOW MANY
stocks you hold mattered a lot: random picks from the buyable Playbook list spanned 0 to +29% with 10
names and +4.7 to +24% with 20. A long OVERLAPPING window had faked a big order effect.

So every unseen window now records one more reading (scan_records_job.measure_window, `hold`): 20 stocks
picked at random, many times with a FIXED seed, from the matches a Rs 1 lakh account can actually hold —
Mega, Large, Mid or Small cap (S.HOLD_CAPS: the user's call 2026-10-13, replacing the Rs 2 Cr a day floor
for this reading only) AND one share at most S.HOLD_MAX_PRICE (13 of 185 buyable stocks cost more on
2026-10-12) — after the record's round-trip cost: the typical result, its 10th-90th range,
how often it lost money, and where the top 20 by momentum and by Rank land among those random picks.

And the rule, fixed now (S.ORDER_RULE): a scan's table switches to momentum order only if, over its first
four separate (non-overlapping), finished unseen windows with at least 30 holdable matches, the top 20 by
momentum beats on average at least 70% of random picks AND averages higher than Rank's top 20. Luck alone
passes the 70% average about 1 time in 12 (8.6%); a "3 of 4 above half" rule would pass 31% of the time.
Each scan decides on its own windows; only the Playbook and Leading-Sector are judged.

Run with: pytest tests/test_scan_hold_reading.py -v
"""
import html as _html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui import scan_records_job as J  # noqa: E402
from ui import ui_scans as S  # noqa: E402


# ── 1. The reading is fixed before any result ───────────────────────────────────────────────────────
def test_the_hold_reading_and_the_order_rule_are_declared_before_any_result():
    assert (S.HOLD_K, S.HOLD_MAX_PRICE, S.HOLD_DRAWS) == (20, 5000.0, 2000)
    assert S.HOLD_CAPS == ("Mega Cap", "Large Cap", "Mid Cap", "Small Cap"), "no Micro or Nano caps"
    assert isinstance(S.HOLD_SEED, int), "a FIXED seed: a window's numbers can be re-checked and never move"
    assert S.ORDER_RULE == dict(scans=("playbook", "leaders"), windows=4, min_avg_pct=70.0, min_n=30)
    assert S.ORDER_RULE["min_n"] == J.MIN_N


# ── 2. One window's reading ─────────────────────────────────────────────────────────────────────────
def _market(n_hold=60):
    """n_hold cheap Small-cap matches whose return rises with momentum (and falls with Rank's order) — HALF
    of them trading only Rs 0.01 Cr a day, because liquidity no longer decides (2026-10-13); 10 Micro-cap
    and 10 Rs 6,000-a-share matches that soar (+500%) and must be left out; 100 flat non-matches."""
    rows = []
    for i in range(n_hold):
        rows.append(dict(company_id=f"H{i:03d}", flag=1, close=100.0, vol=1e6 if i % 2 else 1e3, ret=i / 100.0,
                         cap="Small Cap", momentum_score=float(i), rank=i + 1))   # Rank's top = the worst returns
    for i in range(10):
        rows.append(dict(company_id=f"I{i:03d}", flag=1, close=100.0, vol=1e6, ret=5.0,       # liquid, but Micro
                         cap="Micro Cap", momentum_score=999.0, rank=1000 + i))
        rows.append(dict(company_id=f"X{i:03d}", flag=1, close=6000.0, vol=1e6, ret=5.0,      # one share Rs 6,000
                         cap="Large Cap", momentum_score=999.0, rank=2000 + i))
    for i in range(100):
        rows.append(dict(company_id=f"N{i:03d}", flag=0, close=100.0, vol=1e6, ret=0.0,
                         cap="Small Cap", momentum_score=0.0, rank=3000 + i))
    d = pd.DataFrame(rows)
    early = pd.DataFrame({"company_id": d["company_id"], "close_price": d["close"], "vol_sma_20d": d["vol"],
                          "market_category": d["cap"], "equity_shares": 10.0, "flag": d["flag"],
                          "momentum_score": d["momentum_score"], "rank": d["rank"]})
    late = early[["company_id", "equity_shares"]].assign(close_price=d["close"] * (1 + d["ret"]))
    return early, late


_SCAN = {"key": "toy", "mask": lambda d: d["flag"] == 1}


def _hold(early, late):
    r = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")
    return r["hold"]


def test_only_stocks_a_small_account_can_hold_are_picked():
    """Mega to Small caps with one share at most Rs 5,000 — whatever they trade a day (the user's call,
    2026-10-13): the 30 illiquid Small caps count, the Micro caps and the Rs 6,000 shares do not."""
    h = _hold(*_market())
    assert h["n"] == 60, "the Micro-cap and the Rs 6,000-a-share matches are not holdable; illiquid Small caps are"
    assert (h["k"], h["max_price"], h["caps"], h["cost_pp"]) == (20, 5000.0, list(S.HOLD_CAPS), S.NET_COST_PP)
    assert "min_cr" not in h, "no traded-value floor on this reading"
    # The holdable matches return 0..59% (mean 29.5); a +500% stock slipping in would lift these far above.
    assert 20 < h["median_pp"] + S.NET_COST_PP < 40 and h["p90_pp"] < 45 and h["p10_pp"] > 15


def test_the_reading_is_after_costs_and_counts_losses():
    early, late = _market()
    late["close_price"] = early["close_price"] * 1.004          # +0.4% everywhere: a loss after a 1pp round trip
    h = _hold(early, late)
    assert h["median_pp"] == pytest.approx(0.4 - S.NET_COST_PP, abs=0.01)
    assert h["loss_pct"] == 100


def test_the_orders_are_placed_among_random_picks():
    h = _hold(*_market())
    assert h["momentum_pct"] >= 99, "momentum's top 20 holds the 20 best returns here"
    assert h["rank_pct"] <= 1, "Rank's top 20 holds the 20 worst here"


def test_the_reading_never_moves_once_written():
    early, late = _market()
    a = _hold(early, late)
    b = _hold(early.sample(frac=1.0, random_state=3).reset_index(drop=True), late)
    assert a == b, "a fixed seed over the matches in a fixed order: the same window always reads the same"


def test_a_window_with_too_few_holdable_matches_is_noted_and_not_read():
    h = _hold(*_market(n_hold=29))
    assert h["n"] == 29 and "median_pp" not in h and "momentum_pct" not in h


# ── 3. The rule that decides the order ──────────────────────────────────────────────────────────────
_CHAIN = [("2026-10-31", "2026-11-30"), ("2026-11-30", "2026-12-31"), ("2026-12-31", "2027-01-29"),
          ("2027-01-29", "2027-02-26"), ("2027-02-26", "2027-03-31")]


def _w(span, mom, rank, n=60, final=True):
    hold = dict(k=20, n=n)
    if n >= 30:
        hold.update(momentum_pct=mom, rank_pct=rank)
    return dict(start=span[0], end=span[1], final=final, hold=hold)


def test_the_order_rule_waits_for_four_separate_finished_windows():
    ws = [_w(s, 90, 40) for s in _CHAIN[:3]]
    v = S.order_rule(ws)
    assert v["state"] == "pending" and v["counted"] == 3
    v = S.order_rule(ws + [_w(_CHAIN[3], 90, 40, final=False)])
    assert v["state"] == "pending" and v["counted"] == 3, "a window still running does not count"
    v = S.order_rule(ws + [_w(("2026-11-30", "2027-01-29"), 90, 40)])
    assert v["state"] == "pending" and v["counted"] == 3, "an overlapping window does not count"
    v = S.order_rule(ws + [_w(_CHAIN[3], None, None, n=25)])
    assert v["state"] == "pending" and v["counted"] == 3, "fewer than 30 holdable matches: the next one counts"


def test_momentum_order_is_earned_only_by_the_declared_bar():
    earned = S.order_rule([_w(s, m, 50) for s, m in zip(_CHAIN, (80, 75, 66, 72))])
    assert earned["state"] == "earned" and earned["mom_avg"] == pytest.approx(73.25)
    short = S.order_rule([_w(s, m, 50) for s, m in zip(_CHAIN, (80, 75, 60, 62))])
    assert short["state"] == "not earned", "an average under 70 is not enough"
    behind = S.order_rule([_w(s, 80, 85) for s in _CHAIN[:4]])
    assert behind["state"] == "not earned", "momentum must also average above Rank's top 20"


def test_only_the_first_four_separate_windows_decide():
    ws = [_w(s, 80, 50) for s in _CHAIN[:4]] + [_w(_CHAIN[4], 0, 100)]
    v = S.order_rule(ws)
    assert v["state"] == "earned" and v["counted"] == 4, "a later window cannot reopen a decided rule"


# ── 4. On the card ──────────────────────────────────────────────────────────────────────────────────
def _unseen_with_hold():
    hold = dict(k=20, max_price=5000.0, caps=list(S.HOLD_CAPS), cost_pp=S.NET_COST_PP, draws=2000, n=97,
                median_pp=12.6, p10_pp=3.7, p90_pp=23.1, loss_pct=2, momentum_pct=56, rank_pct=41)
    return [dict(start="2026-10-31", end="2026-11-30", days=30, n=120, median_excess_pp=1.0, beat_pct=55,
                 final=True, hold=hold)]


def _playbook_record():
    return {"windows": [dict(start="2026-06-17", end="2026-08-22", days=66, median_excess_pp=7.1, beat_pct=69,
                             n=150)], "out_of_sample": _unseen_with_hold()}


def test_the_full_record_shows_what_holding_20_gave_and_where_the_rule_stands():
    h = _html.unescape(S.card_html(S.SCANS[0], 235, "measured", _playbook_record(), selected=False,
                                   latest="03 Dec"))
    for s in ("Holding 20 of them", "+12.6%", "+3.7 to +23.1", "lost money in 2%",
              "momentum's top 20 beat 56%", "Rank's top 20 41%", "Order rule: 1 of 4"):
        assert s in h, s
    assert h.index("NEVER SAW") < h.index("Holding 20 of them"), "it reads under the unseen record"
    line = h[h.index("Holding 20 of them"):]
    assert "Mega, Large, Mid and Small cap matches" in line and "one share at most ₹5,000" in line
    assert "Cr+ a day" not in line[:line.index("Order rule")], "this reading has no traded-value floor"


def test_the_compounder_shows_the_reading_but_no_order_rule():
    comp = next(s for s in S.SCANS if s["key"] == "compounder")
    import json
    study = json.loads(S.RECORDS_PATH.read_text(encoding="utf-8"))["scans"]["compounder"]
    rec = dict(study, out_of_sample=_unseen_with_hold())
    h = _html.unescape(S.card_html(comp, 320, "measured", rec, selected=False, latest="03 Dec"))
    assert "Holding 20 of them" in h and "Order rule" not in h, "🌱 keeps its order; only two scans are judged"


def test_no_hold_line_before_a_window_carries_it():
    rec = _playbook_record()
    rec["out_of_sample"][0].pop("hold")
    h = S.card_html(S.SCANS[0], 235, "measured", rec, selected=False, latest="03 Dec")
    assert "Holding 20" not in h and "Order rule" not in h
