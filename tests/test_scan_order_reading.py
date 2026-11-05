"""
test_scan_order_reading.py — the order check reads RAM Rank, on what Kite can trade (2026-11-05, the operator's call).

Until now the check (S.ORDER_RULE) asked whether momentum's top 20 keeps beating random picks of what a small account can
hold. The operator: "Change it to RAM Rank from PRISM's momentum score", then "and yes, they should have filters also, same
as we use in the algo". So each unseen window carries one more reading (scan_records_job.order_reading, `order`): from the
scan's matches that pass Kite's filters as far as PRISM can see them, the same draws as "Holding 20", and where RAM Rank's
top 20 and Rank's top 20 land among them. The check reads only that, so a window from before RAM never counts: it starts
again from the first window that has it. Pinned here, each against the break it exists to catch:

  1. The reading: RAM Rank low first, ties by id, blanks dropped, never momentum, and None when fewer than 20 have a rank.
  2. The filters, one case each, on both sides of every line.
  3. A copy missing anything reads n = 0 or None and never raises (the job's own run: tests/test_scan_records_job.py).
  4. "Holding 20" and every other reading moved no number; only momentum's figure is gone.
  5. order_rule reads `order` alone, so an old window does not count.
  6. The words on the card and in the guide; today's records file still loads and renders.
  8. The seal: while S.ORDER_SEALED, the job draws nothing and records no figure, and the card shows none (section 8).
  7. On 🚀 the check reads list C, the list Kite trades from, after the filters (the operator: "only on list C ... and that also
     after filter"). Everything else in 🚀's record stays on the 🚀 list until May.

The word is "trade", not the "buy" of the request: tests/test_scans.py::test_no_buy_wording and
tests/test_scan_guide.py::test_the_guide_promises_nothing_and_never_says_buy hold every scan surface to it.

Run with: pytest tests/test_scan_order_reading.py -v
"""
import html as _html
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui import playbook_feed_job as F  # noqa: E402
from ui import scan_guide as G  # noqa: E402
from ui import scan_records_job as J  # noqa: E402
from ui import ui_scans as S  # noqa: E402

@pytest.fixture(autouse=True)
def _after_the_seal(request, monkeypatch):
    """These tests read what the order check does once the seal is lifted, after Kite's sealed run. The seal itself: the tests
    with "seal" in their name (tests/test_scan_order_reading.py)."""
    if "seal" not in request.node.name:
        monkeypatch.setattr(S, "ORDER_SEALED", False)


_PASSES = dict(shariah_status="Shariah", market_category="Small Cap", close_price=100.0, vol_sma_20d=1_000_000.0,  # ₹10 Cr a day
               dist_to_vstop=10.0, weinstein_stage="📈 Stage 2 Advancing")
_FILTER_COLUMNS = ("company_id", *_PASSES)
_SCAN = {"key": "toy", "mask": lambda d: d["flag"] == 1}
_BY_KEY = {s["key"]: s for s in S.SCANS}
_FILTER_WORDS = "Shariah, NSE, Mega to Small, ₹1 Cr a day, up to ₹5,000 a share, 0–25% above the V-Stop, Stage 1 or 2"
_NOT_IN_DATA = "The price band, ASM, GSM and the SME lot size are not in PRISM's data"
_FROM_RAM = "Older windows have no RAM, so the check counts from the first window that has it"


def _market(n=40):
    """n matches that pass every filter, the first earning the most. RAM Rank runs with the returns, so its top 20 holds the
    20 best; Rank runs against them, so its top 20 holds the 20 worst; momentum's two halves are the odd and even rows,
    so neither end of it can pass for RAM's."""
    early = pd.DataFrame({"company_id": [f"NSE:K{i:03d}" for i in range(n)], **_PASSES, "flag": 1, "equity_shares": 10.0,
                          "ram_rank": np.arange(1.0, n + 1), "rank": np.arange(float(n), 0.0, -1.0),
                          "momentum_score": np.arange(n) % 2 * 50.0 + np.arange(n) / 100})
    return early, pd.Series(np.arange(float(n), 0.0, -1.0), index=early.index)


def _read(early, ret, match=None):
    return J.order_reading(early, ret, pd.Series(True, index=early.index) if match is None else match)


def _late(early, ret):
    return early[["company_id", "equity_shares"]].assign(close_price=early["close_price"] * (1 + ret / 100))


def _recalculated(early, ret, by):
    """The placing worked out by hand, its settings frozen: seed 20261013, 2,000 picks of 20 over the ids in order, a 1-point
    round trip on both sides; low first, ties by id, blanks dropped."""
    pool = pd.DataFrame({"id": early["company_id"], "ret": ret, "by": pd.to_numeric(early[by], errors="coerce")}).sort_values("id")
    r = pool["ret"].to_numpy()
    picks = np.random.default_rng(20261013).random((2000, len(r))).argsort(axis=1)[:, :20]
    sims = r[picks].mean(axis=1) - 1.0
    top = pool.dropna(subset=["by"]).sort_values(["by", "id"])["ret"].head(20).mean() - 1.0
    return int(round(float((sims < top).mean() * 100)))


# ── 1. The reading ───────────────────────────────────────────────────────────────────────────────────────────────────
def test_it_places_ram_ranks_top_twenty_and_ranks_among_random_picks():
    r = _read(*_market())
    assert r["ram_pct"] >= 99, "low RAM Rank first: its top 20 holds the 20 best returns"
    assert r["rank_pct"] <= 1, "Rank's top 20 holds the 20 worst"


def test_momentum_never_stands_in_for_ram():
    """Whichever end of momentum a reading took, it would give a number here: this copy has momentum and no RAM."""
    early, ret = _market()
    r = _read(early.drop(columns="ram_rank"), ret)
    assert r["ram_pct"] is None and r["rank_pct"] <= 1


def test_it_reads_ram_rank_alone():
    """Not the sheet's other RAM columns: the reading's own `ram_pct` is a share of picks, not the sheet's RAM %."""
    early, ret = _market()
    other = early.assign(ram_pct=np.arange(40.0, 0.0, -1.0), ram_score=np.arange(40.0, 0.0, -1.0), ram_date="2026-11-04")
    assert _read(other, ret) == _read(early, ret)


@pytest.mark.parametrize("blank", [np.nan, None, "", "n/a"], ids=["nan", "none", "empty", "words"])
def test_a_blank_rank_is_dropped_never_kept_and_never_the_best(blank):
    """Ten stocks with the worst returns have no rank: filled with 0, or kept in front, they would drag the top 20 down to
    about the middle of the picks."""
    early, ret = _market()
    early["ram_rank"] = early["ram_rank"].astype(object)
    early.loc[30:, "ram_rank"] = blank
    r = _read(early, ret)
    assert r["n"] == 40, "a stock with no rank is still in the pool the picks are drawn from"
    assert r["ram_pct"] >= 99, "the top 20 of the 30 ranked ones"


def test_fewer_than_twenty_ranks_give_none():
    early, ret = _market()
    early.loc[19:, "ram_rank"] = np.nan
    assert _read(early, ret)["ram_pct"] is None, "19 ranks: no top 20, and the blanks must not make up the number"
    early.loc[19, "ram_rank"] = 20.0
    assert _read(early, ret)["ram_pct"] is not None, "20 ranks are enough"


def test_a_tie_goes_to_the_lower_company_id():
    early, ret = _market()
    early.loc[[19, 20], "ram_rank"] = 20.0              # K019 and K020 tie for the last place in the top 20
    ret.loc[[19, 20]] = [-500.0, 500.0]                 # the lower id is the disaster
    worse = _read(early, ret)["ram_pct"]
    ret.loc[[19, 20]] = [500.0, -500.0]                 # the lower id is the fortune
    better = _read(early, ret)["ram_pct"]
    assert worse < better


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_it_agrees_with_a_recalculation_whatever_the_rows_order(seed):
    """Ties, blanks and a shuffled frame on random returns: the same seed, id order and round trip as "Holding 20"."""
    rng = np.random.default_rng(seed)
    early, _ = _market(70)
    early["ram_rank"] = rng.integers(1, 40, 70).astype(float)
    early.loc[rng.choice(70, 15, replace=False), "ram_rank"] = np.nan
    early["rank"] = rng.integers(1, 40, 70).astype(float)
    ret = pd.Series(rng.normal(3.0, 20.0, 70), index=early.index)
    r = _read(early, ret)
    assert (r["ram_pct"], r["rank_pct"]) == (_recalculated(early, ret, "ram_rank"), _recalculated(early, ret, "rank"))
    order = rng.permutation(70)
    shuffled = early.iloc[order].reset_index(drop=True)
    assert _read(shuffled, pd.Series(ret.to_numpy()[order], index=shuffled.index)) == r, "a window always reads the same"


def test_the_picks_come_from_the_pool_alone():
    """Twenty matches outside the pool soar: drawn from, they would lift the picks far above RAM Rank's top 20."""
    early, ret = _market(60)
    early.loc[40:, "company_id"] = [f"BSE:K{i:03d}" for i in range(40, 60)]
    ret.loc[40:] = 1000.0
    r = _read(early, ret)
    assert r["n"] == 40 and r["ram_pct"] == _recalculated(early.iloc[:40], ret.iloc[:40], "ram_rank") >= 99


def test_it_draws_exactly_the_picks_holding_twenty_draws():
    """On the same pool, the same stocks are the top 20 by Rank, so the two readings must place them the same."""
    early, ret = _market()
    w = J.measure_window(early, _late(early, ret), _SCAN, "2026-10-31", "2026-11-28")
    assert w["order"]["n"] == w["hold"]["n"] == 40 and w["order"]["rank_pct"] == w["hold"]["rank_pct"]


def test_it_records_the_filters_with_its_size_and_only_those_under_thirty():
    want = dict(k=20, shariah="Shariah", exchange="NSE", caps=["Mega Cap", "Large Cap", "Mid Cap", "Small Cap"], min_cr=1.0,
                max_price=5000.0, zone=[0.0, 25.0], stages=["1", "2"], cost_pp=1.0, draws=2000)
    assert _read(*_market(29)) == {**want, "n": 29}, "under MIN_N: the size and the filters, nothing more"
    r = _read(*_market(30))
    assert {k: r[k] for k in (*want, "n")} == {**want, "n": 30} and set(r) == {*want, "n", "ram_pct", "rank_pct"}
    assert json.loads(json.dumps(r)) == r, "the job writes it to the records file"


# ── 2. The filters, one case each ────────────────────────────────────────────────────────────────────────────────────
_OUT = {
    "a_bse_stock": {"company_id": "BSE:K030"},
    "an_id_with_no_exchange": {"company_id": "K030"},
    "nse_only_inside_the_id": {"company_id": "BSE:NSE:K030"},
    "non_shariah": {"shariah_status": "Non-Shariah"},
    "review": {"shariah_status": "Review"},
    "doubtful": {"shariah_status": "Doubtful"},
    "no_shariah_label": {"shariah_status": np.nan},
    "micro_cap": {"market_category": "Micro Cap"},
    "nano_cap": {"market_category": "Nano Cap"},
    "no_cap": {"market_category": np.nan},
    "under_a_crore_a_day": {"vol_sma_20d": 99_000.0},
    "no_traded_value": {"vol_sma_20d": np.nan},
    "5001_rupees": {"close_price": 5001.0},
    "no_price": {"close_price": np.nan},
    "26_percent_above_the_stop": {"dist_to_vstop": 26.0},
    "just_over_25_percent": {"dist_to_vstop": 25.01},
    "just_below_the_stop": {"dist_to_vstop": -0.01},
    "no_stop": {"dist_to_vstop": np.nan},
    "stage_3": {"weinstein_stage": "⚠️ Stage 3 Top"},
    "stage_4": {"weinstein_stage": "📉 Stage 4 Declining"},
    "unknown_stage": {"weinstein_stage": "❔ Unknown"},
    "no_stage": {"weinstein_stage": np.nan},
    "a_label_nobody_knows": {"weinstein_stage": "Stage 2"},
}
_IN = {
    "the_plain_stock": {},
    "mega_cap": {"market_category": "Mega Cap"},
    "large_cap": {"market_category": "Large Cap"},
    "mid_cap": {"market_category": "Mid Cap"},
    "exactly_a_crore_a_day": {"vol_sma_20d": 100_000.0},
    "exactly_5000_rupees": {"close_price": 5000.0},
    "on_the_stop": {"dist_to_vstop": 0.0},
    "exactly_25_percent": {"dist_to_vstop": 25.0},
    "stage_1": {"weinstein_stage": "🔄 Stage 1 Basing"},
}


def _thirty_and_one(change):
    """30 stocks that pass, and a 31st with `change`."""
    early, ret = _market(31)
    for col, value in change.items():
        early.loc[30, col] = value
    return early, ret


@pytest.mark.parametrize("change", _OUT.values(), ids=_OUT.keys())
def test_a_stock_outside_a_filter_is_out(change):
    assert _read(*_thirty_and_one(change))["n"] == 30


@pytest.mark.parametrize("change", _IN.values(), ids=_IN.keys())
def test_a_stock_on_the_line_of_a_filter_is_in(change):
    assert _read(*_thirty_and_one(change))["n"] == 31


def test_only_the_scans_matches_are_in_the_pool():
    early, ret = _market(35)
    assert _read(early, ret, pd.Series([True] * 30 + [False] * 5, index=early.index))["n"] == 30


def test_the_stage_mapping_is_the_feeds_own(monkeypatch):
    """Reused, not copied: move a label in the feed's map and the reading follows; and no stage label is typed in the job."""
    early, ret = _thirty_and_one({"weinstein_stage": "⚠️ Stage 3 Top"})
    monkeypatch.setitem(F.STAGE_CODES, "⚠️ Stage 3 Top", "2")
    assert _read(early, ret)["n"] == 31
    assert "Stage 2 Advancing" not in (ROOT / "ui" / "scan_records_job.py").read_text(encoding="utf-8")


def test_the_filters_are_read_from_their_settings(monkeypatch):
    early, ret = _thirty_and_one({"dist_to_vstop": 26.0})
    monkeypatch.setattr(S, "ORDER_ZONE", (0.0, 30.0))
    r = _read(early, ret)
    assert r["n"] == 31 and r["zone"] == [0.0, 30.0]


# ── 3. A copy missing anything: a reading, never a crash ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("column", _FILTER_COLUMNS)
def test_a_copy_missing_a_filter_column_reads_nothing_and_never_raises(column):
    early, ret = _market()
    r = _read(early.drop(columns=column), ret)
    assert r["n"] == 0 and not {"ram_pct", "rank_pct"} & set(r)


def test_a_copy_with_no_columns_at_all_reads_nothing_and_never_raises():
    early, ret = _market()
    assert _read(pd.DataFrame(index=early.index), ret)["n"] == 0


@pytest.mark.parametrize("missing", ["absent", "all blank"])
def test_a_copy_with_no_ram_records_rank_and_none_for_ram(missing):
    early, ret = _market()
    r = _read(early.drop(columns="ram_rank") if missing == "absent" else early.assign(ram_rank=np.nan), ret)
    assert r["n"] == 40 and r["ram_pct"] is None and r["rank_pct"] <= 1


def test_a_copy_with_no_rank_records_ram_and_none_for_rank():
    early, ret = _market()
    r = _read(early.drop(columns="rank"), ret)
    assert r["ram_pct"] >= 99 and r["rank_pct"] is None


# ── 4. Every other reading moved no number ───────────────────────────────────────────────────────────────────────────
def _seeded(n=300, seed=20261105):
    rng = np.random.default_rng(seed)
    early = pd.DataFrame({
        "company_id": [f"NSE:G{i:03d}" for i in rng.permutation(n)],
        "flag": (rng.integers(0, 4, n) > 0).astype(int),
        "close_price": rng.uniform(50.0, 6500.0, n).round(2),
        "vol_sma_20d": rng.uniform(1e3, 1e6, n).round(0),
        "equity_shares": 10.0,
        "market_category": rng.choice(["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap"], n),
        "momentum_score": rng.uniform(0.0, 100.0, n).round(2),
        "rank": rng.permutation(n) + 1.0,
        "shariah_status": rng.choice(["Shariah", "Non-Shariah"], n)})
    late = early[["company_id", "equity_shares"]].assign(
        close_price=(early["close_price"] * (1 + rng.normal(0.03, 0.2, n))).round(2))
    return early, late


_CAPS = ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap"]
_FALLS = [["NSE:G054", -59.7], ["NSE:G155", -45.0], ["NSE:G132", -40.8], ["NSE:G199", -40.0], ["NSE:G148", -37.8],
          ["NSE:G099", -37.0], ["NSE:G106", -35.5], ["NSE:G296", -33.4], ["NSE:G215", -32.1], ["NSE:G183", -31.5],
          ["NSE:G122", -30.4], ["NSE:G024", -30.2]]
# Written by the code before this change (af5cef2) on the frame above, with its two momentum_pct figures (15 and 41) taken out.
_BEFORE = {
    "start": "2026-10-31", "end": "2026-11-28", "days": 28, "median_excess_pp": 1.4, "beat_pct": 53, "n": 224,
    "dropped_no_return": 0, "dropped_corp_action": 0, "falls": _FALLS, "falls_pct": 30.0,
    "net": {"min_cr": 2.0, "cost_pp": 1.0, "n": 221, "median_excess_pp": 0.7, "beat_pct": 51},
    "hold": {"k": 20, "max_price": 5000.0, "caps": _CAPS, "cost_pp": 1.0, "draws": 2000, "n": 134, "median_pp": 1.2,
             "p10_pp": -3.8, "p90_pp": 6.0, "loss_pct": 39, "rank_pct": 94},
    "shariah": {"caps": _CAPS, "min_cr": 1.0, "sme_excluded": False, "n": 81, "pool": 115, "median_excess_pp": 1.0,
                "beat_pct": 53, "net": {"min_cr": 2.0, "cost_pp": 1.0, "n": 81, "median_excess_pp": -0.0, "beat_pct": 49,
                                        "portfolio_pp": 0.22},
                "hold": {"k": 20, "max_price": 5000.0, "caps": _CAPS, "cost_pp": 1.0, "draws": 2000, "n": 65, "median_pp": 0.1,
                         "p10_pp": -4.5, "p90_pp": 4.8, "loss_pct": 49, "rank_pct": 10}},
    "to_check": [sid for sid, _ in _FALLS]}


def test_every_other_reading_moved_no_number_and_only_momentums_figure_is_gone():
    early, late = _seeded()
    w = J.measure_window(early, late, _SCAN, "2026-10-31", "2026-11-28")
    assert {k: v for k, v in w.items() if k != "order"} == _BEFORE
    assert w["order"]["n"] == 0, "this frame has no filter columns but the Shariah label and the cap"


def test_every_window_carries_the_reading_on_every_scan():
    early, ret = _market()
    for key in ("toy", "leaders", "compounder"):
        w = J.measure_window(early, _late(early, ret), {"key": key, "mask": _SCAN["mask"]}, "2026-10-31", "2026-11-28")
        assert w["order"]["ram_pct"] >= 99, key
    w = J.measure_window(early, _late(early, ret), {"key": "playbook", "mask": _SCAN["mask"]}, "2026-10-31", "2026-11-28")
    assert "order" in w, "🚀 carries it too, read on list C: its own section below"


def test_holding_twenty_keeps_its_own_pool():
    """The "Holding 20" line still answers what a small account can hold: ten BSE stocks and ten thin ones are held there.
    The BSE ones soar, so its picks are worth far more with them drawn than without, and Rank's top 20 sits among them."""
    early, ret = _market(60)
    early.loc[40:49, "company_id"] = [f"BSE:K{i:03d}" for i in range(40, 50)]
    early.loc[50:, "vol_sma_20d"] = 1_000.0
    early["rank"] = np.arange(1.0, 61.0)
    ret.loc[40:49] = 500.0
    w = J.measure_window(early, _late(early, ret), _SCAN, "2026-10-31", "2026-11-28")
    assert (w["hold"]["n"], w["order"]["n"]) == (60, 40)
    assert w["hold"]["rank_pct"] == _recalculated(early, ret, "rank") < 50, "drawn from all 60"


def test_a_set_aside_stock_leaves_the_reading_and_stays_in_the_one_beside_it():
    early, ret = _market()
    check = {"company_id": "NSE:K005", "from": "2026-10-31", "through": "2026-11-28", "demerger": "2026-11-07",
             "subject": "Demerger", "checked_on": "2026-11-28"}
    w = J.measure_window(early, _late(early, ret), _SCAN, "2026-10-31", "2026-11-28", checks=[check])
    assert w["set_aside"] == [["NSE:K005", "2026-11-07", "Demerger"]]
    assert (w["order"]["n"], w["as_recorded"]["order"]["n"]) == (39, 40)


# ── 5. The check reads `order` alone ─────────────────────────────────────────────────────────────────────────────────
_CHAIN = [("2026-10-31", "2026-11-30"), ("2026-11-30", "2026-12-31"), ("2026-12-31", "2027-01-29"),
          ("2027-01-29", "2027-02-26"), ("2027-02-26", "2027-03-31"), ("2027-03-31", "2027-04-30"),
          ("2027-04-30", "2027-05-31"), ("2027-05-31", "2027-06-30")]


def _new(span, ram=80, rank=50, n=60, **hold):
    return dict(start=span[0], end=span[1], final=True, hold=dict(k=20, n=60, **hold), order=dict(k=20, n=n, ram_pct=ram, rank_pct=rank))


def _old(span, momentum=99, rank=1):
    """A window as the job wrote it until today: momentum and Rank in `hold`, no `order`."""
    return dict(start=span[0], end=span[1], final=True, hold=dict(k=20, n=60, momentum_pct=momentum, rank_pct=rank))


def test_an_old_window_does_not_count():
    v = S.order_rule([_old(s) for s in _CHAIN[:4]])
    assert v == dict(counted=0, state="pending"), "four windows of old evidence decide nothing"


def test_the_check_starts_again_from_the_first_window_with_ram():
    """Four old windows first (with a bar they would clear), then four with RAM that miss it: the new four decide alone."""
    ws = [_old(s) for s in _CHAIN[:4]] + [_new(s, ram=60) for s in _CHAIN[4:]]
    v = S.order_rule(ws)
    assert v["state"] == "did not hold" and v["counted"] == 4 and v["ram_avg"] == pytest.approx(60.0)
    assert S.order_rule([_old(s) for s in _CHAIN[:4]] + [_new(s) for s in _CHAIN[4:7]])["counted"] == 3, "three in so far"


def test_an_old_window_cannot_block_the_first_window_with_ram():
    """An old window that ends inside the first one with RAM would come first by earliest end, and put that one out of the
    separate windows, if it counted at all."""
    overlapping = _old(("2026-10-31", "2026-12-15"))
    v = S.order_rule([overlapping, *[_new(s) for s in _CHAIN[1:5]]])
    assert v["state"] == "held" and v["counted"] == 4


@pytest.mark.parametrize("window", [
    _new(_CHAIN[0], ram=None), _new(_CHAIN[0], rank=None), _new(_CHAIN[0], n=29),
    dict(start=_CHAIN[0][0], end=_CHAIN[0][1], final=True, order=dict(k=20, n=2)),
    dict(start=_CHAIN[0][0], end=_CHAIN[0][1], final=True, order=None),
    dict(_new(_CHAIN[0]), final=False)],
    ids=["no_ram_place", "no_rank_place", "under_30_stocks", "only_the_size", "order_blank", "still_running"])
def test_a_window_without_both_places_and_thirty_stocks_does_not_count(window):
    assert S.order_rule([window])["counted"] == 0


def test_it_compares_with_ranks_place_in_the_same_pool_not_in_holding_twenty():
    """Rank's place in `hold` is another pool's: at 99 it would sink RAM's 80, and at 1 it would lift a RAM below Rank."""
    assert S.order_rule([_new(s, ram=80, rank=50, rank_pct=99) for s in _CHAIN[:4]])["state"] == "held"
    assert S.order_rule([_new(s, ram=80, rank=90, rank_pct=1) for s in _CHAIN[:4]])["state"] == "did not hold"


def test_held_means_ram_at_the_bar_and_above_rank():
    held = S.order_rule([_new(s, ram=70) for s in _CHAIN[:4]])
    assert held["state"] == "held" and (held["ram_avg"], held["rank_avg"]) == (70.0, 50.0) and "mom_avg" not in held
    assert S.order_rule([_new(s, ram=69) for s in _CHAIN[:4]])["state"] == "did not hold", "under the bar"
    assert S.order_rule([_new(s, ram=80, rank=80) for s in _CHAIN[:4]])["state"] == "did not hold", "level with Rank is not above it"


# ── 6. The words, and today's file ───────────────────────────────────────────────────────────────────────────────────
_HOLD = dict(k=20, max_price=5000.0, caps=list(S.HOLD_CAPS), cost_pp=S.NET_COST_PP, draws=2000, n=97, median_pp=12.6,
             p10_pp=3.7, p90_pp=23.1, loss_pct=2, rank_pct=41)
_CHECKED = ("playbook", "leaders")
_RECORDS = S.load_records()
_MOMENTUMS_ORDER = re.compile(r"momentum'?s (top|order)|top \d+ by momentum|momentum order|momentum top", re.I)


def _window(span=_CHAIN[0], order="default", **hold):
    w = dict(start=span[0], end=span[1], days=30, n=120, median_excess_pp=1.0, beat_pct=55, final=True, hold=dict(_HOLD, **hold))
    if order is not None:
        w["order"] = dict(k=20, n=49, ram_pct=56, rank_pct=38) if order == "default" else order
    return w


def _text(h: str) -> str:
    return " ".join(_html.unescape(re.sub(r"<[^>]+>", " ", h)).split())


def _card(key, windows):
    rec = {"windows": [dict(start="2026-06-17", end="2026-08-22", days=66, median_excess_pp=7.1, beat_pct=69, n=150)],
           "out_of_sample": windows}
    t = _text(S.card_html(_BY_KEY[key], 235, "measured", rec, selected=False, latest="03 Dec"))
    return t[t.index("Holding 20 of them"):]


@pytest.mark.parametrize("key", _CHECKED)
def test_the_card_says_what_the_check_reads_and_names_the_filters_once(key):
    t = _card(key, [_window()])
    assert "Among what Kite can trade, RAM Rank's top 20 beat 56% of random picks, Rank's top 20 38%" in t
    assert f"the order check reads RAM Rank. What Kite can trade: {_FILTER_WORDS}. {_NOT_IN_DATA}." in t
    assert "The check holds if RAM Rank's top 20 beats 70%+ of random picks of those stocks on average" in t
    for words in (_FILTER_WORDS, _NOT_IN_DATA, _FROM_RAM, "Order check: 1 of 4 separate windows in"):
        assert t.count(words) == 1, words
    assert f"{_FROM_RAM}, and only with 30+ of those stocks." in t, "or a dash on a thin list reads as a window with no RAM"
    assert not _MOMENTUMS_ORDER.search(t), _MOMENTUMS_ORDER.search(t)


@pytest.mark.parametrize("key", [*_CHECKED, "compounder"])
def test_the_holding_twenty_sentence_no_longer_mentions_momentum(key):
    rec = dict(_RECORDS["scans"][key], out_of_sample=[_window()])
    t = _text(S.card_html(_BY_KEY[key], 235, "measured", rec, selected=False, latest="03 Dec"))
    t = t[t.index("Holding 20 of them"):t.index("From the Mega")]
    assert "momentum" not in t.lower() and "Rank's top 20 beat 41% of them" in t


def test_the_card_says_whose_order_is_whose():
    play, lead = _card("playbook", [_window()]), _card("leaders", [_window()])
    assert "🏆 Core and 📈 Technical list by momentum since 13 Oct and 🔢 RAM by RAM Rank; the order check reads RAM Rank." in play
    assert "The table lists by momentum since 13 Oct; the order check reads RAM Rank." in lead
    assert "🔢 RAM" not in lead, "🌬️ has no 🔢 RAM view"


def test_the_compounder_card_says_nothing_of_the_check():
    rec = dict(_RECORDS["scans"]["compounder"], out_of_sample=[_window()])
    t = _text(S.card_html(_BY_KEY["compounder"], 320, "measured", rec, selected=False, latest="03 Dec"))
    t = t[t.index("Holding 20 of them"):]
    assert not re.search(r"\bRAM\b|Kite can trade|Order check", t), t


def test_a_decided_check_speaks_of_ram_ranks_order():
    ws = [_window(s, dict(k=20, n=49, ram_pct=m, rank_pct=50)) for s, m in zip(_CHAIN, (80, 75, 66, 72))]
    t = _card("playbook", ws)
    assert ("Order check after 4 separate windows: RAM Rank's order held (its top 20 beat 73% of random picks on average, "
            "Rank's 50%)") in t and "beat 80% / 75% / 66% / 72% of random picks" in t


def test_an_old_window_shows_a_dash_and_never_momentums_figure():
    t = _card("playbook", [_window(order=None, momentum_pct=97)])
    assert "RAM Rank's top 20 beat – of random picks, Rank's top 20 – ·" in t
    assert "97%" not in t and "Order check: 0 of 4" in t
    assert "Rank's top 20 beat 41% of them" in t, "Holding 20's own Rank figure is still on file and still shown"


def test_old_and_new_windows_sit_side_by_side():
    t = _card("playbook", [_window(_CHAIN[0], None, momentum_pct=97), _window(_CHAIN[1])])
    assert "RAM Rank's top 20 beat – / 56% of random picks, Rank's top 20 – / 38%" in t and "Order check: 1 of 4" in t


def test_a_thin_list_shows_a_dash_too():
    t = _card("leaders", [_window(order=dict(k=20, n=2))])
    assert "RAM Rank's top 20 beat – of random picks, Rank's top 20 –" in t and "Order check: 0 of 4" in t


def test_a_copy_with_no_ram_shows_a_dash_for_ram_and_ranks_figure_beside_it():
    t = _card("playbook", [_window(order=dict(k=20, n=49, ram_pct=None, rank_pct=38))])
    assert "RAM Rank's top 20 beat – of random picks, Rank's top 20 38%" in t and "Order check: 0 of 4" in t


def test_todays_records_file_still_loads_and_renders():
    """The job rewrites the file, so this holds no figure of it: only that it loads and every surface draws it."""
    assert _RECORDS, "S.load_records gives nothing for a file of another version"
    for scan in S.SCANS:
        state, rec = S.record_state(_RECORDS, scan)
        assert state == "measured", scan["key"]
        assert S.card_html(scan, 235, state, rec, selected=False, latest="03 Dec") and G.guide_html(scan, rec, state)
    assert G.reference_html(_RECORDS) and G.guide_markdown(_RECORDS)


# A window as the job wrote it until today, copied from ui/scan_records.json on 2026-11-05: 🚀's first design window.
_ON_FILE = {"start": "2026-06-17", "end": "2026-08-22", "days": 66, "median_excess_pp": 7.1, "beat_pct": 70, "n": 148,
            "dropped_no_return": 0, "dropped_corp_action": 10,
            "net": {"min_cr": 2.0, "cost_pp": 1.0, "n": 119, "median_excess_pp": 6.1, "beat_pct": 66},
            "hold": {"k": 20, "max_price": 5000.0, "caps": ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap"], "cost_pp": 1.0,
                     "draws": 2000, "n": 119, "median_pp": 9.7, "p10_pp": 5.2, "p90_pp": 14.9, "loss_pct": 0, "momentum_pct": 81,
                     "rank_pct": 31},
            "revenue_led": {"n_led": 118, "n_rest": 30, "led_mean_pp": 10.06, "rest_mean_pp": 6.59, "led_median_pp": 7.37,
                            "rest_median_pp": 3.48, "mean_gap_pp": 3.48, "median_gap_pp": 3.89}}


@pytest.mark.parametrize("key", _CHECKED)
def test_a_window_on_file_reads_as_a_dash_and_does_not_count(key):
    ws = [dict(_ON_FILE, final=True)]
    t = _card(key, ws)
    assert "RAM Rank's top 20 beat – of random picks, Rank's top 20 – ·" in t and "81%" not in t
    assert "Rank's top 20 beat 31% of them" in t, "Holding 20's own figure stays"
    assert S.order_rule(ws) == dict(counted=0, state="pending") and "So far 0 of 4 windows are in." in _guide(key, ws)


@pytest.mark.parametrize("key", _CHECKED)
def test_the_rank_tip_says_what_the_check_reads(key):
    tip = S.table_config("03 Dec", _BY_KEY[key])["rank"]["help"]
    assert "order check records whether RAM Rank's order keeps beating random picks" in tip and "that order" not in tip


def _guide(key, windows=()):
    rec = dict(_RECORDS["scans"][key], out_of_sample=list(windows))
    return _text(G.guide_html(_BY_KEY[key], rec, "measured"))


@pytest.mark.parametrize("key", _CHECKED)
def test_the_guide_says_what_the_check_reads_and_names_the_filters_once(key):
    t = _guide(key)
    assert "among what Kite can trade" in t and "does the top 20 by RAM Rank beat at least 70% of random picks" in t
    assert "and beat the top 20 by Rank?" in t and "at least 30 such stocks" in t
    for words in (_FILTER_WORDS, _NOT_IN_DATA, _FROM_RAM):
        assert t.count(words) == 1, words
    assert "reads RAM Rank, whichever way the table lists" in t, "its order section says it is not the table's order"
    assert not _MOMENTUMS_ORDER.search(t), _MOMENTUMS_ORDER.search(t)


def test_the_general_section_says_it_too():
    t = " ".join(map(str, dict(G.general_sections())["How every scan is measured"]))
    for words in (_FILTER_WORDS, _NOT_IN_DATA, _FROM_RAM):
        assert t.count(words) == 1, words
    assert "where the top 20 by Rank lands among the random picks" in t, "Holding 20 reads Rank alone"
    assert "top 20 by RAM Rank" in t and not _MOMENTUMS_ORDER.search(t)


def test_a_decided_guide_speaks_of_ram_ranks_order():
    ws = [_window(s, dict(k=20, n=49, ram_pct=80, rank_pct=50)) for s in _CHAIN[:4]]
    assert "Decided: RAM Rank's order held (its top 20 beat 80% of random picks on average, Rank's 50%)" in _guide("playbook", ws)
    assert "Order check" not in _guide("compounder"), "🌱 is not order-checked"


def test_no_surface_still_says_momentums_top():
    for text in (G.reference_html(_RECORDS), G.guide_markdown(_RECORDS),
                 *(G.guide_html(s, _RECORDS["scans"][s["key"]], "measured") for s in S.SCANS),
                 " ".join(map(str, sum((items for _, items in G.general_sections()[1:]), [])))):
        assert not _MOMENTUMS_ORDER.search(_text(text)), _MOMENTUMS_ORDER.search(_text(text))


def test_the_filter_words_follow_their_settings(monkeypatch):
    assert S.order_pool_words() == _FILTER_WORDS
    monkeypatch.setattr(S, "HOLD_MAX_PRICE", 7000.0)
    monkeypatch.setattr(S, "SHARIAH_MIN_CR", 2.0)
    monkeypatch.setattr(S, "ORDER_ZONE", (5.0, 30.0))
    monkeypatch.setattr(S, "ORDER_STAGES", ("2",))
    monkeypatch.setattr(S, "ORDER_EXCHANGE", "BSE")
    monkeypatch.setattr(S, "HOLD_CAPS", ("Large Cap", "Mid Cap"))
    want = "Shariah, BSE, Large to Mid, ₹2 Cr a day, up to ₹7,000 a share, 5–30% above the V-Stop, Stage 2"
    assert S.order_pool_words() == want
    assert want in _guide("playbook")
    assert want in _card("playbook", [_window()])


# ── 7. On 🚀 the check reads list C, after the filters ──
def _list_c_market():
    """120 stocks that pass every filter. The first 41 are the market's top third on momentum and breakout (80 of 120 is on the line, and
    the line is in) and above their stop, so list C; the first 32 of those are also the scan's own matches, as 🚀's are inside C. RAM Rank runs with the returns."""
    n = 120
    early = pd.DataFrame({"company_id": [f"NSE:K{i:03d}" for i in range(n)], **_PASSES, "equity_shares": 10.0,
                          "flag": (np.arange(n) < 32).astype(int), "ram_rank": np.arange(1.0, n + 1),
                          "rank": np.arange(float(n), 0.0, -1.0), "momentum_score": np.arange(float(n), 0.0, -1.0),
                          "breakout_score": np.arange(float(n), 0.0, -1.0)})
    return early, pd.Series(np.arange(float(n), 0.0, -1.0), index=early.index)


def _measured(key, early, ret):
    return J.measure_window(early, _late(early, ret), {"key": key, "mask": _SCAN["mask"]}, "2026-10-31", "2026-11-28")


def test_list_c_is_the_ladder_layer_of_price_only():
    layer = next(x for x in S.PLAYBOOK_LAYERS if x["key"] == S.ORDER_LIST["playbook"])
    assert layer["tag"] == "C" and set(layer["conditions"]) == {"momentum", "breakout", "stop"}
    assert set(S.ORDER_LIST) == {"playbook"}, "only 🚀's check moves to another list"


def test_the_playbooks_check_reads_list_c_not_its_own_matches():
    early, ret = _list_c_market()
    assert _measured("playbook", early, ret)["order"]["n"] == 41, "list C after the filters, not the 32 on the scan"


def test_every_other_scan_reads_its_own_matches():
    early, ret = _list_c_market()
    for key in ("leaders", "compounder", "toy"):
        assert _measured(key, early, ret)["order"]["n"] == 32, key


def test_the_playbooks_other_readings_stay_on_its_own_matches():
    early, ret = _list_c_market()
    w = _measured("playbook", early, ret)
    assert w["n"] == 32 and w["hold"]["n"] == 32, "only the order check moves to list C; the rest of the record is May's"


@pytest.mark.parametrize("change", [dict(shariah_status="Non-Shariah"), dict(company_id="BSE:K"), dict(market_category="Micro Cap"),
                                    dict(close_price=5001.0), dict(dist_to_vstop=26.0), dict(weinstein_stage="📉 Stage 3 Topping")],
                         ids=["label", "exchange", "cap", "price", "zone", "stage"])
def test_list_c_is_read_after_every_filter(change):
    early, ret = _list_c_market()
    for col, value in change.items():
        early.loc[35, col] = value + "035" if col == "company_id" else value
    assert _measured("playbook", early, ret)["order"]["n"] == 40, change


def test_a_list_c_stock_with_no_return_is_out_as_it_is_for_the_scan():
    early, ret = _list_c_market()
    late = _late(early, ret)
    late.loc[35, "close_price"] = np.nan
    w = J.measure_window(early, late, {"key": "playbook", "mask": _SCAN["mask"]}, "2026-10-31", "2026-11-28")
    assert w["order"]["n"] == 40 and w["order"]["ram_pct"] is not None


def test_a_set_aside_stock_leaves_list_cs_reading_too():
    early, ret = _list_c_market()
    gone = pd.Series(early.index == 35, index=early.index)
    assert J._order_matches(early, ret.where(~gone), pd.Series(True, index=early.index), {"key": "playbook"}).sum() == 40


def test_a_copy_with_no_list_c_inputs_reads_nothing_and_never_raises():
    early, ret = _list_c_market()
    w = _measured("playbook", early.drop(columns=["momentum_score", "breakout_score"]), ret)
    assert w["order"]["n"] == 0, "the job's other readings go on"


def test_the_playbooks_card_and_the_guides_say_it_reads_list_c():
    assert S.ORDER_ON_C in _card("playbook", [_window()])
    assert S.ORDER_ON_C not in _card("leaders", [_window()])
    assert S.ORDER_ON_C in _guide("playbook")


# ── 8. The seal. (37), "Sealed until then": before the one sealed run, nothing computes a book that orders by RAM over a session
#      from 30 Nov 2026, in any code, and a figure of one, computed or seen, makes the paper test no evidence. RAM's top 20 among
#      what Kite can trade is such a book. So the check reads RAM Rank today and works out nothing until the seal is lifted. ──
def test_the_seal_is_on_until_the_sealed_run():
    assert S.ORDER_SEALED is True, "set it to False only after the sealed run's read-out, and change this test with it"


def test_a_sealed_reading_draws_nothing_and_records_no_figure(monkeypatch):
    def no_draw(*a, **k):
        raise AssertionError("a sealed reading must not draw")
    monkeypatch.setattr(J, "_random_picks", no_draw)
    monkeypatch.setattr(J, "_placed", no_draw)
    early, ret = _market()
    r = _read(early, ret)
    assert r["sealed"] is True and r["n"] == 40, "its size is an input, no figure of a book"
    assert "ram_pct" not in r and "rank_pct" not in r


def test_a_sealed_window_gives_the_check_nothing_to_count():
    early, ret = _list_c_market()
    w = dict(_measured("playbook", early, ret), final=True)
    assert w["order"]["n"] == 41 and "ram_pct" not in w["order"], "list C after the filters, sealed"
    assert S.order_rule([w])["counted"] == 0


def test_the_seal_leaves_the_other_readings_alone():
    early, ret = _list_c_market()
    w = _measured("playbook", early, ret)
    assert "median_pp" in w["hold"] and w["hold"]["rank_pct"] is not None and "momentum_pct" not in w["hold"]


def test_a_sealed_card_shows_no_figure_of_the_check_even_from_a_stored_one():
    t = _card("playbook", [_window()])
    assert S.ORDER_SEALED_WORDS in t and "Order check: sealed until May 2027" in t
    assert "of random picks, Rank's top 20" not in t and "56%" not in t


def test_a_sealed_guide_says_so():
    assert S.ORDER_SEALED_WORDS in _guide("playbook") and S.ORDER_SEALED_WORDS in _guide("leaders")


def test_todays_records_file_renders_under_the_seal():
    records = S.load_records()["scans"]
    for key in _CHECKED:
        assert "Order check: sealed until May 2027" in _card(key, records[key]["windows"])
