"""
ui/scan_guide.py — the 🎯 Scans guide: how every scan works, in plain words (2026-10-14).

ONE source, three places: the Reference tab's 🎯 Scans section (searchable), the Reference's Markdown
download, and the Scans tab's "📖 How it works" panel for the selected scan — so they can never disagree.

The user asked that the guide stay right whenever a scan changes or a new one is added. So it holds two
kinds of text:
- FACTS the code already holds are written from it at render time and never typed twice: the one-line
  recipe, the table order, the comparison index, the record's numbers and timing (from the records file),
  the costs, the Holding-20 reading, the order check, why a 🌱 stock left, the candidates being measured.
  They are read as `S.<name>` at call time, so they follow the code (and a test can change them).
- Each scan's EXPLANATION (GUIDES) is written by hand and stamped with the scan it was written for:
  guide_stamp() = its recipe fingerprint (the recipe and every score it reads) plus each setting the guide
  explains. tests/test_scan_guide.py fails while a stamp is stale, a scan has no guide, or a guide has no
  scan: an explanation never outlives its scan, the rule the scan record already follows.
- A scan whose record is a STUDY (🌱) also says how that backtest was run — the data, the StockScans
  columns, the formulas, the traps and how to run it again (GUIDES "backtest"), stamped with
  study_stamp(): a re-run or any edit of the study block makes it be re-read.

Numbers quoted in GUIDES are dated measurements; re-measure them, by the method noted beside them, before
re-stamping. Pure: no Streamlit calls and no I/O (the records are passed in).
"""
import hashlib
import html
import json
import re
from datetime import date

from config import BENCHMARK_INDICES, COLORS
from ui import ui_scans as S

# The scan settings the explanation depends on, beside the recipe itself. A column width explains nothing.
_STAMPED = ("name", "emoji", "recipe", "benchmark", "order", "designed_on", "horizon_days", "start_month",
            "fixed", "extra")
# The hand-written parts of every GUIDES entry (written_for is the stamp).
PROSE_PARTS = ("short", "idea", "recipe", "enter_leave", "benchmark_why", "limits", "changes")
GENERAL_TITLE = "📏 How every scan works"


def guide_stamp(scan: dict) -> str:
    """The scan a guide was written for: its recipe fingerprint plus every setting in _STAMPED — 16 hex
    characters, like the fingerprint itself."""
    settings = {k: scan.get(k) for k in _STAMPED}
    blob = json.dumps({"recipe": S.recipe_fingerprint(scan), "settings": settings}, sort_keys=True,
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def study_stamp(study: dict) -> str:
    """The study a backtest method was written for: every number and note of its record's study block, so a
    re-run (or any edit) makes the method be re-read — 16 hex characters."""
    return hashlib.sha256(json.dumps(study, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


# ── The explanations, one per scan ─────────────────────────────────────────────────────────────────────
# Item grammar (every section): a string is a paragraph, "- " starts a bullet, **words** are bold.
# "Still in the list" figures: membership = scan["mask"] on the saved snapshots, prepared exactly as
# tools/scan_records.py prepares them (improvement_score rebuilt), pairs on the same universe only (the
# vendor grew it from ~2,110 to ~2,715 stocks between 22 Aug and 19 Sep 2026, which moves every rank).
GUIDES = {
    "playbook": dict(
        written_for="44b1f1fe7352f4c6",
        short="strong price trends backed by a fresh strong quarter",
        idea=[
            "Stocks the market is already rewarding — strong, rising prices close to a breakout — whose "
            "latest quarter backs the move with real numbers, whose business got better over the last year, "
            "and whose uptrend has not broken. It reads the price and the results together, so a stock needs "
            "both.",
        ],
        recipe=[
            "- **Momentum in the top third of the market.** PRISM's Momentum score (0–100) blends relative "
            "strength against the market, trend quality, how close the price is to a breakout, volume "
            "confirmation and sector leadership. The stock must rank in the best third of all stocks on it.",
            "- **Breakout in the top third.** The Breakout score measures how close the price is to breaking "
            "out of a base to new highs. Again, the best third of the market.",
            "- **A 🔥 Strong Quarter.** The latest reported quarter against the same quarter last year: "
            "revenue up more than 15%, profit up more than 20%, and the EBITDA margin wider. A company that "
            "made a loss a year earlier cannot be judged, so it does not pass.",
            "- **Improvement in the top half.** PRISM's Improvement score asks whether the business got better "
            "over the last full year — profit growth, revenue growth and the change in operating margin, each "
            "against the year before. The stock must rank in the better half of the market.",
            "- **The price at or above its stop.** The Volatility Stop is a trend line that trails the price "
            "at a distance set by how much the stock normally moves. A price below it means the uptrend has "
            "broken.",
            "Every \"top third\" and \"top half\" is a rank among all the stocks PRISM covers, never among a "
            "filtered slice — so a filter can hide a stock but can never let one in.",
        ],
        enter_leave=[
            "PRISM rebuilds this list from scratch every time its data refreshes. A stock is in it only while "
            "all five conditions hold that day, and it drops out the first day any one of them fails. Being in "
            "the list yesterday never keeps a stock in today.",
            "The usual ways out: the price falls below its stop (the uptrend broke); momentum or breakout slips "
            "out of the market's top third — which can happen while the price holds, because other stocks "
            "moved more; or a new quarterly result is not strong.",
            "So a breakout that fails does not sit in the list for months: its stop or its momentum rank "
            "removes it. Measured on the saved snapshots, 82% of the list was still in it two weeks later "
            "(19 Sep → 3 Oct 2026) and 47% about two months later (17 Jun → 22 Aug 2026).",
        ],
        benchmark_why="a momentum index drawn from the whole market, the closest index fund to what this "
                      "list does.",
        limits=[
            "- **It is a momentum list.** When the market turns, the strongest trends can fall hardest; the "
            "stop condition removes a stock once its uptrend breaks, not before.",
            "- **The results condition looks back one quarter.** A strong quarter says what happened, not what "
            "comes next.",
            "- **Every match is listed, however thinly it trades.** A small stock can be hard to trade in "
            "size.",
        ],
        changes=[
            "Its recipe changes only through a new, measured test written down before the data. The December "
            "2026 review re-tests the measured red flags, may add the \"Cycle ↓\" reading aid to this table — "
            "only if stocks with it beat stocks without it by at least 1 point in every window — and decides "
            "PRISM's default scoring mode, which changes what Rank means, not this list. The order check keeps "
            "reading until it decides.",
        ],
    ),
    "leaders": dict(
        written_for="c5b0c3512a06fdd2",
        short="every stock in the five sectors leading the market",
        idea=[
            "Money tends to move by sector. This scan finds the five sectors where the most companies are "
            "trading near their 52-week highs and reporting strong quarters, then lists every stock in them — "
            "the strong and the weak together. The bet is on the sector, not on the single stock.",
        ],
        recipe=[
            "- **Score every sector.** For each sector with at least 5 stocks, two shares from the 📈 Sectors "
            "tab: the share of its stocks trading within 5% of their 52-week high, and the share whose latest "
            "quarter was strong (revenue up more than 15%, profit up more than 20% and the EBITDA margin wider, "
            "against the same quarter last year).",
            "- **Keep the top five.** Each sector is ranked on each share; the five with the best combined rank "
            "lead. A tie goes to the sector with more stocks near their highs, then alphabetically — so the "
            "answer never depends on the order of the rows.",
            "- **List every stock in those five sectors.** There is no test on the stock itself: a weak stock "
            "in a leading sector is in, and a strong stock in a lagging sector is out.",
        ],
        enter_leave=[
            "The list changes in blocks. A stock enters when its sector joins the top five and leaves when its "
            "sector drops out — one data refresh can swap a whole sector, and a sector that leaves can come "
            "back.",
            "A stock's own price does not decide anything: it can fall below its stop and still stay while its "
            "sector leads. Read its Trend and its distance to the stop before acting.",
            "Measured on the saved snapshots, 44% of the list was still in it two weeks later (19 Sep → "
            "3 Oct 2026) and 34% about two months later (17 Jun → 22 Aug 2026).",
        ],
        benchmark_why="whole sectors of every size, so the broad market is the fair test.",
        limits=[
            "- **No check on the stock itself** — by design. Its trend, its results and its red flags are "
            "yours to read.",
            "- **Uneven lists.** One large sector can make up most of the list.",
            "- **Sector labels come from the data vendor**, mapped to PRISM's names; a company filed under an "
            "odd sector rides with that sector.",
        ],
        changes=[
            "Its recipe changes only through a new, measured test written down before the data. The December "
            "2026 review may add the \"Cycle ↓\" reading aid to this table on the same bar as the Playbook's, "
            "re-tests the measured red flags, and the order check keeps reading until it decides.",
        ],
    ),
    "compounder": dict(
        written_for="63ba93225ce77da7",
        short="profitable growth earned by selling more, read a year at a time",
        idea=[
            "Companies whose last full year shows growth the business earned: sales up more than 10%, profit "
            "growing faster than sales, and margins widening — each among the strongest in the market. It is "
            "a long-horizon research list, read a year at a time; prices are not part of the recipe at all.",
        ],
        recipe=[
            "- **Profit growth in the top 40% of the market.** Last full year's profit against the year "
            "before, only where both years were profits: a loss year has no growth rate, so it cannot pass. "
            "The stock must rank above 60% of the stocks that can be judged.",
            "- **Operating margin widening in the top 40%.** The change in operating margin over the last full "
            "year, ranked across the market the same way.",
            "- **Revenue-led.** Sales up more than 10% AND profit growing faster than sales — margins widened "
            "by selling more, not by cutting costs in a shrinking business. In the backtest the edge all but "
            "vanished without this condition, so it is part of the recipe, not an option.",
            "Full-year figures only. Both top-40% ranks are taken across the whole market before the "
            "revenue-led condition narrows the list.",
        ],
        enter_leave=[
            "The recipe reads only annual results, so the list barely moves from day to day: between 19 Sep and "
            "3 Oct 2026, 250 of its 251 stocks stayed. It changes when a company's new annual numbers reach the "
            "data — and all at once when the data rolls to a new fiscal year: from 17 Jun to 22 Aug 2026, as "
            "the annual figures rolled for 93% of companies, only 52 of 322 stayed.",
            "A falling share price never removes a stock, because the recipe does not read prices.",
            "The Scans tab's \"Left 🌱\" list shows every stock that was here on the last month-end copy and is "
            "not now, with why:",
        ],
        benchmark_why="the December 2026 plan's main test for this list: an index fund of small companies "
                      "chosen for quality and momentum.",
        limits=[
            "- **A year is the unit.** A month of price moves says almost nothing about this list; judge it on "
            "its 12-month windows.",
            "- **The backtest is in-sample.** The recipe was found and tested on the same three years, on "
            "companies that are still listed today.",
            "- **\"Cycle ↓\"** marks stocks whose cash conversion cycle shortened by 15+ days while sales grew "
            "more than 15% in the same year. It did well inside this list in the backtest, but it is a reading "
            "aid — not part of the recipe or the record.",
        ],
        changes=[
            "The December 2026 plan fixes this recipe until its first 12-month unseen window has finished. The "
            "candidates being measured beside it can join only through their own measured test.",
        ],
        # How the backtest behind the record was run (2026-10-14, the user's ask: "so in future we can check
        # and do it again"). Verified against the 🌱 commit of 2026-10-07 (e4193c5), the record's own notes,
        # tools/compounder_years.py and the 2026-10-03 code whose study it reproduced to the stock (median
        # 3Y CAGR; doubled = 26%+ a year). Stamped with the study it describes (study_stamp): a re-run that
        # changes the record makes this be re-read (tests/test_scan_guide.py).
        backtest=dict(
            written_for="b49a4a17fde1607a",
            items=[
                "**The data.** Two downloads joined on each company's StockScans id: PRISM's scored export of 6 "
                "Oct 2026 — 2,712 stocks, whose yearly profit, sales and margin history are PRISM's own StockScans "
                "columns — and a StockScans export of 7 Oct 2026 with every stock's returns.",
                "**The StockScans columns.** On that data the profit-and-loss had rolled to FY26, so:",
                "- **PAT**, **PAT 1 Year Back** … **PAT 5 Years Back** — yearly profit, FY26 back to FY21.",
                "- **Revenue**, **Revenue 1 Year Back** … **Revenue 5 Years Back** — yearly sales, FY26 back to "
                "FY21.",
                "- **OPM 3 Years Back** and **OPM 5 Years Back** — operating margin in FY23 and FY21. PRISM's data "
                "has no OPM 2 or 4 Years Back, so the FY22 and FY24 margins are missing.",
                "- **Returns 1Y**, **Returns 2Y** and **Returns 3Y** — each a per-year rate over the last 1, 2 and 3 "
                "years, not a total, adjusted for splits and bonuses (checked on 3 Oct 2026: HUL, Infosys, TCS and "
                "Reliance reproduce their October 2023 prices).",
                "- **Listed Days** — how long the stock has traded, so it counts only in windows it was listed for.",
                "**The 3-year test, Oct 2023 → Oct 2026.** Each stock is judged on its numbers as they stood in FY23, "
                "public by mid-2023:",
                "- Profit growth FY22 → FY23 = PAT 3 Years Back ÷ PAT 4 Years Back − 1, and sales growth FY22 → FY23 "
                "= Revenue 3 Years Back ÷ Revenue 4 Years Back − 1 — each only where both years were positive.",
                "- Margin change FY21 → FY23 = OPM 3 Years Back − OPM 5 Years Back: two years, because the FY22 "
                "margins are missing — the one way the backtest differs from the live one-year recipe.",
                "- The recipe's own rules on those numbers: profit growth and margin change each in the top 40% of "
                "the stocks that can be judged, ranked across the whole market, AND sales up more than 10% with "
                "profit growing faster.",
                "- The return is Returns 3Y. A group's yearly figure is its median stock's; \"doubled\" means a "
                "per-year return of at least 26% (1.26³ ≈ 2). A stock counts if it has a 3-year return — listed "
                "for 3+ years — and the numbers the recipe reads; the rest are the stocks that count and did not "
                "match.",
                "**The year-by-year test — three one-year windows**, each judged only on the annual results public "
                "at its start:",
                "- Oct 2023 → Oct 2024 on FY23 (PAT and Revenue 3 and 4 Years Back), Oct 2024 → Oct 2025 on FY24 "
                "(2 and 3 Years Back), Oct 2025 → Oct 2026 on FY25 (1 and 2 Years Back).",
                "- One year's return from the per-year rates, with R1, R2 and R3 = Returns 1Y, 2Y and 3Y as "
                "fractions (12.5% = 0.125): 23→24 = (1 + R3)³ ÷ (1 + R2)² − 1; 24→25 = (1 + R2)² ÷ (1 + R1) − 1; "
                "25→26 = R1. A stock counts only if its Listed Days cover the window's start: 3, 2 and 1 years.",
                "- Margin change is net margin (PAT ÷ Revenue) against the year before, because the data has no "
                "yearly operating-margin history.",
                "- Points above the market are measured against the median of every stock with a return in that "
                "window; the record also reads each stock against its own industry's median (industries of 5+ "
                "stocks), within terciles of the previous year's return, and within terciles of start-year sales.",
                "**Checked before use:** Returns 1Y, 2Y and 3Y are per-year rates. Never use Returns Vs Nifty 500 "
                "3Y: it subtracts a 3-year total (Nifty 500 Returns 3Y) from a per-year rate.",
                "**To run it again:**",
                "- On one day, download PRISM's scored export and a StockScans export with Returns 1Y, 2Y and 3Y and "
                "Listed Days for every stock — and keep both. The 7 Oct 2026 StockScans download was not kept, so "
                "these exact numbers can be re-run but not re-checked.",
                "- Map every Years Back column to its fiscal year on that day's data: a year later, each one has "
                "moved by a year.",
                "- The year-by-year windows are the local script tools/compounder_years.py (a PRISM export and a "
                "folder of StockScans exports in, the year rows out as JSON); its year map is fixed to the 6 Oct "
                "2026 data, so move it a year for each later download. The 3-year test's script was not kept; the "
                "steps above rebuild it.",
                "- The scan's own unseen record, above, is the real test: it is measured automatically from the "
                "daily archive, with no backtest needed.",
            ],
        ),
    ),
}

# What each table view is for — keyed by the view's own name, so a new or renamed view needs its words here
# (tests/test_scan_guide.py).
VIEW_WHAT = {
    "🏆 Core": "why a stock is in the list.",
    "📈 Technical": "price and trend readings: is now a good moment, how far is the stop, and does one share "
                   "fit a small account.",
}

# What each measured flag means, in plain words — keyed by the flag's column, so a new flag needs its words
# here (tests/test_scan_guide.py). Read from core/forensic_engine.py on 2026-10-14.
FLAG_WHAT = {
    "rf_wc_double_squeeze": "over the last year customers began paying 10+ days slower while suppliers were "
                            "paid 10+ days faster — cash squeezed from both sides",
    "rf_receivables_bloat": "over the last year the time customers take to pay rose 20+ days more than the "
                            "typical rise in its sector",
    "rf_negative_fcf": "the business burns cash: free cash flow and operating cash flow both negative",
}

_NO_NUMBERS = {
    "unmeasured": "Not measured yet — no numbers.",
    "stale": "The recipe changed since its record was measured; no numbers until it is measured again.",
    "unverified": "The record could not be checked just now; no numbers are shown.",
}


# ── The facts, written from the code ───────────────────────────────────────────────────────────────────
def _day(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day} {d:%b %Y}"


def _count_words(n: int) -> str:
    return {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}.get(n, str(n))


def _first_due(scan: dict) -> str:
    """When the first unseen window can finish — the card's own date (ui_scans), with its year always shown."""
    due = S._first_unseen_due(scan)
    return due if re.search(r"\d{4}$", due) else f"{due} {date.fromisoformat(S.scan_designed_on(scan)).year}"


def _span(lo, hi, fmt: str, sep: str = "–") -> str:
    """'a–b' in one format, or one value when the two are equal. Signed numbers read better joined by
    ' to ' ('+2.9 to +10.3', not '+2.9–+10.3')."""
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}{sep}{fmt.format(hi)}"


def _index(scan: dict) -> str:
    return BENCHMARK_INDICES.get(scan.get("benchmark"), str(scan.get("benchmark")))


def _order_words(scan: dict) -> str:
    return "Momentum, high to low" if S.momentum_ordered(scan) else "PRISM Rank"


def _read_over(scan: dict) -> str:
    """How long one record window runs: '25+ days' or '12 months'."""
    h = S.scan_horizon(scan)
    return f"{round(h / 30.4)} months" if h >= 90 else f"{h}+ days"


def _windows_words(scan: dict) -> str:
    m = S.scan_start_month(scan)
    return (f"one window a year, starting each {date(2000, m, 1):%B}" if m
            else "a window from every month-end")


def _caps_words() -> str:
    names = [str(v).replace(" Cap", "") for v in S.HOLD_CAPS]
    return ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else "".join(names)


def _glance(scan: dict) -> list:
    return [
        f"- **Recipe:** {scan['recipe']}",
        f"- **Measured over:** {_read_over(scan)} — {_windows_words(scan)}.",
        f"- **Listed by:** {_order_words(scan)}.",
        f"- **Compared with:** {_index(scan)}.",
        f"- **Recipe fixed on:** {_day(S.scan_designed_on(scan))} — every window that starts later is unseen.",
    ]


def _order(scan: dict) -> list:
    reading = "The order is for reading only: every match is the scan, and the record counts them all equally."
    if not S.momentum_ordered(scan):
        return ["By PRISM Rank — 1 is PRISM's best overall score.", reading]
    out = ["By Momentum, high to low — the strongest price trends first.",
           "Rank stays a column: click its header, then Sort ascending, to list by it.", reading]
    if scan["key"] in S.ORDER_RULE["scans"]:
        out.append(f"The order check (see Its record, below) records whether the top {S.HOLD_K} by momentum "
                   f"keeps beating random picks.")
    return out


def _exit_reasons(scan: dict) -> list:
    if "exit_reason" not in scan:
        return []
    return [f"- {mark} **{label}** — {why}" for mark, label, why in S.EXIT_REASONS.values()]


def _design_windows(ws: list) -> list:
    if not ws:
        return ["No design-period windows are recorded."]
    n = len(ws)
    ex, bt = [w["median_excess_pp"] for w in ws], [w["beat_pct"] for w in ws]
    out = [f"**Design period (in-sample):** {n} windows between saved snapshots, "
           f"{_day(min(w['start'] for w in ws))} → {_day(max(w['end'] for w in ws))}, studied while the recipe "
           f"was chosen."]
    points = _span(min(ex), max(ex), "{:+.1f}", " to ") + " points"
    beat = _span(min(bt), max(bt), "{:d}") + "%"
    out.append(f"In every window the list's median stock returned more than the market's median stock — by "
               f"{points} — and {beat} of its stocks beat the market's median stock." if min(ex) > 0 else
               f"The list's median stock against the market's median stock: {points} across the windows; "
               f"{beat} of its stocks beat it.")
    net = [w["net"] for w in ws if (w.get("net") or {}).get("median_excess_pp") is not None]
    if net:
        nx, nb = [x["median_excess_pp"] for x in net], [x["beat_pct"] for x in net]
        out.append(f"After costs — only stocks trading ₹{net[0]['min_cr']:g} Cr+ a day, less a "
                   f"{net[0]['cost_pp']:g}-point round trip: {_span(min(nx), max(nx), '{:+.1f}', ' to ')} points, "
                   f"{_span(min(nb), max(nb), '{:d}')}% beating.")
    k = S.non_overlapping(ws)
    if k < n:
        out.append(f"The {n} windows share days: they are {k} separate stretches of the market, not {n} "
                   f"independent results.")
    return out


def _study(st_: dict) -> list:
    out = [f"**{st_['title']} (in-sample), {st_['from']} → {st_['to']}:** stocks matching the recipe compounded "
           f"{st_['match_cagr_pct']:+.1f}% a year against {st_['rest_cagr_pct']:+.1f}% for the rest "
           f"({st_['n_match']:,} against {st_['n_rest']:,} stocks), and {st_['match_doubled_pct']:.1f}% of them "
           f"doubled against {st_['rest_doubled_pct']:.1f}%."]
    if st_.get("years"):
        out.append("Year by year, points above the market's median stock: " + "; ".join(
            f"{y['label']} {y['match_excess_pp']:+.1f} (the rest {y['rest_excess_pp']:+.1f})"
            for y in st_["years"]) + ".")
    if st_.get("cleanest"):
        out.append(f"The cleanest year is {st_['cleanest']['label']}: {st_['cleanest']['why']}.")
    notes = " ".join(x for x in (st_.get("caveats"), st_.get("definition_note"), st_.get("years_note")) if x)
    if notes:
        out.append(f"Caveats: {notes}")
    return out


def _record(scan: dict, rec, state: str) -> list:
    if state != "measured" or not rec:
        return [_NO_NUMBERS.get(state, _NO_NUMBERS["unmeasured"])]
    return _study(rec["study"]) if "study" in rec else _design_windows(rec.get("windows") or [])


def _unseen(scan: dict, rec, state: str) -> list:
    lead = (f"**Unseen record:** windows that start on a month-end copy after "
            f"{_day(S.scan_designed_on(scan))} — data nobody saw while the recipe was chosen.")
    if state != "measured":
        return [lead]
    final = [w for w in (rec or {}).get("out_of_sample") or [] if w.get("final")]
    if not final:
        return [f"{lead} None has finished yet; the first is due around {_first_due(scan)}."]
    return [f"{lead} {len(final)} finished so far — each one is on the scan's card (📊 Full record on the "
            f"Scans tab), written once and never re-measured."]


def _order_check(scan: dict, rec, state: str) -> list:
    if scan["key"] not in S.ORDER_RULE["scans"]:
        return []
    rule = S.ORDER_RULE
    text = (f"**Order check:** over the first {rule['windows']} separate unseen windows, does the top "
            f"{S.HOLD_K} by momentum beat at least {rule['min_avg_pct']:.0f}% of random picks on average, and "
            f"beat the top {S.HOLD_K} by Rank? \"Held\" or \"did not hold\" — evidence about the order, never "
            f"an automatic switch.")
    if state == "measured":
        v = S.order_rule((rec or {}).get("out_of_sample") or [])
        text += (f" So far {v['counted']} of {rule['windows']} windows are in." if v["state"] == "pending" else
                 f" Decided: momentum order {v['state']} (its top {S.HOLD_K} beat {v['mom_avg']:.0f}% of random "
                 f"picks on average, Rank's {v['rank_avg']:.0f}%).")
    return [text]


def scan_sections(scan: dict, rec=None, state: str = "unmeasured") -> list:
    """One scan's guide as (title, items) sections: its explanation (GUIDES) between the facts read from the
    code and from its record. `rec` / `state` are what ui_scans.record_state returns for it."""
    g = GUIDES[scan["key"]]
    # A scan whose record is a study says, right under it, how that backtest was run (GUIDES "backtest").
    method = [("How the backtest was run", list(g["backtest"]["items"]))] if g.get("backtest") else []
    return [
        ("At a glance", _glance(scan)),
        ("The idea", list(g["idea"])),
        ("The recipe, condition by condition", list(g["recipe"])),
        ("When a stock enters and leaves", list(g["enter_leave"]) + _exit_reasons(scan)),
        ("How the table is ordered", _order(scan)),
        ("Its record", _record(scan, rec, state) + _unseen(scan, rec, state)
         + [f"**Compared with {_index(scan)}** — {g['benchmark_why']}"] + _order_check(scan, rec, state)),
        *method,
        ("What it cannot see", list(g["limits"])
         + [f"The limits every scan shares — SME lots, surveillance, forecasts, data timing — are under "
            f"{GENERAL_TITLE} in 📖 Reference → 🎯 Scans."]),
        ("What could change it", list(g["changes"])),
    ]


def general_sections(scans=None) -> list:
    """What every scan shares, as (title, items) sections. The first, "Which scan for what", is the overview
    table the Reference section opens with."""
    scans = S.SCANS if scans is None else scans
    rule = S.ORDER_RULE
    checked = " and ".join(f"{s['emoji']} {s['name']}" for s in scans if s["key"] in rule["scans"])
    return [
        ("Which scan for what", [(
            "table", ("Scan", "Looks for", "Measured over", "Listed by", "Compared with"),
            [(f"{s['emoji']} {s['name']}", GUIDES[s["key"]]["short"][:1].upper() + GUIDES[s["key"]]["short"][1:],
              _read_over(s), _order_words(s), _index(s)) for s in scans])]),
        ("What a scan is", [
            "A scan is a fixed recipe that PRISM runs on the whole market every time its data refreshes — "
            "normally once each trading day, after the close. Every stock that passes every condition is in the "
            "list, and no other stock is. Nothing is picked by hand.",
            "A scan narrows the market to the stocks worth a closer look. It does not tell you to act: open a "
            "stock's Tear-Sheet, read its red flags and its chart, and decide.",
            "Each scan is defined once, in code. The list on the Scans tab, its record and this guide all read "
            "that one definition, so the record describes exactly the stocks you see.",
        ]),
        ("How every scan is measured", [
            "- **Snapshots and month-end copies:** snapshots are PRISM's saved copies of the whole market on "
            "chosen days — the design-period records and the \"still in the list\" figures were measured on "
            "them. Month-end copies are the daily archive's copy from each month's last trading day; unseen "
            "windows and the 🆕 marks start from them.",
            "- **Excess:** a stock's return over a window minus the return of the market's median stock over the "
            "same window. A record gives the median excess of the scan's stocks and the share of them that beat "
            "the market's median stock.",
            "- **In-sample:** the months studied while a recipe was chosen. A good record there shows that the "
            "recipe fitted that period — not that it will work next.",
            "- **Unseen record:** every window that starts on a month-end copy after a scan's design date is out "
            "of sample — data nobody saw while choosing the recipe. A finished window is written once and never "
            "re-measured, so a record cannot be tuned after the fact.",
            f"- **After costs:** only matches trading ₹{S.NET_MIN_CR:g} Cr+ a day, less a {S.NET_COST_PP:g}-point "
            f"round trip for brokerage, taxes and slippage. Each unseen window also compares those stocks, held "
            f"as one equal-weight portfolio, with the scan's index fund after the fund's "
            f"{S.FUND_FEE_PCT_YR:g}% a year fee.",
            f"- **Holding {S.HOLD_K}:** for each unseen window, {S.HOLD_DRAWS:,} random picks of {S.HOLD_K} "
            f"stocks from the {_caps_words()} cap matches with one share at most ₹{S.HOLD_MAX_PRICE:,.0f} — what "
            f"₹1 lakh spread across {S.HOLD_K} stocks can hold — after the round trip: the typical result, its "
            f"10th–90th percentile range, how often a pick lost money, and where the top {S.HOLD_K} by momentum "
            f"and by Rank land among the random picks.",
            "On June–October 2026 data, random picks of 10 Playbook stocks ranged from 0 to +29% (10th–90th "
            "percentile) and picks of 20 from +4.7 to +24%: how many stocks you hold changed the result far "
            "more than which order you picked them in.",
            f"- **Order check** ({checked}): over the first {rule['windows']} separate unseen windows with at "
            f"least {rule['min_n']} holdable matches, does the top {S.HOLD_K} by momentum beat at least "
            f"{rule['min_avg_pct']:.0f}% of random picks on average, and beat the top {S.HOLD_K} by Rank? "
            f"\"Held\" or \"did not hold\" — evidence about the order, never an automatic switch.",
        ]),
        ("How to read the table", [
            *(f"- **{view}** — {VIEW_WHAT[view]}" for view in S.SCAN_VIEWS),
            "- **Filters** only hide rows of the finished list. The recipe always ranks the whole market, so a "
            "filter can never put a stock into a scan.",
            "- **🆕** — not in the scan on the last archived month-end copy. **📄** — its quarterly result was "
            "declared since then. Both together: it entered on a fresh result; 🆕 alone: it entered on price "
            "movement.",
            "- **Category** — the company's market-cap band from the data vendor: Mega, Large, Mid, Small, Micro "
            "or Nano Cap.",
            "- **⚠ Measured flags** — red flags that went with weaker returns when measured on 6 Oct 2026, "
            "provisional until the December 2026 review re-tests them. A flagged stock is shown, never removed: "
            "open its Forensics tab before deciding.",
            *(f"- **⚠ {label}** — {FLAG_WHAT[col]}." if col in FLAG_WHAT else f"- **⚠ {label}**"
              for col, label in S.MEASURED_FLAGS.items()),
            f"- **{S.LIQUIDITY_HEADER}** — average daily traded value over 20 days. Keep one order under about "
            f"{S.ORDER_SHARE:.0%} of it, so the market can take it without moving the price much.",
            "- **Trend** — the Weinstein stage. Being above the stop is not the same line as being above the "
            "30-week average, so a Stage 3 or 4 can appear in a scan.",
            "- **Click a row** to load the stock into the Tear-Sheet tab.",
        ]),
        ("What no scan can see", [
            "- **SME stocks** trade only in lots, often worth more than a small account can hold. The data has "
            "no lot size, so no scan can leave them out.",
            f"- **Surveillance** — {S.SURVEILLANCE_NOTE}",
            "- **Forecasts:** PRISM reads reported results and prices only — no analyst estimates, company "
            "guidance or order books.",
            "- **Timing:** results reach the data a little after they are declared, and the vendor rolls the "
            "profit-and-loss to a new fiscal year months before the balance sheet.",
            "- **Survivors:** a stock with no price at a window's end (delisted or suspended) drops out of that "
            "window's numbers, which flatters a record a little.",
        ]),
        ("Being measured — not scans yet", [
            *(f"- **{name}** — {rule_}" for name, rule_ in S.CANDIDATES),
            "The December 2026 review decides whether any becomes a scan; until then none has a list or a "
            "record.",
        ]),
        ("When a scan changes", [
            "Each scan's explanation here is dated to the exact recipe and settings it describes. If a recipe — "
            "or a score it reads — changes, the scan's record shows no numbers until it is measured again, and "
            "PRISM's own checks refuse the change until this guide has been re-read and updated. The table "
            "order, the comparison index, the costs, the Holding-20 reading, the order check and the reasons a "
            "stock left are written from the code itself, so they change with it. A new scan cannot ship "
            "without its own guide.",
        ]),
    ]


# ── Rendering: one text model, HTML for the app and Markdown for the download ──────────────────────────
_C = COLORS


def _inline(text) -> str:
    """Escaped text with **bold** turned into <strong>."""
    return re.sub(r"\*\*(.+?)\*\*", lambda m: f'<strong style="color:{_C["text_primary"]};">{m.group(1)}</strong>',
                  html.escape(str(text)))


def _table_html(item) -> str:
    _, headers, rows = item
    cell = f"padding:6px 10px;border-bottom:1px solid {_C['border']};text-align:left;vertical-align:top;"
    head = "".join(f'<th style="{cell}font-size:0.66rem;font-weight:700;color:{_C["text_muted"]};'
                   f'text-transform:uppercase;letter-spacing:0.5px;">{_inline(h)}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f'<td style="{cell}font-size:0.78rem;color:{_C["text_secondary"]};">'
                                    f'{_inline(c)}</td>' for c in row) + "</tr>" for row in rows)
    return (f'<div style="overflow-x:auto;margin:4px 0 10px 0;"><table style="border-collapse:collapse;'
            f'width:100%;"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


def _section_html(title: str, items: list) -> str:
    parts, bullets = [], []

    def flush():
        if bullets:
            lis = "".join(f'<li style="margin:3px 0;">{_inline(b)}</li>' for b in bullets)
            parts.append(f'<ul style="margin:2px 0 6px 0;padding-left:18px;font-size:0.78rem;'
                         f'color:{_C["text_secondary"]};line-height:1.55;">{lis}</ul>')
            bullets.clear()

    for item in items:
        if isinstance(item, tuple):
            flush()
            parts.append(_table_html(item))
        elif item.startswith("- "):
            bullets.append(item[2:])
        else:
            flush()
            parts.append(f'<div style="font-size:0.78rem;color:{_C["text_secondary"]};line-height:1.55;'
                         f'margin:3px 0 6px 0;">{_inline(item)}</div>')
    flush()
    return (f'<div style="font-size:0.66rem;font-weight:800;color:{_C["text_muted"]};text-transform:uppercase;'
            f'letter-spacing:0.8px;margin:14px 0 4px 0;">{html.escape(title)}</div>' + "".join(parts))


def _sections_html(sections: list) -> str:
    return "".join(_section_html(title, items) for title, items in sections)


def _block_html(title: str, sub: str, inner: str, open_: bool) -> str:
    return (f'<details{" open" if open_ else ""} style="border:1px solid {_C["border"]};border-radius:10px;'
            f'background:{_C["bg_secondary"]};padding:10px 16px;margin:10px 0;">'
            f'<summary style="cursor:pointer;font-size:0.9rem;font-weight:800;color:{_C["text_primary"]};">'
            f'{html.escape(title)}<span style="font-size:0.76rem;font-weight:500;color:{_C["text_muted"]};">'
            f' — {html.escape(sub)}</span></summary>{inner}</details>')


def _plain(items: list) -> str:
    """A section's words for searching: every paragraph, bullet and table cell, without the ** marks."""
    words = []
    for item in items:
        if isinstance(item, tuple):
            words += [*item[1], *(c for row in item[2] for c in row)]
        else:
            words.append(item)
    return " ".join(map(str, words)).replace("**", "")


def guide_html(scan: dict, rec=None, state: str = "unmeasured") -> str:
    """One scan's guide, every section shown — the Scans tab's 📖 How it works panel."""
    return _sections_html(scan_sections(scan, rec, state))


def _blocks(records: dict, scans) -> list:
    out = []
    for s in scans:
        state, rec = S.record_state(records or {}, s)
        out.append((f"{s['emoji']} {s['name']}", GUIDES[s["key"]]["short"], scan_sections(s, rec, state)))
    return out


def reference_html(records: dict, query: str = "", scans=None) -> str:
    """The Reference tab's 🎯 Scans section. Browsing (no query): a short lead, the overview table, then one
    closed block per scan and one for what every scan shares. Searching: only the sections holding EVERY
    word of the query (token-AND, as in the other corpora), each block open; "" when nothing matches."""
    scans = S.SCANS if scans is None else scans
    general = general_sections(scans)
    blocks = _blocks(records, scans)
    tokens = (query or "").lower().split()
    if not tokens:
        lead = (f"{_count_words(len(scans))} measured lists. Each narrows the market to the stocks worth a "
                f"closer look; the decision stays yours. Open a scan for its recipe in plain words, when a stock "
                f"enters and leaves, its record and what it cannot see.")
        return (_sections_html([("Which scan for what", [lead, *general[0][1]])])
                + "".join(_block_html(t, sub, _sections_html(secs), False) for t, sub, secs in blocks)
                + _block_html(GENERAL_TITLE, "measured, read and kept honest", _sections_html(general[1:]), False))
    out = []
    for t, sub, secs in blocks + [(GENERAL_TITLE, "measured, read and kept honest", general)]:
        kept = [sec for sec in secs
                if all(tok in f"{t}\n{sec[0]}\n{_plain(sec[1])}".lower() for tok in tokens)]
        if kept:
            out.append(_block_html(t, sub, _sections_html(kept), True))
    return "".join(out)


def _section_md(title: str, items: list, level: str) -> list:
    out, bullets = [f"{level} {title}", ""], []

    def flush():
        if bullets:
            out.extend(bullets + [""])
            bullets.clear()

    for item in items:
        if isinstance(item, tuple):
            flush()
            _, headers, rows = item
            cells = lambda row: "| " + " | ".join(str(c).replace("|", "/") for c in row) + " |"  # noqa: E731
            out += [cells(headers), "|" + "---|" * len(headers), *(cells(r) for r in rows), ""]
        elif item.startswith("- "):
            bullets.append(item)
        else:
            flush()
            out += [item, ""]
    flush()
    return out


def guide_markdown(records: dict, scans=None) -> str:
    """The 🎯 Scans part of the Reference download — the same sections the screen shows, as Markdown."""
    scans = S.SCANS if scans is None else scans
    general = general_sections(scans)
    out = ["## 🎯 Scans", "",
           f"_{_count_words(len(scans))} measured lists — each narrows the market; the decision stays yours._", ""]
    out += _section_md(*general[0], "###")
    for (title, sub, secs) in _blocks(records, scans):
        out += [f"### {title} — {sub}", ""]
        for sec in secs:
            out += _section_md(*sec, "####")
    out += [f"### {GENERAL_TITLE}", ""]
    for sec in general[1:]:
        out += _section_md(*sec, "####")
    return "\n".join(out).rstrip() + "\n"
