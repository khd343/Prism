"""
🔢 RAM in plain words: the Reference tab's page (2026-11-05, the operator: "write about RAM, all time best").

One page, the same words on screen, in the tab's search and in its download. PRISM does not compute RAM: the RAM job in
stockscans-sync does, from NSE's own files, and writes four columns to the sheet. So its facts cannot be imported here. They
were read from that job's code at 7a9a867 (ram/compute.py; the frozen vendor/kite_v2/scoretest/scores.py) and from its record for
2026-11-05, and tests/test_ram_reference.py pins them. What PRISM holds itself, Momentum's weights and the view's name, is read
from the code. On every RAM surface: no word of action, no person, no promise, no return figure, and the order check stays sealed.
"""
from __future__ import annotations

from config import MOMENTUM_WEIGHTS
from ui import ui_scans as S
from ui.scan_guide import _plain, _section_md, _sections_html

MOMENTUM_PARTS = {"relative_strength": "relative strength", "trend_quality": "trend quality", "breakout_proximity": "breakout proximity",
                  "sector_leadership": "sector leadership", "volume_confirm": "volume"}


def momentum_words() -> str:
    """Momentum's five parts and weights, largest first, as config has them."""
    parts = sorted(MOMENTUM_WEIGHTS.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{MOMENTUM_PARTS[key]} {weight:.0%}" for key, weight in parts)


SECTIONS = [
    ("What RAM is", [
        "RAM is a risk-adjusted momentum score. It asks which stocks have risen the most **for the ups and downs it took to get "
        "there**. Of two stocks that rose by the same amount in a year, the one that climbed smoothly scores higher than the one that swung "
        "wildly.",
        "It is worked out every trading evening from NSE's own closing prices, not the data vendor's. It is separate from PRISM's "
        "Momentum score, its Score and its Rank.",
    ]),
    ("How it is worked out", [
        "- **Five look-backs.** 1, 3, 6, 9 and 12 calendar months back from that evening.",
        "- **Return for the risk taken.** In each look-back, the stock's log return divided by how much its daily returns swung: "
        "their standard deviation times the square root of the days in the look-back.",
        "- **Against the other stocks.** Each look-back becomes a z-score across the ranked stocks: how far above or below their "
        "average the stock stands. Extremes are first pulled in to ±3 standard deviations, and no z-score passes ±3.",
        "- **Blended.** 1 month 10%, 3 months 30%, 6 months 30%, 9 months 20%, 12 months 10%. The result is RAM.",
        "- **Ranked.** RAM Rank 1 is the highest RAM, and ties share a rank. RAM % is the same order as a percentile.",
        "Prices before a split, bonus or rights issue are put on that evening's basis with NSE's own list of corporate actions, so "
        "a split is not read as a fall. Dividends are not added back.",
    ]),
    ("Which stocks get a rank", [
        "A stock must pass every step, in this order. The counts are from the first evening, 5 Nov 2026.",
        ("table", ("Step", "Stocks left"), [
            ("In NSE's equity list", "2,629"),
            ("On the main board: SME stocks are never ranked", "2,620"),
            ("Trades ₹1 Cr or more a day, averaged over 20 sessions", "1,623"),
            ("Has a full year of prices, so all five look-backs exist", "1,417"),
            ("No doubt over its prices or a corporate action", "1,412"),
        ]),
        "That evening 1,245 of the 2,692 stocks on PRISM's sheet had a rank.",
        "**A dash does not mean a weak stock.** It means RAM did not rank it, for one of these reasons:",
        "- The sheet lists it under its BSE code, and RAM reads NSE's files only.",
        "- It trades under ₹1 Cr a day.",
        "- It has under a year of prices.",
        "- A check doubted its data, or NSE lists a rights issue or demerger that RAM cannot price.",
        "- It is an SME stock.",
    ]),
    ("How to read the numbers", [
        "- **RAM Rank** — the stock's place among the ranked stocks, about 1,400 of them. Lower is stronger, and 1 is the strongest.",
        "- **RAM %** — the same place as a percentile: 100 is the strongest and 50 the middle.",
        "- **RAM** — the raw score. 0 is about average, and the strongest stocks sit above 2.",
        "- **RAM Date** — the evening the numbers are from, shown as \"RAM of 5 Nov\". If an evening's RAM could not be made, the "
        "columns stay blank rather than show an older day's number as today's.",
        "**Ranks move a lot.** Replayed on past prices at 13 month-ends, Sep 2025 to Sep 2026, on stocks still listed: of those ranked 20 or better at a "
        "month-end, 44% were still 20 or better a month later, and 10% had fallen beyond 150.",
    ]),
    ("RAM and PRISM's Momentum", [
        ("table", ("", "RAM", "Momentum"), [
            ("What it reads", "Price return for the risk taken, over five look-backs", f"Five parts: {momentum_words()}"),
            ("Its prices", "NSE's own closes", "The data vendor's columns"),
            ("Which stocks", "About 1,400 liquid NSE main-board stocks", "Every stock on the sheet"),
            ("Its scale", "Rank 1 is the strongest", "0 to 100, higher is stronger"),
            ("What reads it", "No score or list: only the order check on the cards", "PRISM's Score and the scans' recipes"),
        ]),
        "A stock can stand high on one and not the other, because they ask different questions. Which order works better is not "
        "known yet: no test on data nobody had seen has finished.",
    ]),
    ("Where you see it", [
        f"- **🎯 Scans** — every list opens on the {S.RAM_VIEW} table, by RAM Rank, strongest first, no RAM last. 📈 Technical shows "
        "RAM Rank beside the technical readings.",
        f"- **🔍 Deep Scanner** — a {S.RAM_VIEW} view of the whole market, RAM Rank in 📈 Technical, and a RAM Rank ↑ sort in the other "
        "views.",
        "- **🔬 The Tear-Sheet** — its own RAM RANK row, with RAM % and the date.",
        "- **📥 The downloads** — the Deep Scanner's CSV and the full download carry the numbers.",
    ]),
    ("What it does not do", [
        "- It is display only. Every Score, Rank, verdict, flag and list is the same with or without it, and no scan's recipe reads it.",
        f"- The order check on the 🚀 and 🌬️ cards is built on RAM Rank{', and its figures stay sealed for now' if S.ORDER_SEALED else ''}: see 🎯 Scans.",
    ]),
    ("What it cannot tell you", [
        "- **It only reads price.** Nothing here looks at the business, its results, its debt or what it costs.",
        "- **It looks back.** A high rank says what the price has done, not what it will do. Strong stocks can fall fast and "
        "together, and past swings do not limit future ones.",
        "- **It has no record here yet.** Its first evening was 5 Nov 2026, so no test has yet followed its ranks on data nobody had "
        "seen. The month-ends above are a replay on past prices, not a record.",
        "A list narrows the market; the decision stays yours.",
    ]),
]


def reference_html(query: str = "") -> str:
    """The Reference tab's 🔢 RAM page. No query: every section. A query: only the sections holding EVERY word of it (token-AND, as
    in the other corpora); "" when nothing matches."""
    tokens = (query or "").lower().split()
    kept = [sec for sec in SECTIONS if all(tok in f"{sec[0]}\n{_plain(sec[1])}".lower() for tok in tokens)]
    return _sections_html(kept)


def markdown() -> str:
    """The same page for the Reference download."""
    out = ["## 🔢 RAM", ""]
    for title, items in SECTIONS:
        out += _section_md(title, items, "###")
    return "\n".join(out)
