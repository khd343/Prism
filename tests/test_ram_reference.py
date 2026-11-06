"""
test_ram_reference.py — the 🔢 RAM page of the Reference tab (2026-11-05, the operator: "write about RAM, all time best").

RAM is computed outside PRISM, by the RAM job in stockscans-sync, so its facts cannot be imported here: they are typed in
ui/ram_guide.py from that job's code (7a9a867) and its record for 5 Nov 2026, and pinned below. What PRISM itself holds (Momentum's
weights, the view's name) is read from the code. No word of action, no person and no promise, as on every RAM surface.
"""
import ast
import html
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

from config import MOMENTUM_WEIGHTS  # noqa: E402
from ui import ram_guide as R  # noqa: E402
from ui import ui_scans as S  # noqa: E402
from ui.ui_reference import build_reference_markdown  # noqa: E402

APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()


def _page() -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", R.reference_html())))


def test_every_section_is_on_the_page_in_a_readers_order():
    want = ["What RAM is", "How it is worked out", "Which stocks get a rank", "How to read the numbers", "RAM and PRISM's Momentum",
            "Where you see it", "What it does not do", "What it cannot tell you"]
    assert [title for title, _ in R.SECTIONS] == want
    at = [_page().upper().index(t.upper()) for t in want]
    assert at == sorted(at)


@pytest.mark.parametrize("fact", [
    "1, 3, 6, 9 and 12 calendar months",
    "1 month 10%, 3 months 30%, 6 months 30%, 9 months 20%, 12 months 10%",
    "standard deviation",
    "±3",
    "NSE's own closing prices",
    "Dividends are not added back",
    "₹1 Cr or more a day",
    "20 sessions",
    "full year of prices",
    "SME stocks are never ranked",
])
def test_the_formula_and_the_universe_match_the_ram_jobs_code(fact):
    assert fact in _page(), fact


def test_the_first_evenings_counts_are_the_jobs_record_for_5_nov():
    rows = next(item for item in dict(R.SECTIONS)["Which stocks get a rank"] if isinstance(item, tuple))[2]
    assert [r[1] for r in rows] == ["2,629", "2,620", "1,623", "1,417", "1,412"]
    assert "5 Nov 2026" in _page() and "1,245 of the 2,692" in _page()


def test_momentums_weights_are_read_from_the_engine_not_typed():
    words = R.momentum_words()
    for key, part in R.MOMENTUM_PARTS.items():
        assert f"{part} {MOMENTUM_WEIGHTS[key]:.0%}" in words, key
    assert set(R.MOMENTUM_PARTS) == set(MOMENTUM_WEIGHTS), "a new part of Momentum needs its words here"
    assert words in _page()


def test_a_dash_is_said_not_to_mean_weak_and_each_cause_is_named():
    t = _page()
    assert "does not mean a weak stock" in t
    for cause in ("BSE code", "under ₹1 Cr a day", "under a year of prices", "rights issue or demerger", "SME"):
        assert cause in t, cause


def test_it_says_where_it_shows_from_the_apps_own_names():
    t = _page()
    assert S.RAM_VIEW in t and "RAM Rank ↑" in APP and "RAM Rank ↑" in t
    for words in ("🎯 Scans", "🔍 Deep Scanner", "🔬 The Tear-Sheet", "📈 Technical"):
        assert words in t, words


def test_it_is_said_to_be_display_only_and_to_have_no_record():
    t = _page()
    assert "display only" in t.lower() and "no scan's recipe reads it" in t
    assert "first evening was 5 Nov 2026" in t and "decision stays yours" in t


_FORBIDDEN = re.compile(r"\b(buy|buys|buying|bought|sell|sells|selling|sold|hold|holds|avoid|kite|operator|paresh)\b", re.I)
_PROMISE = re.compile(r"(will (?:outperform|rise|deliver|beat|double)|guaranteed|never fails|sure[- ]?shot)", re.I)
_RETURN_FIGURE = re.compile(r"[+-]\d+(?:\.\d+)?%|\b(?:rose|risen|gained|up|returned|fell|fallen|lost|down)\b[^.%]{0,20}\d+(?:\.\d+)?%", re.I)


def test_no_word_of_action_no_person_no_promise_and_no_return_figure():
    for text in (_page(), R.markdown()):
        assert not _FORBIDDEN.search(text), _FORBIDDEN.search(text)
        assert not _PROMISE.search(text), _PROMISE.search(text)
        assert not _RETURN_FIGURE.search(text), _RETURN_FIGURE.search(text)


def test_the_page_never_reads_a_replay_as_a_record_and_says_what_reads_ram():
    t = _page()
    assert "Replayed on past prices at 13 month-ends" in t and "a replay on past prices, not a record" in t
    assert "no result exists" not in t and "has measured nothing" not in t, "looks at past prices exist; only an unseen record does not"
    assert "Nothing: it is display only" not in t and "No score or list: only the order check on the cards" in t


def test_the_page_speaks_of_the_seal_only_while_it_holds():
    assert ("stay sealed for now" in _page()) == bool(S.ORDER_SEALED)
    assert "S.ORDER_SEALED" in open(R.__file__, encoding="utf-8").read(), "the words follow the flag, as the other surfaces do"


def test_the_page_prints_no_figure_of_the_sealed_order_check():
    t = _page()
    assert "sealed" in t and "Decided" not in t and "random picks" not in t


def test_search_keeps_only_sections_holding_every_word():
    hit = R.reference_html("liquid ₹1 Cr") or R.reference_html("₹1 Cr sessions")
    assert hit and "WHICH STOCKS GET A RANK".lower() in hit.lower() and "What it cannot tell you".lower() not in hit.lower()
    assert R.reference_html("zzz-no-such-word") == ""
    assert R.reference_html("") == R.reference_html(), "no query: the whole page"


def test_the_download_carries_the_same_page():
    md = R.markdown()
    assert md.startswith("## 🔢 RAM")
    for title, _ in R.SECTIONS:
        assert f"### {title}" in md, title
    whole = build_reference_markdown({"T": "d"}, {}, {}, as_of="2026-11-05", ram=md)
    assert "## 🔢 RAM" in whole and "1 month 10%" in whole


def test_the_tab_passes_the_page_to_the_download_and_runs_it_in_both_modes():
    assert "ram=ram_markdown()" in APP
    tree = ast.parse(APP)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_show_ram")
    assert "ram_reference_html(q)" in ast.unparse(fn)


def test_the_page_says_what_the_tear_sheets_row_shows():
    assert "its own 🔢 RAM row: RAM Rank, RAM % and the date. Green at RAM % 70 or above and red at 30 or below, as the RS chip beside it." in R.markdown()
