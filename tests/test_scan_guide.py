"""
test_scan_guide.py — the 🎯 Scans guide (2026-10-14): every scan explained in plain words, in the Reference
tab, its Markdown download and the Scans tab — and an explanation never outlives its scan.

The user asked that the guide stay right when a scan changes or a new one is added. So ui/scan_guide.py
writes two kinds of text, and this file pins both:
- FACTS the code already holds (the one-line recipe, the table order, the comparison index, the record's
  numbers and timing, the costs, the Holding-20 reading, the order check, why a 🌱 stock left, the
  candidates being measured) are written FROM the code — section 3 changes the code under the guide and
  watches the text follow, and section 4 reads the record from the records file;
- each scan's EXPLANATION is stamped with the scan it was written for (its recipe fingerprint plus every
  setting the guide explains) — section 1 fails the moment a recipe, a score it reads or one of those
  settings changes, or a scan is added or removed, until the explanation is re-read and re-stamped: the
  rule the scan record already follows (tests/test_scans.py, "a record never outlives its recipe").

Run with: pytest tests/test_scan_guide.py -v
"""
import ast
import html as _html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import BENCHMARK_INDICES  # noqa: E402
from ui import scan_guide as G  # noqa: E402
from ui import ui_scans as S  # noqa: E402

_BY_KEY = {s["key"]: s for s in S.SCANS}
_RECORDS = S.load_records()


def _text(h: str) -> str:
    """Rendered HTML as the reader sees it: tags out, entities decoded, whitespace collapsed."""
    return " ".join(_html.unescape(re.sub(r"<[^>]+>", " ", h)).split())


def _rec(key: str) -> dict:
    """A private copy of a scan's committed record, safe to edit."""
    return json.loads(json.dumps(_RECORDS["scans"][key]))


def _guide(key: str, rec=None, state="measured", **changes) -> str:
    """One scan's guide as text, optionally for a changed copy of the scan."""
    return _text(G.guide_html(dict(_BY_KEY[key], **changes), rec, state))


# ── 1. Every scan is explained, and no explanation outlives its scan ─────────────────────────────────
def test_every_scan_has_a_guide_and_none_is_orphaned():
    assert sorted(G.GUIDES) == sorted(_BY_KEY), (
        "a scan without a guide, or a guide for a scan that is gone — write or remove it in ui/scan_guide.py")


def test_a_guide_never_outlives_its_scan():
    stale = [k for k, s in _BY_KEY.items() if G.GUIDES[k]["written_for"] != G.guide_stamp(s)]
    assert not stale, (
        f"{stale}: the scan changed since its guide was written. Re-read every sentence of its GUIDES entry "
        f"against the new recipe and settings (re-measure every number it quotes, by the method named beside "
        f"it), then set written_for: " + ", ".join(f"{k}={G.guide_stamp(_BY_KEY[k])}" for k in stale))


def test_the_stamp_follows_the_recipe_and_every_setting_the_guide_explains():
    play = _BY_KEY["playbook"]
    base = G.guide_stamp(play)
    other = dict(play, mask=S.compounder_candidates, uses=_BY_KEY["compounder"]["uses"], inputs=())
    assert G.guide_stamp(other) != base, "another recipe needs another explanation"
    for key, value in (("benchmark", "NSE:CNX500"), ("order", None), ("recipe", "other words"),
                       ("name", "Renamed"), ("emoji", "🧪"), ("designed_on", "2027-01-01"),
                       ("horizon_days", 90), ("start_month", 4), ("fixed", ()), ("extra", ("_cycle",))):
        assert G.guide_stamp(dict(play, **{key: value})) != base, f"changing {key} must need a re-read"
    assert G.guide_stamp(dict(play, widths={"name": 999})) == base, "a column width explains nothing"
    assert re.fullmatch(r"[0-9a-f]{16}", base)


# ── 1b. A backtest's method never outlives its study (2026-10-14, the user's ask: "so in future we can
# check and do it again") ─────────────────────────────────────────────────────────────────────────────
# The StockScans columns, formulas and traps the 🌱 study used — verified 2026-10-14 against the 🌱 commit
# (e4193c5), the record's own notes, tools/compounder_years.py and the 2026-10-03 code it reproduced.
_METHOD_WORDS = ("PAT 3 Years Back", "PAT 4 Years Back", "Revenue 3 Years Back", "Revenue 4 Years Back",
                 "OPM 3 Years Back", "OPM 5 Years Back", "Returns 1Y", "Returns 2Y", "Returns 3Y", "Listed Days",
                 "FY21 → FY23", "median", "26%", "(1 + R3)³ ÷ (1 + R2)² − 1", "(1 + R2)² ÷ (1 + R1) − 1",
                 "Returns Vs Nifty 500 3Y", "per-year", "tools/compounder_years.py")


def test_a_backtest_method_never_outlives_its_study():
    recs = _RECORDS["scans"]
    for key, g in G.GUIDES.items():
        study, method = (recs.get(key) or {}).get("study"), g.get("backtest")
        if study is None:
            assert method is None, f"{key}: a backtest method, but its record holds no study"
            continue
        assert method and method.get("items"), f"{key}: its record is a study — say how it was run"
        assert method["written_for"] == G.study_stamp(study), (
            f"{key}: the study changed since its method was written. Re-read every line of GUIDES['{key}']"
            f"['backtest'] against the new study, then set written_for={G.study_stamp(study)}")


def test_the_study_stamp_follows_every_number_and_note():
    study = _rec("compounder")["study"]
    base = G.study_stamp(study)
    assert re.fullmatch(r"[0-9a-f]{16}", base) and G.study_stamp(json.loads(json.dumps(study))) == base
    assert G.study_stamp(dict(reversed(list(study.items())))) == base, (
        "the weekly job may re-save the records with its keys in another order — that is not a new study")
    for change in ({"n_match": 183}, {"source": "another export"}, {"definition_note": "other legs"},
                   {"years": study["years"][:2]}):
        assert G.study_stamp(dict(study, **change)) != base, change


def test_the_method_names_every_column_formula_and_trap():
    text = " ".join(G.GUIDES["compounder"]["backtest"]["items"])
    missing = [w for w in _METHOD_WORDS if w not in text]
    assert not missing, f"the 🌱 method never mentions {missing}"
    assert G.GUIDES["compounder"]["backtest"]["items"][0].startswith("**"), "it opens on a named step"


def test_the_method_shows_under_the_record_and_only_for_a_study():
    comp = _guide("compounder", _rec("compounder"))
    assert "How the backtest was run" in comp and comp.index("Its record") < comp.index("How the backtest was run")
    assert comp.index("How the backtest was run") < comp.index("What it cannot see")
    for key in ("playbook", "leaders"):
        assert "How the backtest was run" not in _guide(key, _rec(key)), f"{key} is measured on snapshots"
    t = _text(G.reference_html(_RECORDS, "Returns 2Y"))
    assert "Compounder Candidates" in t and "How the backtest was run" in t, "the Reference search finds it"
    assert "#### How the backtest was run" in G.guide_markdown(_RECORDS), "the download carries it"


def test_every_guide_fills_every_part():
    for key, g in G.GUIDES.items():
        for part in G.PROSE_PARTS:
            assert g.get(part), f"{key}: {part} is empty"
        # _plain reads a table's cells too (a part may hold one since 2026-10-14)
        assert len(G._plain(g["recipe"])) >= 300, f"{key}: the recipe is not explained in detail"
        assert len(G._plain(g["enter_leave"])) >= 200, f"{key}: entering and leaving is not explained"


# ── 1c. 🚀: every way out, measured (2026-10-14, the user's question "when does a stock get out?") ───────
_WAYS_OUT = ("Breakout", "Momentum", "quarter", "Improvement", "stop")   # one per condition of the recipe


def test_the_playbook_shows_every_way_out_measured():
    items = G.GUIDES["playbook"]["enter_leave"]
    tables = [i for i in items if isinstance(i, tuple)]
    assert len(tables) == 1, "one table of the ways out"
    _, headers, rows = tables[0]
    assert len(headers) == 4 and all(len(r) == 4 for r in rows)
    assert len(rows) == len(_WAYS_OUT), "one way out per condition of the recipe — no more, no fewer"
    for word in _WAYS_OUT:
        assert sum(word in r[0] for r in rows) == 1, f"the way out through {word!r} is missing or doubled"
    for r in rows:
        assert re.fullmatch(r"\d+%", r[2]) and re.fullmatch(r"\d+%", r[3]), r
    text = G._plain(items)
    assert "The stop is the last line" in text and "about 15%" in text and "17.5%" in text, (
        "measured, the stop removes a stock last — the guide must not lead with it")
    assert text.index("Results season reshuffles the list") > text.index("The stop is the last line")


def test_the_quoted_score_parts_are_the_engines():
    """THE TRUTH PIN for the 🚀 recipe's numbers: every weight the guide quotes is the engine's own (config),
    "best within a month" is the freshest age zone, and the stop is the vendor column PRISM maps."""
    from config import BREAKOUT_SIGNALS, HIGH_AGE_ZONES, MOMENTUM_WEIGHTS
    from core.data_engine import TECHNICAL_COLS
    text = " ".join(G.GUIDES["playbook"]["recipe"])
    for phrase, w in (("relative strength against the market (", MOMENTUM_WEIGHTS["relative_strength"]),
                      ("trend quality (", MOMENTUM_WEIGHTS["trend_quality"]),
                      ("how close the price is to a breakout (", MOMENTUM_WEIGHTS["breakout_proximity"]),
                      ("sector leadership (", MOMENTUM_WEIGHTS["sector_leadership"]),
                      ("volume confirmation (", MOMENTUM_WEIGHTS["volume_confirm"]),
                      ("from the 52-week high (", BREAKOUT_SIGNALS["52wh_distance"]),
                      ("from the 13-week high (", BREAKOUT_SIGNALS["13wh_distance"]),
                      ("high was made (", BREAKOUT_SIGNALS["52wh_days"]),
                      ("breakout window (", BREAKOUT_SIGNALS["breakout_window"]),
                      ("all-time high (", BREAKOUT_SIGNALS["ath_distance"])):
        assert f"{phrase}{w:.0%}" in text, f"the guide quotes {phrase!r} with a weight the engine does not use"
    assert HIGH_AGE_ZONES["breakout_fresh"] == {"min": 0, "max": 31, "score": 100}, "'best within a month'"
    assert TECHNICAL_COLS.get("VSTOP 14W 2.5") == "vstop_value", "the stop the guide names is the one PRISM reads"


def test_the_ways_out_render_and_are_searchable():
    h = G.guide_html(_BY_KEY["playbook"], _rec("playbook"), "measured")
    section = h[h.index("When a stock enters and leaves"):h.index("How the table is ordered")]
    assert "<table" in section and "Results season" in _text(section)
    # Layout (seen 2026-10-14 at 1536px): the labels wrapped onto two lines while the share columns stood
    # wide around "75%". A label column stays on one line; an all-number column is as narrow as its numbers.
    first_cell = re.search(r"<td style=\"([^\"]*)\">", section).group(1)
    assert "white-space:nowrap" in first_cell, "the way-out label stays on one line"
    share = re.search(r"<td style=\"([^\"]*)\">\d+%</td>", section).group(1)     # any share cell (re-measured)
    assert "width:1%" in share and "text-align:right" in share and "white-space:nowrap" in share
    text_cell = re.search(r"<td style=\"([^\"]*)\">The price slipped back", section).group(1)
    assert "nowrap" not in text_cell and "width:1%" not in text_cell, "a text column wraps and takes the rest"
    assert "overflow-x:auto" in section, "a narrow screen scrolls the table sideways instead of breaking it"
    assert "| Way out |" in G.guide_markdown(_RECORDS), "the download carries the table"
    t = _text(G.reference_html(_RECORDS, "fresher highs"))
    assert "Playbook Shortlist" in t, "a word that only the table holds is found by the Reference search"
    stop = " ".join(G.GUIDES["playbook"]["recipe"])
    assert "VSTOP 14W 2.5" in stop and "52-week high" in stop and "13-week high" in stop


# The words each recipe's conditions are explained with — checked against ui/ui_scans.py on 2026-10-14.
# Change them only after re-reading the recipe.
_CONDITIONS = {
    "playbook": ("Momentum", "Breakout", "top third", "Strong Quarter", "15%", "20%", "Improvement",
                 "top half", "stop"),
    "leaders": ("at least 5 stocks", "within 5%", "52-week high", "strong", "top five", "every stock"),
    "compounder": ("profit growth", "top 40%", "operating margin", "more than 10%", "faster than sales",
                   "loss year"),
}


def test_each_guide_explains_every_condition_of_its_recipe():
    assert sorted(_CONDITIONS) == sorted(_BY_KEY), "a new scan needs its conditions listed here"
    for key, words in _CONDITIONS.items():
        text = " ".join(G.GUIDES[key]["recipe"]).lower()
        missing = [w for w in words if w.lower() not in text]
        assert not missing, f"{key}: the recipe explanation never mentions {missing}"


# ── 2. Where the guide shows ─────────────────────────────────────────────────────────────────────────
def test_the_reference_section_shows_every_scan_in_order_with_the_overview_on_top():
    h = G.reference_html(_RECORDS)
    t = _text(h)
    first = [t.index(s["name"]) for s in S.SCANS]
    assert first == sorted(first), "the overview lists the scans in their own order"
    assert h.count("<details") == len(S.SCANS) + 1, "one closed block per scan, and one for how scans work"
    assert "<details open" not in h, "browsing opens nothing by itself"
    assert t.index("Which scan for what") < t.index(G.GENERAL_TITLE)
    for title, _ in G.general_sections():
        assert title in t, title


def test_search_narrows_to_the_matching_sections_and_misses_cleanly():
    t = _text(G.reference_html(_RECORDS, "volatility stop"))
    assert "Playbook Shortlist" in t and "The recipe, condition by condition" in t
    assert "Compounder Candidates" not in t, "🌱's guide never mentions the Volatility Stop"
    assert "<details open" in G.reference_html(_RECORDS, "volatility stop"), "a search result shows open"
    assert G.reference_html(_RECORDS, "sector zzzz-no-such-word") == "", "every word must match"
    assert G.reference_html(_RECORDS, "zzzz-no-such-word") == ""


def test_the_download_carries_the_whole_guide():
    from ui.ui_reference import build_reference_markdown
    md = build_reference_markdown({"T": "d"}, {"C": [("l", "e")]}, {}, scans=G.guide_markdown(_RECORDS))
    assert "## 🎯 Scans" in md
    for s in S.SCANS:
        assert f"### {s['emoji']} {s['name']}" in md
    for title, _ in G.general_sections():
        assert f"#### {title}" in md or f"### {title}" in md, title
    assert "<" not in G.guide_markdown(_RECORDS), "the download is Markdown, not HTML"


def test_app_wires_the_guide_into_the_reference_and_the_scans_tab():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "scan_reference_html(_scan_recs, q)" in src
    assert "scans=scan_guide_markdown(_scan_recs)" in src, "the download no longer carries the guide"
    tree = ast.parse(src)
    block = ast.unparse(next(n for n in tree.body if isinstance(n, ast.With)
                             and ast.unparse(n.items[0].context_expr) == "tabs[1]"))
    assert "📖 How it works" in block
    assert "scan_guide_html(_sc_sel, _sc_rec_sel, _sc_state_sel)" in block, (
        "the Scans tab shows the SELECTED scan's guide, with the record state its card uses")
    assert block.index("📊 Full record") < block.index("📖 How it works"), "it sits beside the full record"


# ── 3. The facts are written FROM the code ───────────────────────────────────────────────────────────
def test_the_table_order_is_written_from_the_scan():
    """In all three places it is stated: the scan's order section, its "Listed by" line, and the overview
    table's row (a mutation that typed the line and the row survived the first draft of this test)."""
    for key in ("playbook", "leaders"):
        t = _guide(key)
        assert "By Momentum, high to low" in t and "Listed by: Momentum, high to low" in t, key
    comp = _guide("compounder")
    assert "By PRISM Rank" in comp and "Listed by: PRISM Rank" in comp and "Momentum, high to low" not in comp
    flipped = _guide("playbook", order=None)
    assert "By PRISM Rank" in flipped and "Listed by: PRISM Rank" in flipped and "By Momentum" not in flipped
    table = dict(G.general_sections())["Which scan for what"][0][2]
    assert [row[3] for row in table] == [G._order_words(s) for s in S.SCANS] == [
        "Momentum, high to low" if S.momentum_ordered(s) else "PRISM Rank" for s in S.SCANS]


def test_the_index_the_dates_and_the_recipe_line_are_written_from_the_scan():
    for s in S.SCANS:
        t = _guide(s["key"], _rec(s["key"]))
        assert BENCHMARK_INDICES[s["benchmark"]] in t, s["key"]
        assert s["recipe"] in t, s["key"]
        assert S._first_unseen_due(s) in t, s["key"]
    moved = _guide("playbook", benchmark="NSE:CNX500")
    assert "Compared with Nifty 500" in moved and "Nifty500 Momentum 50" not in moved
    assert "starting each October" in _guide("compounder") and "every month-end" in _guide("playbook")


def test_the_measurement_settings_are_read_not_typed(monkeypatch):
    monkeypatch.setattr(S, "HOLD_K", 25)
    monkeypatch.setattr(S, "HOLD_MAX_PRICE", 7000.0)
    monkeypatch.setattr(S, "NET_COST_PP", 1.5)
    monkeypatch.setattr(S, "ORDER_RULE", dict(S.ORDER_RULE, windows=6, min_avg_pct=75.0))
    t = _text(G.reference_html(_RECORDS))
    for s in ("Holding 25", "top 25 by momentum", "₹7,000", "1.5-point round trip", "first 6 separate",
              "at least 75%"):
        assert s in t, s


def test_the_lists_the_guide_explains_are_read_not_typed(monkeypatch):
    real = " ".join(map(str, dict(G.general_sections())["Being measured — not scans yet"]))
    for name, rule in S.CANDIDATES:
        assert f"**{name}** — {rule}" in real, f"candidate {name} is missing from the guide"
    monkeypatch.setattr(S, "CANDIDATES", (("Test candidate", "a rule under test"),))
    monkeypatch.setattr(S, "EXIT_REASONS", dict(S.EXIT_REASONS, new=("🟣", "a new reason", "why it left")))
    monkeypatch.setattr(S, "MEASURED_FLAGS", dict(S.MEASURED_FLAGS, rf_test="Test flag"))
    sections = dict(G.general_sections())
    cands = " ".join(map(str, sections["Being measured — not scans yet"]))
    assert "Test candidate" in cands and "WC Bullies" not in cands
    assert "Test flag" in " ".join(map(str, sections["How to read the table"]))
    assert "a new reason" in _guide("compounder") and "a new reason" not in _guide("playbook")


def test_every_table_view_and_every_measured_flag_is_explained():
    assert sorted(G.VIEW_WHAT) == sorted(S.SCAN_VIEWS), "a new or renamed view needs its words here"
    assert sorted(G.FLAG_WHAT) == sorted(S.MEASURED_FLAGS), "a new measured flag needs its plain words here"
    t = " ".join(map(str, dict(G.general_sections())["How to read the table"]))
    for view in S.SCAN_VIEWS:
        assert view in t, view
    for col, label in S.MEASURED_FLAGS.items():
        assert f"**⚠ {label}** — {G.FLAG_WHAT[col]}" in t, label
    if "market_category" in S.TABLE_COLUMNS:                 # the Core column added 2026-10-14
        assert "**Category** — the company's market-cap band" in t, "the guide explains the Category column"


# ── 4. The record is read from the records file, honestly ────────────────────────────────────────────
def test_the_design_record_is_read_from_the_records_file():
    for key in ("playbook", "leaders"):
        rec = _rec(key)
        ws = rec["windows"]
        t = _guide(key, rec)
        ex, bt = [w["median_excess_pp"] for w in ws], [w["beat_pct"] for w in ws]
        assert f"{min(ex):+.1f} to {max(ex):+.1f} points" in t, key
        assert f"{min(bt)}–{max(bt)}%" in t, key
        net = [w["net"] for w in ws]
        assert f"{min(n['median_excess_pp'] for n in net):+.1f} to {max(n['median_excess_pp'] for n in net):+.1f}" in t
        k = S.non_overlapping(ws)
        assert f"they are {k} separate stretches" in t, "overlapping windows must be called what they are"
        assert "In every window" in t, "every committed design window beat the market's median stock"


def test_every_window_is_claimed_only_when_every_window_beat():
    rec = _rec("playbook")
    rec["windows"][0]["median_excess_pp"] = -1.2
    t = _guide("playbook", rec)
    assert "In every window" not in t and "-1.2 to" in t


def test_the_study_record_is_read_from_the_records_file():
    rec = _rec("compounder")
    st_ = rec["study"]
    t = _guide("compounder", rec)
    assert f"{st_['match_cagr_pct']:+.1f}% a year against {st_['rest_cagr_pct']:+.1f}%" in t
    assert f"{st_['n_match']:,} against {st_['n_rest']:,}" in t
    for y in st_["years"]:
        assert f"{y['label']} {y['match_excess_pp']:+.1f} (the rest {y['rest_excess_pp']:+.1f})" in t
    assert st_["caveats"] in t and st_["cleanest"]["label"] in t


def test_no_numbers_from_a_record_nobody_can_vouch_for():
    rec = _rec("playbook")
    best = f"{max(w['median_excess_pp'] for w in rec['windows']):+.1f}"
    for state, words in (("stale", "measured again"), ("unverified", "could not be checked"),
                         ("unmeasured", "Not measured yet")):
        t = _guide("playbook", rec, state)
        assert best not in t and "None has finished" not in t and words in t, state


def test_the_unseen_record_and_the_order_check_follow_the_records():
    rec = _rec("playbook")
    assert "None has finished yet" in _guide("playbook", rec)
    assert "So far 0 of" in _guide("playbook", rec)
    hold = dict(k=20, n=60, momentum_pct=80, rank_pct=50)
    chain = [("2026-10-31", "2026-11-30"), ("2026-11-30", "2026-12-31"), ("2026-12-31", "2027-01-29"),
             ("2027-01-29", "2027-02-26")]
    rec["out_of_sample"] = [dict(start=a, end=b, final=True, hold=hold) for a, b in chain]
    t = _guide("playbook", rec)
    assert "4 finished so far" in t and "Decided: momentum order held" in t
    assert "Order check" not in _guide("compounder", _rec("compounder")), "🌱 is not order-checked"


# ── 5. Tone, safety, purity ──────────────────────────────────────────────────────────────────────────
_PROMISE = re.compile("(will (?:outperform|rise|deliver|beat|double)|guaranteed returns?|never fails|"
                      "is a buy|should buy|sure[- ]?shot returns?)", re.I)


def test_the_guide_promises_nothing_and_never_says_buy():
    everything = _text(G.reference_html(_RECORDS)) + "\n" + G.guide_markdown(_RECORDS)
    assert not _PROMISE.search(everything), _PROMISE.search(everything)
    assert not re.search(r"\bbuys?\b", everything, re.I), "PRISM narrows; the user decides"


def test_the_module_is_pure_and_escapes_what_it_shows():
    path = ROOT / "ui" / "scan_guide.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    st_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "st"]
    assert not st_calls and "import streamlit" not in path.read_text(encoding="utf-8")
    h = G.guide_html(dict(_BY_KEY["playbook"], recipe="x <script>alert(1)</script> & y"), None, "unmeasured")
    assert "&lt;script&gt;" in h and "<script>" not in h
    assert G.reference_html(_RECORDS) == G.reference_html(_RECORDS), "the same input renders the same page"
