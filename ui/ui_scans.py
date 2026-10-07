"""
🎯 Scans — measured stock lists, one definition each (2026-10-06).

A screener usually shows a list with no evidence. Each scan here is shown beside ITS OWN record: how
its stocks did against the market on PRISM's saved snapshots. Only recipes that beat the market in
every test window get a card (docs/december-2026-plan.md section 8); the rest wait as "being
measured" until December's review.

THREE RULES, each pinned in tests/test_scans.py:
1. ONE DEFINITION. Each recipe is one function below. The tab calls it on the whole market and the
   local record tool (tools/scan_records.py) imports the same function, so a record always describes
   exactly the stocks on screen.
2. A RECORD NEVER OUTLIVES ITS RECIPE. scan_records.json stores a fingerprint of the logic it
   measured (the recipe plus every engine function it reads). Change the logic and the card shows no
   numbers until it is re-measured.
3. NOTHING HIDDEN, NOTHING PROMISED. A flagged stock stays in its list with the flag named; a missing
   value never passes a condition; the wording is "matched / record / provisional".

Each window also carries what the headline leaves out (2026-10-07): how many matches had no return
(no price at the end: delisted, suspended or missing) or were dropped as corporate actions, and a NET
record: only matches trading NET_MIN_CR a day or more, each charged NET_COST_PP for the round trip.
The card labels the design-period record "in-sample" in so many words: the recipe was chosen on it.

Display only, like ui_movers: no widgets here. app.py owns the scan choice, the liquidity floor and
the row click.
"""
import ast
import hashlib
import html
import inspect
import json
import textwrap
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from config import COLORS
from core.forward_returns import MIN_HORIZON_DAYS
from core.scoring_engine import _compute_improvement_score, compute_momentum_score, strong_quarter
from ui.ui_group_views import MEASURES, group_measures

RECORDS_PATH = Path(__file__).with_name("scan_records.json")
# The net record (2026-10-07): what a buyer could have kept. Only matches trading at least NET_MIN_CR
# a day count, and each is charged NET_COST_PP for the round trip (0.5% in + 0.5% out: taxes, spread,
# impact). The benchmark (the market median) is charged nothing, so the cost side errs conservative.
NET_MIN_CR = 2.0
NET_COST_PP = 1.0
RECORD_VERSION = 2
# The day the recipes were chosen. Every window that STARTS on or after it is out of sample: data
# nobody looked at while choosing. Windows before it are the design-period record (in sample).
DESIGNED_ON = "2026-10-06"


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """The column as numbers; all-NaN when absent, so a missing input passes nothing."""
    return (pd.to_numeric(df[col], errors="coerce") if col in df.columns
            else pd.Series(np.nan, index=df.index))


def traded_value_cr(df: pd.DataFrame) -> pd.Series:
    """Average daily traded value over 20 days (average volume × price), ₹ crore; NaN when unknown. One
    definition for the list's floor and the net record."""
    return _num(df, "vol_sma_20d") * _num(df, "close_price") / 1e7


def _pct(df: pd.DataFrame, col: str) -> pd.Series:
    """Percentile rank (0-1] across the frame given — the whole market, never a filtered slice."""
    return _num(df, col).rank(pct=True)


# ── The recipes ────────────────────────────────────────────────────────────────────────────────────
def playbook_shortlist(df: pd.DataFrame) -> pd.Series:
    """Momentum and Breakout in the market's top third, a 🔥 Strong Quarter (the engine's own
    definition; a quarter that cannot be judged is not strong), Improvement in the top half, and the
    price at or above its stop. In-sample record (2026-10-06, ui/scan_records.json): +7.1 / +10.3 /
    +3.3 / +2.9pp against the market median, 61-70% of matches beating it."""
    return ((_pct(df, "momentum_score") >= 2 / 3)
            & (_pct(df, "breakout_score") >= 2 / 3)
            & (strong_quarter(df) == 1)
            & (_pct(df, "improvement_score") >= 0.5)
            & (_num(df, "dist_to_vstop") >= 0)).fillna(False).astype(bool)


def leading_sectors(df: pd.DataFrame, n: int = 5, min_stocks: int = 5) -> list:
    """The n sectors (of at least min_stocks stocks) ranked best on the 📈 Sectors tab's own two
    breadth shares: stocks within 5% of their 52-week high, and stocks reporting a strong quarter.
    Ties break on the near-high share, then the sector name, so the answer never depends on row
    order."""
    if "sector" not in df.columns:
        return []
    g = group_measures(df, "sector")
    if not {"grp_near_high", "grp_strong_quarter"} <= set(g.columns):
        return []
    g = g.join(df["sector"].value_counts().rename("_n"))
    g = g[g["_n"] >= min_stocks].dropna(subset=["grp_near_high", "grp_strong_quarter"])
    order = pd.DataFrame({
        "score": (g["grp_near_high"].rank(ascending=False) + g["grp_strong_quarter"].rank(ascending=False)).to_numpy(),
        "near": g["grp_near_high"].to_numpy(), "sector": g.index.astype(str)})
    order = order.sort_values(["score", "near", "sector"], ascending=[True, False, True])
    return order["sector"].head(n).tolist()


def leading_sector_stocks(df: pd.DataFrame) -> pd.Series:
    """Every stock in the leading sectors — the whole sector, not only its leaders. In-sample record
    (2026-10-06, ui/scan_records.json): +3.9 / +11.2 / +2.4 / +3.4pp against the market median,
    58-67% of matches beating it."""
    if "sector" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["sector"].isin(leading_sectors(df)).astype(bool)


SCANS = (
    # `fixed`: table columns the recipe itself fixes, so they would repeat one value on every row — the
    # Playbook requires a 🔥 Strong Quarter, so that column is ✓ on 100% of its stocks. Only columns
    # fixed BY CONSTRUCTION are hidden; Trend stays, because "above its stop" is not "above its 30-week
    # average" and a Stage 3 or 4 is possible (tests/test_scans.py). `widths` gives the hidden column's
    # 40px to Stock and Sector (measured 2026-10-07: names fit 79% -> 90%, sectors 51% -> 75%).
    # `inputs`: the engine functions that compute the columns a recipe reads. The fingerprint follows
    # them, and everything they call, so a re-defined score makes the record stale (2026-10-07).
    # dist_to_vstop is NOT followed: it comes from data_engine.compute_derived_signals, which computes
    # hundreds of columns, so following it would stale the record on every unrelated engine change.
    # Nor is the regime momentum boost in run_scoring_pipeline (one factor for every stock, so it
    # moves no percentile below the 100 clip).
    dict(key="playbook", emoji="🚀", name="Playbook Shortlist", mask=playbook_shortlist,
         uses=(strong_quarter, _pct, _num), inputs=(compute_momentum_score, _compute_improvement_score),
         fixed=("_strong_q",), widths={"name": 190, "sector": 130},
         recipe="Momentum and Breakout in the top third of the market, a 🔥 Strong Quarter, Improvement "
                "in the top half, and the price above its stop."),
    dict(key="leaders", emoji="🌬️", name="Leading-Sector Stocks", mask=leading_sector_stocks,
         uses=(leading_sectors, group_measures, strong_quarter),
         # It reads two of the Sectors measures, so it follows those two, not the whole table: a new
         # Sectors column must not stale this record (2026-10-07).
         values={f"MEASURES[{k}]": MEASURES[k] for k in ("grp_near_high", "grp_strong_quarter")},
         skip=("MEASURES",),
         recipe="Every stock in the 5 sectors (of 5+ stocks) with the most stocks near their 52-week "
                "high and reporting strong quarters, as ranked on 📈 Sectors."),
)

# Recorded in tools/validate.py; December decides (docs/december-2026-plan.md, sections 2d and 8).
CANDIDATES = (
    ("Supply-chain velocity", "cash conversion cycle down 15+ days and revenue up more than 15%"),
    ("WC Bullies", "3-year revenue growth above 20%, cash cycle not worse, cash flow at least profit"),
    ("J-Curve inflection", "operating margin and profit accelerating while capital work converts"),
)


# ── A record never outlives its recipe ─────────────────────────────────────────────────────────────
def _recipe_functions(scan: dict) -> list:
    return [scan["mask"], *scan.get("uses", ())]


_PROJECT_MODULES = ("config", "core", "ui")
_PLAIN = (bool, int, float, str, type(None), dict, list, tuple, set, frozenset)


def _is_project(obj) -> bool:
    return (getattr(obj, "__module__", None) or "").split(".")[0] in _PROJECT_MODULES


def _logic(fn) -> str:
    """A named function's LOGIC: its parsed code without the docstring or its own name, so a reworded
    comment or a rename keeps a record valid and a changed threshold does not. A lambda (one that sits
    inside a settings table, e.g. a Sectors measure) has no source of its own, so its compiled code
    stands in: instructions, constants and names, never line numbers."""
    if fn.__name__ == "<lambda>":
        return _code_logic(fn.__code__)
    node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
        node.body = node.body[1:]
    node.name = "_"
    return ast.dump(node)


def _code_logic(code) -> str:
    consts = [_code_logic(c) if inspect.iscode(c) else repr(c) for c in code.co_consts]
    return f"{code.co_code.hex()}|{consts}|{code.co_names}"


def _canon(obj, found: list) -> str:
    """A settings value as stable text: dicts and sets sorted (set order changes between runs), a
    function inside it reduced to its logic and followed like any other dependency."""
    if isinstance(obj, dict):
        return "{" + ",".join(sorted(f"{_canon(k, found)}:{_canon(v, found)}" for k, v in obj.items())) + "}"
    if isinstance(obj, (set, frozenset)):
        return "{" + ",".join(sorted(_canon(v, found) for v in obj)) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_canon(v, found) for v in obj) + "]"
    if inspect.isfunction(obj):
        found.append(obj)
        return f"<fn {_fn_key(obj)}>"
    return repr(obj)


def _fn_key(fn) -> str:
    """A function's name for the record; lambdas share one name, so theirs carries their logic."""
    key = f"{fn.__module__}.{fn.__qualname__}"
    if fn.__name__ == "<lambda>":
        key += "#" + hashlib.sha256(_logic(fn).encode("utf-8")).hexdigest()[:12]
    return key


def _referenced_globals(fn) -> set:
    """The global names a function's code mentions: its own, its lambdas' and its comprehensions'."""
    names, todo = set(), [fn.__code__]
    while todo:
        code = todo.pop()
        names |= set(code.co_names)
        todo += [c for c in code.co_consts if inspect.iscode(c)]
    return names


def _dependencies(funcs, values=None, skip=()) -> dict:
    """Every PRISM function and settings value the given functions reach, followed to the end: the
    helpers they call, the helpers those call, the weight tables they read (config included). Names
    are looked up where each function lives, so a value imported from config counts wherever it is
    read. Libraries (numpy, pandas) are not followed; their versions are pinned in requirements.txt.
    `values` adds named settings directly (and follows the functions inside them); `skip` names a
    settings table that is covered that way instead of whole."""
    out, todo = {}, list(funcs)
    for label, value in sorted((values or {}).items()):
        found = []
        out[f"value {label}"] = _canon(value, found)
        todo += [f for f in found if _is_project(f)]
    while todo:
        fn = todo.pop()
        key = f"fn {_fn_key(fn)}"
        if key in out:
            continue
        out[key] = _logic(fn)
        for name in sorted(_referenced_globals(fn)):
            if name not in fn.__globals__:      # a method name or a builtin, not a module-level value
                continue
            obj = fn.__globals__[name]
            if inspect.isfunction(obj) and _is_project(obj):
                todo.append(obj)
            elif isinstance(obj, _PLAIN) and not name.startswith("__") and name not in skip:
                found = []
                out[f"value {name}"] = _canon(obj, found)
                todo += [f for f in found if _is_project(f)]
    return out


def _fingerprint_of(funcs, values=None, skip=()) -> str:
    """Hash of the LOGIC a record depends on: the given functions and everything they reach
    (_dependencies). Widened 2026-10-07: the fingerprint used to cover the recipe's own code only, so a
    change to how momentum_score or improvement_score is computed changed the list on screen while the
    card still read 'measured'."""
    deps = _dependencies(funcs, values, skip)
    roots = [_logic(fn) for fn in funcs]
    for fn in funcs:                         # hashed above without their names; a rename is no change
        deps.pop(f"fn {_fn_key(fn)}", None)
    text = "\n".join(roots + [f"{k}={v}" for k, v in sorted(deps.items())])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def recipe_fingerprint(scan: dict) -> str:
    return _fingerprint_of([*_recipe_functions(scan), *scan.get("inputs", ())], scan.get("values"),
                           scan.get("skip", ()))


def load_records(path: Path = RECORDS_PATH) -> dict:
    """The committed records; {} when missing, unreadable or another version (cards then say "Not yet
    measured" rather than guess)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != RECORD_VERSION or "scans" not in data:
        return {}
    return data


def record_state(records: dict, scan: dict):
    """("measured", rec) · ("stale", rec) when the recipe changed since it was measured ·
    ("unmeasured", None)."""
    rec = (records.get("scans") or {}).get(scan["key"])
    if not rec:
        return "unmeasured", None
    if rec.get("fingerprint") != recipe_fingerprint(scan):
        return "stale", rec
    return "measured", rec


# ── The list ───────────────────────────────────────────────────────────────────────────────────────
MEASURED_FLAGS = {"rf_wc_double_squeeze": "Squeeze", "rf_receivables_bloat": "Receivables",
                  "rf_negative_fcf": "Neg FCF"}
LIQUIDITY_STEPS = (None, 0.5, 1.0, 2.0, 5.0)    # ₹ crore a day; None = any

TABLE_COLUMNS = ("rank", "_new", "name", "sector", "composite_score", "momentum_score",
                 "improvement_score", "_strong_q", "_measured_flags", "weinstein_stage", "_liq_cr")
# Set widths (an auto-sized text column grows to its longest cell): 1,025px, inside the 1,030px a
# 1536px laptop leaves after the 44px row-selection column (the Deep Scanner's measured budget).
# Measured 2026-10-07 in the grid's own font (13px Source Sans + 16px padding): every short Trend label
# fits 90, "Receivables · Neg FCF" needs 133, the headers of the number columns need their width.
TABLE_WIDTHS = {"rank": 55, "_new": 55, "name": 170, "sector": 110, "composite_score": 105,
                "momentum_score": 85, "improvement_score": 100, "_strong_q": 40,
                "_measured_flags": 135, "weinstein_stage": 90, "_liq_cr": 80}
# The engine's Weinstein labels, shortened so the column fits; the header tip carries the words.
TREND_SHORT = {"📈 Stage 2 Advancing": "📈 Stage 2", "🔄 Stage 1 Basing": "🔄 Stage 1",
               "⚠️ Stage 3 Top": "⚠️ Stage 3", "📉 Stage 4 Declining": "📉 Stage 4"}


def table_columns(scan: dict) -> tuple:
    """The columns this scan shows: every table column except those its recipe fixes."""
    return tuple(c for c in TABLE_COLUMNS if c not in scan.get("fixed", ()))


def table_widths(scan: dict) -> dict:
    """Column widths for this scan: the shared widths, with the scan's own overrides."""
    return {**TABLE_WIDTHS, **scan.get("widths", {})}


def scan_table(df: pd.DataFrame, scan: dict, members_at_latest=None, min_cr=None,
               keep_ids=None, result_window_days=None) -> pd.DataFrame:
    """The scan's stocks, sorted by PRISM Rank. The recipe runs on the frame given (the whole market);
    the liquidity floor and the sidebar's kept stocks (keep_ids) only narrow the FINISHED list, so a
    listed stock is exactly the match its record measured. A stock with no traded value cannot clear
    a floor. Flagged stocks stay, with their measured flags named. 🆕 marks a stock that was not in the
    scan on the 🆕 baseline (the last archived month-end copy; no baseline = nothing marked). 📄 marks
    a stock whose quarterly result was DECLARED inside the last `result_window_days` (the days since that
    baseline): result_age_days counts days since the last declared result and is negative for one only
    scheduled, so 0 <= age < window is "declared since the baseline" — the rule Movers uses. The two
    clocks differ on purpose: technical legs move daily, fundamentals only when a result lands, so 🆕 + 📄
    together means the stock entered on a fresh result."""
    out = df.loc[scan["mask"](df)].copy()
    out["_liq_cr"] = traded_value_cr(out)
    if min_cr is not None:
        out = out[out["_liq_cr"] >= min_cr]
    if keep_ids is not None:
        out = out[out["company_id"].astype(str).isin(keep_ids)]
    named = pd.Series("", index=out.index, dtype=object)
    for col, lab in MEASURED_FLAGS.items():                 # three columns, never a loop over rows
        named = named.where(~(_num(out, col) == 1), np.where(named == "", lab, named + " · " + lab))
    out["_measured_flags"] = named
    new = (pd.Series(False, index=out.index) if members_at_latest is None
           else ~out["company_id"].astype(str).isin(set(members_at_latest)))
    age = _num(out, "result_age_days")
    fresh = ((age >= 0) & (age < result_window_days) if result_window_days
             else pd.Series(False, index=out.index))
    out["_new"] = np.select([new & fresh, new, fresh], ["🆕 📄", "🆕", "📄"], default="")
    out["_strong_q"] = np.where(strong_quarter(out) == 1, "✓", "")
    if "weinstein_stage" in out.columns:
        out["weinstein_stage"] = out["weinstein_stage"].replace(TREND_SHORT)
    out = out.sort_values("rank", kind="mergesort")
    for c in TABLE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out[[*TABLE_COLUMNS, "company_id"]].reset_index(drop=True)


def table_config(members_date_label: str, scan: dict = None) -> dict:
    """Column headers — the same header each column carries in the Deep Scanner — widths and tips."""
    w = table_widths(scan or {})
    cc = st.column_config
    since = (f"on {members_date_label}, the last archived month-end copy" if members_date_label
             else "on the last archived month-end copy")
    return {
        "rank": cc.NumberColumn("Rank", format="%.0f", width=w["rank"],
                                help="PRISM's overall rank. The list is sorted by it for reading only: inside "
                                     "this scan the top of the Rank did not beat the rest in the 2026-10-06 "
                                     "test, so every match is the scan."),
        "_new": cc.TextColumn("New", width=w["_new"],
                              help=f"🆕 = not in this scan {since}. 📄 = its quarterly result was declared "
                                   f"since then, so its Strong Quarter and Improvement readings are fresh. "
                                   f"🆕 📄 together = it entered on a new result; 🆕 alone = it entered on "
                                   f"price movement (technical legs move daily — check it is not flickering "
                                   f"across the line)."),
        "name": cc.TextColumn("Stock", width=w["name"]),
        "sector": cc.TextColumn("Sector", width=w["sector"]),
        "composite_score": cc.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f",
                                             width=w["composite_score"]),
        "momentum_score": cc.NumberColumn("Momentum", format="%.0f", width=w["momentum_score"]),
        "improvement_score": cc.NumberColumn("Improvement", format="%.0f", width=w["improvement_score"]),
        "_strong_q": cc.TextColumn("🔥", width=w["_strong_q"],
                                   help="🔥 Strong Quarter: latest quarter vs the same quarter last year — "
                                        "revenue up more than 15%, profit up more than 20%, margin wider."),
        "_measured_flags": cc.TextColumn(
            "⚠ Measured flags", width=w["_measured_flags"],
            help="The red flags that went with weaker returns in the 2026-10-06 measurement (provisional; "
                 "December re-tests them): working-capital double squeeze (customers pay 10+ days slower "
                 "AND suppliers are paid 10+ days faster) — weaker even for improving businesses; "
                 "receivables bloat and negative free cash flow — weaker mainly for businesses that are "
                 "not improving, and negative free cash flow alongside capex mirage was not weaker. "
                 "Shown, never removed: open the Tear-Sheet's Forensics tab before deciding."),
        "weinstein_stage": cc.TextColumn(
            "Trend", width=w["weinstein_stage"],
            help="Weinstein stage — the price against its 30-week average, and that average against the "
                 "200-day: 📈 Stage 2 advancing · 🔄 Stage 1 basing · ⚠️ Stage 3 topping · 📉 Stage 4 "
                 "declining · ❔ not enough price history. Being above its stop is not the same line, so "
                 "a Stage 3 or 4 can appear in a scan; open the Tear-Sheet before deciding."),
        "_liq_cr": cc.NumberColumn("₹ Cr/day", format="%.1f", width=w["_liq_cr"],
                                   help="Average daily traded value over 20 days (average volume × price), "
                                        "in ₹ crore."),
    }


# ── The card ───────────────────────────────────────────────────────────────────────────────────────
def _month_span(w: dict) -> str:
    a, b = date.fromisoformat(w["start"]), date.fromisoformat(w["end"])
    return f"{a:%b}→{b:%b}"


def _range(vals, fmt) -> str:
    lo, hi = min(vals), max(vals)
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}–{fmt.format(hi)}"


def _bars(ws, c, esc) -> str:
    """One bar per window: height by size (12 points fills it), green above the median, red below."""
    out = ""
    for w in ws:
        v = float(w["median_excess_pp"])
        h = 6 + round(28 * min(abs(v) / 12.0, 1.0))
        col = c["green"] if v > 0 else c["red"]
        out += (f'<div style="display:flex;flex-direction:column;align-items:center;justify-content:flex-end;'
                f'min-width:46px;" title="{esc(w["start"])} → {esc(w["end"])}, {int(w["days"])} days, '
                f'{int(w["n"])} stocks">'
                f'<div style="width:14px;height:{h}px;background:{col};border-radius:3px;"></div>'
                f'<div style="font-size:0.78rem;font-weight:800;color:{col};margin-top:3px;">{v:+.1f}</div>'
                f'<div style="font-size:0.6rem;color:{c["text_muted"]};">{_month_span(w)}</div></div>')
    return out


def _first_unseen_due() -> str:
    """When the first unseen window can exist: a start must be a month-end copy taken after
    DESIGNED_ON, and a window needs MIN_HORIZON_DAYS — so about that many days after the end of the
    month the recipes were chosen in."""
    d = date.fromisoformat(DESIGNED_ON)
    month_end = (date(d.year + d.month // 12, d.month % 12 + 1, 1) - timedelta(days=1))
    return f"{month_end + timedelta(days=MIN_HORIZON_DAYS):%d %b}"


def _extras(ws, c, esc) -> str:
    """Under a record: the NET reading (liquid matches only, charged the round trip) and how many matches
    the headline left out. Windows measured before these were recorded say so instead of showing 0."""
    nets = [w.get("net") for w in ws]
    if not any(nets):
        return (f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:3px;">Net-of-costs record '
                f'and left-out counts: not measured for these windows yet</div>')
    cells = []
    for nt in nets:
        if nt and "median_excess_pp" in nt:
            col = c["green"] if nt["median_excess_pp"] > 0 else c["red"]
            cells.append(f'<span style="color:{col};font-weight:700;">{nt["median_excess_pp"]:+.1f}</span>')
        else:
            cells.append('<span title="fewer than 30 liquid matches">–</span>')
    nt0 = next(nt for nt in nets if nt)
    measured = [nt for nt in nets if nt and "median_excess_pp" in nt]
    tail = (f' · {_range([int(nt["beat_pct"]) for nt in measured], "{}")}% beat · '
            f'{_range([int(nt["n"]) for nt in measured], "{}")} stocks' if measured else
            ' · too few liquid matches to measure')
    out = (f'<div style="font-size:0.7rem;color:{c["text_secondary"]};margin-top:5px;">Net of a '
           f'{float(nt0["cost_pp"]):.1f}pp round trip, ₹{float(nt0["min_cr"]):g} Cr+/day matches only: '
           f'{" / ".join(cells)}{tail}</div>')
    if all("dropped_no_return" in w for w in ws):
        gone = _range([int(w["dropped_no_return"]) for w in ws], "{}")
        ca = _range([int(w["dropped_corp_action"]) for w in ws], "{}")
        out += (f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;">Left out per window: '
                f'{gone} with no price at the end (delisted, suspended or missing) · {ca} corporate '
                f'actions</div>')
    return out


def card_html(scan: dict, n_now: int, state: str, rec, selected: bool, latest: str) -> str:
    """One scan card: name, today's count, the recipe in words, and two records — the IN-SAMPLE one
    (the design period: windows studied while the recipe was chosen, frozen) and the OUT-OF-SAMPLE one
    (windows that start after DESIGNED_ON, measured automatically from the PRISM Archive). Each shows
    one bar per window, the share of matches that beat the market, the sample size, the net-of-costs
    reading, what was left out and "provisional". A stale or missing record shows no numbers at all."""
    c = COLORS
    esc = html.escape
    border = c["gold"] if selected else c["border"]
    head = (f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;">'
            f'<span style="font-size:1.0rem;font-weight:800;color:{c["text_primary"]};">'
            f'{scan["emoji"]} {esc(scan["name"])}</span>'
            f'<span style="font-size:1.35rem;font-weight:900;color:{c["gold"]};">{n_now:,}'
            f'<span style="font-size:0.7rem;font-weight:600;color:{c["text_muted"]};"> now</span></span></div>'
            f'<div style="font-size:0.76rem;color:{c["text_secondary"]};margin:4px 0 10px 0;line-height:1.35;">'
            f'{esc(scan["recipe"])}</div>')
    if state == "measured" and rec and rec.get("windows"):
        # One ROW per record — the words on the left, the bars on the right — so the card is about half
        # the height of a stacked one and the list starts on the first screen (2026-10-07).
        ws = rec["windows"]
        beat = _range([int(w["beat_pct"]) for w in ws], "{}") + "%"
        n = _range([int(w["n"]) for w in ws], "{}")
        starts = len({w["start"] for w in ws})
        built = f"{date.fromisoformat(DESIGNED_ON):%d %b}"
        unseen = rec.get("out_of_sample") or []
        bars_tip = ("Points above the market median in each window, before costs; hover a bar for its "
                    "dates and size. In-sample = windows studied while the recipe was chosen.")
        # Named in so many words (2026-10-07): "provisional" alone read as "proven, just early".
        body = (f'<div style="display:flex;gap:14px;align-items:flex-end;justify-content:space-between;">'
                f'<div style="flex:1;min-width:0;">'
                f'<div style="font-size:0.66rem;font-weight:700;color:{c["text_muted"]};letter-spacing:.3px;">'
                f'IN-SAMPLE · DESIGN-PERIOD RECORD</div>'
                f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;">The recipe was chosen '
                f'on these windows: worth testing, not proven</div>'
                f'<div style="font-size:0.74rem;color:{c["text_secondary"]};margin-top:3px;">'
                f'{beat} of matches beat the market · before costs</div>'
                f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;">{n} stocks per '
                f'window · provisional</div>'
                f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;">{len(ws)} short '
                f'windows from {starts} start date{"s" if starts != 1 else ""}'
                f'{f", measured to {esc(latest)}" if latest else ""}</div>'
                + ("" if unseen else
                   f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;" title="Windows '
                   f'that start after the recipe was chosen ({built}) — data nobody looked at while '
                   f'choosing it.">Out-of-sample: pending — unseen windows: none yet, the first is due '
                   f'about {_first_unseen_due()}</div>')
                + f'</div><div style="display:flex;gap:4px;align-items:flex-end;" title="{esc(bars_tip)}">'
                f'{_bars(ws, c, esc)}</div></div>{_extras(ws, c, esc)}')
        if unseen:
            u_beat = _range([int(w["beat_pct"]) for w in unseen], "{}") + "%"
            body += (f'<div style="display:flex;gap:14px;align-items:flex-end;justify-content:space-between;'
                     f'border-top:1px solid {c["border"]};margin-top:8px;padding-top:7px;">'
                     f'<div style="flex:1;min-width:0;">'
                     f'<div style="font-size:0.66rem;font-weight:700;color:{c["text_muted"]};letter-spacing:.3px;">'
                     f'OUT-OF-SAMPLE · SINCE IT WAS BUILT ({esc(built.upper())}) — WINDOWS IT NEVER SAW</div>'
                     f'<div style="font-size:0.74rem;color:{c["text_secondary"]};margin-top:3px;">'
                     f'{u_beat} of matches beat the market · before costs · the most trustworthy record</div></div>'
                     f'<div style="display:flex;gap:4px;align-items:flex-end;">{_bars(unseen, c, esc)}</div></div>'
                     f'{_extras(unseen, c, esc)}')
    elif state == "stale":
        body = (f'<div style="font-size:0.74rem;color:{c["gold"]};">The recipe changed since its record was '
                f'measured — no record is shown until it is re-measured.</div>')
    else:
        body = f'<div style="font-size:0.74rem;color:{c["text_muted"]};">Not yet measured.</div>'
    return (f'<div style="background:{c["bg_secondary"]};border:1px solid {border};border-radius:10px;'
            f'padding:12px 14px;">{head}{body}</div>')


def members_label(records: dict) -> str:
    """'30 Sep' — the archived month-end copy the 🆕 marks compare against; '' without one."""
    d = records.get("members_date") if records else None
    try:
        return f"{date.fromisoformat(d):%d %b}"
    except (TypeError, ValueError):
        return ""


def latest_label(records: dict) -> str:
    """'03 Oct' for the last snapshot a record was measured on; '' with no records."""
    d = records.get("latest_snapshot") if records else None
    try:
        return f"{date.fromisoformat(d):%d %b}"
    except (TypeError, ValueError):
        return ""
