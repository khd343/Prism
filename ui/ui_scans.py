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

from config import BENCHMARK_INDICES, COLORS
from core.data_engine import traded_value_cr        # ONE liquidity number (moved there 2026-10-09)
from core.forward_returns import MIN_HORIZON_DAYS
from core.scoring_engine import _compute_improvement_score, compute_momentum_score, strong_quarter
from ui.ui_group_views import MEASURES, group_measures
from ui.ui_scanner import _SCANNER_HEADER_TIPS        # the Deep Scanner's header tips: one tip per column
# LIQUIDITY_STEPS and step_label are also re-exported for app.py's Scans selector (its one ui_scans import).
from ui.ui_liquidity import (HEADER as LIQUIDITY_HEADER, LIQUIDITY_STEPS, ORDER_SHARE, SURVEILLANCE_NOTE,
                             order_guide, order_size_lakh, step_label)

RECORDS_PATH = Path(__file__).with_name("scan_records.json")
# The net record (2026-10-07): what a buyer could have kept. Only matches trading at least NET_MIN_CR
# a day count, and each is charged NET_COST_PP for the round trip (0.5% in + 0.5% out: taxes, spread,
# impact). The benchmark (the market median) is charged nothing, so the cost side errs conservative.
NET_MIN_CR = 2.0
NET_COST_PP = 1.0
# Each scan against the index fund you could hold instead (2026-10-08, docs/december-2026-plan.md rule
# 23, written before any result). The portfolio you would actually hold — the scan's NET_MIN_CR+ matches,
# equal-weighted, charged the NET_COST_PP round trip each window — against the fund you would actually
# own: the index's PLAIN level change (not total return; the matches' returns carry no dividends either)
# less FUND_FEE_PCT_YR for the window's days. 0.2% a year is about what the cheapest direct plans charge
# (Nifty 500 ~0.17%, Nifty500 Momentum 50 0.16-0.19%, Oct 2026); funds on the smaller factor indices
# charge more (Smallcap250 Momentum Quality 100: 0.47-0.64%), so for those the test errs strict — it
# never flatters a scan. Each scan's main index is its `benchmark`; every declared index is recorded.
FUND_FEE_PCT_YR = 0.2
# "What if you held 20 of them?" (2026-10-13, fixed before the first unseen window; December plan rule
# 26 E-F, rule 27). Measured first: on windows that do not overlap, the ORDER inside a scan barely
# mattered (Rank's top 25 at the 38/33/73th percentile of random picks, momentum's at 93/63/83), while
# HOW MANY you hold mattered a lot (random picks from the buyable Playbook list: 10 names 0 to +29%, 20
# names +4.7 to +24%). So each unseen window also records HOLD_K stocks picked at random HOLD_DRAWS
# times with a FIXED seed, from the matches a Rs 1 lakh account can hold — NET_MIN_CR+ a day AND one
# share at most HOLD_MAX_PRICE (13 of 185 buyable stocks cost more on 2026-10-12) — after NET_COST_PP:
# the typical result, its 10th-90th range, how often it lost money, and where the top HOLD_K by
# momentum and by Rank land among those picks.
HOLD_K = 20
HOLD_MAX_PRICE = 5000.0
HOLD_DRAWS = 2000
HOLD_SEED = 20261013
# The rule that decides a scan's table order, written before any unseen window: it switches to momentum
# order only if, over the scan's FIRST `windows` separate (non-overlapping), finished unseen windows with
# at least `min_n` holdable matches, momentum's top HOLD_K beats on average at least `min_avg_pct`% of
# the random picks AND averages above Rank's top HOLD_K. Luck alone passes a 70% average over four
# windows about 1 time in 12 (8.6%); "3 of 4 above half" would pass 31% of the time. Each scan decides
# on its own windows; 🌱 keeps its order (its 3-year study found momentum inside it flips year to year).
ORDER_RULE = dict(scans=("playbook", "leaders"), windows=4, min_avg_pct=70.0, min_n=30)
RECORD_VERSION = 2
# The day the recipes were chosen. Every window that STARTS on or after it is out of sample: data
# nobody looked at while choosing. Windows before it are the design-period record (in sample).
DESIGNED_ON = "2026-10-06"


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """The column as numbers; all-NaN when absent, so a missing input passes nothing."""
    return (pd.to_numeric(df[col], errors="coerce") if col in df.columns
            else pd.Series(np.nan, index=df.index))


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


def _growth(df: pd.DataFrame, now: str, prev: str) -> pd.Series:
    """now / prev - 1, only where BOTH years are positive: a loss year has no growth RATE, so it is no
    evidence (NaN, which passes nothing). The guard PRISM's improvement score uses."""
    a, b = _num(df, now), _num(df, prev)
    return (a / b - 1.0).where((a > 0) & (b > 0))


def revenue_led(df: pd.DataFrame) -> pd.Series:
    """Sales up more than 10% over the last year AND profit growing faster than sales: margin expansion
    the business earned by selling more, not by cutting costs on a shrinking business."""
    sales, profit = _growth(df, "revenue", "revenue_1yb"), _growth(df, "pat", "pat_1yb")
    return ((sales > 0.10) & (profit > sales)).fillna(False).astype(bool)


def compounder_candidates(df: pd.DataFrame) -> pd.Series:
    """Profit growth AND margin widening over the last year, each in the top 40% of the stocks that can
    be judged on it (whole market), AND revenue-led (sales up more than 10%, profit growing faster).

    Backtest of 2026-10-07 (PRISM's export of 2026-10-06 for the fiscal history, joined to StockScans
    returns, 2,712 stocks; traits as they stood in FY23, public by mid-2023): matches compounded
    +20.4%/yr against +4.4%/yr for the rest from Oct 2023 to Oct 2026. Year by year, each judged only
    on results public at its start, points above the market's median stock: 23→24 +31.2 against -6.8,
    24→25 +13.5 against -2.6, 25→26 +23.6 against -2.2. The same data reproduces the study in
    docs/december-2026-plan.md section 1 to the stock (358 vs 1,102 matches), and without the
    revenue-led leg the edge all but vanishes, so it is part of the recipe, not an option."""
    profit = _growth(df, "pat", "pat_1yb")
    margin = _num(df, "opm") - _num(df, "opm_1yb")
    # Top 40% = percentile rank ABOVE 0.6: of 10 stocks, ranks 7-10 (rank 6 sits at exactly 0.6). The
    # percentiles rank the whole market BEFORE the revenue-led leg narrows it, as in the backtest.
    top = (profit.rank(pct=True) > 0.6) & (margin.rank(pct=True) > 0.6)
    return (top & revenue_led(df)).fillna(False).astype(bool)


# ── Why a stock LEFT 🌱 (2026-10-07) ──────────────────────────────────────────────────────────────
# Measured in the yearly backtest (December section 9c): a 🌱 stock judged on its NEXT annual results,
# points against the market's median stock the following year — still revenue-led +9.8 / +26.6, growing
# but no longer revenue-led +2.4 / +0.5, profit fell -11.1 / -8.1. Only the first two reasons below are
# measured; the others say so. Order = how bad the news is: a stock that fell AND stopped being
# revenue-led is listed under "profit fell".
EXIT_REASONS = {
    "profit_fell": ("🔴", "profit fell",
                    "measured: a 🌱 stock whose next annual profit fell trailed the market's median "
                    "stock by 8 to 11 points the following year"),
    "not_revenue_led": ("🟠", "no longer revenue-led",
                        "measured: profit did not fall, but sales are no longer up 10%+ with profit "
                        "growing faster; its edge was gone (+2.4 and +0.5 points against +9.8 and +26.6 "
                        "for those that stayed revenue-led)"),
    "rank_slipped": ("⚪", "others grew faster",
                     "not measured: still revenue-led with profit up, but its profit growth or margin "
                     "change slipped out of the market's top 40%"),
    "unjudged": ("⚪", "a figure is missing", "not measured: a number the recipe needs is missing"),
    "gone": ("⚪", "not in today's data", "not measured: delisted, suspended or renamed"),
}


def compounder_exit_reason(df: pd.DataFrame) -> pd.Series:
    """For each row, why it is NOT a 🌱 match (read only for stocks that left): a key of EXIT_REASONS."""
    pat, pat_1yb = _num(df, "pat"), _num(df, "pat_1yb")
    fell = pat.notna() & pat_1yb.notna() & (pat < pat_1yb)
    needed = ("pat", "pat_1yb", "revenue", "revenue_1yb", "opm", "opm_1yb")
    unjudged = pd.concat([_num(df, c).isna() for c in needed], axis=1).any(axis=1)
    reason = np.select([fell, unjudged, ~revenue_led(df)], ["profit_fell", "unjudged", "not_revenue_led"],
                       default="rank_slipped")
    return pd.Series(reason, index=df.index, dtype=object)


def cycle_shortened(df: pd.DataFrame) -> pd.Series:
    """Cash conversion cycle down 15+ days AND sales up more than 15%, BOTH in the same fiscal year as
    the P&L 🌱 is judged on — supply-chain velocity, the December candidate, as a READING AID on the 🌱
    table (not part of its recipe or record).

    Measured 2026-10-07 over three yearly windows: inside 🌱, same-year velocity beat the market by +24
    to +54 points, ahead of 🌱 without it every year. A YEAR-OLD cash-cycle improvement did not help
    (+9.4 vs +15.9 and +32.0 vs +32.0), and the vendor rolls the balance sheet months after the P&L
    (cf_year_lag == 1 on ~83% of stocks in October). So the ✓ needs the balance sheet ON the P&L's year
    (cf_year_lag == 0): a lagging or undecided stock gets no ✓ until its balance sheet rolls."""
    same_year = _num(df, "cf_year_lag") == 0
    change = _num(df, "ccc") - _num(df, "ccc_1yb")
    sales = _growth(df, "revenue", "revenue_1yb")
    return (same_year & (change <= -15) & (sales > 0.15)).fillna(False).astype(bool)


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
    # `benchmark`: the index fund each scan is judged against on its card — fixed 2026-10-08, before any
    # result (December plan rule 23). Not part of the recipe, so not in the fingerprint.
    dict(key="playbook", emoji="🚀", name="Playbook Shortlist", mask=playbook_shortlist,
         benchmark="NSE:NIF500MO50",                  # a momentum list across the whole market
         uses=(strong_quarter, _pct, _num), inputs=(compute_momentum_score, _compute_improvement_score),
         fixed=("_strong_q",), widths={"name": 190, "sector": 130},
         recipe="Momentum and Breakout in the top third of the market, a 🔥 Strong Quarter, Improvement "
                "in the top half, and the price above its stop."),
    dict(key="leaders", emoji="🌬️", name="Leading-Sector Stocks", mask=leading_sector_stocks,
         benchmark="NSE:CNX500",                      # whole sectors, every size
         uses=(leading_sectors, group_measures, strong_quarter),
         # It reads two of the Sectors measures, so it follows those two, not the whole table: a new
         # Sectors column must not stale this record (2026-10-07).
         values={f"MEASURES[{k}]": MEASURES[k] for k in ("grp_near_high", "grp_strong_quarter")},
         skip=("MEASURES",),
         recipe="Every stock in the 5 sectors (of 5+ stocks) with the most stocks near their 52-week "
                "high and reporting strong quarters, as ranked on 📈 Sectors."),
    # 🌱 (2026-10-07): a LONG-horizon list. Its record is a 3-year backtest, not saved windows, so its
    # unseen windows are 12 months long (`horizon_days`) and start after its own design date.
    dict(key="compounder", emoji="🌱", name="Compounder Candidates", mask=compounder_candidates,
         benchmark="NSE:NS250MQ100",                  # the plan's main test for 🌱
         uses=(revenue_led, _growth, _num), designed_on="2026-10-07", horizon_days=365,
         # One unseen window a YEAR, starting each October (2026-10-07): annual results are public by
         # then, as in the yearly backtest. A window from every month-end would put ~12 bars on the card
         # by 2028 that overlap by 11 months — one year of evidence that looks like twelve.
         start_month=10,
         # "Cycle ↓" (2026-10-07): a reading aid, not part of the recipe or record; its 60px come from
         # Stock, Sector and Score so the table keeps the laptop budget.
         extra=("_cycle",), widths={"name": 155, "sector": 95, "composite_score": 75},
         # "Left 🌱" (2026-10-07): why a stock in 🌱 on the 🆕 baseline is not in it now. Not part of the
         # recipe, so not in the fingerprint (which follows mask, uses and inputs only).
         exit_reason=compounder_exit_reason,
         recipe="Sales up more than 10% with profit growing faster, and profit growth and margin "
                "widening each in the top 40% of the market. Starting points for research: better odds, "
                "not a promise."),
)


def scan_designed_on(scan: dict) -> str:
    """The day this scan's recipe was chosen: windows starting after it are out of sample."""
    return scan.get("designed_on", DESIGNED_ON)


def scan_horizon(scan: dict) -> int:
    """The shortest window that counts as evidence for this scan, in days."""
    return int(scan.get("horizon_days", MIN_HORIZON_DAYS))


def scan_start_month(scan: dict):
    """The one calendar month this scan's unseen windows start in (1-12), or None: every month-end."""
    return scan.get("start_month")


def measured_on_snapshots(scan: dict) -> bool:
    """True when the scan's design-period record is measured on the saved snapshots by the local
    tools/scan_records.py, whose windows are MIN_HORIZON_DAYS long. A longer-horizon scan (🌱, a year)
    carries a study instead: a 25-day window is no evidence for it, and writing one would overwrite the
    study (2026-10-08)."""
    return scan_horizon(scan) <= MIN_HORIZON_DAYS


def _month_end(year: int, month: int) -> date:
    return date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)

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


class SourceMismatch(RuntimeError):
    """The file on disk no longer holds the code that is running — the app was updated while it ran
    (Streamlit Cloud, 2026-10-09: the old module stayed in memory, the new file arrived, and reading
    the logic by the old line numbers crashed the Scans tab and every tab after it). No fingerprint is
    made from a mismatch; the record is shown as one that could not be checked."""


def _same_code(a, b) -> bool:
    """Two code objects that do the same thing: instructions, names and constants (nested code compared
    the same way). Line numbers are not compared — they are what a twin compiled elsewhere differs in."""
    if a.co_code != b.co_code or a.co_names != b.co_names or len(a.co_consts) != len(b.co_consts):
        return False
    return all(_same_code(x, y) if inspect.iscode(x) and inspect.iscode(y) else repr(x) == repr(y)
               for x, y in zip(a.co_consts, b.co_consts))


def _twin(node: ast.Lambda, code):
    """The lambda's source compiled again, like for like: a lambda that uses its enclosing function's
    variables (a closure) is compiled inside a function that binds them, as where it was written."""
    if code.co_freevars:
        outer = ast.FunctionDef(
            name="_outer", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[ast.Assign(targets=[ast.Name(id=v, ctx=ast.Store())], value=ast.Constant(None))
                  for v in code.co_freevars] + [ast.Return(value=node)],
            decorator_list=[], type_params=[])
        compiled = compile(ast.fix_missing_locations(ast.Module(body=[outer], type_ignores=[])),
                           code.co_filename, "exec")
    else:
        compiled = compile(ast.Expression(body=node), code.co_filename, "eval")
    todo = [compiled]
    while todo:
        for c in todo.pop().co_consts:
            if inspect.iscode(c):
                if c.co_name == "<lambda>":
                    return c
                todo.append(c)
    return None


def _lambda_node(fn) -> ast.Lambda:
    """A lambda's own source node: found in its module by the line it starts on, and confirmed by
    compiling it back to the code that is running — so two lambdas on one line are told apart and a
    file that changed under the running code is refused, never misread."""
    code = fn.__code__
    try:
        tree = ast.parse(inspect.getsource(inspect.getmodule(fn)))
    except (OSError, TypeError, SyntaxError) as e:
        raise SourceMismatch(f"no readable source for the lambda at line {code.co_firstlineno}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda) and node.lineno == code.co_firstlineno:
            twin = _twin(node, code)
            if twin is not None and _same_code(twin, code):
                return node
    raise SourceMismatch(f"{code.co_filename}:{code.co_firstlineno} no longer holds this lambda")


def _logic(fn) -> str:
    """A function's LOGIC: its parsed SOURCE without the docstring or its own name, so a reworded
    comment or a rename keeps a record valid and a changed threshold does not. A lambda (one that sits
    inside a settings table, e.g. a Sectors measure) is read from its module's source the same way.
    Never the compiled bytecode: it differs between Python versions (3.13 here and in the weekly job,
    3.14 on Streamlit Cloud), and a record must read the same everywhere — on bytecode the 🌬️ record
    read 'recipe changed' on the live app from 2026-10-07 to 2026-10-09."""
    if fn.__name__ == "<lambda>":
        return ast.dump(_lambda_node(fn))
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    except (OSError, TypeError, SyntaxError) as e:
        raise SourceMismatch(f"no readable source for {fn.__module__}.{fn.__qualname__}") from e
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != fn.__name__:
        raise SourceMismatch(f"{fn.__module__}.{fn.__qualname__}: line {fn.__code__.co_firstlineno} of its "
                             f"file no longer holds it")
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
        node.body = node.body[1:]
    node.name = "_"
    return ast.dump(node)


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
    """("measured", rec) · ("stale", rec) when the recipe changed since it was measured · ("unverified",
    rec) when the recipe's logic could not be read just now · ("unmeasured", None). A record nobody
    could check shows no numbers, and it never takes the app down: on Streamlit Cloud a push can land
    while the app runs (old code in memory, new files on disk), and on 2026-10-09 reading the logic
    then crashed the Scans tab and every tab after it. A page reload clears it."""
    rec = (records.get("scans") or {}).get(scan["key"])
    if not rec:
        return "unmeasured", None
    try:
        current = recipe_fingerprint(scan)
    except Exception:          # SourceMismatch, or a file unreadable mid-deploy: never a crash
        return "unverified", rec
    return ("measured" if rec.get("fingerprint") == current else "stale"), rec


# ── The list ───────────────────────────────────────────────────────────────────────────────────────
MEASURED_FLAGS = {"rf_wc_double_squeeze": "Squeeze", "rf_receivables_bloat": "Receivables",
                  "rf_negative_fcf": "Neg FCF"}
# The liquidity floor's steps, the order-size rule and the traded-value number itself are shared with
# the sidebar, the Deep Scanner and the Tear-Sheet (ui/ui_liquidity.py, core.data_engine.traded_value_cr).


def liquidity_help() -> str:
    """The Liquidity selector's help: what the number is, that it is a preference, and which step fits
    an order size — plus the one tradability risk the data cannot see."""
    guide = order_guide()
    return ("Minimum average daily traded value over 20 days (average volume × price). Your preference, "
            "not part of the record: the record counts every match.\n\n"
            f"**Which step?** Keep one order under about {ORDER_SHARE:.0%} of a day's traded value, so the "
            f"market can take it without moving the price much:\n\n{guide}\n\n"
            + SURVEILLANCE_NOTE)

TABLE_COLUMNS = ("rank", "_new", "name", "sector", "composite_score", "momentum_score",
                 "improvement_score", "_strong_q", "_measured_flags", "weinstein_stage", "_liq_cr")
# Set widths (an auto-sized text column grows to its longest cell): 1,025px, inside the 1,030px a
# 1536px laptop leaves after the 44px row-selection column (the Deep Scanner's measured budget).
# Measured 2026-10-07 in the grid's own font (13px Source Sans + 16px padding): every short Trend label
# fits 90, "Receivables · Neg FCF" needs 133, the headers of the number columns need their width.
TABLE_WIDTHS = {"rank": 55, "_new": 55, "name": 170, "sector": 110, "composite_score": 105,
                "momentum_score": 85, "improvement_score": 100, "_strong_q": 40,
                "_measured_flags": 135, "weinstein_stage": 90, "_liq_cr": 80, "_cycle": 60,
                # 📈 Technical (2026-10-13): the Deep Scanner's widths for the same columns, Breakout as a
                # number like Momentum here; "🔥 OVERHEATED" fits 110.
                "close_price": 80, "dist_to_vstop": 60, "d49_momentum_quality": 110, "rsi_14d": 50,
                "breakout_score": 85, "crs_52w": 60}
# The table's views (2026-10-13), by the app-wide names (ui_group_views.VIEWS — the Deep Scanner's and
# Market Pulse's). 🏆 Core, the default, answers "why is it in the list"; 📈 Technical answers "is now a
# good moment, how far is the stop, can I buy one share": the Deep Scanner's Technical columns (without
# Readiness and Off ATH, both folded into Breakout) plus Price and ₹ Cr/day, which decide whether a
# small account can buy a share at all. Mom. Quality sits between the two numbers it reads. Adding these
# to Core would have pushed it ~450px past a 1536px laptop. A display choice: it changes no list, record
# or order, and the 🧹 Clear never resets it.
SCAN_VIEWS = ("🏆 Core", "📈 Technical")
TECHNICAL_COLUMNS = ("rank", "_new", "name", "close_price", "_liq_cr", "dist_to_vstop", "momentum_score",
                     "d49_momentum_quality", "rsi_14d", "breakout_score", "crs_52w", "weinstein_stage")
# The engine's Weinstein labels, shortened so the column fits; the header tip carries the words.
TREND_SHORT = {"📈 Stage 2 Advancing": "📈 Stage 2", "🔄 Stage 1 Basing": "🔄 Stage 1",
               "⚠️ Stage 3 Top": "⚠️ Stage 3", "📉 Stage 4 Declining": "📉 Stage 4"}


# Columns a scan may ADD to the shared table (2026-10-07), each placed after an anchor column. Only a
# scan that lists one in `extra` shows it; the others keep exactly the shared table.
EXTRA_COLUMNS = {"_cycle": "_strong_q"}


def table_columns(scan: dict, view: str = SCAN_VIEWS[0]) -> tuple:
    """The columns this scan shows in `view`. 🏆 Core: every table column except those its recipe fixes,
    plus its own extra columns, each right after its anchor. 📈 Technical: TECHNICAL_COLUMNS, the same
    for every scan (none of them is fixed by a recipe)."""
    if view == SCAN_VIEWS[1]:
        return tuple(c for c in TECHNICAL_COLUMNS if c not in scan.get("fixed", ()))
    cols = [c for c in TABLE_COLUMNS if c not in scan.get("fixed", ())]
    for extra in scan.get("extra", ()):
        cols.insert(cols.index(EXTRA_COLUMNS[extra]) + 1, extra)
    return tuple(cols)


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
    out["_cycle"] = np.where(cycle_shortened(out), "✓", "")
    if "weinstein_stage" in out.columns:
        out["weinstein_stage"] = out["weinstein_stage"].replace(TREND_SHORT)
    out = out.sort_values("rank", kind="mergesort")
    shown = [*TABLE_COLUMNS, *(c for c in TECHNICAL_COLUMNS if c not in TABLE_COLUMNS), *EXTRA_COLUMNS]
    for c in shown:
        if c not in out.columns:
            out[c] = np.nan
    return out[[*shown, "company_id"]].reset_index(drop=True)


def _change(df: pd.DataFrame, now: str, prev: str) -> pd.Series:
    """now / prev - 1 in %, wherever the earlier year is positive (a fall into a loss still shows)."""
    a, b = _num(df, now), _num(df, prev)
    return ((a / b - 1.0) * 100).where(b > 0)


def left_scan(df: pd.DataFrame, scan: dict, members_at_latest=None, keep_ids=None) -> pd.DataFrame:
    """Stocks in the scan on the 🆕 baseline that are not in it now, each with why: columns company_id,
    name, sector, _why, _sales_chg, _profit_chg, reason — the worst news first, then by name.

    Empty for a scan without an exit_reason or without a baseline (members_at_latest is None, as it is
    whenever the record is stale — the same rule the 🆕 marks follow). The recipe runs on the whole
    market; the liquidity floor does NOT narrow this list (a stock does not leave 🌱 by trading less),
    keep_ids does, and a stock gone from today's data is kept only when nothing narrows the list."""
    cols = ["company_id", "name", "sector", "_why", "_sales_chg", "_profit_chg", "reason"]
    if members_at_latest is None or "exit_reason" not in scan:
        return pd.DataFrame(columns=cols)
    ids = df["company_id"].astype(str)
    left = set(map(str, members_at_latest)) - set(ids[scan["mask"](df)])
    if keep_ids is not None:
        left &= set(map(str, keep_ids))
    here = df.loc[ids.isin(left)]
    out = pd.DataFrame({"company_id": ids[here.index], "name": here.get("name"), "sector": here.get("sector"),
                        "_sales_chg": _change(here, "revenue", "revenue_1yb"),
                        "_profit_chg": _change(here, "pat", "pat_1yb"),
                        "reason": scan["exit_reason"](here)})
    gone = sorted(left - set(out["company_id"]))
    if gone:
        out = pd.concat([out, pd.DataFrame({"company_id": gone, "name": gone, "sector": "",
                                            "reason": "gone"})], ignore_index=True)
    out["_why"] = out["reason"].map(lambda r: f"{EXIT_REASONS[r][0]} {EXIT_REASONS[r][1]}")
    order = {r: k for k, r in enumerate(EXIT_REASONS)}
    out = out.assign(_o=out["reason"].map(order)).sort_values(["_o", "name"], kind="mergesort")
    return out[cols].reset_index(drop=True)


def left_summary(left: pd.DataFrame) -> str:
    """'🔴 2 profit fell · 🟠 1 no longer revenue-led' — the counts, worst first; '' when none left."""
    counts = left["reason"].value_counts()
    return " · ".join(f"{EXIT_REASONS[r][0]} {counts[r]} {EXIT_REASONS[r][1]}"
                      for r in EXIT_REASONS if r in counts)


def left_legend() -> str:
    """One line per reason: its mark, its words and whether it is measured."""
    return "\n".join(f"- {m} **{lab}** — {why}" for m, lab, why in EXIT_REASONS.values())


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
        "_cycle": cc.TextColumn(
            "Cycle ↓", width=w["_cycle"],
            help="✓ = cash conversion cycle down 15+ days with sales up more than 15%, both in the same "
                 "year as the profit and sales 🌱 is judged on (supply-chain velocity). Inside 🌱, stocks "
                 "like this beat the market by +24 to +54 points in each of three years (Oct 2023 to "
                 "Oct 2026), ahead of 🌱 stocks without it (in-sample). A year-old improvement did not help, "
                 "so a company whose balance sheet is still last year's stays blank until it updates. A "
                 "reading aid, not part of 🌱's recipe or record."),
        "_liq_cr": cc.NumberColumn(LIQUIDITY_HEADER, format="%.2f", width=w["_liq_cr"],
                                   help="Average daily traded value over 20 days (average volume × price), "
                                        f"in ₹ crore. Keep one order under about {ORDER_SHARE:.0%} of it: "
                                        f"2.0 → about ₹{order_size_lakh(2.0):g} lakh."),
        # 📈 Technical (2026-10-13): the Deep Scanner's header, format and tip for each column.
        "close_price": cc.NumberColumn("Price ₹", format="%.2f", width=w["close_price"],
                                       help=_SCANNER_HEADER_TIPS["close_price"]),
        "dist_to_vstop": cc.NumberColumn("Stop Δ", format="%.1f%%", width=w["dist_to_vstop"],
                                         help=_SCANNER_HEADER_TIPS["dist_to_vstop"]),
        "d49_momentum_quality": cc.TextColumn("Mom. Quality", width=w["d49_momentum_quality"],
                                              help=_SCANNER_HEADER_TIPS["d49_momentum_quality"]),
        "rsi_14d": cc.NumberColumn("RSI", format="%.0f", width=w["rsi_14d"],
                                   help=_SCANNER_HEADER_TIPS["rsi_14d"]),
        "breakout_score": cc.NumberColumn("Breakout", format="%.0f", width=w["breakout_score"],
                                          help=_SCANNER_HEADER_TIPS["breakout_score"]),
        "crs_52w": cc.NumberColumn("RS 52W", format="%.0f", width=w["crs_52w"],
                                   help=_SCANNER_HEADER_TIPS["crs_52w"]),
    }


# ── The card ───────────────────────────────────────────────────────────────────────────────────────
def _month_span(w: dict) -> str:
    a, b = date.fromisoformat(w["start"]), date.fromisoformat(w["end"])
    return f"{a:%b}→{b:%b}"


def _range(vals, fmt) -> str:
    lo, hi = min(vals), max(vals)
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}–{fmt.format(hi)}"


def non_overlapping(ws) -> int:
    """The most windows that share no days (2026-10-13). A window ending on the day the next starts is
    separate — the first's end price is the second's start. Greedy by earliest end, which is optimal. The
    design record's 4 windows from 3 snapshot dates are 2 independent stretches, not 4 results."""
    n, last = 0, ""
    for w in sorted(ws, key=lambda w: (w["end"], w["start"])):
        if w["start"] >= last:
            n, last = n + 1, w["end"]
    return n


def order_rule(ws) -> dict:
    """ORDER_RULE applied to a scan's unseen windows: state 'pending' until it has ORDER_RULE['windows']
    separate (non-overlapping, earliest end first — the same greedy choice as non_overlapping), finished
    windows carrying a hold reading of at least min_n matches; then 'earned' or 'not earned' on those
    first windows only, so a later window can never reopen it. counted = windows that count so far."""
    usable = [w for w in ws if w.get("final") and int((w.get("hold") or {}).get("n", 0)) >= ORDER_RULE["min_n"]
              and w["hold"].get("momentum_pct") is not None and w["hold"].get("rank_pct") is not None]
    picked, last = [], ""
    for w in sorted(usable, key=lambda w: (w["end"], w["start"])):
        if w["start"] >= last and len(picked) < ORDER_RULE["windows"]:
            picked.append(w)
            last = w["end"]
    out = dict(counted=len(picked), state="pending")
    if len(picked) == ORDER_RULE["windows"]:
        mom = float(np.mean([float(w["hold"]["momentum_pct"]) for w in picked]))
        rank = float(np.mean([float(w["hold"]["rank_pct"]) for w in picked]))
        earned = mom >= ORDER_RULE["min_avg_pct"] and mom > rank
        out.update(state="earned" if earned else "not earned", mom_avg=mom, rank_avg=rank)
    return out


def _hold_line(scan: dict, ws, c: dict, esc) -> str:
    """Under the unseen record: per window, HOLD_K matches a Rs 1 lakh account can hold, picked at random
    — the typical result after costs and its 10th-90th range — and where momentum's and Rank's top HOLD_K
    land among those picks; then, for the scans ORDER_RULE judges, where the rule stands. Empty until a
    window carries the reading."""
    rows = [w for w in ws if "median_pp" in (w.get("hold") or {})]
    if not rows:
        return ""
    k = int(rows[0]["hold"]["k"])
    res = " / ".join(f'{float(w["hold"]["median_pp"]):+.1f}% ({float(w["hold"]["p10_pp"]):+.1f} to '
                     f'{float(w["hold"]["p90_pp"]):+.1f})' for w in rows)
    loss = " / ".join(f'{int(w["hold"]["loss_pct"])}%' for w in rows)
    pct = lambda v: "–" if v is None else f"{int(v)}%"
    mom = " / ".join(pct(w["hold"].get("momentum_pct")) for w in rows)
    rnk = " / ".join(pct(w["hold"].get("rank_pct")) for w in rows)
    h0 = rows[0]["hold"]
    rule = note = ""
    if scan.get("key") in ORDER_RULE["scans"]:
        note = (f' The order rule was written before any result: momentum order only if its top {k} beats '
                f'{ORDER_RULE["min_avg_pct"]:.0f}%+ of random picks on average over {ORDER_RULE["windows"]} '
                f'separate windows, and beats Rank\'s.')
        v = order_rule(ws)
        rule = (f' · Order rule: {v["counted"]} of {ORDER_RULE["windows"]} separate windows in'
                if v["state"] == "pending" else
                f' · Order rule decided: momentum order {v["state"]} (its top {k} beat {v["mom_avg"]:.0f}% of '
                f'random picks on average, Rank\'s {v["rank_avg"]:.0f}%)')
    return (f'<div style="font-size:0.7rem;color:{c["text_secondary"]};margin-top:5px;">Holding {k} of them, '
            f'picked at random: {esc(res)} after costs · lost money in {esc(loss)} of picks · '
            f'momentum\'s top {k} beat {esc(mom)} of them, Rank\'s top {k} {esc(rnk)}{esc(rule)}</div>'
            f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;">From the matches trading '
            f'₹{float(h0["min_cr"]):g} Cr+ a day with one share at most ₹{float(h0["max_price"]):,.0f} (what ₹1 '
            f'lakh across {k} can hold), {int(h0["draws"]):,} random picks each window with a fixed seed, a '
            f'{float(h0["cost_pp"]):.1f}pp round trip.{esc(note)}</div>')


def _overlap_note(ws, sep: str, end: str = "") -> str:
    """'<sep>K non-overlapping<end>' when the windows overlap; nothing when every one is independent."""
    k = non_overlapping(ws)
    return f"{sep}{k} non-overlapping{end}" if k < len(ws) else ""


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


def _first_unseen_due(scan: dict = None) -> str:
    """When the first unseen window can exist: a start must be a month-end copy taken after the scan's
    design date (in the scan's start month, if it has one), and a window needs the scan's horizon — so
    about that many days after that first start. The year is shown when it differs from the design
    year."""
    scan = scan or {}
    d = date.fromisoformat(scan_designed_on(scan))
    m = scan_start_month(scan)
    start = _month_end(d.year, d.month)
    if m is not None:
        start = _month_end(d.year, m)
        if start <= d:
            start = _month_end(d.year + 1, m)
    due = start + timedelta(days=scan_horizon(scan))
    return f"{due:%d %b %Y}" if due.year != d.year else f"{due:%d %b}"


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


def _index_readings(scan: dict, ws) -> list:
    """For each unseen window that carries the comparison: (window, edge) — the portfolio's return after
    costs minus the scan's own index after the fund fee, in points; edge None where the window had too
    few liquid matches for a portfolio or no reading of that index."""
    code = scan.get("benchmark")
    out = []
    for w in ws:
        v = w.get("vs_index")
        if not code or not v:
            continue
        p, i = v.get("portfolio_pp"), (v.get("indices") or {}).get(code)
        out.append((w, None if p is None or i is None else float(p) - float(i)))
    return out


def _vs_index(scan: dict, ws, c: dict, esc) -> str:
    """Under the unseen record (December plan rule 23): the scan's liquid matches as one equal-weight
    portfolio against the index fund it is judged on, both after costs, per window — every recorded
    index one hover away. A progress reading: the plan's verdict needs 3+ years. Empty until a window
    carries the comparison (the indices are archived from 2026-10-08)."""
    rows = _index_readings(scan, ws)
    if not rows:
        return ""
    code = scan["benchmark"]
    name = BENCHMARK_INDICES.get(code, code)
    cells = []
    for w, edge in rows:
        v = w["vs_index"]
        if edge is None:
            why = "fewer than 30 liquid matches" if v.get("portfolio_pp") is None else f"no {name} reading"
            cells.append(f'<span title="{esc(why)}">–</span>')
            continue
        p = float(v["portfolio_pp"])
        every = " · ".join(f"{BENCHMARK_INDICES.get(k, k)} {p - float(x):+.1f}" for k, x in v["indices"].items())
        tip = (f'{w["start"]} → {w["end"]}: portfolio {p:+.1f}% vs {name} {float(v["indices"][code]):+.1f}%, '
               f'both after costs, {int(v["n"])} stocks. Against every index: {every}')
        col = c["green"] if edge > 0 else c["red"]
        cells.append(f'<span title="{esc(tip)}" style="color:{col};font-weight:700;">{edge:+.1f}</span>')
    measured = [e for _, e in rows if e is not None]
    tail = (f' · ahead in {sum(e > 0 for e in measured)} of {len(measured)}' if measured
            else ' · too few liquid matches to measure')
    v0 = rows[0][0]["vs_index"]
    return (f'<div style="font-size:0.7rem;color:{c["text_secondary"]};margin-top:5px;">Against the '
            f'{esc(name)} index fund — what you could hold instead — after costs: {" / ".join(cells)} pts'
            f'{tail} · a progress reading, the verdict needs 3+ years</div>'
            f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;">Equal-weight portfolio of '
            f'the ₹{float(v0["min_cr"]):g} Cr+/day matches, charged a {float(v0["cost_pp"]):.1f}pp round trip '
            f'each window, against the plain index (no dividends, like the stocks) less a '
            f'{float(v0["fee_pct_yr"]):.1f}%/yr fund fee. Hover a number for every index.</div>')


def _year_bars(years: list, cleanest: str, c: dict, esc) -> str:
    """One bar per one-year window: the matches' median return above the market's median stock, in
    points. Green above, red below; scaled to the largest year, so the years compare by height. The
    cleanest test carries a ✦."""
    out = ""
    top = max([abs(float(y["match_excess_pp"])) for y in years] + [1.0])
    for y in years:
        v = float(y["match_excess_pp"])
        col = c["green"] if v > 0 else c["red"]
        h = 6 + round(28 * abs(v) / top)
        mark = " ✦" if y["label"] == cleanest else ""
        tip = (f'{y["window"]}, judged on {y["known"]} results: matches {v:+.1f} vs rest '
               f'{float(y["rest_excess_pp"]):+.1f} points above the market median · beat it '
               f'{int(y["match_beat_pct"])}% vs {int(y["rest_beat_pct"])}% · {int(y["n_match"])} vs '
               f'{int(y["n_rest"])} stocks')
        out += (f'<div style="display:flex;flex-direction:column;align-items:center;justify-content:flex-end;'
                f'min-width:46px;" title="{esc(tip)}">'
                f'<div style="width:14px;height:{h}px;background:{col};border-radius:3px;"></div>'
                f'<div style="font-size:0.78rem;font-weight:800;color:{col};margin-top:3px;">{v:+.1f}</div>'
                f'<div style="font-size:0.6rem;color:{c["text_muted"]};">{esc(y["label"])}{mark}</div></div>')
    return out


def year_claims(years: list) -> tuple:
    """(years ahead, of how many, the controls that held in EVERY year) — computed from the record, so
    the card never claims more than it measured. A control a year did not measure (no momentum before
    the first window) is judged on the years that did."""
    ahead = sum(float(y["match_excess_pp"]) > float(y["rest_excess_pp"]) for y in years)
    controls = []
    if years and all(all(float(m) > float(r) for m, r in y["size_pp"].values()) for y in years):
        controls.append("inside every size group")
    if years and all(float(y["match_vs_industry_pp"]) > float(y["rest_vs_industry_pp"]) for y in years):
        controls.append("against its own industry")
    held = [float(y["momentum_held_pp"]) for y in years if y.get("momentum_held_pp") is not None]
    if held and all(v > 0 for v in held):
        controls.append("with last year's momentum held")
    return ahead, len(years), controls


def _study_body(scan: dict, rec: dict, c: dict, esc) -> str:
    """A record that is a STUDY, not saved windows (🌱, 2026-10-07): traits known at the start, returns
    over the years after. Named in-sample (the recipe was chosen on it), its caveats on the card, and
    the share of matches that did NOT double, so 'better odds' never reads as 'will compound'. When the
    study has a year-by-year record, one bar per year (each judged only on results public at its
    start) with the controls that held in every year. Then the out-of-sample section: the scan's own
    long windows, or when the first is due."""
    st_ = rec["study"]
    muted = f'font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;'
    years = st_.get("years") or []
    cleanest = (st_.get("cleanest") or {}).get("label", "")
    if years:
        bars = _year_bars(years, cleanest, c, esc)
        bars_tip = "Each year: matches' median return above the market's median stock, in points"
    else:
        bars, bars_tip = "", ""
        for label, v in (("Matches", st_["match_cagr_pct"]), ("Rest", st_["rest_cagr_pct"])):
            col = c["green"] if v > 0 else c["red"]
            h = 6 + round(28 * min(abs(v) / 12.0, 1.0))
            bars += (f'<div style="display:flex;flex-direction:column;align-items:center;justify-content:flex-end;'
                     f'min-width:46px;"><div style="width:14px;height:{h}px;background:{col};border-radius:3px;">'
                     f'</div><div style="font-size:0.78rem;font-weight:800;color:{col};margin-top:3px;">'
                     f'{v:+.1f}%</div><div style="font-size:0.6rem;color:{c["text_muted"]};">{label}</div></div>')
    body = (f'<div style="display:flex;flex-wrap:wrap;gap:6px 14px;align-items:flex-end;justify-content:space-between;">'
            f'<div style="flex:1 1 200px;min-width:0;">'
            f'<div style="font-size:0.66rem;font-weight:700;color:{c["text_muted"]};letter-spacing:.3px;">'
            f'{esc(st_["title"].upper())} · IN-SAMPLE</div>'
            f'<div style="{muted}">Traits known at the start ({esc(st_["from"])}); returns to '
            f'{esc(st_["to"])}</div>'
            f'<div style="font-size:0.74rem;color:{c["text_secondary"]};margin-top:3px;">'
            f'<b>{st_["match_cagr_pct"]:+.1f}%/yr</b> vs {st_["rest_cagr_pct"]:+.1f}%/yr for the rest</div>'
            f'<div style="{muted}">Doubled: {st_["match_doubled_pct"]:.1f}% vs {st_["rest_doubled_pct"]:.1f}% · '
            f'{int(st_["n_match"]):,} vs {int(st_["n_rest"]):,} stocks</div>'
            f'</div><div style="display:flex;gap:4px;align-items:flex-end;" title="{esc(bars_tip)}">{bars}</div></div>')
    if years:
        ahead, total, controls = year_claims(years)
        head = (f"Ahead in all {total} years" if ahead == total else f"Ahead in {ahead} of {total} years")
        beat_m = " / ".join(f'{int(y["match_beat_pct"])}%' for y in years)
        beat_r = " / ".join(f'{int(y["rest_beat_pct"])}%' for y in years)
        body += (f'<div style="font-size:0.7rem;color:{c["text_secondary"]};margin-top:6px;">'
                 f'<b>Year by year</b> (each judged on results public at its start): {esc(head)}'
                 f'{" — " + esc(", ".join(controls)) if controls else ""}.</div>'
                 f'<div style="{muted}">Beat the market: {beat_m} of matches, against {beat_r} of the rest.</div>')
        if cleanest:
            body += (f'<div style="{muted}">✦ Cleanest test: {esc(cleanest)} — '
                     f'{esc(st_["cleanest"].get("why", ""))}.</div>')
    body += (f'<div style="font-size:0.7rem;color:{c["text_secondary"]};margin-top:5px;">Better odds, not a '
             f'promise: {100 - st_["match_doubled_pct"]:.0f}% of matches did not double.</div>'
             f'<div style="{muted}">{esc(st_["caveats"])} {esc(st_["definition_note"])}'
             f'{" " + esc(st_["years_note"]) if years and st_.get("years_note") else ""}</div>')
    unseen = rec.get("out_of_sample") or []
    built = f"{date.fromisoformat(scan_designed_on(scan)):%d %b %Y}"
    years = scan_horizon(scan) / 365
    span = f"{years:g}-year" if years >= 1 else f"{scan_horizon(scan)}-day"
    if unseen:
        u_beat = _range([int(w["beat_pct"]) for w in unseen], "{}") + "%"
        body += (f'<div style="display:flex;flex-wrap:wrap;gap:6px 14px;align-items:flex-end;justify-content:space-between;'
                 f'border-top:1px solid {c["border"]};margin-top:8px;padding-top:7px;">'
                 f'<div style="flex:1 1 200px;min-width:0;">'
                 f'<div style="font-size:0.66rem;font-weight:700;color:{c["text_muted"]};letter-spacing:.3px;">'
                 f'OUT-OF-SAMPLE · SINCE IT WAS BUILT ({esc(built.upper())}) — WINDOWS IT NEVER SAW</div>'
                 f'<div style="font-size:0.74rem;color:{c["text_secondary"]};margin-top:3px;">'
                 f'{u_beat} of matches beat the market · before costs · the most trustworthy record{_overlap_note(unseen, " · ")}</div></div>'
                 f'<div style="display:flex;gap:4px;align-items:flex-end;">{_bars(unseen, c, esc)}</div></div>'
                 f'{_extras(unseen, c, esc)}{_vs_index(scan, unseen, c, esc)}{_hold_line(scan, unseen, c, esc)}')
    else:
        body += (f'<div style="border-top:1px solid {c["border"]};margin-top:8px;padding-top:7px;{muted}">'
                 f'<b style="color:{c["text_secondary"]};">Out-of-sample: pending</b> — this list is judged '
                 + (f'on one {span} window a year, each starting at the end of '
                    f'{date(2000, scan_start_month(scan), 1):%B}, after it was built ({built})'
                    if scan_start_month(scan) else f'on {span} windows that start after it was built ({built})')
                 + f'; the first is due about {_first_unseen_due(scan)}.</div>')
    return body


def _card_head(scan: dict, n_now: int, c: dict, esc) -> str:
    """Name, today's count and the recipe in words — the same on the summary card and the full record."""
    return (f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;">'
            f'<span style="font-size:1.0rem;font-weight:800;color:{c["text_primary"]};">'
            f'{scan["emoji"]} {esc(scan["name"])}</span>'
            f'<span style="font-size:1.35rem;font-weight:900;color:{c["gold"]};">{n_now:,}'
            f'<span style="font-size:0.7rem;font-weight:600;color:{c["text_muted"]};"> now</span></span></div>'
            f'<div style="font-size:0.76rem;color:{c["text_secondary"]};margin:4px 0 10px 0;line-height:1.35;">'
            f'{esc(scan["recipe"])}</div>')


def card_html(scan: dict, n_now: int, state: str, rec, selected: bool, latest: str,
              with_head: bool = True) -> str:
    """One scan card: name, today's count, the recipe in words, and two records — the IN-SAMPLE one
    (the design period: windows studied while the recipe was chosen, frozen) and the OUT-OF-SAMPLE one
    (windows that start after DESIGNED_ON, measured automatically from the PRISM Archive). Each shows
    one bar per window, the share of matches that beat the market, the sample size, the net-of-costs
    reading, what was left out and "provisional". A stale or missing record shows no numbers at all."""
    c = COLORS
    esc = html.escape
    border = c["gold"] if selected else c["border"]
    head = _card_head(scan, n_now, c, esc) if with_head else ""
    if state == "measured" and rec and rec.get("study"):
        body = _study_body(scan, rec, c, esc)
    elif state == "measured" and rec and rec.get("windows"):
        # One ROW per record — the words on the left, the bars on the right — so the card is about half
        # the height of a stacked one and the list starts on the first screen (2026-10-07).
        ws = rec["windows"]
        beat = _range([int(w["beat_pct"]) for w in ws], "{}") + "%"
        n = _range([int(w["n"]) for w in ws], "{}")
        starts = len({w["start"] for w in ws})
        built = f"{date.fromisoformat(scan_designed_on(scan)):%d %b}"
        unseen = rec.get("out_of_sample") or []
        bars_tip = ("Points above the market median in each window, before costs; hover a bar for its "
                    "dates and size. In-sample = windows studied while the recipe was chosen.")
        # Named in so many words (2026-10-07): "provisional" alone read as "proven, just early".
        body = (f'<div style="display:flex;flex-wrap:wrap;gap:6px 14px;align-items:flex-end;justify-content:space-between;">'
                f'<div style="flex:1 1 200px;min-width:0;">'
                f'<div style="font-size:0.66rem;font-weight:700;color:{c["text_muted"]};letter-spacing:.3px;">'
                f'IN-SAMPLE · DESIGN-PERIOD RECORD</div>'
                f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;">The recipe was chosen '
                f'on these windows: worth testing, not proven</div>'
                f'<div style="font-size:0.74rem;color:{c["text_secondary"]};margin-top:3px;">'
                f'{beat} of matches beat the market · before costs</div>'
                f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;">{n} stocks per '
                f'window · provisional</div>'
                f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;">{len(ws)} short '
                f'windows from {starts} start date{"s" if starts != 1 else ""}{_overlap_note(ws, ", ")}'
                f'{f", measured to {esc(latest)}" if latest else ""}</div>'
                + ("" if unseen else
                   f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;" title="Windows '
                   f'that start after the recipe was chosen ({built}) — data nobody looked at while '
                   f'choosing it.">Out-of-sample: pending — unseen windows: none yet, the first is due '
                   f'about {_first_unseen_due(scan)}</div>')
                + f'</div><div style="display:flex;gap:4px;align-items:flex-end;" title="{esc(bars_tip)}">'
                f'{_bars(ws, c, esc)}</div></div>{_extras(ws, c, esc)}')
        if unseen:
            u_beat = _range([int(w["beat_pct"]) for w in unseen], "{}") + "%"
            body += (f'<div style="display:flex;flex-wrap:wrap;gap:6px 14px;align-items:flex-end;justify-content:space-between;'
                     f'border-top:1px solid {c["border"]};margin-top:8px;padding-top:7px;">'
                     f'<div style="flex:1 1 200px;min-width:0;">'
                     f'<div style="font-size:0.66rem;font-weight:700;color:{c["text_muted"]};letter-spacing:.3px;">'
                     f'OUT-OF-SAMPLE · SINCE IT WAS BUILT ({esc(built.upper())}) — WINDOWS IT NEVER SAW</div>'
                     f'<div style="font-size:0.74rem;color:{c["text_secondary"]};margin-top:3px;">'
                     f'{u_beat} of matches beat the market · before costs · the most trustworthy record{_overlap_note(unseen, " · ")}</div></div>'
                     f'<div style="display:flex;gap:4px;align-items:flex-end;">{_bars(unseen, c, esc)}</div></div>'
                     f'{_extras(unseen, c, esc)}{_vs_index(scan, unseen, c, esc)}{_hold_line(scan, unseen, c, esc)}')
    elif state == "unverified":
        body = (f'<div style="font-size:0.74rem;color:{c["text_muted"]};">The record could not be checked against '
                f'the recipe just now — the app may be updating. Reload the page in a minute.</div>')
    elif state == "stale":
        body = (f'<div style="font-size:0.74rem;color:{c["gold"]};">The recipe changed since its record was '
                f'measured — no record is shown until it is re-measured.</div>')
    else:
        body = f'<div style="font-size:0.74rem;color:{c["text_muted"]};">Not yet measured.</div>'
    return (f'<div style="background:{c["bg_secondary"]};border:1px solid {border};border-radius:10px;'
            f'padding:12px 14px;">{head}{body}</div>')


def _pts(vals) -> str:
    """'+2.9 to +10.3' (or one value): points above the market's median stock."""
    lo, hi = min(vals), max(vals)
    return f"{lo:+.1f}" if lo == hi else f"{lo:+.1f} to {hi:+.1f}"


# Every summary card is this tall, so the three "Show list" buttons line up: the tallest summary any
# record can produce, at 345px wide. Measured 2026-10-09 in the app at 1536px: today 192 / 192 / 210;
# with the out-of-sample line against the index fund, Playbook and Leading-Sector 207 and 🌱 224 (its
# long fund name takes a second line from Oct 2027). 224 keeps them aligned in every state.
SUMMARY_MIN_HEIGHT = 224


def card_summary_html(scan: dict, n_now: int, state: str, rec, selected: bool) -> str:
    """The card the user CHOOSES from (2026-10-08): name, count, recipe, and one honest headline —
    in-sample named, before and after costs, how many windows or years it was ahead in, and when the
    first out-of-sample test is due. Every number here is computed from the same record card_html
    shows in full under the cards, so the two cannot disagree. Three full cards side by side were
    406 / 408 / 596px tall and pushed the list ~240px below a laptop's first screen."""
    c = COLORS
    esc = html.escape
    border = c["gold"] if selected else c["border"]
    label = f'font-size:0.66rem;font-weight:700;color:{c["text_muted"]};letter-spacing:.3px;'
    main = f'font-size:0.74rem;color:{c["text_secondary"]};margin-top:3px;line-height:1.35;'
    muted = f'font-size:0.66rem;color:{c["text_muted"]};margin-top:3px;line-height:1.35;'
    unseen = (rec or {}).get("out_of_sample") or []
    if unseen:
        u_ahead = sum(float(w["median_excess_pp"]) > 0 for w in unseen)
        oos = (f'Out-of-sample: ahead in {u_ahead} of {len(unseen)} unseen window'
               f'{"s" if len(unseen) != 1 else ""} ({_pts([float(w["median_excess_pp"]) for w in unseen])} pts)'
               f'{_overlap_note(unseen, " · ")}')
    else:
        oos = f"Out-of-sample: pending · first test due about {_first_unseen_due(scan)}"
    # Once the unseen windows carry the index comparison (December plan rule 23), it IS the out-of-sample
    # line: after costs, against the fund the user could hold instead — the strictest reading, in the
    # same space, so the three cards stay one height (the full record below keeps both readings).
    read = [(w, e) for w, e in _index_readings(scan, unseen) if e is not None]
    edges = [e for _, e in read]
    if edges:
        oos = (f"Out-of-sample, after costs: ahead of the {BENCHMARK_INDICES.get(scan['benchmark'], scan['benchmark'])}"
               f" index fund in {sum(e > 0 for e in edges)} of {len(edges)} window{'s' if len(edges) != 1 else ''}"
               f" ({_pts(edges)} pts){_overlap_note([w for w, _ in read], ' · ')}")
    if state == "measured" and rec and rec.get("study"):
        st_ = rec["study"]
        years = st_.get("years") or []
        ahead, total, _ = year_claims(years) if years else (0, 0, [])
        body = (f'<div style="{label}">IN-SAMPLE · {esc(st_["title"].upper())} · BEFORE COSTS</div>'
                f'<div style="{main}"><b>{st_["match_cagr_pct"]:+.1f}%/yr</b> vs '
                f'{st_["rest_cagr_pct"]:+.1f}%/yr for the rest</div>'
                f'<div style="{muted}">' + (f'Ahead in {ahead} of {total} years · ' if total else "")
                + f'{100 - st_["match_doubled_pct"]:.0f}% of matches did not double</div>'
                f'<div style="{muted}">{esc(oos)}</div>')
    elif state == "measured" and rec and rec.get("windows"):
        ws = rec["windows"]
        gross = [float(w["median_excess_pp"]) for w in ws]
        nets = [float(w["net"]["median_excess_pp"]) for w in ws
                if (w.get("net") or {}).get("median_excess_pp") is not None]
        after = ""
        if nets:
            after = f", {_pts(nets)} after costs" + ("" if len(nets) == len(ws) else
                                                    f" ({len(nets)} of {len(ws)} windows)")
        body = (f'<div style="{label}">IN-SAMPLE · DESIGN-PERIOD RECORD</div>'
                f'<div style="{main}">Ahead of the market in <b>{sum(v > 0 for v in gross)} of {len(ws)}</b> '
                f'windows{_overlap_note(ws, " (", ")")}: {_pts(gross)} pts before costs{after}</div>'
                f'<div style="{muted}">{esc(oos)}</div>')
    elif state == "unverified":
        body = (f'<div style="font-size:0.74rem;color:{c["text_muted"]};">The record could not be checked against '
                f'the recipe just now — the app may be updating. Reload the page in a minute.</div>')
    elif state == "stale":
        body = (f'<div style="font-size:0.74rem;color:{c["gold"]};">The recipe changed since its record was '
                f'measured — no record is shown until it is re-measured.</div>')
    else:
        body = f'<div style="font-size:0.74rem;color:{c["text_muted"]};">Not yet measured.</div>'
    return (f'<div style="background:{c["bg_secondary"]};border:1px solid {border};border-radius:10px;'
            f'padding:12px 14px;min-height:{SUMMARY_MIN_HEIGHT}px;">{_card_head(scan, n_now, c, esc)}{body}</div>')


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
