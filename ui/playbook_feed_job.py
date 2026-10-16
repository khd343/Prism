"""
🚀 The Playbook feed (2026-10-12) — PRISM's daily list for Kite Algo. Run by GitHub Actions
(.github/workflows/playbook-feed.yml), or locally once to back-fill:

    python -m ui.playbook_feed_job --feed feed                                      # needs SHEET_ID
    python -m ui.playbook_feed_job --feed <dir> --backfill "Other Resources/archive-cache"

PRISM picks the stocks and Kite Algo trades them, and the two engines never share code: they meet at ONE
file per trading session on the `playbook-feed` branch (the contract is readme_text() below; the design is
docs/superpowers/specs/2026-10-12-playbook-feed-design.md). A wrong list would be TRADED, while a missing one
only pauses entries, so every rule here fails closed:

* The session comes from the sheet's own name, read from the SAME response as the data. Prism.gs renames
  the sheet only after all six tabs are written and pass its sanity gate, so the name is the "data
  complete" marker; the clock, or a second request, could stamp one session's list with another's date.
* PRISM's own 🩺 checks (ui/data_check_job.py, imported, never copied) gate the publish. Any failure
  publishes NOTHING and fails the run — GitHub emails the owner.
* The recipe is ui_scans.playbook_shortlist, imported, never re-typed; its fingerprint rides in the
  manifest and Kite pins it, so a recipe change is a deliberate change in both repositories.
* A published session is never rewritten: identical bytes are a no-op, different bytes an error.
* Codes, never display labels: PRISM's labels carry emoji and have been renamed before.
* Byte-deterministic, so a local back-fill and a CI publish of one session hash the same.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from core import data_engine as de
from core import run_scoring_pipeline
from core.sheet_meta import IST, _DATA_READY, _parse_title, expected_session, fetch_sheet_title, parse_data_date
from ui import data_check_job as D
from ui import ui_scans as S
from ui.scan_records_job import engine_version, sheet_id_from_secret

SCHEMA = 2                    # 2 (2026-10-16): the manifest also carries `blanks`; every earlier session is schema 1
SESSIONS = "sessions"
COLUMNS = ("symbol", "name", "sector", "industry", "cap", "composite_score", "crs_52w", "dist_to_vstop",
           "momentum_score", "breakout_score", "traded_value_cr", "rsi_14d", "stage", "close_price",
           "equity_shares")
TEXT = ("name", "sector", "industry")
NUMBERS = ("composite_score", "crs_52w", "dist_to_vstop", "momentum_score", "breakout_score",
           "traded_value_cr", "rsi_14d", "close_price")
SOURCE_COLUMNS = ("company_id", "market_category", "weinstein_stage", "equity_shares", *TEXT, *NUMBERS)
# The six vendor categories and the engine's five stage labels — every value seen on all 24 archives
# (2026-10-12). A blank source becomes an empty code (unknown); a label not listed here fails the publish.
CAP_CODES = {"Mega Cap": "MEGA", "Large Cap": "LARGE", "Mid Cap": "MID", "Small Cap": "SMALL",
             "Micro Cap": "MICRO", "Nano Cap": "NANO"}
STAGE_CODES = {"📈 Stage 2 Advancing": "2", "🔄 Stage 1 Basing": "1", "⚠️ Stage 3 Top": "3",
               "📉 Stage 4 Declining": "4", "❔ Unknown": ""}
FLOAT_FORMAT = "%.10g"
# Measured 2026-10-12 on all 24 archives: 128 members on 29 Jun (a 2,107-stock universe), 262-288 on every
# other copy. Wide enough that a normal day never trips it; an empty or exploded list always does.
_MEASURED_BOUNDS = (100, 450)
MIN_ROWS, MAX_ROWS = _MEASURED_BOUNDS
SYMBOL = re.compile(r"(NSE|BSE):[A-Z0-9&_.-]+")
GITATTRIBUTES = b"* -text\n"


class FeedError(Exception):
    """A list that must not be published. The run fails and GitHub emails; nothing is written."""


def playbook_scan() -> dict:
    return next(s for s in S.SCANS if s["key"] == "playbook")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ── The contract: one row per Playbook member ─────────────────────────────────────────────────────
def _codes(raw: pd.Series, codes: dict, what: str) -> pd.Series:
    filled = raw.notna() & (raw.astype(str).str.strip() != "")
    unknown = sorted(set(raw[filled].astype(str)) - set(codes))
    if unknown:
        raise FeedError(f"unmapped {what} label(s) {unknown} — add them to the code map, never guess")
    return raw.astype(str).map(codes).where(filled, "")


def build_frame(scored: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(SOURCE_COLUMNS) - set(scored.columns))
    if missing:
        raise FeedError(f"the scored data lacks {missing}")
    pb = scored.loc[S.playbook_shortlist(scored)]
    out = pd.DataFrame(index=pb.index)
    out["symbol"] = pb["company_id"].astype(str)
    for c in TEXT:
        out[c] = pb[c].where(pb[c].notna(), "").astype(str)
    out["cap"] = _codes(pb["market_category"], CAP_CODES, "market_category")
    for c in NUMBERS:
        out[c] = pd.to_numeric(pb[c], errors="coerce")
    out["stage"] = _codes(pb["weinstein_stage"], STAGE_CODES, "weinstein_stage")
    out["equity_shares"] = pd.to_numeric(pb["equity_shares"], errors="coerce").round().astype("Int64")
    return out[list(COLUMNS)].sort_values("symbol", kind="mergesort").reset_index(drop=True)


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, float_format=FLOAT_FORMAT, lineterminator="\n").encode("utf-8")


def validate(frame: pd.DataFrame) -> None:
    n = len(frame)
    if not MIN_ROWS <= n <= MAX_ROWS:
        raise FeedError(f"{n} Playbook stocks, outside the measured {MIN_ROWS}-{MAX_ROWS}")
    dup = sorted(frame.loc[frame["symbol"].duplicated(), "symbol"].unique())
    if dup:
        raise FeedError(f"duplicate symbol(s) {dup[:5]}")
    bad = sorted(frame.loc[~frame["symbol"].str.fullmatch(SYMBOL.pattern), "symbol"])
    if bad:
        raise FeedError(f"malformed symbol(s) {bad[:5]}")
    for c in ("symbol", "name"):
        if (frame[c].str.strip() == "").any():
            raise FeedError(f"empty {c} on {int((frame[c].str.strip() == '').sum())} row(s)")
    px = frame["close_price"]
    if (px.isna() | (px <= 0)).any():
        raise FeedError(f"close_price missing or not positive on {int((px.isna() | (px <= 0)).sum())} row(s)")


# ── What a list could not judge (2026-10-16) ──────────────────────────────────────────────────────────────────────
# Kite sells a held stock once it is out of the list 3 sessions running. A stock can be out because an input of the
# recipe came in BLANK — the vendor sent no CRS for it that day, a quarter half-filled in results season — and then the
# recipe never judged it out (day to day, 10 Sep - 9 Oct 2026: 14 times in 6,017 member-days a member lost an input it
# had the day before, and 2 of those left the list that day). So each list states its blanks and Kite decides with
# history: an absence counts as nothing when every condition the stock failed rests on an input it had on its last
# list. What each of ui_scans.playbook_conditions reads, pinned to the recipe's code and on a real copy
# (tests/test_playbook_blanks.py); "dist_to_vstop" stands for the stop, which the trend score reads too.
RECIPE_INPUTS = {
    "momentum": ("adx_14w", "close_price", "crs_26w", "crs_50d", "crs_52w", "dist_13wh", "dist_52wh", "dist_52wh_days",
                 "dist_ath", "dist_to_vstop", "last_vstop_change", "ret_vs_industry_1y", "ret_vs_industry_3m",
                 "rsi_14d", "sector", "sma_200d", "vol_sma_20d", "volume"),
    "breakout": ("dist_13wh", "dist_52wh", "dist_52wh_days", "dist_ath"),
    "quarter": ("ebitda_lq", "ebitda_pyq", "pat_lq", "pat_pyq", "rev_lq", "rev_pyq"),
    "improvement": ("opm", "opm_1yb", "pat", "pat_1yb", "revenue", "revenue_1yb"),
    "evidence": ("opm", "opm_1yb", "pat", "pat_1yb", "revenue", "revenue_1yb"),
    "stop": ("close_price", "dist_to_vstop"),
}
# A blank that is the vendor's ANSWER, not a gap (StockScans' own definitions): no earlier close that high, no golden
# cross in force. The recipe reads them as such, so they are never listed; only their floors guard an outage.
BLANK_IS_A_FACT = ("breakout_window", "golden_cross_days")
TEXT_INPUTS = ("sector",)


def filled(scored: pd.DataFrame, col: str) -> pd.Series:
    """True where a stock carries the reading: a number, or for a text input non-blank text. An absent column is
    empty on every stock, never skipped."""
    if col not in scored.columns:
        return pd.Series(False, index=scored.index)
    return D._filled(scored[col]) if col in TEXT_INPUTS else pd.to_numeric(scored[col], errors="coerce").notna()


def _named(flags: pd.DataFrame, ids: pd.Series) -> dict:
    """{ids[row]: [the columns True in that row, sorted]} for every row with one, keys sorted. Rows are positions."""
    hit = flags.stack()
    hit = hit[hit]
    pairs = pd.DataFrame({"symbol": ids.to_numpy()[hit.index.get_level_values(0).to_numpy(dtype=int)],
                          "input": hit.index.get_level_values(1)}).sort_values(["symbol", "input"], kind="mergesort")
    return {s: g.tolist() for s, g in pairs.groupby("symbol", sort=True)["input"]}


def blanks(scored: pd.DataFrame) -> dict:
    """The manifest's `blanks` (schema 2): {"in": {member: [inputs that came in blank]}, "out": {stock: {condition:
    [inputs]}}}. "out" holds only a stock off the list whose EVERY failed condition has a blank input — one condition
    failed on the data it has is a judgement — with each failed condition's blank inputs. Blank-is-a-fact columns are
    never listed. Vectorized; positions throughout, so a repeated index label cannot misalign a row."""
    ids = scored["company_id"].astype(str).reset_index(drop=True)
    cond = S.playbook_conditions(scored).reset_index(drop=True)
    inputs = sorted({c for v in RECIPE_INPUTS.values() for c in v})
    blank = pd.DataFrame({c: ~filled(scored, c).to_numpy() for c in inputs})
    member = cond.all(axis=1)
    rests = ~member & pd.concat([cond[c] | blank[list(v)].any(axis=1) for c, v in RECIPE_INPUTS.items()],
                                axis=1).all(axis=1)
    out = {}
    for c, v in RECIPE_INPUTS.items():
        for sym, names in _named(blank.loc[rests & ~cond[c], list(v)], ids).items():
            out.setdefault(sym, {})[c] = names
    listed = set(ids[member])                  # a repeated company_id can be a member on one row: it is not off the list
    return {"in": _named(blank.loc[member], ids),
            "out": {s: dict(sorted(out[s].items())) for s in sorted(out) if SYMBOL.fullmatch(s) and s not in listed}}


def _counts(s: pd.Series, codes: dict) -> dict:
    return {k: int((s == k).sum()) for k in sorted(set(codes.values()) | {""})}


def manifest(frame: pd.DataFrame, csv: bytes, session: str, source_title: str, source: bytes, origin: str,
             trigger: str, checks: dict, now: datetime, blanks: dict) -> dict:
    return {
        "schema": SCHEMA, "session": session, "rows": int(len(frame)), "columns": list(COLUMNS),
        "csv_sha256": sha256(csv), "recipe": "playbook",
        "recipe_fingerprint": S.recipe_fingerprint(playbook_scan()),
        "engine_commit": os.environ.get("FEED_CODE_SHA") or os.environ.get("GITHUB_SHA") or engine_version(),
        "source_title": source_title, "source_sha256": sha256(source),
        "origin": origin, "trigger": trigger, "published_at": now.isoformat(timespec="seconds"),
        "counts": {"cap": _counts(frame["cap"], CAP_CODES), "stage": _counts(frame["stage"], STAGE_CODES)},
        "checks": checks,
        "blanks": blanks,
    }


def readme_text() -> str:
    rows = "\n".join(f"| {i} | `{c}` |" for i, c in enumerate(COLUMNS, 1))
    return (f"# Playbook feed — schema {SCHEMA}\n\n"
            "PRISM's 🚀 Playbook Shortlist, one session per file, published by `ui/playbook_feed_job.py` on "
            "`main`. Each `sessions/<YYYY-MM-DD>.csv` is written once, never modified, never deleted; its "
            "`.json` manifest carries the session (from the sheet's own name), the row count, the CSV's "
            "sha256, the recipe fingerprint, the engine commit, the source workbook's sha256, and whether "
            "it was `recorded` as the session closed or `backfilled` later.\n\n"
            "Only Playbook members are listed: for a valid session, absence means \"not in the list today\". "
            "A missing or invalid session means UNKNOWN, never \"out of the list\".\n\n"
            f"| # | Column |\n|---|---|\n{rows}\n\n"
            f"`cap`: {' '.join(sorted(set(CAP_CODES.values())))} or empty. "
            "`stage`: Weinstein 1 2 3 4 or empty. Rows sorted by `symbol`; UTF-8, `\\n`; empty = not known.\n\n"
            "Since schema 2 (2026-10-16) the manifest also carries `blanks`, the recipe inputs that came in blank: "
            "`in` names each member's (it passed without them); `out` names each stock off the list whose every "
            f"failed condition ({', '.join(RECIPE_INPUTS)}) rests on a blank input, with those inputs per condition. "
            "A stock off the list and not in `out` failed a condition on the data it has. Symbols and inputs sorted. "
            "A schema 1 manifest has no `blanks`: it says nothing about blank inputs.\n")


# ── The branch: written once ──────────────────────────────────────────────────────────────────────
def sessions_on_branch(feed: Path) -> list:
    d = Path(feed) / SESSIONS
    return sorted(p.stem for p in d.glob("*.csv")) if d.is_dir() else []


def write_session(feed: Path, session: str, csv: bytes, meta: dict) -> bool:
    """True when written; False when exactly this session is already there. Different bytes raise."""
    d = Path(feed) / SESSIONS
    path, mpath = d / f"{session}.csv", d / f"{session}.json"
    if path.exists():
        if path.read_bytes() != csv:
            raise FeedError(f"{session} is already published with different content — a published session "
                            "is never rewritten")
        if mpath.exists():
            return False
    d.mkdir(parents=True, exist_ok=True)
    path.write_bytes(csv)
    mpath.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if sha256(path.read_bytes()) != meta["csv_sha256"]:
        raise FeedError(f"{session}: the written file does not match its manifest's checksum")
    readme = Path(feed) / "README.md"
    if not readme.exists() or readme.read_text(encoding="utf-8") != readme_text():
        readme.write_text(readme_text(), encoding="utf-8", newline="\n")
    # Windows git (core.autocrlf) would check a session out with CRLF, breaking its sha256 and making an
    # identical re-publish read as "different content". The branch forbids any conversion.
    attrs = Path(feed) / ".gitattributes"
    if not attrs.exists() or attrs.read_bytes() != GITATTRIBUTES:
        attrs.write_bytes(GITATTRIBUTES)
    return True


# ── Scoring and the gate ──────────────────────────────────────────────────────────────────────────
# A back-filled archive predates two things the vendor added later, which are "not there yet", not broken:
# the index rows (2026-10-08; check_benchmarks is skipped) and the listed_days column (~2026-09-21). Exactly
# that detail is tolerated — anything else on the same check still blocks, and nothing is tolerated live.
BACKFILL_TOLERATED = {("Key columns", "listed_days absent")}
# THE LIVE GATE'S INDEX ROWS ARE THE SEVEN KITE'S TEST OPENED WITH (2026-10-13). The Playbook never reads an index
# row; this check is part of the ingest gate the window opened with on 12 Oct, so it stays exactly that. Indices added
# later for display (2a87ac4: the MidSmallcap 400 pair) must not cost Kite a session: the daily data check still
# requires every one, so a missing row is still mailed.
FEED_GATE_INDICES = ("NSE:CNX500", "NSE:SMALLCA250", "NSE:NFMICRO250", "NSE:NIF500MO50", "NSE:NI200MOM30",
                     "NSE:NQUALITY30", "NSE:NS250MQ100")
# THE RECIPE'S OWN INPUTS (2026-10-14, independent review I4). The data check guards the columns its daily email is
# about; the Playbook also reads the quarterly levels behind 🔥 Strong Quarter, the stop distance, and the moving
# averages behind the stage and the trend scores. A vendor that thins one shrinks and skews the list while every other
# check stays green. Floors sit about ten points under the lowest share measured on all 24 archive copies (29 Jun -
# 9 Oct 2026): latest quarter 99.2%, the stop 99.6%, last year's quarter 95.0%, sma_30w 95.2%, sma_200d 93.0%. The
# scores are always filled (the engine fills a missing input), so a broken input shows as a FLAT score instead.
# EVERY INPUT OF EVERY 🚀 CONDITION (2026-10-16, RECIPE_INPUTS): the same rule — the lowest share on the 24 copies less
# 8 points, rounded down to 0.05, which gives back each floor above — for the rest of what the conditions read. A
# blank-is-a-fact column swings with the market (stocks at an all-time high, golden crosses in force), so its floor
# only catches an outage. The shares behind each: tests/test_playbook_feed.py.
RECIPE_FLOORS = {"rev_lq": 0.90, "pat_lq": 0.90, "ebitda_lq": 0.90, "dist_to_vstop": 0.90,
                 "rev_pyq": 0.85, "pat_pyq": 0.85, "ebitda_pyq": 0.85, "sma_30w": 0.85, "sma_200d": 0.85,
                 "crs_50d": 0.80, "crs_52w": 0.75, "crs_26w": 0.80, "ret_vs_industry_1y": 0.50,
                 "ret_vs_industry_3m": 0.50, "adx_14w": 0.85, "sma_50d": 0.85, "rsi_14d": 0.90,
                 "last_vstop_change": 0.90, "vol_sma_20d": 0.90, "volume": 0.90, "close_price": 0.90, "sector": 0.90,
                 "dist_52wh": 0.90, "dist_52wh_days": 0.90, "dist_13wh": 0.90, "dist_ath": 0.90, "pat": 0.90,
                 "pat_1yb": 0.90, "revenue": 0.90, "revenue_1yb": 0.90, "opm": 0.90, "opm_1yb": 0.90,
                 "breakout_window": 0.50, "golden_cross_days": 0.05}


def check_recipe_inputs(scored: pd.DataFrame) -> list:
    def num(c):                                          # an absent column is empty, never skipped
        return (pd.to_numeric(scored[c], errors="coerce") if c in scored.columns
                else pd.Series(float("nan"), index=scored.index))
    share = {c: float(filled(scored, c).mean()) for c in RECIPE_FLOORS}
    thin = [f"{c} {share[c]:.1%}" for c in RECIPE_FLOORS if share[c] < RECIPE_FLOORS[c]]
    flat = [c for c in ("breakout_score", "improvement_score") if num(c).round(1).nunique() < D.MIN_DISTINCT]
    return [D._ok(not thin, "Recipe inputs", f"below the floor: {thin}" if thin
                  else f"all {len(RECIPE_FLOORS)} at or above their floor"),
            D._ok(not flat, "Recipe scores vary", f"flat: {flat}" if flat else "breakout and improvement vary")]


def score(workbook: bytes):
    """The app's own path on the downloaded bytes (data_engine's upload branch IS the sheet's parse)."""
    with contextlib.redirect_stdout(io.StringIO()):
        datasets = de.load_all_csvs("upload", {"workbook": io.BytesIO(workbook)})
        stocks, bench = de._clean_from(datasets), de.benchmark_table(datasets)
        scored = run_scoring_pipeline(stocks.copy())
    return datasets, stocks, bench, scored


def gate(datasets, stocks, bench, scored, backfill: bool) -> dict:
    results = (D.check_tabs(datasets) + D.check_universe(stocks)
               + ([] if backfill else D.check_benchmarks(bench, FEED_GATE_INDICES))
               + D.check_coverage(stocks) + D.check_types(datasets) + D.check_taxonomy(stocks)
               + D.check_identities(stocks) + D.check_engine(scored)
               + ([] if backfill else check_recipe_inputs(scored)))
    tolerated = [r for r in results if backfill and r.status == "fail" and (r.name, r.detail) in BACKFILL_TOLERATED]
    failed = [r for r in results if r.status == "fail" and r not in tolerated]
    if failed:
        raise FeedError("data check failed — " + "; ".join(f"{r.name}: {r.detail}" for r in failed))
    return {"passed": sum(r.status == "ok" for r in results),
            "warnings": [f"{r.name}: {r.detail}" for r in results if r.status == "warn" or r in tolerated]}


# A STALE COPY IS NOT A SESSION. The sync names a session by the weekday, with no holiday calendar, so on an
# exchange holiday it fetched the vendor's unchanged data and named it as that day: 14 Sep and 2 Oct 2026 were
# published, with 90% and 87% of their closes identical to the day before -- against about 2% on real trading
# days (measured on all 22 daily sessions, 2026-10-12). Half identical is the line, with room on both sides. A
# vendor that fails to update on a real trading day is caught the same way: nothing is published, and Kite's
# 08:45 alarm says there is no list -- no trade on stale data. Kite also checks the exchange calendar itself.
STALE_SHARE = 0.5
MIN_COMMON = 50          # fewer shared stocks than this and two lists are too different to be copies


def stale_problem(frame: pd.DataFrame, feed: Path, session: str) -> str:
    """Why `frame` is a stale copy of the latest session published before `session`, or "" when it is not (or
    cannot be judged: nothing before it, or too few stocks in common). Closes are compared as the exact text the
    feed writes, so floating-point noise can neither hide a copy nor invent one."""
    before = [d for d in sessions_on_branch(feed) if d < session]
    if not before:
        return ""
    prev = before[-1]
    old = pd.read_csv(Path(feed) / SESSIONS / f"{prev}.csv", dtype=str, keep_default_na=False)
    new = pd.read_csv(io.BytesIO(to_csv_bytes(frame)), dtype=str, keep_default_na=False)
    old, new = old.set_index("symbol")["close_price"], new.set_index("symbol")["close_price"]
    common = new.index.intersection(old.index)
    if len(common) < MIN_COMMON:
        return ""
    same = float((new.loc[common] == old.loc[common]).mean())
    if same < STALE_SHARE:
        return ""
    return (f"stale: {same:.0%} of the {len(common)} stocks in both lists close exactly as on {prev} -- not a new "
            f"trading session (an exchange holiday, or the vendor did not update)")


def publish(workbook: bytes, title: str, session: str, feed: Path, *, origin: str, trigger: str,
            now: datetime) -> bool:
    """Score, gate, select, validate, write once. True when a new session was written."""
    datasets, stocks, bench, scored = score(workbook)
    checks = gate(datasets, stocks, bench, scored, backfill=origin == "backfilled")
    frame = build_frame(scored)
    validate(frame)
    problem = stale_problem(frame, feed, session)
    if problem:
        raise FeedError(problem)
    csv = to_csv_bytes(frame)
    return write_session(feed, session, csv, manifest(frame, csv, session, title, workbook, origin, trigger,
                                                      checks, now, blanks(scored)))


# ── The live run ──────────────────────────────────────────────────────────────────────────────────
# sync.py does not wait for proof of the ingest (its report printed empty on 9 Oct), so a dispatched run may
# start before Prism.gs renames the sheet: it re-reads the name every minute for up to 15. A scheduled or
# manual run never waits — it publishes what is there or exits.
WAIT_SECONDS, WAIT_STEP = 900, 60
TITLE_TRIES, TITLE_PAUSE = 3, 20            # a failed name read is read again (independent review M1, 2026-10-14)
TRIGGERS = ("dispatch", "schedule", "manual")


def download(sheet_id: str):
    """(sheet name, workbook bytes) from ONE response: the name in its header is the session of its body."""
    import requests
    r = requests.get(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx", timeout=180)
    r.raise_for_status()
    return _parse_title(r.headers.get("Content-Disposition", "")), r.content


def _session(title) -> str | None:
    d = parse_data_date(title)
    return d.isoformat() if d else None


def _new_session(title, have: list, now: datetime) -> str | None:
    """The session to publish, None when there is nothing new, or FeedError when it cannot be trusted."""
    session = _session(title)
    if session is None:
        raise FeedError(f"the sheet's name {title!r} carries no session date — unverifiable is not published")
    if session in have:
        print(f"{session} is already published — nothing to do")
        return None
    if have and session < have[-1]:
        print(f"the sheet holds {session}, older than the newest published {have[-1]} — nothing to do")
        return None
    # ONLY THE SESSION EXPECTED NOW IS PUBLISHED LIVE (2026-10-14, independent review I1). Prism.gs certifies a sheet
    # only by renaming it after a complete, sane ingest; a partial ingest, a failed rename or a BUSY run keeps the OLD
    # name over new data, and when that old session was never published (a refused holiday copy, say) it went out
    # under the wrong date. A missed session belongs to --backfill, never to the live path.
    if date.fromisoformat(session) < expected_session(now):
        raise FeedError(f"the sheet holds {session}, but the session expected now is {expected_session(now)}: an "
                        f"ingest that did not finish keeps the old name over new data, so it is never published live")
    if date.fromisoformat(session) > expected_session(now):
        raise FeedError(f"the sheet claims {session}, ahead of the newest possible session "
                        f"{expected_session(now)}")
    return session


# A NEW SESSION IS NEVER PUBLISHED IN MARKET HOURS (2026-10-13, independent review C1). Prism.gs names a sheet after
# the session its fetch clock implies -- before 19:00 IST (core.sheet_meta._DATA_READY) the previous weekday. That is
# true before the market opens and false from 09:15: a manual sync then (`--force` skips the settle gates) holds the
# day's moving prices under yesterday's name, and a published session is permanent. A list that late was past Kite's
# 08:45 decision anyway.
_MARKET_OPEN = (9, 15)


def _in_market_hours(now: datetime) -> bool:
    t = now.astimezone(IST)
    return t.weekday() < 5 and _MARKET_OPEN <= (t.hour, t.minute) < _DATA_READY


def _title(sheet_id: str, sleep) -> str | None:
    """The sheet's name, read up to TITLE_TRIES times: one network blip used to fail the whole run."""
    for attempt in range(TITLE_TRIES):
        title = fetch_sheet_title(sheet_id)
        if title or attempt == TITLE_TRIES - 1:
            return title
        sleep(TITLE_PAUSE)


def _not_renamed(title, have: list, now: datetime) -> bool:
    """A dispatch can land before Prism.gs renames the sheet: its name is still a published session, or older than
    the session expected now (the day after a refused copy the old name is unpublished, and the rename still comes)."""
    s = _session(title)
    return s is not None and (s in have or date.fromisoformat(s) < expected_session(now))


def run_live(sheet_id: str, feed: Path, *, trigger: str, now: datetime | None = None,
             sleep=time.sleep) -> str | None:
    """Publish the sheet's session if it is new. Returns the session written, or None."""
    now = now or datetime.now(IST)
    have = sessions_on_branch(feed)
    title = _title(sheet_id, sleep)
    waited = 0
    while trigger == "dispatch" and _not_renamed(title, have, now) and waited < WAIT_SECONDS:
        sleep(WAIT_STEP)
        waited += WAIT_STEP
        title = _title(sheet_id, sleep)
    if _new_session(title, have, now) is None:
        return None
    if _in_market_hours(now):
        raise FeedError(f"{_session(title)} is not published between 09:15 and 19:00 IST on a weekday: a sheet "
                        f"refreshed then holds the day's moving prices under a settled session's name")
    for _ in range(2):                                   # a rename between the two requests: read again
        got, workbook = download(sheet_id)
        if got == title:
            break
        title = _title(sheet_id, sleep)
    else:
        raise FeedError("the sheet was renamed during the download twice — the next run retries")
    session = _new_session(title, have, now)
    if session is None:
        return None
    return session if publish(workbook, title, session, feed, origin="recorded", trigger=trigger,
                              now=now) else None


# ── Back-fill (local, once) ───────────────────────────────────────────────────────────────────────
def backfill(folder: Path, feed: Path, *, now: datetime | None = None) -> list:
    """Every `PRISM YYYY-MM-DD*.xlsx` in `folder` not yet on the branch, oldest first, as `backfilled`.
    A copy that fails a check is reported and skipped — the others still go in."""
    now = now or datetime.now(IST)
    have = set(sessions_on_branch(feed))
    done, refused = [], []
    for path in sorted(Path(folder).glob("PRISM *.xlsx")):
        session = _session(path.stem)
        if session is None or session in have:
            continue
        try:
            if publish(path.read_bytes(), path.stem, session, feed, origin="backfilled", trigger="backfill",
                       now=now):
                done.append(session)
        except FeedError as exc:
            refused.append(f"{session}: {exc}")
    for line in refused:
        print(f"  ✗ not back-filled {line}")
    return done


# ── The run ───────────────────────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Publish PRISM's Playbook list to the playbook-feed branch.")
    ap.add_argument("--feed", required=True, type=Path, help="a checkout of the playbook-feed branch")
    ap.add_argument("--backfill", type=Path, help="a folder of archived 'PRISM YYYY-MM-DD.xlsx' copies")
    args = ap.parse_args(argv)
    if args.backfill:
        done = backfill(args.backfill, args.feed)
        print(f"back-filled {len(done)} session(s): {done}")
        return 0
    raw = os.environ.get("SHEET_ID", "")
    sheet_id = sheet_id_from_secret(raw)
    print(D._secret_shape(raw, sheet_id))
    if not sheet_id:
        raise SystemExit("SHEET_ID holds no spreadsheet id — paste the live data sheet's id (the part of its "
                         "address between /d/ and /edit).")
    trigger = os.environ.get("FEED_TRIGGER") or "manual"
    if trigger not in TRIGGERS:
        raise SystemExit(f"FEED_TRIGGER {trigger!r} is not one of {TRIGGERS}")
    try:
        session = run_live(sheet_id, args.feed, trigger=trigger)
    except FeedError as exc:
        print(f"❌ nothing published: {exc}")
        return 1
    out = os.environ.get("GITHUB_OUTPUT")
    if session and out:
        rows = json.loads((args.feed / SESSIONS / f"{session}.json").read_text(encoding="utf-8"))["rows"]
        with open(out, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"session={session}\nmessage=feed: {session} · {rows} stocks · recorded ({trigger})\n")
    print(f"✅ published {session}" if session else "nothing new to publish")
    return 0


if __name__ == "__main__":
    sys.exit(main())
