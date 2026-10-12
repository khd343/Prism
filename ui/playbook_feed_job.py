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

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from ui import ui_scans as S
from ui.scan_records_job import engine_version

SCHEMA = 1
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


def _counts(s: pd.Series, codes: dict) -> dict:
    return {k: int((s == k).sum()) for k in sorted(set(codes.values()) | {""})}


def manifest(frame: pd.DataFrame, csv: bytes, session: str, source_title: str, source: bytes, origin: str,
             trigger: str, checks: dict, now: datetime) -> dict:
    return {
        "schema": SCHEMA, "session": session, "rows": int(len(frame)), "columns": list(COLUMNS),
        "csv_sha256": sha256(csv), "recipe": "playbook",
        "recipe_fingerprint": S.recipe_fingerprint(playbook_scan()),
        "engine_commit": os.environ.get("GITHUB_SHA") or engine_version(),
        "source_title": source_title, "source_sha256": sha256(source),
        "origin": origin, "trigger": trigger, "published_at": now.isoformat(timespec="seconds"),
        "counts": {"cap": _counts(frame["cap"], CAP_CODES), "stage": _counts(frame["stage"], STAGE_CODES)},
        "checks": checks,
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
            "`stage`: Weinstein 1 2 3 4 or empty. Rows sorted by `symbol`; UTF-8, `\\n`; empty = not known.\n")


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
