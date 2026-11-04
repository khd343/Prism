"""
📈 NSE's index closes for the sidebar's market box (2026-11-04) — the nightly job. Display only.

    python -m ui.index_closes_job --out closes [--backfill-from YYYY-MM-DD]

<out> is a checkout of the branch config.INDEX_CLOSES_BRANCH (never main): one file, index_closes.csv, each declared
index's close per session as NSE wrote it (core/index_closes.py reads it). The job asks NSE's own daily index file
(config.NSE_INDEX_FILE_URL) for every day it does not hold, from LOOKBACK_DAYS before its newest day (a file NSE
published late, or a weekend Muhurat session, is picked up the next run) — or from --backfill-from, or from
config.INDEX_CLOSES_START on an empty branch.

  * 404 = no file for that day (a holiday, a weekend, or not published yet): nothing is added and the run stays
    green — a holiday never turns it red.
  * A file carrying another date is a stale copy: skipped, with a warning.
  * A file that is not NSE's index table stops the run (red): NSE changed its file, and someone should look.
  * A close once kept is never rewritten.

Nothing here reaches a score, a scan, a record or Kite.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from config import INDEX_CLOSES_BRANCH, INDEX_CLOSES_START, NSE_INDEX_FILE_URL
from core.index_closes import from_csv, merge, parse_nse_file, to_csv
from core.sheet_meta import IST

LOOKBACK_DAYS = 7
PAUSE_S = 0.3                     # between NSE requests: a polite reader of an archive
ATTEMPTS = 3
FILE = "index_closes.csv"
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"}
_README = f"""# NSE index closes for PRISM's market box

Written by `ui/index_closes_job.py` (the `Index closes` workflow) on this branch only, never on main. Display only:
the sidebar's market box reads `{FILE}` for each index's stage and its line against the 200-week average. No score,
scan, record or Kite reads it.

`{FILE}`: `date,company_id,close`, one row per session and declared index, the close exactly as NSE wrote it in its
daily file (`ind_close_all_DDMMYYYY.csv`). A close, once kept, is never rewritten. Days with no NSE file (holidays,
weekends) have no rows.
"""


def ist_today() -> date:
    return datetime.now(IST).date()


def days_to_ask(kept: set, today: date, start: date, backfill_from: date | None = None) -> list:
    """Every day not kept, from LOOKBACK_DAYS before the newest kept day (never before `start`; `start` itself when
    nothing is kept; `backfill_from` when given) to `today`."""
    first = backfill_from or (max(start, max(kept) - timedelta(days=LOOKBACK_DAYS)) if kept else start)
    n = (today - first).days
    return [first + timedelta(days=i) for i in range(n + 1) if first + timedelta(days=i) not in kept]


def _session():
    import requests
    s = requests.Session()
    s.headers.update(_UA)
    return s


def fetch_day(day: date, session) -> str | None:
    """NSE's file for `day`, or None when NSE has none (404). A passing outage is retried; anything else, after the
    last attempt, stops the run with what NSE answered."""
    url = NSE_INDEX_FILE_URL.format(day)
    last = None
    for attempt in range(ATTEMPTS):
        time.sleep(PAUSE_S if attempt == 0 else 5.0 * attempt)
        try:
            r = session.get(url, timeout=20)
        except Exception as exc:                     # noqa: BLE001 -- a network error is retried, then named
            last = type(exc).__name__
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 200:
            return r.text
        last = f"HTTP {r.status_code}"
    raise RuntimeError(f"NSE's index file for {day} could not be read: {last}")


def run(out: Path, today: date, fetch, start: date | None = None, backfill_from: date | None = None) -> tuple:
    """Ask for the missing days, keep what NSE published, write the branch's files. -> (changed, message)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / FILE
    kept_rows = from_csv(path.read_text(encoding="utf-8")) if path.exists() else []
    kept = {date.fromisoformat(d) for d, _, _ in kept_rows}
    start = start or date.fromisoformat(INDEX_CLOSES_START)
    new, sessions = [], []
    for day in days_to_ask(kept, today, start, backfill_from):
        text = fetch(day)
        if text is None:
            continue
        got = parse_nse_file(text, day)
        if got is None:
            print(f"  ⚠️ NSE's file for {day} carries another date: a stale copy, skipped")
            continue
        sessions.append(day)
        new += [(day.isoformat(), cid, close) for cid, close in got.items()]
    rows = merge(kept_rows, new)
    if not rows:
        return False, "index closes: nothing new (nothing kept yet)"
    text = to_csv(rows)
    changed = not path.exists() or path.read_text(encoding="utf-8") != text
    for name, body in ((".gitattributes", "* -text\n"), ("README.md", _README)):
        f = out / name
        if not f.exists() or f.read_text(encoding="utf-8") != body:
            f.write_text(body, encoding="utf-8", newline="\n")
            changed = True
    if not changed:
        return False, f"index closes: nothing new (kept to {max(kept) if kept else 'nothing yet'})"
    path.write_text(text, encoding="utf-8", newline="\n")
    newest = max(date.fromisoformat(d) for d, _, _ in rows) if rows else None
    added = len({d for d, _, _ in rows} - {d.isoformat() for d in kept})
    return True, (f"index closes: {newest} · {len({c for d, c, _ in rows if d == str(newest)})} indices · "
                  f"{added} new day{'s' if added != 1 else ''}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NSE's index closes for the market box")
    ap.add_argument("--out", required=True, help=f"a checkout of the {INDEX_CLOSES_BRANCH} branch")
    ap.add_argument("--backfill-from", default="", help="YYYY-MM-DD: also ask for every day from here")
    a = ap.parse_args(argv)
    backfill = date.fromisoformat(a.backfill_from) if a.backfill_from.strip() else None
    session = _session()
    changed, message = run(Path(a.out), ist_today(), lambda d: fetch_day(d, session), backfill_from=backfill)
    print(message)
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a", encoding="utf-8") as fh:
            fh.write(f"changed={'true' if changed else 'false'}\nmessage={message}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
