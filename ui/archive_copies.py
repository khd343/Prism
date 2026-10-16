"""
📦 The PRISM Archive's copies for a date range (2026-10-16): the saved sheets window 1's chosen run is rebuilt from
(Kite's docs/readout-window-1.md; its frozen page, addendum 2026-10-15).

    $env:ARCHIVE_INDEX_SHEET_ID = '<the PRISM Archive Index sheet's id or address>'
    python -m ui.archive_copies --out <folder> --from 2026-10-12 --to 2026-11-20

The ingest script (stockscans_sync/Prism.gs) archives a copy of the sheet each trading day, once its six tabs are
written and verified, and lists it in the PRISM Archive Index: `ok` for the month's current copy, `daily` for one kept
for good after a newer copy took the month's slot (since 2026-10-13). Read-only: each copy is downloaded as
`PRISM <date>.xlsx` — the name ui/playbook_feed_job.py --backfill reads a session from — read back by the engine's own
loader with every tab holding exactly the stocks the index counted, written whole or not at all, and never over a file
that is there already. The index id is a secret: logged by its shape, never its value; no id or address is printed.

Measured 2026-10-16: the copies of 12-15 Oct, back-filled with the feed's pinned code (68b38e0), rebuilt each list Kite
received byte for byte, 4 of 4 — though each export's workbook bytes differ, the lists do not.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from core import data_engine as de
from ui.scan_records_job import sheet_id_from_secret

KEPT = ("ok", "daily")          # a copy kept for good: the month's current one, or an older day's (2026-10-13 on)
TABS = ("balance", "cashflow", "income", "ratio", "shareholding", "technical")   # data_engine's names for the six tabs
_EXPORT = "https://docs.google.com/spreadsheets/d/{}/export?format=xlsx"         # one workbook, every tab (CLAUDE.md §0)
_ID_LIKE = re.compile(r"[A-Za-z0-9_-]{25,}")


class ArchiveError(RuntimeError):
    """A copy that cannot be fetched as the index describes it — said in words, never with an id in them."""


def _scrub(text: str) -> str:
    return _ID_LIKE.sub(lambda m: f"<id of {len(m.group(0))} chars>", str(text))


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        body, kind = resp.read(), resp.headers.get("Content-Type", "")
    if "spreadsheetml" not in kind:
        raise ArchiveError(f"the download is not a workbook ({kind.split(';')[0] or 'no type'}): is it shared by link?")
    return body


def _engine_rows(workbook: bytes) -> dict:
    """{tab: rows} as the engine's own loader reads the workbook — data_engine.load_all_csvs, the feed's own path."""
    with contextlib.redirect_stdout(io.StringIO()):
        datasets = de.load_all_csvs("upload", {"workbook": io.BytesIO(workbook)})
    return {k: len(v) for k, v in datasets.items()}


def _iso(day: str) -> bool:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(day)):
        return False
    try:
        date.fromisoformat(day)
    except ValueError:
        return False
    return True


def kept(index: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """The index's kept copies dated start..end inclusive, oldest first: vintage_date, spreadsheet_id and
    rows_technicals (the stock count). Two kept rows for one day are refused: which is that day's would be a guess."""
    day = index["vintage_date"].astype(str).str.strip().str[:10]
    status = index["status"].astype(str).str.strip()
    mask = day.between(start, end) & status.isin(KEPT)
    rows = index[mask].assign(vintage_date=day[mask])     # day[mask]: a whole column on an empty frame would add rows
    twice = sorted(set(rows["vintage_date"][rows["vintage_date"].duplicated()]))
    if twice:
        raise ArchiveError(f"the index keeps two copies of {', '.join(twice)}")
    rows = rows.sort_values("vintage_date", kind="mergesort")
    return pd.DataFrame({"vintage_date": rows["vintage_date"].to_numpy(),
                         "spreadsheet_id": rows["spreadsheet_id"].astype(str).str.strip().to_numpy(),
                         "rows_technicals": pd.to_numeric(rows["rows_technicals"], errors="coerce").to_numpy()})


def _check(day: str, count, rows: dict) -> None:
    missing = sorted(set(TABS) - set(rows))
    if missing:
        raise ArchiveError(f"{day}: the copy lacks the tab(s) {missing}")
    if pd.isna(count):
        raise ArchiveError(f"{day}: the index gives no stock count, so its copy cannot be checked")
    off = {t: rows[t] for t in TABS if rows[t] != int(count)}
    if off:
        raise ArchiveError(f"{day}: the index counts {int(count)} stocks, the copy's tabs hold {off}")


def fetch(index_id: str, out: Path, start: str, end: str, get=None, rows=None) -> list:
    """Every kept copy dated start..end into `out` as `PRISM <date>.xlsx`; one line per day, oldest first. A file that
    is there already is checked and kept, never replaced; a copy that fails its check stops the fetch, unwritten."""
    get, rows = get or _get, rows or _engine_rows
    if not (_iso(start) and _iso(end) and start <= end):
        raise ArchiveError(f"the range {start} to {end} is not two dates in order (YYYY-MM-DD)")
    index = pd.read_excel(io.BytesIO(get(_EXPORT.format(index_id))), sheet_name=0, dtype=str)
    copies = kept(index, start, end)
    if copies.empty:
        raise ArchiveError(f"the index has no kept copy dated {start} to {end}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for day, sid, count in copies.itertuples(index=False):
        dest = out / f"PRISM {day}.xlsx"
        if dest.exists():
            _check(day, count, rows(dest.read_bytes()))
            lines.append(f"{day} kept: {dest.name} was there already and holds the {int(count)} stocks")
            continue
        body = get(_EXPORT.format(sid))
        _check(day, count, rows(body))
        part = dest.with_name(dest.name + ".part")
        part.write_bytes(body)
        os.replace(part, dest)
        lines.append(f"{day} fetched: {dest.name}, {int(count)} stocks in every tab")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download the PRISM Archive's kept copies for a date range, read-only.")
    ap.add_argument("--out", required=True, type=Path, help="the folder the copies go to (outside any repository)")
    ap.add_argument("--from", required=True, dest="start", help="the first day, YYYY-MM-DD")
    ap.add_argument("--to", required=True, dest="end", help="the last day, YYYY-MM-DD")
    a = ap.parse_args(argv)
    raw = os.environ.get("ARCHIVE_INDEX_SHEET_ID", "")
    index_id = sheet_id_from_secret(raw)
    # Shape only, never the value: enough to tell a quoted id, a pasted line or an address apart (CLAUDE.md §5).
    print(f"secret: {len(raw.strip())} chars{', quoted' if any(q in raw for q in chr(34) + chr(39)) else ''}"
          f"{', a full line' if '=' in raw else ''}{', an address' if '/' in raw else ''}"
          f" -> sheet id of {len(index_id)} chars")
    if not index_id:
        print("ARCHIVE_INDEX_SHEET_ID holds no spreadsheet id: set it to the PRISM Archive Index sheet's id (the part "
              "of its address between /d/ and /edit) or to its address")
        return 1
    try:
        lines = fetch(index_id, a.out, a.start, a.end)
    except (ArchiveError, OSError, ValueError) as e:      # said in words, never a traceback carrying an id
        print(f"refused: {_scrub(e)}")
        return 1
    print("\n".join(lines))
    print(f"{len(lines)} kept copies in {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
