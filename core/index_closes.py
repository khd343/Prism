"""
📈 The market box's numbers from NSE's own daily index file (2026-11-04) — display only.

WHY. The vendor's history for Nifty500 Shariah was measured wrong on 3 Nov 2026 against NSE's own files (its 200-day
average 7,835.0 against NSE's 6,824.3, its 30-week 7,481.1 against 6,842.7), so the box read Stage 4 where NSE's
numbers give Stage 3; the other nine indices matched NSE within 1%. And the operator asked (4 Nov) for the 200-week
line Kite's shadow book now gates on: NIFTY MIDSML 400 against the simple average of its last 200 weekly closes
(addendum 37, rule 8). So every number in the box is NSE's, and each row shows that line.

WHAT. ui/index_closes_job.py keeps each declared index's daily NSE close on the branch config.INDEX_CLOSES_BRANCH
(one CSV, a close written once per day and index, never rewritten). The app reads it (read_history), and box_view
turns it into one row per index: the stage by the one rule every stock uses (data_engine.weinstein_stage: the close
against the 30-week average, that average against the 200-day), and vs_200w, the close against the 200-week average.
Kite's definition, to the decimal: a week's close is its last session's (Monday to Sunday), and the current week
counts with its latest close. An average needs its whole count of weeks or sessions, or it is NaN (the box shows —):
an index younger than 200 weeks is never given a 200-week line it has not earned.

Nothing here reaches a score, a scan, a record or Kite.
"""
from __future__ import annotations

import csv
import io
import math
from datetime import date, datetime

import numpy as np
import pandas as pd

from config import BENCHMARK_INDICES, INDEX_WEEKS_LONG
from core.data_engine import weinstein_stage

HEADER = ("date", "company_id", "close")
SMA_WEEKS, SMA_DAYS = 30, 200          # the stage rule's averages, as the vendor's SMA 30W and SMA 200D
# A stray fragment never joins an average: measured on the seed (4 Nov 2026), the three young indices each appear ONCE
# in NSE's file of 12 Apr 2023 and then not for 172-284 sessions until they launch, while the seven older ones never
# miss a session. A stretch of fewer than MIN_RUN_SESSIONS sessions followed by a gap of more than MAX_GAP_SESSIONS is
# such a fragment. A longer stretch before a gap is the same index, bridged (review, 4 Nov 2026: a six-session outage
# would otherwise have taken the Nifty 500's 200-week line away for four years); the newest stretch always counts.
MAX_GAP_SESSIONS = 5
MIN_RUN_SESSIONS = 20
# How NSE may write a day. Its files write it day-first, except where they don't: measured on 205 files (4 Nov 2026),
# one wrote 6 Apr 2023 month-first (04-06-2023). A cell names the day when any reading of it does; a cell no reading
# parses stops the run (review, 4 Nov 2026: a new format would otherwise have read as a stale copy every night, quietly).
_DATE_FORMATS = ("%d-%m-%Y", "%m-%d-%Y", "%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y")
VIEW_COLUMNS = ["company_id", "name", "last_date", "close", "sma_30w", "sma_200d", "sma_200w", "weeks", "stage",
                "vs_200w"]


class NseFileError(ValueError):
    """NSE's file is not the index table it should be: someone should look before anything is kept."""


class HistoryError(ValueError):
    """The kept history file is damaged."""


def _norm(name) -> str:
    return " ".join(str(name).lower().split())


_DECLARED = {_norm(name): cid for cid, name in BENCHMARK_INDICES.items()}


def _priced(text: str) -> bool:
    try:
        x = float(text)
    except ValueError:
        return False
    return math.isfinite(x) and x > 0


def _readings(cell: str) -> set:
    """Every day `cell` can be read as."""
    out = set()
    for fmt in _DATE_FORMATS:
        try:
            out.add(datetime.strptime(cell, fmt).date())
        except ValueError:
            pass
    return out


def parse_nse_file(text: str, day: date) -> dict | None:
    """{vendor id: close as NSE wrote it, thousands commas dropped} for the declared indices NSE's file for `day`
    prices on that day; None when none of its declared rows carries the day (a stale copy served under the day's name,
    never kept). A declared row carrying another date is not that day's close and is left out. Raises NseFileError
    when it is not NSE's index table, names none of the declared indices, carries a date no reading parses, or prices
    none of them on the day."""
    try:
        rows = list(csv.reader(io.StringIO(text.lstrip("﻿"))))
    except csv.Error as exc:
        raise NseFileError(f"not NSE's index table ({exc})") from None
    head = [_norm(h) for h in rows[0]] if rows else []
    try:
        i_name, i_date, i_close = head.index("index name"), head.index("index date"), head.index("closing index value")
    except ValueError:
        raise NseFileError(f"not NSE's index table (header {(rows[0] if rows else [])[:6]})") from None
    body = [r for r in rows[1:] if len(r) > max(i_name, i_date, i_close) and r[i_name].strip()]
    if not body:
        raise NseFileError("NSE's index table has no rows")
    declared = [r for r in body if _norm(r[i_name]) in _DECLARED]
    if not declared:
        raise NseFileError(f"none of the declared indices in NSE's file for {day}")
    readings = {r[i_date].strip(): _readings(r[i_date].strip()) for r in declared}
    unread = sorted(cell for cell, got in readings.items() if not got)
    if unread:
        raise NseFileError(f"cannot read the date {unread[0]!r} in NSE's file for {day}")
    on_day = [r for r in declared if day in readings[r[i_date].strip()]]
    if not on_day:
        return None
    out: dict = {}
    for r in on_day:
        cid, close = _DECLARED[_norm(r[i_name])], r[i_close].strip().replace(",", "")
        if not _priced(close):
            continue
        if out.get(cid, close) != close:
            raise NseFileError(f"two closes for {BENCHMARK_INDICES[cid]} on {day}: {out[cid]} and {close}")
        out[cid] = close
    if not out:
        raise NseFileError(f"NSE's file for {day} prices none of the declared indices")
    return out


def merge(rows, new_rows) -> list:
    """The kept rows plus the new ones, each (ISO date, vendor id, close text), sorted. A (date, id) already kept is
    never changed: a close, once kept, is the record."""
    out = {(d, c): v for d, c, v in rows}
    for d, c, v in new_rows:
        out.setdefault((d, c), v)
    return sorted((d, c, v) for (d, c), v in out.items())


def to_csv(rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(HEADER)
    w.writerows(sorted(rows))
    return buf.getvalue()


def from_csv(text: str) -> list:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or tuple(rows[0]) != HEADER:
        raise HistoryError(f"the history's header is {rows[0] if rows else 'missing'}, not {list(HEADER)}")
    out = []
    for n, r in enumerate(rows[1:], start=2):
        if len(r) != 3:
            raise HistoryError(f"line {n} has {len(r)} fields")
        try:
            date.fromisoformat(r[0])
        except ValueError:
            raise HistoryError(f"line {n}: {r[0]!r} is not a date") from None
        if not _priced(r[2]):
            raise HistoryError(f"line {n}: {r[2]!r} is not a close")
        out.append((r[0], r[1], r[2]))
    return sorted(out)


def read_history(url: str, timeout: float = 8.0) -> pd.DataFrame:
    """The kept closes (date, company_id, close) from the branch, or an empty frame when it cannot be read or is
    damaged: the box then shows the sheet's stages and says so. A display never breaks the sidebar."""
    try:
        import requests
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return pd.DataFrame(from_csv(r.text), columns=list(HEADER))
    except Exception as exc:                     # noqa: BLE001 -- unreadable is shown, never raised
        print(f"  ⚠️ NSE index closes not read ({type(exc).__name__})")
        return pd.DataFrame(columns=list(HEADER))


def _last(series: pd.Series) -> pd.Series:
    """Each index's LAST value (its newest), NaN kept: GroupBy.last() would skip a NaN and hand back an older one."""
    return series.groupby(level=0).tail(1).droplevel(1)


def box_view(closes: pd.DataFrame | None, as_of: date | None = None) -> pd.DataFrame:
    """One row per declared index with closes, in config order: its newest session on or before `as_of` (the sheet's
    own date when known), the close, the 30-week, 200-day and 200-week averages (each only on its whole count), the
    number of weeks, the stage (the one rule) and vs_200w (% above, negative below)."""
    if closes is None or closes.empty:
        return pd.DataFrame(columns=VIEW_COLUMNS)
    df = closes[closes["company_id"].isin(BENCHMARK_INDICES)][["date", "company_id", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[df["date"].notna() & (df["close"] > 0)]
    if as_of is not None:
        df = df[df["date"] <= pd.Timestamp(as_of)]
    if df.empty:
        return pd.DataFrame(columns=VIEW_COLUMNS)
    df = df.sort_values(["company_id", "date"]).reset_index(drop=True)
    sessions = pd.Index(np.sort(df["date"].unique()))                  # every session any index has
    pos = pd.Series(sessions.get_indexer(df["date"]), index=df.index)  # each row's place among them
    missed = pos.groupby(df["company_id"]).diff().fillna(1) - 1         # sessions this index missed before it
    run = (missed > MAX_GAP_SESSIONS).groupby(df["company_id"]).cumsum()          # each stretch between long gaps
    size = run.groupby([df["company_id"], run]).transform("size")
    newest_run = run == run.groupby(df["company_id"]).transform("max")
    df = df[newest_run | (size >= MIN_RUN_SESSIONS)].reset_index(drop=True)
    by_id = df.groupby("company_id", sort=True)
    newest = by_id.tail(1).set_index("company_id")
    sma_200d = _last(by_id["close"].rolling(SMA_DAYS, min_periods=SMA_DAYS).mean())
    iso = df["date"].dt.isocalendar()
    df["week"] = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    weekly = df.groupby(["company_id", "week"], sort=True)["close"].last()          # each week's last session
    w_by = weekly.groupby(level=0)
    sma_30w = _last(w_by.rolling(SMA_WEEKS, min_periods=SMA_WEEKS).mean().droplevel(0))
    sma_200w = _last(w_by.rolling(INDEX_WEEKS_LONG, min_periods=INDEX_WEEKS_LONG).mean().droplevel(0))
    out = pd.DataFrame({"close": newest["close"], "last_date": newest["date"].dt.date,
                        "sma_30w": sma_30w, "sma_200d": sma_200d, "sma_200w": sma_200w,
                        "weeks": w_by.size()})
    out["stage"] = weinstein_stage(out["close"], out["sma_30w"], out["sma_200d"])
    with np.errstate(divide="ignore", invalid="ignore"):
        out["vs_200w"] = np.where(out["sma_200w"] > 0.0, (out["close"] / out["sma_200w"] - 1.0) * 100.0, np.nan)
    order = [cid for cid in BENCHMARK_INDICES if cid in out.index]
    out = out.loc[order].rename_axis("company_id").reset_index()
    out["name"] = out["company_id"].map(BENCHMARK_INDICES)
    return out[VIEW_COLUMNS]
