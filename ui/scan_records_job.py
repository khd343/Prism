"""
🎯 Scans — the records job (2026-10-06). Run weekly by GitHub Actions (.github/workflows/scan-records.yml):

    python -m ui.scan_records_job          # needs ARCHIVE_INDEX_SHEET_ID in the environment

It refreshes ui/scan_records.json from the PRISM Archive (the monthly copies the ingest script keeps,
listed in the 'PRISM Archive Index' sheet), re-scoring every copy with TODAY's engine and measuring
every scan with the SAME recipe function the tab runs (ui/ui_scans.py — one definition):

- UNSEEN windows: from each month-end copy dated after the scan's design date whose month is over
  (frozen), to the first copy at least the scan's horizon later (MIN_HORIZON_DAYS, or 365 days for
  🌱 Compounder Candidates) and to the latest copy. These are windows the
  recipes were never chosen on — the most trustworthy record a card shows.
- 🆕 baseline: the scan's members on the latest frozen month-end copy, and its date.

The DESIGN-PERIOD windows (in-sample, from the saved snapshots) are written only by the local tool
tools/scan_records.py; this job never touches them. If a recipe changed since they were measured the
job stops with an error (GitHub emails the failure), because a new unseen record beside a stale
design record would describe two different recipes.

The file is rewritten only when a measured value changed, so a quiet week makes no commit.
"""
import json
import os
import re
import subprocess
import sys
from datetime import date

import pandas as pd

from core import fetch_and_clean_data, run_scoring_pipeline
from core.forward_returns import MIN_HORIZON_DAYS, forward_returns
from ui import ui_scans as S

MIN_N = 30           # a window with fewer matches is not evidence and is not recorded
_VOLATILE = ("generated", "engine_version")


# ── Pure parts (tested) ───────────────────────────────────────────────────────────────────────────
def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def frozen(dates) -> list:
    """Copies whose month is over: every month before the latest copy's month (the latest month's
    copy is replaced daily until the month ends, so it cannot be a start)."""
    dates = sorted(dates)
    return [d for d in dates if d[:7] < dates[-1][:7]] if dates else []


def unseen_windows(dates, designed_on: str = S.DESIGNED_ON, min_days: int = MIN_HORIZON_DAYS) -> list:
    """(start, end) pairs the recipes were never chosen on: each frozen copy dated after designed_on,
    to the first copy at least min_days later and to the latest copy."""
    dates = sorted(dates)
    out = []
    for a in (d for d in frozen(dates) if d > designed_on):
        later = [b for b in dates if b > a and _days(a, b) >= min_days]
        for b in dict.fromkeys(later[:1] + later[-1:]):
            out.append((a, b))
    return out


def scan_windows(dates, scan: dict) -> list:
    """This scan's unseen windows: after ITS design date, at least ITS horizon long."""
    return unseen_windows(dates, S.scan_designed_on(scan), S.scan_horizon(scan))


def baseline_date(dates):
    """The latest frozen month-end copy — what 🆕 compares against; None before one exists."""
    f = frozen(dates)
    return f[-1] if f else None


def measure_window(early: pd.DataFrame, late: pd.DataFrame, scan: dict, start: str, end: str):
    """The scan's record over one window. The recipe runs on the WHOLE early frame (as the tab runs it
    on the whole market); returns exclude corporate actions; excess = a match's return minus the
    median of every stock measured. None when fewer than MIN_N matches have a return.

    Beside the headline (2026-10-07), what it leaves out: dropped_no_return = matches with no price at
    the end (delisted, suspended or missing: their fate is unknown, so they cannot be scored and the
    headline is the survivors' record) and dropped_corp_action = matches whose share count moved.
    And net = the same reading on matches trading at least S.NET_MIN_CR a day, each charged
    S.NET_COST_PP; its median and beat rate only when it has MIN_N matches, else just its n."""
    fr = forward_returns(early, late, "company_id")
    corp = early["company_id"].isin(set(fr.loc[fr["corp_action"], "company_id"]))
    m = fr[~fr["corp_action"]][["company_id", "fwd_return"]]
    ret = early[["company_id"]].merge(m, on="company_id", how="left")["fwd_return"].to_numpy() * 100
    ret = pd.Series(ret, index=early.index)
    in_scan = scan["mask"](early)
    match = in_scan & ret.notna()
    if int(match.sum()) < MIN_N:
        return None
    excess = ret - ret.median()
    liquid = match & (S.traded_value_cr(early) >= S.NET_MIN_CR)
    net = dict(min_cr=S.NET_MIN_CR, cost_pp=S.NET_COST_PP, n=int(liquid.sum()))
    if net["n"] >= MIN_N:
        kept = excess[liquid] - S.NET_COST_PP
        net.update(median_excess_pp=round(float(kept.median()), 1),
                   beat_pct=int(round(float((kept > 0).mean() * 100))))
    return dict(start=start, end=end, days=_days(start, end),
                median_excess_pp=round(float(excess[match].median()), 1),
                beat_pct=int(round(float((excess[match] > 0).mean() * 100))),
                n=int(match.sum()),
                dropped_no_return=int((in_scan & ret.isna() & ~corp).sum()),
                dropped_corp_action=int((in_scan & corp).sum()),
                net=net)


def members(frame: pd.DataFrame, scan: dict) -> list:
    return sorted(frame.loc[scan["mask"](frame), "company_id"].astype(str).unique())


def check_fresh(existing: dict) -> None:
    """Refuse to add unseen evidence beside a design-period record of a different recipe."""
    for scan in S.SCANS:
        rec = (existing.get("scans") or {}).get(scan["key"])
        if not rec or rec.get("fingerprint") != S.recipe_fingerprint(scan):
            raise SystemExit(f"{scan['key']}: the recipe changed since its design-period record was "
                             f"measured — run tools/scan_records.py locally first.")


def merge(existing: dict, updates: dict, top: dict):
    """existing + per-scan field updates + top-level fields → (new, changed). `changed` ignores the
    run's own timestamps, so re-running on unchanged data writes nothing."""
    new = json.loads(json.dumps(existing))
    new["version"] = S.RECORD_VERSION
    new["designed_on"] = S.DESIGNED_ON
    new.update(top)
    for key, fields in updates.items():
        new.setdefault("scans", {}).setdefault(key, {}).update(fields)
    strip = lambda d: {k: v for k, v in d.items() if k not in _VOLATILE}
    return new, strip(new) != strip(existing)


def write(records: dict, path=S.RECORDS_PATH) -> None:
    text = json.dumps(records, ensure_ascii=False, indent=1) + "\n"
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    with open(tmp, encoding="utf-8") as fh:
        if fh.read() != text:
            os.remove(tmp)
            raise SystemExit("read-back mismatch — nothing written")
    os.replace(tmp, path)


def sheet_id_from_secret(raw: str) -> str:
    """The spreadsheet id inside whatever was pasted into the secret: a bare id, the id in quotes, the
    whole Streamlit line (ARCHIVE_INDEX_SHEET_ID = "…") or the sheet's address. '' when none is there.
    The first live run (2026-10-06) failed HTTP 400 on exactly this: the value reached Google as pasted."""
    raw = raw or ""
    for pattern in (r"/spreadsheets/d/([A-Za-z0-9_-]{25,})",     # the sheet's address
                    r"[\"']([A-Za-z0-9_-]{25,})[\"']",            # a quoted id, alone or in a line
                    r"(?<![A-Za-z0-9_-])([A-Za-z0-9_-]{25,})(?![A-Za-z0-9_-])"):   # a bare id
        m = re.search(pattern, raw)
        if m:
            return m.group(1)
    return ""


def engine_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# ── The run ──────────────────────────────────────────────────────────────────────────────────────
def main() -> int:
    from ui.ui_movers import load_index
    raw = os.environ.get("ARCHIVE_INDEX_SHEET_ID", "")
    index_id = sheet_id_from_secret(raw)
    # Shape only, never the value: enough to tell a quoted id, a pasted line or an address apart.
    print(f"secret: {len(raw.strip())} chars{', quoted' if any(q in raw for q in chr(34) + chr(39)) else ''}"
          f"{', a full line' if '=' in raw else ''}{', an address' if '/' in raw else ''}"
          f" -> sheet id of {len(index_id)} chars")
    if not index_id:
        raise SystemExit("ARCHIVE_INDEX_SHEET_ID holds no spreadsheet id — paste the 'PRISM Archive Index' "
                         "sheet's id (the part of its address between /d/ and /edit).")
    existing = S.load_records()
    if not existing:
        raise SystemExit(f"{S.RECORDS_PATH.name} is missing or unreadable — run tools/scan_records.py first.")
    check_fresh(existing)
    idx = load_index(index_id)                                     # status 'ok' rows only
    ids = dict(zip(idx["vintage_date"], idx["spreadsheet_id"]))
    dates = sorted(ids)
    # Each scan on its own clock (2026-10-07): its own design date and shortest window, so a 🌱 list
    # judged over 12 months is never recorded on a 1-month window.
    windows = {s["key"]: scan_windows(dates, s) for s in S.SCANS}
    base = baseline_date(dates)
    need = sorted({d for ws in windows.values() for w in ws for d in w} | ({base} if base else set()))
    print(f"archive copies {dates}\nunseen windows {windows}\n🆕 baseline {base}")
    frames = {}
    for d in need:
        frames[d] = run_scoring_pipeline(fetch_and_clean_data("sheet", None, ids[d]))
        print(f"  scored {d}: {len(frames[d])} stocks")
    updates = {}
    for scan in S.SCANS:
        rows = [r for a, b in windows[scan["key"]] if (r := measure_window(frames[a], frames[b], scan, a, b))]
        updates[scan["key"]] = {"out_of_sample": rows}
        if base:
            updates[scan["key"]]["members_at_latest"] = members(frames[base], scan)
        print(f"{scan['emoji']} {scan['name']}: {len(rows)} unseen window(s)"
              + "".join(f"\n   {r['start']} -> {r['end']} {r['median_excess_pp']:+.1f}pp beat {r['beat_pct']}% n {r['n']}"
                        for r in rows))
    top = {"generated": date.today().isoformat(), "engine_version": engine_version()}
    if base:
        top["members_date"] = base
    new, changed = merge(existing, updates, top)
    if changed:
        write(new)
        print(f"wrote {S.RECORDS_PATH}")
    else:
        print("no measured value changed — nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
