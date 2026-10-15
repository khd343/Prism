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
- Against the index fund you could hold instead (2026-10-08, December plan rule 23): each unseen window
  also records the scan's liquid matches as one equal-weight portfolio after a round trip, and every
  declared benchmark index's price change after a fund fee, read from the index rows each archive copy
  carries (set aside from the stocks by core.load_market, from the same download).

The DESIGN-PERIOD windows (in-sample, from the saved snapshots) are written only by the local tool
tools/scan_records.py; this job never touches them. If a recipe changed since they were measured the
job stops with an error (GitHub emails the failure), because a new unseen record beside a stale
design record would describe two different recipes.

Since 2026-10-15 (the scan-records audit, before the first unseen window): every unseen window carries the
recipe fingerprint it was measured with, a final window is final for ITS recipe, and the cards count the
current recipe's windows only (ui_scans.record_state). A recipe changed after unseen windows were recorded
must move its design date past them (check_recipe_dates), and an archive whose newest copy is older than
MAX_ARCHIVE_AGE_DAYS fails the run (archive_problem) — before, it stayed green while the record stood still.

The file is rewritten only when a measured value changed, so a quiet week makes no commit.
"""
import json
import os
import re
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd

from config import BENCHMARK_INDICES
from core import load_market, run_scoring_pipeline
from core.forward_returns import MIN_HORIZON_DAYS, forward_returns
from ui import ui_scans as S

MIN_N = 30           # a window with fewer matches is not evidence and is not recorded
_VOLATILE = ("generated", "engine_version")
# The archive's current-month copy is re-taken every trading day, so its newest copy is a few days old at
# most (3 on a Monday morning, more over a long holiday). Older than this, the archive has stopped taking
# copies and the unseen record cannot grow — the job fails so GitHub emails the owner (2026-10-15 audit:
# before this, a stalled archive left the job green every Monday while the record silently stopped).
MAX_ARCHIVE_AGE_DAYS = 10


def _today() -> date:
    return date.today()


# ── Pure parts (tested) ───────────────────────────────────────────────────────────────────────────
def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def frozen(dates) -> list:
    """Copies whose month is over: every month before the latest copy's month (the latest month's
    copy is replaced daily until the month ends, so it cannot be a start)."""
    dates = sorted(dates)
    return [d for d in dates if d[:7] < dates[-1][:7]] if dates else []


def unseen_windows(dates, designed_on: str = S.DESIGNED_ON, min_days: int = MIN_HORIZON_DAYS,
                   start_month=None) -> list:
    """(start, end) pairs the recipes were never chosen on: each frozen copy dated after designed_on,
    to the first copy at least min_days later and to the latest copy.

    With a start_month (🌱, 2026-10-07): only frozen copies dated in that month start a window, and
    each runs to the first copy at least min_days later ONLY — one non-overlapping window a year, so a
    year of evidence is shown as one bar, not as a dozen windows that overlap by 11 months."""
    dates = sorted(dates)
    out = []
    for a in (d for d in frozen(dates) if d > designed_on):
        if start_month is not None and int(a[5:7]) != int(start_month):
            continue
        later = [b for b in dates if b > a and _days(a, b) >= min_days]
        ends = later[:1] if start_month is not None else later[:1] + later[-1:]
        for b in dict.fromkeys(ends):
            out.append((a, b))
    return out


def scan_windows(dates, scan: dict) -> list:
    """This scan's unseen windows: after ITS design date, at least ITS horizon long, and — for a scan
    with a start month — one a year."""
    return unseen_windows(dates, S.scan_designed_on(scan), S.scan_horizon(scan), S.scan_start_month(scan))


def baseline_date(dates):
    """The latest frozen month-end copy — what 🆕 compares against; None before one exists."""
    f = frozen(dates)
    return f[-1] if f else None


def archive_problem(dates, today: date) -> str:
    """Why the archive cannot extend the unseen record, in words for the failure email; '' when it can."""
    if not dates:
        return "The PRISM Archive Index lists no usable copy, so the unseen record cannot grow."
    newest = max(dates)
    age = (today - date.fromisoformat(newest)).days
    if age > MAX_ARCHIVE_AGE_DAYS:
        return (f"The PRISM Archive's newest copy is {newest}, {age} days old (the limit is "
                f"{MAX_ARCHIVE_AGE_DAYS}): the archive has stopped taking copies, so the unseen record cannot "
                f"grow. Check the script that saves the monthly copies.")
    return ""


def _closes(bench) -> pd.Series:
    """Each index's level in one copy's benchmark table, keyed by its id; empty without one."""
    if bench is None or bench.empty or not {"company_id", "close_price"} <= set(bench.columns):
        return pd.Series(dtype=float)
    b = bench.drop_duplicates("company_id")
    return pd.Series(pd.to_numeric(b["close_price"], errors="coerce").to_numpy(), index=b["company_id"].astype(str))


def index_returns(bench_early, bench_late, days: int) -> dict:
    """Each declared index's level change over a window, %, less the fund fee for the window's days
    (S.FUND_FEE_PCT_YR a year), in config.BENCHMARK_INDICES order. An index missing at either end, or
    without a positive level, has no reading. The PLAIN index, not a total-return one: the matches'
    returns are price changes too, so neither side counts dividends."""
    both = pd.DataFrame({"a": _closes(bench_early), "b": _closes(bench_late)}).dropna()
    both = both[(both["a"] > 0) & (both["b"] > 0)]
    r = (both["b"] / both["a"] - 1.0) * 100 - S.FUND_FEE_PCT_YR * days / 365
    return {k: round(float(r[k]), 2) for k in BENCHMARK_INDICES if k in r.index}


def hold_reading(early: pd.DataFrame, ret: pd.Series, match: pd.Series) -> dict:
    """"What if you held S.HOLD_K of them?" (2026-10-13, ui_scans.HOLD_* and ORDER_RULE). From the matches
    that are a Mega, Large, Mid or Small cap (S.HOLD_CAPS, whatever they trade a day) and whose one share
    costs at most S.HOLD_MAX_PRICE at the window's start — what a Rs 1 lakh account can hold — S.HOLD_DRAWS
    random picks of S.HOLD_K, equal-weighted, less S.NET_COST_PP: their
    median, 10th and 90th percentile and the share that lost money; and the share of those picks that
    momentum's top S.HOLD_K and Rank's top S.HOLD_K each beat. A FIXED seed over the matches in
    company_id order, so a window always reads the same. Only n when fewer than MIN_N can be held."""
    price = pd.to_numeric(early["close_price"], errors="coerce")
    cap = early["market_category"] if "market_category" in early.columns else pd.Series(np.nan, index=early.index)
    ok = match & cap.isin(S.HOLD_CAPS) & (price <= S.HOLD_MAX_PRICE)
    out = dict(k=S.HOLD_K, max_price=S.HOLD_MAX_PRICE, caps=list(S.HOLD_CAPS), cost_pp=S.NET_COST_PP,
               draws=S.HOLD_DRAWS, n=int(ok.sum()))
    if out["n"] < MIN_N:
        return out
    col = lambda c: (pd.to_numeric(early[c], errors="coerce") if c in early.columns
                     else pd.Series(np.nan, index=early.index))
    held = pd.DataFrame({"company_id": early["company_id"].astype(str), "ret": ret,
                         "mom": col("momentum_score"), "rank": col("rank")})[ok]
    held = held.sort_values("company_id").reset_index(drop=True)
    r = held["ret"].to_numpy()
    rng = np.random.default_rng(S.HOLD_SEED)
    picks = rng.random((S.HOLD_DRAWS, len(r))).argsort(axis=1)[:, :S.HOLD_K]
    sims = r[picks].mean(axis=1) - S.NET_COST_PP

    def placed(col: str, ascending: bool):
        top = held.dropna(subset=[col]).sort_values([col, "company_id"], ascending=[ascending, True])
        if len(top) < S.HOLD_K:
            return None
        return int(round(float((sims < top["ret"].head(S.HOLD_K).mean() - S.NET_COST_PP).mean() * 100)))

    out.update(median_pp=round(float(np.median(sims)), 1), p10_pp=round(float(np.percentile(sims, 10)), 1),
               p90_pp=round(float(np.percentile(sims, 90)), 1), loss_pct=int(round(float((sims < 0).mean() * 100))),
               momentum_pct=placed("mom", False), rank_pct=placed("rank", True))
    return out


def measure_window(early: pd.DataFrame, late: pd.DataFrame, scan: dict, start: str, end: str,
                   bench_early=None, bench_late=None):
    """The scan's record over one window. The recipe runs on the WHOLE early frame (as the tab runs it
    on the whole market); returns exclude corporate actions; excess = a match's return minus the
    median of every stock measured. None when fewer than MIN_N matches have a return.

    Beside the headline (2026-10-07), what it leaves out: dropped_no_return = matches with no price at
    the end (delisted, suspended or missing: their fate is unknown, so they cannot be scored and the
    headline is the survivors' record) and dropped_corp_action = matches whose share count moved.
    And net = the same reading on matches trading at least S.NET_MIN_CR a day, each charged
    S.NET_COST_PP; its median and beat rate only when it has MIN_N matches, else just its n.

    vs_index (2026-10-08, December plan rule 23), when the two copies' benchmark tables give at least
    one index reading: those same liquid matches as ONE equal-weight portfolio — the mean return, which
    is what a portfolio earns, less S.NET_COST_PP (portfolio_pp, only with MIN_N of them) — and every
    index's return less the fund fee for the window (index_returns)."""
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
    out = dict(start=start, end=end, days=_days(start, end),
               median_excess_pp=round(float(excess[match].median()), 1),
               beat_pct=int(round(float((excess[match] > 0).mean() * 100))),
               n=int(match.sum()),
               dropped_no_return=int((in_scan & ret.isna() & ~corp).sum()),
               dropped_corp_action=int((in_scan & corp).sum()),
               net=net)
    out["hold"] = hold_reading(early, ret, match)
    indices = index_returns(bench_early, bench_late, _days(start, end))
    if indices:
        vs = dict(fee_pct_yr=S.FUND_FEE_PCT_YR, cost_pp=S.NET_COST_PP, min_cr=S.NET_MIN_CR, n=net["n"],
                  indices=indices)
        if vs["n"] >= MIN_N:
            vs["portfolio_pp"] = round(float(ret[liquid].mean()) - S.NET_COST_PP, 2)
        out["vs_index"] = vs
    return out


def _window_key(w: dict) -> tuple:
    """A window is its dates AND the recipe that measured it (2026-10-15): final is final per recipe."""
    return w["start"], w["end"], w.get("fingerprint") or ""


def keep_final(old: list, fresh: list, dates, engine: str = "") -> list:
    """The unseen record, final windows first-written (2026-10-12). A window whose END copy is a finished month is
    FINAL: recorded once, stamped with the engine that measured it, never re-measured and never dropped. Every
    copy is re-scored with TODAY's engine, and the recipe fingerprint guards the recipe, not the engine under it —
    so without this a later scoring change would silently rewrite evidence already recorded. A window still ending
    on the latest, unfinished copy is RUNNING: measured again each week, unstamped (like the top-level
    engine_version, a stamp alone must not count as a change). Final for its OWN recipe (2026-10-15): another
    recipe's final window on the same dates is kept beside it, never mistaken for it."""
    done = set(frozen(dates))
    out = {_window_key(w): w for w in old if w.get("final")}
    for w in fresh:
        key = _window_key(w)
        if key not in out:
            out[key] = {**w, "final": True, "engine": engine} if w["end"] in done else {**w, "final": False}
    return [out[k] for k in sorted(out)]


def members(frame: pd.DataFrame, scan: dict) -> list:
    return sorted(frame.loc[scan["mask"](frame), "company_id"].astype(str).unique())


def check_fresh(existing: dict) -> None:
    """Refuse to add unseen evidence beside a design-period record of a different recipe."""
    for scan in S.SCANS:
        rec = (existing.get("scans") or {}).get(scan["key"])
        if not rec or rec.get("fingerprint") != S.recipe_fingerprint(scan):
            raise SystemExit(f"{scan['key']}: the recipe changed since its design-period record was "
                             f"measured — run tools/scan_records.py locally first.")


def check_recipe_dates(existing: dict, fingerprints: dict) -> None:
    """A recipe chosen after unseen windows of an earlier recipe were recorded has SEEN them (2026-10-15): its
    own unseen record may only start after the last of them. Until its design date is moved there, the job
    stops — otherwise data the new recipe was fitted on would be recorded as unseen."""
    for scan in S.SCANS:
        rows = ((existing.get("scans") or {}).get(scan["key"]) or {}).get("out_of_sample") or []
        seen = [w["end"] for w in rows if w.get("fingerprint") != fingerprints[scan["key"]]]
        if seen and S.scan_designed_on(scan) < max(seen):
            raise SystemExit(f"{scan['key']}: the recipe changed after unseen windows to {max(seen)} were recorded "
                             f"— move its design date to {max(seen)} or later, so its own unseen record starts "
                             f"on data it never saw.")


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
    # The recipe each window is measured with (2026-10-15): stamped on every window, final per recipe.
    fps = {s["key"]: S.recipe_fingerprint(s) for s in S.SCANS}
    check_recipe_dates(existing, fps)
    idx = load_index(index_id)                                     # status 'ok' rows only
    ids = dict(zip(idx["vintage_date"], idx["spreadsheet_id"]))
    dates = sorted(ids)
    problem = archive_problem(dates, _today())
    if problem:
        raise SystemExit(problem)
    # Each scan on its own clock (2026-10-07): its own design date and shortest window, so a 🌱 list
    # judged over 12 months is never recorded on a 1-month window.
    old = {s["key"]: ((existing.get("scans") or {}).get(s["key"]) or {}).get("out_of_sample") or [] for s in S.SCANS}
    final = {k: {_window_key(w) for w in rows if w.get("final")} for k, rows in old.items()}
    # A FINAL window on file is never measured again for its recipe (keep_final) — so its copies are not even
    # downloaded for it.
    windows = {s["key"]: [w for w in scan_windows(dates, s) if (*w, fps[s["key"]]) not in final[s["key"]]]
               for s in S.SCANS}
    base = baseline_date(dates)
    ev = engine_version()
    need = sorted({d for ws in windows.values() for w in ws for d in w} | ({base} if base else set()))
    print(f"archive copies {dates}\nunseen windows {windows}\n🆕 baseline {base}")
    frames, benches = {}, {}
    for d in need:
        # ONE download per copy: the stocks (benchmark rows set aside) and the index rows beside them.
        stocks, benches[d] = load_market("sheet", None, ids[d])
        frames[d] = run_scoring_pipeline(stocks)
        print(f"  scored {d}: {len(frames[d])} stocks, {len(benches[d])} benchmark indices")
    updates = {}
    for scan in S.SCANS:
        rows = [{**r, "fingerprint": fps[scan["key"]]} for a, b in windows[scan["key"]]
                if (r := measure_window(frames[a], frames[b], scan, a, b, benches[a], benches[b]))]
        rows = keep_final(old[scan["key"]], rows, dates, ev)
        updates[scan["key"]] = {"out_of_sample": rows}
        if base:
            updates[scan["key"]]["members_at_latest"] = members(frames[base], scan)
        print(f"{scan['emoji']} {scan['name']}: {len(rows)} unseen window(s)")
        for r in rows:
            vs, code = r.get("vs_index") or {}, scan.get("benchmark")
            idx = (vs.get("indices") or {}).get(code)
            tail = (f" · vs {BENCHMARK_INDICES[code]} {vs['portfolio_pp'] - idx:+.1f}pp after costs"
                    if idx is not None and "portfolio_pp" in vs else "")
            print(f"   {r['start']} -> {r['end']} {r['median_excess_pp']:+.1f}pp beat {r['beat_pct']}% n {r['n']}{tail}")
    top = {"generated": date.today().isoformat(), "engine_version": ev}
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
