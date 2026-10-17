"""
📋 Kite v2's lists (2026-10-17): PRISM's two lists for Kite v2, both from ONE commit (Kite-v2 design §3.1, addendum A5).

    python -m ui.v2_lists_job --lists <a v2-lists checkout>                        # live, needs SHEET_ID
    python -m ui.v2_lists_job --lists <dir> --backfill <folder of "PRISM YYYY-MM-DD.xlsx">

For every session, two files go to the orphan branch `v2-lists`, written once:
- the 🚀 Playbook list, exactly the bytes the feed's own code (ui/playbook_feed_job.py) builds at this commit — not
  the `playbook-feed` branch's, whose workflow is pinned to an older commit;
- the rank list: every scored stock with the columns v2 reads, among them its momentum percentile over the WHOLE
  market, and the dates of its last and next results.

Everything that decides a list is the feed's own and is imported, never copied: the scoring, the 🩺 gate, the
recipe, the Playbook list's builder, the stale-copy guard and the session rules. This module adds the rank list, its
checks, the two-file store and the carry-forward of `last_result_date`. The feed itself (branch `playbook-feed`,
pinned 68b38e0) is untouched: Kite v1 reads it until it retires, and v2 never mixes the two recipes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import SHARIAH_STATUSES
from core.sheet_meta import IST, _prev_weekday, expected_session
from ui import data_check_job as D
from ui import playbook_feed_job as F
from ui import ui_scans as S
from ui.archive_copies import _scrub
from ui.scan_records_job import engine_version, sheet_id_from_secret

# ── The rank list: one row per scored stock (design §3.1) ────────────────────────────────────────────────────────────
RANK_COLUMNS = ("symbol", "shariah_status", "cap", "traded_value_cr", "momentum_score", "momentum_pct", "vstop_value",
                "close_price", "sector", "stage", "last_result_date", "next_result_date", "equity_shares")
RANK_NUMBERS = ("traded_value_cr", "momentum_score", "vstop_value", "close_price")
RANK_SOURCE = ("company_id", "shariah_status", "market_category", "traded_value_cr", "momentum_score", "vstop_value",
               "close_price", "sector", "weinstein_stage", "days_from_result", "equity_shares")
# Measured 2026-10-17 on all 29 saved sheets (29 Jun - 16 Oct 2026): 2,107 to 2,716 scored stocks, no company id ever
# repeated or outside Kite's EXCHANGE:SYMBOL form. Wide enough that a normal day never trips it; an empty, truncated
# or exploded market always does.
_MEASURED_RANK_BOUNDS = (1800, 3500)
MIN_RANK_ROWS, MAX_RANK_ROWS = _MEASURED_RANK_BOUNDS
# THE RANK LIST'S OWN INPUTS, floored by the feed's rule (RECIPE_FLOORS): the lowest share on the 29 sheets less 8
# points, rounded down to 0.05. Lowest shares: shariah_status 100% (every sheet), market_category 99.96% (14 Sep),
# traded_value_cr 98.86% (10 Sep), momentum_score 100%, vstop_value 99.62% (29 Jun), sector 100%, days_from_result
# 99.89% (10 Sep), equity_shares 99.96% (14 Sep), a positive close 100%, a known Weinstein stage 93.04% (10 Sep).
RANK_FLOORS = {"shariah_status": 0.90, "market_category": 0.90, "traded_value_cr": 0.90, "momentum_score": 0.90,
               "vstop_value": 0.90, "sector": 0.90, "days_from_result": 0.90, "equity_shares": 0.90,
               "close_price": 0.90, "weinstein_stage": 0.85}
TEXT_FLOORS = ("shariah_status", "market_category", "sector")
UNKNOWN_STAGE = "❔ Unknown"
# A SCREEN THAT LABELS EVERYTHING DOUBTFUL STILL FILLS EVERY ROW, so the fill floor cannot see it. The share labelled
# Shariah ran 20.98% (29 Jun) to 30.91% (3 Sep) on the 29 sheets; less and plus 8 points, rounded outward to 0.05.
SHARIAH_SHARE_MIN, SHARIAH_SHARE_MAX = 0.10, 0.40
# A FIGURE THAT CANNOT BE REAL IS A MISSING ONE IN DISGUISE (review M2, 2026-10-17): a close or stop of 0 or less, or no
# shares, is written empty, never a number an exit could act on (anti-goal: a sale caused by a blank or missing figure).
# Never seen on the 29 sheets; the floors count only what is written.
POSITIVE = ("close_price", "vstop_value", "equity_shares")
# days_from_result ran about -660 (the oldest last result, 27 Dec 2024) to +30 on the 24 sessions 11 Sep - 16 Oct 2026.
# Past 1,000 days it is unknown: an Excel serial date in the wrong column (~46,000) would be a date in 2152, and past
# ~86,000 days pandas overflows, which would stop the day's publish (review M3).
MAX_RESULT_DAYS = 1000


def build_rank(scored: pd.DataFrame, session: str, previous: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """The rank list of `session`, and {"carried": n, "unknown_days": n}.

    `previous` is the newest earlier rank list on the branch, read as text (None when there is none). The momentum
    percentile is the 🚀 recipe's own (`S._pct` over the whole frame), so the 2/3 line reads exactly as the recipe
    reads it. The result dates come from days_from_result (negative = days since the last result):
    - next_result_date = session + days when days >= 0 (an announced date, or today);
    - last_result_date = session + days when days <= 0; otherwise (an announced date, or a blank) that symbol's value
      on `previous`, or '' when there is none.
    A non-integral or blank day is unknown. Every label is a code, and an unknown one refuses the list."""
    missing = sorted(set(RANK_SOURCE) - set(scored.columns))
    if missing:
        raise F.FeedError(f"the scored data lacks {missing}")
    status = scored["shariah_status"]
    named = status.notna() & (status.astype(str).str.strip() != "")
    unknown = sorted(set(status[named].astype(str)) - set(SHARIAH_STATUSES))
    if unknown:
        raise F.FeedError(f"unmapped shariah_status value(s) {unknown} — add them to the screen, never guess")
    out = pd.DataFrame(index=scored.index)
    out["symbol"] = scored["company_id"].astype(str)
    out["shariah_status"] = status.astype(str).where(named, "")
    out["cap"] = F._codes(scored["market_category"], F.CAP_CODES, "market_category")
    for c in RANK_NUMBERS:
        x = pd.to_numeric(scored[c], errors="coerce")
        out[c] = x.where(x > 0) if c in POSITIVE else x
    pct = S._pct(scored, "momentum_score")                     # the WHOLE market, as the recipe ranks it
    out["momentum_pct"] = pct.astype(str).where(pct.notna(), "")   # str(float) is its shortest exact repr
    out["sector"] = scored["sector"].where(scored["sector"].notna(), "").astype(str)
    out["stage"] = F._codes(scored["weinstein_stage"], F.STAGE_CODES, "weinstein_stage")

    raw = pd.to_numeric(scored["days_from_result"], errors="coerce")
    integral = raw.notna() & (raw.abs() <= MAX_RESULT_DAYS) & np.isclose(raw, np.round(raw))
    days = np.round(raw).where(integral)                       # NaN = unknown
    when = (pd.Timestamp(session) + pd.to_timedelta(days, unit="D")).dt.strftime("%Y-%m-%d")
    out["next_result_date"] = when.where(days >= 0, "")
    if previous is None or previous.empty:
        carried = pd.Series(np.nan, index=scored.index, dtype=object)
    else:
        carried = out["symbol"].map(previous.set_index("symbol")["last_result_date"]).replace("", np.nan)
    out["last_result_date"] = when.where(days <= 0, carried).fillna("")
    shares = pd.to_numeric(scored["equity_shares"], errors="coerce")
    out["equity_shares"] = shares.where(shares > 0).round().astype("Int64")
    stats = {"carried": int((~(days <= 0) & carried.notna()).sum()), "unknown_days": int((~integral).sum())}
    frame = out[list(RANK_COLUMNS)].sort_values("symbol", kind="mergesort").reset_index(drop=True)
    return frame, stats


def rank_csv_bytes(frame: pd.DataFrame) -> bytes:
    """UTF-8, \\n, numbers as the feed writes them; momentum_pct is already its exact text."""
    return frame.to_csv(index=False, float_format=F.FLOAT_FORMAT, lineterminator="\n").encode("utf-8")


def validate_rank(frame: pd.DataFrame) -> None:
    n = len(frame)
    if not MIN_RANK_ROWS <= n <= MAX_RANK_ROWS:
        raise F.FeedError(f"{n} stocks in the rank list, outside the measured {MIN_RANK_ROWS}-{MAX_RANK_ROWS}")
    dup = sorted(frame.loc[frame["symbol"].duplicated(), "symbol"].unique())
    if dup:
        raise F.FeedError(f"duplicate symbol(s) in the rank list {dup[:5]}")
    bad = sorted(frame.loc[~frame["symbol"].str.fullmatch(F.SYMBOL.pattern), "symbol"])
    if bad:
        raise F.FeedError(f"malformed symbol(s) in the rank list {bad[:5]}")


def _share(scored: pd.DataFrame, col: str) -> float:
    """The share of stocks carrying the reading; an absent column carries none (never skipped)."""
    if col not in scored.columns:
        return 0.0
    s = scored[col]
    if col in TEXT_FLOORS:
        ok = D._filled(s)
    elif col in POSITIVE:
        ok = pd.to_numeric(s, errors="coerce") > 0
    elif col == "weinstein_stage":
        ok = D._filled(s) & (s.astype(str) != UNKNOWN_STAGE)
    else:
        ok = pd.to_numeric(s, errors="coerce").notna()
    return float(ok.mean())


def check_rank_inputs(scored: pd.DataFrame) -> list:
    """Two 🩺 results in the feed's own form: every rank-list input at or above its floor, and the Shariah share
    inside its measured band."""
    share = {c: _share(scored, c) for c in RANK_FLOORS}
    thin = [f"{c} {share[c]:.1%}" for c in RANK_FLOORS if share[c] < RANK_FLOORS[c]]
    sh = (float((scored["shariah_status"].astype(str) == SHARIAH_STATUSES[0]).mean())
          if "shariah_status" in scored.columns else 0.0)
    return [D._ok(not thin, "Rank list inputs", f"below the floor: {thin}" if thin
                  else f"all {len(RANK_FLOORS)} at or above their floor"),
            D._ok(SHARIAH_SHARE_MIN <= sh <= SHARIAH_SHARE_MAX, "Shariah share",
                  f"{sh:.1%} labelled Shariah, the measured band is {SHARIAH_SHARE_MIN:.0%}-{SHARIAH_SHARE_MAX:.0%}")]


# ── The branch: both lists of a session, written once ────────────────────────────────────────────────────────────────
SCHEMA = 1
RANK_DIR = "rank"


def _table(columns) -> str:
    return "| # | Column |\n|---|---|\n" + "\n".join(f"| {i} | `{c}` |" for i, c in enumerate(columns, 1))


def readme_text() -> str:
    return (f"# Kite v2's lists — schema {SCHEMA}\n\n"
            "PRISM's two lists for Kite v2, published by `ui/v2_lists_job.py` on `main`. Every list of every session "
            "comes from ONE commit, named in each manifest as `engine_commit`. Per session:\n\n"
            "- `sessions/<YYYY-MM-DD>.csv`: the 🚀 Playbook list, exactly the bytes the Playbook feed's code builds at "
            "the manifest's `engine_commit` (not the `playbook-feed` branch, which is pinned to an older commit);\n"
            "- `rank/<YYYY-MM-DD>.csv`: the rank list, one row per stock PRISM scored that session;\n"
            "- `sessions/<YYYY-MM-DD>.json`: the manifest, written LAST. It carries the session (from the sheet's own "
            "name); each list's `rows`, `columns` and `csv_sha256`; the recipe fingerprint; the engine commit; the "
            "source workbook's sha256; whether the session was `recorded` as it closed or `backfilled` later; the "
            "checks; the Playbook recipe's `blanks` (as the Playbook feed's schema 2 states them); the rank list's "
            "counts; and `carried_from`, the earlier session whose `last_result_date` values were carried.\n\n"
            "Every file is written once, never modified, never deleted. A session is valid only when its manifest "
            "exists and both files match its hashes; a missing or invalid session means UNKNOWN.\n\n"
            "## The Playbook list\n\nOnly members are listed: for a valid session, absence means \"not in the list "
            f"today\".\n\n{_table(F.COLUMNS)}\n\n"
            f"## The rank list\n\nEvery scored stock, members or not.\n\n{_table(RANK_COLUMNS)}\n\n"
            f"`shariah_status`: {', '.join(SHARIAH_STATUSES)} or empty. "
            f"`cap`: {' '.join(sorted(set(F.CAP_CODES.values())))} or empty. `stage`: Weinstein 1 2 3 4 or empty. "
            "`momentum_pct`: the stock's momentum percentile over the WHOLE scored market, exactly as the 🚀 recipe "
            "reads it, written as the shortest text that reads back to the same number.\n\n"
            "The result dates come from the vendor's `days_from_result` (negative = days since the last result). "
            "`next_result_date` is the session plus it when it is 0 or more. `last_result_date` is the session plus "
            "it when it is 0 or less, otherwise that stock's `last_result_date` on the newest earlier session of this "
            "branch (`carried_from`), or empty. A blank or non-whole `days_from_result` is unknown.\n\n"
            f"Both lists: rows sorted by `symbol`; UTF-8, `\\n`; numbers `{F.FLOAT_FORMAT}`; dates `YYYY-MM-DD`; "
            "empty = not known.\n")


def _paths(lists: Path, session: str) -> tuple[Path, Path, Path]:
    """(the Playbook list, the rank list, the manifest) of `session`."""
    sessions = Path(lists) / F.SESSIONS
    return sessions / f"{session}.csv", Path(lists) / RANK_DIR / f"{session}.csv", sessions / f"{session}.json"


def _write_whole(path: Path, data: bytes) -> None:
    """All of `data` at `path`, or nothing there: written beside it, then moved into place in one step."""
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    os.replace(part, path)


def write_pair(lists: Path, session: str, playbook: bytes, rank: bytes, meta: dict) -> bool:
    """True when written; False when exactly this session is already there: both files present with the bytes given
    AND with its manifest's own hashes. A file already on the branch with other bytes raises, and so does a manifest
    that does not match its files: nothing is ever rewritten. Each file is written whole or not at all; the rank list
    goes first, then the Playbook list (the file the feed's session rules count), and the manifest LAST, only once both
    files read back with its hashes: a session with a manifest is always whole. Files a stopped run left without their
    manifest are completed."""
    pb_path, rank_path, m_path = _paths(lists, session)
    files = (("rank", rank_path, rank), ("playbook", pb_path, playbook))
    for what, path, data in files:
        if path.exists() and path.read_bytes() != data:
            raise F.FeedError(f"{session}'s {what} list is already on the branch with different content — a "
                              "session's files are written once, never rewritten")
    if m_path.exists():
        sealed = json.loads(m_path.read_text(encoding="utf-8"))
        for what, path, _ in files:
            if not path.exists() or F.sha256(path.read_bytes()) != sealed[what]["csv_sha256"]:
                raise F.FeedError(f"{session}: its manifest does not match its {what} list on the branch — a session "
                                  "with a manifest is never completed or rewritten")
        return False
    for _, path, data in files:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_whole(path, data)
    for what, path, _ in files:
        if F.sha256(path.read_bytes()) != meta[what]["csv_sha256"]:
            raise F.FeedError(f"{session}: the written {what} list does not match its manifest's checksum")
    _write_whole(m_path, (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    readme = Path(lists) / "README.md"
    if not readme.exists() or readme.read_text(encoding="utf-8") != readme_text():
        readme.write_text(readme_text(), encoding="utf-8", newline="\n")
    attrs = Path(lists) / ".gitattributes"             # a CRLF checkout would break every hash
    if not attrs.exists() or attrs.read_bytes() != F.GITATTRIBUTES:
        attrs.write_bytes(F.GITATTRIBUTES)
    return True


def read_rank(lists: Path, session: str) -> pd.DataFrame:
    """A published rank list as text: every value exactly as written, a blank as ''."""
    return pd.read_csv(_paths(lists, session)[1], dtype=str, keep_default_na=False)


# ── The job: the feed's own gate, recipe and session rules, imported ─────────────────────────────────────────────────
TRIGGERS = ("after-feed", "schedule", "manual")


def publish(workbook: bytes, title: str, session: str, lists: Path, *, origin: str, trigger: str,
            now: datetime) -> bool:
    """F.publish's steps with the rank list beside them: score, gate (the feed's checks and the rank list's own),
    build and validate the Playbook list, refuse a stale copy, build the rank list carrying from the newest earlier
    session on the branch, write both once. True when a new session was written."""
    datasets, stocks, bench, scored = F.score(workbook)
    checks = F.gate(datasets, stocks, bench, scored, backfill=origin == "backfilled")
    own = check_rank_inputs(scored)
    failed = [r for r in own if r.status == "fail"]
    if failed:
        raise F.FeedError("data check failed — " + "; ".join(f"{r.name}: {r.detail}" for r in failed))
    frame = F.build_frame(scored)
    F.validate(frame)
    problem = F.stale_problem(frame, lists, session)
    if problem:
        raise F.FeedError(problem)
    before = [s for s in F.sessions_on_branch(lists) if s < session]
    carried_from = before[-1] if before else ""
    rank, stats = build_rank(scored, session, read_rank(lists, carried_from) if carried_from else None)
    validate_rank(rank)
    playbook_csv, rank_csv = F.to_csv_bytes(frame), rank_csv_bytes(rank)
    meta = {
        "schema": SCHEMA, "session": session,
        "playbook": {"rows": int(len(frame)), "columns": list(F.COLUMNS), "csv_sha256": F.sha256(playbook_csv)},
        "rank": {"rows": int(len(rank)), "columns": list(RANK_COLUMNS), "csv_sha256": F.sha256(rank_csv)},
        "recipe": "playbook", "recipe_fingerprint": S.recipe_fingerprint(F.playbook_scan()),
        "engine_commit": os.environ.get("V2_CODE_SHA") or engine_version(),
        "source_title": title, "source_sha256": F.sha256(workbook),
        "origin": origin, "trigger": trigger, "published_at": now.isoformat(timespec="seconds"),
        "checks": {"passed": checks["passed"] + len(own), "warnings": checks["warnings"]},
        "blanks": F.blanks(scored),
        "counts": {"cap": F._counts(rank["cap"], F.CAP_CODES), "stage": F._counts(rank["stage"], F.STAGE_CODES),
                   "shariah": {k: int((rank["shariah_status"] == k).sum()) for k in sorted({*SHARIAH_STATUSES, ""})}},
        "carried_from": carried_from, "carried": stats["carried"], "unknown_days": stats["unknown_days"],
    }
    return write_pair(lists, session, playbook_csv, rank_csv, meta)


# THE TWO ARCHIVE COPIES MEASURED WRONG (14 Oct 2026, against Kite's own daily closes; Kite 327efbb, I5). The stale
# guard compares a copy with the session before it, so alone it would publish '10 Sep' (11 Sep's data under 10 Sep's
# name) and refuse the true 11 Sep as its copy. No correct copy of either day exists, so the branch starts on 11 Sep,
# where addendum A5's in-sample lists start. Keyed by date: every export of one copy has different workbook bytes.
BACKFILL_REFUSED = {
    "2026-09-03": "the archive's copy is a mid-day snapshot: its closes match no trading day's (best 3.4%, against "
                  "Kite's daily bars, measured 14 Oct)",
    "2026-09-10": "the archive's copy holds 11 Sep's data: its NSE closes are Kite's 11 Sep closes on 99.1% of stocks "
                  "and 10 Sep's on 0% (measured 14 Oct)",
}


def backfill(folder: Path, lists: Path, *, now: datetime | None = None) -> tuple[list, list]:
    """Every saved `PRISM ….xlsx` in `folder` whose name carries a session, as `backfilled`, oldest SESSION first
    (never by file name). Returns (done, refused). A copy that fails a check is refused and the others still go in.
    So is a copy measured wrong (BACKFILL_REFUSED), and a new session older than the newest on the branch: its rank
    list would carry from a later session, and every list after it has already carried without it. A session already
    on the branch is rebuilt from its copy with the same carry and compared byte for byte (write_pair): the same bytes
    are a no-op, other bytes are refused. That is how another machine proves it makes the same lists."""
    now = now or datetime.now(IST)
    copies = sorted((s, p.name, p) for p in Path(folder).glob("PRISM *.xlsx") if (s := F._session(p.stem)))
    done, refused = [], []
    for session, _, path in copies:
        if session in BACKFILL_REFUSED:
            refused.append(f"{session}: {BACKFILL_REFUSED[session]}")
            continue
        have = F.sessions_on_branch(lists)
        if session not in have and have and session < have[-1]:
            refused.append(f"{session}: older than {have[-1]}, the newest session on the branch — a back-fill never "
                           "goes behind the carry")
            continue
        try:
            if publish(path.read_bytes(), path.stem, session, lists, origin="backfilled", trigger="backfill",
                       now=now):
                done.append(session)
            else:
                print(f"  = {session} is already on the branch, byte for byte")
        except F.FeedError as exc:
            refused.append(f"{session}: {exc}")
    for line in refused:
        print(f"  ✗ not back-filled {line}")
    return done, refused


# ── The run ──────────────────────────────────────────────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(_REPO), *args], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _pinned_code_problem(git=None, env=None) -> str:
    """Why this run's code is not the ONE commit every manifest it writes will name (A5), or ''. On CI the checkout IS
    V2_CODE_SHA; on the PC a back-fill must run from a clean checkout of it. Untracked files (lists, copies) do not
    count: they are not code."""
    git, env = git or _git, os.environ if env is None else env
    want, head = env.get("V2_CODE_SHA", ""), git("rev-parse", "HEAD")
    if not head or head != want:           # rev-parse gives all 40 characters: a short, empty or other pin never equals it
        return (f"the checkout is {head[:12] or 'not a git checkout'}, but V2_CODE_SHA names "
                f"{want[:12] or 'nothing'}")
    if git("status", "--porcelain", "--untracked-files=no"):
        return "the checkout has uncommitted changes, so its lists would not be the pinned commit's"
    return ""


def _gap(carried_from: str, session: str) -> str:
    """' · ⚠ no session on the branch since X' when the session before this one is older than the weekday before it:
    a weekday holiday, or sessions lost for good (a live run publishes across a gap; no back-fill goes behind it)."""
    if not carried_from or carried_from >= _prev_weekday(date.fromisoformat(session)).isoformat():
        return ""
    return f" · ⚠ no session on the branch since {carried_from}"


def _github_output(path: str, session: str, message: str) -> None:
    """What the workflow commits when something was written: the session, and its message."""
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"session={session}\nmessage=v2-lists: {message}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Publish Kite v2's two lists to the v2-lists branch.")
    ap.add_argument("--lists", required=True, type=Path, help="a checkout of the v2-lists branch")
    ap.add_argument("--backfill", type=Path, help="a folder of archived 'PRISM YYYY-MM-DD.xlsx' copies")
    args = ap.parse_args(argv)
    problem = _pinned_code_problem()
    if problem:
        raise SystemExit(f"refused: {problem}")
    out = os.environ.get("GITHUB_OUTPUT")
    if args.backfill:
        done, refused = backfill(args.backfill, args.lists)
        print(f"back-filled {len(done)} session(s): {done}")
        on = set(F.sessions_on_branch(args.lists))
        broken = [line for line in refused if line.split(":", 1)[0] in on]
        if broken:              # a copy of a session on the branch did not make its bytes: this machine, or the copy
            print(f"❌ {len(broken)} copy(ies) of sessions on the branch did not rebuild them byte for byte: nothing "
                  "to commit")
            return 1
        if done and out:
            _github_output(out, done[-1], f"back-filled {len(done)} session(s), {done[0]} to {done[-1]}")
        return 0
    raw = os.environ.get("SHEET_ID", "")
    sheet_id = sheet_id_from_secret(raw)
    print(D._secret_shape(raw, sheet_id))
    if not sheet_id:
        raise SystemExit("SHEET_ID holds no spreadsheet id — paste the live data sheet's id (the part of its "
                         "address between /d/ and /edit).")
    trigger = os.environ.get("V2_TRIGGER") or "manual"
    if trigger not in TRIGGERS:
        raise SystemExit(f"V2_TRIGGER {trigger!r} is not one of {TRIGGERS}")
    try:
        session = F.run_live(sheet_id, args.lists, trigger=trigger, publish_fn=publish)
    except Exception as exc:           # ANY failure: a download error names its URL, and the URL holds the sheet's id
        kind = "" if isinstance(exc, F.FeedError) else f"{type(exc).__name__}: "
        print(f"❌ nothing published: {kind}{_scrub(str(exc))}")
        print(f"   if {expected_session(datetime.now(IST)).isoformat()} is not on v2-lists before the next session is "
              "published, it can never be added: dispatch this workflow with backfill_from = backfill_to = that day")
        return 1
    if not session:
        print("nothing new to publish")
        return 0
    meta = json.loads(_paths(args.lists, session)[2].read_text(encoding="utf-8"))
    gap = _gap(meta["carried_from"], session)
    if out:
        _github_output(out, session, f"{session} · {meta['playbook']['rows']} Playbook · {meta['rank']['rows']} "
                                     f"ranked · recorded ({trigger}){gap}")
    print(f"✅ published {session}{gap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
