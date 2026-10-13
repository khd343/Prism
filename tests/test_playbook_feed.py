"""
test_playbook_feed.py — the 🚀 Playbook feed (2026-10-12): PRISM's daily list for Kite Algo.

PRISM picks, Kite trades, and the two engines never share code: they meet at one file per trading session
on the `playbook-feed` branch, published by ui/playbook_feed_job.py. A wrong list would be TRADED, while a
missing one only pauses entries — so every rule here fails closed: the session comes from the sheet's own
name, read with the data; a failed check publishes nothing; a published session is never rewritten; the
bytes are deterministic so any copy can be checked against any other.

Run with: pytest tests/test_playbook_feed.py -v
"""
import contextlib
import hashlib
import io
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import BENCHMARK_INDICES  # noqa: E402
from core.data_engine import weinstein_stage  # noqa: E402
from core.sheet_meta import IST  # noqa: E402
from ui import data_check_job as D  # noqa: E402
from ui import playbook_feed_job as F  # noqa: E402
from ui import ui_scans as S  # noqa: E402

NOW = datetime(2026, 10, 12, 20, 15, tzinfo=IST)


@pytest.fixture(autouse=True)
def _small_lists(monkeypatch):
    """The real bounds are 100-450 stocks; the toy universe has 10 members."""
    monkeypatch.setattr(F, "MIN_ROWS", 1)


def _scored(n=30, members=10):
    """n stocks; the last `members` pass every Playbook leg (top-third momentum and breakout, a 🔥 Strong
    Quarter, top-half improvement, price above its stop) and the rest fail the stop and the quarter."""
    i = np.arange(n, dtype=float)
    m = i >= n - members
    return pd.DataFrame({
        "company_id": [f"NSE:S{k:02d}" for k in range(n)],
        "name": [f"Stock {k:02d}" for k in range(n)],
        "sector": "Alpha", "industry": "Widgets",
        "market_category": np.where(i % 2 == 0, "Small Cap", "Mid Cap"),
        "weinstein_stage": "📈 Stage 2 Advancing",
        "composite_score": 40 + i, "crs_52w": 1 + i / 100, "momentum_score": i, "breakout_score": i,
        "improvement_score": i, "dist_to_vstop": np.where(m, 5.0, -1.0),
        "traded_value_cr": 2.5, "rsi_14d": 60.0, "close_price": 100.0 + i, "equity_shares": 1e7 + i,
        "rev_pyq": 100.0, "rev_lq": np.where(m, 130.0, 101.0),
        "pat_pyq": 10.0, "pat_lq": np.where(m, 15.0, 10.0),
        "ebitda_pyq": 15.0, "ebitda_lq": np.where(m, 26.0, 15.0),
    })


def _meta(frame, csv=None, session="2026-10-12"):
    csv = F.to_csv_bytes(frame) if csv is None else csv
    return F.manifest(frame, csv, session, "PRISM 2026-10-12 Mon", b"workbook", "recorded", "dispatch",
                      {"passed": 1, "warnings": []}, NOW)


# ── 1. The rows are exactly the Playbook, from the imported recipe ────────────────────────────────
def test_rows_are_exactly_the_playbook():
    df = _scored()
    frame = F.build_frame(df)
    expected = sorted(df.loc[S.playbook_shortlist(df), "company_id"])
    assert frame["symbol"].tolist() == expected and len(expected) == 10


def test_columns_are_the_contract_in_order():
    assert F.build_frame(_scored()).columns.tolist() == list(F.COLUMNS)
    assert F.COLUMNS == ("symbol", "name", "sector", "industry", "cap", "composite_score", "crs_52w",
                         "dist_to_vstop", "momentum_score", "breakout_score", "traded_value_cr", "rsi_14d",
                         "stage", "close_price", "equity_shares")
    assert F.SCHEMA == 1


def test_a_missing_source_column_fails():
    with pytest.raises(F.FeedError, match="rsi_14d"):
        F.build_frame(_scored().drop(columns=["rsi_14d"]))


# ── 2. Codes, never display labels ────────────────────────────────────────────────────────────────
def test_every_stage_label_the_engine_can_emit_is_mapped():
    """The labels come from the engine's own function, all five branches, so a renamed label fails here."""
    c = pd.Series([10.0, 10.0, 5.0, 5.0, np.nan])
    s30 = pd.Series([8.0, 8.0, 8.0, 8.0, 8.0])
    s200 = pd.Series([6.0, 9.0, 6.0, 9.0, 6.0])
    assert set(weinstein_stage(c, s30, s200)) == set(F.STAGE_CODES)
    assert F.STAGE_CODES["📈 Stage 2 Advancing"] == "2" and F.STAGE_CODES["❔ Unknown"] == ""


def test_cap_codes_cover_the_six_vendor_categories():
    """The six values measured on all 24 archives (2026-10-12)."""
    assert F.CAP_CODES == {"Mega Cap": "MEGA", "Large Cap": "LARGE", "Mid Cap": "MID", "Small Cap": "SMALL",
                           "Micro Cap": "MICRO", "Nano Cap": "NANO"}


def test_an_unknown_label_fails_and_a_blank_one_is_empty():
    df = _scored()
    df.loc[df.index[-1], "market_category"] = "Small-Cap"          # a vendor re-spelling
    with pytest.raises(F.FeedError, match="Small-Cap"):
        F.build_frame(df)
    df.loc[df.index[-1], "market_category"] = np.nan                # PRISM simply has no category
    assert F.build_frame(df)["cap"].iloc[-1] == ""


# ── 3. Byte-deterministic ─────────────────────────────────────────────────────────────────────────
def test_same_input_same_bytes_whatever_the_row_order():
    df = _scored()
    a = F.to_csv_bytes(F.build_frame(df))
    b = F.to_csv_bytes(F.build_frame(df.sample(frac=1.0, random_state=7)))
    assert a == b
    assert not a.startswith(b"\xef\xbb\xbf") and b"\r\n" not in a and a.endswith(b"\n")
    assert a.split(b"\n")[0].decode() == ",".join(F.COLUMNS)


def test_nan_is_written_empty_never_zero():
    df = _scored()
    df.loc[df.index[-1], ["rsi_14d", "equity_shares"]] = np.nan
    row = F.to_csv_bytes(F.build_frame(df)).decode().splitlines()[-1].split(",")
    assert row[F.COLUMNS.index("rsi_14d")] == "" and row[F.COLUMNS.index("equity_shares")] == ""


def test_awkward_text_round_trips():
    df = _scored()
    df.loc[df.index[-1], "name"] = 'Alpha, "Beta"\nGamma Ltd'
    back = pd.read_csv(io.BytesIO(F.to_csv_bytes(F.build_frame(df))), keep_default_na=False)
    assert 'Alpha, "Beta"\nGamma Ltd' in back["name"].tolist()


# ── 4. Validation fails closed ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mutate,match", [
    (lambda f: f.assign(symbol=["NSE:S20"] * len(f)), "duplicate"),
    (lambda f: f.assign(symbol=f["symbol"].str.replace("NSE:", "nse:")), "symbol"),
    (lambda f: f.assign(close_price=0.0), "close_price"),
    (lambda f: f.assign(name=""), "name"),
])
def test_validate_refuses_a_bad_list(mutate, match):
    with pytest.raises(F.FeedError, match=match):
        F.validate(mutate(F.build_frame(_scored())))


def test_validate_refuses_a_list_outside_the_measured_bounds(monkeypatch):
    frame = F.build_frame(_scored())
    monkeypatch.setattr(F, "MIN_ROWS", 11)
    with pytest.raises(F.FeedError, match="outside"):
        F.validate(frame)


def test_the_real_bounds_are_the_measured_ones():
    assert F._MEASURED_BOUNDS == (100, 450)


# ── 5. The manifest ───────────────────────────────────────────────────────────────────────────────
def test_the_manifest_identifies_list_input_and_recipe():
    frame = F.build_frame(_scored())
    csv = F.to_csv_bytes(frame)
    m = _meta(frame, csv)
    assert m["csv_sha256"] == hashlib.sha256(csv).hexdigest()
    assert m["source_sha256"] == hashlib.sha256(b"workbook").hexdigest()
    assert m["recipe_fingerprint"] == S.recipe_fingerprint(F.playbook_scan())
    assert m["rows"] == 10 and m["columns"] == list(F.COLUMNS) and m["schema"] == 1
    assert sum(m["counts"]["cap"].values()) == 10 and sum(m["counts"]["stage"].values()) == 10
    assert m["published_at"] == "2026-10-12T20:15:00+05:30"
    assert {"session", "engine_commit", "source_title", "origin", "trigger", "checks"} <= set(m)


# ── 6. The branch: written once ───────────────────────────────────────────────────────────────────
def test_write_once_identical_is_a_no_op_different_raises(tmp_path):
    frame = F.build_frame(_scored())
    csv = F.to_csv_bytes(frame)
    assert F.write_session(tmp_path, "2026-10-12", csv, _meta(frame, csv)) is True
    assert F.sessions_on_branch(tmp_path) == ["2026-10-12"]
    assert F.write_session(tmp_path, "2026-10-12", csv, _meta(frame, csv)) is False
    with pytest.raises(F.FeedError, match="never rewritten"):
        F.write_session(tmp_path, "2026-10-12", csv + b"x", _meta(frame, csv + b"x"))
    assert (tmp_path / "sessions" / "2026-10-12.csv").read_bytes() == csv
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == F.readme_text()


def test_a_csv_without_its_manifest_is_completed(tmp_path):
    frame = F.build_frame(_scored())
    csv = F.to_csv_bytes(frame)
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "2026-10-12.csv").write_bytes(csv)           # a run died here
    F.write_session(tmp_path, "2026-10-12", csv, _meta(frame, csv))
    assert json.loads((tmp_path / "sessions" / "2026-10-12.json").read_text(encoding="utf-8"))["rows"] == 10


def test_the_branch_tells_git_never_to_touch_the_bytes(tmp_path):
    """Windows git (core.autocrlf) would check a session out with CRLF: its sha256 would stop matching
    its manifest and an identical re-publish would read as 'different content'. `* -text` forbids it."""
    frame = F.build_frame(_scored())
    csv = F.to_csv_bytes(frame)
    F.write_session(tmp_path, "2026-10-12", csv, _meta(frame, csv))
    assert (tmp_path / ".gitattributes").read_bytes() == b"* -text\n"


def test_the_readme_lists_the_contract():
    text = F.readme_text()
    assert all(f"`{c}`" in text for c in F.COLUMNS) and "never modified" in text


# ── 7. The gate fails closed; the back-fill tolerance is exact and never live ─────────────────────
def _all_pass(monkeypatch, **override):
    """Every 🩺 check passes unless overridden with a list of Results."""
    names = ["check_tabs", "check_universe", "check_benchmarks", "check_coverage", "check_types",
             "check_taxonomy", "check_identities", "check_engine"]
    for n in names:
        res = override.get(n, [D.Result(n, "ok", "fine")])
        monkeypatch.setattr(D, n, lambda *a, _r=res: list(_r))


def test_a_failing_check_publishes_nothing(monkeypatch, tmp_path):
    _all_pass(monkeypatch, check_engine=[D.Result("Scores not flat", "fail", "flat: ['momentum_score']")])
    monkeypatch.setattr(F, "score", lambda wb: ({}, None, None, _scored()))
    with pytest.raises(F.FeedError, match="Scores not flat"):
        F.publish(b"wb", "PRISM 2026-10-12 Mon", "2026-10-12", tmp_path, origin="recorded",
                  trigger="dispatch", now=NOW)
    assert F.sessions_on_branch(tmp_path) == [] and not (tmp_path / "sessions").exists()


def test_backfill_tolerates_exactly_the_two_later_additions(monkeypatch):
    _all_pass(monkeypatch, check_benchmarks=[D.Result("Index rows", "fail", "missing or no price: [...]")],
              check_coverage=[D.Result("Key columns", "fail", "listed_days absent")])
    checks = F.gate({}, None, None, None, backfill=True)
    assert "Key columns: listed_days absent" in checks["warnings"]
    _all_pass(monkeypatch, check_coverage=[D.Result("Key columns", "fail", "listed_days absent; pe 50% < 80%")])
    with pytest.raises(F.FeedError, match="pe 50%"):
        F.gate({}, None, None, None, backfill=True)


def test_backfill_tolerance_never_applies_live(monkeypatch):
    _all_pass(monkeypatch, check_benchmarks=[D.Result("Index rows", "fail", "missing or no price: [...]")])
    with pytest.raises(F.FeedError, match="Index rows"):
        F.gate({}, None, None, None, backfill=False)
    _all_pass(monkeypatch, check_coverage=[D.Result("Key columns", "fail", "listed_days absent")])
    with pytest.raises(F.FeedError, match="listed_days"):
        F.gate({}, None, None, None, backfill=False)


def test_warnings_are_recorded_not_blocking(monkeypatch, tmp_path):
    _all_pass(monkeypatch, check_taxonomy=[D.Result("Industry names", "warn", "3 unmapped (0.4%)")])
    monkeypatch.setattr(F, "score", lambda wb: ({}, None, None, _scored()))
    assert F.publish(b"wb", "PRISM 2026-10-12 Mon", "2026-10-12", tmp_path, origin="recorded",
                     trigger="dispatch", now=NOW) is True
    meta = json.loads((tmp_path / "sessions" / "2026-10-12.json").read_text(encoding="utf-8"))
    assert meta["checks"]["warnings"] == ["Industry names: 3 unmapped (0.4%)"]
    assert meta["origin"] == "recorded" and meta["trigger"] == "dispatch"


# ── 8. The live run: the session comes from the sheet, read with the data ─────────────────────────
class _Sheet:
    """A fake sheet: `titles` is what successive name reads return; downloads return `download_title`."""

    def __init__(self, monkeypatch, titles, download_title=None):
        self.titles, self.downloads = list(titles), 0
        self.download_title = download_title
        monkeypatch.setattr(F, "fetch_sheet_title", self._title)
        monkeypatch.setattr(F, "download", self._download)
        monkeypatch.setattr(F, "publish", self._publish)
        self.published = []

    def _title(self, sheet_id):
        return self.titles.pop(0) if len(self.titles) > 1 else self.titles[0]

    def _download(self, sheet_id):
        self.downloads += 1
        t = self.download_title if self.download_title is not None else self.titles[0]
        return t, b"xlsx"

    def _publish(self, workbook, title, session, feed, *, origin, trigger, now):
        self.published.append((title, session, origin, trigger))
        return True


def _published(tmp_path, *sessions):
    (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)
    for s in sessions:
        (tmp_path / "sessions" / f"{s}.csv").write_bytes(b"x")


def test_a_new_session_is_published_with_its_name(monkeypatch, tmp_path):
    _published(tmp_path, "2026-10-09")
    sheet = _Sheet(monkeypatch, ["PRISM 2026-10-12 Mon"])
    assert F.run_live("id", tmp_path, trigger="dispatch", now=NOW) == "2026-10-12"
    assert sheet.published == [("PRISM 2026-10-12 Mon", "2026-10-12", "recorded", "dispatch")]


def test_an_already_published_session_is_a_no_op_without_a_download(monkeypatch, tmp_path):
    _published(tmp_path, "2026-10-12")
    sheet = _Sheet(monkeypatch, ["PRISM 2026-10-12 Mon"])
    assert F.run_live("id", tmp_path, trigger="schedule", now=NOW) is None
    assert sheet.downloads == 0


def test_a_dispatch_waits_for_the_rename(monkeypatch, tmp_path):
    _published(tmp_path, "2026-10-09")
    sheet = _Sheet(monkeypatch, ["PRISM 2026-10-09 Fri", "PRISM 2026-10-09 Fri", "PRISM 2026-10-12 Mon"])
    naps = []
    assert F.run_live("id", tmp_path, trigger="dispatch", now=NOW, sleep=naps.append) == "2026-10-12"
    assert naps == [F.WAIT_STEP, F.WAIT_STEP]


def test_a_dispatch_gives_up_after_the_wait(monkeypatch, tmp_path):
    _published(tmp_path, "2026-10-09")
    sheet = _Sheet(monkeypatch, ["PRISM 2026-10-09 Fri"])
    naps = []
    assert F.run_live("id", tmp_path, trigger="dispatch", now=NOW, sleep=naps.append) is None
    assert sum(naps) == F.WAIT_SECONDS and sheet.downloads == 0


def test_a_scheduled_run_never_waits(monkeypatch, tmp_path):
    _published(tmp_path, "2026-10-09")
    _Sheet(monkeypatch, ["PRISM 2026-10-09 Fri"])
    naps = []
    assert F.run_live("id", tmp_path, trigger="schedule", now=NOW, sleep=naps.append) is None
    assert naps == []


def test_an_unreadable_sheet_name_fails_loudly(monkeypatch, tmp_path):
    _Sheet(monkeypatch, [None])
    with pytest.raises(F.FeedError, match="no session date"):
        F.run_live("id", tmp_path, trigger="schedule", now=NOW)


@pytest.mark.parametrize("title", ["PRISM 2026-10-12 Mon", "PRISM 2026-10-12"])
def test_both_name_forms_give_the_session(monkeypatch, tmp_path, title):
    sheet = _Sheet(monkeypatch, [title])
    assert F.run_live("id", tmp_path, trigger="manual", now=NOW) == "2026-10-12"
    assert sheet.published[0][1] == "2026-10-12"


def test_a_rename_during_the_download_is_read_again(monkeypatch, tmp_path):
    _published(tmp_path, "2026-10-09")
    sheet = _Sheet(monkeypatch, ["PRISM 2026-10-12 Mon"], download_title="PRISM 2026-10-13 Tue")
    with pytest.raises(F.FeedError, match="renamed"):
        F.run_live("id", tmp_path, trigger="manual", now=datetime(2026, 10, 13, 21, 0, tzinfo=IST))
    assert sheet.downloads == 2 and sheet.published == []


def test_an_older_sheet_is_nothing_to_do_and_a_future_one_fails(monkeypatch, tmp_path):
    _published(tmp_path, "2026-10-12")
    _Sheet(monkeypatch, ["PRISM 2026-10-09 Fri"])
    assert F.run_live("id", tmp_path, trigger="manual", now=NOW) is None
    _Sheet(monkeypatch, ["PRISM 2026-10-14 Wed"])
    with pytest.raises(F.FeedError, match="ahead"):
        F.run_live("id", tmp_path, trigger="manual", now=NOW)


# ── 9. Back-fill and the command line ─────────────────────────────────────────────────────────────
def test_backfill_publishes_each_archive_once_and_reports_refusals(monkeypatch, tmp_path):
    archives = tmp_path / "archive"
    archives.mkdir()
    for name in ["PRISM 2026-06-29.xlsx", "PRISM 2026-09-10.xlsx", "PRISM 2026-10-09.xlsx", "notes.xlsx"]:
        (archives / name).write_bytes(name.encode())
    feed = tmp_path / "feed"
    _published(feed, "2026-10-09")                                   # already recorded: never re-written
    seen = []

    def fake_publish(workbook, title, session, feed_, *, origin, trigger, now):
        if session == "2026-06-29":
            raise F.FeedError("Cash-flow year decided: 36.7% undecided")
        seen.append((title, session, origin, trigger))
        return True

    monkeypatch.setattr(F, "publish", fake_publish)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert F.backfill(archives, feed, now=NOW) == ["2026-09-10"]
    assert seen == [("PRISM 2026-09-10", "2026-09-10", "backfilled", "backfill")]
    assert "2026-06-29" in out.getvalue() and "36.7%" in out.getvalue()


_SECRET = "1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789abcd"


def test_main_writes_the_outputs_and_never_prints_the_id(monkeypatch, tmp_path):
    feed = tmp_path / "feed"
    (feed / "sessions").mkdir(parents=True)
    (feed / "sessions" / "2026-10-12.json").write_text(json.dumps({"rows": 271}), encoding="utf-8")
    monkeypatch.setattr(F, "run_live", lambda sid, f, *, trigger: "2026-10-12" if sid == _SECRET else None)
    gh_out = tmp_path / "out.txt"
    monkeypatch.setenv("SHEET_ID", f'SHEET_ID = "{_SECRET}"')
    monkeypatch.setenv("FEED_TRIGGER", "dispatch")
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh_out))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert F.main(["--feed", str(feed)]) == 0
    assert _SECRET not in out.getvalue()
    assert gh_out.read_text(encoding="utf-8") == (
        "session=2026-10-12\nmessage=feed: 2026-10-12 · 271 stocks · recorded (dispatch)\n")


def test_main_turns_a_refusal_into_a_failed_run(monkeypatch, tmp_path):
    def refuse(*a, **k):
        raise F.FeedError("data check failed — Universe: 12 stocks")
    monkeypatch.setattr(F, "run_live", refuse)
    monkeypatch.setenv("SHEET_ID", _SECRET)
    monkeypatch.delenv("FEED_TRIGGER", raising=False)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert F.main(["--feed", str(tmp_path)]) == 1
    assert "Universe: 12 stocks" in out.getvalue()


def test_main_refuses_a_missing_secret_or_an_unknown_trigger(monkeypatch, tmp_path):
    monkeypatch.delenv("SHEET_ID", raising=False)
    with pytest.raises(SystemExit):
        F.main(["--feed", str(tmp_path)])
    monkeypatch.setenv("SHEET_ID", _SECRET)
    monkeypatch.setenv("FEED_TRIGGER", "cron")
    with pytest.raises(SystemExit):
        F.main(["--feed", str(tmp_path)])


# ── 10. On a real archive: the whole path, twice, same bytes ──────────────────────────────────────
_ARCHIVE = ROOT / "Other Resources" / "archive-cache" / "PRISM 2026-10-09.xlsx"


@pytest.mark.skipif(not _ARCHIVE.exists(), reason="the archive copies are local only (gitignored)")
def test_a_real_archive_publishes_within_bounds_and_deterministically(monkeypatch, tmp_path):
    monkeypatch.setattr(F, "MIN_ROWS", F._MEASURED_BOUNDS[0])
    wb = _ARCHIVE.read_bytes()
    for feed in (tmp_path / "a", tmp_path / "b"):
        assert F.publish(wb, "PRISM 2026-10-09", "2026-10-09", feed, origin="backfilled", trigger="backfill",
                         now=NOW) is True
    a = (tmp_path / "a" / "sessions" / "2026-10-09.csv").read_bytes()
    assert a == (tmp_path / "b" / "sessions" / "2026-10-09.csv").read_bytes()
    meta = json.loads((tmp_path / "a" / "sessions" / "2026-10-09.json").read_text(encoding="utf-8"))
    assert 262 <= meta["rows"] <= 288 and meta["checks"]["warnings"] == []


# ── 11. The workflow ──────────────────────────────────────────────────────────────────────────────
_WF_PATH = ROOT / ".github" / "workflows" / "playbook-feed.yml"


def _wf():
    return _WF_PATH.read_text(encoding="utf-8")


def test_the_workflow_is_dispatched_with_a_backstop_schedule():
    y = _wf()
    assert "workflow_dispatch:" in y and re.search(r"options:\s*\[manual, dispatch\]", y)
    assert re.findall(r'cron:\s*"([^"]+)"', y) == ["5 14 * * 1-5", "30 21 * * *"]


def test_the_workflow_writes_only_the_feed_branch():
    y = _wf()
    assert "contents: write" in y and "permissions:" in y
    assert "ref: playbook-feed" in y and "path: feed" in y
    pushes = re.findall(r"git push\s+(.+)", y)
    assert pushes == ["origin HEAD:playbook-feed"], pushes
    assert "main" not in " ".join(pushes)
    assert "git add .gitattributes README.md sessions" in y, "the no-conversion rule travels with the files"


def test_the_workflow_runs_the_job_serially_on_a_pinned_runner():
    y = _wf()
    assert re.search(r"concurrency:\s*\n\s+group: playbook-feed\s*\n\s+cancel-in-progress: false", y)
    assert re.findall(r"runs-on:\s*(\S+)", y) == ["ubuntu-24.04"]
    assert "python -m ui.playbook_feed_job --feed feed" in y
    assert "SHEET_ID: ${{ secrets.SHEET_ID }}" in y
    assert re.findall(r"secrets\.(\w+)", y) == ["SHEET_ID"], "no other secret is handed to this job"


def test_the_commit_message_reaches_git_through_the_environment():
    """Never `${{ }}`-interpolated into the shell: an output is data, not script."""
    y = _wf()
    assert 'git commit -m "$MSG"' in y and "MSG: ${{ steps.publish.outputs.message }}" in y


# ── 12. A stale copy is not a session (2026-10-12) ────────────────────────────────────────────────
# The sync named two NSE HOLIDAYS as sessions (14 Sep, 2 Oct) and the feed published them: 90% and 87% of their
# closes were identical to the day before, against ~2% on real trading days. A copy of yesterday is refused.
def _on_branch(feed, day, df):
    frame = F.build_frame(df)
    csv = F.to_csv_bytes(frame)
    F.write_session(feed, day, csv, _meta(frame, csv, session=day))


def test_a_copy_of_the_previous_session_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "MIN_COMMON", 5)
    _on_branch(tmp_path, "2026-10-09", _scored())
    problem = F.stale_problem(F.build_frame(_scored()), tmp_path, "2026-10-12")
    assert "stale" in problem and "2026-10-09" in problem and "100%" in problem


def test_half_identical_is_the_measured_line(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "MIN_COMMON", 5)
    df = _scored()
    _on_branch(tmp_path, "2026-10-09", df)
    m = S.playbook_shortlist(df)
    members = df.index[m]
    half = df.copy(); half.loc[members[:5], "close_price"] += 1.0          # 5 of 10 moved: 50% same
    assert "stale" in F.stale_problem(F.build_frame(half), tmp_path, "2026-10-12")
    most = df.copy(); most.loc[members[:6], "close_price"] += 1.0          # 6 of 10 moved: 40% same
    assert F.stale_problem(F.build_frame(most), tmp_path, "2026-10-12") == ""


def test_the_first_session_or_a_small_overlap_cannot_be_judged(tmp_path, monkeypatch):
    assert F.stale_problem(F.build_frame(_scored()), tmp_path, "2026-10-12") == ""   # nothing before it
    _on_branch(tmp_path, "2026-10-09", _scored())
    monkeypatch.setattr(F, "MIN_COMMON", 11)                                       # 10 shared < 11
    assert F.stale_problem(F.build_frame(_scored()), tmp_path, "2026-10-12") == ""


def test_only_an_earlier_session_is_compared(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "MIN_COMMON", 5)
    _on_branch(tmp_path, "2026-10-13", _scored())                     # LATER than the one being judged
    assert F.stale_problem(F.build_frame(_scored()), tmp_path, "2026-10-12") == ""


def test_publish_refuses_a_stale_session_and_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(F, "MIN_COMMON", 5)
    _all_pass(monkeypatch)
    monkeypatch.setattr(F, "score", lambda wb: ({}, None, None, _scored()))
    assert F.publish(b"wb", "PRISM 2026-10-09 Fri", "2026-10-09", tmp_path, origin="recorded",
                     trigger="dispatch", now=NOW) is True
    with pytest.raises(F.FeedError, match="stale"):
        F.publish(b"wb", "PRISM 2026-10-12 Mon", "2026-10-12", tmp_path, origin="recorded",
                  trigger="dispatch", now=NOW)
    assert F.sessions_on_branch(tmp_path) == ["2026-10-09"]


_A = ROOT / "Other Resources" / "archive-cache"


@pytest.mark.skipif(not (_A / "PRISM 2026-09-14.xlsx").exists(), reason="the archive copies are local only")
def test_the_real_holiday_is_refused_and_the_next_real_day_is_not(monkeypatch, tmp_path):
    monkeypatch.setattr(F, "MIN_ROWS", F._MEASURED_BOUNDS[0])
    pub = lambda d: F.publish((_A / f"PRISM {d}.xlsx").read_bytes(), f"PRISM {d}", d, tmp_path,   # noqa: E731
                              origin="backfilled", trigger="backfill", now=NOW)
    assert pub("2026-09-11") is True
    with pytest.raises(F.FeedError, match="stale"):
        pub("2026-09-14")                                             # Ganesh Chaturthi: NSE closed
    assert pub("2026-09-15") is True                                  # a real session after the holiday


# ── 13. The live gate's index rows are the seven Kite's test opened with (2026-10-13) ──────────────
# The window opened on 12 Oct with the live feed refusing a session unless these SEVEN index rows were priced.
# PRISM 2a87ac4 (13 Oct) added two display indices to config.BENCHMARK_INDICES and the gate silently grew to nine:
# a missing row of either would have cost Kite a whole session, for rows the Playbook never reads.
_WINDOW_SEVEN = ("NSE:CNX500", "NSE:SMALLCA250", "NSE:NFMICRO250", "NSE:NIF500MO50", "NSE:NI200MOM30",
                 "NSE:NQUALITY30", "NSE:NS250MQ100")


def _index_rows(ids, price=100.0):
    return pd.DataFrame({"company_id": list(ids), "close_price": [price] * len(ids)})


def _live_gate(monkeypatch, bench):
    """The live gate with every other check passing and the REAL index-row check."""
    real = D.check_benchmarks
    _all_pass(monkeypatch)
    monkeypatch.setattr(D, "check_benchmarks", real)
    return F.gate({}, None, bench, None, backfill=False)


def test_the_live_gate_requires_exactly_the_seven_the_window_opened_with():
    assert F.FEED_GATE_INDICES == _WINDOW_SEVEN
    assert set(F.FEED_GATE_INDICES) <= set(BENCHMARK_INDICES), "a misspelt symbol would never be checked"


def test_a_display_index_without_a_price_cannot_cost_kite_a_session(monkeypatch):
    later = sorted(set(BENCHMARK_INDICES) - set(_WINDOW_SEVEN))
    assert later, "without a display-only index this test proves nothing"
    checks = _live_gate(monkeypatch, _index_rows(_WINDOW_SEVEN))                        # the later rows absent
    assert checks == {"passed": 8, "warnings": []}, "all eight checks ran, the index rows among them"
    unpriced = pd.concat([_index_rows(_WINDOW_SEVEN), _index_rows(later, price=float("nan"))])
    assert _live_gate(monkeypatch, unpriced) == {"passed": 8, "warnings": []}           # present, no price


def test_any_of_the_seven_without_a_price_still_publishes_nothing(monkeypatch):
    for gone in _WINDOW_SEVEN:
        with pytest.raises(F.FeedError, match="Index rows"):
            _live_gate(monkeypatch, _index_rows([i for i in BENCHMARK_INDICES if i != gone]))


def test_the_daily_check_still_requires_every_index():
    assert D.check_benchmarks(_index_rows(_WINDOW_SEVEN))[0].status == "fail"    # a display row missing is mailed
    assert D.check_benchmarks(_index_rows(BENCHMARK_INDICES))[0].status == "ok"
