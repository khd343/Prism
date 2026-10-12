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
