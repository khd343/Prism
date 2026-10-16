"""
test_archive_copies.py
======================
Contract for ui/archive_copies.py (2026-10-16): the PRISM Archive's copies for a date range, downloaded read-only as
the saved sheets window 1's chosen run is rebuilt from (Kite's docs/readout-window-1.md, frozen page addendum
2026-10-15).

Measured before writing: the index the operator shared lists 12, 13, 14 Oct (`daily`) and 15 Oct (`ok`); each copy,
back-filled with the feed's pinned code 68b38e0, rebuilt the list Kite received byte for byte (4 of 4). The engine's
own loader reads every tab of each copy with exactly the stocks the index counted.

Run with: pytest tests/test_archive_copies.py -v
"""
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui import archive_copies as A  # noqa: E402

TABS = ("balance", "cashflow", "income", "ratio", "shareholding", "technical")
_SID = {d: f"copy{d.replace('-', '')}" + "x" * 32 for d in ("2026-10-09", "2026-10-12", "2026-10-13", "2026-10-14",
                                                               "2026-10-15", "2026-10-16")}
_ARCHIVE = ROOT / "Other Resources" / "archive-cache" / "PRISM 2026-10-09.xlsx"


def _index(rows=None) -> pd.DataFrame:
    rows = rows if rows is not None else [
        ("2026-10-09", "daily", 2719), ("2026-10-12", "daily", 2717), ("2026-10-13", "daily", 2719),
        ("2026-10-14", "daily", 2704), ("2026-10-15", "ok", 2704)]
    return pd.DataFrame([{"vintage_date": d, "fy_quarter": "FY27Q1 (off-cycle)", "spreadsheet_id": _SID.get(d, ""),
                          "url": "", "archived_at": f"{d} 20:00:00", "rows_technicals": str(n), "status": s}
                         for d, s, n in rows])


def _xlsx(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


class _Archive:
    """A fake Drive: the index workbook, and one 'workbook' per copy whose engine rows are its stock count."""
    def __init__(self, index: pd.DataFrame, counts=None):
        self.index, self.calls = _xlsx(index), []
        self.counts = counts or {r.spreadsheet_id: int(r.rows_technicals) for r in index.itertuples()}

    def get(self, url):
        self.calls.append(url)
        sid = url.split("/d/")[1].split("/")[0]
        return self.index if sid == "INDEX" + "y" * 39 else f"workbook {sid}".encode()

    def rows(self, workbook: bytes) -> dict:
        n = self.counts[workbook.decode().split()[1]]
        return {t: n for t in TABS}


_INDEX_ID = "INDEX" + "y" * 39


def test_kept_is_every_ok_or_daily_copy_in_the_range_oldest_first():
    idx = _index([("2026-10-14", "daily", 2704), ("2026-10-12", "daily", 2717), ("2026-10-15", "ok", 2704),
                  ("2026-10-13", "failed: content", 0), ("2026-10-09", "daily", 2719), ("2026-10-16", "ok", 2704)])
    k = A.kept(idx, "2026-10-12", "2026-10-15")
    assert k["vintage_date"].tolist() == ["2026-10-12", "2026-10-14", "2026-10-15"]
    assert k["rows_technicals"].tolist() == [2717, 2704, 2704]


def test_two_kept_copies_for_one_day_are_refused():
    idx = _index([("2026-10-12", "daily", 2717), ("2026-10-12", "ok", 2717)])
    with pytest.raises(A.ArchiveError, match="2026-10-12"):
        A.kept(idx, "2026-10-12", "2026-10-12")


def test_each_copy_is_written_under_the_name_the_backfill_reads(tmp_path):
    arc = _Archive(_index())
    lines = A.fetch(_INDEX_ID, tmp_path, "2026-10-12", "2026-10-15", get=arc.get, rows=arc.rows)
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"PRISM 2026-10-{d}.xlsx" for d in (12, 13, 14, 15)]
    assert (tmp_path / "PRISM 2026-10-12.xlsx").read_bytes() == f"workbook {_SID['2026-10-12']}".encode()
    assert [ln.split()[0] for ln in lines] == ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]
    assert all("fetched" in ln for ln in lines)


def test_a_file_that_exists_is_checked_and_never_overwritten(tmp_path):
    arc = _Archive(_index())
    A.fetch(_INDEX_ID, tmp_path, "2026-10-12", "2026-10-12", get=arc.get, rows=arc.rows)
    arc.calls.clear()
    lines = A.fetch(_INDEX_ID, tmp_path, "2026-10-12", "2026-10-12", get=arc.get, rows=arc.rows)
    assert arc.calls == [f"https://docs.google.com/spreadsheets/d/{_INDEX_ID}/export?format=xlsx"]   # the index only
    assert "kept" in lines[0]


def test_a_kept_file_that_disagrees_with_the_index_is_refused_never_replaced(tmp_path):
    (tmp_path / "PRISM 2026-10-12.xlsx").write_bytes(f"workbook {_SID['2026-10-13']}".encode())   # 2719, not 2717
    arc = _Archive(_index())
    with pytest.raises(A.ArchiveError, match="2026-10-12.*2717"):
        A.fetch(_INDEX_ID, tmp_path, "2026-10-12", "2026-10-12", get=arc.get, rows=arc.rows)
    assert (tmp_path / "PRISM 2026-10-12.xlsx").read_bytes() == f"workbook {_SID['2026-10-13']}".encode()


def test_a_copy_whose_tabs_disagree_with_the_index_is_refused_and_nothing_is_written(tmp_path):
    idx = _index()
    arc = _Archive(idx, counts={**{r.spreadsheet_id: int(r.rows_technicals) for r in idx.itertuples()},
                                _SID["2026-10-13"]: 2650})
    with pytest.raises(A.ArchiveError, match="2026-10-13"):
        A.fetch(_INDEX_ID, tmp_path, "2026-10-12", "2026-10-15", get=arc.get, rows=arc.rows)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["PRISM 2026-10-12.xlsx"]      # the day before it, whole


def test_a_copy_missing_a_tab_is_refused(tmp_path):
    arc = _Archive(_index())
    with pytest.raises(A.ArchiveError, match="tab"):
        A.fetch(_INDEX_ID, tmp_path, "2026-10-12", "2026-10-12", get=arc.get,
                rows=lambda wb: {t: 2717 for t in TABS if t != "income"})
    assert not list(tmp_path.iterdir())


def test_a_range_with_no_kept_copy_is_refused(tmp_path):
    arc = _Archive(_index())
    with pytest.raises(A.ArchiveError, match="no kept copy"):
        A.fetch(_INDEX_ID, tmp_path, "2026-11-01", "2026-11-30", get=arc.get, rows=arc.rows)


@pytest.mark.parametrize("start,end", [("2026-10-15", "2026-10-12"), ("2026-13-01", "2026-10-15"),
                                       ("12-10-2026", "2026-10-15")])
def test_a_range_that_is_not_two_dates_in_order_is_refused(tmp_path, start, end):
    with pytest.raises(A.ArchiveError, match="range"):
        A.fetch(_INDEX_ID, tmp_path, start, end, get=_Archive(_index()).get, rows=_Archive(_index()).rows)


def test_main_reads_the_secret_by_shape_never_its_value(tmp_path, monkeypatch, capsys):
    arc = _Archive(_index())
    monkeypatch.setattr(A, "_get", arc.get)
    monkeypatch.setattr(A, "_engine_rows", arc.rows)
    monkeypatch.setenv("ARCHIVE_INDEX_SHEET_ID", f"https://docs.google.com/spreadsheets/d/{_INDEX_ID}/edit")
    assert A.main(["--out", str(tmp_path), "--from", "2026-10-12", "--to", "2026-10-13"]) == 0
    out = capsys.readouterr().out
    assert _INDEX_ID not in out and not any(s in out for s in _SID.values()) and "docs.google" not in out
    assert "an address" in out and f"{len(_INDEX_ID)} chars" in out and "2026-10-13" in out


def test_main_without_the_secret_or_with_a_refusal_fails_loudly(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ARCHIVE_INDEX_SHEET_ID", raising=False)
    assert A.main(["--out", str(tmp_path), "--from", "2026-10-12", "--to", "2026-10-13"]) == 1
    assert "ARCHIVE_INDEX_SHEET_ID" in capsys.readouterr().out
    arc = _Archive(_index())
    monkeypatch.setattr(A, "_get", arc.get)
    monkeypatch.setattr(A, "_engine_rows", arc.rows)
    monkeypatch.setenv("ARCHIVE_INDEX_SHEET_ID", _INDEX_ID)
    assert A.main(["--out", str(tmp_path), "--from", "2026-12-01", "--to", "2026-12-31"]) == 1
    assert "no kept copy" in capsys.readouterr().out


@pytest.mark.skipif(not _ARCHIVE.exists(), reason="the archive copies are local only (gitignored)")
def test_the_engine_reads_a_real_archive_copy_whole():
    rows = A._engine_rows(_ARCHIVE.read_bytes())
    assert sorted(rows) == list(TABS) and len(set(rows.values())) == 1 and min(rows.values()) > 2000
