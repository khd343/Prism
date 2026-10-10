"""Contract for the sidebar full-universe CSV export (ui/ui_export.py). Tests the PURE core
(_to_csv_bytes) — Excel-safe BOM, every column preserved, NaN never leaks as a literal. Pins the
2026-06-19 export feature so the full-dump download can't silently regress (wrong encoding / dropped
columns / 'nan' text)."""
import io

import numpy as np
import pandas as pd

from ui.ui_export import _to_csv_bytes


def test_csv_round_trips_with_bom_and_emoji():
    df = pd.DataFrame({"name": ["HDFC", "Infosys"],
                       "tier_label": ["🏆 Crown Jewels", "❌ Not Ready"],
                       "composite_score": [88.5, 12.0]})
    raw = _to_csv_bytes(df)
    assert raw[:3] == b"\xef\xbb\xbf"                            # UTF-8 BOM (Excel-safe)
    back = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")    # round-trips, BOM stripped
    assert list(back.columns) == ["name", "tier_label", "composite_score"]
    assert len(back) == 2 and back["tier_label"].iloc[0] == "🏆 Crown Jewels"


def test_all_columns_preserved():
    df = pd.DataFrame({f"c{i}": [i] for i in range(60)})
    back = pd.read_csv(io.BytesIO(_to_csv_bytes(df)), encoding="utf-8-sig")
    assert list(back.columns) == [f"c{i}" for i in range(60)]    # nothing dropped


def test_missing_values_no_literal_nan_leak():
    raw = _to_csv_bytes(pd.DataFrame({"a": [1.0, np.nan], "b": ["x", None]}))
    text = raw.decode("utf-8-sig")
    assert "nan" not in text.lower() and "None" not in text      # missing → empty field, not literal
    back = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    assert back["a"].isna().iloc[1]                              # NaN round-trips as missing


# ═══════════════════════════════════════════════════════════════════════════
# CACHE-SIGNATURE CONTRACT (2026-08-29) — the wrong-file collision class
# ═══════════════════════════════════════════════════════════════════════════
# THE BUG THIS PINS AGAINST (found by live audit): the download cache was keyed on
# `score_key|len|composite_sum:.2f` — a LOSSY content hash. On live data, 768 of 2,117 stocks
# share a 2dp-rounded composite with another stock (355 tied values), and two real filter states
# (Capital Goods-Electrical + Crown Jewels → Shilchar; Petrochemicals + Crown Jewels → Kothari)
# produced the IDENTICAL signature (1, 90.00) — the second download silently served the first
# stock's CSV while the filename showed the right count. The key must identify the exact row set.

def test_universe_signature_distinguishes_equal_count_equal_sum_sets():
    """The Shilchar/Kothari class: same row count, same composite sum → the old signature
    collided. The digest must differ whenever the underlying stocks differ."""
    from ui.ui_export import universe_signature
    a = pd.Series(["Shilchar Technologies Ltd"])
    b = pd.Series(["Kothari Petrochemicals Ltd"])
    assert universe_signature(a) != universe_signature(b)


def test_universe_signature_is_deterministic_and_order_sensitive():
    """Same set, same order → same digest (cache hits still work). The frame's row order is
    df-order (boolean masks preserve it), and the CSV bytes depend on it — so order is part
    of content identity."""
    from ui.ui_export import universe_signature
    s = pd.Series(["A Ltd", "B Ltd", "C Ltd"])
    assert universe_signature(s) == universe_signature(s.copy())
    assert universe_signature(s) != universe_signature(s.iloc[::-1])


def test_app_download_sig_uses_the_exact_digest_not_the_lossy_sum():
    """Structural pin on the call site: app.py's _dl_sig must include universe_signature(...) and
    must NOT fall back to the 2dp composite-sum hash that collided."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    sig_line = next(l for l in src.splitlines() if "_dl_sig" in l and "=" in l)
    assert "universe_signature(" in sig_line, "download cache key lost its exact row-identity digest"
    assert "composite_score" not in sig_line, (
        "the lossy composite-sum component is back in the download cache key — it collides "
        "(768-stock 2dp tie pool measured 2026-08-29)")
    assert "_score_key" in sig_line, (
        "score_key must stay in the signature — the same stock set has different score columns "
        "under a different analysis mode / profile")


# ── The download IS the Cloud-side snapshot (2026-09-02) ─────────────────────────────────────
def test_stamp_snapshot_prefixes_the_five_provenance_columns():
    """ONE snapshot format for both destinations (the 📥 download in the browser, tools/snapshot.py
    on disk): vintage · source · engine · scored-day · scored-MODE, in front of the untouched frame.
    The vintage is the identity — a snapshot named by the click date mislabels the data it holds.

    scored_mode joined the format on 2026-09-19, the day the default Analysis Mode moved Hybrid ->
    Breakout: tools/validate.py backs framework boosts OUT of the stored composite to compare
    weightings, and that arithmetic needs the weights it was built from. This test grew from four
    columns to five rather than being replaced — the invariant (a fixed provenance prefix in front
    of an untouched frame) is the same one."""
    from ui.ui_export import stamp_snapshot
    df = pd.DataFrame({"name": ["A"], "composite_score": [50.0]})
    out = stamp_snapshot(df, "2026-08-28", "sheet")
    assert list(out.columns[:5]) == ["snapshot_vintage", "snapshot_source", "engine_version",
                                     "scored_at", "scored_mode"]
    assert out["snapshot_vintage"].iloc[0] == "2026-08-28"
    assert out["snapshot_source"].iloc[0] == "sheet"
    assert list(out.columns[5:]) == ["name", "composite_score"], "the frame itself must be untouched"
    assert list(df.columns) == ["name", "composite_score"], "stamping mutated the caller's frame"
    assert stamp_snapshot(df, None, "local")["snapshot_vintage"].iloc[0] == "unknown", (
        "an unknown vintage must say so — never today's date dressed up as the data's date")


def test_engine_version_is_a_short_hash_or_unknown():
    """A delta between two snapshots scored by DIFFERENT engines cannot tell 'the company moved'
    from 'we fixed a bug'. The stamp must be a git short hash, or an honest 'unknown' — never an
    exception, never an empty string."""
    import re
    from ui.ui_export import engine_version
    v = engine_version()
    assert re.fullmatch(r"[0-9a-f]{7,40}|unknown", v), v
    assert engine_version() == v, "the stamp must be stable within a process"


def test_download_is_named_by_the_data_vintage_and_stamped():
    """The sidebar download used today's date in its filename: a download of 28-Aug data taken on
    the 15th was named the 15th. It is named by the sheet's own vintage now, falls back to the
    click date ONLY when the sheet carries no date, passes vintage+source into the stamped cached
    export, and puts vintage + engine into the cache key so a refreshed sheet or a redeployed engine
    can never be served stale bytes."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    i = src.index("with _scored_dl_ph.container():")
    blk = src[i:src.index('mime="text/csv"', i)]
    assert 'file_name=f"prism_scored_{_fname_date}_{len(filt)}stocks.csv"' in blk, (
        "the download filename no longer carries the data vintage")
    assert "_date.today().isoformat()}_{len(filt)}" not in blk, "the click date names the file again"
    assert "_fresh.data_date.isoformat() if _fresh.is_known else None" in blk, (
        "the vintage must come from the sheet's own date (sheet_meta), unknown when it has none")
    assert "scored_universe_csv(_dl_sig, filt, _vintage, st.session_state.data_source)" in blk, (
        "the export is no longer stamped with vintage + source")
    sig = blk[blk.index("_dl_sig = ("):blk.index("st.download_button(")]
    # The engine enters as its CODE signature since 2026-10-10 (tests/test_engine_signature.py): the git
    # commit needs a git binary and may read 'unknown' on the host, which would never change the key.
    assert "_vintage" in sig and "engine_signature()" in sig, (
        "vintage and engine must be in the cache key — a stale CSV would otherwise survive a refresh")


def test_download_cache_is_bounded():
    """12.3 MB per full-universe entry, one entry per distinct filter state, container RAM is
    finite: the cache must declare max_entries so a long filter session can't grow unbounded."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "ui" / "ui_export.py").read_text(encoding="utf-8")
    deco = src[src.index("@st.cache_data"):src.index("def scored_universe_csv")]
    assert "max_entries" in deco, "scored_universe_csv lost its max_entries bound — unbounded 12MB entries"
