"""Full-universe CSV export for the sidebar download button. The pure core (_to_csv_bytes) and the
cache-key digest (universe_signature) are unit-tested; the cached entry (scored_universe_csv) is
what app.py calls."""
import hashlib
import subprocess
from datetime import date as _date
from functools import lru_cache
from pathlib import Path

import pandas as pd
import streamlit as st

from config import DEFAULT_ANALYSIS_MODE

_REPO_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def engine_version() -> str:
    """Short git commit hash of the engine that produced this frame — or 'unknown'.

    WHY (2026-09-02): a delta between two scored snapshots mixes two things — the DATA changed
    and the ENGINE changed — and between June and September the engine changed enormously (the
    wealth-tier reserves basis, the EP sign guard, dozens of gate corrections). A file that does
    not say which engine scored it cannot tell "the company moved" from "we fixed a bug"; the
    cross-year-basis rule in different clothes. Stamped once per process; never raises."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=3, cwd=str(_REPO_ROOT))
        h = out.stdout.strip()
        return h if out.returncode == 0 and h else "unknown"
    except Exception:
        return "unknown"


def engine_signature(root: Path = _REPO_ROOT) -> str:
    """WHICH CODE IS RUNNING, for cache keys: a hash of config.py and every core/*.py — or 'unknown'.

    WHY (2026-10-10): after a push the app kept serving numbers the OLD engine had derived until someone
    pressed Clear Cache & Reload — the data caches were keyed on the data alone (the new Liquidity
    column waited on exactly that). engine_version above is the wrong key: it needs a git binary and a
    .git folder, may read 'unknown' where neither exists, and moves on commits that change no engine
    code. This hash needs neither and moves exactly when the code that derives and scores does. It is
    redone only when a file's time or size moves (~9 ms a time; the app reruns on every click)."""
    files = [root / "config.py", *sorted((root / "core").glob("*.py"))]
    try:
        stats = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
        return _hash_sources(stats)
    except OSError:
        return "unknown"


@lru_cache(maxsize=8)
def _hash_sources(stats) -> str:
    """The content hash behind engine_signature, cached on the files' times and sizes."""
    h = hashlib.sha1()
    for path, _mtime, _size in stats:
        h.update(Path(path).name.encode())
        h.update(Path(path).read_bytes())
    return h.hexdigest()[:12]


def stamp_snapshot(df: pd.DataFrame, vintage, source, analysis_mode=None) -> pd.DataFrame:
    """THE ONE snapshot format — five provenance columns in front of the untouched full frame:

        snapshot_vintage  the DATA's own date (the sheet's name / the dated CSV drop), or 'unknown'
        snapshot_source   'sheet' | 'upload' | 'local' — the two sources are NOT the same data
        engine_version    which engine scored it (see engine_version)
        scored_at         the day the scoring ran — distinct from the vintage on purpose
        scored_mode       the Analysis Mode whose weights produced composite_score (2026-09-19)

    Used by the sidebar 📥 download (the Cloud-side snapshot) AND tools/snapshot.py (the local
    one), so the archive in Drive and the archive on disk are the same file. The vintage — not
    today's date — is the identity of a snapshot: the data only changes when the sheet is
    refreshed, so two snapshots of one vintage under one engine are identical and a download
    named by the day it was clicked mislabels it. Pure — never mutates `df`.

    scored_mode ADDED 2026-09-19, the day the default moved Hybrid -> Breakout. Without it a
    snapshot is an orphan: tools/validate.py has to BACK OUT the framework boosts to compare
    weightings, and that arithmetic needs the weights the composite was actually built from —
    guessing "Hybrid" would silently mis-reconstruct every snapshot taken after the switch.
    Read from df.attrs["analysis_mode"] (set by run_full_scoring) unless passed explicitly;
    falls back to the CURRENT default rather than a literal, so it cannot rot. Snapshots taken
    BEFORE this column existed carry no scored_mode and validate.py reads them as Hybrid, which
    is what they were."""
    snap = df.copy()
    _mode = analysis_mode or df.attrs.get("analysis_mode") or DEFAULT_ANALYSIS_MODE
    snap.insert(0, "scored_mode", str(_mode))
    snap.insert(0, "scored_at", _date.today().isoformat())
    snap.insert(0, "engine_version", engine_version())
    snap.insert(0, "snapshot_source", str(source))
    snap.insert(0, "snapshot_vintage", str(vintage) if vintage else "unknown")
    return snap


def _to_csv_bytes(df) -> bytes:
    """Full frame → UTF-8-with-BOM CSV bytes. The BOM (utf-8-sig) makes Excel render the emoji/unicode
    column labels and Indian stock names correctly instead of mojibake. NaN → empty field (to_csv
    default na_rep=''), so no literal 'nan'/'None' leaks. Pure — no st.*; unit-tested."""
    return df.to_csv(index=False).encode("utf-8-sig")


def universe_signature(names) -> str:
    """Exact, order-sensitive identity digest of a filtered frame's row set — md5 over the joined
    stock names (unique + never-NaN in this universe; ~2 ms on 2,117 rows).

    WHY EXACT (2026-08-29 audit): the previous cache key hashed content as
    `len|composite_sum:.2f` — lossy. On live data 768 of 2,117 stocks share a 2dp-rounded
    composite with another stock, and two real one-stock filter states (Shilchar vs Kothari, both
    90.00) produced the identical key, so the second download silently served the FIRST stock's
    CSV. Different row sets must never share a cache entry; identical ones still hit it."""
    return hashlib.md5("|".join(names.astype(str)).encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False, max_entries=8)
def scored_universe_csv(score_key: str, _df, vintage, source) -> bytes:
    """Cached on `score_key` — `<scoring key>|<universe_signature digest>|<vintage>|<engine>|<day>`
    from app.py, which identifies the exact (data, mode, profile, row set, provenance): to_csv
    runs ONCE per distinct filter state, not on every sidebar rerun. `_df`'s leading underscore
    tells st.cache_data NOT to hash the 2117×~700 frame each call (Streamlit skips underscore
    args); `vintage` and `source` are plain strings and DO enter the key, so a new data vintage
    can never be served last vintage's bytes. max_entries=8 caps the cache: each full-universe
    entry is ~12.3 MB (measured), and without a bound every filter state ever visited stays
    resident for the life of the Cloud container. The bytes are the stamped snapshot format —
    this download IS the Cloud-side snapshot."""
    return _to_csv_bytes(stamp_snapshot(_df, vintage, source))
