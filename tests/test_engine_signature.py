"""
test_engine_signature.py — every data cache misses when the engine's code changes (2026-10-10).

After a push the app kept serving numbers the OLD engine had derived until someone pressed Clear Cache
& Reload: app.get_clean_data was keyed on the data source alone, Movers' archived copy on its sheet id
alone, and Movers' re-score and the sidebar download on the git commit — which needs a git binary and a
.git folder, may read 'unknown' on a host that has neither, and moves on commits that change no engine
code. ui_export.engine_signature hashes config.py and core/*.py — the code that derives and scores — and
all four caches take it. engine_version (the commit) stays the provenance stamp inside exported files.

Run with: pytest tests/test_engine_signature.py -v
"""
import ast
import os
import re
import time

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
_TREE = ast.parse(_APP)


@pytest.fixture
def engine(tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "config.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "core" / "data_engine.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "core" / "notes.txt").write_text("not code\n", encoding="utf-8")
    return tmp_path


def _bump(path, text):
    path.write_text(text, encoding="utf-8")
    t = time.time() + 5
    os.utime(path, (t, t))


def test_the_signature_is_stable_while_nothing_changes(engine):
    from ui.ui_export import engine_signature
    s = engine_signature(engine)
    assert re.fullmatch(r"[0-9a-f]{12}", s) and engine_signature(engine) == s


def test_it_moves_when_the_engine_or_config_changes(engine):
    from ui.ui_export import engine_signature
    s0 = engine_signature(engine)
    _bump(engine / "core" / "data_engine.py", "def f():\n    return 2\n")
    s1 = engine_signature(engine)
    _bump(engine / "config.py", "A = 2\n")
    s2 = engine_signature(engine)
    assert len({s0, s1, s2}) == 3, "a change to core/ or config.py must change the signature"


def test_it_follows_content_not_file_times(engine):
    """A checkout that rewrites a file with the same bytes is the same engine."""
    from ui.ui_export import engine_signature
    s = engine_signature(engine)
    _bump(engine / "core" / "data_engine.py", "def f():\n    return 1\n")
    assert engine_signature(engine) == s
    _bump(engine / "core" / "notes.txt", "still not code\n")
    assert engine_signature(engine) == s, "only .py sources are the engine"


def test_it_needs_no_git_and_never_raises(tmp_path):
    from ui import ui_export as X
    fn = next(n for n in ast.parse(open(X.__file__, encoding="utf-8").read()).body
              if isinstance(n, ast.FunctionDef) and n.name == "engine_signature")
    assert "subprocess" not in ast.unparse(fn), "the cache identity must not depend on a git binary"
    assert X.engine_signature(tmp_path / "nowhere") == "unknown"


def test_the_live_signature_covers_the_real_engine():
    from ui.ui_export import engine_signature
    assert re.fullmatch(r"[0-9a-f]{12}", engine_signature())


# ── every data cache takes it ─────────────────────────────────────────────────────────────────────
def _fn(name):
    return next(n for n in _TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls(name):
    return [n for n in ast.walk(_TREE) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]


@pytest.mark.parametrize("fn", ["get_clean_data", "_load_vintage_clean", "_score_vintage"])
def test_each_cached_loader_is_keyed_on_the_engine(fn):
    """Streamlit drops underscored parameters from the key: the engine must be a plain one."""
    args = [a.arg for a in _fn(fn).args.args]
    assert "engine" in args, f"{fn} has no engine parameter — a deploy would serve the old engine's frame"


@pytest.mark.parametrize("fn", ["get_clean_data", "_load_vintage_clean", "_score_vintage"])
def test_every_call_passes_the_content_signature(fn):
    calls = _calls(fn)
    assert calls, f"{fn} is never called"
    pos = [a.arg for a in _fn(fn).args.args].index("engine")
    for c in calls:
        kw = [k.value for k in c.keywords if k.arg == "engine"]
        val = kw[0] if kw else (c.args[pos] if len(c.args) > pos else None)
        passed = ast.unparse(val) if val is not None else None
        assert passed == "engine_signature()", f"{fn} is keyed on {passed}, not the engine's own code"


def test_the_download_cache_key_takes_the_content_signature():
    i = _APP.index("_dl_sig = (")
    sig = _APP[i:_APP.index("st.download_button(", i)]
    assert "engine_signature()" in sig and "engine_version()" not in sig
