"""
test_fresh_modules.py — after a deploy, one run imports ONE version of PRISM's code (2026-10-10).

Streamlit Cloud re-reads app.py on every run but keeps the first-party modules it imported before a
push, so a NEW app.py can meet an OLD module. The live app went down twice that way, each time until a
reboot: 2026-10-09 (AttributeError in ui_scans._logic — the old module against the new file) and
2026-10-10 (ImportError: cannot import name 'step_label' from 'ui.ui_scans'). app.py now drops every
config/core/ui module when any of their files changed since this process imported them. These tests
run app.py's own guard (lifted out of the file with ast) on a throwaway package.

Run with: pytest tests/test_fresh_modules.py -v
"""
import ast
import importlib
import os
import sys
import textwrap

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
_TREE = ast.parse(_APP)
_GUARD = ("_first_party", "_file_times", "_drop_stale_first_party")


def _guard_ns(roots):
    """app.py's guard functions, executed as written, with the first-party roots pointed at `roots`."""
    nodes = [n for n in _TREE.body if isinstance(n, ast.FunctionDef) and n.name in _GUARD]
    assert sorted(n.name for n in nodes) == sorted(_GUARD), "app.py lost a piece of the fresh-modules guard"
    ns = {"os": os, "_FIRST_PARTY": roots}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), ns)
    return ns


@pytest.fixture
def pkg(tmp_path):
    """A throwaway package `zzfresh`: a.py defines X; b.py imports it — like app.py importing ui_scans."""
    root = tmp_path / "zzfresh"
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "a.py").write_text("X = 1\n", encoding="utf-8")
    (root / "b.py").write_text("from zzfresh.a import X\n", encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    importlib.import_module("zzfresh.b")
    yield root
    sys.path.remove(str(tmp_path))
    for n in [n for n in sys.modules if n.split(".")[0] == "zzfresh"]:
        del sys.modules[n]


def _push(root):
    """A deploy: a.py gains a name, and its file time moves on."""
    path = root / "a.py"
    path.write_text("X = 1\nstep_label = 'new'\n", encoding="utf-8")
    t = os.path.getmtime(path) + 10
    os.utime(path, (t, t))
    importlib.invalidate_caches()


def test_without_the_guard_a_new_name_meets_the_old_module(pkg):
    """The 2026-10-10 crash, reproduced: the file has the name, the cached module does not."""
    _push(pkg)
    with pytest.raises(ImportError):
        from zzfresh.a import step_label  # noqa: F401


def test_the_guard_drops_every_first_party_module_when_one_changed(pkg):
    g = _guard_ns(("zzfresh",))
    seen = g["_file_times"](sys.modules, g["_first_party"](sys.modules))
    _push(pkg)
    assert g["_drop_stale_first_party"](sys.modules, seen) == ["zzfresh.a"]
    assert not [n for n in sys.modules if n.split(".")[0] == "zzfresh"], (
        "ALL of them must go: b still holds a's old objects if only a is dropped")
    assert importlib.import_module("zzfresh.a").step_label == "new"
    assert seen == {}, "the memory restarts, to be refilled after this run's imports"


def test_the_guard_leaves_everything_alone_when_nothing_changed(pkg):
    g = _guard_ns(("zzfresh",))
    seen = g["_file_times"](sys.modules, g["_first_party"](sys.modules))
    before = {n: sys.modules[n] for n in g["_first_party"](sys.modules)}
    assert g["_drop_stale_first_party"](sys.modules, seen) == []
    assert all(sys.modules[n] is m for n, m in before.items())


def test_a_module_never_recorded_is_not_called_stale(pkg):
    """First run of a process: nothing recorded yet, nothing dropped."""
    g = _guard_ns(("zzfresh",))
    _push(pkg)
    assert g["_drop_stale_first_party"](sys.modules, {}) == []
    assert "zzfresh.a" in sys.modules


def test_app_runs_the_guard_before_its_first_party_imports_and_records_after_them():
    """The guard only works at the very top: before any config/core/ui import, and the times are
    recorded after the last one, so the next run can tell what changed."""
    def first_party(node):
        mod = node.module if isinstance(node, ast.ImportFrom) else None
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else [mod or ""]
        return any((n or "").split(".")[0] in ("config", "core", "ui") for n in names)

    body = _TREE.body
    imports = [i for i, n in enumerate(body) if isinstance(n, (ast.Import, ast.ImportFrom)) and first_party(n)]
    # The guard call sits in a `with _first_party_lock():` block since 2026-10-16 (the test below pins the lock).
    calls = [i for i, n in enumerate(body) if isinstance(n, (ast.Expr, ast.With))
             and "_drop_stale_first_party(" in ast.unparse(n)]
    records = [i for i, n in enumerate(body) if "_first_party_file_times().update(" in ast.unparse(n)]
    assert len(calls) == 1 and calls[0] < imports[0], "the guard must run before the first first-party import"
    assert len(records) == 1 and records[0] > imports[-1], "the times must be recorded after the last one"
    assert '_FIRST_PARTY = ("config", "core", "ui")' in _APP


def test_a_module_mid_import_and_its_package_are_never_dropped(pkg):
    """PRODUCTION CRASH, 2026-10-16 ("KeyError: 'config'", right after a deploy): two sessions ran this guard and
    their imports at once, one dropped the first-party modules while the other was still importing config, and
    Python's import, finding its half-built module gone from sys.modules, raised KeyError. A module mid-import is
    being built from the file as it is NOW, so it is not stale; neither is the package it is importing into, which
    the import reads back when it finishes. Everything else still goes."""
    import threading
    import types
    gate = types.ModuleType("zzfresh_gate")
    gate.entered, gate.release = threading.Event(), threading.Event()
    sys.modules["zzfresh_gate"] = gate
    (pkg / "c.py").write_text("import zzfresh_gate\nzzfresh_gate.entered.set()\nzzfresh_gate.release.wait(10)\n"
                              "Y = 2\n", encoding="utf-8")
    importlib.invalidate_caches()
    g = _guard_ns(("zzfresh",))
    seen = g["_file_times"](sys.modules, g["_first_party"](sys.modules))
    _push(pkg)
    errors = []

    def importer():
        try:
            importlib.import_module("zzfresh.c")
        except BaseException as e:      # the import's own failure is the evidence
            errors.append(e)

    t = threading.Thread(target=importer, daemon=True)
    try:
        t.start()
        assert gate.entered.wait(10), "the importer never reached its module body"
        assert g["_drop_stale_first_party"](sys.modules, seen) == ["zzfresh.a"]
        kept = sorted(n for n in sys.modules if n.split(".")[0] == "zzfresh")
    finally:
        gate.release.set()
        t.join(10)
        sys.modules.pop("zzfresh_gate", None)
    assert not errors, errors
    assert kept == ["zzfresh", "zzfresh.c"], "only the module mid-import and the package it imports into stay"
    assert sys.modules["zzfresh.c"].Y == 2


def test_two_sessions_never_drop_at_the_same_moment():
    """The guard runs under ONE lock for the whole process (a cache_resource), so two sessions a deploy reruns
    together cannot both find the old record and drop modules under each other's re-import (2026-10-16)."""
    withs = [n for n in _TREE.body if isinstance(n, ast.With) and "_drop_stale_first_party(" in ast.unparse(n)]
    assert len(withs) == 1 and [ast.unparse(i.context_expr) for i in withs[0].items] == ["_first_party_lock()"]
    lock = [n for n in _TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_first_party_lock"]
    assert len(lock) == 1 and any("cache_resource" in ast.unparse(d) for d in lock[0].decorator_list)
    assert ast.unparse(lock[0].body[-1]) == "return threading.Lock()"
