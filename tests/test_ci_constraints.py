"""
test_ci_constraints.py — every GitHub job that installs the engine installs the WHOLE dependency tree pinned
(.github/constraints.txt), not just the eight packages requirements.txt names.

Why (independent review, 24 Oct 2026): each run resolved streamlit's and plotly's dependencies afresh, so the jobs
ran on whatever was newest that day. Measured: python-dateutil 2.9.0.post0 until 21 Oct 2026, 3.0.0 from 22 Oct
(a major release, which pandas reads dates through), with certifi and tornado moving too -- under lists that one
frozen commit is supposed to make. Nothing changed in them (a back-fill on GitHub rebuilt 21-23 Oct byte for byte
with the new set), but one breaking release would turn every feed and v2-lists run red until May. The pinned set is
the one that rebuilt 21-23 Oct.

Run with: pytest tests/test_ci_constraints.py -v
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_CONSTRAINTS = ROOT / ".github" / "constraints.txt"
_WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([0-9][0-9A-Za-z.+!-]*)$")
# The jobs whose code checkout is a pinned commit (its own requirements.txt, no constraints file): main's .github/
# beside it.
_PINNED_CODE = {"playbook-feed.yml", "v2-lists.yml"}


def _lines(path):
    return [ln.split("#", 1)[0].strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.split("#", 1)[0].strip()]


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def test_the_constraints_pin_every_package_exactly_in_ascii():
    raw = _CONSTRAINTS.read_bytes()
    assert raw.isascii(), "pip reads this on a Linux box with its own locale"
    lines = _lines(_CONSTRAINTS)
    assert len(lines) >= 40, len(lines)
    loose = [ln for ln in lines if not _PIN.match(ln)]
    assert not loose, f"not pinned with ==: {loose}"
    names = [_norm(_PIN.match(ln).group(1)) for ln in lines]
    assert len(names) == len(set(names)), "a package pinned twice"


def test_the_constraints_hold_every_requirement_at_its_own_pin():
    """A requirement bumped without its constraint makes pip refuse the install: every job red."""
    pinned = {_norm(m.group(1)): m.group(2) for m in map(_PIN.match, _lines(_CONSTRAINTS))}
    for req in _lines(ROOT / "requirements.txt"):
        m = _PIN.match(req)
        assert m, req
        assert pinned.get(_norm(m.group(1))) == m.group(2), (req, pinned.get(_norm(m.group(1))))


def test_every_job_that_installs_the_engine_installs_it_with_the_constraints():
    installing = [wf for wf in _WORKFLOWS if "pip install" in wf.read_text(encoding="utf-8")]
    assert {wf.name for wf in installing} >= {"data-check.yml", "scan-records.yml", "playbook-feed.yml",
                                               "v2-lists.yml"}
    for wf in installing:
        y = wf.read_text(encoding="utf-8")
        installs = re.findall(r"run: (pip install .*)$", y, re.M)
        where = "main-ci/.github/constraints.txt" if wf.name in _PINNED_CODE else ".github/constraints.txt"
        assert installs == [f"pip install -r requirements.txt -c {where}"], (wf.name, installs)


def test_a_job_on_pinned_code_reads_mains_constraints_beside_it():
    """The code checkout is the frozen commit (no constraints file there); main's .github/ comes in a sparse checkout
    of its own, so moving a pin never moves the code."""
    for name in sorted(_PINNED_CODE):
        y = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert re.search(r"- uses: actions/checkout@v7\n\s+with:\n\s+path: main-ci\n\s+sparse-checkout: \.github\n", y), \
            name
        assert y.index("path: main-ci") < y.index("pip install"), name


def test_the_constraints_file_is_never_ignored():
    """.gitignore ignores every *.txt but requirements.txt: a constraints file it caught would never reach GitHub, and
    every install would fail on a missing -c file (found 24 Oct 2026, when ci/constraints.txt would not add).
    --no-index asks the rules themselves: git never calls a tracked file ignored, which would hide a lost exception."""
    import subprocess
    p = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", "--no-index",
                        _CONSTRAINTS.relative_to(ROOT).as_posix()])
    assert p.returncode == 1, "git's rules ignore the constraints file"
