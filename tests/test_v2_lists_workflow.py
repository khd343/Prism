"""
test_v2_lists_workflow.py — the workflow that publishes Kite v2's lists (.github/workflows/v2-lists.yml): when it
runs, what it may write, how values reach the shell, and the ONE commit every list is made by (Kite-v2 design,
addendum A5).

Run with: pytest tests/test_v2_lists_workflow.py -v
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui import v2_lists_job as V  # noqa: E402

_WF = ROOT / ".github" / "workflows" / "v2-lists.yml"


def _wf():
    return _WF.read_text(encoding="utf-8")


def _scripts(y):
    """Every `run:` script of the workflow: a one-line value, or the lines of a `run: |` block."""
    lines, out = y.splitlines(), []
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)run:\s*(.*)$", line)
        if not m:
            continue
        if m.group(2) != "|":
            out.append(m.group(2))
            continue
        block = []
        for nxt in lines[i + 1:]:
            if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= len(m.group(1)):
                break
            block.append(nxt.strip())
        out.append("\n".join(block))
    return out


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, encoding="utf-8")


def test_the_workflow_runs_after_every_feed_run_with_two_backstops():
    y = _wf()
    assert re.search(r'workflow_run:\s*\n\s+workflows: \["Playbook feed"\]\s*\n\s+types: \[completed\]', y)
    assert re.findall(r'cron:\s*"([^"]+)"', y) == ["45 14 * * 1-5", "40 21 * * *"]
    assert re.search(r"workflow_dispatch:\s*\n\s+inputs:\s*\n\s+backfill_from:", y) and "backfill_to:" in y


def test_each_trigger_reaches_the_job_as_its_own_word():
    y = _wf()
    assert ("V2_TRIGGER: ${{ github.event_name == 'workflow_run' && 'after-feed' || github.event_name == 'schedule' "
            "&& 'schedule' || 'manual' }}") in y
    assert V.TRIGGERS == ("after-feed", "schedule", "manual")


def test_the_workflow_writes_only_the_v2_lists_branch():
    y = _wf()
    assert "permissions:" in y and "contents: write" in y
    assert "ref: v2-lists" in y and "path: lists" in y
    assert re.findall(r"git push\s+(.+)", y) == ["origin HEAD:v2-lists"]
    assert "git add .gitattributes README.md sessions rank" in y, "the no-conversion rule travels with the files"
    assert "if: steps.publish.outputs.session != ''" in y, "a run that wrote nothing commits nothing"


def test_the_workflow_runs_the_job_serially_on_a_pinned_runner():
    y = _wf()
    assert re.search(r"concurrency:\s*\n\s+group: v2-lists\s*\n\s+cancel-in-progress: false", y)
    assert re.findall(r"runs-on:\s*(\S+)", y) == ["ubuntu-24.04"]
    assert 'python-version: "3.13"' in y and "pip install -r requirements.txt" in y
    assert sorted(set(re.findall(r"secrets\.(\w+)", y))) == ["ARCHIVE_INDEX_SHEET_ID", "SHEET_ID"]


def test_live_and_backfill_run_the_job():
    y = _wf()
    scripts = "\n".join(_scripts(y))
    assert "if: inputs.backfill_from != ''" in y, "the archive is fetched only for a back-fill"
    assert 'python -m ui.archive_copies --out copies --from "$FROM" --to "$TO"' in scripts
    assert 'python -m ui.v2_lists_job --lists lists --backfill "$BACKFILL"' in scripts
    assert re.search(r"^python -m ui\.v2_lists_job --lists lists$", scripts, re.M)


def test_every_value_reaches_the_shell_through_the_environment():
    """Never `${{ }}`-interpolated into a script: an input or an output is data, not code."""
    y = _wf()
    scripts = _scripts(y)
    assert len(scripts) == 4 and not any("${{" in s for s in scripts), scripts
    assert 'git commit -m "$MSG"' in y and "MSG: ${{ steps.publish.outputs.message }}" in y
    assert "FROM: ${{ inputs.backfill_from }}" in y and "TO: ${{ inputs.backfill_to }}" in y


def test_every_list_is_made_by_one_named_commit():
    """A5: the code checkout is ONE named commit (V2_CODE_SHA), which every manifest names. It is in this history and
    it is the job's own code: moving either is a deliberate edit of the workflow (a new freeze)."""
    y = _wf()
    pins = re.findall(r"^\s+V2_CODE_SHA:\s*(\S+)\s*$", y, re.M)
    assert len(pins) == 1 and re.fullmatch(r"[0-9a-f]{40}", pins[0]), pins
    assert re.search(r"- uses: actions/checkout@v7\n\s+with:\n\s+ref: \$\{\{ env\.V2_CODE_SHA \}\}\n", y)
    if _git("rev-parse", "--is-shallow-repository").stdout.strip() != "false":
        pytest.skip("a shallow clone cannot check the pin against history")
    assert _git("merge-base", "--is-ancestor", pins[0], "HEAD").returncode == 0, "the pin is not in this history"
    assert _git("diff", "--quiet", pins[0], "HEAD", "--", "ui/v2_lists_job.py").returncode == 0, \
        "the job's code moved after the pin: move V2_CODE_SHA with it, or the lists would not be the code you read"
