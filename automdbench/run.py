"""Run one agent on one task, then grade it.

    python -m automdbench.run TASK --agent "bash agents/codex/run.sh agents/codex/prompt.txt" \
        --run-dir runs/codex/support_1 [--budget-min 720] [--inject agents/codex/autoresearch]

The agent starts in a fresh copy of the task's public files, with TASK and BUDGET_MIN (minutes) set.
Every new version of ./submission.csv is archived while it runs. When the budget ends the agent's
process group receives SIGTERM, and SIGKILL 60 s later. For AIDE and AI Scientist v2 the agent's own
best node is then selected (select.py); the result is graded with the hidden labels.
"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from automdbench.grade import grade
from automdbench.select import select

POLL_S, GRACE_S = 15, 60


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Archive:
    """Keeps every distinct version of submission.csv, named and stamped by its write time."""

    def __init__(self, sub, folder):
        self.sub, self.folder, self.records, self.last = sub, folder, [], None

    def poll(self, t0):
        try:
            mtime, sha = self.sub.stat().st_mtime, _sha(self.sub)
        except OSError:
            return
        if self.last and sha == self.last["sha"]:
            if abs(mtime - self.last["mtime"]) > 0.5:  # same content written again
                self.last["mtime"] = mtime
                self.records.append({"file": self.last["file"], "mtime": round(mtime, 3),
                                     "elapsed_s": round(time.time() - t0)})
            return
        name = f"submission_{time.strftime('%Y%m%d-%H%M%S', time.localtime(mtime))}_{len(self.records):03d}.csv"
        try:
            shutil.copy2(self.sub, self.folder / name)
        except OSError:
            return
        self.last = {"sha": sha, "mtime": mtime, "file": name}
        self.records.append({"file": name, "mtime": round(mtime, 3), "elapsed_s": round(time.time() - t0)})


def _stop(proc):
    if os.name != "posix":
        proc.terminate()
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=GRACE_S)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait()


def run(task, agent, run_dir, budget_min=720, inject=None, packages="packages"):
    run_dir = Path(run_dir).resolve()
    sandbox, archive = run_dir / "sandbox", run_dir / "archive"
    run_dir.mkdir(parents=True)
    archive.mkdir()
    shutil.copytree(Path(packages) / task / "public", sandbox)
    for f in sorted(Path(inject).iterdir()) if inject else []:
        if f.is_file():
            shutil.copy2(f, sandbox / f.name)

    env = {**os.environ, "TASK": task, "BUDGET_MIN": f"{budget_min:g}"}
    t0, timed_out = time.time(), False
    with open(run_dir / "agent.log", "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(agent, shell=True, cwd=sandbox, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=os.name == "posix")
        arc = Archive(sandbox / "submission.csv", archive)
        while True:
            try:
                proc.wait(timeout=POLL_S)
                break
            except subprocess.TimeoutExpired:
                arc.poll(t0)
            if time.time() - t0 > budget_min * 60:
                timed_out = True
                _stop(proc)
                break
        arc.poll(t0)

    meta = {"task": task, "agent": agent, "budget_min": budget_min, "elapsed_s": round(time.time() - t0, 1),
            "exit_code": proc.returncode, "timed_out": timed_out, "archives": arc.records,
            "selection": select(sandbox, arc.records)}
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    report = grade(task, sandbox / "submission.csv", packages)
    (run_dir / "grade.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--agent", required=True, help="shell command, started inside the sandbox")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--budget-min", type=float, default=720)
    ap.add_argument("--inject", help="directory whose files are copied into the sandbox before the start")
    ap.add_argument("--packages", default="packages")
    a = ap.parse_args()
    report = run(a.task, a.agent, a.run_dir, a.budget_min, a.inject, Path(a.packages).resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
