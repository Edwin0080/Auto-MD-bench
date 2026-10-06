"""Choose the submission to grade for AIDE and AI Scientist v2, using only the agent's own records.

AIDE: rank the nodes not marked buggy by the agent's validation metric and take the highest one that
wrote a submission file. A node's file is the archived version whose write time falls within the
node's execution.

AI Scientist v2: walk the sub-stages from last to first. In each, take the node named in
best_node_id.txt if it is not buggy, otherwise the best non-buggy node under the agent's metric
rule. Grade the first such node that saved its own file in submissions_by_node/.

Codex manages ./submission.csv itself, so nothing is selected for it.
"""
import json
import re
import shutil
from pathlib import Path


def _nodes(journal):
    try:
        return json.loads(Path(journal).read_text(encoding="utf-8", errors="replace")).get("nodes") or []
    except (OSError, json.JSONDecodeError):
        return []


def _place(src, sandbox):
    tmp = sandbox / "submission.csv.selecting"
    shutil.copyfile(src, tmp)
    tmp.replace(sandbox / "submission.csv")


def _aide(sandbox, journals, archives):
    def key(n):
        m = n.get("metric") or {}
        if n.get("is_buggy") or not isinstance(m.get("value"), (int, float)) or m.get("maximize") is None:
            return None
        return m["value"] if m["maximize"] else -m["value"]

    ranked = sorted((n for j in journals for n in _nodes(j) if key(n) is not None), key=key, reverse=True)
    for n in ranked:
        if not isinstance(n.get("ctime"), (int, float)):
            continue
        end = n["ctime"] + (n.get("exec_time") or 0.0)
        hits = [a for a in archives if n["ctime"] - 5 <= a["mtime"] <= end + 5]
        if hits:
            src = Path(sandbox).parent / "archive" / max(hits, key=lambda a: a["mtime"])["file"]
            _place(src, sandbox)
            return {"agent": "aide", "node_id": n.get("id"), "step": n.get("step"), "file": src.name}
    return {"agent": "aide", "node_id": None, "reason": "no ranked node has a submission file"}


def _ais_value(n):
    v = (n.get("metric") or {}).get("value")
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, dict):
        return None
    if "metric_names" in v:
        xs = [d["final_value"] for m in v["metric_names"] or [] for d in m.get("data") or []
              if isinstance(d.get("final_value"), (int, float))]
    else:
        xs = [x for x in v.values() if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def _ais_maximize(n):
    m = n.get("metric") or {}
    v = m.get("value")
    if isinstance(v, dict) and v.get("metric_names"):
        try:
            return not v["metric_names"][0]["lower_is_better"]
        except (KeyError, IndexError, TypeError):
            pass
    return bool(m.get("maximize"))


def _ais_good(n):
    return not n.get("is_buggy") and not n.get("is_buggy_plots")


def _ais_best(journal):
    """AI Scientist v2's best node of one sub-stage."""
    nodes = _nodes(journal)
    try:
        named = (Path(journal).parent / "best_node_id.txt").read_text(encoding="utf-8").strip()
    except OSError:
        named = None
    n = next((x for x in nodes if x.get("id") == named), None)
    if n is not None and _ais_good(n):
        return n
    best = None
    for n in (x for x in nodes if _ais_good(x) and _ais_value(x) is not None):
        a, b = _ais_value(n), _ais_value(best) if best else None
        if best is None or (a != b and ((a > b) if _ais_maximize(n) else (a < b))):
            best = n
    return best


def _ais(sandbox, journals):
    def order(j):
        m = re.match(r"stage_(\d+)_[a-z_]+?_(\d+)_", Path(j).parent.name)
        return (int(m.group(1)), int(m.group(2))) if m else (99, 99)

    substages = {Path(j).parent.name: j for j in journals}
    for name, j in reversed(sorted(substages.items(), key=lambda kv: order(kv[1]))):
        n = _ais_best(j)
        if n is None:
            continue
        files = sorted(Path(sandbox).glob(f"experiments/*/submissions_by_node/{n.get('id')}.csv"))
        if files:
            _place(files[0], sandbox)
            return {"agent": "ais2", "substage": name, "node_id": n.get("id"), "file": files[0].name}
    return {"agent": "ais2", "node_id": None, "reason": "no sub-stage best has a submission file"}


def select(sandbox, archives):
    """Replace sandbox/submission.csv with the agent's chosen file. Returns a record, or None for Codex."""
    sandbox = Path(sandbox)
    aide = sorted(sandbox.glob(".aide/logs/*/journal.json"))
    if aide:
        return _aide(sandbox, aide, archives)
    ais = sorted(sandbox.glob("experiments/*/logs/*/*/journal.json"))
    if ais:
        return _ais(sandbox, ais)
    return None
