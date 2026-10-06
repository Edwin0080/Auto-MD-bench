"""Recompute the paper's result tables from runs.csv and reference.csv.

    python results/tables.py
"""
from pathlib import Path
from statistics import mean, stdev

import pandas as pd

from automdbench.tasks import get, normalize

HERE = Path(__file__).resolve().parent
AGENTS = ["AIDE", "Codex", "Autoresearch", "AI Scientist v2"]
CLASSES = {
    "Survival analysis": ["support", "metabric", "rotterdam", "rotterdam_gbsg", "seer", "seer_later_period", "nhanes"],
    "Classification": ["uci_diabetes", "physionet2012", "mimiciv", "eicu", "eicu_new_hospitals"],
    "Sequential prediction": ["sepsis", "sepsis_new_hospital", "sepsis_500_labels"],
}
TASKS = sum(CLASSES.values(), [])


def pm(xs, d=3):
    return f"{mean(xs):.{d}f} ± {stdev(xs):.{d}f}" if len(xs) > 1 else f"{mean(xs):.{d}f}"


def ranks(scores):
    """{agent: score} -> {agent: rank}; 1 = best, ties share the average rank."""
    s = pd.Series(scores)
    return s.rank(ascending=False, method="average").to_dict()


def main_table(runs, ref):
    main = runs[(runs.llm == "Qwen3.8 Flash") & runs.task.isin(TASKS)]
    by_task = {a: main[main.agent == a].groupby("task") for a in AGENTS}
    pen = {t: ranks({a: g.get_group(t).normalized.mean() for a, g in by_task.items()}) for t in TASKS}
    non = {t: ranks({a: g.get_group(t)[g.get_group(t).valid].normalized.mean() for a, g in by_task.items()}) for t in TASKS}
    ref_norm = mean(normalize(get(t).metric, ref.loc[t, "primary"]) for t in TASKS)
    print(f"Table: primary task performance (15 tasks)\n  {'Simple baseline models':22s} normalized {ref_norm:.3f}")
    for a in AGENTS:
        r = main[main.agent == a]
        per_rep = r.groupby("repetition")
        penalized = per_rep.normalized.mean().tolist()
        valid = (100 * per_rep.valid.mean()).tolist()
        if r.valid.all():
            nonpen = pm(per_rep.normalized.mean().tolist())
        else:  # average each task over its valid runs, then the tasks
            nonpen = f"{r[r.valid].groupby('task').normalized.mean().mean():.3f}"
        rp, rn = [pen[t][a] for t in TASKS], [non[t][a] for t in TASKS]
        print(f"  {a:22s} penalized {pm(penalized)} | non-penalized {nonpen} | "
              f"mean rank {pm(rp, 2)} / {pm(rn, 2)} | valid runs {pm(valid, 1)}%")


def ece_table(runs, ref):
    main = runs[(runs.llm == "Qwen3.8 Flash") & runs.task.isin(TASKS) & runs.valid]
    print("\nTable: ECE by task class (mean over tasks of the per-task mean over valid runs)")
    print("  " + " | ".join(["All"] + list(CLASSES)))
    for a in AGENTS + ["Simple baseline models"]:
        if a in AGENTS:
            per_task = main[main.agent == a].groupby("task").ece.mean()
        else:
            per_task = ref.ece
        cells = [mean(per_task[t] for t in TASKS)] + [mean(per_task[t] for t in ts) for ts in CLASSES.values()]
        print(f"  {a:22s} " + " | ".join(f"{c:.3f}" for c in cells))


def task_table(runs, ref, col, title):
    main = runs[(runs.llm == "Qwen3.8 Flash") & runs.task.isin(TASKS)]
    print(f"\nTable: {title}")
    for t in TASKS:
        cells = []
        for a in AGENTS:
            r = main[(main.agent == a) & (main.task == t)]
            xs = (r if col == "normalized" else r[r.valid])[col].tolist()
            cells.append(pm(xs) + ("" if len(xs) == 3 else f" (n={len(xs)})"))
        print(f"  {t:20s} ref {ref.loc[t, 'ece' if col == 'ece' else 'primary']:.3f} | " + " | ".join(cells))


def llm_table(runs):
    print("\nTable: Codex with three LLMs")
    codex = runs[(runs.agent == "Codex") & runs.task.isin(TASKS)]
    for llm, r in codex.groupby("llm", sort=False):
        score = r.groupby("repetition").normalized.mean().tolist()
        ece = r[r.valid].groupby("task").ece.mean().mean()
        print(f"  {llm:18s} normalized {pm(score)} | ECE {ece:.3f}")


def calibration_table(runs):
    print("\nTable: calibration control on UCI Diabetes (valid runs of the four agents)")
    for task, name in (("uci_diabetes", "AUROC"), ("uci_diabetes_brier", "Brier score"),
                       ("uci_diabetes_ece", "AUROC and ECE")):
        r = runs[(runs.llm == "Qwen3.8 Flash") & (runs.task == task) & runs.valid]
        print(f"  {name:14s} AUROC {pm(r.primary.tolist())} | ECE {pm(r.ece.tolist())}")


if __name__ == "__main__":
    runs = pd.read_csv(HERE / "runs.csv")
    ref = pd.read_csv(HERE / "reference.csv").set_index("task")
    main_table(runs, ref)
    ece_table(runs, ref)
    task_table(runs, ref, "normalized", "normalized score by task (invalid runs score 0)")
    task_table(runs, ref, "ece", "ECE by task, valid runs")
    llm_table(runs)
    calibration_table(runs)
