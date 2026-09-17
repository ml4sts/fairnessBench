"""Loaders for the cleaned FairnessBench result CSVs.

`explode_results.py` writes one timestamped CSV per kind into ``path.CSV_FILES``;
these loaders pick up the newest one, so nothing needs editing after a
regeneration. Pin an older file with ``constants.PERFORMANCE_CSV`` /
``constants.BASELINE_CSV``, or pass ``csv=`` for a one-off.
"""

import pandas as pd

from path import CSV_FILES
from constants import (
    PERFORMANCE_CSV,
    BASELINE_CSV,
    DOLLARSTREET_CSV,
    PERFORMANCE_PREFIX,
    BASELINE_PREFIX,
    DOLLARSTREET_PREFIX,
    RUN_COUNTS_CSV,
    RUN_COUNTS_PREFIX,
)
import cleaning


def latest_csv(prefix):
    """Name of the newest CSV in CSV_FILES starting with ``prefix``.

    Filenames embed an ISO timestamp, so lexicographic max == newest.
    """
    matches = sorted(p.name for p in CSV_FILES.glob(prefix + "*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"no {prefix}*.csv in {CSV_FILES} — run `python explode_results.py` first"
        )
    return matches[-1]


def _resolve(csv, pinned, prefix):
    path = CSV_FILES / (csv or pinned or latest_csv(prefix))
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_performance(csv=None, clean=True):
    """Agent performance + flake8 scores, one row per (model, task, run).

    With ``clean=True`` the task is split into dataset / task_metric /
    research_problem / dem and models are renamed.
    """
    path = _resolve(csv, PERFORMANCE_CSV, PERFORMANCE_PREFIX)
    df = pd.read_csv(path)
    if clean:
        df = cleaning.rename_models(cleaning.split_task(df))
    df.attrs["source"] = f"{path.name}: {len(df)} runs, {df['task'].nunique()} tasks"
    return df


def load_baseline(csv=None):
    """Unmodified-train.py scores, one row per task, ready to merge on task."""
    df = pd.read_csv(_resolve(csv, BASELINE_CSV, BASELINE_PREFIX))
    drop = [c for c in ("run_ts", "run_id", "baseline_score_count", "baseline_0") if c in df.columns]
    return df.drop(columns=drop)


def load_dollarstreet(csv=None):
    """DollarStreet runs: accuracy for the Advantaged and Disadvantaged groups.

    Kept out of the performance CSV because it is scored per income group
    rather than by the usual fairness metrics.
    """
    path = _resolve(csv, DOLLARSTREET_CSV, DOLLARSTREET_PREFIX)
    df = cleaning.rename_models(pd.read_csv(path))
    df.attrs["source"] = f"{path.name}: {len(df)} runs"
    return df


def load_run_counts(csv=None):
    """Runs attempted, completed and successfully scored, per model and task.

    Includes failed runs, so it can answer "how often did this work?" - the
    performance CSV only keeps runs that produced a score.
    """
    path = _resolve(csv, RUN_COUNTS_CSV, RUN_COUNTS_PREFIX)
    df = cleaning.rename_models(cleaning.split_task(pd.read_csv(path)))
    df.attrs["source"] = f"{path.name}: {df['runs'].sum()} runs over {len(df)} model/task pairs"
    return df


def load_perf_and_baseline(csv=None, baseline_csv=None):
    """Performance merged with baseline columns (rows without both dropped)."""
    perf = load_performance(csv, clean=False)
    base = load_baseline(baseline_csv)
    merged = cleaning.merge_baseline(perf, base)
    return cleaning.rename_models(cleaning.split_task(merged))
