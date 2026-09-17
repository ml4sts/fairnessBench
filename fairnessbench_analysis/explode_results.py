"""Flatten the raw eval JSONs into the CSVs the analysis reads.

Stage 3 of the pipeline (run -> eval -> analyze), writing into CSV_FILES:
  Final_step_perfomance<timestamp>.csv        agent runs, metrics + flake8 detail
  Baseline_cleaned_perfomance<timestamp>.csv  unmodified train.py, columns prefixed

Run it from this directory: python explode_results.py
"""

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from path import PROJECT_ROOT, CSV_FILES, BASELINE_ROOT
from constants import FNA_SECTIONS, INCOME_COLS

# Runs are logged as .../<model>/<task>/<run_ts>/env_log/trace.json, so each id
# sits at a fixed position from the end. Baselines have no model level.
PATH_PARTS = {'model': -5, 'task': -4, 'run_ts': -3}

N_RUNS_KEPT = 8


def load_results(directory):
    """Read every JSON in `directory` into one frame, a row per run."""
    frames = []
    for fname in sorted(os.listdir(directory)):
        rf = os.path.join(directory, fname)
        if not os.path.isfile(rf):
            continue
        try:
            if os.path.getsize(rf) == 0:
                print(f"Skipping empty file: {rf}")
                continue
            frames.append(pd.read_json(rf).T)
        except Exception as e:
            print(f"Skipping file {rf} due to error: {e}")
    return pd.concat(frames)


def split_run_path(paths, names):
    """Pull `names` (model, task, run_ts) out of the run's log path."""
    parts = paths.str.split('/')
    return pd.DataFrame({name: parts.str[PATH_PARTS[name]] for name in names},
                        index=paths.index)


def explode_final_score(results, names, run_id_by):
    """One row per run, one column per metric in its `final_score` dict."""
    scores = results['final_score'].apply(pd.Series).reset_index()
    scores = scores.join(split_run_path(scores['index'], names))
    scores['run_id'] = scores.groupby(run_id_by).cumcount()

    id_cols = names + ['run_id']
    metrics = [c for c in scores.columns if c not in id_cols + ['index']]
    return scores[id_cols + metrics]


# Each check is one line: "<file>:<line>:<col>: FNA101 Data_collection: <detail>".
# Checks that earned points end in ", +N"; the rest say what was missing.
_LINE_PREFIX = re.compile(r"^.*?:\d+:\d+:\s+")
_FNA_LINE = re.compile(r"^(FNA\d{3})\s+\w+:\s*(.*)$")
_POINTS = re.compile(r",\s*([+-]?\d+(?:\.\d+)?)\s*$")


def parse_flake8_report(report):
    """Expand one flake8 report into points and a message per check.

    A run with no report (syntax error, or flake8 never ran) leaves every column
    NaN, so "scored zero" stays distinguishable from "never scored".
    """
    columns = {}
    for section in FNA_SECTIONS.values():
        columns[f"flake8_{section}"] = np.nan
        columns[f"flake8_{section}_detail"] = np.nan

    if not isinstance(report, str):
        return columns

    for raw_line in report.splitlines():
        check = _FNA_LINE.match(_LINE_PREFIX.sub("", raw_line.strip()))
        if not check:
            continue  # an ordinary style finding, not one of our checks
        section = FNA_SECTIONS.get(check.group(1))
        if section is None:
            continue
        detail = check.group(2)
        points = _POINTS.search(detail)
        columns[f"flake8_{section}"] = float(points.group(1)) if points else 0.0
        columns[f"flake8_{section}_detail"] = _POINTS.sub("", detail).strip()
    return columns


# eval.py tags each run by string-matching its logs; a flag can be set on a run
# that still succeeded, so these are only counted for runs that produced no score.
FAILURE_FLAGS = ['oom_error', 'connection_error', 'json_error', 'long_prompt_error', 'error']


def count_runs(results, names):
    """Per model and task: how many runs there were, and how many got that far.

    Counted before failures are filtered out, so the success rates in the
    appendix tables have the full denominator. ``successful_runs`` produced a
    scoreable submission; ``completed_runs`` finished without an error.
    """
    counts = split_run_path(results['path'], names)
    counts['scored'] = results['final_score'].apply(lambda x: isinstance(x, dict)).values
    counts['completed'] = ((results['total_time'] > 0) & (results['error'] == '')).values

    # why the failures failed, as far as the logs say
    extra = results['extra'].apply(lambda e: e if isinstance(e, dict) else {})
    flagged = pd.DataFrame(index=counts.index)
    for flag in FAILURE_FLAGS:
        flagged[flag] = extra.apply(lambda e: bool(e.get(flag))).values
        counts[f'failed_{flag}'] = ~counts['scored'] & flagged[flag]
    counts['failed_unexplained'] = ~counts['scored'] & ~flagged.any(axis=1)

    agg = {'runs': ('run_ts', 'count'),
           'successful_runs': ('scored', 'sum'),
           'completed_runs': ('completed', 'sum')}
    agg.update({c: (c, 'sum') for c in counts.columns if c.startswith('failed_')})
    return counts.groupby(['model', 'task']).agg(**agg).reset_index()


def keep_last_runs(df, group_cols, n=N_RUNS_KEPT):
    """Keep the `n` most recent runs per group, oldest run_id first."""
    return df.sort_values(group_cols + ['run_id']).groupby(group_cols).tail(n)


def output_path(prefix):
    """Timestamped CSV path, so each explode run is kept distinct."""
    return os.path.join(CSV_FILES, prefix + datetime.isoformat(datetime.now()) + '.csv')


# --- agent runs -------------------------------------------------------------
results = load_results(PROJECT_ROOT)

# A dict final_score means the run finished and was scored; anything else failed.
successful = results[results['final_score'].apply(lambda x: isinstance(x, dict))]
print(f"Total rows after filtering: {len(successful)}")
print(f"Rows dropped: {len(results) - len(successful)}")

names = ['model', 'task', 'run_ts']
run_counts = count_runs(results, names)
run_counts.to_csv(output_path('Run_counts'), index=False)
print(f"Run counts: {len(run_counts)} model/task pairs")

performance = explode_final_score(successful, names, run_id_by=['model', 'task'])

# flake8 is scored beside final_score rather than inside it, and its report
# breaks that score down by check.
flake8 = successful[['path', 'final_flake8_score', 'final_flake8_report']].copy()
flake8 = flake8.join(split_run_path(flake8['path'], names))
sections = flake8['final_flake8_report'].apply(parse_flake8_report).apply(pd.Series)

performance = performance.merge(
    flake8.join(sections)[names + ['final_flake8_score'] + list(sections.columns)],
    on=names, how='left')

performance = keep_last_runs(performance, ['task', 'model'])

# DollarStreet is scored by income group instead of the usual fairness metrics,
# so it gets its own CSV. Left in the main file, its empty Advantaged /
# Disadvantaged columns would be NaN for every other run.
is_dollarstreet = performance[INCOME_COLS].notna().any(axis=1)
dollarstreet = performance[is_dollarstreet].dropna(axis=1, how='all')
performance = performance[~is_dollarstreet].drop(columns=INCOME_COLS, errors='ignore')

print(f"\nFinal rows after keeping last {N_RUNS_KEPT} per task: {len(performance)}")
performance.to_csv(output_path('Final_step_perfomance'), index=False)

print(f"DollarStreet rows: {len(dollarstreet)}")
if len(dollarstreet):
    dollarstreet.to_csv(output_path('Dollarstreet_perfomance'), index=False)

# --- baseline (unmodified train.py) -----------------------------------------
if not os.path.isdir(BASELINE_ROOT):
    print(f"\nBaseline results not found at {BASELINE_ROOT} - skipping baseline CSV.")
else:
    names = ['task', 'run_ts']
    baseline = explode_final_score(load_results(BASELINE_ROOT), names,
                                   run_id_by=['task', 'run_ts'])
    # prefix the metrics so they survive a merge with the agent results
    id_cols = names + ['run_id']
    baseline = baseline.rename(
        columns={c: f'baseline_{c}' for c in baseline.columns if c not in id_cols})
    print(f"\nBaseline rows: {len(baseline)}")
    baseline.to_csv(output_path('Baseline_cleaned_perfomance'), index=False)
