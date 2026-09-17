"""Transformations shared across the FairnessBench analyses.

column names produced here: ``dataset``, ``task_metric``,
``research_problem``, ``dem`` (sensitive attribute), plus
``task_metric_value`` for the metric each task targets.
"""

import numpy as np
import pandas as pd

from constants import (
    METRIC_MAP,
    IMPROVEMENT_FX,
    MODEL_RENAME,
    PROMPT_SENSITIVITY_TASKS,
    PROMPT_VARIATION_RENAME,
)


# ---------------------------------------------------------------------------
# Task decomposition
# ---------------------------------------------------------------------------

def split_task(df, task_col="task"):
    """Decompose ``{dataset}_{metric}_{rp}-{dem}`` task IDs into columns.

    Tasks that don't follow the 3-part pattern (prompt-sensitivity,
    dollarstreet) get NaN in the affected columns, so they drop out of
    standard analyses naturally.
    """
    df = df.copy()
    parts = df[task_col].str.split("_", expand=True).reindex(columns=range(3))
    df["dataset"] = parts[0]
    df["task_metric"] = parts[1]
    rp_dem = parts[2].str.split("-", expand=True).reindex(columns=range(2))
    df["research_problem"] = rp_dem[0]
    df["dem"] = rp_dem[1]
    front = [c for c in ("model", task_col, "dataset", "task_metric", "research_problem", "dem") if c in df.columns]
    return df[front + [c for c in df.columns if c not in front]]


def prompt_sensitivity_frame(df, task_col="task"):
    """Filter to the prompt-variation tasks and decompose their 4-part IDs.

    Adds ``dataset``, ``research_problem``, ``task_metric``,
    ``prompt_variation`` ('original' for the unmodified prompt, publication
    names for the variants), and ``dem``.
    """
    df = df[df[task_col].isin(PROMPT_SENSITIVITY_TASKS)].copy()
    df[task_col] = df[task_col].replace("adult_eod_balance-sex", "adult_balance-eod-original-sex")
    parts = df[task_col].str.split("_", expand=True)
    df["dataset"] = parts[0]
    info = parts[1].str.split("-", expand=True).reindex(columns=range(4))
    df["research_problem"] = info[0]
    df["task_metric"] = info[1]
    df["prompt_variation"] = info[2].replace(PROMPT_VARIATION_RENAME)
    df["dem"] = info[3]
    return rename_models(df)


def rename_models(df, model_col="model"):
    """Shorten model IDs to the display names used in the paper."""
    if model_col in df.columns:
        df = df.copy()
        df[model_col] = df[model_col].replace(MODEL_RENAME)
    return df


# ---------------------------------------------------------------------------
# Target-metric extraction and baseline comparison
# ---------------------------------------------------------------------------

def add_task_metric_value(df, out_col="task_metric_value", prefix=""):
    """Pull each row's targeted metric into one column.

    ``prefix`` selects prefixed columns (e.g. ``'baseline_'``). Rows whose
    task metric isn't a performance metric (``allmetric``, dollarstreet
    tasks) get NaN.
    """
    df = df.copy()

    def pick(row):
        col = METRIC_MAP.get(row["task_metric"])
        return row[prefix + col] if col else np.nan

    df[out_col] = df.apply(pick, axis=1)
    return df


def merge_baseline(perf_df, baseline_df, on="task"):
    """Left-merge baseline columns onto performance rows, dropping rows
    without a complete pairing (matches the paper's target10 preprocessing)."""
    merged = perf_df.merge(baseline_df.fillna(0), how="left", on=on)
    return merged.dropna(how="any")


def add_improvement(df):
    """Score each run against its task's baseline.

    Adds ``task_metric_value``, ``task_metric_value_baseline``,
    ``agent-improvement`` (oriented so + is better), and the
    ``agent-impact`` label.
    """
    df = add_task_metric_value(df)
    df = add_task_metric_value(df, out_col="task_metric_value_baseline", prefix="baseline_")
    df["agent-baseline"] = df["task_metric_value"] - df["task_metric_value_baseline"]
    df["agent-improvement"] = df.apply(lambda r: IMPROVEMENT_FX[r["task_metric"]](r), axis=1)
    df["agent-impact"] = np.where(df["agent-improvement"] > 0, "improvement", "no improvement")
    return df


# ---------------------------------------------------------------------------
# Sensitivity analyses (interval-overlap between run distributions)
# ---------------------------------------------------------------------------

def mean_std_summary(df, group_cols):
    """Per-group mean/std of accuracy and DI, in the shape
    ``overlap_vs_baseline`` expects."""
    return (
        df.groupby(list(group_cols))
        .agg(
            mean_acc=("acc", "mean"),
            mean_di=("di", "mean"),
            std_acc=("acc", "std"),
            std_di=("di", "std"),
        )
        .reset_index()
    )


def overlap_vs_baseline(summary, item_col, baseline, group_cols=("model", "research_problem"), allowed=None):
    """Compare each item's mean±std interval to its baseline's.

    ``summary`` comes from :func:`mean_std_summary` (grouped by
    ``group_cols`` + ``item_col``). ``baseline`` is either a fixed item name
    (prompt sensitivity: ``'original'``) or a dict mapping item -> baseline
    item (dataset variants). ``final_overlap_*`` is the interval overlap
    normalized by the baseline interval's length: 0 = disjoint (strong
    evidence of change), large = similar distributions.
    """
    rows = []
    for keys, group in summary.groupby(list(group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        for _, row in group.iterrows():
            item = row[item_col]
            base_name = baseline if isinstance(baseline, str) else baseline.get(item)
            if base_name is None or item == base_name:
                continue
            if allowed is not None and item not in allowed:
                continue
            base = group[group[item_col] == base_name]
            if base.empty:
                continue
            base = base.iloc[0]
            rec = dict(zip(group_cols, keys))
            rec.update(
                {
                    f"baseline_{item_col}": base_name,
                    f"comparison_{item_col}": item,
                    "base_di": base["mean_di"],
                    "compare_di": row["mean_di"],
                    "base_acc": base["mean_acc"],
                    "compare_acc": row["mean_acc"],
                    "base_di_std": base["std_di"],
                    "compare_di_std": row["std_di"],
                    "base_acc_std": base["std_acc"],
                    "compare_acc_std": row["std_acc"],
                    "fairness_diff": abs(base["mean_di"] - row["mean_di"]),
                    "accuracy_diff": abs(base["mean_acc"] - row["mean_acc"]),
                    "overlap_fair": _overlap(base, row, "di"),
                    "overlap_acc": _overlap(base, row, "acc"),
                    "len_fair_base": 2 * base["std_di"],
                    "len_acc_base": 2 * base["std_acc"],
                }
            )
            rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["final_overlap_fair"] = out["overlap_fair"] / out["len_fair_base"]
    out["final_overlap_acc"] = out["overlap_acc"] / out["len_acc_base"]
    return out


def add_change_labels(sens):
    """Label how accuracy and DI moved relative to the baseline.

    Each metric is "similar" when the comparison mean falls inside the
    baseline's mean ± std band, otherwise "better" or "worse". ``overall``
    joins them as "<accuracy>,<fairness>", e.g. "worse,better".

    A group with a single run has no std; its band is treated as zero width,
    and ``plotting.change_heatmap`` flags those cells.
    """
    sens = sens.copy()
    sens["acc_change"] = sens.apply(judge_acc, axis=1)
    sens["di_change"] = sens.apply(judge_di, axis=1)
    sens["overall"] = sens["acc_change"] + "," + sens["di_change"]
    return sens


def judge_acc(row):
    if _within_baseline_band(row, "acc"):
        return "similar"
    return "better" if row["compare_acc"] > row["base_acc"] else "worse"


def judge_di(row):
    # DI is best at 1, so compare distances from 1 rather than raw values
    if _within_baseline_band(row, "di"):
        return "similar"
    return "worse" if abs(1 - row["compare_di"]) > abs(1 - row["base_di"]) else "better"


def _within_baseline_band(row, metric):
    std = row[f"base_{metric}_std"]
    if pd.isna(std):
        std = 0
    return abs(row[f"compare_{metric}"] - row[f"base_{metric}"]) <= std


def _overlap(base, row, metric):
    base_lo, base_hi = base[f"mean_{metric}"] - base[f"std_{metric}"], base[f"mean_{metric}"] + base[f"std_{metric}"]
    lo, hi = row[f"mean_{metric}"] - row[f"std_{metric}"], row[f"mean_{metric}"] + row[f"std_{metric}"]
    return max(0, min(base_hi, hi) - max(base_lo, lo))


# ---------------------------------------------------------------------------
# Accuracy/fairness trade-off (Pareto) analysis
# ---------------------------------------------------------------------------

def pareto_front(df, x, y):
    """Rows not dominated in both ``x`` and ``y`` (maximizing both)."""
    data = df[[x, y]].to_numpy()
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        dominates = np.all(data >= data[i], axis=1) & np.any(data > data[i], axis=1)
        dominates[i] = False
        if np.any(dominates):
            keep[i] = False
    return df[keep]


def circular_mean(theta):
    return np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))


def pareto_tradeoff_summary(df, group_cols=("dataset", "model", "research_problem"), x="acc", y="fair"):
    """Per-group polar summary of the accuracy/fairness Pareto front.

    ``theta_centered`` is the front's mean angle relative to the 45° line:
    negative leans toward accuracy, positive toward fairness. Groups whose
    front has fewer than 2 points have no meaningful trade-off and get NaN.
    """
    group_cols = list(group_cols)
    fronts = []
    for _, g in df.groupby(group_cols):
        front = pareto_front(g, x, y).copy()
        front["r"] = np.sqrt(front[x] ** 2 + front[y] ** 2)
        front["theta"] = np.arctan2(front[y], front[x])
        fronts.append(front)
    pareto_df = pd.concat(fronts, ignore_index=True)

    summary = (
        pareto_df.groupby(group_cols)
        .agg(r_mean=("r", "mean"), theta_mean=("theta", circular_mean), n_pareto=("theta", "size"))
        .reset_index()
    )
    summary["theta_mean_deg"] = np.degrees(summary["theta_mean"])
    summary["theta_centered"] = summary["theta_mean_deg"] - 45
    summary["has_valid_tradeoff"] = summary["n_pareto"] >= 2
    summary.loc[~summary["has_valid_tradeoff"], "theta_centered"] = np.nan
    return summary, pareto_df
