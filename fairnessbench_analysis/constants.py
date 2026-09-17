"""Shared vocabulary for the FairnessBench analysis.

Task IDs decompose as ``{dataset}_{metric}_{researchproblem}-{sensitive_attr}``.
Prompt-sensitivity tasks use ``{dataset}_{rp}-{metric}-{variation}-{sensitive_attr}``.
"""

# ---------------------------------------------------------------------------
# CSV files
# ---------------------------------------------------------------------------
# explode_results.py writes one timestamped CSV per kind; every analysis reads
# the newest one. Set these to a filename in CSV_FILES to pin an older run
# (e.g. to reproduce a published figure); None means "use the newest".
PERFORMANCE_CSV = None
BASELINE_CSV = None

PERFORMANCE_PREFIX = "Final_step_perfomance"
BASELINE_PREFIX = "Baseline_cleaned_perfomance"
DOLLARSTREET_PREFIX = "Dollarstreet_perfomance"
DOLLARSTREET_CSV = None
RUN_COUNTS_PREFIX = "Run_counts"
RUN_COUNTS_CSV = None

# DollarStreet scores accuracy per income group rather than the usual metrics
INCOME_COLS = ["Advantaged", "Disadvantaged"]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
# task-metric code (as it appears in the task ID) -> column in the performance CSV
METRIC_MAP = {
    "acc": "acc",
    "di": "di",
    "spd": "statistical_parity_diff",
    "eod": "equal_opp_diff",
    "err": "error_rate_ratio",
    "erd": "error_rate_diff",
    "ford": "false_omission_rate_diff",
}

METRIC_DISPLAY = {
    "acc": "Accuracy",
    "di": "Disparate Impact",
    "spd": "Statistical Parity Diff.",
    "eod": "Equal Opp Diff",
    "err": "Error Rate Ratio",
    "erd": "Error Rate Diff.",
    "ford": "False Omission Rate Diff.",
}

# Value each metric takes for a perfectly fair (or perfect) model.
METRIC_IDEAL = {
    "acc": 1,
    "di": 1,
    "spd": 0,
    "eod": 0,
    "err": 1,
    "erd": 0,
    "ford": 0,
}

# Direction conventions used for the "metric ↑ / metric ↓" axis labels.
HIGH_GOOD = ["acc", "p", "r", "di", "err"]
LOW_GOOD = ["spd", "eod", "erd", "ford"]

# Improvement over baseline, oriented so + is better and - is worse.
# Ratio metrics (di, err) measure distance from 1.
IMPROVEMENT_FX = {
    "acc": lambda r: r["task_metric_value"] - r["task_metric_value_baseline"],
    "di": lambda r: abs(1 - r["task_metric_value_baseline"]) - abs(1 - r["task_metric_value"]),
    "spd": lambda r: r["task_metric_value_baseline"] - r["task_metric_value"],
    "eod": lambda r: r["task_metric_value_baseline"] - r["task_metric_value"],
    "err": lambda r: abs(1 - r["task_metric_value_baseline"]) - abs(1 - r["task_metric_value"]),
    "erd": lambda r: r["task_metric_value_baseline"] - r["task_metric_value"],
    "ford": lambda r: r["task_metric_value_baseline"] - r["task_metric_value"],
}

PERF_COLS = [
    "acc",
    "precision",
    "recall",
    "di",
    "statistical_parity_diff",
    "equal_opp_diff",
    "error_rate_diff",
    "error_rate_ratio",
    "false_omission_rate_diff",
]
FAIRNESS_COLS = PERF_COLS[3:]

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODEL_RENAME = {"claude-3-7-sonnet-20250219": "claude"}

# ---------------------------------------------------------------------------
# Dataset-variant sensitivity (Figures 8 / 12)
# ---------------------------------------------------------------------------
VARIANT_DATASETS = [
    "randoadult",
    "sampadult",
    "nondescriptive",
    "health",
    "sampcreditdefault",
    "samphealth",
    "sampgerman",
]
VARIANT_BASELINE = {
    "randoadult": "adult",
    "sampadult": "adult",
    "nondescriptive": "adult",
    "health": "adult",
    "sampcreditdefault": "creditdefault",
    "samphealth": "adult",
    "sampgerman": "german",
}

# ---------------------------------------------------------------------------
# Prompt-variation sensitivity (Figures 10 / 11)
# ---------------------------------------------------------------------------
PROMPT_SENSITIVITY_TASKS = [
    "adult_balance-eod-sex",
    "adult_eod_balance-sex",
    "adult_balance-eod-nosuccess-sex",
    "adult_balance-eod-shortgoal-sex",
    "adult_balance-eod-noreq-sex",
    "adult_balance-eod-nochange-sex",
    "adult_balance-eod-nohow-sex",
    "adult_balance-eod-rephrased01-sex",
    "adult_balance-eod-rephrased10-sex",
    "adult_balance-eod-rephrased06-sex",
    "adult_balance-eod-rephrased05-sex",
    "adult_balance-eod-rephrased03-sex",
    "adult_balance-eod-altmetricdetail-sex",
    "adult_balance-eod-altmetricdetail2-sex",
]

PROMPT_VARIATION_RENAME = {
    "altmetricdetail": "altmetricnames",
    "altmetricdetail2": "informalgoal",
    "rephrased01": "informaldirect",
    "rephrased03": "altnowork",
    "rephrased05": "verbosedetail",
    "rephrased06": "informationalpassive",
    "rephrased10": "passivedata",
}

ALLOWED_PROMPT_VARIATIONS = [
    "altmetricnames",
    "informalgoal",
    "nochange",
    "nohow",
    "nosuccess",
    "informaldirect",
    "altnowork",
    "verbosedetail",
    "informationalpassive",
    "passivedata",
    "shortgoal",
]


# ---------------------------------------------------------------------------
# Flake8 fairness plugin (FNA101-FNA109)
# ---------------------------------------------------------------------------
# The plugin prints one line per check, awarding points when it finds evidence
# and explaining what was missing when it doesn't. explode_results.py turns each
# check into a points column (0.0 when the check found nothing, NaN when the run
# produced no report at all) plus a `_detail` column holding the message.
FNA_SECTIONS = {
    "FNA101": "data_collection",
    "FNA102": "column_identification",
    "FNA103": "preprocessing",
    "FNA104": "categorical_encoding",
    "FNA105": "sensitive_features",
    "FNA106": "bias_mitigation_libraries",
    "FNA107": "fairness_metrics",
    "FNA108": "model_training",
    "FNA109": "evaluation",
}

FLAKE8_SECTION_COLS = [f"flake8_{name}" for name in FNA_SECTIONS.values()]
FLAKE8_DETAIL_COLS = [f"{col}_detail" for col in FLAKE8_SECTION_COLS]
