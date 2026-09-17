"""Plot helpers shared by the FairnessBench figures."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from path import GRAPHS
from constants import METRIC_IDEAL


def save_fig(name, dpi=300):
    """Save the current figure into GRAPHS (created if missing) and return the path."""
    os.makedirs(GRAPHS, exist_ok=True)
    output = os.path.join(GRAPHS, name)
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    return output


def ref_lines(grid, h=None, v=None, alpha=0.2):
    """Draw the same horizontal/vertical reference line on every facet."""
    for ax in grid.axes.flat:
        if h is not None:
            ax.axhline(y=h, color="black", linestyle="-.", alpha=alpha)
        if v is not None:
            ax.axvline(x=v, color="black", linestyle="-.", alpha=alpha)
    return grid


def metric_ref_lines(grid, row_metrics, v=1.0):
    """Per-row ideal-value lines for a FacetGrid whose rows are metrics.

    ``row_metrics`` lists the task-metric code of each facet row; the
    horizontal line lands on that metric's ideal value (1 for ratios,
    0 for differences), plus an optional vertical line (accuracy = 1).
    """
    for ax_row, metric in zip(grid.axes, row_metrics):
        ideal = METRIC_IDEAL[metric]
        for ax in ax_row:
            ax.axhline(y=ideal, color="black", linestyle="-.", alpha=0.3 if ideal == 1 else 0.6)
            if v is not None:
                ax.axvline(x=v, color="black", linestyle="-.", alpha=0.2)
    return grid


def rotate_xticklabels(grid, rotation=30, ha=None):
    for ax in grid.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(rotation)
            if ha:
                label.set_ha(ha)
    return grid


def grouped_heatmap(
    pivot,
    xlabel,
    title=None,
    cbar_label=None,
    cmap="viridis",
    center=None,
    vmin=None,
    vmax=None,
    fmt=".2g",
    figsize=(7, 9),
    annot_size=10,
    group_label_x=-0.30,
):
    """Heatmap over a (research_problem, model) MultiIndex.

    Rows show only the model name; research problems appear as bold group
    labels on the far left with thick separators between groups — the layout
    used by the sensitivity and trade-off figures.
    """
    sns.set_context("paper", font_scale=1.8)
    fig, ax = plt.subplots(figsize=figsize)

    cbar_kws = {"shrink": 0.5, "pad": 0.03}
    if cbar_label:
        cbar_kws["label"] = cbar_label
    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws=cbar_kws,
        annot_kws={"size": annot_size},
        ax=ax,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel(xlabel, labelpad=10)
    if title:
        ax.set_title(title, pad=12)
    label_research_problem_groups(ax, pivot.index, group_label_x)

    return fig, ax


def label_research_problem_groups(ax, index, label_x):
    """Show only model names on the y axis, grouped under research problems.

    ``index`` is a (research_problem, model) MultiIndex; each group gets a bold
    label on the far left and a thick line separating it from the next.
    """
    n_models = index.get_level_values(1).nunique()
    ax.set_yticklabels([model for _, model in index], rotation=0)
    ax.set_ylabel("")

    for i in range(n_models, len(index), n_models):
        ax.axhline(i, color="black", linewidth=2)

    groups = dict.fromkeys(index.get_level_values(0))
    for i, group in enumerate(groups):
        ax.text(label_x, i * n_models + n_models / 2, str(group).capitalize(),
                transform=ax.get_yaxis_transform(), ha="center", va="center",
                fontsize=14, fontweight="bold", rotation=90)


# Cells are coloured by "<accuracy>,<fairness>" change labels from
# cleaning.add_change_labels; numbers index into CHANGE_COLORS.
CHANGE_CODES = {
    "better,worse": 0, "better,similar": 1, "better,better": 2,
    "similar,worse": 3, "similar,similar": 4, "similar,better": 5,
    "worse,worse": 6, "worse,similar": 7, "worse,better": 8,
}
CHANGE_COLORS = [
    "#40dba7", "#408fa7", "#403593",
    "#a5e8cd", "#a58fb6", "#a53593",
    "#f7fcf5", "#f78fb6", "#f73593",
]

# Cases where a mean±std comparison is degenerate, marked in the cell.
EDGE_CASE_FLAGS = [
    (r"$\ast$", "comparison accuracy = 1", lambda r: r["compare_acc"] == 1),
    (r"$\dagger$", "baseline accuracy = 1", lambda r: r["base_acc"] == 1),
    (r"$\circ$", "baseline DI std = 0", lambda r: r["base_di_std"] == 0),
    (r"$\bullet$", "baseline accuracy std = 0", lambda r: r["base_acc_std"] == 0),
    (r"$\diamond$", "comparison DI std = 0", lambda r: r["compare_di_std"] == 0),
    (r"$\ddagger$", "comparison accuracy std = 0", lambda r: r["compare_acc_std"] == 0),
    (r"$\flat$", "baseline is a single run", lambda r: pd.isna(r["base_acc_std"])),
    (r"$\sharp$", "comparison is a single run", lambda r: pd.isna(r["compare_acc_std"])),
]


def change_heatmap(sens, comparison_col, xlabel, title):
    """Heatmap of how accuracy and DI changed against the baseline.

    ``sens`` comes from cleaning.add_change_labels. Rows are models grouped by
    research problem, columns are ``comparison_col`` (the dataset variant or
    prompt). A 3x3 legend decodes the colours, and symbols mark edge cases.
    """
    sens = sens.copy()
    sens["code"] = sens["overall"].map(CHANGE_CODES)
    sens["flags"] = sens.apply(
        lambda r: "".join(symbol for symbol, _, applies in EDGE_CASE_FLAGS if applies(r)), axis=1)

    index = ["research_problem", "model"]
    codes = sens.pivot_table(index=index, columns=comparison_col, values="code")
    flags = sens.pivot_table(index=index, columns=comparison_col, values="flags",
                             aggfunc="first").reindex_like(codes).fillna("")

    sns.set_context("paper", font_scale=1.5)
    fig, axd = plt.subplot_mosaic([["main", "color_legend"],
                                   ["main", "symbol_legend"]],
                                  figsize=(14, 8), gridspec_kw={"width_ratios": [2, 1]})

    ax = axd["main"]
    sns.heatmap(codes, annot=flags, fmt="", cmap=CHANGE_COLORS, vmin=0, vmax=8,
                linewidths=0.5, linecolor="white", annot_kws={"size": 11}, cbar=False, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_title(title, pad=12)
    label_research_problem_groups(ax, codes.index, label_x=-0.18)

    labels = ["worse", "similar", "better"]
    legend = pd.DataFrame(
        [[CHANGE_CODES[f"{acc},{fair}"] for acc in labels] for fair in labels[::-1]],
        index=pd.Index(labels[::-1], name="fairness"),
        columns=pd.Index(labels, name="accuracy"))
    sns.heatmap(legend, cmap=CHANGE_COLORS, vmin=0, vmax=8, cbar=False,
                ax=axd["color_legend"])

    axd["symbol_legend"].axis("off")
    axd["symbol_legend"].legend(
        handles=[Line2D([0], [0], marker="None", color="none", label=f"{symbol}  {text}")
                 for symbol, text, _ in EDGE_CASE_FLAGS],
        loc="upper right", bbox_to_anchor=(1.02, 0.65), frameon=False,
        title="Edge-case flags", title_fontsize=12, fontsize=10,
        handlelength=0, handletextpad=0)

    return fig, axd


def symmetric_limits(values):
    """(vmin, vmax) centered on zero, spanning the data's magnitude."""
    vmax = np.ceil(np.nanmax(np.abs(values)))
    return -vmax, vmax
