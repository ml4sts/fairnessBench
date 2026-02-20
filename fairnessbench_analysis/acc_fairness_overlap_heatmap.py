import os
import pandas as pd
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
from path import CSV_FILES,GRAPHS


file= CSV_FILES/'Final_step_perfomance2026-02-07T02:23:17.201613.csv'
perf_alt = pd.read_csv(file)

perf= ['acc','precision', 'recall', 'di', 'statistical_parity_diff', 
                  'equal_opp_diff', 'error_rate_diff', 'error_rate_ratio', 
                  'false_omission_rate_diff','final_flake8_score']

perf_alt = perf_alt.replace([np.inf, -np.inf], np.nan).fillna(0)

# expanding the task to sub columns 
task_data_metric = perf_alt['task'].str.split('_').apply(pd.Series).rename(columns ={0:'task_dataset',1:'task_metric',2:'task-dem'})
wider_code = pd.concat([perf_alt, task_data_metric],axis=1)
wider_cols=['model','task','task_dataset','task_metric','task-dem','run_ts','run_id','acc','precision','recall','di','statistical_parity_diff','equal_opp_diff','error_rate_diff','error_rate_ratio','false_omission_rate_diff','final_flake8_score']
wider_code = wider_code[wider_cols]

task_task_dem = wider_code['task-dem'].str.split('-').apply(pd.Series).rename(columns ={0:'rsch_prob',1:'dem'})
wider = pd.concat([wider_code, task_task_dem],axis=1)
cols= ['model','task','task_dataset','task_metric','task-dem','rsch_prob','dem','run_ts','run_id','acc','precision','recall','di','statistical_parity_diff','equal_opp_diff','error_rate_diff','error_rate_ratio','false_omission_rate_diff','final_flake8_score']
wider=wider[cols]
wider=wider.replace('claude-3-7-sonnet-20250219','claude-3-7-sonnet')

# summary df of the mean and std 
summary = (
    wider.groupby(["model", "task_dataset", "rsch_prob"])
      .agg(
          mean_acc=("acc","mean"),
          mean_di=("di","mean"),
          std_acc=("acc","std"),
          std_di=("di","std")
      )
      .reset_index()
)

allowed_datasets = ["randoadult","sampadult","nondescriptive",'health']

results = []

# here we loop over each (model, research problem) group
for (model, rp), group in summary.groupby(["model","rsch_prob"]):

    # find baseline (adult)
    base_row = group[group["task_dataset"]=="adult"]
#if a model+research-problem doesn’t have an Adult row, I can’t compute diffs, so skip it
    if base_row.empty:
        continue
# extracting adults mean n std , .iloc[0] grabs the first row value (assumes only one Adult row exists in this group).
    base_di = base_row["mean_di"].iloc[0]
    base_acc = base_row["mean_acc"].iloc[0]
    # build baseline DI interval using variance 
    base_min_di = base_di - base_row["std_di"].iloc[0] 
    base_max_di = base_di + base_row['std_di'].iloc[0]
    # build baseline ACC interval using std
    base_min_acc = base_acc - base_row["std_acc"].iloc[0] 
    base_max_acc = base_acc + base_row['std_acc'].iloc[0]
# iterate through each dataset std (adult, randoadult, sampadult, …) for this model+rproblem
    for _, row in group.iterrows():
        # get dataset name for this row
        dataset = row["task_dataset"]
        # skip baseline itself, don't compare adult to itself
        if dataset == "adult":
            continue
        if dataset not in allowed_datasets:
            continue

        change_type = "data_change"
        if row["task_dataset"] == "nondescriptive":
            change_type = "context_change"
        row_max_di = (row["mean_di"] + row['std_di'])
        row_min_di = (row["mean_di"] - row['std_di'])

        row_max_acc = (row["mean_acc"] + row['std_acc'])
        row_min_acc = (row["mean_acc"] - row['std_acc'])
        results.append({
            "model": model,
            "rsch_prob": rp,
            "baseline_dataset": "adult",
            "comparison_dataset": dataset,
            # Differences in mean fairness and accuracy, these measure how far the means move from Adult baseline.
            "fairness_diff": abs(base_di - row["mean_di"]),
            "accuracy_diff": abs(base_acc - row["mean_acc"]),
            'min_fair_dff':row_min_di,
            'max_fair_diff':row_max_di,
            'min_acc_diff':row_min_acc,
            'max_acc_diff':row_max_acc,
            # compute overlap between baseline interval and comparison interval
            # overlap = 0  intervals are disjoint (stronger evidence of change)
            # overlap large  intervals similar/overlapping (weaker evidence of change)
            'overlap_fair': max(0,min(base_max_di,row_max_di)-max(base_min_di,row_min_di)), # width
            # Baseline interval length
            'len_man_min': base_max_di - base_min_di, 
            'overlap_acc': max(0,min(base_max_acc,row_max_acc)-max(base_min_acc,row_min_acc)), # width
            'len_acc_minmax':base_max_acc - base_min_acc, 
            "change_type": change_type
        })


sens_df = pd.DataFrame(results)
sens_df['final_overlap_fair'] = sens_df['overlap_fair'] / sens_df['len_man_min']
sens_df['final_overlap_acc'] = sens_df['overlap_acc'] / sens_df['len_acc_minmax']

# heatmap for fairness_overlap
heatmap_data = sens_df.pivot_table(
    index=["rsch_prob", "model"],
    columns="comparison_dataset",
    values="final_overlap_fair",
    aggfunc="mean"
).sort_index(level=["rsch_prob", "model"])

plt.figure(figsize=(12,6))

sns.heatmap(
    heatmap_data,
    annot=True,        
    cmap="viridis",
    linewidths=0.5
)

plt.title("Fairness Sensitivity Across Dataset Variants")
plt.xlabel("Comparison Dataset")
plt.ylabel("Research Problem | Model")

output= os.path.join(GRAPHS,'fairness_overlap_heatmap.png')  
plt.savefig(output, dpi=300, bbox_inches='tight')

# heatmap for acc_overlap
heatmap_data = sens_df.pivot_table(
    index=["rsch_prob", "model"],
    columns="comparison_dataset",
    values="final_overlap_acc",
    aggfunc="mean"
).sort_index(level=["rsch_prob", "model"])

plt.figure(figsize=(12,6))

sns.heatmap(
    heatmap_data,
    annot=True,        
    cmap="viridis",
    linewidths=0.5
)

plt.title("Accuracy Sensitivity Across Dataset Variants")
plt.xlabel("Comparison Dataset")
plt.ylabel("Research Problem | Model")
output= os.path.join(GRAPHS,'acc_overlap_heatmap.png')  
plt.savefig(output, dpi=300, bbox_inches='tight')