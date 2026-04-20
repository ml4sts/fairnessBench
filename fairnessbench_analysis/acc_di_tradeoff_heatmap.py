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
task_data_metric = perf_alt['task'].str.split('_').apply(pd.Series).rename(columns ={0:'dataset',1:'task_metric',2:'task-dem'})
wider_code = pd.concat([perf_alt, task_data_metric],axis=1)
wider_cols=['model','task','dataset','task_metric','task-dem','run_ts','run_id','acc','precision','recall','di','statistical_parity_diff','equal_opp_diff','error_rate_diff','error_rate_ratio','false_omission_rate_diff','final_flake8_score']
wider_code = wider_code[wider_cols]

task_task_dem = wider_code['task-dem'].str.split('-').apply(pd.Series).rename(columns ={0:'rsch_prob',1:'dem'})
wider = pd.concat([wider_code, task_task_dem],axis=1)
cols= ['model','task','dataset','task_metric','task-dem','rsch_prob','dem','run_ts','run_id','acc','precision','recall','di','statistical_parity_diff','equal_opp_diff','error_rate_diff','error_rate_ratio','false_omission_rate_diff','final_flake8_score']
wider=wider[cols]
wider=wider.replace('claude-3-7-sonnet-20250219','claude-3-7-sonnet')

# Filtering only balance task from the dataframe
wider_balance = wider[wider['rsch_prob'].isin(["balance", "implicit", "best"])].copy()

# Adding fairness column
df = wider_balance.copy()
df["fair"] = 1 - (df['di'] - 1).abs()
df["fair"] = df["fair"].clip(lower=0)  

def pareto_max(df, x, y):
    """
    Return Pareto optimal points (maximizing both x and y).
    """
    data = df[[x, y]].to_numpy()
    keep = np.ones(len(df), dtype=bool)

    for i in range(len(df)):
        # point j dominates i if:
        # j is >= in both AND > in at least one
        dominates = np.all(data >= data[i], axis=1) & np.any(data > data[i], axis=1)
        dominates[i] = False
        if np.any(dominates):
            keep[i] = False

    return df[keep]

pareto_all = []

# loop through each panel and research problem
for (dataset, model, prob), g in df.groupby(['dataset','model', 'rsch_prob']):
    front = pareto_max(g, 'acc', "fair")
    
    # radius and angle for Pareto points
    front = front.copy()
    front["r"] = np.sqrt(front['acc']**2 + front["fair"]**2)
    front["theta"] = np.arctan2(front["fair"], front['acc'])  # radians
    
    pareto_all.append(front)

pareto_df = pd.concat(pareto_all, ignore_index=True)

r_summary = pareto_df.groupby(['dataset','model', 'rsch_prob'])["r"].mean().reset_index()
r_summary = r_summary.rename(columns={"r": "r_mean"})

# Function to calculate circular mean
def circ_mean(theta):
    return np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))

theta_summary = (
    pareto_df.groupby(['dataset','model', 'rsch_prob'])["theta"]
    .apply(circ_mean)
    .reset_index()
    .rename(columns={"theta": "theta_mean"})
)

summary = r_summary.merge(theta_summary, on=['dataset','model', 'rsch_prob'])
summary["theta_mean_deg"] = np.degrees(summary["theta_mean"])

angle_ranges = summary.groupby("rsch_prob")["theta_mean_deg"].agg(["min","max"])

# Function to calculate the overlap between two ranges
def range_overlap(a_min, a_max, b_min, b_max):
    overlap = max(0, min(a_max, b_max) - max(a_min, b_min))
    total = max(a_max, b_max) - min(a_min, b_min)
    return overlap / total if total > 0 else 0
    
pairs = [("balance","implicit"),
         ("balance","best"),
         ("implicit","best")]


# centering 
summary["theta_centered"] = summary["theta_mean_deg"] - 45

sns.set_context(context='poster',font_scale= 0.75)
cmap = sns.diverging_palette(145, 300, as_cmap=True)
summary["pm"] = summary["rsch_prob"].astype(str) + "-" + summary["model"].astype(str)
pivot = summary.pivot(
    index="pm",
    columns="dataset",
    values="theta_centered"
)


plt.figure(figsize=(12,8))

sns.heatmap(
        pivot,
        cmap=cmap,    
        center=0,
        vmin=-45,
        vmax=45,
        annot=True,
        fmt=".1f"
    )


plt.xlabel("Dataset")
plt.ylabel("Research Problem | Model")
plt.tight_layout()
output= os.path.join(GRAPHS,'acc_di_tradeoff_heatmap.png')  
plt.savefig(output, dpi=300, bbox_inches='tight')
