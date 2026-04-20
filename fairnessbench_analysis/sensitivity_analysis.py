
import os
import pandas as pd
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns

 
df_final = pd.read_csv("Final_step_perfomance2026-02-07T02:23:17.201613.csv")
#df_baseline = pd.read_csv("Baseline_cleaned_perfomance2026-02-07T02:23:19.806782.csv")

 
df_final.columns

 
df_final.head(2)

 
#df_baseline.head(2)

 
cols_to_keep = ['model', 'task', 'run_ts', 'run_id', 'acc', 'precision', 'recall', 'di', 'statistical_parity_diff',
'equal_opp_diff', 'error_rate_diff', 'error_rate_ratio', 'false_omission_rate_diff']

 
df_final = df_final[cols_to_keep]
df_final.head(2)
 
list_of_values = ['adult_balance-eod-sex', 'adult_eod_balance-sex', 'adult_balance-eod-nosuccess-sex', 
'adult_balance-eod-shortgoal-sex', 'adult_balance-eod-noreq-sex', 'adult_balance-eod-nochange-sex', 'adult_balance-eod-nohow-sex',
'adult_balance-eod-rephrased01-sex', 'adult_balance-eod-rephrased10-sex', 'adult_balance-eod-rephrased06-sex', 
'adult_balance-eod-rephrased05-sex', 'adult_balance-eod-rephrased03-sex', 'adult_balance-eod-altmetricdetail-sex', 
'adult_balance-eod-altmetricdetail2-sex']

filtered_df = df_final[df_final['task'].isin(list_of_values)]
filtered_df

 
filtered_df['task']=filtered_df['task'].replace('adult_eod_balance-sex', 'adult_balance-eod-original-sex')

 
filtered_df['task'].value_counts()

 
data_task = filtered_df['task'].str.split('_').apply(pd.Series).rename(columns={0: 'data', 1: 'task_info'})
data_task

 
data_task2 = data_task['task_info'].str.split('-').apply(pd.Series).rename(columns={0:'rp', 1:'f_metric', 2:'prompt_variation',3:'dem'})
data_task2

 
df_wide = pd.concat([filtered_df, data_task, data_task2],axis=1)
cols=['model','data','rp','f_metric','prompt_variation','dem', 'run_ts','run_id','acc','precision','recall','di','statistical_parity_diff','equal_opp_diff','error_rate_diff','error_rate_ratio','false_omission_rate_diff']
df_wide = df_wide[cols]
df_wide=df_wide.replace('claude-3-7-sonnet-20250219','claude-3-7-sonnet')
df_wide.head()


 
def get_metric(df, model, rp, data, f_metric, prompt_variation):
    return df[
        (df["model"]==model) &
        (df["rsch_prob"]==rp) &
        (df["task_dataset"]==data)&
        (df["prompt_variation"]==prompt_variation)
    ][metric].values

 
df_wide['prompt_variation'].value_counts()

 
count = df_wide.groupby(['model', 'prompt_variation'])["rp"].count()

 
count.to_csv("counts.csv")

 
df_wide = df_wide[df_wide["prompt_variation"] != "noreq"]

 
df_wide['prompt_variation'].value_counts()

 
df_wide["prompt_variation"] = df_wide["prompt_variation"].replace({
    'altmetricdetail': 'altmetricnames',
    'altmetricdetail2': 'informalgoal',
    'rephrased01': 'informaldirect',
    'rephrased03': 'altnowork',
    'rephrased05': 'verbosedetail',
    'rephrased06': 'informationalpassive',
    'rephrased10': 'passivedata'})

 
df_wide['prompt_variation'].unique()

 
summary = (
    df_wide.groupby(["model", "rp", "prompt_variation"])
      .agg(
          mean_acc=("acc","mean"),
          mean_di=("di","mean"),
          std_acc=("acc","std"),
          std_di=("di","std")
      )
      .reset_index()
)
summary 
 
allowed_prompts = ['altmetricnames', 'informalgoal', 'nochange', 'nohow', 'nosuccess', 'informaldirect', 'altnowork', 
'verbosedetail', 'informationalpassive', 'passivedata', 'shortgoal']

results = []

for (model, rp), group in summary.groupby(["model", "rp"]):

    # baseline = original prompt
    base_row = group[group["prompt_variation"] == "original"]

    if base_row.empty:
        continue

    base_row = base_row.iloc[0]

    # baseline interval
    base_min_di  = base_row["mean_di"]  - base_row["std_di"]
    base_max_di  = base_row["mean_di"]  + base_row["std_di"]
    base_min_acc = base_row["mean_acc"] - base_row["std_acc"]
    base_max_acc = base_row["mean_acc"] + base_row["std_acc"]

    for _, row in group.iterrows():

        prompt = row["prompt_variation"]

        if prompt == "original":
            continue

        if prompt not in allowed_prompts:
            continue

        # variation interval
        row_min_di  = row["mean_di"]  - row["std_di"]
        row_max_di  = row["mean_di"]  + row["std_di"]
        row_min_acc = row["mean_acc"] - row["std_acc"]
        row_max_acc = row["mean_acc"] + row["std_acc"]

        results.append({
            "model": model,
            "rsch_prob": rp,
            "baseline_prompt": "original",
            "comparison_prompt": prompt,
            # mean shifts
            "fairness_diff": abs(base_row["mean_di"] - row["mean_di"]),
            "accuracy_diff": abs(base_row["mean_acc"] - row["mean_acc"]),
            # overlap between two intervals
            "overlap_fair": max(0, min(base_max_di, row_max_di) - max(base_min_di, row_min_di)),
            "overlap_acc": max(0, min(base_max_acc, row_max_acc) - max(base_min_acc, row_min_acc)),
            # baseline interval length 
            "len_fair_base": base_max_di - base_min_di,
            "len_acc_base": base_max_acc - base_min_acc})

sens_df = pd.DataFrame(results)

 
sens_df['comparison_prompt'].unique()

 
sens_df['final_overlap_fair'] = sens_df['overlap_fair'] / sens_df['len_fair_base']
sens_df['final_overlap_acc'] = sens_df['overlap_acc'] / sens_df['len_acc_base']
sens_df

 
heatmap_data = sens_df.pivot_table(
    index=["rsch_prob", "model"],
    columns="comparison_prompt",
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
plt.title("Fairness Sensitivity Across Prompt Variations")
plt.xlabel("Comparison Prompts")
plt.ylabel("Research Problem | Model")

plt.savefig("prompt_sensitivity_fair_overlap.png", dpi=200, bbox_inches="tight")


heatmap_data = sens_df.pivot_table(
    index=["rsch_prob", "model"],
    columns="comparison_prompt",
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

plt.title("Accuracy Sensitivity Across Prompt Variants")
plt.xlabel("Comparison Prompts")
plt.ylabel("Research Problem | Model")
plt.savefig("prompt_sensitivity_acc_overlap.png", dpi=200, bbox_inches="tight")
