import os
import pandas as pd
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns

# Loading useful dataframes
os.chdir('csv_files')
code_eval = pd.read_csv('Result_Final_code_clean2025-08-13T10:44:12.639136.csv')
os.chdir('..')

# Removing missing rows 
code_eval= code_eval.dropna(how="any")

task_data_metric = code_eval['task'].str.split('_').apply(pd.Series).rename(columns ={0:'task_dataset',1:'task_metric',2:'task-dem'})
task_data_dem = task_data_metric['task-dem'].str.split('-').apply(pd.Series).rename(columns ={0:'resrch_prob',1:'dem'})
wider_code = pd.concat([code_eval, task_data_metric,task_data_dem],axis=1)
wider_cols=['model','task','task_dataset','task_metric','resrch_prob','dem','run_ts','run_id','total_llm_score',"1. Data Collection and Processing","2. Bias Detection and Mitigation","3. Fairness Metric Selection","4. Model Selection and Training","5. Evaluation and Testing",'final_flake8_score']
wider_code = wider_code[wider_cols]

# Correlation between flake8 and code llm eval on claude_adult_di_erd task 
code_cols=['1. Data Collection and Processing','2. Bias Detection and Mitigation','3. Fairness Metric Selection','4. Model Selection and Training',	'5. Evaluation and Testing']
group_cols = ["model", "task_dataset", "resrch_prob", "task_metric"]  # Add 'task_dem' if needed

def flake8_corr_matrix(group):
    # Compute correlation between flake8_score and each rubric section
    corrs = [group["final_flake8_score"].corr(group[rubric]) for rubric in code_cols]
    return pd.Series(corrs, index=code_cols)

corrs = (
    wider_code.groupby(group_cols)
      .apply(flake8_corr_matrix)
      .reset_index()
)
corrs=corrs.fillna(0)

group_filter = (
    (corrs['model'] == 'claude-3-7-sonnet-20250219') &
    (corrs['task_dataset'] == 'adult') &
    (corrs['resrch_prob'] == 'balance') &
    (corrs['task_metric'] == 'erd')
)
corr_row = corrs.loc[group_filter, code_cols]

plt.figure(figsize=(8, 2))
sns.heatmap(
    corr_row.values.reshape(1, -1), 
    annot=True,
    cmap='coolwarm',
    xticklabels=code_cols,
    yticklabels=['Flake8 score']
)
plt.title("Flake8 vs Rubric Correlation (claude-3, adult, balance, erd)")

os.chdir('graphs/')    
plt.savefig('flake8_vs_code_correlation.png',bbox_inches='tight')