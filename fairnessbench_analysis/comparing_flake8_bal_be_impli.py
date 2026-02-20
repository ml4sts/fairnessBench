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
sns.set_context(context='poster',font_scale= 0.75)
g=sns.catplot(
    data=wider_balance,
    x="rsch_prob",        
    y="final_flake8_score",
    col="model",
    row="dataset",
    kind="bar",
    height=4,
    aspect=1
)
g.set_titles(template='{col_var}: {col_name}\n{row_var}: {row_name}')

for ax in g.axes.flat:
    ax.axhline(y=85, color='black', linestyle='--')

output= os.path.join(GRAPHS,'comparing_flake8_bal_be_impli.png')  
plt.savefig(output, dpi=300, bbox_inches='tight')