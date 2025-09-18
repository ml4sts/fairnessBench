import os
import pandas as pd
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
from path import CSV_FILES,GRAPHS

# Loading useful dataframes
file = CSV_FILES/'Final_step_perfomance2025-09-18T00:48:26.025263.csv'
perf_alt = pd.read_csv(file)

# Removing missing rows 
perf= ['acc','precision', 'recall', 'di', 'statistical_parity_diff', 
                  'equal_opp_diff', 'error_rate_diff', 'error_rate_ratio', 
                  'false_omission_rate_diff']
perf_alt= perf_alt.dropna(subset=perf, how='all')
perf_alt = perf_alt.replace([np.inf, -np.inf], np.nan).fillna(0)

# expanding the task to sub columns 
task_data_metric = perf_alt['task'].str.split('_').apply(pd.Series).rename(columns ={0:'task_dataset',1:'task_metric',2:'task-dem'})
wider_code = pd.concat([perf_alt, task_data_metric],axis=1)
wider_cols=['model','task','task_dataset','task_metric','task-dem','run_ts','run_id','acc','precision','recall','di','statistical_parity_diff','equal_opp_diff','error_rate_diff','error_rate_ratio','false_omission_rate_diff','score_count']
wider_code = wider_code[wider_cols]

task_task_dem = wider_code['task-dem'].str.split('-').apply(pd.Series).rename(columns ={0:'rsch_prob',1:'dem'})
wider = pd.concat([wider_code, task_task_dem],axis=1)
cols= ['model','task','task_dataset','task_metric','task-dem','rsch_prob','dem','run_ts','run_id','acc','precision','recall','di','statistical_parity_diff','equal_opp_diff','error_rate_diff','error_rate_ratio','false_omission_rate_diff','score_count']
wider=wider[cols]

# Filtering only balance task from the dataframe
wider_balance = wider[wider['rsch_prob']=='balance']
wider_balance=wider_balance.replace('claude-3-7-sonnet-20250219','claude-3-7-sonnet')

wider_balance= wider_balance.copy()
metric_map = {
    'acc': 'acc',
    'di': 'di',
    'spd': 'statistical_parity_diff',
    'eod': 'equal_opp_diff',
    'err' : 'error_rate_ratio',
    'erd' : 'error_rate_diff',
    'ford': 'false_omission_rate_diff',
}
wider_balance.loc[:, 'task_metric_value'] = wider_balance.apply(lambda row: row[metric_map[row['task_metric']]], axis=1)

sns.set_context(context='poster',font_scale=1.0)
g=sns.relplot(data=wider_balance,x='acc',y='task_metric_value',hue='task_dataset',style='dem',row='task_metric',col='model',kind='scatter',
           aspect=1)
g.set_titles(template='{col_var}: {col_name}\n{row_var}: {row_name}')

# add horizontal lines at di=1 for the first row and a vertical line for acc=1
ax=g.axes
for i in range(len(ax)):
    for j in range(len(ax[0])):
        if i in [0,3]:  # first row (di)
            ax[i,j].axhline(y=1.0, color='black', linestyle='-.', alpha=0.3)
        elif i in [1, 2, 4, 5]:  # other fairness metrics
            ax[i,j].axhline(y=0.0, color='black', linestyle='-.', alpha=0.3)
        ax[i,j].axvline(x=1.0, color='black', linestyle='-.', alpha=0.2)

output = os.path.join(GRAPHS,'balancing_fairness.png')   
plt.savefig(output,dpi=400,bbox_inches='tight')