import os
import pandas as pd
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
from path import CSV_FILES,GRAPHS

# Loading useful dataframes
file= CSV_FILES/'Final_step_perfomance2026-02-07T02:23:17.201613.csv'
perf_alt = pd.read_csv(file)

# Removing missing rows 
perf= ['acc','precision', 'recall', 'di', 'statistical_parity_diff', 
                  'equal_opp_diff', 'error_rate_diff', 'error_rate_ratio', 
                  'false_omission_rate_diff']
#perf_alt= perf_alt.dropna(subset=perf, how='all')
perf_alt = perf_alt.replace([np.inf, -np.inf], np.nan).fillna(0)

# expanding the task to sub columns 
task_data_metric = perf_alt['task'].str.split('_').apply(pd.Series).rename(columns ={0:'task_dataset',1:'task_metric',2:'task-dem'})
wider_code = pd.concat([perf_alt, task_data_metric],axis=1)
wider_cols=['model','task','task_dataset','task_metric','task-dem','run_ts','run_id','acc','precision','recall','di','statistical_parity_diff','equal_opp_diff','error_rate_diff','error_rate_ratio','false_omission_rate_diff']
wider_code = wider_code[wider_cols]

# Filtering only DI from the dataframe
wider_di = wider_code[wider_code['task_metric']=='di']
wider_DI = (
    wider_di.groupby(['task_dataset','task-dem'])[['di','acc']].mean().reset_index()
)
dem_df= wider_di['task-dem'].str.split('-').apply(pd.Series).rename(columns ={0:'resrch_prob',1:'dem'})
wider_DI=pd.concat([wider_di,dem_df],axis=1)
wider_DI=wider_DI.replace('claude-3-7-sonnet-20250219','claude-3-7-sonnet')

# ploting the scatter plot for di vs acc 
sns.set_context(context='poster',font_scale=0.8)
g=sns.relplot(data=wider_DI,x='acc',y='di',hue='task_dataset',row='resrch_prob',style='dem',col='model',kind='scatter',aspect=1)
g.set_titles(template='{col_var}: {col_name}\n{row_var}: {row_name}')

# adding horizontal lines at di=1 to each cell plot 
for i, ax in enumerate(g.axes.flat):
    ax.axhline(y=1.0, color='black', linestyle='-.', alpha=0.2)
    ax.axvline(x=1.0, color='black', linestyle='-.', alpha=0.2)    
# saving the plot 
output= os.path.join(GRAPHS,'di_vs_acc_scatter.png')  
plt.savefig(output, dpi=300, bbox_inches='tight')


# checking SPD vs Acc 
wider_spd = wider_code[wider_code['task_metric']=='spd']
wider_SPD = (
    wider_spd.groupby(['task_dataset','task-dem'])[['statistical_parity_diff','acc']].mean().reset_index()
)
dem_df= wider_spd['task-dem'].str.split('-').apply(pd.Series).rename(columns ={0:'resrch_prob',1:'dem'})
wider_SPD=pd.concat([wider_spd,dem_df],axis=1)
wider_SPD=wider_SPD.replace('claude-3-7-sonnet-20250219','claude-3-7-sonnet')
# ploting the scatter plot for spd vs acc 
sns.set_context(context='poster',font_scale=0.8)
g=sns.relplot(data=wider_SPD,x='acc',y='statistical_parity_diff',hue='task_dataset',row='resrch_prob',style='dem',col='model',kind='scatter',aspect=1)
g.set_titles(template='{col_var}: {col_name}\n{row_var}: {row_name}')
# adding horizontal lines at di=1 to each cell plot 
for i, ax in enumerate(g.axes.flat):
    ax.axhline(y=0.0, color='black', linestyle='-.', alpha=0.2)
    ax.axvline(x=1.0, color='black', linestyle='-.', alpha=0.2)    
# saving the plot 
output= os.path.join(GRAPHS,'spd_vs_acc_scatter.png')  
plt.savefig(output, dpi=300, bbox_inches='tight')


# checking EOD vs Acc 
wider_eod = wider_code[wider_code['task_metric']=='eod']
wider_EOD = (
    wider_eod.groupby(['task_dataset','task-dem'])[['equal_opp_diff','acc']].mean().reset_index()
)
dem_df= wider_eod['task-dem'].str.split('-').apply(pd.Series).rename(columns ={0:'resrch_prob',1:'dem'})
wider_EOD=pd.concat([wider_eod,dem_df],axis=1)
wider_EOD=wider_EOD.replace('claude-3-7-sonnet-20250219','claude-3-7-sonnet')
# ploting the scatter plot for eod vs acc 
sns.set_context(context='poster',font_scale=0.8)
g=sns.relplot(data=wider_EOD,x='acc',y='equal_opp_diff',hue='task_dataset',row='resrch_prob',style='dem',col='model',kind='scatter',aspect=1)
g.set_titles(template='{col_var}: {col_name}\n{row_var}: {row_name}')
# adding horizontal lines at di=1 to each cell plot 
for i, ax in enumerate(g.axes.flat):
    ax.axhline(y=0.0, color='black', linestyle='-.', alpha=0.2)
    ax.axvline(x=1.0, color='black', linestyle='-.', alpha=0.2)    
# saving the plot 
output= os.path.join(GRAPHS,'eod_vs_acc_scatter.png')  
plt.savefig(output, dpi=300, bbox_inches='tight')