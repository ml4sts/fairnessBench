import os
import pandas as pd
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns

# Loading useful dataframes
os.chdir('csv_files')
perf_alt = pd.read_csv('Final_step_perfomance2025-08-13T10:44:02.216469.csv')
os.chdir('..')

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
os.chdir('graphs/')    
plt.savefig('di_vs_acc_scatter.png', dpi=300, bbox_inches='tight')