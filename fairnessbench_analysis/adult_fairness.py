import os
import pandas as pd
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns

# Loading useful dataframes
os.chdir('csv_files')
perf_alt = pd.read_csv('Final_step_perfomance2025-08-13T08:50:41.399910.csv')
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

# Filtering only adult dataset from the dataframe
wider_adult = wider_code[wider_code['task_dataset']=='adult']
fairness_metrics= ['di','error_rate_ratio','statistical_parity_diff','equal_opp_diff','error_rate_diff','false_omission_rate_diff']
wider_ADULT = (
    wider_adult.groupby(['model','task-dem','task_metric'])[fairness_metrics].mean()
).reset_index()
ad_df= wider_ADULT['task-dem'].str.split('-').apply(pd.Series).rename(columns ={0:'resrch_prob',1:'dem'})
wider_ADULT=pd.concat([wider_ADULT,ad_df],axis=1)
wider_ADULT=wider_ADULT.replace('claude-3-7-sonnet-20250219','claude-3-7-sonnet')

metric_map = {
    'acc': 'acc',
    'di': 'di',
    'spd': 'statistical_parity_diff',
    'eod': 'equal_opp_diff',
    'err' : 'error_rate_ratio',
    'erd' : 'error_rate_diff',
    'ford': 'false_omission_rate_diff',
}
wider_ADULT.loc[:, 'task_metric_value'] = wider_ADULT.apply(lambda row: row[metric_map[row['task_metric']]], axis=1)

sns.set_context(context='poster',font_scale=1.0)
g=sns.catplot(data=wider_ADULT,x='resrch_prob',y='task_metric_value',hue='dem',row='task_metric',col='model',kind='bar'
           ,aspect=1)

g.set_titles(template='{col_var}: {col_name}\n{row_var}: {row_name}')
# adding  horizontal lines at di=1 to each cell plot 
ax=g.axes
for i in range(4):
    ax[0,i].axhline(y=1.0, color='black', linestyle='-.', alpha=0.3)
    ax[1,i].axhline(y=0.0, color='black', linestyle='-.', alpha=0.6)
    ax[2,i].axhline(y=0.0, color='black', linestyle='-.', alpha=0.6)
    ax[3,i].axhline(y=1.0, color='black', linestyle='-.', alpha=0.3)
    ax[4,i].axhline(y=0.0, color='black', linestyle='-.', alpha=0.6)
    ax[5,i].axhline(y=0.0, color='black', linestyle='-.', alpha=0.6)

os.chdir('graphs/')    
plt.savefig('adult_fairness.png',dpi=300, bbox_inches='tight')
