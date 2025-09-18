import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from path import CSV_FILES,GRAPHS

# Loading useful dataframes
file = CSV_FILES/'Final_step_perfomance2025-09-18T00:48:26.025263.csv'
res = pd.read_csv(file)

task_data_metric = res['task'].str.split('_').apply(pd.Series).rename(columns ={0:'task_dataset',1:'task_metric',2:'task-dem'})
wider_code = pd.concat([res, task_data_metric],axis=1)
wider_cols=['model','task','task_dataset','task_metric','task-dem','run_ts','run_id','acc','precision','recall','di','statistical_parity_diff','equal_opp_diff','error_rate_diff','error_rate_ratio','false_omission_rate_diff','score_count']
wider_code = wider_code[wider_cols]

wider_adrecon = wider_code[wider_code['task_dataset']=='adrecon'].reset_index(drop=True)

# % of times (per model/etc) that actually gets a final result
allmetrics = ['acc', 'precision', 'recall', 'di', 'statistical_parity_diff', 'equal_opp_diff', 
              'error_rate_diff', 'error_rate_ratio', 'false_omission_rate_diff']

wider_adrecon['all_metric_vals'] = wider_adrecon[allmetrics].notna().all(axis=1)

res = wider_adrecon.groupby('model')['all_metric_vals'].mean() * 100
res.round(2)
wider_adrecon = wider_adrecon.dropna()

# regular performance of final models 
wider_adrecon = wider_adrecon.rename(columns={'statistical_parity_diff': 'spd', 'equal_opp_diff': 'eod', 
                                              'error_rate_diff': 'erd', 'error_rate_ratio': 'err', 
                                              'false_omission_rate_diff': 'ford', 'precision': 'p', 
                                              'recall': 'r'})

allmet = ['acc', 'p', 'r', 'di', 'spd', 'eod', 'erd', 'err', 'ford']
adrec_res = (wider_adrecon.groupby(['model','task-dem'])[allmet].mean()).reset_index()
adrec_task_dem= adrec_res['task-dem'].str.split('-').apply(pd.Series).rename(columns ={0:'resrch_prob',1:'dem'})
adrec_res=pd.concat([adrec_res,adrec_task_dem],axis=1)
adrec_res=adrec_res.replace('claude-3-7-sonnet-20250219','claude-3-7-sonnet')
allmet = ['acc', 'p', 'r', 'di', 'spd', 'eod', 'erd', 'err', 'ford']

adrec_long = pd.melt(adrec_res, id_vars=['model', 'task-dem', 'resrch_prob', 'dem'], value_vars= allmet, var_name='task_metric', value_name='task_metric_value')
adrec_long['model-dem'] = adrec_long['model'] + '-' + adrec_long['dem']
adrec_long = adrec_long.rename(columns={'task_metric': 'task_metrics'})

high_good = ['acc', 'p', 'r', 'di','err']  #
low_good  = ['spd', 'eod', 'erd', 'ford']

metric_rename = {}

for m in high_good:
    metric_rename[m] = f"{m} ↑"

for m in low_good:
    metric_rename[m] = f"{m} ↓"

adrec_long['task_metric'] = adrec_long['task_metrics'].map(metric_rename)
g = sns.catplot(data=adrec_long,kind='bar',x='task_metric',y='task_metric_value',hue='model',col='dem',height=4,aspect=1.5)

output= os.path.join(GRAPHS,"adrec_allmetric.png")
plt.savefig(output,dpi=400,bbox_inches='tight')