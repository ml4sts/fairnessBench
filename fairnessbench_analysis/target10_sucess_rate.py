import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('csv_files')
perf_df= pd.read_csv('Final_step_perfomance2025-08-13T10:44:02.216469.csv')
os.chdir('..') 

# Removing missing rows 
perf= ['acc','precision', 'recall', 'di', 'statistical_parity_diff', 
                  'equal_opp_diff', 'error_rate_diff', 'error_rate_ratio', 
                  'false_omission_rate_diff']
perf_df= perf_df.dropna(subset=perf, how='all')
perf_df = perf_df.replace([np.inf, -np.inf], np.nan).fillna(0)

# loading baseline 
os.chdir('csv_files')
baseline_df= pd.read_csv('Baseline_cleaned_perfomance2025-08-13T10:44:21.444178.csv')
os.chdir('..')

base= ['baseline_acc','baseline_precision', 'baseline_recall', 'baseline_di', 'baseline_statistical_parity_diff', 
                  'baseline_equal_opp_diff', 'baseline_error_rate_diff', 'baseline_error_rate_ratio', 
                  'baseline_false_omission_rate_diff']
baseline_df= baseline_df.dropna(subset=base, how='all')
baseline_df= baseline_df.fillna(0)
baseline_df= baseline_df.drop(columns=['run_ts','run_id','baseline_score_count'])

# merging both dfs
merged_results= perf_df.merge(baseline_df, how='left',on=['task'])
merged_results= merged_results.dropna(how='any')
# rearranging cols 
task_data_metric = merged_results['task'].str.split('_').apply(pd.Series).rename(columns ={0:'task_dataset',1:'task_metric',2:'task-dem'})
task_10 = task_data_metric['task-dem'].str.split('-').apply(pd.Series).rename(columns ={0:'resch_prob',1:'dem'})
wider = pd.concat([task_10, task_data_metric],axis=1)
col= ['task_dataset','task_metric','resch_prob','dem']
wider=wider[col]
clean_df = pd.concat([merged_results, wider], axis=1)

columns=['model','task','task_dataset','task_metric','resch_prob','dem','run_ts','run_id',
'acc','baseline_acc','precision','baseline_precision','recall',
'baseline_recall','di','baseline_di','statistical_parity_diff','baseline_statistical_parity_diff','equal_opp_diff',
'baseline_equal_opp_diff','error_rate_diff','baseline_error_rate_diff','error_rate_ratio','baseline_error_rate_ratio',
'false_omission_rate_diff','baseline_false_omission_rate_diff','score_count']
clean_df=clean_df[columns]

# filtering target10 task 
df= clean_df[clean_df['resch_prob'] == 'target10']

metric_map = {
    'acc': 'acc',
    'di': 'di',
    'spd': 'statistical_parity_diff',
    'eod': 'equal_opp_diff',
    'err' : 'error_rate_ratio',
    'erd' : 'error_rate_diff',
    'ford': 'false_omission_rate_diff',
}

metric_best = {
    'acc': 1,
    'di': 1,
    'spd': 0,
    'eod': 0,
    'err' :1,
    'erd' : 0,
    'ford': 0,
}

# subtract diff directions so that + is improvement and - is worse in result
metric_best_fx = {
    'acc': lambda r: r['task_metric_value'] - r['task_metric_value_baseline'],
    'di': lambda r: abs(1- r['task_metric_value_baseline'])- abs(1- r['task_metric_value']),
    'spd': lambda r: r['task_metric_value_baseline'] -  r['task_metric_value'],
    'eod': lambda r:r['task_metric_value_baseline'] -  r['task_metric_value'],
    'err' : lambda r: abs(1- r['task_metric_value_baseline'])- abs(1- r['task_metric_value']),
    'erd' : lambda r:r['task_metric_value_baseline'] -  r['task_metric_value'],
    'ford': lambda r:r['task_metric_value_baseline'] - r['task_metric_value'],
}

imp_text = {True:'improvement', False:'no improvement'}
def improvement(r):
    return imp_text[r['agent-improvement']>0]

df.loc[:,'task_metric_value'] = df.apply(lambda r: r[metric_map[r['task_metric']]],axis=1)
df.loc[:,'task_metric_value_baseline'] = df.apply(lambda r: r['baseline_'+metric_map[r['task_metric']]],axis=1)
df.loc[:,'agent-baseline'] = df.loc[:,'task_metric_value'] - df.loc[:,'task_metric_value_baseline']
df.loc[:,'agent-improvement'] = df.apply(lambda r: metric_best_fx[r['task_metric']](r),axis=1)
df.loc[:,'agent-impact'] = df.apply(improvement,axis=1)

def success(s):
    return sum(s>.1)

def total(s):
    return len(s)

def improvement(s):
    return sum(s>0)

df_improvement_stats = df.groupby(['model','task_dataset',])['agent-improvement'].agg(['mean',success,total,improvement]).reset_index()
df_improvement_stats_tall = df_improvement_stats.melt(id_vars=['model','task_dataset'],
                                                      value_vars=['total','success','improvement'],var_name='count_type',value_name='count')
sns.set_context(context='poster',font_scale = .5)
os.chdir('graphs/')
sns.catplot(df_improvement_stats_tall, col = 'model',x='task_dataset', y='count',hue='count_type',kind='bar').savefig('target10_success.png')

