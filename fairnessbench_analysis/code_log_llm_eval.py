import os
import pandas as pd
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns

# Loading useful dataframes
os.chdir('csv_files')
code_eval = pd.read_csv('Result_Final_code_clean2025-08-13T10:44:12.639136.csv')
log_eval = pd.read_csv('Results_Final_log_clean2025-08-13T10:44:21.146989.csv')
perf_df= pd.read_csv('Final_step_perfomance2025-08-13T10:44:02.216469.csv')
os.chdir('..')

# Removing missing rows 
code_eval= code_eval.dropna(how="any")
code_eval = code_eval.fillna(0)
log_eval= log_eval.dropna(how='any')
perf= ['acc','precision', 'recall', 'di', 'statistical_parity_diff', 
                  'equal_opp_diff', 'error_rate_diff', 'error_rate_ratio', 
                  'false_omission_rate_diff']
perf_df= perf_df.dropna(subset=perf, how='all')
perf_df = perf_df.fillna(0)

task_data_metric = code_eval['task'].str.split('_').apply(pd.Series).rename(columns ={0:'task_dataset',1:'task_metric',2:'task-dem'})
task_data_dem = task_data_metric['task-dem'].str.split('-').apply(pd.Series).rename(columns ={0:'resrch_prob',1:'dem'})
wider_code = pd.concat([code_eval, task_data_metric,task_data_dem],axis=1)
wider_cols=['model','task','task_dataset','task_metric','resrch_prob','dem','run_ts','run_id','final_flake8_score',"1. Data Collection and Processing","2. Bias Detection and Mitigation","3. Fairness Metric Selection","4. Model Selection and Training","5. Evaluation and Testing"]
wider_code = wider_code[wider_cols]

score_cols = ["1. Data Collection and Processing","2. Bias Detection and Mitigation","3. Fairness Metric Selection","4. Model Selection and Training","5. Evaluation and Testing"]
code_tall = wider_code.melt(id_vars=['model','task','task_dataset','task_metric','resrch_prob','dem','run_ts','run_id'],
                            value_vars=score_cols,var_name='score')
sns.catplot(code_tall,col='model',row='resrch_prob',x= 'task_dataset',y='value',hue='score',kind='bar').savefig('codeval')


# log eval
task_data_metric = log_eval['task'].str.split('_').apply(pd.Series).rename(columns ={0:'task_dataset',1:'task_metric',2:'task-dem'})
task_data_dem = task_data_metric['task-dem'].str.split('-').apply(pd.Series).rename(columns ={0:'resrch_prob',1:'dem'})
wider_log = pd.concat([log_eval, task_data_metric,task_data_dem],axis=1)
wider_cols=['model','task','task_dataset','task_metric','resrch_prob','dem','run_ts','run_id',"1. Model Overview",	"2. Stakeholder Identification and Fairness Definition","3. Data Collection and Processing","4. Bias Detection and Mitigation","5. Fairness Metric Selection","6. Model Selection and Training","7. Evaluation and Testing"]
wider_log = wider_log[wider_cols]
wider_log.head()

score_cols = ["1. Model Overview",	"2. Stakeholder Identification and Fairness Definition","3. Data Collection and Processing","4. Bias Detection and Mitigation","5. Fairness Metric Selection","6. Model Selection and Training","7. Evaluation and Testing"]
log_tall = wider_log.melt(id_vars=['model','task','task_dataset','task_metric','resrch_prob','dem','run_ts','run_id'],
                            value_vars=score_cols,var_name='score')

                            
os.chdir('graphs/')    
sns.catplot(log_tall,col='model',x='resrch_prob',row= 'task_dataset',y='value',hue='score',kind='bar').savefig('logval')

