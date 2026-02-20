import os
import pandas as pd
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
from path import CSV_FILES,GRAPHS

# Loading useful dataframes
file = CSV_FILES/'Result_Final_code_clean2025-09-18T00:48:40.584077.csv'
code_eval = pd.read_csv(file) 


# Removing missing rows 
code_eval= code_eval.dropna(how="any")

task_data_metric = code_eval['task'].str.split('_').apply(pd.Series).rename(columns ={0:'task_dataset',1:'task_metric',2:'task-dem'})
task_data_dem = task_data_metric['task-dem'].str.split('-').apply(pd.Series).rename(columns ={0:'resrch_prob',1:'dem'})
wider_code = pd.concat([code_eval, task_data_metric,task_data_dem],axis=1)
wider_cols=['model','task','task_dataset','task_metric','resrch_prob','dem','run_ts','run_id','total_llm_score',"1. Data Collection and Processing","2. Bias Detection and Mitigation","3. Fairness Metric Selection","4. Model Selection and Training","5. Evaluation and Testing",'final_flake8_score']
wider_code = wider_code[wider_cols]

sns.set_context(context='poster',font_scale=0.8)
g = sns.relplot(data=wider_code,
                 x='final_flake8_score', 
                 y='total_llm_score',
                 col='model',
                 row='task_dataset',
                 hue='resrch_prob',
                 kind='scatter',
                 alpha=0.7,
                 height=4,
                 aspect=1)

g.set_axis_labels('Flake8 Score', 'LLM Code Score')
g.set_titles(template='{col_var}: {col_name}\n{row_var}: {row_name}')
# add horizontal lines at di=1 to each cell plot 
for i, ax in enumerate(g.axes.flat):
    ax.axhline(y=85.0, color='black', linestyle='-.', alpha=0.2)
    ax.axvline(x=85.0, color='black', linestyle='-.', alpha=0.2)

output = os.path.join(GRAPHS,'performance_flake8_code.png')   
plt.savefig(output,dpi=300)


