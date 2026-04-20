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


# Removing missing rows fairnessBench
code_eval= code_eval.dropna(how="any")

task_data_metric = code_eval['task'].str.split('_').apply(pd.Series).rename(columns ={0:'task_dataset',1:'task_metric',2:'task-dem'})
task_data_dem = task_data_metric['task-dem'].str.split('-').apply(pd.Series).rename(columns ={0:'resrch_prob',1:'dem'})
wider_code = pd.concat([code_eval, task_data_metric,task_data_dem],axis=1)
wider_cols=['model','task','task_dataset','task_metric','resrch_prob','dem','run_ts','run_id','total_llm_score',"1. Data Collection and Processing","2. Bias Detection and Mitigation","3. Fairness Metric Selection","4. Model Selection and Training","5. Evaluation and Testing"]
wider_code = wider_code[wider_cols]

# filtering the adult dataset and the di task_metric
adult= wider_code[wider_code['task_dataset']=='adult']
adult_di=adult[adult['task_metric']=='di']

long_df = adult_di.melt(
    id_vars=['model','task_dataset','task_metric','resrch_prob','dem'],  
    value_vars=[
        '1. Data Collection and Processing',
        '2. Bias Detection and Mitigation',
        '3. Fairness Metric Selection',
        '4. Model Selection and Training',
        '5. Evaluation and Testing'
    ],
    var_name='rubric_section',
    value_name='score'
)

sns.set_context(context='poster',font_scale=1.0)
plt.figsize=(16,12)
m=sns.catplot(
    data=long_df,       
    x="rubric_section",        
    y="score",                 
    hue="model",               
    col="resrch_prob", 
    row='dem',      
    kind="bar",
    aspect=2
)
m.set_titles(template='{col_var}: {col_name}\n{row_var}: {row_name}')
ax=m.axes
ax = m.axes
for ax in m.axes.flatten():
    plt.setp(ax.get_xticklabels(), rotation=30)
    ax.axhline(y=4.0, color='black', linestyle='-.', alpha=0.3)

output = os.path.join(GRAPHS, 'adult_di_code_llm_eval.png')
plt.savefig(output, dpi=400 , bbox_inches='tight')