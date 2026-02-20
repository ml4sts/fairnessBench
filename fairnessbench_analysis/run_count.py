import pandas as pd
import os
from datetime import datetime

# Read and combine all result files
result_path = '/scratch3/workspace/ayman_sandouk_uri_edu-fairness/fairnessBench/final_results'
result_files = [
    os.path.join(result_path, resjson)
    for resjson in os.listdir(result_path)
    if os.path.isfile(os.path.join(result_path, resjson))
]
result_list = [pd.read_json(rf).T for rf in result_files]
result_df = pd.concat(result_list)
print(f"Total rows before filtering: {len(result_df)}")

end_series = lambda s: pd.Series(s[-5:])
sp = result_df['path'].str.split('/').apply(end_series)
sp = sp.rename(columns={i: c for i, c in enumerate(['model', 'task', 'run_ts'])})
exp_score = result_df.join(sp[['model', 'task', 'run_ts']])
exp_score['run_id'] = exp_score.groupby(['model', 'task']).cumcount()

cols = ['model', 'task', 'run_ts','run_id'] + [col for col in exp_score.columns if col not in ['model', 'task', 'run_ts', 'run_id']]
exp_score = exp_score[cols]


no_final = ~exp_score["final_score"].apply(lambda x: isinstance(x, dict))
has_final = ~no_final
no_err = exp_score["error"] == ""
time = exp_score["total_time"] > 0 

# aggregate counts per model/task
summary = (
    exp_score
    .assign(
        has_final_score = has_final,
        time_no_error = (time & no_err)
    )
    .groupby(["model", "task"])
    .agg(
        runs=("run_ts", "count"),
        sucessful_runs=("has_final_score", "sum"),
        completed_runs =('time_no_error','sum')
    )
    .reset_index()
)
task_decomp = summary['task'].str.split('_').apply(pd.Series).rename(
    columns={i:col for i,col in enumerate(['dataset','target_metric','task-dem'])})
task_dem = task_decomp['task-dem'].str.split('-').apply(pd.Series).rename(
    columns={i:col for i,col in enumerate(['research_problem','dem'])})

df = pd.concat([summary,task_decomp,task_dem],axis=1)
df
new_df = (
    df.groupby(['model','research_problem'])[
        ['runs','completed_runs','sucessful_runs']
    ]
    .sum()          
    .reset_index()
)

new_df['success_rate'] = new_df['sucessful_runs'] / new_df['runs']
#new_df['completion_rate'] = new_df['completed_runs'] / new_df['runs']
print(new_df.to_latex())

m = new_df.groupby(['model','research_problem'])['success_rate'].mean().unstack()
print(m.to_latex())
