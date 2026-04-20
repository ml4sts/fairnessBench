import os
import pandas as pd
import numpy as np
from datetime import datetime
import json 
from path import PROJECT_ROOT,CSV_FILES

# loading the performance results
perf_path = PROJECT_ROOT 
result_files = [
    os.path.join(perf_path, fname)
    for fname in os.listdir(perf_path)
    if os.path.isfile(os.path.join(perf_path, fname))
]
result_list = []
for rf in result_files:
    try:
        if os.path.getsize(rf) == 0:
            print(f"Skipping empty file: {rf}")
            continue
        df = pd.read_json(rf).T
        result_list.append(df)
    except Exception as e:
        print(f"Skipping file {rf} due to error: {e}")
performance_df = pd.concat(result_list)
# Drop unsuccessful runs - keep only dict type (successful runs)
result_df_successful = performance_df[performance_df['final_score'].apply(lambda x: isinstance(x, dict))]
print(f"Total rows after filtering: {len(result_df_successful)}")
print(f"Rows dropped: {len(performance_df) - len(result_df_successful)}")

end_series = lambda s: pd.Series(s[-5:])
model_run = result_df_successful['path'].str.split('/').apply(end_series).rename(columns = 
                                            {i:c for i,c in enumerate(['model','task','run_ts'])})

model_run['run_id']= model_run.groupby(['model','task']).cumcount()
mr_keep = ['model','task','run_ts','run_id']

# extracting the final and the steps performance scores for the results to save in a csv file
exploded_score = result_df_successful['final_score'].apply(pd.Series).reset_index()
sp = exploded_score['index'].str.split('/').apply(end_series)
sp = sp.rename(columns={i: c for i, c in enumerate([ 'model', 'task', 'run_ts'])})

exploded_score = exploded_score.join(sp[['model', 'task', 'run_ts']])

exploded_score['run_id'] = exploded_score.groupby(['model', 'task']).cumcount()

cols = ['model', 'task', 'run_ts','run_id'] + [col for col in exploded_score.columns if col not in ['model', 'task', 'run_ts', 'run_id']]
exploded_score = exploded_score[cols]
exploded_score = exploded_score.drop(exploded_score.columns[4],axis=1)

# adding flake8 results to  performnce df 
flake8_df = result_df_successful[['path', 'final_flake8_score']].copy()
sps = flake8_df['path'].str.split('/').apply(end_series)
sps = sps.rename(columns={i: c for i, c in enumerate([ 'model', 'task', 'run_ts'])})
flake8_df = flake8_df.join(sps[['model', 'task', 'run_ts']])

exploded_score = exploded_score.merge(
    flake8_df[['model', 'task', 'run_ts', 'final_flake8_score']],
    on=['model', 'task', 'run_ts'],
    how='left'
)
# Function to get last 8 runs
def get_last_best8(df):
    if len(df) > 8:
        return df.sort_values('run_id').tail(8)
    else:
        return df

# Keep only last 8 successful runs per task
exp_score_filtered = exploded_score.groupby(['task','model']).apply(get_last_best8).reset_index(drop=True)
print(f"\nFinal rows after keeping last 8 per task: {len(exp_score_filtered)}")

output_file=os.path.join(CSV_FILES, 'Final_step_perfomance' + datetime.isoformat(datetime.now()) +'.csv')
exp_score_filtered.to_csv(output_file,index=False)

# loading baseline results 
result_path = '/scratch3/workspace/ayman_sandouk_uri_edu-fairness/fairnessBench/baseline_results'
result_files = [
    os.path.join(result_path, resjson)
    for resjson in os.listdir(result_path)
    if os.path.isfile(os.path.join(result_path, resjson))
]

result_list = [pd.read_json(rf).T for rf in result_files]
result_df = pd.concat(result_list)

end_series = lambda s: pd.Series(s[-4:])
model_run = result_df['path'].str.split('/').apply(end_series).rename(columns = 
                                            {i:c for i,c in enumerate(['task','run_ts'])})

model_run['run_id']= model_run.groupby(['task','run_ts']).cumcount()
mr_keep = ['task','run_ts','run_id']
exploded_score = result_df['final_score'].apply(pd.Series).reset_index()
exploded_score['score_count'] = exploded_score.groupby('index').cumcount()
sp = exploded_score['index'].str.split('/').apply(end_series)
sp = sp.rename(columns={i: c for i, c in enumerate([ 'task', 'run_ts'])})

exploded_score = exploded_score.join(sp[['task', 'run_ts']])

exploded_score['run_id'] = exploded_score.groupby(['task','run_ts']).cumcount()

cols = [ 'task', 'run_ts','run_id'] + [col for col in exploded_score.columns if col not in [ 'task', 'run_ts', 'run_id']]
exploded_score = exploded_score[cols]
exploded_score = exploded_score.drop(exploded_score.columns[3],axis=1)
cols_to_prefix = [col for col in exploded_score.columns if col not in ['task', 'run_ts', 'run_id']]
exploded_score = exploded_score.rename(
    columns={col: f'baseline_{col}' for col in cols_to_prefix}
)

output_file=os.path.join(CSV_FILES, 'Baseline_cleaned_perfomance' + datetime.isoformat(datetime.now()) +'.csv')
exploded_score.to_csv(output_file,index=False)
