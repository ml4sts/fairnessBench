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

end_series = lambda s: pd.Series(s[-5:])
model_run = performance_df['path'].str.split('/').apply(end_series).rename(columns = 
                                            {i:c for i,c in enumerate(['model','task','run_ts'])})

model_run['run_id']= model_run.groupby(['model','task']).cumcount()
mr_keep = ['model','task','run_ts','run_id']

# extracting the performance scores for the results to save in a csv file
exploded_score = performance_df['final_score'].apply(pd.Series).reset_index().drop(columns=[0])
exploded_score['score_count'] = exploded_score.groupby('index').cumcount()
sp = exploded_score['index'].str.split('/').apply(end_series)
sp = sp.rename(columns={i: c for i, c in enumerate([ 'model', 'task', 'run_ts'])})

exploded_score = exploded_score.join(sp[['model', 'task', 'run_ts']])

exploded_score['run_id'] = exploded_score.groupby(['model', 'task']).cumcount()

cols = ['model', 'task', 'run_ts','run_id'] + [col for col in exploded_score.columns if col not in ['model', 'task', 'run_ts', 'run_id']]
exploded_score = exploded_score[cols]
exploded_score = exploded_score.drop(exploded_score.columns[4],axis=1)

output_file=os.path.join(CSV_FILES, 'Final_step_perfomance' + datetime.isoformat(datetime.now()) +'.csv')
exploded_score.to_csv(output_file,index=False)
# loading llm eval results 
result_path = PROJECT_ROOT
result_files = [
    os.path.join(result_path, fname)
    for fname in os.listdir(result_path)
    if os.path.isfile(os.path.join(result_path, fname))
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
result_df = pd.concat(result_list)

end_series = lambda s: pd.Series(s[-5:])
model_run = result_df['path'].str.split('/').apply(end_series).rename(columns = 
                                            {i:c for i,c in enumerate(['model','task','run_ts'])})

model_run['run_id']= model_run.groupby(['model','task']).cumcount()
mr_keep = ['model','task','run_ts','run_id']
# extracting llm code evaluation 
raw_df= result_df[["final_llm_score"]].explode('final_llm_score',)['final_llm_score'].apply(pd.Series).reset_index().drop(columns=[0])
exp_code= raw_df["raw_scores"].apply(pd.Series).drop(columns=[0])
exp_code = raw_df.join(raw_df["raw_scores"].apply(pd.Series)).drop(columns= ['raw_scores', 'justifications', 'subtotals',0])
splits = exp_code['index'].str.split('/').apply(end_series)
splits = splits.rename(columns={i: c for i, c in enumerate([ 'model', 'task', 'run_ts'])})

exp_code = exp_code.join(splits[['model', 'task', 'run_ts']])

exp_code['run_id'] = exp_code.groupby(['model', 'task']).cumcount()

cols = ['model', 'task', 'run_ts','run_id'] + [col for col in exp_code.columns if col not in ['model', 'task', 'run_ts', 'run_id']]
exp_code = exp_code[cols]
exp_code = exp_code.drop(exp_code.columns[4],axis=1)

# adding flake8 results to the code llm eval df 
flake8_df = result_df[['path', 'final_flake8_score']].copy()
sps = flake8_df['path'].str.split('/').apply(end_series)
sps = sps.rename(columns={i: c for i, c in enumerate([ 'model', 'task', 'run_ts'])})
flake8_df = flake8_df.join(sps[['model', 'task', 'run_ts']])

# merging both dfs 
exp_code = exp_code.merge(
    flake8_df[['model', 'task', 'run_ts', 'final_flake8_score']],
    on=['model', 'task', 'run_ts'],
    how='left'
)
output_file=os.path.join(CSV_FILES, 'Result_Final_code_clean' + datetime.isoformat(datetime.now()) +'.csv')
exp_code.to_csv(output_file,index=False)
# extracting log llm eval results 
raw_log= result_df[["final_log_score"]].explode('final_log_score',)['final_log_score'].apply(pd.Series).reset_index().drop(columns = [0])
exp_log= raw_log["raw_scores"].apply(pd.Series).drop(columns = [0])
exp_log = raw_log.join(raw_log["raw_scores"].apply(pd.Series)).drop(columns= ['raw_scores', 'justifications', 'subtotals',0])
exp_log = exp_log.rename(columns={"total_llm_score":"total_log_score"})
split = exp_log['index'].str.split('/').apply(end_series)
split = split.rename(columns={i: c for i, c in enumerate([ 'model', 'task', 'run_ts'])})

exp_log = exp_log.join(split[['model', 'task', 'run_ts']])

exp_log['run_id'] = exp_log.groupby(['model', 'task']).cumcount()

cols = ['model', 'task', 'run_ts','run_id'] + [col for col in exp_log.columns if col not in ['model', 'task', 'run_ts', 'run_id']]
exp_log = exp_log[cols]
exp_log = exp_log.drop(exp_log.columns[4],axis=1)

output_file=os.path.join(CSV_FILES, 'Results_Final_log_clean' + datetime.isoformat(datetime.now()) +'.csv')
exp_log.to_csv(output_file,index=False)
# loading baseline results 
result_path = '/project/pi_brownsarahm_uri_edu/ayman_uri/fairnessBench/sanity_results'
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

exploded_score = result_df[['score']].explode('score',)['score'].apply(pd.Series).reset_index().drop(columns = [0])
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
