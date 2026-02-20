
import pandas as pd


df_final = pd.read_csv("Final_step_perfomance2026-02-07T02:23:17.201613.csv")
df_final.head()


df_baseline = pd.read_csv("Baseline_cleaned_perfomance2026-02-07T02:23:19.806782.csv")
df_baseline.head(2)

df_dollar_baseline = df_baseline[df_baseline['baseline_Advantaged'].notna() & df_baseline['baseline_Disadvantaged'].notna()]

cols_to_keep = ['task', 'run_ts', 'run_id','baseline_Advantaged', 'baseline_Disadvantaged']

df_dollar_baseline = df_dollar_baseline[cols_to_keep]
df_dollar_baseline


df_dollar_baseline['model'] = 'baseline'
df_dollar_baseline


df_final.columns


cols_to_keep = ['model', 'task', 'run_ts', 'run_id','Advantaged', 'Disadvantaged']


df_dollar_res = df_final[df_final['Advantaged'].notna() & df_final['Disadvantaged'].notna()]


df_dollar_res = df_dollar_res[cols_to_keep]
df_dollar_res.head(2)


df_dollar_res['model'].value_counts()

 
df_dollar_baseline = df_dollar_baseline.rename(columns={"baseline_Advantaged": "Advantaged", "baseline_Disadvantaged": "Disadvantaged"})


df_combined = pd.concat([df_dollar_baseline, df_dollar_res], ignore_index=True)
df_combined.head(2)

df_combined['model'] = df_combined['model'].replace('claude-3-7-sonnet-20250219', 'claude-3-7-sonnet')
df_combined


df_avg = (df_combined.groupby('model').agg(avg_adv_acc=("Advantaged", "mean"), 
avg_disadv_acc=("Disadvantaged", "mean"), std_adv_acc=("Advantaged", "std"), 
std_disadv_acc=("Disadvantaged", "std"), n_runs=("Advantaged", "count")).reset_index())
df_avg


df_avg['disparity']= df_avg['avg_adv_acc']- df_avg['avg_disadv_acc']
df_avg




