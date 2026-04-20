import os
import pandas as pd
from path import FILES
 
gemma_df = pd.read_csv(FILES/'Gemma_cv.csv')
deepseek_df= pd.read_csv(FILES/'Deepseek_cv.csv')
granite_df=pd.read_csv(FILES/'Granite_cv.csv')

gemma_df['eval'] = 'gemma'
deepseek_df['eval'] = 'deepseek'
granite_df['eval'] = 'granite'

cols = ['eval', 'model', 'task'] + [c for c in gemma_df.columns if c not in ['eval', 'model', 'task']]
gemma_df = gemma_df[cols]
deepseek_df = deepseek_df[cols]
granite_df = granite_df[cols]

all_eval_cv = pd.concat([gemma_df, deepseek_df, granite_df], axis=0, ignore_index=True)

all_eval_cv.to_csv('cv_scores_evalmodel.csv',index=False)