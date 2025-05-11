import os
import numpy as np
import pandas as pd 
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, RandomOverSampler

adult = fetch_ucirepo(id=2) 

X = adult.data.features.copy()
y = adult.data.targets.copy()
data = pd.concat([X, y], axis=1)
data["income"] = data["income"].str.rstrip('.')
data["income"] = data["income"].apply(lambda i: 1 if i == '>50K' else 0)
data["race"] = data["race"].apply(lambda i: 1 if i == 'White' else 0)
data = data.replace('?', np.nan)


protected_attributes = ['sex','race', 'age', 'marital-status', 'relationship']
feature_cat = ['workclass','education', 'occupation','native-country']
features = protected_attributes+feature_cat
target = 'income'

data['all_p_att'] = data[protected_attributes].apply(lambda p_att: '_'.join(p_att.astype(str)), axis=1)

ros = RandomOverSampler()  

X_resampled, y_resampled = ros.fit_resample(data.drop(columns='all_p_att'), data['all_p_att'])
resampled_df = X_resampled
resampled_df['all_p_att'] = y_resampled
resampled_df = resampled_df.drop(['age','sex','marital-status', 'race', 'relationship'], axis = 1)

p_cols = ['sex', 'race', 'age', 'marital-status', 'relationship']
final_df = resampled_df['all_p_att'].str.split('_', expand=True)
resampled_df[p_cols] = final_df
resampled_df = resampled_df.drop(columns=['all_p_att'])

categorical_columns = ["workclass", "education", "occupation", "sex", 
                       "native-country", "marital-status", "relationship"]
#print(resampled_df.columns)
X_enc = pd.get_dummies(resampled_df, columns=categorical_columns)
X_enc = X_enc.replace({True: 1, False: 0})

train_X, test_X, train_y, test_y = train_test_split(X_enc, X_enc['income'], test_size=0.2, random_state=1)

os.chdir("../env/")
train_X.to_csv("train_X.csv")
train_y.to_csv("train_y.csv")
test_X.to_csv("test_X.csv")
test_y.to_csv("test_y.csv")