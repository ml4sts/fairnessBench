# prepared by user when setting up

import os
import numpy as np
import pandas as pd 
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

adult = fetch_ucirepo(id=2) 

X = adult.data.features.copy()
y = adult.data.targets.copy()

y["income"] = y["income"].str.rstrip('.')
y["income"] = y["income"].apply(lambda i: 1 if i == '>50K' else 0)
X["race"] = X["race"].apply(lambda i: 1 if i == 'White' else 0)
X = X.replace('?', np.nan)

X = X.rename(columns={'age': 'patt2', 'workclass': 'feat1', 'education': 'feat2', 'marital-status': 'patt3', 'occupation': 'feat3','relationship': 'patt4', 'race': 'patt1', 'sex': 'patt5', 'capital-gain': 'feat4', 'capital-loss':'feat5', 'hours-per-week': 'feat6', 'native-country': 'feat7'})

y=y.rename(columns={'income':'target'})

categorical_columns = ["feat1", "feat2", "patt3", "feat3", "patt4", "patt5", "feat7"]

X_enc = pd.get_dummies(X, columns=categorical_columns)
X_enc = X_enc.replace({True: 1, False: 0})


train_X, test_X, train_y, test_y = train_test_split(X_enc, y, test_size=0.2, random_state=1)

os.chdir("../env/")
train_X.to_csv("train_X.csv")
train_y.to_csv("train_y.csv")
test_X.to_csv("test_X.csv")
test_y.to_csv("test_y.csv")

# feat1 workclass	cat
# feat2 education	cat
# patt1 (race)	cat
# patt2 (age)	num
# patt3 (ms)	cat
# patt4 (relationship)	cat
# patt5 (sex)	cat
# feat3 occupation	cat
# feat4 cap gain	num
# feat5 cap loss	num
# feat6 hours-per-week	num
# feat7 native country	cat
# target income bin cat