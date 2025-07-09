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
X["sex"] = X["sex"].apply(lambda i: 1 if i == 'Male' else 0)
X = X.replace('?', np.nan)

categorical_columns = ["workclass", "education", "marital-status", "occupation", "relationship", 
                       "race", "native-country"]

X_enc = pd.get_dummies(X, columns=categorical_columns)
X_enc = X_enc.replace({True: 1, False: 0}) # bool vals to 0 and 1

train_X, test_X, train_y, test_y = train_test_split(X_enc, y, test_size=0.2, random_state=1)

os.chdir("../env/")
train_X.to_csv("train_X.csv")
train_y.to_csv("train_y.csv")
test_X.to_csv("test_X.csv")
test_y.to_csv("test_y.csv")