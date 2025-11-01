# prepared by user when setting up

import os
import numpy as np
import pandas as pd 
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

adult = fetch_ucirepo(id=2) 

X = adult.data.features.copy()
y = adult.data.targets.copy()

y["income"] = y["income"].str.rstrip('.')
y["income"] = y["income"].apply(lambda i: 1 if i == '>50K' else 0)
X["race"] = X["race"].apply(lambda i: 1 if i == 'White' else 0)
X = X.replace('?', np.nan)
#print((X.head()))
cholesterol = MinMaxScaler(feature_range=(100, 300))
X['cholesterol'] = cholesterol.fit_transform(X[['capital-gain']])

bp = MinMaxScaler(feature_range=(80, 180))
X['blood-pressure'] = bp.fit_transform(X[['capital-loss']])

activity = MinMaxScaler(feature_range=(0, 24))
X['daily-activity-hours'] = activity.fit_transform(X[['hours-per-week']])

X.drop(['capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt'], axis=1, inplace=True)
#print((X.head()))
#print((X.columns))

X.rename(columns={'age': 'patient-age', 'workclass': 'lifestyle', 'education': 'insurance_provider', 'occupation': 'pocedure_on_the_visite', 'sex':'gender', 'native-country': 'reason_for_visit'}, inplace=True)
y.rename(columns={'income': 'health-risk'}, inplace = True)

# print((X.columns))
# print((y.columns))

categorical_columns = ["lifestyle", "insurance_provider", "marital-status", "pocedure_on_the_visite", "relationship", 
                       "gender", "reason_for_visit"]

X_enc = pd.get_dummies(X, columns=categorical_columns)
X_enc = X_enc.replace({True: 1, False: 0})

train_X, test_X, train_y, test_y = train_test_split(X_enc, y, test_size=0.2, random_state=1)

os.chdir("../env/")
train_X.to_csv("train_X.csv")
train_y.to_csv("train_y.csv")
test_X.to_csv("test_X.csv")
test_y.to_csv("test_y.csv")