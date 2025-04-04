# prepared by user when setting up

import os
import pandas as pd 
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# pip install ucimlrepo, install this before running
# used this link to get data : https://archive.ics.uci.edu/dataset/2/adult

# Fetch dataset
adult = fetch_ucirepo(id=2) 

# data (as pandas dataframes) 
X = adult.data.features.copy()
y = adult.data.targets.copy()

y["income"] = y["income"].str.rstrip('.')
y["income"] = y["income"].apply(lambda i: 1 if i == '>50K' else 0)
X["sex"] = X["sex"].apply(lambda i: 1 if i == 'Male' else 0)

# SR: categorical columns in the data 
categorical_columns = ["workclass", "education", "marital-status", "occupation", "relationship", 
                       "race", "native-country"]

# SR: encoding categorical values to numerical values
def categorical_feature_encoder(df, features):
    # AS: This function takes the values of each column and changes the object values to float then to int
    """
    Encode categorical columns in a DataFrame by mapping them to unique numerical values.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the categorical columns to be encoded.
    features : list of str
        A list of column names in the DataFrame that contain categorical data to be encoded.
    
    Returns
    -------
    df : pandas.DataFrame
        The DataFrame with the specified categorical columns replaced by corresponding numerical values.
    """
    enc = {}
    for cur_f in features:
        encoder = OrdinalEncoder()
        df[cur_f] = encoder.fit_transform(df[[cur_f]]).astype(int)
        enc[cur_f] = encoder
    return df, enc

X, enc = categorical_feature_encoder(X, categorical_columns)

# SR: Splitting to test and train 
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

os.chdir("../env/")

# SR: saving test and train files to read in train.py script
train_X.to_csv("train_X.csv")
train_y.to_csv("train_y.csv")
test_X.to_csv("test_X.csv")
test_y.to_csv("test_y.csv")
