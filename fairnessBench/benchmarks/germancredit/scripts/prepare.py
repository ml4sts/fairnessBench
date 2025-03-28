# prepared by user when setting up

import os
import pandas as pd 
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# SR: pip install ucimlrepo, install this before running
# SR: used this link to get data : https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
statlog_german_credit_data = fetch_ucirepo(id=144) 

# SR: Include code here that  the data from the prepared files
X = statlog_german_credit_data.data.features.copy() 
y = statlog_german_credit_data.data.targets.copy()

# SR: Attribute 9: Personal status and sex (qualitative) since both cols are together, I have separated it in two diff cols as we will need demographic attributes later for fair metric calculation. below decodings are from uci repo
#       A91 : male   : divorced/separated
#       A92 : female : divorced/separated/married
#       A93 : male   : single
#       A94 : male   : married/widowed
#       A95 : female : single

#SR: mapping the decodings 
sex_ms_mapping = {
    "A91": ("male", "d/s"),  # SR: d/s = divorced/separated , just made it smaller to take less space in the table
    "A92": ("female", "d/s/m"),
    "A93": ("male", "single"),
    "A94": ("male", "m/w"),
    "A95": ("female", "single")}

X["sex"] = X["Attribute9"].map(lambda x: sex_ms_mapping[x][0]) # SR: picks 0th index for sex

#X["ms"] = X["Attribute9"].map(lambda x: sex_ms_mapping[x][1]) # SR: picks 1st index val for ms, commenting for now because we may not need this columns and also the decodings are not accurate.

categorical_columns = ["Attribute1", "Attribute3", "Attribute4", "Attribute6", "Attribute7", 
                       "Attribute9", "Attribute10", "Attribute12", "Attribute14", "Attribute15",
                      "Attribute17", "Attribute19", "Attribute20", "sex", "ms"]


# SR: function to convert categorical values to numerical values       
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

# SR: splitting to train and test: 80-20
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# SR: saving test and train files to read in train.py script
os.chdir("../env/")
train_X.to_csv("train_X.csv")
train_y.to_csv("train_y.csv")
test_X.to_csv("test_X.csv")
test_y.to_csv("test_y.csv")
