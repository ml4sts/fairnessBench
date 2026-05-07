import os
import pandas as pd 
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

statlog_german_credit_data = fetch_ucirepo(id=144) 

X = statlog_german_credit_data.data.features.copy() 
y = statlog_german_credit_data.data.targets.copy()

# Mapping the decodings 
sex_ms_mapping = {
    "A91": ("male", "d/s"),  # d/s = divorced/separated , just made it smaller to take less space in the table
    "A92": ("female", "d/s/m"),
    "A93": ("male", "single"),
    "A94": ("male", "m/w"),
    "A95": ("female", "single")}

X["sex"] = X["Attribute9"].map(lambda x: sex_ms_mapping[x][0]) # Picks 0th index for sex
X['sex'] = X['sex'].map({'male': 1, 'female': 0})
y = y.replace({2: 0}) 
categorical_columns = ["Attribute1", "Attribute3", "Attribute4", "Attribute6", "Attribute7", 
                       "Attribute9", "Attribute10", "Attribute12", "Attribute14", "Attribute15",
                      "Attribute17", "Attribute19", "Attribute20"]

X_enc = pd.get_dummies(X, columns=categorical_columns)
X_enc = X_enc.replace({True: 1, False: 0}) # bool vals to 0 and 1

# Splitting to train and test: 80-20
train_X, test_X, train_y, test_y = train_test_split(X_enc, y, test_size=0.2, random_state=1)

# Saving test and train files to read in train.py script
os.chdir("../env/")

train_X.to_csv("train_X.csv")
train_y.to_csv("train_y.csv")
test_X.to_csv("test_X.csv")
test_y.to_csv("test_y.csv")
