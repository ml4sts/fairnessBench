import os
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

ad_rec = pd.read_csv("https://raw.githubusercontent.com/socialfoundations/folktables/refs/heads/main/adult_reconstruction.csv")

ad_rec["race"] = ad_rec["race"].apply(lambda i: 1 if i == 'White' else 0)

X = ad_rec.drop(["income", "education-num"], axis =1)
y = ad_rec['income']

categorical_columns = ["workclass", "education", "marital-status", "occupation", "relationship", 
                        "gender", "native-country", "marital-status", "relationship"]

X_enc = pd.get_dummies(X, columns=categorical_columns)
X_enc = X_enc.replace({True: 1, False: 0}) 

train_X, test_X, train_y, test_y = train_test_split(X_enc, y, test_size=0.2, random_state=1)

os.chdir("../env/")
train_X.to_csv("train_X.csv")
train_y.to_csv("train_y.csv")
test_X.to_csv("test_X.csv")
test_y.to_csv("test_y.csv")