import os
import pandas as pd 
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

default_of_credit_card_clients = fetch_ucirepo(id=350) 
  
X = default_of_credit_card_clients.data.features.copy() 
y = default_of_credit_card_clients.data.targets.copy()
 
X.rename(columns={'X2':'gender', 'X4':'marital_status'}, inplace=True)
X["gender"] = X["gender"].apply(lambda i: 1 if i == 1 else 0) 

# SR: splitting to train and test: 80-20
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)

os.chdir("../env/")
train_X.to_csv("train_X.csv")
train_y.to_csv("train_y.csv")
test_X.to_csv("test_X.csv")
test_y.to_csv("test_y.csv")