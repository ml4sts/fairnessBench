# prepared by user when setting up

import os
import pandas as pd # Works on unity not on WSL
# import random
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
adult = fetch_ucirepo(id=2) 

X = adult.data.features.copy()
y = adult.data.targets.copy()

# with open("testX.txt", "w") as file:
#     file.write(X)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
os.chdir("../env/")
train_X.to_csv("train_X.csv", index=False)
train_y.to_csv("train_y.csv", index=False)
test_X.to_csv("test_X.csv", index=False)
test_y.to_csv("test_y.csv", index=False)
