# Import helpful libraries
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from aif360.sklearn.metrics import disparate_impact_ratio

# SR: we did the data reading and splitting in prepare.py file, below files are read after split done.
train_X = pd.read_csv("train_X.csv", index_col=0)
train_y = pd.read_csv("train_y.csv", index_col=0)
test_X = pd.read_csv("test_X.csv", index_col=0)
test_y = pd.read_csv("test_y.csv", index_col=0)


# SR: train_y and test_y are in the form of column vector hence ravel transforms it into a 1D array for model training. 
# SR: I added this step because the model showed a warning interms of shape while fitting data.

train_y = train_y.values.ravel()
test_y = test_y.values.ravel()

# AS: Traning and testing?? 
# AS: lr = model???
lr = LogisticRegression(max_iter=1000) # SR: model
lr.fit(train_X, train_y) # SR: training

# AS: make prediction
pred_y = lr.predict(test_X) 
pred_y = pd.DataFrame(pred_y) # SR: creating a dataframe of prediction we got from the model using test data

prot_attr = test_X['sex'] # SR: we have considered sex as protected attribute from the data 

# task is to improve score for fair metric disparate impact (optimal score is 1, worse score is 0)
disparate_impact = disparate_impact_ratio(y_true=test_y, y_pred=pred_y, prot_attr=prot_attr, priv_group=1) 
test_accuracy = accuracy_score(test_y, pred_y) # accuracy


print('accuracy score:', test_accuracy)
print('disparate impact ratio', disparate_impact)

# SR: creating a dataframe with prediction_y column, test_y column and the protected_att for using in eval.py script.
output = pd.DataFrame({
    "Actual_y": test_y,
    "Predicted_y": pred_y.values.ravel(),
    "sex" : test_X["sex"]})
output.to_csv("submission.csv", index=False) # this will be read in eval.py 
