import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from aif360.sklearn.metrics import disparate_impact_ratio, statistical_parity_difference, equal_opportunity_difference

train_X = pd.read_csv("train_X.csv", index_col=0)
train_y = pd.read_csv("train_y.csv", index_col=0)
test_X = pd.read_csv("test_X.csv", index_col=0)
test_y = pd.read_csv("test_y.csv", index_col=0)


lr = LogisticRegression(max_iter=1000) 
lr.fit(train_X, train_y) 

pred_y = lr.predict(test_X) 

prot_attr = test_X['gender']  

test_accuracy = accuracy_score(test_y, pred_y) # accuracy
recall = recall_score(test_y, pred_y)
precision = precision_score(test_y, pred_y)
f1 = f1_score(test_y, pred_y)
    
disparate_impact = disparate_impact_ratio(y_true=test_y, y_pred=pred_y, prot_attr=prot_attr, priv_group=1)
statistical_parity_diff = statistical_parity_difference(y_true=test_y, y_pred=pred_y, prot_attr=prot_attr, priv_group=1)
equal_opportunity_diff = equal_opportunity_difference(y_true=test_y, y_pred=pred_y, prot_attr=prot_attr, priv_group=1)
    
print('accuracy score:', test_accuracy)
print('recall score:', recall)
print('precision score:', precision)
print('f1 score:', f1)
print('disparate impact ratio', disparate_impact)
print('statistical_parity_difference', statistical_parity_diff)
print('equal_opportunity_difference', equal_opportunity_diff)

output = pd.DataFrame({"Actual_y": test_y.values.ravel(), "Predicted_y": pred_y, "gender" : test_X["gender"]})
output.to_csv("submission.csv", index=False)  

