import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from aif360.sklearn.metrics import disparate_impact_ratio

# SR: we did the data reading and splitting in prepare.py file, below files are read after split done.
train_X = pd.read_csv("train_X.csv", index_col=0)
train_y = pd.read_csv("train_y.csv", index_col=0)
test_X = pd.read_csv("test_X.csv", index_col=0)
test_y = pd.read_csv("test_y.csv", index_col=0)

lr = LogisticRegression(max_iter=1000) # SR: model
lr.fit(train_X, train_y) # SR: training

pred_y = lr.predict(test_X)

prot_attr = test_X['gender'] # SR: we have considered sex as protected attribute from the data 

disparate_impact = disparate_impact_ratio(y_true=test_y, y_pred=pred_y, prot_attr=prot_attr, priv_group=1) 
test_accuracy = accuracy_score(test_y, pred_y) # accuracy

print('accuracy score:', test_accuracy)
print('disparate impact ratio', disparate_impact)

output = pd.DataFrame({"Actual_y": test_y.values.ravel(), "Predicted_y": pred_y, "sex" : test_X["gender"]})
output.to_csv("submission.csv", index=False) # this will be read in eval.py 

