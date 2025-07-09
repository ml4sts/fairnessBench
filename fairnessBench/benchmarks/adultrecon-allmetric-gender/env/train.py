import os
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

train_X = pd.read_csv("train_X.csv", index_col=0)
train_y = pd.read_csv("train_y.csv", index_col=0).squeeze()
test_X = pd.read_csv("test_X.csv", index_col=0)
test_y = pd.read_csv("test_y.csv", index_col=0).squeeze()

prot_attr = test_X['gender']  


# **********************************************************************************************
# In this part of the code, write and train the model on the above dataset to perform the task.
# This part should populate the variable accuracy, precision, recall, statistical_parity_diff, equal_opp_diff and false_omission_rate_diff on the model selected
# **********************************************************************************************







# ***********************************************
# End of the main training module
# ***********************************************

pred_y = model.predict(test_X)

output = pd.DataFrame({"Actual_y": test_y,
                       "Predicted_y": pred_y, 
                       "gender": prot_attr})

privileged_groups = [{'gender': 1}]
unprivileged_groups = [{'gender': 0}]

binaryLabelDataset = BinaryLabelDataset(df=output, 
                                        label_names=['Actual_y'], 
                                        protected_attribute_names=['gender'])
binaryLabelDataset_pred = binaryLabelDataset.copy()
binaryLabelDataset_pred.labels = output['Predicted_y'].values.reshape(-1, 1)

metric = ClassificationMetric(binaryLabelDataset, binaryLabelDataset_pred, 
                                  unprivileged_groups=unprivileged_groups, 
                                  privileged_groups=privileged_groups)

accuracy = metric.accuracy()
precision = metric.precision()
recall = metric.true_positive_rate()

statistical_parity_diff = metric.statistical_parity_difference()
equal_opp_diff = metric.equal_opportunity_difference()
false_omission_rate_diff = metric.false_omission_rate_difference()

print('best threshold:', best_threshold)
print('accuracy score:', accuracy)
print('precision score:', precision)
print('recall score:', recall)
print('statistical_parity_diff', statistical_parity_diff)
print('equal_opp_diff', equal_opp_diff)
print('false_omission_rate_diff', false_omission_rate_diff)
    
output.to_csv("submission.csv", index=False)