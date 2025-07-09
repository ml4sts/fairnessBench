import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.linear_model import LogisticRegression


train_X = pd.read_csv("train_X.csv", index_col=0)
train_y = pd.read_csv("train_y.csv", index_col=0)
test_X = pd.read_csv("test_X.csv", index_col=0)
test_y = pd.read_csv("test_y.csv", index_col=0)

lr = LogisticRegression(max_iter=1000) 
lr.fit(train_X, train_y) 

pred_y = lr.predict(test_X) 

prot_attr = test_X['race']  

output = pd.DataFrame({"Actual_y": test_y.values.ravel(), "Predicted_y": pred_y, "race" : prot_attr})

privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]

binaryLabelDataset = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, 
                                        df=output, label_names=['Actual_y'], 
                                        protected_attribute_names=['race'])

binaryLabelDataset_pred = binaryLabelDataset.copy()
binaryLabelDataset_pred.labels = output['Predicted_y'].values.reshape(-1, 1)

metric = ClassificationMetric(binaryLabelDataset, binaryLabelDataset_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)

accuracy = metric.accuracy()

false_omission_rate_diff = metric.false_omission_rate_difference()

print('accuracy score:', accuracy)
print('false omission rate difference', false_omission_rate_diff)

output.to_csv("submission.csv", index=False)  

