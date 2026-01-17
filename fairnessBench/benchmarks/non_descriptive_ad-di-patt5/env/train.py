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

prot_attr = test_X['patt5']  

output = pd.DataFrame({"Actual_y": test_y.values.ravel(), "Predicted_y": pred_y, "patt5" : prot_attr})

privileged_groups = [{'patt5': 1}]
unprivileged_groups = [{'patt5': 0}]

binaryLabelDataset = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, 
                                        df=output, label_names=['Actual_y'], 
                                        protected_attribute_names=['patt5'])

binaryLabelDataset_pred = binaryLabelDataset.copy()
binaryLabelDataset_pred.labels = output['Predicted_y'].values.reshape(-1, 1)

metric = ClassificationMetric(binaryLabelDataset, binaryLabelDataset_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)

accuracy = metric.accuracy()

disparate_impact = metric.disparate_impact()

print('accuracy score:', accuracy)
print('disparate impact ratio', disparate_impact)

output.to_csv("submission.csv", index=False)  

