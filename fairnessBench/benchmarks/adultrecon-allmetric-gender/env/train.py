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

prot_attr = test_X['gender']  

output = pd.DataFrame({"Actual_y": test_y.values.ravel(), "Predicted_y": pred_y, "gender" : prot_attr})

privileged_groups = [{'gender': 1}]
unprivileged_groups = [{'gender': 0}]

binaryLabelDataset = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, 
                                        df=output, label_names=['Actual_y'], 
                                        protected_attribute_names=['gender'])

binaryLabelDataset_pred = binaryLabelDataset.copy()
binaryLabelDataset_pred.labels = output['Predicted_y'].values.reshape(-1, 1)

metric = ClassificationMetric(binaryLabelDataset, binaryLabelDataset_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)

accuracy = metric.accuracy()
precision = metric.precision()
recall = metric.true_positive_rate()

disparate_impact = metric.disparate_impact()
statistical_parity_diff = metric.statistical_parity_difference()
equal_opp_diff = metric.equal_opportunity_difference()
error_rate_diff = metric.error_rate_difference()
error_rate_ratio = metric.error_rate_ratio()
false_omission_rate_diff = metric.false_omission_rate_difference()

print('accuracy score:', accuracy)
print('precision score:', precision)
print('recall score:', recall)
print('disparate_impact', disparate_impact)
print('statistical_parity_diff', statistical_parity_diff)
print('equal_opp_diff', equal_opp_diff)
print('error_rate_diff', error_rate_diff)
print('error_rate_ratio', error_rate_ratio)
print('false_omission_rate_diff', false_omission_rate_diff)

output.to_csv("submission.csv", index=False)  

