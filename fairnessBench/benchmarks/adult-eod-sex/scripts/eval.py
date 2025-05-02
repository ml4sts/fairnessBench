import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

def get_score(agent_files):  
    submission_path = os.path.join(agent_files, "submission.csv")
    submission = pd.read_csv(submission_path)
    
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    binaryLabelDataset = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, 
                                            df=submission, label_names=['Actual_y'], 
                                            protected_attribute_names=['sex'])
    
    binaryLabelDataset_pred = binaryLabelDataset.copy()
    binaryLabelDataset_pred.labels = submission['Predicted_y'].values.reshape(-1, 1)

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
    #equalized_odds_diff = metric.equalized_odds_difference()
    
    
    performance = {"acc": accuracy, "precision": precision, "recall": recall, "di": disparate_impact, 
                  "statistical_parity_diff": statistical_parity_diff, "equal_opp_diff": equal_opp_diff, 
                  "error_rate_diff": error_rate_diff, "error_rate_ratio" : error_rate_ratio, 
                   "false_omission_rate_diff": false_omission_rate_diff}
    return performance


if __name__ == "__main__":
    print(get_score())
    