import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from aif360.sklearn.metrics import disparate_impact_ratio, statistical_parity_difference, equal_opportunity_difference

def get_score(agent_files):  
    submission_path = os.path.join(agent_files, "submission.csv")
    submission = pd.read_csv(submission_path)
    
    test_y = submission['Actual_y']
    pred_y = submission['Predicted_y']
    prot_attr = submission['gender']
    
    accuracy = accuracy_score(test_y, pred_y)
    recall = recall_score(test_y, pred_y)
    precision = precision_score(test_y, pred_y)
    f1 = f1_score(test_y, pred_y)
    
    disparate_impact = disparate_impact_ratio(y_true=test_y, y_pred=pred_y, prot_attr=prot_attr, priv_group=1)
    statistical_parity_diff = statistical_parity_difference(y_true=test_y, y_pred=pred_y, prot_attr=prot_attr, priv_group=1)
    equal_opportunity_diff = equal_opportunity_difference(y_true=test_y, y_pred=pred_y, prot_attr=prot_attr, priv_group=1)
    
    performance = {"acc": accuracy, "recall": recall, "precision": precision, "f1": f1, "di": disparate_impact, "spd": statistical_parity_diff, "eod": equal_opportunity_diff}
    return performance

if __name__ == "__main__":
    print(get_score())
