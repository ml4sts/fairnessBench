import sys
import os
sys.path.append("../env")
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from aif360.sklearn.metrics import statistical_parity_difference, disparate_impact_ratio, equal_opportunity_difference

# the df we are usinf here contains y test, y pred and a protected attribute


def get_score(submission_folder = "../env"): # needs to be fixed as per location of submission.csv file 
    submission_path = os.path.join(submission_folder, "submission.csv")
    submission = pd.read_csv(submission_path)
    
    test_y = submission['Actual_y']
    pred_y = submission['Predicted_y']
    prot_attr = submission['sex']
    
    # Calculate performance metrics
    accuracy = accuracy_score(test_y, pred_y)
    precision = precision_score(test_y, pred_y)
    recall = recall_score(test_y, pred_y)
    f1 = f1_score(test_y, pred_y)
    
    # Calculate fairness metrics
    sp = statistical_parity_difference(y_true=test_y, y_pred=pred_y, prot_attr=prot_attr, priv_group=1)
    di = disparate_impact_ratio(y_true=test_y, y_pred=pred_y, prot_attr=prot_attr, priv_group=1)
    eod = equal_opportunity_difference(y_true = test_y, y_pred = pred_y, prot_attr = prot_attr, priv_group=1)
    
    performace_df = pd.DataFrame({
        'acc': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1': [f1],
        'statistical_parity': [sp],
        'disparate_impact': [di],
        'equal_opportunity_difference':[eod]})
    
    return performace_df

if __name__ == "__main__":
    print(get_score())
    