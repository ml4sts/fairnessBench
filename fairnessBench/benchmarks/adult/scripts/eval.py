import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from aif360.sklearn.metrics import statistical_parity_difference, disparate_impact_ratio, equal_opportunity_difference


# sys.path.append("../env")
# SR: the submission.csv we are using here contains y_test, y_pred and protected_attribute columns and we'll use these to calculate any fair/ performance metric

def get_score(): # needs to be fixed as per location of submission.csv file 
    submission_path = "submission.csv"
    submission = pd.read_csv(submission_path)
    
    test_y = submission['Actual_y']
    pred_y = submission['Predicted_y']
    prot_attr = submission['sex']
    
    # SR: Calculate performance metrics
    accuracy = accuracy_score(test_y, pred_y)
    
    # Calculate fairness metrics
    di = disparate_impact_ratio(y_true=test_y, y_pred=pred_y, prot_attr=prot_attr, priv_group=1)
    
    performace_df = pd.DataFrame({
        'acc': [accuracy],
        'disparate_impact': [di]})
    return performace_df

if __name__ == "__main__":
    print(get_score())
    