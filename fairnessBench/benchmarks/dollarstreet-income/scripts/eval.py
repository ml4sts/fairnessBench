import os
import ast
import pandas as pd
from torchvision import datasets

def get_score(agent_files):
    submission_path = os.path.join(agent_files, "submission.csv")
    submission_df= pd.read_csv(submission_path)

    test_df = pd.read_csv(os.path.join("../env", "images_v2_imagenet_test.csv"))

    test_df["imagenet_sysnet_id"] = test_df["imagenet_sysnet_id"].apply(ast.literal_eval)
    test_df["synset"] = test_df["imagenet_sysnet_id"].apply(lambda x: x[0])
    
    all_synsets = sorted(set(test_df["synset"]))
    synset2idx = {syn: idx for idx, syn in enumerate(all_synsets)}
    test_df["true_label_idx"] = test_df["synset"].map(synset2idx)

    test_df["predicted_label"] = submission_df.values.argmax(axis=1)

    test_df["correct"] = test_df["predicted_label"] == test_df["true_label_idx"]
    test_df["income_q"] = pd.qcut(test_df["income"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    test_df["income_group"] = test_df["income_q"].apply(lambda q: "Disadvantaged" if q == "Q1" else "Advantaged")
    group_acc = test_df.groupby("income_group")["correct"].mean() * 100
    
    return group_acc.round(2).to_dict()

if __name__ == "__main__":
    print(get_score())
    