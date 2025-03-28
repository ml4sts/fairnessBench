# Import helpful libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OrdinalEncoder

train_X = pd.read_csv("train_X.csv", index=True)
train_y = pd.read_csv("train_y.csv", index=True)
test_X = pd.read_csv("test_X.csv", index=True)
test_y = pd.read_csv("test_y.csv", index=True)


# AS: Traning and testing?? 
# AS: lr = model???
lr = LogisticRegression(max_iter=1000)
lr.fit(train_X, train_y)

# AS: make prediction
pred_y = lr.predict(test_X)


# ***********************************************
# In this part of the code, write and train the model on the above dataset to perform the task.
# This part should populate the variable train_accuracy on the model selected
# ***********************************************


# ***********************************************
# End of the main training module
# ***********************************************


train_accuracy = accuracy_score(test_y, pred_y)
print('accuracy score:', train_accuracy)

output = pd.DataFrame({ "Actual_y": test_y, "Predicted_y": pred_y, "sex" : test_X["sex"]})

output.to_csv("submission.csv", index=False) # this will be read in eval.py and other matrics will be calculated

