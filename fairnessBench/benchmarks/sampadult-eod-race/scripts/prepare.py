import os
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

n_samples = 45222

income = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])

cap_gain = income + np.random.normal(0, 0.1, size=n_samples)  
cap_loss = income + np.random.normal(0, 0.1, size=n_samples) 

x1 = np.random.normal(0, 1, size=n_samples)
x2 = np.random.normal(0, 1, size=n_samples)

sex = np.random.choice(['Male', 'Female'], size=n_samples)  
race = np.random.choice(['White', 'Black', 'Asian', 'Other'], size=n_samples)
age = np.random.randint(18, 70, size=n_samples)  
marital_status = np.random.choice(['Single', 'Married', 'Divorced'], size=n_samples)  

ad_df = pd.DataFrame({'income': income, 'cap_gain': cap_gain, 'cap_loss': cap_loss, 'employment_score': x1, 
                      'education_score': x2, 'sex': sex, 'race': race, 'age': age, 
                      'marital_status': marital_status})

ad_df["race"] = ad_df["race"].apply(lambda i: 1 if i == 'White' else 0)

categorical_columns = ["marital_status", "sex"]

X_enc = pd.get_dummies(ad_df, columns=categorical_columns)
X_enc = X_enc.replace({True: 1, False: 0})

train_X, test_X, train_y, test_y = train_test_split(X_enc, X_enc['income'], test_size=0.2, random_state=1)

os.chdir("../env/")
train_X.to_csv("train_X.csv")
train_y.to_csv("train_y.csv")
test_X.to_csv("test_X.csv")
test_y.to_csv("test_y.csv")