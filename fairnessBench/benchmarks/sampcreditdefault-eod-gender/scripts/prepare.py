import os
import pandas as pd 
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

default_of_credit_card_clients = fetch_ucirepo(id=350) 
  
X = default_of_credit_card_clients.data.features.copy() 
y = default_of_credit_card_clients.data.targets.copy()
 
X.rename(columns={'X2':'gender', 'X4':'marital_status'}, inplace=True)
X["gender"] = X["gender"].apply(lambda i: 1 if i == 1 else 0) 

cd = pd.concat([X, y], axis=1)
cd = cd.dropna()

#### syn_data generator part
os.chdir("../env/")
cd.to_csv('credit_default_original.csv', index=False)

input_data = "credit_default_original.csv"
description_file = "credit_default_description.json"
synthetic_data = "credit_default_synthetic.csv"

categorical_attributes = {
    'gender': True,   # SEX
    'X3': True,   # EDUCATION
    'marital_status': True,   # MARRIAGE
    'X6': True,   # PAY_0
    'X7': True,   # PAY_2
    'X8': True,   # PAY_3
    'X9': True,   # PAY_4
    'X10': True,  # PAY_5
    'X11': True   # PAY_6
}

epsilon = 1
degree_of_bayesian_network = 3
num_tuples_to_generate = 30000

describer = DataDescriber(category_threshold=15)
describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
                                                        epsilon=epsilon, 
                                                        k=degree_of_bayesian_network, 
                                                        attribute_to_is_categorical=categorical_attributes)
describer.save_dataset_description_to_file(description_file)

generator = DataGenerator()
generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
generator.synthetic_dataset.to_csv(synthetic_data, index=False)

cd_syn = pd.read_csv(synthetic_data)
####

cat_cols= ['gender','X3', 'marital_status','X6','X7','X8','X9','X10', 'X11']
df_syn_enc = pd.get_dummies(cd_syn, columns=cat_cols)
df_syn_enc = df_syn_enc.astype(int)

cdX = df_syn_enc.drop(columns=['Y'])   
cdy = df_syn_enc['Y']

train_X, test_X, train_y, test_y = train_test_split(cdX, cdy, test_size=0.2, random_state=1)

os.chdir("../env/")
train_X.to_csv("train_X.csv")
train_y.to_csv("train_y.csv")
test_X.to_csv("test_X.csv")
test_y.to_csv("test_y.csv")