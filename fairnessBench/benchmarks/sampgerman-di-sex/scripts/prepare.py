import os
import pandas as pd 
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

statlog_german_credit_data = fetch_ucirepo(id=144) 

X = statlog_german_credit_data.data.features.copy() 
y = statlog_german_credit_data.data.targets.copy()

#mapping the decodings 
sex_ms_mapping = {
    "A91": ("male", "d/s"),  # d/s = divorced/separated , just made it smaller to take less space in the table
    "A92": ("female", "d/s/m"),
    "A93": ("male", "single"),
    "A94": ("male", "m/w"),
    "A95": ("female", "single")}

X["sex"] = X["Attribute9"].map(lambda x: sex_ms_mapping[x][0]) #  picks 0th index for sex
X['sex'] = X['sex'].map({'male': 1, 'female': 0})
y = y.replace({2: 0}) 

# syn data generation
df_german = pd.concat([X,y], axis=1)
df_german = df_german.dropna()
os.chdir("../env/")
df_german.to_csv("german_org.csv")

input_data = "german_org.csv"
description_file = "german_description.json"
synthetic_data = "german_synthetic.csv"

threshold_value = 20

categorical_attributes = {
    'Attribute1': True,
    'Attribute3': True,
    'Attribute4': True,
    'Attribute6': True,
    'Attribute7': True,
    'Attribute9': True,
    'Attribute10': True,
    'Attribute12': True,
    'Attribute14': True,
    'Attribute15': True,
    'Attribute17': True,
    'Attribute19': True,
    'Attribute20': True}

epsilon = 1
degree_of_bayesian_network = 2
num_tuples_to_generate = 1000

describer = DataDescriber(category_threshold=threshold_value)
describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data,epsilon=epsilon,k=degree_of_bayesian_network, attribute_to_is_categorical=categorical_attributes)
describer.save_dataset_description_to_file(description_file)

generator = DataGenerator()
generator.generate_dataset_in_correlated_attribute_mode(1000,description_file=description_file)

synthetic_df = generator.synthetic_dataset
synthetic_df.to_csv(synthetic_data, index=False)

df_german = pd.read_csv(synthetic_data)

# Drop target from features
X = df_german.drop(columns=["class"])
y = df_german["class"]   

categorical_columns = ["Attribute1", "Attribute3", "Attribute4", "Attribute6", "Attribute7", 
                       "Attribute9", "Attribute10", "Attribute12", "Attribute14", "Attribute15",
                      "Attribute17", "Attribute19", "Attribute20"]

X_enc = pd.get_dummies(X, columns=categorical_columns)
X_enc = X_enc.replace({True: 1, False: 0}) # bool vals to 0 and 1

# splitting to train and test: 80-20
train_X, test_X, train_y, test_y = train_test_split(X_enc, y, test_size=0.2, random_state=1)

# saving test and train files to read in train.py script
#os.chdir("../env/")

train_X.to_csv("train_X.csv")
train_y.to_csv("train_y.csv")
test_X.to_csv("test_X.csv")
test_y.to_csv("test_y.csv")
