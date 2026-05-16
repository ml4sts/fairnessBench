# prepared by user when setting up

import os
import numpy as np
import pandas as pd 
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator


adult = fetch_ucirepo(id=2) 

X = adult.data.features.copy()
y = adult.data.targets.copy()

y["income"] = y["income"].str.rstrip('.')
y["income"] = y["income"].apply(lambda i: 1 if i == '>50K' else 0)
X["sex"] = X["sex"].apply(lambda i: 1 if i == 'Male' else 0)
X = X.replace('?', np.nan)

ad = pd.concat([X, y], axis=1)
ad = ad.dropna()

#### syn_data generator part
os.chdir("../env/")
ad.to_csv('adult_temp.csv', index=False)
input_data = "adult_temp.csv"
description_file = "adult_description.json"
synthetic_data = "adult_synthetic.csv"

categorical_attributes = {
    'workclass': True,
    'education': True,
    'marital-status': True,
    'occupation': True,
    'relationship': True,
    'race': True,
    'sex': True,
    'native-country': True,
    'income': True,}

#cat = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']

describer = DataDescriber(category_threshold=15)
describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
                                                        epsilon=0, 
                                                        k=2, 
                                                        attribute_to_is_categorical=categorical_attributes)
describer.save_dataset_description_to_file(description_file)

generator = DataGenerator()
generator.generate_dataset_in_correlated_attribute_mode(48842, description_file)
generator.synthetic_dataset.to_csv(synthetic_data, index=False)

df_syn = pd.read_csv(synthetic_data)
####

# scaling the values in cols capital-gain, capital-loss and hours-per-week into a realistic range. It preserves relative ordering while only changing the scale for respective cols.

cholesterol_scale = MinMaxScaler(feature_range=(100, 300))
df_syn['cholesterol'] = cholesterol_scale.fit_transform(df_syn[['capital-gain']])
 
bp_scale = MinMaxScaler(feature_range=(80, 180))
df_syn['blood-pressure'] = bp_scale.fit_transform(df_syn[['capital-loss']])

activity_scale = MinMaxScaler(feature_range=(0, 24))
df_syn['daily-activity-hours'] = activity_scale.fit_transform(df_syn[['hours-per-week']])

# dropping unwanted cols, we have never used these two cols ('fnlwgt', 'education-num') in any analysis so far with adult data.
df_syn = df_syn.drop(columns=['capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt', 'education-num'])

# column values mapping eg: occupation = Adm-clerical is mapped to procedure = Administrative-check
occupation_to_procedure = {'Adm-clerical': 'Administrative-check', 'Exec-managerial': 'Specialist-consultation', 
                           'Handlers-cleaners': 'Minor-procedure', 'Prof-specialty': 'Major-surgery', 
                           'Other-service': 'General-checkup', 'Sales': 'Diagnostic-test', 
                           'Craft-repair': 'Rehabilitation', 'Transport-moving': 'Emergency', 
                           'Farming-fishing': 'Preventive-screening', 'Machine-op-inspct': 'Therapy-session', 
                           'Tech-support': 'Follow-up', 'Protective-serv': 'Mental-health-consultation', 
                           'Armed-Forces': 'Chronic-disease-management', 'Priv-house-serv': 'Vaccination'}


# column values mapping eg: country = United-States is mapped to reason = General-checkup
country_to_reason = {'United-States': 'General-checkup', 'Cuba': 'Routine-follow-up', 'Jamaica': 'New-symptom', 
                     'India': 'Fever', 'Mexico': 'Cold-Flu-Cough', 'South': 'Headache-migraine', 
                     'Puerto-Rico': 'Heart-concerns', 'Honduras': 'Respiratory-issue', 'England': 'Digestive-issue', 
                     'Canada': 'Pain-related', 'Germany': 'Skin-concern', 'Iran': 'Eye', 'Philippines': 'ENT', 
                     'Italy': 'Mental-health-concerns', 'Poland': 'Stress', 'Columbia': 'Fatigue-weakness', 
                     'Cambodia': 'Weight-concerns', 'Thailand': 'Diabetes/Bloodsugar', 'Ecuador': 'BP-concern', 
                     'Laos': 'Cholesterol-lipid-concern', 'Taiwan': 'Allergy-symptoms', 'Haiti': 'Prenatal', 
                     'Portugal': 'Gynecological-concern', 'Dominican-Republic': 'Pediatrics', 
                     'El-Salvador': 'Age-related', 'France': 'Chronic-disease', 'Guatemala': 'Medication-side-effect', 
                     'China': 'Injury', 'Japan': 'Trauma', 'Yugoslavia': 'Preventive-counseling', 'Peru': 'Genetic-risks', 
                     'Outlying-US(Guam-USVI-etc)': 'Vaccination-inquiry', 'Scotland': 'Health-counseling', 
                     'Trinadad&Tobago': 'Substance-concern', 'Greece': 'Surgical-consultation', 
                     'Nicaragua': 'Specialist-referral', 'Vietnam': 'Discharge', 'Hong': 'Lab-result-discussion', 
                     'Ireland': 'Second-opinion', 'Hungary': 'Administrative-inquiry', 'Holand-Netherlands': 'Other'}

# column values mapping eg: workclass = Private is mapped to lifestyle = Sedentary
workclass_to_lifestyle = { 'Private':'Sedentary', 'Self-emp-not-inc':'Moderately-active', 'Local-gov':'Active', 
              'State-gov': 'Highly-active', 'Self-emp-inc': 'Unhealthy', 'Federal-gov': 'Healthy', 
              'Without-pay': 'At-risk'}

# column values mapping eg: education = HS-grad is mapped to insurance_provider = UnitedHealthcare
education_to_insurance_provider = {'HS-grad':'UnitedHealthcare', 'Some-college': 'Blue Cross Blue Shield', 'Bachelors': 'Aetna', 
                     'Masters': 'Cigna', 'Assoc-voc': 'Humana', '11th': 'Kaiser Permanente', 'Assoc-acdm': 'Anthem', 
                     '10th': 'Molina Healthcare', '7th-8th': 'Centene Corporation', 'Prof-school': 'Highmark Health',
                     '9th': 'Health Net', '12th':'WellCare Health Plans', 'Doctorate':'CareFirst', 
                    '5th-6th': 'Independence Blue Cross', '1st-4th': 'Fidelis Care', 'Preschool': 'Oscar Health'}

new_col_list = []

new_col_list.append(pd.Series(df_syn['occupation'].map(occupation_to_procedure),name='procedure_on_the_day'))
new_col_list.append(pd.Series(df_syn['native-country'].map(country_to_reason),name='reason_for_visit'))
new_col_list.append(pd.Series(df_syn['workclass'].map(workclass_to_lifestyle), name='lifestyle'))
new_col_list.append(pd.Series(df_syn['education'].map(education_to_insurance_provider),name='insurance_provider'))

df_syn = pd.concat([df_syn] + new_col_list, axis=1)

df_syn = df_syn.drop(columns = ['occupation', 'native-country', 'workclass', 'education'])
df_syn = df_syn.rename(columns={'age': 'patient_age', 'sex': 'gender', 'income': 'health_risk'})

cat_cols = ['marital-status', 'relationship', 'race', 'lifestyle', 
            'insurance_provider', 'procedure_on_the_day', 'reason_for_visit']


df_syn_enc = pd.get_dummies(df_syn, columns=cat_cols)
df_syn_enc = df_syn_enc.astype(int)

adX = df_syn_enc.drop(columns=['health_risk'])   
ady = df_syn_enc['health_risk']

train_X, test_X, train_y, test_y = train_test_split(adX, ady, test_size=0.2, random_state=1)

#os.chdir("../env/")
train_X.to_csv("train_X.csv")
train_y.to_csv("train_y.csv")
test_X.to_csv("test_X.csv")
test_y.to_csv("test_y.csv")