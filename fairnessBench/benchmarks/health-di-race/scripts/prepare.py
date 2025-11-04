# prepared by user when setting up

import os
import numpy as np
import pandas as pd 
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

adult = fetch_ucirepo(id=2) 

X = adult.data.features.copy()
y = adult.data.targets.copy()

y["income"] = y["income"].str.rstrip('.')
y["income"] = y["income"].apply(lambda i: 1 if i == '>50K' else 0)
X["race"] = X["race"].apply(lambda i: 1 if i == 'White' else 0)
X = X.replace('?', np.nan)

ad = pd.concat([X, y], axis=1)
ad = ad.dropna()

cholesterol_scale = MinMaxScaler(feature_range=(100, 300))
ad['cholesterol'] = cholesterol_scale.fit_transform(ad[['capital-gain']])

bp_scale = MinMaxScaler(feature_range=(80, 180))
ad['blood-pressure'] = bp_scale.fit_transform(ad[['capital-loss']])

activity_scale = MinMaxScaler(feature_range=(0, 24))
ad['daily-activity-hours'] = activity_scale.fit_transform(ad[['hours-per-week']])

ad.drop(['capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt', 'education-num'], axis=1, inplace=True)

occupation_to_procedure = {'Adm-clerical': 'Administrative-check', 'Exec-managerial': 'Specialist-consultation', 
                           'Handlers-cleaners': 'Minor-procedure', 'Prof-specialty': 'Major-surgery', 
                           'Other-service': 'General-checkup', 'Sales': 'Diagnostic-test', 
                           'Craft-repair': 'Rehabilitation', 'Transport-moving': 'Emergency', 
                           'Farming-fishing': 'Preventive-screening', 'Machine-op-inspct': 'Therapy-session', 
                           'Tech-support': 'Follow-up', 'Protective-serv': 'Mental-health-consultation', 
                           'Armed-Forces': 'Chronic-disease-management', 'Priv-house-serv': 'Vaccination'}


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

workclass_to_lifestyle = { 'Private':'Sedentary', 'Self-emp-not-inc':'Moderately-active', 'Local-gov':'Active', 
              'State-gov': 'Highly-active', 'Self-emp-inc': 'Unhealthy', 'Federal-gov': 'Healthy', 
              'Without-pay': 'At-risk'}

education_to_insurance_provider = {'HS-grad':'UnitedHealthcare', 'Some-college': 'Blue Cross Blue Shield', 'Bachelors': 'Aetna', 
                     'Masters': 'Cigna', 'Assoc-voc': 'Humana', '11th': 'Kaiser Permanente', 'Assoc-acdm': 'Anthem', 
                     '10th': 'Molina Healthcare', '7th-8th': 'Centene Corporation', 'Prof-school': 'Highmark Health',
                     '9th': 'Health Net', '12th':'WellCare Health Plans', 'Doctorate':'CareFirst', 
                    '5th-6th': 'Independence Blue Cross', '1st-4th': 'Fidelis Care', 'Preschool': 'Oscar Health'}

ad['procedure_on_the_day'] = ad['occupation'].map(occupation_to_procedure)
ad['reason_for_visit'] = ad['native-country'].map(country_to_reason)
ad['lifestyle'] = ad['workclass'].map(workclass_to_lifestyle)
ad['insurance_provider'] = ad['education'].map(education_to_insurance_provider)

ad = ad.rename(columns={'age': 'patient_age', 'sex': 'gender', 'income': 'health_risk'})

ad.drop(['occupation', 'native-country', 'workclass', 'education'], axis=1, inplace=True)

cat_cols = ['marital-status', 'race', 'relationship', 'gender', 'lifestyle', 
            'insurance_provider', 'procedure_on_the_day', 'reason_for_visit']


ad_enc = pd.get_dummies(ad, columns=cat_cols)
ad_enc = ad_enc.astype(int)

adX = ad_enc.drop(columns=['health_risk'])   
ady = ad_enc['health_risk']

train_X, test_X, train_y, test_y = train_test_split(adX, ady, test_size=0.2, random_state=1)

os.chdir("../env/")
train_X.to_csv("train_X.csv")
train_y.to_csv("train_y.csv")
test_X.to_csv("test_X.csv")
test_y.to_csv("test_y.csv")