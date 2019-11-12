# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:33:36 2019

@author: halad
"""

###################################################################################################
######### Import the necessary libraries
import chardet
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

SEED = 17

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
###################################################################################################
FalsePositive_Alerts_Filename = 'C:/Users/halad.CORPDOM/OneDrive - Micro Focus/vertica-related/False_Positives_12_AUG.csv'
True_Alerts_Filename = 'C:/Users/halad.CORPDOM/OneDrive - Micro Focus/vertica-related/TrueAlerts_12_AUG.csv'
Combined_Filename = 'C:/Users/halad.CORPDOM/OneDrive - Micro Focus/vertica-related/Sec_Alerts_12_AUG.csv'
###################################################################################################
######### Read False Positive Alerts File
with open(FalsePositive_Alerts_Filename, 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large
    print (result)
    
FalsePoistive_Alerts_DataFrame = pd.read_csv(FalsePositive_Alerts_Filename, encoding=result['encoding'])
FalsePoistive_Alerts_DataFrame.shape

######### Read True Alerts File File
with open(True_Alerts_Filename, 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large
    print (result)

True_Alerts_DataFrame = pd.read_csv(True_Alerts_Filename, encoding=result['encoding'])
True_Alerts_DataFrame.shape

######### New columns are created and assigned the binary value to indicate te type of alert.
FalsePoistive_Alerts_DataFrame['Alert Type'] = 0
True_Alerts_DataFrame['Alert Type'] = 1

######### Ensuring the column names are correct and in order in both the data frames.
FalsePoistive_Alerts_DataFrame.columns == True_Alerts_DataFrame.columns

######### Combining the datasets
dataset = FalsePoistive_Alerts_DataFrame.append(True_Alerts_DataFrame)
dataset.shape

######### Finding the attributes whose values are null and percentage > 50
columns = dataset.columns
columns_null_50_percent = []
result_series = (dataset.isnull().sum()/dataset.shape[0])*100
for col in columns :
    if result_series[col] > 50 :
        columns_null_50_percent.append(col)
print(len(columns_null_50_percent))

######### Create a new dataset leaving the attribues the attributes whose values are null and percentage > 50
new_col_list = [ele for ele in columns if ele not in columns_null_50_percent] 
print(len(new_col_list))
new_col_list.remove('End Time')
print(len(new_col_list))

######### Create a new dataset and shuffle
new_dataset = dataset[new_col_list]

from sklearn.utils import shuffle
new_dataset_shuffled = shuffle(new_dataset)

new_dataset_shuffled['Alert Type'].value_counts()

######### Shuffled data is copied to the new csv file
new_dataset_shuffled.to_csv(Combined_Filename)

######### The NaN values are filled with meaningfull information respect to the attribues
new_dataset_shuffled['Message'].fillna('No Message', inplace=True)
new_dataset_shuffled['Category Outcome'].fillna('No Category', inplace=True)
new_dataset_shuffled['Source Address'].fillna('0.0.0.0', inplace=True)
new_dataset_shuffled['Source Zone'].fillna('No Source Zone', inplace=True)
new_dataset_shuffled['Source Zone Name'].fillna('No Source Zone Name', inplace=True)
new_dataset_shuffled['Destination Host Name'].fillna('No Destination Host Name', inplace=True)
new_dataset_shuffled['Destination Port'].fillna('No Port', inplace=True)

new_dataset_shuffled['Source Zone'] =  new_dataset_shuffled['Source Zone'].apply(lambda x: x.replace('<','').replace('/>','')) 
new_dataset_shuffled['Destination Host Name'] = new_dataset_shuffled['Destination Host Name'].apply(lambda x: x.replace('$dstHostName','No Destination Host Name'))

new_dataset_shuffled['Message']=new_dataset_shuffled['Message'].astype('category')
new_dataset_shuffled['Category Outcome']=new_dataset_shuffled['Category Outcome'].astype('category')
new_dataset_shuffled['Source Address']=new_dataset_shuffled['Source Address'].astype('category')
new_dataset_shuffled['Source Zone']=new_dataset_shuffled['Source Zone'].astype('category')
new_dataset_shuffled['Source Zone Name']=new_dataset_shuffled['Source Zone Name'].astype('category')
new_dataset_shuffled['Destination Host Name']=new_dataset_shuffled['Destination Host Name'].astype('category')
new_dataset_shuffled['Destination Port']=new_dataset_shuffled['Destination Port'].astype('category')

new_dataset_shuffled.isnull().sum()

###################################################################################################
######### Split the dataset into training and test with a ratio 80:20
X_Columns = ['Name', 'Message', 'Device Product', 'Device Vendor',
       'Device Event Category', 'Device Severity', 'Category Outcome',
       'Source Address', 'Source Zone','Source Zone Name',
       'Destination Host Name', 'Destination Port']

X = new_dataset_shuffled[X_Columns]
y = new_dataset_shuffled['Alert Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

print("X Train Shape",X_train.shape)
print("X Test Shape",X_test.shape)
print("Y Train Shape",y_train.shape)
print("Y Test Shape",y_test.shape)

###################################################################################################
######### Creating a helper function to get the scores for each encoding method.
y_pred = []
def get_score(model, X, y, X_test, y_test):
    model.fit(X, y)
    y_pred = model.predict_proba(X_test)[:,1]
    score = roc_auc_score(y_test, y_pred)
    return score

######### Creating objects for 2 classification models.
logit = LogisticRegression(random_state=SEED)
rf = RandomForestClassifier(random_state=SEED)

###################################################################################################
######### Apply One Hot Encoding
from category_encoders import OneHotEncoder
onehot_enc = OneHotEncoder(cols=X_Columns)
onehot_enc.fit(X_train, y_train)

print('Original number of features: \n', X_train.shape[1], "\n")
data_ohe_train = onehot_enc.fit_transform(X_train)
data_ohe_test = onehot_enc.transform(X_test)
print('Features after OHE: \n', data_ohe_train.shape[1])

######### Logistic Regression
onehot_logit_score = get_score(logit, data_ohe_train, y_train, data_ohe_test, y_test)
print('Logistic Regression score with One hot encoding:', onehot_logit_score)

######### Random Forest
onehot_rf_score = get_score(rf, data_ohe_train, y_train, data_ohe_test, y_test)
print('Random Forest score with One hot encoding:', onehot_logit_score)

###################################################################################################
######### Apply Hashing Encoding
from category_encoders import HashingEncoder
hashing_enc = HashingEncoder(n_components=10000,cols=X_Columns)
hashing_enc.fit(X_train, y_train)

print('Original number of features: \n', X_train.shape[1], "\n")
X_train_hashing = hashing_enc.transform(X_train.reset_index(drop=True))
X_test_hashing = hashing_enc.transform(X_test.reset_index(drop=True))
print('Features after OHE: \n', X_train_hashing.shape[1])

######### Logistic Regression
hashing_logit_score = get_score(logit, X_train_hashing, y_train, X_test_hashing, y_test)
print('Logistic Regression score with Hashing encoding:', hashing_logit_score)

######### Random Forest
hashing_rf_score = get_score(rf, X_train_hashing, y_train, X_test_hashing, y_test)
print('Random Forest score with Hashing encoding:', hashing_rf_score)
