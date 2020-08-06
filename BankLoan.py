# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 16:04:30 2020

@author: oyedeepak
"""

import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import math
from datetime import date
from scipy import stats
import pickle

# warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\Users\oyedeepak\Pictures\P 27\bank_final.csv")

df_r = df.drop(['MIS_Status', 'ChgOffDate'], axis=1)

df = df.dropna(axis=0, subset=['Name', 'State', 'City', 'Bank', 'BankState', 'RevLineCr', 'DisbursementDate'])

df['MIS_Status'] = df['MIS_Status'].replace({'P I F': 0, 'CHGOFF': 1})

currency_col = ['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']
df[currency_col] = df[currency_col].replace('[\$,]', '', regex=True).astype(float)

df['LowDoc'] = df['LowDoc'].map({'[C, 1]': np.nan, 'N': 0, 'Y':1})
df['LowDoc'].fillna(0, inplace=True)
df['LowDoc'] = df['LowDoc'].astype(int)

df['RevLineCr'] = df['RevLineCr'].replace({'N': 0, 'Y': 1, })
df['RevLineCr'] = df['RevLineCr'].replace({'0': 0, '1': 1, })

df['RevLineCr'] = np.where((df['RevLineCr'] != 0) & (df['RevLineCr'] != 1), np.nan, df.RevLineCr)


df['NewExist'] = np.where((df['NewExist'] == 2), 0, df.NewExist).astype(int)

#df['NewExist'] = df['NewExist']
df['NewExist'].value_counts() #New Business = 0, Existing Business =1

# df['NoEmp'] = np.where((df['NoEmp'] != 0), 1, df.NoEmp)


df = df.rename(columns={'CCSC': 'NAICS'})

temp = []
for item in df['NAICS']:
    if item == 0:
        temp.append(0)
    else:
        a = list(str(item))[:2]
        b = ''.join(a)
        temp.append(b)

df['NAICS'] = temp

df['NAICS'] = np.where((df['NAICS'] == 0), np.nan, df.NAICS)

df['NAICS'] = df['NAICS'].map({
    '11': 'Agriculture, Forestry, Fishing & Hunting',
    '21': 'Mining',
    '22': 'Utilities',
    '23': 'Construction',
    '31': 'Manufacturing',
    '32': 'Manufacturing',
    '33': 'Manufacturing',
    '42': 'Wholesale trade',
    '44': 'Retail trade',
    '45': 'Retail trade',
    '48': 'Transportation & Warehousing',
    '49': 'Transportation & Warehousing',
    '51': 'Information',
    '52': 'Finance & Insurance',
    '53': 'Real Estate, Rental & Leasing',
    '54': 'Professional, Scientific & Technical Services',
    '55': 'Management of Companies and Enterprises',
    '56': 'Administrative and Support, Waste Management & Remediation Services',
    '61': 'Educational',
    '62': 'Health Care & Social Assistance',
    '71': 'Arts, Entertainment & Recreation',
    '72': 'Accomodation & Food Services',
    '81': 'Other Services (except Public Administration)',
    '92': 'Public Administration'
})

from sklearn.impute import SimpleImputer

simple_NAICS = SimpleImputer(strategy='constant', fill_value='Unknown')
simple_imp = SimpleImputer(strategy='most_frequent')
df['NAICS'] = simple_NAICS.fit_transform(df[['NAICS']]).ravel()
df['RevLineCr'] = simple_imp.fit_transform(df[['RevLineCr']]).ravel()
df['MIS_Status'] = simple_imp.fit_transform(df[['MIS_Status']]).ravel()

df = df.dropna(axis=0, subset=['NewExist'])

df['FranchiseCode'] = df['FranchiseCode'].replace(1, 0)  # replace 1 with 0 since both says no code.
df['FranchiseCode'] = np.where((df.FranchiseCode != 0), 1,
                               df.FranchiseCode)  # assigning 1 to rows which have Franchise Code

int_col = ['ApprovalFY', 'Term', 'NoEmp', 'NewExist', 'CreateJob', 'RetainedJob', 'FranchiseCode', 'RevLineCr',
           'LowDoc', 'DisbursementGross', 'BalanceGross', 'MIS_Status', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']

# Removing the rows where Term = 0

df['Term'] = np.where((df['Term'] == 0), np.nan, df['Term'])
df = df.dropna(axis=0, subset=['Term'])

col_to_take = ['State', 'NAICS', 'NoEmp', 'NewExist', 'RevLineCr', 'LowDoc', 'GrAppv', 'SBA_Appv', 'Term', 'MIS_Status']

df_new = df[col_to_take]

label_col =['State','NAICS','NoEmp','NewExist','RevLineCr','LowDoc','GrAppv','SBA_Appv','Term']

from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
mapping_dict = {}
for col in label_col:
    df_new[col] = labelEncoder.fit_transform(df[col])

    le_name_mapping = dict(zip(labelEncoder.classes_,
                               labelEncoder.transform(labelEncoder.classes_)))

    mapping_dict[col] = le_name_mapping
# print(mapping_dict)

from sklearn.model_selection import train_test_split

train, test = train_test_split(df_new, test_size=0.2, random_state=101)

x_train = train.drop(['MIS_Status'], axis=1)
y_train = train['MIS_Status']

x_test = test.drop(['MIS_Status'], axis=1)
y_test = test['MIS_Status']

#from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier()

rfmodel.fit(x_train, y_train)

y_pred = rfmodel.predict(x_test)

# np.mean(y_test == y_pred)

# train_acc = np.mean(rfmodel.predict(x_train)==y_train)
# print(train_acc)

# test accuracy
# test_acc = np.mean(rfmodel.predict(x_test)==y_test)
# print(test_acc)


pickle.dump(rfmodel, open("rfmodel.pkl", "wb"))