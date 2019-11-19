#!/usr/bin/env python
# coding: utf-8
"""
Created by Imane 
"""

import pandas as pd
import numpy as np

#df1 = pd.read_csv("/Users/HP/Desktop/PROJECTS/ML/data_banknote_authentication.txt")
df = pd.read_csv("/Users/HP/Desktop/PROJECTS/ML/kidney_disease.csv")

print(df.head())


#  Data Summary

columns = list(df.columns)

print("Number of features:" ,len(df.loc[0]))
print("Number of samples:" ,len(df))
print("Features names:" ,columns)


# a dictionnary to display the type of each feature 
#this is an important information that will help us choosing the method to replace the missing values
types_dict = dict()
for col in columns:
    types_dict[col] = type(df.loc[0][col])
types_dict['rbc'] = str
types_dict




# an array showing the number of missing values in each feature
print (df.isnull().sum())


# ### Replacing missing values

from sklearn.impute import SimpleImputer

#imputer for features of numerical type: replaces each missing value "NaN" with the median
imp_num = SimpleImputer(missing_values=np.NaN, strategy='median')

#imputer for features of string type: replaces each missing value "NaN" with the most frequent value in the column
imp_str = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')

#we separate columns of str type and numerical type according to the types dictionnary defined above
for i in range(5):
    imp_num = imp_num.fit(df.loc[:,[columns[i],columns[i+1]]])  
    df.loc[:,[columns[i],columns[i+1]]] = imp_num.transform(df.loc[:,[columns[i],columns[i+1]]])
for i in range(6,9):
    imp_str = imp_str.fit(df.loc[:,[columns[i],columns[i+1]]])  
    df.loc[:,[columns[i],columns[i+1]]] = imp_str.transform(df.loc[:,[columns[i],columns[i+1]]])
for i in range(10,15):
    imp_num = imp_num.fit(df.loc[:,[columns[i],columns[i+1]]])  
    df.loc[:,[columns[i],columns[i+1]]] = imp_num.transform(df.loc[:,[columns[i],columns[i+1]]])
for i in range(16,25):
    imp_str = imp_str.fit(df.loc[:,[columns[i],columns[i+1]]])  
    df.loc[:,[columns[i],columns[i+1]]] = imp_str.transform(df.loc[:,[columns[i],columns[i+1]]])

#check if all missing values have been replaced
print(df.isnull().values.any())





