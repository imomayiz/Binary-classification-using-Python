#!/usr/bin/env python
# coding: utf-8

def preprocessing(f, missing_values=['NaN'], irrelevant_features=[], chars=[], categorical_features=[]):
    """
    Description:
    This function cleans the dataset by deleting irrelevant columns and replacing the missing values. Afterwards, it encodes the categorical 
    features.
    It returns a list where the first element is a numpy array of the data (without the labels), the second is a numpy array of the labels
    and the last one is the list of features names.
    
    Parameters:
    - f: path of the dataset to load
    - missing_values: list of values that should be considered as missing, for instance: NaN,'na','?','--' etc
    They are replaced with either the mean (in the case of numerical values) or the most frequent value (in the case of string values)
    - irrelevant_features: list of features that are not relevant for the training, for instance the 'id' of patients. These features are deleted 
    in our function
    - chars: list of string characters to omit usually referring to separators like '\t' or ' ' respectively for tabulation and space
    - categorical_features: features referring to categories, they are encoded using pandas dummies
    """
    
    #libraries to import
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer
    
    #loading the dataset in a pandas dataframe
    df = pd.read_csv(f, na_values = missing_values)

    #dropping irrelevant data
    for f in irrelevant_features:
        df = df.drop(f,axis=1)

    #storing features names
    names = df.columns 

    #remove unnecessary string characters like tabulation etc
    str_features = [f for f in names if df[f].dtypes==object] #the characters can only be present in columns of type str
    for c in chars:
        for feature in str_features:
            df[feature] = df[feature].str.replace(c, '') 

    #filling missing values
    for feature in names:
        #fills with the mean value over the column if it's numerical
        if df[feature].dtypes == float:
            mean = round(df[feature].mean())
            df[feature].fillna(mean,inplace=True) 
        #fills with most frequent value if it's not numerical
        else:
            df[feature].fillna(df[feature].mode()[0],inplace=True) 

    #encoding categorical features
    for f in categorical_features:
        dummies = pd.get_dummies(df[f],prefix=str(f))
        df = pd.concat([df,dummies],axis=1)
        df.drop(f,axis=1,inplace=True) 

    #separating labels and data and converting them to a numpy array
    label = df[df.columns[-1]].to_numpy()
    data = df.drop(df.columns[-1], axis=1).to_numpy()
    
    return([data, label, names])


##To test the function on Kidney disease dataset

#f = "/Users/HP/Desktop/PROJECTS/ML/kidney_disease.csv" #path to modify
#missing_values = ["NaN","nan","\t?"]
#irrelevant_features = ["id"]
#chars = ['\t', ' ']
#categorical_features = ["pc","rbc","pcc","ba","htn","dm","cad","appet","pe","ane","classification"]
#preprocessing(f, missing_values, irrelevant_features, chars, categorical_features)

