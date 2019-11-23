#!/usr/bin/env python
# coding: utf-8

def preprocessing(f, missing_values=['NaN'], irrelevant_features=[], chars=[], categorical_features=[]):
    
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
    str_features = [f for f in names if df[f].dtypes==object]
    for c in chars:
        for feature in str_features:
            df[feature] = df[feature].str.replace(c, '')

    #filling missing values
    for feature in names:
        if df[feature].dtypes == float:
            mean = round(df[feature].mean())
            df[feature].fillna(mean,inplace=True) #fill with the mean value over the column if it's numerical
        else:
            df[feature].fillna(df[feature].mode()[0],inplace=True) #fill with most frequent value if it's not numerical

    #encoding categorical features
    for f in categorical_features:
        dummies = pd.get_dummies(df[f],prefix=str(f))
        df = pd.concat([df,dummies],axis=1)
        df.drop(f,axis=1,inplace=True)

    #separating labels and data
    label = df[df.columns[-1]].to_numpy()
    data = df.drop(df.columns[-1], axis=1).to_numpy()
    
    return([data,label,names])




#f = "/Users/HP/Desktop/PROJECTS/ML/kidney_disease.csv"
#missing_values = ["NaN","nan","\t?"]
#irrelevant_features = ["id"]
#chars = ['\t', ' ']
#categorical_features = ["pc","rbc","pcc","ba","htn","dm","cad","appet","pe","ane","classification"]
#preprocessing(f, missing_values, irrelevant_features, chars, categorical_features)

