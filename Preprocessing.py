#!/usr/bin/env python
# coding: utf-8


def check_header(df):
    """
    Description: 
    This function takes a dataframe as an input and outputs True if the dataframe contains a header and False otherwise.
    We consider the header to be a list of only string elements. Hence, we if we detect one numerical element in the dataframe's columns then
    we consider that the dataframe has no header.
    
    """
    def check_header(df):
    i = 0 #counter of numerical values
    
    for f in df.columns:
        try:
            int(f)
            i+=1
            break #if int(f) outputs no error it means that we detected a numerical value in the header
        except ValueError:
            pass
        
    if i!=0:
            return(False)
            
    return(True)





def preprocessing(f, missing_values, irrelevant_features, chars, categorical_features, header=[]):
    """
    Description:
    This function cleans the dataset by deleting irrelevant columns and replacing the missing values. Afterwards, it encodes the categorical 
    features.
    It returns a list where the first element is a numpy array of the data (without the labels), the second is a numpy array of the labels
    and the last one is the list of features+labels names.
    
    Parameters:
    - f: path of the dataset to load
    
    - missing_values: list of values that should be considered as missing, for instance: NaN,'na','?','--' etc
    They are replaced with either the mean (in the case of numerical values) or the most frequent value (in the case of string values).
    Please pass an empty list if there are no missing values in your dataset.
    
    - irrelevant_features: list of features that are not relevant for the training, for instance the 'id' of patients. These features are deleted 
    in our function. Please pass an empty list if you have no irrelevant features to declare.
    
    - chars: list of string characters to omit usually referring to separators like '\t' or ' ' respectively for tabulation and space.
    Please pass an empty list if you have no characters to declare.
    
    - categorical_features: features referring to categories, they are encoded using pandas dummies. 
    Please pass an empty list if there are no categorical features in the dataset.
    
    - header: Initialized to an empty list. If a header is given though, sets the list to columns names of the dataframe.
    If not given and if the dataset already has a header then the function works with the latter.
    Otherwise (ie no initial header and no header passed to the function's parameters), the function returns an error.
    """
    
    #libraries to import
    import pandas as pd
    import numpy as np
    
    #setting the header
    if check_header(df): #function defined above
        pass    #in this case the dataset already contains a header                             
    elif len(header)!=0:
        df.columns = header #sets the header according to the list passed in the parameters
    else:  #if no parameter "header" is passed to the function and the dataset doens't contain a header 
        return ("Error: the dataset has no header, please enter the columns names")

    
    #dropping irrelevant data
    for f in irrelevant_features:
        df = df.drop(f,axis=1)
    
    #storing columns names 
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
        df.drop(dummies.columns[0],axis=1,inplace=True)
        
    #storing the new features names
    names = df.columns

    #separating labels and data
    label = df[df.columns[-1]].to_numpy()
    data = df.drop(df.columns[-1], axis=1).to_numpy()
    
    return([data,label,names])



##Examples

##Kidney disease dataset
#f = "/Users/HP/Desktop/PROJECTS/ML/kidney_disease.csv" #path to modify
#missing_values = ["NaN","nan","\t?"]
#irrelevant_features = ["id"]
#chars = ['\t', ' ']
#categorical_features = ["pc","rbc","pcc","ba","htn","dm","cad","appet","pe","ane","classification"]
#preprocessing(f, missing_values, irrelevant_features, chars, categorical_features)

##Banknote dataset
#f = "/Users/HP/Desktop/PROJECTS/ML/data_banknote_authentication.txt"
#names = ['variance','skewness','curtosis','entropy','class']
#preprocessing(f,[],[],[],[],names)
