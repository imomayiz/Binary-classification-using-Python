#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def check_header(df):
    """
    Author: Imane M\n
    Check if a dataframe contains a header (list of only string elements)
    If we detect one numerical element in the dataframe's columns then we consider that the dataframe has no header.
    
    Args:
        df: pandas dataframe
    
    Returns:
        True if the dataframe contains a header and False otherwise
    """
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


def preprocess(f, missing_values, irrelevant_features, chars, categorical_features, header=[]):
    """
    Author: Imane M\n
    Load, clean and normalize the dataset by deleting irrelevant columns and replacing the missing values, encode the categorical features.
    
    Args:
        f: relative path of the dataset to load
        
        missing_values: list of values that should be considered as missing, for instance: NaN,'na','?','--' etc. They are replaced with either the mean (in the case of numerical values) or the most frequent value (in the case of string values). Please pass an empty list if there are no missing values in your dataset.
        
        irrelevant_features: list of features that are not relevant for the training, for instance the 'id' of patients. These features are deleted in our function. Please pass an empty list if you have no irrelevant features to declare.
        
        chars: list of string characters to omit usually referring to separators like '\t' or ' ' respectively for tabulation and space. Please pass an empty list if you have no characters to declare.
        
        categorical_features: features referring to categories, they are encoded using pandas dummies. Please pass an empty list if there are no categorical features in the dataset.
        
        header: Initialized to an empty list. If a header is given though, sets the list to columns names of the dataframe. If not given and if the dataset already has a header then the function works with the latter. Otherwise (ie no initial header and no header passed to the function's parameters), the function returns an error.
    
    Returns:
        List composed of original data as numpy array (without the labels), label as numpy array of dimension 1, list of feature+label names.
    """
    
    #loading the dataset in a pandas dataframe
    if header != []:
        df = pd.read_csv(f, na_values = missing_values, names=header)
    else:
        df = pd.read_csv(f, na_values = missing_values)
    
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
    names = df.columns.to_list()
    
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

    #standardizing data
    numerical_features = set(df.columns) - set(categorical_features) #we only standardize the numerical data
    for f in numerical_features:
        df[f] = preprocessing.scale(df[f])


    #encoding categorical features
    for f in categorical_features:
        dummies = pd.get_dummies(df[f],prefix=str(f))
        df = pd.concat([df,dummies],axis=1)
        df.drop(f,axis=1,inplace=True)
        df.drop(dummies.columns[0],axis=1,inplace=True)


    #separating labels and data
    label = df[df.columns[-1]].to_numpy()
    data = df.drop(df.columns[-1], axis=1).to_numpy()
    
    return([data,label,names])

def split_data(data,labels,test_size=0.25):
    """
    Author: Imane M\n
    Splits data into training set and test set

    Args:
        data : preprocessed data, first element of output list of the function "preprocess"
        labels : preprocessed labels, second element of output list of the function "preprocess"
        test_size : the absolute number of the test samples, default value is 0.25

    Returns:
        np.array [data_train, data_test, labels_train, labels_test]
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size)
    return (np.array([X_train, X_test, y_train, y_test]))

def pca(datalist, n):
    """
    Author: Guillaume S\n
    Transform original data to data with n features, generated by PCA algorithm.
    
    Args:
        datalist: list returned by function "preprocess"
        n: number of components from PCA algorithm to keep (integer)

    Returns:
        Transformed data (thanks to PCA) as numpy array, label as numpy array of dimension 1, list of feature names ("feature_i").
    """
    model = PCA(n_components=n)
    data_transformed = model.fit_transform(datalist[0])
    return [data_transformed, datalist[1], ["feature_"+str(i) for i in range(1,n+1)]]


def tsne(datalist, n):
    """
    Author: Guillaume S\n
    Transform original data to data with n features, generated by t-SNE algorithm.
    
    Args:
        datalist: list returned by function "preprocess"
        n: number of components from t-SNE algorithm to keep (integer)

    Returns:
        Transformed data (thanks to t-SNE) as numpy array, label as numpy array of dimension 1, list of feature names ("feature_i").
    """
    model = TSNE(n_components=n)
    data_transformed = model.fit_transform(datalist[0])
    return [data_transformed, datalist[1], ["feature_"+str(i) for i in range(1,n+1)]]

def preprocess_main():
    """
    Author: Guillaume S\n
    Generate [data,label,names] lists for both dataset : original data, pca and t-sne for each.
    
    Returns:
        kidney original list, banknote original list, kidney pca list, banknote pca list, sidney t-sne list, banknote t-sne list.
    
    """
    f = "kidney_disease.csv"
    f2 = "data_banknote_authentication.txt"
    names = ['variance','skewness','curtosis','entropy','class']
    missing_values = ["NaN","nan","\t?"]
    irrelevant_features = ["id"]
    chars = ['\t', ' ']
    categorical_features = ["pc","rbc","pcc","ba","htn","dm","cad","appet","pe","ane","classification"]
    l = preprocess(f, missing_values, irrelevant_features, chars, categorical_features)
    l2 = preprocess(f2,[],[],[],[],names)
    l_pca = pca(l,2)
    l2_pca = pca(l2,2)
    l_tsne = tsne(l,2)
    l2_tsne = tsne(l2,2)
    #plt.scatter(l2_tsne[0][:,0],l2_tsne[0][:,1], c=l2_tsne[1])
    #plt.scatter(l2_pca[0][:,0],l2_pca[0][:,1], c=l2_pca[1])
    #plt.show()
    return l,l2,l_pca,l2_pca,l_tsne,l2_tsne

if __name__ == '__main__':
    l,l2,l_pca,l2_pca,l_tsne,l2_tsne = preprocess_main()
    print(l[0].shape)
    print("-----")
    print(l2[0].shape)


##Examples

##Kidney disease dataset
#f = "/Users/HP/Desktop/PROJECTS/ML/kidney_disease.csv" #path to modify
#missing_values = ["NaN","nan","\t?"]
#irrelevant_features = ["id"]
#chars = ['\t', ' ']
#categorical_features = ["pc","rbc","pcc","ba","htn","dm","cad","appet","pe","ane","classification"]
#preprocess(f, missing_values, irrelevant_features, chars, categorical_features)

##Banknote dataset
#f = "/Users/HP/Desktop/PROJECTS/ML/data_banknote_authentication.txt"
#names = ['variance','skewness','curtosis','entropy','class']
#preprocess(f,[],[],[],[],names)
