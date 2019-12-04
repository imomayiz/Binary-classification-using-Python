#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import warnings
import copy
warnings.filterwarnings("ignore")

def split_train_test(data, labels, test_size) :
    """
    Author: Karel Kedemos\n
    Split arrays or matrices into random train and test subsets. 
    The train and test sets keep the same percentages of each class of the original data set 
    (sklearn.model_selection.train_test_split does not keep these percentages).
    
    Args:
        data: numpy array or list of the dataset to split
        labels : numpy array or list of the labelset to split
        test_size : float between 0.0 and 1.0 that represent the proportion of the dataset to include in the test split

    Returns:
        4 arrays : the train dataset, the train labelset, the test dataset and the test labelset
    """
    n = len(labels)
    list_labels = []
    data_split = []
    labels_split = []
    for i in range(n) :
        if labels[i] not in list_labels :
            list_labels.append(labels[i])
            data_split.append([data[i]])
            labels_split.append([labels[i]])
        else :
            data_split[list_labels.index(labels[i])].append(data[i])
            labels_split[list_labels.index(labels[i])].append(labels[i])
    X_train_tot = []
    X_test_tot = []
    y_train_tot = []
    y_test_tot = []

    for i in range(len(list_labels)) :
        X_train, X_test, y_train, y_test = train_test_split(data_split[i], labels_split[i], test_size = test_size)
        X_train_tot += copy.deepcopy(X_train) 
        X_test_tot += copy.deepcopy(X_test)
        y_train_tot += copy.deepcopy(y_train)
        y_test_tot += copy.deepcopy(y_test)

    X_train_tot = np.array(X_train_tot)
    X_test_tot = np.array(X_test_tot)
    y_train_tot = np.array(y_train_tot)
    y_test_tot = np.array(y_test_tot)

    indexes_train = np.arange(len(X_train_tot))
    indexes_test = np.arange(len(X_test_tot))

    np.random.shuffle(indexes_train)
    np.random.shuffle(indexes_test)

    return X_train_tot[indexes_train], X_test_tot[indexes_test], y_train_tot[indexes_train], y_test_tot[indexes_test]

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
        if f!="class":
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


def pca(data, data_train, data_test, n):
    """
    Author: Guillaume S\n
    Transform original data to data with n features, generated by PCA algorithm.
    
    Args:
        data: numpy array with data without labels
        data_train: numpy array with data from training set without labels
        data_test: numpy array with data from testing set without labels
        n: number of components from PCA algorithm to keep (integer)

    Returns:
        Transformed data (thanks to PCA) as numpy array, transformed data from training set as numpy array, transformed data from testing set as numpy array
    """
    model = PCA(n_components=n)
    data_pca = model.fit_transform(data)
    return (data_pca, model.transform(data_train), model.transform(data_test))


def tsne(data, data_train, data_test, n):
    """
    Author: Guillaume S\n
    Transform original data to data with n features, generated by t-SNE algorithm.
    
    Args:
        data: numpy array with data without labels
        data_train: numpy array with data from training set without labels
        data_test: numpy array with data from testing set without labels
        n: number of components from t-SNE algorithm to keep (integer)

    Returns:
        Transformed data (thanks to t-SNE) as numpy array, transformed data from training set as numpy array, transformed data from testing set as numpy array
    """
    model = TSNE(n_components=n)
    return (model.fit_transform(data), model.fit_transform(data_train), model.fit_transform(data_test))

def preprocess_main(n_pca=2, n_tsne=2, test_size=0.25):
    """
    Author: Guillaume S and Imane M\n
    Generate dictionaries for both dataset with the following keys :\n
    -data : numpy array of the whole data without labels\n
    -labels : label list\n
    -names : feature names in a list\n
    -data_train : numpy array with data from training set without labels\n
    -label_train : label list from training dataset\n
    -data_test : numpy array with data from testing set without labels\n
    -label_test : label list from testing dataset\n
    -dataset_name : name of the dataset (for automatisation only)\n
    -class_possibilities : list of strings with class possibilities (for decision tree plotting only)
    
    Returns:
        kidney dictionary, banknote dictionary, kidney pca dictionary, banknote pca dictionary, sidney t-sne dictionary, banknote t-sne dictionary
    
    """
    #kidney_disease dataset
    f = "kidney_disease.csv"
    missing_values = ["NaN","nan","\t?"]
    irrelevant_features = ["id"]
    chars = ['\t', ' ']
    categorical_features = ["pc","rbc","pcc","ba","htn","dm","cad","appet","pe","ane","classification"]
    l = preprocess(f, missing_values, irrelevant_features, chars, categorical_features)

    #banknote dataset
    f2 = "data_banknote_authentication.txt"
    names = ['variance','skewness','curtosis','entropy','class']
    l2 = preprocess(f2,[],[],[],[],names)

    #data split
    X_train, X_test, y_train, y_test = split_train_test(l[0], l[1], test_size = test_size)
    X2_train, X2_test, y2_train, y2_test = split_train_test(l2[0], l2[1], test_size = test_size)
    
    #PCA
    all_pca, train_pca, test_pca = pca(l[0], X_train, X_test, n_pca)
    all_pca2, train_pca2, test_pca2 = pca(l2[0], X2_train, X2_test, n_pca)
    feature_names_pca = ["feature_"+str(i) for i in range(1,n_pca+1)]

    #TSNE
    all_tsne, train_tsne, test_tsne = tsne(l[0], X_train, X_test, n_tsne)
    all_tsne2, train_tsne2, test_tsne2 = tsne(l2[0], X2_train, X2_test, n_tsne)
    feature_names_tsne = ["feature_"+str(i) for i in range(1,n_tsne+1)]
    
    #parse into dictonnaries
    kidney = {"data": l[0], "labels": l[1], "names": l[2], "data_train": X_train, "data_test": X_test, "label_train": y_train, "label_test": y_test, "dataset_name": "kidney", "class_possibilities": ["cdk","notcdk"]}
    banknote = {"data": l2[0], "labels": l2[1], "names": l2[2], "data_train": X2_train, "data_test": X2_test, "label_train": y2_train, "label_test": y2_test, "dataset_name": "banknote", "class_possibilities": ["0","1"]}
    
    kidney_pca = {"data": all_pca, "labels": l[1], "names": feature_names_pca, "data_train": train_pca, "data_test": test_pca, "label_train": y_train, "label_test": y_test, "dataset_name": "kidney_pca", "class_possibilities": ["cdk","notcdk"]}
    banknote_pca = {"data": all_pca2, "labels": l2[1], "names": feature_names_pca, "data_train": train_pca2, "data_test": test_pca2, "label_train": y2_train, "label_test": y2_test, "dataset_name": "banknote_pca", "class_possibilities": ["0","1"]}}
    
    kidney_tsne = {"data": all_tsne, "labels": l[1], "names": feature_names_tsne, "data_train": train_tsne, "data_test": test_tsne, "label_train": y_train, "label_test": y_test, "dataset_name": "kidney_tsne", "class_possibilities": ["cdk","notcdk"]}}
    banknote_tsne = {"data": all_tsne2, "labels": l[1], "names": feature_names_tsne, "data_train": train_tsne2, "data_test": test_tsne2, "label_train": y2_train, "label_test": y2_test, "dataset_name": "banknote_tsne", "class_possibilities": ["0","1"]}}
    
    return kidney, banknote, kidney_pca, banknote_pca, kidney_tsne, banknote_tsne

if __name__ == '__main__':
    #to import and preprocess the 2 datasets, use the following line
    kidney, banknote, kidney_pca, banknote_pca, kidney_tsne, banknote_tsne = preprocess_main()
    print(banknote.get("data_train"))
    print("-----")
    print(banknote.get("label_train"))
    print("---------------")
    print(kidney.get("data_train"))
    print("-----")
    print(kidney.get("label_train"))

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
