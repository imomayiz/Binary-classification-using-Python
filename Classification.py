#!/usr/bin/env python
# coding: utf-8
import Preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import svm


def make_meshgrid(datadict, h=.02):
    """
    Author: Romain N\n
    Create a mesh of points to plot in
    
    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py
    
    Returns:
        A REMPLIR ROMAIN
    """
    x_min, x_max = datadict.get("data").min() - 1, datadict.get("data").max() + 1
    y_min, y_max = datadict.get("labels").min() - 1, datadict.get("labels").max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def log_reg(datadict) :
    """
    DOCSTRING DE ROMAIN
    """
    lr = LogisticRegression(solver = 'lbfgs',multi_class='auto')
    lr.fit(datadict.get("data_train"),datadict.get("label_train"))
    
    return lr.score(datadict.get("data_test"),datadict.get("label_test"))
    
def knn(datadict,n=30) :
    """
    Author: Romain N\n
    Train and test kNN algorithm on the dataset contained in datadict, with n_neighbors up to n

    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py
        n: maximal int to try as n_neighbors
    
    Returns:
        Accuracy of kNN on the testing set, best parameters found during grid search
    """
    k = KNeighborsClassifier(n_neighbors = n)
    parameters = {'n_neighbors':[i for i in range(1,n)]}
    model = GridSearchCV(k, parameters, cv=5, refit=True)
    model.fit(datadict.get("data_train"),datadict.get("label_train"))
    label_pred = model.predict(datadict.get("data_test"))
    return accuracy_score(label_pred, datadict.get("label_test")), model.best_params_

if __name__ == '__main__':
    kidney, banknote, kidney_pca, banknote_pca, kidney_tsne, banknote_tsne = Preprocessing.preprocess_main()
    print(knn(kidney))
    print(knn(banknote))