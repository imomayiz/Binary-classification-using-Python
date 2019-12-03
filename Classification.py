#!/usr/bin/env python
# coding: utf-8
import Preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from Preprocessing import preprocess_main
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

def log_reg(datadict) :
    """
    Author: Romain N\n   
    Train and test Logistic Regression algorithm on the dataset contained in datadict
    
    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py
    
    Returns:
        Accuracy of Logistic regresion on the testing set
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


def SVM(datadict):
    """
    Author: Imane M\n
    Implementation of Support vector machines algorithm using sklearn
    
    Args: training and test datasets (data and labels separated)
    
    Returns: accuracy of the model on the test set
    """
    model = SVC()
    model.fit(datadict.get("data_train"),datadict.get("label_train"))
    predictions = model.predict(datadict.get("data_test"))
    #param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
    #grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=1)
    #grid.fit(X_train,y_train)
    #grid.best_params_
    #grid.best_estimator_
    #grid_predictions = grid.predict(X_test)
    performance = confusion_matrix(y_test,predictions),classification_report(y_test,predictions)
    return(accuracy_score(predictions, datadict.get("label_test")))
    


if __name__ == '__main__':
    kidney, banknote, kidney_pca, banknote_pca, kidney_tsne, banknote_tsne = Preprocessing.preprocess_main()
    print("Kidney knn: ",knn(kidney))
    print("Kidney logistic regression: ", log_reg(kidney))
    print("Banknote knn: ",knn(banknote))
    print("Banknote logistic regression: ",log_reg(banknote))