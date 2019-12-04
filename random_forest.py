#!/usr/bin/env python
# coding: utf-8
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import Preprocessing

def rdforest(datadict):
    """
    Author: Guillaume S\n
    Train and test random forest algorithm with the dataset contained in datadict.
    
    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py

    Returns:
        Accuracy of random forest algorithm on the testing set, best parameters found during grid search
    """
    parameters = {'n_estimators':[10,100,200], 'criterion':('gini','entropy'), 'max_depth':(5,10,100), 'min_samples_split':(2,4,6), 'min_samples_leaf':[1,3], "max_samples": [0.5, 1.0]}
    rf = RandomForestClassifier()
    gs = GridSearchCV(rf, parameters, cv=5, refit=True)
    gs.fit(datadict.get("data_train"),datadict.get("label_train"))
    label_pred = gs.predict(datadict.get("data_test"))
    return accuracy_score(label_pred, datadict.get("label_test")), gs.best_params_

if __name__ == '__main__':
    kidney, banknote, kidney_pca, banknote_pca, kidney_tsne, banknote_tsne = Preprocessing.preprocess_main()
    score, param = rdforest(kidney)
    print("Accuracy with kidney dataset : " + str(score))
    print(param)
    print('-'*20)
    score2, param2 = rdforest(banknote)
    print("Accuracy with banknote dataset : " + str(score2))
    print(param2)
    


