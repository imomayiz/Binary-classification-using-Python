# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:32:26 2019

@author: Romain N
"""
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets


iris = datasets.load_iris() #Pour le test mais l'algo est générique
X = iris.data

def pca_variance_curves(X) :
    #standardisation of the data
    n_attributs = len(X[1])
    for i in range(n_attributs) :
        X[:,i] /= np.sum(X[:,i])
    pca = PCA()
    pca.fit(X)
    
    variances_ratio = pca.explained_variance_ratio_
    sum_variances = np.copy(variances_ratio)
    for i in range(1,n_attributs) :
        sum_variances[i] += sum_variances[i-1]
    
    plt.plot(variances_ratio,label="attributs'variance")
    plt.plot(sum_variances,label='sum variance')
    plt.legend()
    print(sum_variances)
    print(variances_ratio)
    return 2

pca_variance_curves(X)