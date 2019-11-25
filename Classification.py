# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:57:56 2019

@author: Romain N
"""
import Preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm



iris = datasets.load_iris()
X = iris.data  # Temporary dataset.
y = iris.target 

def make_meshgrid(x, y, h=.02):
    """
    Create a mesh of points to plot in
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def log_reg(X,y,n) :
    pca = PCA(n_components=n)
    pca.fit(X)
    X = pca.transform(X)
    
    
    
    X_train,x_test,Y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    lr = LogisticRegression(solver = 'lbfgs',multi_class='auto')
    lr.fit(X_train,Y_train)
    
    return lr.score(x_test,y_test)
    
def knn(X,y,n) :
    k = KNeighborsClassifier(n_neighbors = n)
    X_train,x_test,Y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    k.fit(X_train,Y_train)
    
    return k.score(x_test,y_test)

def knn_perf() :
    scores = [knn(X,y,i) for i in range(1,50)]
    plt.scatter(range(1,50),scores)
    return scores.index(max(scores)) +1


def display2components(X,y) :
    p = PCA(n_components=2)   
    Xp = p.fit(X).transform(X)   
    X0, X1 = Xp[:, 0], Xp[:, 1]
    plt.scatter(X0,X1,c=y)
    return

if __name__ == '__main__':
    l,l2,l_pca,l2_pca,l_tsne,l2_tsne = Preprocessing.preprocessing_main()
    print(l2)