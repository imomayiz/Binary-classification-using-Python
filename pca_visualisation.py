#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import Preprocessing

def pca_variance_curves(datadict, name):
    """
    Author: Romain N\n
    Compute and plot obtained variances according to each PCA component
    
    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py
        name: string with the name of the dataset
    """
    n_attributs = len(datadict.get("data")[1])
    pca = PCA()
    pca.fit(datadict.get("data"))
    
    variances_ratio = pca.explained_variance_ratio_
    sum_variances = np.copy(variances_ratio)
    for i in range(1,n_attributs) :
        sum_variances[i] += sum_variances[i-1]
    
    plt.title("PCA variance with " + name + " dataset")
    plt.plot(variances_ratio,label="attributs'variance")
    plt.plot(sum_variances,label='sum variance')
    plt.legend()
    plt.show()
    print(sum_variances)
    print(variances_ratio)

if __name__ == '__main__':
    kidney, banknote, kidney_pca, banknote_pca, kidney_tsne, banknote_tsne = Preprocessing.preprocess_main()
    pca_variance_curves(kidney,"kidney")