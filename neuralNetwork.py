import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn import datasets
import Preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data
y = iris.target 
sc = StandardScaler()
X = sc.fit_transform(X)
datalist = [X, y]


l,l2,l_pca,l2_pca,l_tsne,l2_tsne = Preprocessing.preprocessing_main()


def neuralNetwork(datalist, nbneurons, epochs) :
    """
    Author: Karel Kedemos\n
    Train and execute a neural network with len(nbneurons) + 1 layers, each with nbneurons[i] neurons, for binary classification.
    
    Args:
        datalist: list returned by function "preprocessing"
        nbneurons : list containing the number of neurons for each layer, except the last one which has 1 output neuron. The number of layers is equal to len(nbneurons) + 1.

    Returns:
        The accuracy 
    """
    nb = len(nbneurons)
    if nb < 1 : 
        return 'nb must be greater than 1'

    X_train, X_test, y_train, y_test = train_test_split(datalist[0], datalist[1], test_size = 0.25)
    model = Sequential()
    model.add(Dense(nbneurons[0], activation = "relu", input_dim = X_train.shape[1], kernel_initializer = 'random_normal'))
    for layer in range(1, nb) :
        model.add(Dense(nbneurons[layer], activation = "relu", kernel_initializer = 'random_normal'))
    model.add(Dense(1, activation = "sigmoid", kernel_initializer = 'random_normal'))

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit(X_train, y_train, epochs = epochs)
    test_loss, test_acc = model.evaluate(X_test, y_test)

    return test_acc

