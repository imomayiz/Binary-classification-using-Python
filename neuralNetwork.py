#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn.model_selection import train_test_split
import Preprocessing
from keras.models import Sequential
from keras.layers import Dense

def neuralNetwork(datadict, nbneurons, epochs) :
    """
    Author: Karel Kedemos\n
    Train and execute a neural network with len(nbneurons) + 1 layers, each with nbneurons[i] neurons, for binary classification.
    
    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py
        nbneurons : list containing the number of neurons for each layer, except the last one which has 1 output neuron. The number of layers is equal to len(nbneurons) + 1
        epochs : number of epochs for the training

    Returns:
        The model, and the accuracy of the model on the testing data.
    """
    nb = len(nbneurons)
    if nb < 1 : 
        return 'nb must be greater than 1'

    model = Sequential()
    model.add(Dense(nbneurons[0], activation = "relu", input_dim = datadict.get("data_train").shape[1], kernel_initializer = 'random_normal'))
    for layer in range(1, nb) :
        model.add(Dense(nbneurons[layer], activation = "relu", kernel_initializer = 'random_normal'))
    model.add(Dense(1, activation = "sigmoid", kernel_initializer = 'random_normal'))

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit(datadict.get("data_train"), datadict.get("label_train"), epochs = epochs)
    test_loss, test_acc = model.evaluate(datadict.get("data_test"), datadict.get("label_test"))

    return model, test_acc

def neuralNetworkGridSearch(datadict, param_grid) :
    """
    Author: Karel Kedemos\n
    Train several neural network models with different parameters, and return the best one based on the accuracy of the testing data.
    
    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py
        param_grid : Dictionary with parameters names (string) as keys and lists of parameter settings to try as values. This enables searching over any sequence of parameter settings

    Returns:
        The model with the best parameters, the best parameters, and the accuracy of the best model on the testing data
    """
    best_test_acc_sum = 0
    best_model = Sequential()
    best_params = []
    for nbneurons in param_grid["nbneurons"] :
        for epochs in param_grid["epochs"] :
            test_acc_sum = 0
            for i in range(3) :
                model, test_acc = neuralNetwork(datadict, nbneurons, epochs)
                test_acc_sum += test_acc
            if test_acc_sum > best_test_acc_sum : 
                best_test_acc_sum = test_acc_sum
                best_model = model
                best_params = [nbneurons, epochs]
    return best_model, best_params, best_test_acc_sum/3

if __name__ == '__main__':
    kidney, banknote, kidney_pca, banknote_pca, kidney_tsne, banknote_tsne = Preprocessing.preprocess_main()
    data = kidney
    
    model, test_acc = neuralNetwork(data, [32,64,16,8], 300)
    param_grid = {"nbneurons" : [[1], [5,6,7,8], [32,64,16,8]], "epochs" : [100,200,300]}
    best_model, best_params, best_test_acc = neuralNetworkGridSearch(data, param_grid)
    print(best_params)
    print(best_test_acc)
