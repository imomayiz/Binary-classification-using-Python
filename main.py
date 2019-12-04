#!/usr/bin/env python
# coding: utf-8
import Preprocessing
import neuralNetwork
import DecisionTree
import Classification
import random_forest

def main_dico(datadict, datadict_pca, datadict_tsne):
    """
    Author: Guillaume S\n
    Try every classification algorithm on datadict, and print resulting accuracies.
    
    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py
    """
    accuracies = {}
    parameters = {}
    print("\n--------------For "+ datadict.get("dataset_name") +" dataset--------------")
    
    print("Run neural networks...")
    param_grid = {"nbneurons" : [[4,12,8], [16,32,24,12], [32,64,16,8]], "epochs" : [100,200,300]}
    _, neural_network_param, neural_network_acc = neuralNetwork.neuralNetworkGridSearch(datadict, param_grid)
    _, neural_network_param_pca, neural_network_acc_pca = neuralNetwork.neuralNetworkGridSearch(datadict_pca, param_grid)
    _, neural_network_param_tsne, neural_network_acc_tsne = neuralNetwork.neuralNetworkGridSearch(datadict_tsne, param_grid)
    accuracies["neural_network"] = neural_network_acc
    accuracies["neural_network_pca"] = neural_network_acc_pca
    accuracies["neural_network_tsne"] = neural_network_acc_tsne
    parameters["neural_network"] = neural_network_param
    parameters["neural_network_pca"] = neural_network_param_pca
    parameters["neural_network_tsne"] = neural_network_param_tsne
    
    print("Run kNN...")
    knn_acc, knn_param  = Classification.knn(datadict)
    knn_acc_pca, knn_param_pca  = Classification.knn(datadict_pca)
    knn_acc_tsne, knn_param_tsne  = Classification.knn(datadict_tsne)
    accuracies["knn"] = knn_acc
    accuracies["knn_pca"] = knn_acc_pca
    accuracies["knn_tsne"] = knn_acc_tsne
    parameters["knn"] = knn_param
    parameters["knn_pca"] = knn_param_pca
    parameters["knn_tsne"] = knn_param_tsne
    
    print("Run logistic regression...")
    logreg_acc = Classification.log_reg(datadict)
    logreg_acc_pca = Classification.log_reg(datadict_pca)
    logreg_acc_tsne = Classification.log_reg(datadict_tsne)
    accuracies["logreg"] = logreg_acc
    accuracies["logreg_pca"] = logreg_acc_pca
    accuracies["logreg_tsne"] = logreg_acc_tsne
    parameters["logreg"] = "No parameters for this algorithm"
    parameters["logreg_pca"] = "No parameters for this algorithm"
    parameters["logreg_tsne"] = "No parameters for this algorithm"
    
    print("Run SVM...")
    svm_acc, svm_param = Classification.SVM(datadict)
    svm_acc_pca, svm_param_pca = Classification.SVM(datadict_pca)
    svm_acc_tsne, svm_param_tsne = Classification.SVM(datadict_tsne)
    accuracies["svm"] = svm_acc
    accuracies["svm_pca"] = svm_acc_pca
    accuracies["svm_tsne"] = svm_acc_tsne
    parameters["svm"] = svm_param
    parameters["svm_pca"] = svm_param_pca
    parameters["svm_tsne"] = svm_param_tsne
    
    print("Run stochastic gradient...")
    gradient_acc, gradient_param = Classification.gradient(datadict)
    gradient_acc_pca, gradient_param_pca = Classification.gradient(datadict_pca)
    gradient_acc_tsne, gradient_param_tsne = Classification.gradient(datadict_tsne)
    accuracies["stochastic_gradient"] = gradient_acc
    accuracies["stochastic_gradient_pca"] = gradient_acc_pca
    accuracies["stochastic_gradient_tsne"] = gradient_acc_tsne
    parameters["stochastic_gradient"] = gradient_param
    parameters["stochastic_gradient_pca"] = gradient_param_pca
    parameters["stochastic_gradient_tsne"] = gradient_param_tsne
    
    print("Run random forest...")
    randomforest_acc, randomforest_param = random_forest.rdforest(datadict)
    randomforest_acc_pca, randomforest_param_pca = random_forest.rdforest(datadict_pca)
    randomforest_acc_tsne, randomforest_param_tsne = random_forest.rdforest(datadict_tsne)
    accuracies["randomforest"] = randomforest_acc
    accuracies["randomforest_pca"] = randomforest_acc_pca
    accuracies["randomforest_tsne"] = randomforest_acc_tsne
    parameters["randomforest"] = randomforest_param
    parameters["randomforest_pca"] = randomforest_param_pca
    parameters["randomforest_tsne"] = randomforest_param_tsne
    
    print("Run decision tree...")
    _, _, decision_tree_acc_list, _ = DecisionTree.decisiontree_main(datadict)
    accuracies["decision_tree"] = max(decision_tree_acc_list)
    parameters["decision_tree"] = "No parameters for this algorithm"
    print("Done.\n")
    
    #Take the biggest accuracy, algorithm name and parameters
    best_accuracy = max(accuracies.values())
    best_accuracy_name = max(accuracies, key=accuracies.get)
    best_param = parameters.get(best_accuracy_name)
    
    print("---Results for "+datadict.get("dataset_name")+" dataset---")
    print("Best accuracy : "+str(best_accuracy))
    print("Obtained with : "+str(best_accuracy_name))
    print("Best parameters : "+str(best_param))
    print("Every accuracy : ")
    for name in accuracies.keys():
        print("    "+ name + " : " + str(accuracies.get(name)))
    print(" ")

def main():
    """
    Author: Guillaume S\n
    Main algorithm (to launch) : call all classification algorithms on every dataset and sum up their results.
    """
    liste = Preprocessing.preprocess_main()
    if len(liste)%3!=0:
        print("Warning : the number of datadict returned by preprocess_main is not divisible by 3.")
    for i in range(len(liste)//3):
        main_dico(liste[3*i], liste[3*i+1], liste[3*i+2])

if __name__ == '__main__':
    main()