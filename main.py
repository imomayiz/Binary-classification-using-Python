#!/usr/bin/env python
# coding: utf-8
import Preprocessing
import neuralNetwork
import DecisionTree
import Classification
import random_forest

def main():
    """
    Author: Guillaume S\n
    Call all classification algorithms and sum up their results
    """
    print("Preprocessing...")
    kidney, banknote, kidney_pca, banknote_pca, kidney_tsne, banknote_tsne = Preprocessing.preprocess_main()
    kidney_accuracies = {}
    banknote_accuracies = {}
    kidney_parameters = {}
    banknote_parameters = {}

    #Neural network
    print("Run neural networks...")
    param_grid = {"nbneurons" : [[4,12,8], [16,32,24,12], [32,64,16,8]], "epochs" : [100,200,300]}
    _, kidney_nn_param, kidney_nn_acc = neuralNetwork.neuralNetworkGridSearch(kidney,param_grid)
    _, kidney_pca_nn_param, kidney_pca_nn_acc = neuralNetwork.neuralNetworkGridSearch(kidney_pca,param_grid)
    _, kidney_tsne_nn_param, kidney_tsne_nn_acc = neuralNetwork.neuralNetworkGridSearch(kidney_tsne,param_grid)
    _, banknote_nn_param, banknote_nn_acc = neuralNetwork.neuralNetworkGridSearch(banknote,param_grid)
    _, banknote_pca_nn_param, banknote_pca_nn_acc = neuralNetwork.neuralNetworkGridSearch(banknote_pca,param_grid)
    _, banknote_tsne_nn_param, banknote_tsne_nn_acc = neuralNetwork.neuralNetworkGridSearch(banknote_tsne,param_grid)
    kidney_accuracies["kidney_nn"] = kidney_nn_acc
    kidney_accuracies["kidney_pca_nn"] = kidney_pca_nn_acc
    kidney_accuracies["kidney_tsne_nn"] = kidney_tsne_nn_acc
    kidney_parameters["kidney_nn"] = kidney_nn_param
    kidney_parameters["kidney_pca_nn"] = kidney_pca_nn_param
    kidney_parameters["kidney_tsne_nn"] = kidney_tsne_nn_param
    banknote_accuracies["banknote_nn"] = banknote_nn_acc
    banknote_accuracies["banknote_pca_nn"] = banknote_pca_nn_acc
    banknote_accuracies["banknote_tsne_nn"] = banknote_tsne_nn_acc
    banknote_parameters["banknote_nn"] = banknote_nn_param
    banknote_parameters["banknote_pca_nn"] = banknote_pca_nn_param
    banknote_parameters["banknote_tsne_nn"] = banknote_tsne_nn_param

    #Classification
    print("Run kNN...")
    kidney_knn_acc, kidney_knn_param  = Classification.knn(kidney)
    kidney_pca_knn_acc, kidney_pca_knn_param  = Classification.knn(kidney_pca)
    kidney_tsne_knn_acc, kidney_tsne_knn_param  = Classification.knn(kidney_tsne)
    banknote_knn_acc, banknote_knn_param  = Classification.knn(banknote)
    banknote_pca_knn_acc, banknote_pca_knn_param  = Classification.knn(banknote_pca)
    banknote_tsne_knn_acc, banknote_tsne_knn_param  = Classification.knn(banknote_tsne)
    kidney_accuracies["kidney_knn"] = kidney_knn_acc
    kidney_accuracies["kidney_pca_knn"] = kidney_pca_knn_acc
    kidney_accuracies["kidney_tsne_knn"] = kidney_tsne_knn_acc
    kidney_parameters["kidney_knn"] = kidney_knn_param
    kidney_parameters["kidney_pca_knn"] = kidney_pca_knn_param
    kidney_parameters["kidney_tsne_knn"] = kidney_tsne_knn_param
    banknote_accuracies["banknote_knn"] = banknote_knn_acc
    banknote_accuracies["banknote_pca_knn"] = banknote_pca_knn_acc
    banknote_accuracies["banknote_tsne_knn"] = banknote_tsne_knn_acc
    banknote_parameters["banknote_knn"] = banknote_knn_param
    banknote_parameters["banknote_pca_knn"] = banknote_pca_knn_param
    banknote_parameters["banknote_tsne_knn"] = banknote_tsne_knn_param

    print("Run logistic regression...")
    kidney_logreg_acc = Classification.log_reg(kidney)
    kidney_pca_logreg_acc = Classification.log_reg(kidney_pca)
    kidney_tsne_logreg_acc = Classification.log_reg(kidney_tsne)
    banknote_logreg_acc = Classification.log_reg(banknote)
    banknote_pca_logreg_acc = Classification.log_reg(banknote_pca)
    banknote_tsne_logreg_acc = Classification.log_reg(banknote_tsne)
    kidney_accuracies["kidney_logreg"] = kidney_logreg_acc
    kidney_accuracies["kidney_pca_logreg"] = kidney_pca_logreg_acc
    kidney_accuracies["kidney_tsne_logreg"] = kidney_tsne_logreg_acc
    banknote_accuracies["banknote_logreg"] = banknote_logreg_acc
    banknote_accuracies["banknote_pca_logreg"] = banknote_pca_logreg_acc
    banknote_accuracies["banknote_tsne_logreg"] = banknote_tsne_logreg_acc

    print("Run SVM...")
    kidney_svm_acc = Classification.SVM(kidney)
    kidney_pca_svm_acc = Classification.SVM(kidney_pca)
    kidney_tsne_svm_acc = Classification.SVM(kidney_tsne)
    banknote_svm_acc = Classification.SVM(banknote)
    banknote_pca_svm_acc = Classification.SVM(banknote_pca)
    banknote_tsne_svm_acc = Classification.SVM(banknote_tsne)
    kidney_accuracies["kidney_svm"] = kidney_svm_acc
    kidney_accuracies["kidney_pca_svm"] = kidney_pca_svm_acc
    kidney_accuracies["kidney_tsne_svm"] = kidney_tsne_svm_acc
    banknote_accuracies["banknote_svm"] = banknote_svm_acc
    banknote_accuracies["banknote_pca_svm"] = banknote_pca_svm_acc
    banknote_accuracies["banknote_tsne_svm"] = banknote_tsne_svm_acc


    print("Run decision tree...")
    _, decision_tree_graph_kidney, decision_tree_scores_kidney, decision_tree_accuracy_kidney = DecisionTree.decisiontree_main(kidney, ["cdk","notcdk"])
    _, decision_tree_graph_banknote, decision_tree_scores_banknote, decision_tree_accuracy_banknote = DecisionTree.decisiontree_main(banknote, ["0","1"])
    kidney_accuracies["kidney_decision_tree"] = decision_tree_scores_kidney
    banknote_accuracies["banknote_decision_tree"] = decision_tree_scores_banknote
    '''
    pca, tnse pas trop d'interet pour decision tree + marche pas car nombre de features renvoyes par presprocessing pas modifie pour pca et tnse

    _, decision_tree_graph_kidney_pca, decision_tree_scores_kidney_pca, decision_tree_accuracy_kidney_pca = DecisionTree.decisiontree_main(kidney_pca, ["cdk","notcdk"])
    _, decision_tree_graph_kidney_tnse, decision_tree_scores_kidney_tnse, decision_tree_accuracy_kidney_tnse = DecisionTree.decisiontree_main(kidney_tnse, ["cdk","notcdk"])
    _, decision_tree_graph_banknote_pca, decision_tree_scores_banknote_pca, decision_tree_accuracy_banknote_pca = DecisionTree.decisiontree_main(banknote_pca, ["0","1"])
    _, decision_tree_graph_banknote_tnse, decision_tree_scores_banknote_tnse, decision_tree_accuracy_banknote_tnse = DecisionTree.decisiontree_main(banknote_tnse, ["0","1"])
    kidney_accuracies["kidney_pca_decision_tree"] = decision_tree_scores_kidney_pca
    kidney_accuracies["kidney_tsne_decision_tree"] = decision_tree_scores_kidney_tnse
    banknote_accuracies["banknote_pca_decision_tree"] = decision_tree_scores_banknote_pca
    banknote_accuracies["banknote_tsne_decision_tree"] = banknote_tsne_decision_tree_acc
    '''




    #Random_forest
    #A faire

    #Take the biggest accuracy, algorithm name and parameters
    kidney_best_accuracy = max(kidney_accuracies.values())
    banknote_best_accuracy = max(banknote_accuracies.values())
    kidney_best_accuracy_name = max(kidney_accuracies, key=kidney_accuracies.get)
    banknote_best_accuracy_name = max(banknote_accuracies, key=banknote_accuracies.get)
    kidney_best_param = kidney_parameters.get(kidney_best_accuracy_name)
    banknote_best_param = banknote_parameters.get(banknote_best_accuracy_name)

    print("-------------------------Kidney dataset-------------------------")
    print("Best accuracy : "+str(kidney_best_accuracy))
    print("Obtained with : "+str(kidney_best_accuracy_name))
    print("Best parameters : "+str(kidney_best_param))
    print("------------------------Banknote dataset------------------------")
    print("Best accuracy : "+str(banknote_best_accuracy))
    print("Obtained with : "+str(banknote_best_accuracy_name))
    print("Best parameters : "+str(banknote_best_param))
    print("------------Results per algorithm : kidney dataset--------------")
    print(kidney_accuracies)
    print("-----------Results per algorithm : banknote dataset-------------")
    print(banknote_accuracies)

if __name__ == '__main__':
    main()