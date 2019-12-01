#!/usr/bin/env python
# coding: utf-8
import Preprocessing
from sklearn import svm
from sklearn import datasets
from sklearn import tree
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import random
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import graphviz
style.use("ggplot")


def create_tree(datadict, class_names=None,feature_names=None criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """
    Author: Thomas K\n
    Builds a decision tree from a training set

    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py
        class_names :Names of each of the target classes in ascending numerical order
        feature_names :Names of each of the features
        criterion: 'gini' or 'entropy', criterion to use
        max_depth: int or None, maximum depth of the tree
        min_samples_split: int, the minimum number of samples required to split an internal node
        min_samples_leaf: int or float, the minimum number of samples required to be at a leaf node

    Returns:
        Decision tree fited with the training dataset, graph view of the tree
    """
    #Create tree
    clf = tree.DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
    #Train tree
    clf = clf.fit(datadict.get("data_train"),datadict.get("label_train"))
    #Create the graph view of the tree
    dot_data = tree.export_graphviz(clf, out_file=None, filled=True,feature_names=feature_names, class_names=class_names)
    graph = graphviz.Source(dot_data)
    return clf, graph


def accuracy_tree(datadict, clf):
    """
    Author: Thomas K\n
    Computes the tree's accuracy

    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py
        clf : decision tree returned by function 'CreateTree'

    Returns:
        Accuracy of the tree on the testing dataset
    """
    label_pred = clf.predict(datadict.get("data_test"))
    accuracy = metrics.accuracy_score(datadict.get("label_test"), label_pred)
    return accuracy


def cross_validation_accuracy(datadict, classifier):
    """
    Author: Thomas K\n
    Computes a classifier's accuracy

    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py
        clf : a classifier

    Returns:
        Array of scores of the estimator for each run of the cross validation
    """
    scores = cross_val_score(classifier, datadict.get("data_train"), datadict.get("label_train"), cv=10)
    print("Accuracy : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores


def dummy_classifier(datadict):
    """
    Author: Thomas K\n
    Creates a dummy classifier

    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py

    Returns:
        Accuracy of the dummy classifier on the training dataset
    """
    # Create dummy classifer
    dummy = DummyClassifier(strategy='uniform', random_state=1)
    # "Train" model
    dummy.fit(datadict.get("data_train"), datadict.get("label_train"))
    label_pred = dummy.predict(datadict.get("data_test"))
    accuracy = metrics.accuracy_score(datadict.get("label_test"), label_pred)
    return accuracy

def DecisionTreeMain(datadict,class_names=None, feature_names = None, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """
    Author: Thomas K\n
    Builds a decision tree and compute the accuracy

    Args:
        datadict : dictionary returned by Preprocessing.preprocessing_main
        class_names :Names of each of the target classes in ascending numerical order
        feature_names :Names of each of the features
        criterion : 'gini' or 'entropy'
        max_depth : int or None, maximum depth of the tree
        min_samples_split : int, the minimum number of samples required to split an internal node
        min_samples_leaf : int or float, the minimum number of samples required to be at a leaf node
    Returns:
        A decision tree fited with the training dataset, a graph view of the tree, the accuracy using cross validation, the accuracy without cross validation
    """
    clf, graph = create_tree(datadict, class_names,feature_names, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    #Accuracy using cross validation
    scores = CrossValidationAccuracy(datadict, clf)
    #Accuracy without cross validation
    accuracy = AccuracyTree(datadict, clf)
    return clf, graph, scores, accuracy


if __name__ == '__main__':
    kidney, banknote, kidney_pca, banknote_pca, kidney_tsne, banknote_tsne = Preprocessing.preprocess_main()
    print("dummy : ")
    print(dummy_classifier(kidney))
