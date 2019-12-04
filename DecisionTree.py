#!/usr/bin/env python
# coding: utf-8
import Preprocessing
from sklearn import tree
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import random
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import graphviz
style.use("ggplot")


def create_tree(datadict):
    """
    Author: Thomas K\n
    Builds a the best decision tree from a training set using GridSearchCV

    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py
        class_names : names of the classes
    Returns:
        Decision tree fited with the training dataset, graph view of the tree
    """
    #Create tree
    clf = tree.DecisionTreeClassifier()
    parameters = {'criterion' : ['gini', 'entropy'],'max_depth':[2,10,20,50,80], 'min_samples_split':[2,5,10,20,40,70], 'min_samples_leaf':[2,5,10,20,40,70] }
    model = GridSearchCV(clf,parameters, cv = 5, refit = True)
    #Train tree
    model = model.fit(datadict.get("data_train"),datadict.get("label_train"))
    #Create the graph view of the best estimator tree
    dot_data = tree.export_graphviz(model.best_estimator_, out_file=None, filled=True, feature_names=datadict.get("names")[:-1], class_names=datadict.get("class_possibilities"))
    graph = graphviz.Source(dot_data)
    return model.best_estimator_, graph


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

def decisiontree_main(datadict):
    """
    Author: Thomas K\n
    Builds a decision tree and compute the accuracy

    Args:
        datadict : dictionary returned by Preprocessing.preprocessing_main
        class_names : names of the classes

    Returns:
        A decision tree fited with the training dataset, a graph view of the tree, the accuracy using cross validation, the accuracy without cross validation
    """
    clf, graph = create_tree(datadict)
    #Accuracy using cross validation
    scores = cross_validation_accuracy(datadict, clf)
    #Accuracy without cross validation
    accuracy = accuracy_tree(datadict, clf)
    return clf, graph, scores, accuracy


if __name__ == '__main__':
    kidney, banknote, kidney_pca, banknote_pca, kidney_tsne, banknote_tsne = Preprocessing.preprocess_main()
    #print("dummy : ")
    #print(dummy_classifier(kidney))
    _, graph_kidney, scores_kidney, accuracy_kidney = decisiontree_main(kidney)
    _, graph_banknote, scores_banknote, accuracy_banknote = decisiontree_main(banknote)

    print("Accuracy(with cross_validation) : %0.2f (+/- %0.2f)" % (scores_kidney.mean(), scores_kidney.std() * 2))
    print("Accuracy(without cross_validation) : %0.2f" % accuracy_kidney.mean())
    print("Accuracy(with cross_validation) : %0.2f (+/- %0.2f)" % (scores_banknote.mean(), scores_banknote.std() * 2))
    print("Accuracy(without cross_validation) : %0.2f" % accuracy_banknote.mean())
    graph_kidney.view()
    graph_banknote.view()
