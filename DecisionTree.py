##Imports

import Preprocessing
from sklearn import svm
from sklearn import datasets
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import random
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import graphviz


style.use("ggplot")




def DataSplit(X,Y,splitVal):
    """
    Author: Thomas K\n
    Mixes the data and splits it into the test and train datasets.

    Args:
        X : Initial data values as numpy array
        Y : Initial data labels as numpy array
        splitVal : int or float, percentage of the data used for the testing (between 0 and 100)

    Returns:
        Training dataset values as numpy array, training dataset labels as numpy array, testing dataset values as numpy array, testing dataset values as numpy array
    """
    #Data mix
    number  = random.randrange(1, 9999)
    random.seed(number)
    np.random.shuffle(X)
    random.seed(number)
    random.shuffle(Y)
    #Data split
    n = len(X)
    nsplit = round(n*splitVal/100)
    test_X=X[:nsplit]
    test_Y=Y[:nsplit]
    train_X=X[nsplit:]
    train_Y=Y[nsplit:]
    return train_X, train_Y, test_X, test_Y



def CreateTree(train_X, train_Y, feature_names, class_names, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """
    Author: Thomas K\n
    Builds a decision tree from a training set

    Args:
        train_X : training dataset values, numpy array returned by function 'DataSplit'
        train_Y : training dataset labels, numpy array returned by function 'DataSplit'
        feature_names :
        class_names :
        criterion : 'gini' or 'entropy'
        max_depth : int or None, maximum depth of the tree
        min_samples_split : int, the minimum number of samples required to split an internal node
        min_samples_leaf : int or float, the minimum number of samples required to be at a leaf node
    Returns:
        A decision tree fited with the training dataset, a graph view of the tree
    """
    #Create tree
    clf = tree.DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
    #Train tree
    clf = clf.fit(train_X,train_Y)
    #Create the graph view of the tree
    dot_data = tree.export_graphviz(clf, out_file=None, filled=True,feature_names=feature_names, class_names=class_names)
    graph = graphviz.Source(dot_data)
    return clf, graph


def AccuracyTree(test_X, test_Y, clf):
    """
    Author: Thomas K\n
    Computes the tree's accuracy
    Args:
        test_X : testing dataset values, numpy array returned by function 'DataSplit'
        test_Y : testing dataset labels, numpy array returned by function 'DataSplit'
        clf : decision tree returned by function 'CreateTree'
    Returns:
        The accuracy of the tree according to the testing dataset.
    """
    Y_pred = clf.predict(test_X)
    accuracy = metrics.accuracy_score(test_Y, Y_pred)
    return accuracy


def CrossValidationAccuracy(X, Y, classifier):
    """
    Author: Thomas K\n
    Computes a classifier's accuracy
    Args:
        X : the complete dataset values
        Y : the complete dataset labels
        clf : a classifier
    Returns:
        An array of scores of the estimator for each run of the cross validation.
    """
    scores = cross_val_score(classifier, X, Y, cv=10)
    print("Accuracy : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores


def DummyClassifier(train_X, train_y):
    """
    Author: Thomas K\n
    Creates a dummy classifier
    Args:
        train_X : training dataset values, numpy array returned by function 'DataSplit'
        train_Y : training dataset labels, numpy array returned by function 'DataSplit'
    Returns:
        A dummy classifier fited with the training dataset
    """
    # Create dummy classifer
    dummy = DummyClassifier(strategy='uniform', random_state=1)
    # "Train" model
    dummy.fit(train_X, train_y)
    return dummy


if __name__ == '__main__':
    l,l2,l_pca,l2_pca,l_tsne,l2_tsne = Preprocessing.preprocessing_main()
    print(l2)