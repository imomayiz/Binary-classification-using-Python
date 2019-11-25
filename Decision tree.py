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


#Mesure de précision
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


'''
A CONTINUER

#With cross validation
scores = cross_val_score(clfGini, iris.data, iris.target, cv=10)
print("Accuracy of Gini: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(clfEntropy, iris.data, iris.target, cv=10)
print("Accuracy of Entropy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(dummy, iris.data, iris.target, cv=10)
print("Accuracy of Dummy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(classifier, iris.data, iris.target, cv=10)
print("Accuracy of KNN: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#Changing number of minimum observations for a node
iArray=[]
scores=[]
for i in range(2,100):
    clfNode = tree.DecisionTreeClassifier( min_samples_split=i)
    clfNode = clfNode.fit(train_X,train_y)
    scoreBuffer=cross_val_score(clfNode, iris.data, iris.target, cv=10)
    scores.append(scoreBuffer.mean())
    iArray.append(i)
plt.plot(iArray, scores)
plt.show()


#Changing number of minimum sample by leaf
iArray=[]
scores=[]
for i in range(1,100):
    clfNode = tree.DecisionTreeClassifier(min_samples_leaf=i)
    clfNode = clfNode.fit(train_X,train_y)
    scoreBuffer=cross_val_score(clfNode, iris.data, iris.target, cv=10)
    scores.append(scoreBuffer.mean())
    iArray.append(i)
plt.plot(iArray, scores)
plt.show()



#Changing number of minimum observations for a node for KNN classifier
iArray=[]
scores=[]
for i in range(2,100):
    classifier = tree.DecisionTreeClassifier( min_samples_split=i)
    classifier = classifier.fit(train_X,train_y)
    scoreBuffer=cross_val_score(classifier, iris.data, iris.target, cv=10)
    scores.append(scoreBuffer.mean())
    iArray.append(i)
plt.plot(iArray, scores)
plt.show()



#Changing number of minimum sample by leaf for KNN
iArray=[]
scores=[]
for i in range(1,100):
    classifier = tree.DecisionTreeClassifier(min_samples_leaf=i)
    classifier = classifier.fit(train_X,train_y)
    scoreBuffer=cross_val_score(classifier, iris.data, iris.target, cv=10)
    scores.append(scoreBuffer.mean())
    iArray.append(i)
plt.plot(iArray, scores)
plt.show()



#LDA

# Instantiate learning model (n = 1)
lda = LDA(n_components=1)

# Fitting the model
lda.fit(train_X, train_y)

# Predicting the Test set results
y_pred = lda.predict(test_X)

accuracy = metrics.accuracy_score(test_y, y_pred)
print(accuracy)
'''

if __name__ == '__main__':
    l,l2,l_pca,l2_pca,l_tsne,l2_tsne = Preprocessing.preprocessing_main()
    print(l2)