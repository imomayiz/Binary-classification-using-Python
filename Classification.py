#!/usr/bin/env python
# coding: utf-8
import Preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier



def log_reg(datadict) :
    """
    Author: Romain N\n   
    Train and test Logistic Regression algorithm on the dataset contained in datadict
    
    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py
    
    Returns:
        Accuracy of Logistic regresion on the testing set
    """
    lr = LogisticRegression(solver = 'lbfgs',multi_class='auto')
    lr.fit(datadict.get("data_train"),datadict.get("label_train"))
    
    return lr.score(datadict.get("data_test"),datadict.get("label_test"))
    
def knn(datadict,n=30) :
    """
    Author: Romain N\n
    Train and test kNN algorithm on the dataset contained in datadict, with n_neighbors up to n

    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py
        n: maximal int to try as n_neighbors
    
    Returns:
        Accuracy of kNN on the testing set, best parameters found during grid search
    """
    k = KNeighborsClassifier(n_neighbors = n)
    parameters = {'n_neighbors':[i for i in range(1,n)]}
    model = GridSearchCV(k, parameters, cv=5, refit=True)
    model.fit(datadict.get("data_train"),datadict.get("label_train"))
    label_pred = model.predict(datadict.get("data_test"))
    return accuracy_score(label_pred, datadict.get("label_test")), model.best_params_


def SVM(datadict):
    """
    Author: Imane M\n
    Implementation of Support vector machines algorithm using sklearn
    
    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py
    
    Returns: 
        Accuracy of the model on the test set
    """
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001], 'kernel': ['rbf','poly']}
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=1)
    grid.fit(datadict.get("data_train"),datadict.get("label_train"))
    predictions = grid.predict(datadict.get("data_test"))
    return(accuracy_score(predictions, datadict.get("label_test")),grid.best_params_)
    
def gradient(datadict):
    """
    Author: Romain N\n
    Train and test stochastic gradient algorithm on the dataset contained in datadict

    Args:
        datadict: dictionary returned by function "preprocessing_main" in Preprocessing.py
    
    Returns:
        Accuracy of stochastic gradient on the testing set, best parameters found during grid search
    """
    param_grid={'loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],'penalty':['none','l2','l1']}
    grid = GridSearchCV(SGDClassifier(),param_grid,refit=True,cv=5)
    grid.fit(datadict.get("data_train"),datadict.get("label_train"))
    pred = grid.predict(datadict.get("data_test"))
    return(accuracy_score(pred,datadict.get("label_test")),grid.best_params_)
    


if __name__ == '__main__':
    """
    Author: Imane M\n
    Function printing the summary of three models (KNN, log regression and SVM)
    performance on both data sets
    """
    kidney, banknote, kidney_pca, banknote_pca, kidney_tsne, banknote_tsne = Preprocessing.preprocess_main()
    
    kd_svm_score , kd_svm_par = SVM(kidney)
    bn_svm_score , bn_svm_par = SVM(banknote)
    
    #KD dataset classification 
    print('*'*50)
    print('\n')
    print("Classifiers performance and parameters on KD dataset:")
    print('\n')
    print("KNN: ")
    print("score: "+str(knn(kidney)[0]))
    print("Parameters:",' ')
    print(knn(kidney)[1])
    print('-'*40)
    print("Log regression: ")
    print(log_reg(kidney))
    print('-'*40)
    print("SVM:")
    print("score: "+str(kd_svm_score))
    print("Parameters:",' ')
    print(kd_svm_par)
    print('\n')
    print("Stochastic Gradient:")
    print("score: "+str(gradient(kidney)[0]))
    print("Parameters:",' ')
    print(gradient(kidney)[1])
    print('\n')
    print('*'*50)
    
    #BN dataset classification
    
    print("Classifiers performance and parameters on Banknote dataset:")
    print('\n')
    print("KNN: ")
    print("score: "+str(knn(banknote)[0]))
    print("Parameters:",' ')
    print(knn(banknote)[1])
    print('-'*40)
    print("Log regression: ")
    print(log_reg(banknote))
    print('-'*40)
    print("SVM:")
    print("score: "+str(bn_svm_score))
    print("Parameters:",' ')
    print(bn_svm_par)
    print('-'*40)
    print("Stochastic Gradient:")
    print("score: "+str(gradient(banknote)[0]))
    print("Parameters:",' ')
    print(gradient(kidney)[1])
    print('\n')
    
    
#    print("Kidney logistic regression: ", log_reg(kidney))
#    print("Banknote knn: ",knn(banknote))
#    print("Banknote logistic regression: ",log_reg(banknote))
#    score_kd, parameters_kd = SVM(kidney)
#    score_bn, parameters_bn = SVM(banknote)
#    print("Kidney SVM: ")
#    print("score:)
#    print("Banknote SVM: ")
#    