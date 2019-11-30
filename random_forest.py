from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import Preprocessing

def rdforest(datalist):
    X_train, X_test, y_train, y_test = train_test_split(datalist[0], datalist[1], test_size = 0.25)
    parameters = {'n_estimators':[10,100,200], 'criterion':('gini','entropy'), 'max_depth':(3,6,10,1000), 'min_samples_split':(2,4,6,10,15), 'min_samples_leaf':[1,2,4]}
    rf = RandomForestClassifier()
    gs = GridSearchCV(rf, parameters, cv=5, refit=True)
    gs.fit(X_train,y_train)
    return gs.best_score_, gs.best_params_

if __name__ == '__main__':
    data, data2, data_pca, data2_pca, data_tsne, data2_tsne = Preprocessing.preprocessing_main()
    score, param = rdforest(data)
    print(score)
    print('-'*10)
    print(param)
    print('-'*20)
    score2, param2 = rdforest(data2)
    print(score2)
    print('-'*10)
    print(param2)
    


