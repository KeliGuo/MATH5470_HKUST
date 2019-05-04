import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

feature_selected = ['Danceability', 
                 'Energy', 
                 'Speechiness', 
                 'Acousticness', 
                 'Instrumentalness', 
                 'Liveness',
                 'Valence',
                 'Loudness',
                 'Tempo',
                 'Artist_Score']

Data = pd.read_excel('./tmp/feature_complete_normalized_1990_2019.xlsx')
kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(Data)

i=1
store_train = []
store_test =[]
for train_index, test_index in kf.split(Data):
    train_set = Data.loc[train_index,]
    Xtrain = np.array(train_set[feature_selected])
    Ytrain = np.array(train_set['label'], dtype=float)
    test_set =  Data.loc[test_index,]
    Xtest = np.array(test_set[feature_selected])
    Ytest = np.array(test_set['label'], dtype=float)
    clf = tree.DecisionTreeClassifier(max_depth = 10)
    clf.fit(Xtrain, Ytrain)
    train_predict = clf.predict(Xtrain)
    train_accuracy = (train_predict==Ytrain).mean()
    test_predict = clf.predict(Xtest)
    test_accuracy = (test_predict==Ytest).mean()
    store_train.append(train_accuracy)
    store_test.append(test_accuracy)

averge_train_accuracy = np.mean(store_train)
print('Train:',averge_train_accuracy)
averge_test_accuracy = np.mean(store_test)
print('Test:',averge_test_accuracy)


'''try max_depth=n, max_depth=1.5n, max_depth=2n...; n is the number of features'''