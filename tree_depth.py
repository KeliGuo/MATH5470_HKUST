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
store_train_p = []
store_test_p =[]
store_train_r = []
store_test_r =[]
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
    tp1 = sum(((train_predict == 1) & (Ytrain == 1)) * 1)
    fp1 = sum(((train_predict == 0) & (Ytrain == 1)) * 1)
    tn1 = sum(((train_predict == 1) & (Ytrain == 0)) * 1)
    fn1 = sum(((train_predict == 0) & (Ytrain == 0)) * 1)
    train_precision = tp1 / (tp1 + fp1)
    train_recall = tp1 / (tp1 + fn1)
    store_train_p.append(train_precision)
    store_train_r.append(train_recall)
    tp2 = sum(((test_predict == 1) & (Ytest == 1)) * 1)
    fp2 = sum(((test_predict == 0) & (Ytest == 1)) * 1)
    tn2 = sum(((test_predict == 1) & (Ytest == 0)) * 1)
    fn2 = sum(((test_predict == 0) & (Ytest == 0)) * 1)
    test_precision = tp2 / (tp2 + fp2)
    test_recall = tp2 / (tp2 + fn2)
    store_test_p.append(test_precision)
    store_test_r.append(test_recall)


averge_train_accuracy = np.mean(store_train)
print('Train:',averge_train_accuracy)
averge_test_accuracy = np.mean(store_test)
print('Test:',averge_test_accuracy)
averge_train_precision = np.mean(store_train_p)
print("Train precision:",averge_train_precision)
averge_test_precision = np.mean(store_test_p)
print("Test precision:",averge_test_precision)
averge_train_recall = np.mean(store_train_r)
print("Train recall:",averge_train_recall)
averge_test_recall = np.mean(store_test_r)
print("Test recall:",averge_test_recall)



'''try max_depth=p, max_depth=1.5p, max_depth=2p...; p is the number of features'''