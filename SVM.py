import pandas as pd
import numpy as np
from sklearn import svm

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


train_set = pd.read_excel('./train_set/train.xlsx')
Xtrain = np.array(train_set[feature_selected])
Ytrain = np.array(train_set['label'], dtype=float)
test_set = pd.read_excel('./test_set/test.xlsx')
Xtest = np.array(test_set[feature_selected])
Ytest = np.array(test_set['label'], dtype=float)

clf_lin = svm.SVC(gamma='scale',kernel='linear')
clf_rbf = svm.SVC(gamma='scale',kernel='rbf')
clf_poly = svm.SVC(gamma='scale',kernel='poly')

clf_lin.fit(Xtrain, Ytrain)
clf_rbf.fit(Xtrain, Ytrain)
clf_poly.fit(Xtrain, Ytrain)


train_predict_lin = clf_lin.predict(Xtrain)
train_predict_rbf = clf_rbf.predict(Xtrain)
train_predict_poly = clf_poly.predict(Xtrain)
train_accuracy_lin = (train_predict_lin==Ytrain).mean()
train_accuracy_rbf = (train_predict_rbf==Ytrain).mean()
train_accuracy_poly = (train_predict_poly==Ytrain).mean()
print("Train accuracy_linear:", train_accuracy_lin)
print("Train accuracy_rbf:", train_accuracy_rbf)
print("Train accuracy_polynomial:", train_accuracy_poly)


test_predict_lin = clf_lin.predict(Xtest)
test_predict_rbf = clf_rbf.predict(Xtest)
test_predict_poly = clf_poly.predict(Xtest)
test_accuracy_lin = (test_predict_lin==Ytest).mean()
test_accuracy_rbf = (test_predict_rbf==Ytest).mean()
test_accuracy_poly = (test_predict_poly==Ytest).mean()
print("Test accuracy_linear:", test_accuracy_lin)
print("Test accuracy_rbf:", test_accuracy_rbf)
print("Test accuracy_polynomial:", test_accuracy_poly)

