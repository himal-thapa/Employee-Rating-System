import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset_1 = pd.read_csv('001.csv')
dataset_2 = pd.read_csv('002.csv')
dataset_3 = pd.read_csv('003.csv')
dataset_4 = pd.read_csv('004.csv')
dataset_5 = pd.read_csv('005.csv')
dataset_6 = pd.read_csv('006.csv')
dataset_7 = pd.read_csv('007.csv')
dataset_8 = pd.read_csv('008.csv')
dataset_9 = pd.read_csv('009.csv')
dataset_10 = pd.read_csv('010.csv')


X_1 = dataset_1.iloc[:, [1,2,4,5,6,7,8,9,10]].values
y_1 = dataset_1['Rating'].values

X_2 = dataset_2.iloc[:, [1,2,4,5,6,7,8,9,10]].values
y_2 = dataset_2['Rating'].values

X_3 = dataset_3.iloc[:, [1,2,4,5,6,7,8,9,10]].values
y_3 = dataset_3['Rating'].values

X_4 = dataset_4.iloc[:, [1,2,4,5,6,7,8,9,10]].values
y_4 = dataset_4['Rating'].values

X_5 = dataset_5.iloc[:, [1,2,4,5,6,7,8,9,10]].values
y_5 = dataset_5['Rating'].values

X_6 = dataset_6.iloc[:, [1,2,4,5,6,7,8,9,10]].values
y_6 = dataset_6['Rating'].values

X_7 = dataset_7.iloc[:, [1,2,4,5,6,7,8,9,10]].values
y_7 = dataset_7['Rating'].values

X_8 = dataset_8.iloc[:, [1,2,4,5,6,7,8,9,10]].values
y_8 = dataset_8['Rating'].values

X_9 = dataset_9.iloc[:, [1,2,4,5,6,7,8,9,10]].values
y_9 = dataset_9['Rating'].values

X_10 = dataset_10.iloc[:, [1,2,4,5,6,7,8,9,10]].values
y_10 = dataset_10['Rating'].values

X = np.concatenate((X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9, X_10))
y = np.concatenate((y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10))

del(y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9, X_10)

X = np.nan_to_num(X, 0)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_train = sc_x.fit_transform(X_train)
#Y_train = sc_y.fit_transform(Y_train.reshape(-1, 1))

X_test = sc_x.transform(X_test)
#Y_test = sc_y.transform(Y_test.reshape(-1,1))


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=9)
lda.fit(X_train, Y_train)
print(lda.explained_variance_ratio_)


'''
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
'''


from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, Y_train)


'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
classifier.fit(X_train, Y_train)
'''


y_pred = (classifier.predict(X_test))

Y_test_dataframe = pd.DataFrame(Y_test)
Y_pred_dataframe = pd.DataFrame(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

'''
from sklearn.model_selection import GridSearchCV
params = [{ 'gamma': [0.0001, 0.01, 0.1, 1]}]
gridSearchCV = GridSearchCV(estimator=classifier, param_grid=params, scoring='accuracy', cv=10, n_jobs=-1)
gridSearchCV = gridSearchCV.fit(X, y)
gridSearchCV.best_params_
'''

'''
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X, y=y, scoring='accuracy', cv=10)
accuracies.mean()
'''

score = classifier.score(X_test, Y_test)

y_new = classifier.predict(sc_x.transform([[10, 9, 15, 2,5,5,5,5,5]]))

pickle.dump(classifier, open('model.emp', 'wb'))

pickle.dump(sc_x, open("scalerX.sc", "wb"))


model = pickle.load(open('model.emp', 'rb'))
scalerX = pickle.load(open("scalerX.sc", "rb"))
model.predict(scalerX.transform([[10, 9, 15, 2,5,5,5,5,5]]))

























# -*- coding: utf-8 -*-

