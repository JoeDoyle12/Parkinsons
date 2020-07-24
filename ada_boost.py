import pandas as pd
import sklearn
from keras.models import Sequential
from keras.layers import Dense
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

train = pd.read_csv('parkinsons_train.csv')
test = pd.read_csv('parkinsons_test.csv')

features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]
# 64%
X = train[features]
y = train['status']

temp = test['status']

i = 1
j = 1

results = []
regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=100), n_estimators=100, random_state=)
regr.fit(X,y)
'''
for i,j in range(1000):
	acc = accuracy_score(regr.predict(test[jitter], y))
	if results.length() == 0 or acc > results[2]:
		results = [i, j, acc]

print('Use max_depth: ' + results[0] + ' , n_estimators: ' + results[1] + ' to get a maximum accuracy score of ' + results[2])
i = results[0]
j = results[1]
'''
pred5 = regr.predict(test[jitter])

print(str(accuracy_score(pred5, temp)))