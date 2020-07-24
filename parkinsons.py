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

train = pd.read_csv('parkinsons_train.csv', delimiter=',')
test = pd.read_csv('parkinsons_test.csv', delimiter=',')

features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)' ,'MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)', 'MDVP:RAP','MDVP:PPQ','Jitter:DDP', 'MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA', 'NHR','HNR', 'RPDE','DFA', 'spread1','spread2','D2','PPE']

X = train[features]
y = train['status']

temp = test['status']

# ~100%
model = Sequential()
model.add(Dense(23, input_dim=22))
model.add(Dense(20, activation='relu'))
#model.add(Dense(10, activation='relu'))
#model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=1000, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y) # figure out if y or temp
print('Nueral Network: %.2f' % (accuracy*100))
# should print 0
pred = model.predict(test[features])
a = 0
b = 0
for i in range(len(pred)):
	calc = pred[i] - temp[i]
	calc = round(float(calc))
	if calc == 0:
		print('Good accuracy: ' + str(pred[i] - temp[i]))
		a += 1
	else:
		print('Bad accuracy: ' + str(pred[i] - temp[i]) + str(pred[i]))
		b += 1
percent_good = a / (a + b)
percent_bad = b / (a + b)
print('Percent good: ' + str(percent_good))
print('Percent bad: ' + str(percent_bad))
