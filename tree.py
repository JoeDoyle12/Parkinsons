import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score

train = pd.read_csv('parkinsons_train.csv')
test = pd.read_csv('parkinsons_test.csv')

features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]
# 63%
X = train[features]
y = train['status']

temp = test['status']

clf3 = tree.DecisionTreeClassifier()
clf3.fit(X, y)
pred3 = clf3.predict(test[features])
"""
for i in range(len(pred3)):
	print('Got ' + str(pred3[i]) + ', Expected ' + str(temp[i]))
"""
# should get 0
#print(clf3.predict([[214.28900,260.27700,77.97300,0.00567,0.00003,0.00295,0.00317,0.00885,0.01884,0.19000,0.01026,0.01161,0.01373,0.03078,0.04398,21.20900,0.462803,0.664357,-5.724056,0.190667,2.555477,0.148569]]))
print(accuracy_score(pred3, temp))