from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.metrics import accuracy_score

train = pd.read_csv('parkinsons_train.csv')
test = pd.read_csv('parkinsons_test.csv')

features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]
# 64%
X = train[features]
y = train['status']

temp = test['status']

clf = GaussianNB()
clf.fit(X,y)
pred = clf.predict(test[features])
print('GaussianNB: ' + str(accuracy_score(pred, temp)))