import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn import model_selection

start = datetime.now()
print(start)

X = np.load('digits_dataset.npy')
y = np.load('digits_label.npy')


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)


clf = RandomForestClassifier(n_estimators=40)
clf.fit(X_train, y_train)

with open('newrandomforestdigit40.pickle', 'wb') as f:
    pickle.dump(clf, f)
print(datetime.now()-start)

start = datetime.now()
pickle_in = open('newrandomforestdigit40.pickle', 'rb')
clf = pickle.load(pickle_in)
print('score prediction started')
accuracy = clf.score(X_test, y_test)
print(accuracy)
end = datetime.now()
print(end - start)
print(len(X))


