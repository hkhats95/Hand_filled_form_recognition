import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm, model_selection
from datetime import datetime

def Classifier():
    start = datetime.now()
    print(start)
    X = np.load('full_dataset.npy')
    y = np.load('full_label.npy')
    print('creation of training data')
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.01)


    clf = svm.SVC(gamma=0.001, C=100)
    print('training started')
    clf.fit(X_train, y_train)

    with open('characterrecognition.pickle', 'wb') as f:
        pickle.dump(clf, f)
    print(datetime.now()-start)
    start = datetime.now()
    pickle_in = open('characterrecognition.pickle','rb')
    clf = pickle.load(pickle_in)
    print('score prediction started')
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    end = datetime.now()
    print(end - start)


if __name__=='__main__':
    Classifier()


