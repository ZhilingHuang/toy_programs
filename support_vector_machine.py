# Support vector machine: http://scikit-learn.org/stable/modules/svm.html
from sklearn import svm
import numpy as np

train_X = []
train_Y = []
with open('breast_cancer_shuffled_train') as f:
    line = f.readline()
    while line:
        l_split = line.split()
        y = int(float(l_split[0]))
        x = [int(float(t)) for t in l_split[1:]]
        train_X.append(x)
        train_Y.append(y)
        line = f.readline()

clf = svm.SVC(decision_function_shape='ovr',
              kernel='rbf')
print 'Starts training.'
clf.fit(train_X, train_Y)
print 'Finished training.'

dev_X = []
dev_Y = []
with open('breast_cancer_shuffled_dev') as f:
    line = f.readline()
    while line:
        l_split = line.split()
        y = int(float(l_split[0]))
        x = [int(float(t)) for t in l_split[1:]]
        dev_X.append(x)
        dev_Y.append(y)
        line = f.readline()

dev_prediction = clf.predict(dev_X)
print 'dev accuracy: ' + str(sum(np.array(dev_Y) == dev_prediction) * 1.0 / len(dev_X)) + '.'


