from util import data_generator
from sklearn import svm

x_train, x_test, y_train, y_test = data_generator.baseline_crf(train_percentage=0.7, sft=1, future=True, embed=False)

clf = svm.SVR()
clf.fit(x_train, y_train)
clf.predict([[1, 1]])

