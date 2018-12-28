import pickle
from os import path

import numpy
from keras.layers import Dense
from keras.models import Sequential

from util import data_generator

print('Loading data')
for pred_length in range(1, 7):
    print(pred_length)
    x_train, x_test, y_train, y_test = data_generator.generate(30, future=True, train_percentage=0.7, embed=False,
                                                               stride=1, isRegress=False, predict_length=pred_length)
    transformer = pickle.load(open(path.join('exp', 'ding2014', 'tfidf.pkl'), 'rb'))
    print('Transforming')
    # preprocessing
    x_train[x_train == 0] = ''
    x_test[x_test == 0] = ''
    y_test = numpy.array(y_test > 0, dtype=int)
    y_train = numpy.array(y_train > 0, dtype=int)
    # Statistik
    unique, counts = numpy.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))
    unique, counts = numpy.unique(y_test, return_counts=True)
    print(dict(zip(unique, counts)))
    # End
    shape_2, shape_3 = transformer.transform(x_train[0, :, 1]).toarray().shape
    tf_xtrain = numpy.empty(shape=(x_train.shape[0], shape_2 * shape_3))
    tf_xtest = numpy.empty(shape=(x_test.shape[0], shape_2 * shape_3))
    for i in range(x_train.shape[0]):
        tf_xtrain[i] = transformer.transform(x_train[i, :, 1]).toarray().reshape(shape_2 * shape_3)
    for i in range(x_test.shape[0]):
        tf_xtest[i] = transformer.transform(x_test[i, :, 1]).toarray().reshape(shape_2 * shape_3)
    print('Transform finished')
    batch_size = 16

    numpy.random.seed(42)
    # create model
    model = Sequential()
    model.add(Dense(4, input_shape=(shape_2 * shape_3,), activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(tf_xtrain, y_train, validation_data=(tf_xtest, y_test), batch_size=batch_size, epochs=5)
    model.save('ding2014.h5')
