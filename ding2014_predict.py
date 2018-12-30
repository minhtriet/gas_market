import argparse
import pickle
from os import path

import numpy as np
from keras.layers import Dense
from keras.models import Sequential

from util import data_generator

parser = argparse.ArgumentParser(description='data related parameters')
parser.add_argument('--is_regress', type=int, default=1, required=True, choices=[0, 1])
args = parser.parse_args()


print('Loading data')
for pred_length in range(1, 7):
    print(pred_length)
    x_train, x_test, y_train, y_test = data_generator.generate(30, future=True, train_percentage=0.7, embed=False,
                                                               stride=1, is_regress=args.is_regress,
                                                               predict_length=pred_length)
    transformer = pickle.load(open(path.join('exp', 'ding2014', 'tfidf.pkl'), 'rb'))
    print('Transforming')
    # preprocessing
    x_train[x_train == 0] = ''
    x_test[x_test == 0] = ''
    if not args.is_regress:
        y_test = np.array(y_test > 0, dtype=int)
        y_train = np.array(y_train > 0, dtype=int)
    # Statistik
    if not args.is_regress:
        unique, counts = np.unique(y_train, return_counts=True)
        print(dict(zip(unique, counts)))
    print(np.random.choice(y_train.flat, 5))
    if not args.is_regress:
        unique, counts = np.unique(y_test, return_counts=True)
        print(dict(zip(unique, counts)))
    print(np.random.choice(y_test.flat, 5))
    # End
    shape_2, shape_3 = transformer.transform(x_train[0, :, 1]).toarray().shape
    tf_xtrain = np.empty(shape=(x_train.shape[0], shape_2 * shape_3))
    tf_xtest = np.empty(shape=(x_test.shape[0], shape_2 * shape_3))
    for i in range(x_train.shape[0]):
        tf_xtrain[i] = transformer.transform(x_train[i, :, 1]).toarray().reshape(shape_2 * shape_3)
    for i in range(x_test.shape[0]):
        tf_xtest[i] = transformer.transform(x_test[i, :, 1]).toarray().reshape(shape_2 * shape_3)
    print('Transform finished')
    batch_size = 16

    np.random.seed(42)
    # create model
    if args.is_regress:
        model = Sequential()
        model.add(Dense(32, input_shape=(shape_2 * shape_3,), activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(pred_length))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        model.summary()
        model.fit(tf_xtrain, y_train, validation_data=(tf_xtest, y_test), batch_size=batch_size, epochs=32)
        model.save('ding2014.h5')
    else:
        model = Sequential()
        model.add(Dense(4, input_shape=(shape_2 * shape_3,), activation='relu'))
        model.add(Dense(2, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        model.fit(tf_xtrain, y_train, validation_data=(tf_xtest, y_test), batch_size=batch_size, epochs=5)
        model.save('ding2014.h5')
