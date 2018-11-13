from keras import optimizers
from keras.layers import Dense, LSTM
from keras.models import Sequential


def lstm_nowindow(feature, output_len=5):
    model = Sequential()
    model.add(LSTM(64, input_shape=(None, feature), return_sequences=True))
    model.add(LSTM(output_len, return_sequences=True))
    sgd = optimizers.Adam(lr=0.1)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mae'])
    return model


def lstm(input_shape, output_len=5):
    model = Sequential()
    # model.add(LSTM(64, input_shape=input_shape, return_sequences=True))# input_shape=(20,1)
    model.add(LSTM(1, batch_input_shape=input_shape, stateful=True))
    model.add(Dense(output_len))
    sgd = optimizers.Adam(lr=0.1)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mae'])
    return model


def lstm_stateful(batch_input_shape, no_classes):
    # design network
    model = Sequential()
    model.add(LSTM(64, batch_input_shape=batch_input_shape, return_sequences=False, stateful=True))
    model.add(Dense(no_classes, activation='softmax'))
    sgd = optimizers.SGD(lr=0.0001, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
