from keras.layers import LSTM, Embedding
from keras.models import Sequential


def train(x_train, y_train, x_test, y_test, layer_shape, time_steps, epoch, learning_rate, predict_length,
          embed, vocab_size):
    model = Sequential()
    embedding_size = len(embed[0])
    model.add(Embedding(vocab_size + 1, embedding_size, weights=[embed],
                        input_length=None, trainable=False))

    for i in range(len(layer_shape)):
        if i < len(layer_shape) - 1:
            model.add(LSTM(layer_shape[i], return_sequences=True))
        else:
            model.add(LSTM(layer_shape[i], return_sequences=True))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train, y_train, validation_data=(x_test, y_test))
