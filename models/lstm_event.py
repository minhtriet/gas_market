import numpy as np
from keras import Model
from keras.initializers import Constant
from keras.layers import LSTM, Embedding, Input, concatenate


def train(x_train, y_train, x_test, y_test, layer_shape, time_steps, epoch, learning_rate, predict_length,
          embed, vocab_size):
    x_train_price = x_train[:, :, 0]
    x_train_price = np.expand_dims(x_train_price, axis=2)
    x_train_events = x_train[:, :, 1]
    x_test_price = x_test[:, :, 0]
    x_test_price = np.expand_dims(x_test_price, axis=2)
    x_test_events = x_test[:, :, 1]
    seq_length = x_train.shape[1]
    num_words = len(embed) + 1

    price_input = Input(batch_shape=(None, seq_length, 1), dtype=float, name='price_input')
    event_input = Input(batch_shape=(None, seq_length), name='event_input')
    emb = Embedding(input_dim=num_words, output_dim=300, embeddings_initializer=Constant(embed),
                    mask_zero=True, trainable=False)(event_input)
    total_input = concatenate([emb, price_input])
    lstm_1 = LSTM(128, return_sequences=True)(total_input)
    lstm_2 = LSTM(32, return_sequences=True)(lstm_1)
    lstm_3 = LSTM(3)(lstm_2)

    # lstm = []
    # for i in range(len(layer_shape)):
    #     if i == 0:
    #         lstm.append(LSTM(layer_shape[i], return_sequences=True)(total_input))
    #     else:
    #         lstm.append(LSTM(3)(lstm[i - 1]))

    model = Model(inputs=[price_input, event_input], outputs=[lstm_3])
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit([x_train_price, x_train_events], [y_train], validation_data=([x_test_price, x_test_events], [y_test]))
