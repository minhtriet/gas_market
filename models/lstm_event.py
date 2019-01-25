from keras.layers import LSTM, Embedding, Input
import pdb
from keras.initializers import Constant
from keras import Model

def train(x_train, y_train, x_test, y_test, layer_shape, time_steps, epoch, learning_rate, predict_length,
          embed, vocab_size):
    x_train_price = x_train[:, 0]
    x_train_events = x_train[:, 1]
    x_test_price = x_test[:, 0]
    x_test_events = x_test[:, 1]
    seq_length = x_train.shape[1]
    num_words = len(embed) + 1
    input1 = Input(shape=(seq_length,), dtype=float)
    pdb.set_trace()
    emb = Embedding(input_dim=num_words, output_dim = 300, embeddings_initializer=Constant(embed),
            input_length=seq_length, mask_zero=True,trainable=False)(input1)

    input2 = Input(shape=(seq_length,6 ))
    x = keras.layers.concatenate([emb, input2],axis=2)

    lstm = LSTM(64, return_sequences=True)(x)
    model = Model(inputs = [], outputs = []) 
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train, y_train, validation_data=(x_test, y_test))
