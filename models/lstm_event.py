import numpy as np

np.random.seed(0)
from keras import Model
from keras.initializers import Constant
from keras.layers import Embedding, Input, concatenate, Conv3D, Flatten, Dense
from keras.preprocessing.sequence import pad_sequences

import spacy

nlp = spacy.load('en_core_web_sm')


def to_sequence(text, word_index):
    indexes = [word_index[nlp(str(word))[0].lemma_] for word in text if nlp(str(word))[0].lemma_ in word_index]
    return indexes


def _padding(event_array, word_index, events_per_day=5, word_per_news=15):
    '''
    5d array from 2d array
    :param event_array:
    :param max_len:
    :param events_per_day:
    :param word_per_news:
    :return:
    '''
    news_output = np.zeros(shape=(event_array.shape[0], event_array.shape[1], events_per_day, word_per_news),
                           dtype=object)
    # pad events
    for batch in range(event_array.shape[0]):
        for day in range(event_array.shape[1]):
            if (type(event_array[batch, day]) == int) or (event_array[batch, day].size == 0):
                temp = [['0']] * events_per_day
            else:
                temp = [y for x in event_array[batch, day] for y in x]
                if len(temp) > events_per_day:
                    temp = np.take(temp, np.random.choice(range(len(temp)), events_per_day, replace=False)).tolist()
                elif len(temp) < events_per_day:
                    # randomly patch
                    for i in range(events_per_day - len(temp)):
                        temp.append(['0'])
                    np.random.shuffle(temp)
            # if type(temp[0]) == np.str_:
            #     temp = [temp]
            # convert to index
            for event_index in range(len(temp)):
                temp[event_index] = to_sequence(temp[event_index], word_index)
            temp = pad_sequences(temp, word_per_news, dtype=object)
            news_output[batch, day] = temp
    assert type(news_output) == np.ndarray
    # time step, price
    # time step, 2d-word, 300-dim
    # 2d-word
    return news_output


def _process_price(price, event_per_days, words_per_news):
    def _price_transform(ev_per_day, word_per_news, chunk):
        '''
        a chunk of 2d becomes 5d
        :return:
        '''
        temp = [[e] * ev_per_day * word_per_news for e in chunk]
        return np.array(temp).flatten().reshape(-1, ev_per_day, word_per_news)

    x_train_price = price[:, :, 0]
    # x_train_price = np.expand_dims(x_train_price, axis=2)
    x_extend_price = np.array([_price_transform(event_per_days, words_per_news, x_train_price[chuck_index])
                               for chuck_index in range(len(x_train_price))])
    x_extend_price = np.expand_dims(x_extend_price, 4)
    return x_extend_price


def train(x_train, y_train, x_test, y_test, layer_shape, time_steps, epoch, learning_rate, predict_length,
          embed, words_per_news, wordindex):
    event_per_days = 5

    x_train_price = _process_price(x_train, event_per_days, words_per_news)
    x_test_price = _process_price(x_test, event_per_days, words_per_news)

    x_train_events = x_train[:, :, 1]
    x_train_events = _padding(event_array=x_train_events, word_index=wordindex)

    x_test_events = x_test[:, :, 1]
    x_test_events = _padding(event_array=x_test_events, word_index=wordindex)

    num_words = len(embed) + 1
    # price_input = Input(batch_shape=(None, seq_length, event_per_days, words_per_news, 1), name='price_input', dtype="float32")
    # event_input = Input(batch_shape=(None, seq_length, event_per_days, words_per_news), name='event_input')
    price_input = Input(name='price_input', dtype="float32", shape=(10, event_per_days, words_per_news, 1))
    event_input = Input(name='event_input', dtype="int32", shape=(10, event_per_days, words_per_news))
    emb = Embedding(input_dim=num_words, output_dim=300, embeddings_initializer=Constant(embed),
                    mask_zero=False, trainable=False)(event_input)
    total_input = concatenate([emb, price_input], axis=4)
    print(total_input._keras_shape)
    conv1 = Conv3D(layer_shape[0], kernel_size=(event_per_days, 3, 3), activation='relu')(total_input)
    flat = Flatten()(conv1)
    out = Dense(3)(flat)
    print(out._keras_shape)
    # model = Model(inputs=[price_input, event_input], outputs=[out])
    model = Model(inputs=[price_input, event_input], outputs=[out])
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit([x_train_price, x_train_events], [y_train], validation_data=([x_test_price, x_test_events], [y_test]))
