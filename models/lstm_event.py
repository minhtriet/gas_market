import numpy as np
from keras import Model
from keras.initializers import Constant
from keras.layers import Embedding, Input, concatenate, Conv3D, Conv2D
from keras.preprocessing.sequence import pad_sequences


# def _padding(event_array, max_len, events_per_day, word_per_news):
def _padding(event_array, ):
    '''
    5d array from 2d array
    :param event_array:
    :param max_len:
    :param events_per_day:
    :param word_per_news:
    :return:
    '''
    useless_news = 0
    # news_output = np.zeros(shape=(event_array.shape[0], event_array.shape[1], ))
    news_output = [[[] * event_array.shape[1]] * event_array.shape[0]]
    # pad events

    for batch in range(event_array.shape[0]):
        # for day in range(event_array.shape[1])
        # find number of biggest and longest events in this batch
        # fix the integer ops up in news
        for day in range(event_array.shape[1]):
            if type(event_array[batch, day]) == int:
                event_array[batch, day] = [[]]
        eventful_day_events = len(max(event_array[batch, :], key=len))
        for day in range(event_array.shape[1]):
            temp = [y for x in event_array[batch, day] for y in x]
            if len(temp) == 0:
                temp = event_array[batch, day]
            else:
                longest_event = len(max(temp, key=len))  # only update longest event when not null
            try:
                news_output[batch][day].append(pad_sequences(temp, maxlen=longest_event, dtype=object))
            except:
                pass
    print('useless news', useless_news)
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
    # x_extend_price = np.expand_dims(x_extend_price, 4)
    return x_extend_price


def train(x_train, y_train, x_test, y_test, layer_shape, time_steps, epoch, learning_rate, predict_length,
          embed, words_per_news):
    words_per_news = 7
    event_per_days = 5

    x_train_price = _process_price(x_train, event_per_days, words_per_news)
    x_test_price = _process_price(x_test, event_per_days, words_per_news)

    x_train_events = x_train[:, :, 1]
    x_train_events = _padding(x_train_events)

    x_test_events = x_test[:, :, 1]
    # x_test_events = _padding(x_test_events, words_per_news)

    seq_length = x_train.shape[1]
    num_words = len(embed) + 1
    # price_input = Input(batch_shape=(None, seq_length, event_per_days, words_per_news, 1), name='price_input', dtype="float32")
    # event_input = Input(batch_shape=(None, seq_length, event_per_days, words_per_news), name='event_input')
    price_input = Input(name='price_input', dtype="float32", shape=(10, None, 1))
    event_input = Input(name='event_input', shape=(10, None))
    emb = Embedding(input_dim=num_words, output_dim=300, embeddings_initializer=Constant(embed),
                    mask_zero=False, trainable=False)(event_input)
    print(emb._keras_shape)
    total_input = concatenate([emb, price_input], axis=3)
    print(total_input._keras_shape)
    conv1 = Conv2D(layer_shape[0], kernel_size=(event_per_days, 3), activation='relu')(total_input)
    out = Conv2D(3, kernel_size=(1, 1), activation='linear')(conv1)
    # flat = Flatten()(conv1)
    # out = Dense(3)(flat)
    print(out._keras_shape)
    model = Model(inputs=[price_input, event_input], outputs=[out])
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    # assert x_test_events.shape == (339, 10, 5, 7)
    # assert x_test_events.shape == (339, 10, 5, 7)
    # x_train_events = np.zeros(shape=)
    #`x_test_events = np.zeros(shape=(227, 10, 5, 7))
    price = [[]]
    model.fit([x_train_price, x_train_events], [y_train], validation_data=([x_test_price, x_test_events], [y_test]))
