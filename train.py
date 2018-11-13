import argparse

from keras.callbacks import TensorBoard

from models import tcn, lstm, lstm_tf
from util import data_generator

import yaml


def run_tcn(d, x_train, x_test, y_train, y_test):
    model, param_str = tcn.dilated_tcn(output_slice_index='last',  # try 'first'.
                                       num_feat=1,
                                       num_classes=y_train.shape[1],
                                       nb_filters=64,
                                       kernel_size=8,
                                       dilatations=[1, 2, 4, 8],
                                       nb_stacks=8,
                                       max_len=x_train[0:1].shape[1],
                                       activation='norm_relu',
                                       use_skip_connections=False,
                                       return_param_str=True)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_test.shape = {y_test.shape}')

    model.summary()
    tb_callback = TensorBoard(log_dir=d, histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                              write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None,
                              embeddings_data=None)
    model.fit(x_train, y_train, epochs=32, validation_data=(x_test, y_test), callbacks=[tb_callback])


def run_lstm(d, x_train, x_test, y_train, y_test):
    model = lstm.lstm((11, args.window, 1))
    model.summary()
    tb_callback = TensorBoard(log_dir=d, histogram_freq=0, write_graph=True, write_grads=False,
                              write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None, embeddings_data=None)
    model.fit(x_train, y_train, epochs=128, validation_data=(x_test, y_test),
              batch_size=11, callbacks=[tb_callback], shuffle=False)
    return model


parser = argparse.ArgumentParser(description='data related parameters')
parser.add_argument('--reset_state_window', help='Reset state after this length has been reached for stateful lstm',
                    default=30)
parser.add_argument('--stride', type=int, default=5)
parser.add_argument('--predict_length', type=int, default=3)
args = parser.parse_args()

with open('config.yaml') as stream:
    try:
        config = yaml.load(stream)
        window = config['window']
    except yaml.YAMLError as exc:
        print(exc)

x_train, x_test, y_train, y_test = data_generator.generate(window, train_percentage=0.6, stride=args.stride,
                                                           predict_length=args.predict_length)

lstm_tf.train(x_train, y_train, x_test, y_test, time_steps=args.window, layer_shape=[128, 32], learning_rate=0.005,
              epoch=512, predict_length=args.predict_length)
# 24.10 add news
# 30.10 lemmatize news