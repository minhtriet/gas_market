import argparse
import os
import pickle

import yaml
from keras.callbacks import TensorBoard

from models import tcn, lstm, lstm_tf, lstm_regularize_tf
from util import data_generator


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


with open('config.yaml') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)
parser = argparse.ArgumentParser(description='data related parameters')
parser.add_argument('--reset_state_window', help='Reset state after this length stateful lstm', default=30)
parser.add_argument('--stride', type=int, default=5)
parser.add_argument('--predict_length', type=int, default=5)
parser.add_argument('--embed', type=str, required=True, choices=config['embed'])
args = parser.parse_args()
embed = args.embed
window = config['window']
if not os.path.isfile('x_train_%s.pkl' % embed):
    x_train, x_test, y_train, y_test = data_generator.generate(window, future=True, train_percentage=0.6, stride=args.stride, embed=args.embed, predict_length=args.predict_length)
    with open(r"x_train_%s.pickle" % embed, "wb") as output_file:
        pickle.dump(x_train, output_file)
    with open(r"y_train_%s.pickle" % embed, "wb") as output_file:
        pickle.dump(y_train, output_file)
    with open(r"x_test_%s.pickle" % embed, "wb") as output_file:
        pickle.dump(x_test, output_file)
    with open(r"y_test_%s.pickle" % embed, "wb") as output_file:
        pickle.dump(y_test, output_file)
else:
    with open(r"x_train_%s.pickle" % embed, "rb") as output_file:
        x_train = pickle.load(output_file)
    with open(r"y_train_%s.pickle" % embed, "rb") as output_file:
        y_train = pickle.load(output_file)
    with open(r"x_test_%s.pickle" % embed, "rb") as output_file:
        x_test = pickle.load(output_file)
    with open(r"y_test_%s.pickle" % embed, "rb") as output_file:
        y_test = pickle.load(output_file)

# spacy
lstm_regularize_tf.train(x_train, y_train, x_test, y_test, time_steps=window, layer_shape=[128, 32], learning_rate=0.001, epoch=4096, predict_length=args.predict_length)
# fasttext
#lstm_tf.train(x_train, y_train, x_test, y_test, time_steps=window, layer_shape=[128, 32], learning_rate=0.005, epoch=512, predict_length=args.predict_length)
