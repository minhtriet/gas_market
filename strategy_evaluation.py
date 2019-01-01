import argparse
import pickle
from os import path

import tensorflow as tf
import yaml
from sklearn.metrics import mean_squared_error


def load_tf_model(tf_version='04_11_23_15'):
    tf_model_path = path.join('logs', 'tf_lstm_%s' % tf_version, 'model.meta')
    tf_weight_path = path.join('logs', 'tf_lstm_%s' % tf_version)
    sess = tf.Session()
    saver = tf.train.import_meta_graph(tf_model_path)
    graph = tf.get_default_graph()
    saver.restore(sess, tf.train.latest_checkpoint(tf_weight_path))
    return sess, graph


def _feed_past_data(market, d, *arguments):
    """
    :param market: the price series
    :param d: current day
    :param arguments: number of days to look back
    :return: news and price data
    """
    if not arguments:
        return market['price'].loc[d]
    else:
        loc = market.index.get_loc(d)
        return market.iloc[loc - arguments[0]:loc]


with open('config.yaml') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

parser = argparse.ArgumentParser(description='buying strategy parsing')
parser.add_argument('--embed', type=str, required=True, choices=config['embed'])
args = parser.parse_args()
embed = args.embed
sess, graph = load_tf_model('01_01_02_13')

print('Loading test set')
with open(r"x_test_%s.pickle" % embed, "rb") as output_file:
    x_test = pickle.load(output_file)
with open(r"y_test_%s.pickle" % embed, "rb") as output_file:
    y_test = pickle.load(output_file)
print('Loading test completed')
last_n_days = x_test
actual_result = y_test
inputs = graph.get_tensor_by_name('input:0')
output = graph.get_tensor_by_name('output:0')
prediction_result = sess.run(output, feed_dict={inputs: last_n_days})
print(mean_squared_error(prediction_result, actual_result))
