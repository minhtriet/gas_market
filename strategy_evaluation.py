import argparse
from datetime import date
from os import path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

from util import io


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

window = config['window']
predict_length = config['predict_length']
parser = argparse.ArgumentParser(description='buying strategy parsing')
parser.add_argument('--from_day', type=str, help='start buying date', required=True)
parser.add_argument('--to_day', type=str, help='to buying date', required=True)
args = parser.parse_args()

scaler = joblib.load('scaler.pkl')
market = io.read_future_market_v2('gpl')
news = io.load_news('spacy')
market = market.join(news, how='left')
market.fillna(0, inplace=True)

sess, graph = load_tf_model()
# run through days and
args.to_day = date(int(args.to_day.split('-')[0]), int(args.to_day.split('-')[1]), int(args.to_day.split('-')[2]))
date_range = pd.date_range(args.from_day, args.to_day)
for day in date_range:
    last_n_days = pd.DataFrame(_feed_past_data(market, day, window), copy=True)
    last_n_days['price'] = scaler.transform(last_n_days['price'].values.reshape(1, -1)).squeeze()
    last_n_days = last_n_days.values[np.newaxis, ...]
    inputs = graph.get_tensor_by_name('input:0')
    output = graph.get_tensor_by_name('output:0')
    prediction_result = sess.run(output, feed_dict={inputs: last_n_days})
    actual_result = _feed_past_data(market, day + predict_length)
    mean_squared_error(prediction_result, actual_result)
