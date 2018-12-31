import argparse
from datetime import date
from os import path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from numpy import linspace
from sklearn.externals import joblib

from util import io

# scaler for NN methods
scaler = joblib.load('scaler.pkl')

# global vals
min_price = 1000
# buy day is the index at which gas should be bought
buy_day = -1

# keras variables
bought = False
# model_movement = load_model('trained/keras/multistep_std_05_10_09_15.h5')

# tf variable
tf_version = '04_11_23_15'
tf_model_path = path.join('logs', 'tf_lstm_%s' % tf_version, 'model.meta')
tf_weight_path = path.join('logs', 'tf_lstm_%s' % tf_version)
sess = tf.Session()
saver = tf.train.import_meta_graph(tf_model_path)
graph = tf.get_default_graph()
saver.restore(sess, tf.train.latest_checkpoint(tf_weight_path))

# news
news = io.load_news()


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


def buy(market, day, amount):
    assert amount >= 0
    log.loc[day, 'reservoir'] += amount
    log.loc[day, 'cost'] += _feed_past_data(market, day) * amount
    log.loc[day, 'volume'] = amount


def _update_buy_day(prediction):
    """
    :param prediction: an array of prediction
    :return: True if found new minimum, False otherwise
    """
    global min_price, buy_day
    if np.min(prediction) < min_price:
        min_price = np.min(prediction)
        buy_day = np.argmin(prediction) + 1
        return True
    return False


def should_buy(market, method, day):
    global min_price, buy_day
    print(day)
    if log['goal'].loc[day] < log['reservoir'].loc[day]:
        return False
    if method == 'baseline':
        return True
    if method == 'old':
        if _feed_past_data(market, day) > _feed_past_data(market, day, 1)['price'].values[0]:
            return True
    last_n_days = pd.DataFrame(_feed_past_data(market, day, window), copy=True)
    last_n_days['price'] = scaler.transform(last_n_days['price'].values.reshape(1, -1)).squeeze()
    last_n_days = last_n_days.values[np.newaxis, ...]
    if method == 'new_tf':
        inputs = graph.get_tensor_by_name('input:0')
        output = graph.get_tensor_by_name('output:0')
        prediction_result = sess.run(output, feed_dict={inputs: last_n_days})
        if not _update_buy_day(prediction_result):
            buy_day -= 1
    if buy_day == 0:
        return True
    return False


def _evaluate(market, on_day, for_day, movement):
    distance = (for_day - on_day).days
    expected_price = _feed_past_data(market, on_day) + movement * distance
    log.loc[for_day, 'evaluate'] = (log.loc[for_day, 'evaluate'] * (distance - 1) + expected_price)[0][0] / distance


def strategy(market, rank_function_name):
    reservoir_index = list(log.columns).index('reservoir')
    goal_index = list(log.columns).index('goal')

    # loop by position instead of date to prevent date without data
    for i in range(len(log)):
        log.iloc[i, reservoir_index] = log.iloc[i - 1, reservoir_index]
        # if should_buy(market, rank_function_name, log.index[i].date()) or (i == len(log) - 1):
        if should_buy(market, rank_function_name, log.index[i].date()):
            buy(market, log.index[i], log.iloc[i, goal_index] - log.iloc[i, reservoir_index])

    log.to_csv('%s.csv' % rank_function_name, float_format="%.4f")
    with open('total_%s.txt' % rank_function_name, 'w') as f:
        f.write("%f\n%f\n%f" % (log['cost'].sum(), log.iloc[len(log) - 1, reservoir_index],
                                log['cost'].sum() / log.iloc[len(log) - 1]['reservoir']))


with open('config.yaml') as stream:
    try:
        config = yaml.load(stream)
        window = config['window']
    except yaml.YAMLError as exc:
        print(exc)

parser = argparse.ArgumentParser(description='buying strategy parsing')
parser.add_argument('--from_day', type=str, help='start buying date', required=True)
parser.add_argument('--to_day', type=str, help='to buying date', required=True)
args = parser.parse_args()

market = io.read_future_market_v2('gpl')
spot = io.read_spot_market()
news = io.load_news()
market = market.join(news, how='left')
market.fillna(0, inplace=True)
# choose only between start_day - window and end_day
end_goal = 1200

strategy_name = ['new_tf', 'old', 'baseline']
look_back = [window, 1, 1]
look_back_dict = dict(zip(strategy_name, look_back))

args.to_day = date(int(args.to_day.split('-')[0]), int(args.to_day.split('-')[1]), int(args.to_day.split('-')[2]))

for strat in strategy_name:
    log = pd.DataFrame(0, index=market.loc[args.from_day:args.to_day].index,
                       columns=['cost', 'reservoir', 'evaluate', 'volume'])
    log['goal'] = linspace(0, end_goal, len(log), dtype=int)
    strategy(market, strat)
