import argparse
from datetime import date

import numpy as np
import pandas as pd
import spacy
import yaml
from nltk.corpus import stopwords
from numpy import linspace

nlp = spacy.load('en_core_web_lg')
from util import io
from models import lstm_event
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

# keras var
future_price_scaler = joblib.load('scaler.pkl')


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
        market_segment = market.iloc[loc - arguments[0]:loc].values
        market_segment = np.expand_dims(market_segment, axis=0)

        x_train_price = lstm_event._process_price(market_segment, event_per_days=5, words_per_news=15)
        x_train_events = market_segment[:, :, 1]
        x_train_events = lstm_event._padding(event_array=x_train_events, word_index=word2idx)
        return x_train_price, x_train_events


def buy(market, day, amount):
    assert amount >= 0
    global min_price
    log.loc[day, 'reservoir'] += amount
    log.loc[day, 'cost'] += _feed_past_data(market, day) * amount
    log.loc[day, 'volume'] = amount
    min_price = 9999


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


def should_buy(market, model, day):
    global min_price, buy_day
    print(day)
    if log['goal'].loc[day] < log['reservoir'].loc[day]:
        return False
    price, events_padded = _feed_past_data(market, day, 10)
    prediction_result = model.predict([price, events_padded])
    if not _update_buy_day(prediction_result):
        buy_day -= 1
    if buy_day == 0:
        return True
    return False


def _evaluate(market, on_day, for_day, movement):
    distance = (for_day - on_day).days
    expected_price = _feed_past_data(market, on_day) + movement * distance
    log.loc[for_day, 'evaluate'] = (log.loc[for_day, 'evaluate'] * (distance - 1) + expected_price)[0][0] / distance


def strategy(market):
    reservoir_index = list(log.columns).index('reservoir')
    goal_index = list(log.columns).index('goal')
    model = load_model('spacy.m5')
    # loop by position instead of date to prevent date without data
    for i in range(len(log)):
        log.iloc[i, reservoir_index] = log.iloc[i - 1, reservoir_index]
        # if should_buy(market, rank_function_name, log.index[i].date()) or (i == len(log) - 1):
        if should_buy(market, model, log.index[i].date()):
            buy(original_market, log.index[i], log.iloc[i, goal_index] - log.iloc[i, reservoir_index])

    log.to_csv('3d.csv', float_format="%.4f")
    with open('total_3d.txt', 'w') as f:
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

market = io.read_spot_market_v2('gpl')
original_market = market.copy()
market['price'] = future_price_scaler.transform(market['price'].values.reshape(-1, 1))
news = io.load_news(embed='none')
events = io.load_events()
market = market.join(events, how='left')
market.fillna(0, inplace=True)
# choose only between start_day - window and end_day
end_goal = 1200
min_price = 9999
# corpus
corpus = news.loc[:'2013-04-11', 0].values  # 60%
vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, max_features=128000)
x_train_onehot = vectorizer.fit_transform(corpus)
word2idx = {nlp(word)[0].lemma_: idx for idx, word in enumerate(vectorizer.get_feature_names())}
embeddings_index = np.zeros((len(vectorizer.get_feature_names()) + 1, 300))
for word, idx in word2idx.items():
    embedding = nlp.vocab[word].vector
    embeddings_index[idx] = embedding
word2idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}
# end corpus

strategy_name = ['3d']
look_back = [window]
look_back_dict = dict(zip(strategy_name, look_back))

args.to_day = date(int(args.to_day.split('-')[0]), int(args.to_day.split('-')[1]), int(args.to_day.split('-')[2]))

log = pd.DataFrame(0, index=market.loc[args.from_day:args.to_day].index,
                   columns=['cost', 'reservoir', 'evaluate', 'volume'])
log['goal'] = linspace(0, end_goal, len(log), dtype=int)
strategy(market)
