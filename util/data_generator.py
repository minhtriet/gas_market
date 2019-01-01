import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from util import io


def baseline_crf(train_percentage, sft, future, embed='none'):
    """
    news day n results in day n+1 price, shift index by 1 position
    :param train_percentage: train test split
    :return: x_train, y_train, x_test, y_test
    """
    if future:
        train = io.read_future_market_v2('gpl')
    else:
        train = io.read_spot_market_v2('gpl')
        train = train.drop(['ncg', 'Tradingday', 'Liefertag'], axis=1)
    train['price'] = train['price'].diff()
    train['price'] = train['price'] > 0
    news = io.load_news(embed)
    train = train.join(news, how='left')
    train['price'] = train['price'].shift(sft)
    train.dropna(inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(train.drop('price', axis=1), train['price'],
                                                        test_size=1 - train_percentage, shuffle=False)
    return x_train, x_test, y_train, y_test


def generate(window, stride, predict_length, future=True, save_scaler=True, train_percentage=0.6, embed=True,
             is_regress=True):
    """
    :param window: length of the predict data
    :param stride: stride
    :param predict_length: how much days will be predicted
    :param save_scaler: save scaler to be used later in reference
    :param future: read future or spot market
    :param train_percentage: train test split
    :param embed: embed the news or full text
    :param is_regress: False if classification
    :return: x_train, y_train, x_test, y_test
    """
    if future:
        train = io.read_future_market_v2('gpl')
    else:
        train = io.read_spot_market_v2('gpl')
    news = io.load_news(embed)
    # the training label, don't scale
    original_price = np.array(train.values).squeeze()
    if is_regress:
        y = np.array(
            [original_price[i:i + predict_length] for i in range(window, len(original_price) - predict_length, stride)])
    else:
        original_price = original_price[predict_length:] - original_price[0:-predict_length]
        y = np.array([original_price[i] for i in range(window, len(original_price) - predict_length, stride)])

    # do not scale one hot encoding for now, scale later and see if things change, try to explain, gives out the source
    # is data rich enough to form a triplet or knowledge graph, (also for linked news)
    future_price_scaler = StandardScaler()
    train = train.join(news, how='left')
    train.fillna(0, inplace=True)
    split = int(len(train) * train_percentage)

    train.loc[:split, 'price'] = future_price_scaler.fit_transform(
        train['price'][:split].values.reshape(-1, 1)).squeeze()
    train.loc[split:, 'price'] = future_price_scaler.transform(
        train['price'][split:].values.reshape(-1, 1)).squeeze()

    train = np.array(train)
    # scaled_ = np.concatenate((scaled_, future_price_scaler.transform(train[split:])))
    x = np.array([train[i:i + window] for i in range(0, len(train) - window, stride)])
    if len(x) > len(y):
        x = x[:len(y)]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_percentage, shuffle=False)

    # how a news has big effect on event long term and short term
    print("Saving scaler")
    if save_scaler:
        joblib.dump(future_price_scaler, 'scaler.pkl')

    print("Saving scaler done")
    return x_train, x_test, y_train, y_test
