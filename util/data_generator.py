import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from util import io


def generate(window, stride, predict_length=2, save_scaler=False, train_percentage=0.6, embed=True):
    train = io.read_future_market_v2('gpl')
    # train = pd.concat(train, pd.get_dummies(train.index.month, prefix='m'))
    news = io.load_news(embed)
    # the training label, don't scale
    original_price = np.array(train.values).squeeze()
    y = np.array(
        [original_price[i:i + predict_length] for i in range(window, len(original_price) - predict_length, stride)])

    # do not scale one hot encoding for now, scale later and see if things change, try to explain, gives out the source
    # transfer learning word embedding, spacy custom training
    # is data rich enough to form a triplt or knowledge graph, (also for linked news)
    future_price_scaler = StandardScaler()
    train = train.join(news, how='left')
    train.fillna(0, inplace=True)
    split = int(len(train) * train_percentage)

    train['price'][:split] = future_price_scaler.fit_transform(train['price'][:split].values.reshape(-1, 1)).squeeze()
    train['price'][split:] = future_price_scaler.transform(train['price'][split:].values.reshape(-1, 1)).squeeze()
    train = np.array(train)
    # scaled_ = np.concatenate((scaled_, future_price_scaler.transform(train[split:])))
    x = np.array([train[i:i + window] for i in range(0, len(train) - window, stride)])
    if len(x) > len(y):
        x = x[:len(y)]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_percentage, shuffle=False)
    #    np.savetxt("train_x.csv", x[0], delimiter=",", fmt='%10.5f')
    #    np.savetxt("train_y.csv", y, delimiter=",", fmt='%10.3f')

    # how a news has big effect on event long term and short term
    if save_scaler:
        joblib.dump(future_price_scaler, 'scaler.pkl')

    return x_train, x_test, y_train, y_test
