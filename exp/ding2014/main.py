import sys

# meant to be execute in project root
sys.path.append('.')
from util import io
from sklearn.feature_extraction.text import TfidfVectorizer
import subprocess
from os import path


def clean_and_transform(document):
    """
    tfidf features
    :param document:
    :return:
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit(document)
    return tfidf


train_news_fn = 'train_news.txt'
test_news_fn = 'test_news.txt'
train_events_fn = 'reverb_train_news.txt'
train_events_fn = 'reverb_test_news.txt'
base_path = path.join('exp', 'ding2014')
# reverb
if not path.isfile(train_news_fn):
    print('Extracting news')
    news = io.load_news(False)
    length_news = len(news)
    train_news = news[:int(length_news * 0.6)].values.squeeze()
    test_news = news[int(length_news * 0.6):].values.squeeze()
    with open(path.join(base_path, train_news_fn), 'w') as f:
        f.write(' '.join(train_news))
    print('Train file done')
    with open(path.join(base_path, test_news_fn), 'w') as f:
        test_news.tofile(f, test_news_fn)
    print('Test file done')
print("start")
subprocess.call(path.join(base_path, 'extract.sh ') + train_news_fn, shell=True)
subprocess.call(path.join(base_path, 'extract.sh ') + test_news_fn, shell=True)

# convert to word embedding
with open(train_events_fn) as f:
    train_events = f.read()
transformer = clean_and_transform(train_news)
transformer.transform(train_news)
transformer.transform(test_news)
print("end")

# convert news word embedding to event embedding
