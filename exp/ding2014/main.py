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


def join(xtra_path):
    base_path = path.join('exp', 'ding2014')
    return path.join(base_path, xtra_path)


def parse_reverb(documents):
    events = []
    for d in documents:
        d = d.split('\t')        
        if len(d) == 18:
            print('S V O')
            events.append([' '.join([d[2], d[3], d[4]])])
        else:
            events.append([' '.join([d[2], d[3]])])
    return events


train_news_fn = 'train_news.txt'
test_news_fn = 'test_news.txt'
train_events_fn = 'reverb_train_news.txt'
test_events_fn = 'reverb_test_news.txt'
# reverb
if not path.isfile(train_news_fn):
    print('Extracting news')
    news = io.load_news(False)
    length_news = len(news)
    print('Loaded %d news' % length_news)
    train_news = news[:int(length_news * 0.6)].values.squeeze()
    test_news = news[int(length_news * 0.6):].values.squeeze()
    with open(join(train_news_fn), 'w') as f:
        f.write('\n'.join(train_news))
    print('Train file done')
    with open(join(test_news_fn), 'w') as f:
        f.write('\n'.join(test_news))
    print('Test file done')
if path.isfile(train_events_fn) and path.isfile(test_events_fn):
    print("start")
    subprocess.call(join('extract.sh ') + join(train_news_fn) + ' ' + join(train_events_fn), shell=True)
    subprocess.call(join('extract.sh ') + join(test_news_fn) + ' ' + join(test_events_fn), shell=True)

# convert to tfidf
with open(join(train_events_fn)) as f:
    train_events = f.read()
    train_events = train_events.split('\n')
    train_events = parse_reverb(train_events)
with open(test_events_fn) as  f:
    test_events = f.read()
    test_events = test_events.split('\n')
    test_events = parse_reverb(train_events)
transformer = clean_and_transform(train_events)

with open(tfidf_train_events, 'w') as f:
    f.write(transformer.transform(train_news))

with open(tfidf_test_events, 'w') as f:
    f.write(transformer.transform(train_news))
transformer.transform(test_news)
print("end")

# convert news word embedding to event embedding
