import sys
sys.path.insert(0, '.')

from ev_extractor import pipeline
from util import io

news = io.load_news(embed='none')
news = news.groupby(news.index)[0].apply(lambda x: [x])
for i, _ in news.iteritems():
    events = []
    print(news.loc[i])
    for article in news.loc[i]:
        events.append(pipeline.pipeline(article[0]))
    news.loc[i] = events
news.to_csv('event.csv')
