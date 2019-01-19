import sys
sys.path.insert(0, '.')

import numpy as np
from ev_extractor import pipeline
from util import io

news = io.load_news(embed='none')
news = news.groupby(news.index)[0].apply(lambda x: [x])
for i, _ in news.iteritems():
    splits = []
    print(news.loc[i])
    for article in news.loc[i]:
        splits.append(pipeline.pipeline(article[0]))
    try:
        l = np.array(splits).size 
        news.loc[i] = splits
    except:
news.to_csv('event.csv')
