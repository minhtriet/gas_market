import numpy as np

from ev_extractor import pipeline
from util import io

news = io.load_news(embed='none')
for i, _ in news.iterrows():
    splits = []
    print(news.loc[i].values)
    for article in news.loc[i].values:
        splits.append(pipeline.pipeline(article[0]))
    try:
        l = len(splits)
        news.loc[i] = np.array(splits).reshape(l, 1)
    except:
        import pdb

        pdb.set_trace()
news.to_csv('event.csv')
