import glob
import json

import pandas as pd

news = pd.DataFrame()
for i in glob.glob('*.json'):
    print(i)
    data = json.load(open(i))
    today = data["results"][0]['results']
    for article in today:
        if 'title' in article['title']:
            article['pub_date'] = article['lifecycle']['initialPublishDateTime']
            article.pop('lifecycle')
            article['info'] = article['title']['title']
            article.pop('title')
        news = news.append(article, ignore_index=True)

news.rename(columns={'webPublicationDate': 'pub_date', 'webTitle': 'info'}, inplace=True)
news = news.set_index('pub_date')
news.to_csv('data.csv')
