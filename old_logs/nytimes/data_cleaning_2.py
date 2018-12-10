import glob
import json
import re
from os.path import basename

import pandas as pd


def _is_keywords_in_string(keywords, s):
    if s is None:
        return False
    return any(re.search(r"\b" + re.escape(x) + r"\b", s) for x in keywords)


def abstract_headline_integrate():
    for i in glob.glob('raw/*.json'):
        print(i)
        processed = []
        with open(i, encoding='utf8') as infile:
            month_news = json.loads(infile.read())
            for article in month_news['response']['docs']:
                if 'headline' in article and len(article['headline']) > 0 and 'lead_paragraph' in article:
                    article['headline']['main'].append(' ').append(article['lead_paragraph'])
                    processed.append(article)
        with open('temp_%s' % basename(i), 'w', encoding='utf-8') as outfile:
            json.dump(processed, outfile)


abstract_headline_integrate()
news = pd.DataFrame()
for filename in glob.glob('*.json'):
    news = news.append(pd.read_json(filename), sort=True)
news = news.set_index('pub_date')
# news['info'] = news['abstract'].fillna(news['lead_paragraph']).fillna(news['snippet'])
news['info'] = [x['main'] for x in news.headline.values]
news = news[['info', 'abstract', 'headline', 'lead_paragraph', 'snippet', 'keywords']]
news.dropna(inplace=True, subset=['info'])
# news['info'].str.split('.')[0]
news.to_csv('data.csv')
