import glob

import csv
import json
import re
from os.path import basename

import pandas as pd


def _is_keywords_in_string(keywords, s):
    if s is None:
        return False
    return any(re.search(r"\b" + re.escape(x) + r"\b", s) for x in keywords)


def clean_data():
    free_text_keywords = ['gas']
    for i in glob.glob('raw/*.json'):
        print(i)
        processed = []
        with open(i, encoding='utf8') as infile:
            month_news = json.loads(infile.read())
            for article in month_news:
                for key in ['lead_paragraph', 'abstract', 'snippet']:
                    if _is_keywords_in_string(free_text_keywords, article.get(key)):
                        processed.append({'pub_date':article.get('pub_date'), 'info':article.get(key)})
                        break
                    if _is_keywords_in_string(free_text_keywords, article.get('headline').get('main')):
                        processed.append({'pub_date': article.get('pub_date'), 'info': article.get('headline').get('main')})
                        break
        print(len(processed))
        with open('processed_%s' % basename(i), 'w', encoding='utf-8') as outfile:
            json.dump(processed, outfile)


print('Clean data begin')
clean_data()
print('Clean data finished')
news = pd.DataFrame()
for i in glob.glob('*.json'):
    print(i)
    news = news.append(pd.read_json(i))
print('concatenate news completed')
news = news.set_index('pub_date')
news.dropna(inplace=True, subset=['info'])
news.to_csv('data.csv', )
