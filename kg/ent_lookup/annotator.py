import pandas as pd
import requests
from itertools import groupby   # remove duplicates if elements has same text
from bs4 import BeautifulSoup
from operator import attrgetter as ga
from pprint import pprint

base_urls = ['http://api.dbpedia-spotlight.org/en/annotate',
             'http://api.dbpedia-spotlight.org/de/annotate']

news = pd.read_csv('../../old_logs/reuters/temp.csv')
text = 'Germany: Utility Gets Part of Gas Field. E.On Ruhrgas, a German utility, and Gazprom signed a deal giving E.On Ruhrgas a stake in the Yuzhno-Russkoye natural gas field in Siberia.'

for i, row in news.iterrows():
    payload = {'text': row['info'], 'confidence': 0.5}
    entities = []
    for base_url in base_urls:
        response = requests.post('%s' % base_url, data=payload)
        soup = BeautifulSoup(response.text, 'html.parser')
        entities.extend(soup.find_all('a'))
    entities_set = [next(g) for k, g in groupby(sorted(entities, key=ga('string')), key=ga('string'))]
    pprint(entities_set)
    break
