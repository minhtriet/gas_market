import glob
import json
import re
import pandas as pd


def read():
    for i in glob.glob('raw/*.json'):
        print(i)
        processed = []
        with open(i, encoding='utf8') as infile:
            month_news = json.loads(infile.read())
            for article in month_news:
                if 'main' not in article['headline'] or article['headline']['main'] is None:
                    continue
                if 'lead_paragraph' in article and article['lead_paragraph'] is not None:
                    article['headline']['main'] += '. ' + article['lead_paragraph']
                if 'seo' in article['headline'] and article['headline']['seo'] is not None:
                    article['headline']['main'] += '. ' + article['headline']['seo']
                article['headline']['main'] = re.sub('\t|\n|^"|"$', '', article['headline']['main'])
                processed.append(article)
        break
    return processed


all_json = read()
news = pd.read_json(json.dumps(all_json))
# news = news.set_index('pub_date')
news = news.set_index('_id')
news['info'] = [x['main'] for x in news.headline.values]
news = news[['info']]
news.dropna(inplace=True, subset=['info'])
# news['info'].str.split('.')[0]
news.to_csv('data_exp.csv', sep='\t')
