import glob
import json
import re
import pandas as pd
import csv


with open('data_exp.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';')
    for i in glob.glob('raw/*.json'):
        print(i)
        with open(i, encoding='utf8') as infile:
            month_news = json.loads(infile.read())
            for article in month_news:
                if 'main' not in article['headline'] or article['headline']['main'] is None:
                    continue
                # if 'lead_paragraph' in article and article['lead_paragraph'] is not None:
                #     article['headline']['main'] += '. ' + article['lead_paragraph']
                # if 'seo' in article['headline'] and article['headline']['seo'] is not None:
                #     article['headline']['main'] += '. ' + article['headline']['seo']
                # article['headline']['main'] = re.sub('\t|\n|^"|"$', '', article['headline']['main'])
                writer.writerow([article['pub_date'], article['headline']['main']])
