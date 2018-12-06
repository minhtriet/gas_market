import glob
import pandas as pd

news = pd.DataFrame()
for i in glob.glob('*.json'):
    print(i)
    today = pd.read_json(i)['response']["results"]
    if len(today) > 0:
        for article in today:
            if article['webUrl'].count('/') > 7:  # https://www.theguardian.com/business/2008/dec/19/7 is News in brief
                news = news.append(article, ignore_index=True)

news.rename(columns={'webPublicationDate': 'pub_date', 'webTitle': 'info'}, inplace=True)
news = news.set_index('pub_date')
news.to_csv('data.csv')
