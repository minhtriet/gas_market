import pandas as pd

news = pd.read_csv('reuters_now.csv', sep='\t', names=['id', 'pub_date', 'info', 'url'])
print('Load completed')
news['pub_date'] = news['pub_date'].str.replace(r'[^0-9:AMP ]', '').str.strip()      # AMP = AM or PM
news['pub_date'] = news['pub_date'].str.replace(r'(?<=[0-9]) (?=(A|P))', '')     # space between hour and AM/PM
print('Regex completed')
news['pub_date'] = pd.to_datetime(news['pub_date'], format='%Y%m%d%H:%M%p', errors='coerce')
print('Time matching completed')
news = news.dropna()
news = news.set_index('pub_date').drop('id', axis=1)
news['info'] = news['info'].str.replace('"', '')
news.to_csv('data.csv')

