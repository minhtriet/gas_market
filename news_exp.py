import pandas as pd

news = pd.read_csv('temp.csv')
news = news.set_index('pub_date')
news.index = pd.to_datetime(news.index)

temp = news['2008-12-01':'2009-1-30']

