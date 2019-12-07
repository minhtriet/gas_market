import argparse
import time
import json
import requests
from pandas import date_range

argparse.ArgumentParser(description='download news arg parser')
parser = argparse.ArgumentParser(description='buying strategy parsing')
parser.add_argument('--from_day', type=str, help='start buying date', required=True)
parser.add_argument('--to_day', type=str, help='to buying date', required=True)
args = parser.parse_args()

drange = date_range(args.from_day, args.to_day, freq='M')
for i in drange:
    print(i)
    r = ''
    for p in range(1,10):
        payload = {'api-key': '...-...-...-...-...', 'page-size': 200,
                   'from-date': i.replace(day=1).date(), 'type': 'article',
                   'to-date': i.date(), 'q': 'gas', 'page': p}
        r2 = requests.get('https://content.guardianapis.com/search', params=payload).json()
        if p == 1:
            r = r2
        else:
            if r2['response'].get('result'):
                r['response']['results'].extend(r2['response']['results'])
            else:
                break
    with open('g_news_%s.json' % i.date(), 'w', encoding='utf-8') as outfile:
        print('g_news_%s.json' % i.date())
        outfile.write(json.dumps(r))
    time.sleep(0.1)
