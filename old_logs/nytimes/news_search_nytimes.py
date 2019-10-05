import argparse
import json
import requests
from pandas import date_range


payload = {'api-key': '...'}

argparse.ArgumentParser(description='download news arg parser')
parser = argparse.ArgumentParser(description='buying strategy parsing')
parser.add_argument('--from_day', type=str, help='start buying date')
parser.add_argument('--to_day', type=str, help='to buying date')
args = parser.parse_args()

for day in date_range(args.from_day, args.to_day, freq='M'):
    r = requests.get('https://api.nytimes.com/svc/archive/v1/%s/%s.json' % (day.year, day.month), params=payload)
    with open('raw/news_%s.json' % day.date(), 'w', encoding='utf-8') as outfile:
        content = r.json()
        outfile.write(json.dumps(content['response']['docs']))
        print(day.date())
