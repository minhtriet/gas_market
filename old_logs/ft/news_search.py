import requests
import json
from pandas import date_range
import argparse

headers = {"X-Api-Key": '...', 'Content-Type': 'application/json'}

argparse.ArgumentParser(description='download news arg parser')
parser = argparse.ArgumentParser(description='buying strategy parsing')
parser.add_argument('--from_day', type=str, help='start buying date', required=True)
parser.add_argument('--to_day', type=str, help='to buying date', required=True)
args = parser.parse_args()

base_url = "http://api.ft.com/content/search/v1"
for day in date_range(args.from_day, args.to_day, freq='M'):
    begin_time = (day.replace(day=1)).isoformat()  # first day this month, > in FT means >=
    end_time = day.isoformat()
    payload = {
        "queryString": "\"natural gas\" AND lastPublishDateTime:>%sZ AND lastPublishDateTime:<%sZ" % (
        begin_time, end_time),
        "queryContext": {"curations": ["ARTICLES"]},
        "resultContext": {"aspects":
                              ["audioVisual", "editorial", "images", "lifecycle", "location", "master", "metadata",
                               "nature", "provenance", "summary", "title"],
                          "sortField": "lastPublishDateTime", "sortOrder": "ASC",
                          "facets": {"names": ["authors", "brand", "category", "format", "genre",
                                               "icb", "iptc", "organisations", "people", "primarySection",
                                               "primaryTheme", "regions", "sections", "specialReports", "subjects",
                                               "topics"]}
                          }}
    response = requests.post(base_url, data=json.dumps(payload), headers=headers)
    with open('news_%s.json' % day.date(), 'w', encoding='utf-8') as outfile:
        json.dump(response.json(), outfile)
        print(day.date())
