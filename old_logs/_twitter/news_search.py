import argparse
import json
from datetime import timedelta

import pandas as pd
from searchtweets import gen_rule_payload, load_credentials, collect_results

from util.news import filter

import os

argparse.ArgumentParser(description='download news arg parser')
parser = argparse.ArgumentParser(description='buying strategy parsing')
parser.add_argument('--from_day', type=str, help='start buying date')
parser.add_argument('--to_day', type=str, help='to buying date')
args = parser.parse_args()


premium_search_args = load_credentials("twitter_keys.yaml", env_overwrite=False)

for day in pd.date_range(args.from_day, args.to_day)[::2]:
    rule = gen_rule_payload(pt_rule='"natural gas" lang:en', results_per_call=100,
                            from_date=str(day.date()), to_date=str((day + timedelta(days=1)).date()), stringify=True)
    tweets = collect_results(rule, max_results=100, result_stream_args=premium_search_args)

    with open('news_%s.json' % day.date(), 'w') as outfile:
        json.dump(tweets, outfile)
    tweets = filter.filter_tweet(tweets)
    with open('processed_%s.json' % day.date(), 'w') as outfile:
        json.dump(tweets, outfile)
