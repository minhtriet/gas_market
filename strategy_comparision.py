import argparse

import matplotlib.pyplot as plt
from pandas import read_csv, to_datetime

from util import io


def average_cost(log):
    log['avr'] = 0
    total = 0
    volume = 0
    for l in log.itertuples(index=True):
        total += getattr(l, 'cost')
        volume += getattr(l, 'volume')
        if volume == 0:
            log.loc[getattr(l, 'Index'), 'avr'] = 0
        else:
            log.loc[getattr(l, 'Index'), 'avr'] = total / volume
    return log


parser = argparse.ArgumentParser(description='buying strategy parsing')
parser.add_argument('--from_day', type=str, help='start buying date', required=True)
parser.add_argument('--to_day', type=str, help='to buying date', required=True)

args = parser.parse_args()
fig, axes = plt.subplots()

strategy_name = ['old', 'new', 'baseline']
fancy_name = ['Current Buying Strategy', 'AI aided Strategy', 'Buy same volume every day']
fancy_name_dict = dict(zip(strategy_name, fancy_name))

gpl, ncg = io.read_future_market()
gpl = gpl.loc[args.from_day:args.to_day]
for i, strat in enumerate(strategy_name):
    df = read_csv('%s.csv' % strat, index_col=0)
    df.index = to_datetime(df.index)
    df = df.loc[args.from_day:args.to_day]
    df = average_cost(df)
    axes.plot(df['avr'], label=fancy_name_dict[strat])

axes.plot(gpl[0], label='Future 1y price')
axes.set_ylim([15.5, 19])
plt.legend()
plt.ylabel('Average price')
plt.show()
