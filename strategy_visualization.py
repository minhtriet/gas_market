import argparse

import matplotlib
from numpy import nonzero

font = {'size': 6}

matplotlib.rc('font', **font)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pandas import read_csv, to_datetime
from util import io
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description='buying strategy parsing')
parser.add_argument('--from_day', type=str, help='start buying date', required=True)
parser.add_argument('--to_day', type=str, help='to buying date', required=True)
args = parser.parse_args()
# strategies = ['old', 'baseline','new_tf',  '3d']
# fancy_name = ['Current Buying Strategy', 'Buy same volume every day', 'LSTM with word embedding', '3D Convolution prediction']
strategies = ['baseline', 'new_tf', '3d']
fancy_name = ['Baseline', 'LSTM with word embedding', '3D Convolution with Event embedding']
fancy_name_dict = dict(zip(strategies, fancy_name))
fig, axes = plt.subplots(nrows=len(strategies), ncols=1, sharey=True, figsize=(6, 9))
# market = io.read_future_market_v2('gpl')
market = io.read_spot_market_v2('gpl')
market = market.loc[args.from_day:args.to_day]

data = []

for i in range(len(strategies)):
    df = read_csv('%s.csv' % strategies[i], index_col=0)
    df = df.loc[args.from_day:args.to_day]
    assert len(market) == len(df)
    if len(df) == 0:
        raise ValueError('This time frame has not been experienced')
    df.index = to_datetime(df.index)
    data.append(df)

for d in range(len(data)):
    axes[d].bar(data[d].index, data[d]['volume'], color='orange', alpha=0.7)
    buy_plot = axes[d].twinx()
    axes[d].set_xlabel(fancy_name_dict[strategies[d]])
    axes[d].set_ylabel('Buying amount')
    if d > 0:
        buy_days = data[d]['volume'] > 0
        buy_plot.plot(market['price'], '-|', markevery=list(nonzero(buy_days.values)), markeredgecolor='r')
    else:
        buy_plot.plot(market['price'])
    buy_plot.set_ylabel('Price (Euro/Cube meter)')

legend_elements = [Line2D([0], [0], color='b', label='Future price'),
                   Line2D([0], [0], color='orange', label='Volume')]

fig.legend(handles=legend_elements)
# plt.show()
plt.tight_layout()
fig.savefig("foo.pdf")
