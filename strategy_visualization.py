import argparse
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pandas import read_csv, to_datetime
from util import io
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description='buying strategy parsing')
parser.add_argument('--from_day', type=str, help='start buying date', required=True)
parser.add_argument('--to_day', type=str, help='to buying date', required=True)
args = parser.parse_args()
strategies = ['old', 'new_tf', 'baseline']
fancy_name = ['Current Buying Strategy', 'AI aided Strategy', 'Buy same volume every day']
fancy_name_dict = dict(zip(strategies, fancy_name))
fig, axes = plt.subplots(nrows=len(strategies), ncols=1, sharey=True)

market = io.read_future_market_v2('gpl')
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
    axes[d].set_ylabel(fancy_name_dict[strategies[d]])
    buy_plot.plot(market['price'])

legend_elements = [Line2D([0], [0], color='b', label='Future price'),
                   Line2D([0], [0], color='orange', label='Volume')]

fig.legend(handles=legend_elements)
plt.show()
