import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from numpy import nan, array
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

from util import io, visualization


WINDOW_SIZE = 20  # one side 20, 2 side 10
STABLE_THRESES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
STABLE_THRES_MEANSS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
MOVEMENT_THRESES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
COLOR_DICT = {'up': 'orange', 'still': 'cyan', 'down': 'blue', 'stable': 'green', 'chaos': 'red'}


def arima_decompose(data):
    result = seasonal_decompose(data, freq=250)
    fig = result.plot()
    resid = result.resid.dropna()
    seasonal = result.seasonal.dropna()
    seasonal = seasonal[seasonal.index >= resid.head(1).index.values[0]]
    seasonal = seasonal[seasonal.index <= resid.tail(1).index.values[0]]
    chaos = abs(resid) > 0.75
    plt.show()


def window_decompose(data):
    rolls = data.rolling(WINDOW_SIZE, center=True)
    roll_mean = rolls.mean()
    data['chaos'] = nan
    data['movement'] = nan
    data['chaos_means'] = (data[0].values - roll_mean[0].values) ** 2

    model_ols = linear_model.LinearRegression()

    for i in range(len(data) - WINDOW_SIZE + 1):
        data_subset = data.iloc[i: i + WINDOW_SIZE].reset_index()
        model_ols.fit(data_subset.index.values.reshape(-1, 1), data_subset[0].values.reshape(-1, 1))
        predict = model_ols.predict(array(range(WINDOW_SIZE)).reshape(-1, 1))
        data['chaos'].iloc[int((2 * i + WINDOW_SIZE) / 2)] = mean_squared_error(data_subset[0].values, predict)
        data['movement'].iloc[int((2 * i + WINDOW_SIZE) / 2)] = model_ols.coef_[0][0]

    return data


def _convert_to_classes(x, thres=0.05):
    if abs(x) > thres:
        if x > 0:
            return 2    # increase
        else:
            return 0
    else:
        return 1  # stable


def plot(data, movement_thres=0.05, stable_thres=0.06, stable_thres_means=0.05):
    fig, axes = plt.subplots(2, 1, sharex=True)
    data[0].plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Chaos (MSE)')
    data[0].plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Movement')
    # data[0].plot(ax=axes[2], legend=False)

    # create a map function here for data[:, 2]
    data['movement'].apply(lambda x: _convert_to_classes(x, movement_thres))
    data['chaos'] = data['chaos'][data['chaos'].notnull()].apply(lambda x: x > stable_thres)
    data['chaos_means'] = data['chaos_means'][data['chaos_means'].notnull()].apply(lambda xx: xx > stable_thres_means)

    visualization.plot_colormap(fig.axes[0], data['chaos'], True)
    # visualization.plot_colormap(fig.axes[2], data['chaos_means'], True)
    visualization.plot_colormap(fig.axes[1], data['movement'], False)
    patches = []
    for k, v in COLOR_DICT.items():
        patches.append(matplotlib.patches.Patch(color=v, label=k))

    box = axes[-1].get_position()
    axes[-1].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # Put a legend below current axis
    plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)

    # with open('baseline.txt', 'w') as f:
    #    f.write(str(data['chaos'].value_counts()))
    #    f.write(str(data['movement'].value_counts()))

    plt.savefig('%f_%f.png' % (movement_thres, stable_thres), dpi=800)


#_gpl, _ncg = io.read_future_market()
fig, ax1 = plt.subplots(1, 1)
future = io.read_future_market_v2('gpl')
#gpl_analyses = _gpl[0].to_frame()
#arima_decompose(gpl_analyses)
x = future['price']
x.plot(ax=ax1)
spot = io.read_spot_market()

spot.plot(ax=ax1)
#spot.index = spot['Tradingday']
#spot['GPL'].plot()
#spot['NCG'].plot()
#_gpl[0].plot()
#_ncg[0].plot()
plt.legend()
plt.tight_layout()
plt.show()
# data = window_decompose(gpl_analyses)
#
# # save learning data
# with open('config.yaml') as stream:
#     try:
#         config = yaml.load(stream)
#         io.save_df(data, config['train_file'], key=str(WINDOW_SIZE))
#     except yaml.YAMLError as exc:
#         print(exc)
