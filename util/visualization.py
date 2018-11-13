COLOR_DICT = {'up': 'orange', 'still': 'cyan', 'down': 'blue', 'stable': 'green', 'chaos': 'red'}
CHAOS_MAP = {True: 'chaos', False: 'stable'}
MOVEMENT_MAP = {0: 'down', 1: 'still', 2: 'up'}


def plot_colormap(ax, status, is_chaos):
    status = status.dropna()
    changed_index = [status.index[i + 1] for i in range(len(status.index) - 1) if status.values[i] != status.values[i + 1]]
    changed_index.insert(0, status.index[0])
    changed_index.append(status.index[-1])

    changed_value = status[changed_index].values
    if is_chaos:
        values = list(map(CHAOS_MAP.get, changed_value))
    else:
        values = list(map(MOVEMENT_MAP.get, changed_value))
    color_list = list(map(COLOR_DICT.get, values))

    for i in range(len(changed_index)-1):
        ax.axvspan(changed_index[i], changed_index[i+1], facecolor=color_list[i], alpha=0.5)


def plot_buy_graph(ax, df):
    df['cost'].plot(ax)
    ax.show()
