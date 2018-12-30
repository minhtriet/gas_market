# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 18:05:28 2018

@author: GIQNO
"""

import os

import numpy as np
import pandas as pd
import yaml
from util.news import filterer

def _reduce_1y(df, market):
    """
    :param df: Dataframe
    :param market: Which column to look at
    :return: File has future prices of multi years. Now they should return 1 year
    """
    years_col = df.columns[df.columns.str.startswith(market)]
    future_1y = df[years_col].apply(lambda row: row.first_valid_index(), axis=1)
    df['price'] = df.lookup(future_1y.index, future_1y.values)
    return df


def _join_path(*filename):
    """
    :param filename:
    :return: path of the file
    """
    return os.path.join(base_path, *filename)


def _immediate_subdir(a_dir):
    for name in os.listdir(a_dir):
        if os.path.isdir(os.path.join(a_dir, name)) and not name.startswith('_'):   # switch to disable loading a folder
            yield name


def read_train(key):
    return pd.read_hdf(_join_path(config['train_file']), key=str(key))


def save_df(df, filename, key):
    df.to_hdf(_join_path(filename), key=key)


def read_spot_market(market):
    try:
        spot = pd.read_hdf(_join_path('Mappe1.h5'), key='spot')
        spot = spot.rename(columns={market: 'price'})
        spot = spot.set_index(pd.DatetimeIndex(spot['Tradingday']))
        spot.drop('Tradingday', axis=1, inplace=True)
    except (FileNotFoundError, KeyError):
        print('H5 not found, creating from csv')
        xls = pd.read_excel(_join_path('Mappe1.xlsx'), sheet_name=None)
        spot = xls['G_EEX_TRP'].iloc[2:]
        spot.columns = xls['G_EEX_TRP'].iloc[1]
        spot = spot.rename(columns={'Tradingday\nHandelstag': 'Tradingday'})
        spot = spot[['Tradingday', 'NCG', 'GPL', 'TTF']].dropna()
        spot.Tradingday = pd.to_datetime(spot.Tradingday, dayfirst=True)
        spot.to_hdf(_join_path('Mappe1.h5'), key='spot')
    return spot


def read_spot_market_v2(market):
    df = pd.read_csv(_join_path('TRP-EGSI.csv'), delimiter=';')
    df = df.rename(columns={'Handelstag': 'Tradingday'})
    df['Tradingday'] = pd.to_datetime(df['Tradingday'], format='%m/%d/%y')
    df = df.set_index(pd.DatetimeIndex(df['Tradingday']))
    df = df.rename(columns={market: 'price'})
    return df


def read_future_market_v2(market):
    df = pd.read_csv(_join_path('Mappe1_2007.csv'), delimiter=';')
    df = df.rename(columns={'datum': 'Tradingday'})
    df['Tradingday'] = pd.to_datetime(df['Tradingday'], format='%d.%m.%Y')
    df = df.set_index(pd.DatetimeIndex(df['Tradingday']))
    return pd.DataFrame(_reduce_1y(df, market)['price'])


def read_future_market():
    try:
        gpl = pd.read_hdf(_join_path('Mappe1.h5'), key='gpl')
        ncg = pd.read_hdf(_join_path('Mappe1.h5'), key='ncg')
    except (FileNotFoundError, KeyError):
        xls = pd.read_excel(_join_path('Mappe1.xlsx'), sheet_name=None)
        xls['G_EEX_GPL'].columns = xls['G_EEX_GPL'].iloc[1]
        xls['G_EEX_GPL'] = xls['G_EEX_GPL'].iloc[2:]
        gpl = xls['G_EEX_GPL'].rename(columns={'Tradingday\nHandelstag': 'Tradingday'})
        gpl.Tradingday = gpl.Tradingday.map(lambda x: x.date())
        xls['G_EEX_NCG'].columns = xls['G_EEX_NCG'].iloc[1]
        xls['G_EEX_NCG'] = xls['G_EEX_NCG'].iloc[2:]
        ncg = xls['G_EEX_NCG'].rename(columns={'Tradingday\nHandelstag': 'Tradingday'})
        ncg.Tradingday = ncg.Tradingday.map(lambda x: x.date())
        gpl = gpl.replace({'n.a.': np.nan})
        ncg = ncg.replace({'n.a.': np.nan})
        gpl = gpl.reset_index(drop=True)
        ncg = ncg.reset_index(drop=True)
        gpl = _reduce_1y(gpl, 'gpl')
        ncg = _reduce_1y(ncg, 'ncg')
        # convert to float
        gpl.to_hdf(_join_path('Mappe1.h5'), key='gpl')
        ncg.to_hdf(_join_path('Mappe1.h5'), key='ncg')
    return gpl, ncg


def read_demand():
    if not os.path.exists(_join_path('demand.h5')):
        f = demands_files[0]
        xls = pd.read_excel(f, sheet_name=None)
        for sheet in xls:
            xls[sheet] = xls[sheet][['Unnamed: 16', 'Unnamed: 17']]
            xls[sheet].columns = xls[sheet].iloc[1]
            xls[sheet] = xls[sheet].iloc[2:]
        demand = pd.concat(xls.values(), sort=False, ignore_index=True)
        demand = demand.groupby(demand.Zeitstempel.dt.date).sum().reset_index()
        demand.to_hdf(_join_path('demand.h5'), key='demand')
    else:
        demand = pd.read_hdf(_join_path('demand.h5'), key='demand')
    return demand


def load_stock(from_date, to_date):
    """
    Closing price of gas companies stock
    :return: dataframe of closing price of stocks
    """
    df = pd.DataFrame()
    for f in os.listdir(_join_path('stock')):
        s = pd.read_csv(_join_path('stock', f))
        s = s.iloc[1:]
        s.set_index(s['date'])
        s = s.between_time(from_date, to_date)
        stock_name = os.path.splitext(f)[0]
        s = s.rename(index=str, columns={'close': stock_name})

        # df = pd.concat([df, s[stock_name]], axis=
        df[stock_name] = s[stock_name]

        if len(df.index) == 0:
            df.set_index(s.index)
    return df


def load_news(embed, filename='data.csv'):
    big_info_df = pd.DataFrame()
    if not hasattr(filterer, embed):
        raise ValueError('Unknown embedding')
    for subdir in _immediate_subdir('old_logs'):
        df = pd.read_csv(os.path.join('old_logs', subdir, filename))
        df = df.dropna(subset=['info'])
        df = df.drop_duplicates(subset=['info'])
        df['pub_date'] = df['pub_date'].map(lambda x: pd.to_datetime(x).date())
        df = df.set_index('pub_date')
        info_series = df.groupby(df.index)['info'].apply(lambda x: '. '.join(x))
        if embed != 'none':
            info_series = info_series.map(getattr(filterer, embed))  # user different embedding according to arg embed
            num_column = len(info_series.head(1).values[0])
        else:
            num_column = 1
        data = info_series.values.tolist()
        info_df = pd.DataFrame(data=data, index=info_series.index, columns=range(num_column))
        info_df.index = pd.to_datetime(info_df.index)
        big_info_df = big_info_df.append(info_df)
    return big_info_df


base_path = ''
with open('config.yaml') as stream:
    try:
        config = yaml.load(stream)
        base_path = config['base_path']
    except yaml.YAMLError as exc:
        print(exc)

demands_files = ['Weimar_Lastgang 2015_Erdgasbezug.xlsx',
                 'Weimar_Lastgang 2016_Erdgasbezug.xlsx',
                 'Kiel_Lastgang_KVP_KJ2015_07062016.xlsx']
