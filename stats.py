# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:57:17 2019

@author: rdy11
"""

import pandas as pd
import numpy as np
import pickle

projectpath = '/Users/Ryan/OneDrive/thesis code/trimmed'
csvpath = '/Users/Ryan/OneDrive/thesis code/trimmed'
filepath = '/Users/Ryan/OneDrive/thesis code/trimmed/returns_out_trimmed.csv'
returns = pd.DataFrame.from_csv(filepath)
container = pd.DataFrame()


def import_signals():
    indicators = ['MA_a', 'MA_b', 'MA_c',
                  'TRB_a_min', 'TRB_b_min', 'TRB_c_min',
                  'TRB_a_max', 'TRB_b_max', 'TRB_c_max',
                  'ROUND_up', 'ROUND_down']

    csvs = [pd.DataFrame.from_csv(f'{csvpath}/MA_a_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/MA_b_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/MA_c_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/TRB_a_min_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/TRB_b_min_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/TRB_c_min_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/TRB_a_max_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/TRB_b_max_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/TRB_c_max_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/ROUND_up_out.csv'),
            pd.DataFrame.from_csv(f'{csvpath}/ROUND_down_out.csv')]

    signals_dict = dict(zip(indicators, csvs))

    return signals_dict


def generate_container():
    global container

    securities = returns.columns

    indicators = ['MA_a', 'MA_b', 'MA_c',
                  'TRB_a_min', 'TRB_b_min', 'TRB_c_min',
                  'TRB_a_max', 'TRB_b_max', 'TRB_c_max',
                  'ROUND_up', 'ROUND_down']

    samples = ['all', 'signal']

    container = pd.DataFrame(
        {
            indicator: {
                security: {
                    sample: None
                    for sample in samples}
                for security in securities}
            for indicator in indicators}
    )


def fill_container(signals_dict):
    global container

    securities = returns.columns

    indicators = ['MA_a', 'MA_b', 'MA_c',
                  'TRB_a_min', 'TRB_b_min', 'TRB_c_min',
                  'TRB_a_max', 'TRB_b_max', 'TRB_c_max',
                  'ROUND_up', 'ROUND_down']

    for indicator in indicators:
        for security in securities:

            container[indicator][security]['all'] = pd.to_numeric(
                returns[security], errors='coerce').dropna()

            container[indicator][security]['signal'] = pd.Series(
                np.ma.masked_where(
                    signals_dict[indicator][security] == 0,
                    returns[security])
            ).dropna()


def main():
    global container

    generate_container()
    fill_container(import_signals())

    container.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
    container.to_pickle('container_trimmed')