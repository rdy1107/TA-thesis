import pandas as pd
import numpy as np
from scipy import stats
import pickle
import statsmodels.stats.multitest as multitest
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import brewer2mpl

#projectpath = '/Users/Ryan/OneDrive/thesis code/trimmed'
#containerpath = '/Users/Ryan/OneDrive/thesis code/trimmed/container_trimmed'
projectpath = '/Users/Ryan/OneDrive/thesis code/'
containerpath = '/Users/Ryan/OneDrive/thesis code/container'
container = pd.read_pickle(containerpath)
returns_out = pd.read_csv('returns_out.csv')
returns_out = returns_out.set_index('DATE')
#returns_out = returns_out.loc[19570301:]
returns_out = pd.Series(returns_out.values.flatten()).dropna()
colors = brewer2mpl.get_map('Paired', 'Qualitative', 12).mpl_colors

def table_1():
    global returns_out
    global container

    returns_out = pd.Series(returns_out.values.flatten()).dropna()
    returns_out = pd.to_numeric(returns_out, errors='coerce').dropna()
    indicators = ['MA_a', 'MA_b', 'MA_c',
                  'TRB_a_min', 'TRB_b_min', 'TRB_c_min',
                  'TRB_a_max', 'TRB_b_max', 'TRB_c_max',
                  'ROUND_up', 'ROUND_down']
    rows = ['N', 'Median', 'Mean', 'Var', 'Stdev', 'Skewness', 'Kurtosis']
    f = {'Median': np.median, 'Mean': np.mean, 'Var': np.var,
         'Stdev': np.std, 'Skewness': stats.skew, 'Kurtosis': stats.kurtosis}
    securities = container.index
    table = pd.DataFrame(index=rows, columns=indicators)

    for indicator in indicators:
        signal = pd.Series()
        print(f'Starting {indicator}')

        for security in securities:
            a = container[indicator][security]['signal']
            signal = signal.append(a, ignore_index=True)

        for row in rows:
            signal = pd.to_numeric(signal, errors='coerce').dropna()

            if row == 'N':
                table[indicator][row] = pd.Series(signal).count()

            else:
                table[indicator][row] = f[row](signal)

#    table.to_pickle('table_1')
    table.to_pickle('table_1_b')

    return table

def table_2(a):
    p_values_adj_container = pd.read_pickle('p_values_adj_container_full')
    tests = ['Brown-Forsythe', 'Bartlett', 'Levene', 'Fligner-Killeen']
    indicators = ['MA_a', 'MA_b', 'MA_c',
                  'TRB_a_min', 'TRB_b_min', 'TRB_c_min',
                  'TRB_a_max', 'TRB_b_max', 'TRB_c_max',
                  'ROUND_up', 'ROUND_down']

    brown_forsythe = pd.DataFrame.from_dict(p_values_adj_container[tests[0]], orient='index')
    bartlett = pd.DataFrame.from_dict(p_values_adj_container[tests[1]], orient='index')
    levene = pd.DataFrame.from_dict(p_values_adj_container[tests[2]], orient='index')
    fligner_killeen = pd.DataFrame.from_dict(p_values_adj_container[tests[3]], orient='index')

    table = pd.DataFrame(index=indicators, columns=tests)

    for indicator in indicators:
        table['Brown-Forsythe'][indicator] = np.round((np.sum(
            brown_forsythe.loc[indicator] < a) / brown_forsythe.loc[indicator].count() * 100), 1)
        table['Bartlett'][indicator] = np.round((np.sum(
            bartlett.loc[indicator] < a) / bartlett.loc[indicator].count() * 100), 1)
        table['Levene'][indicator] = np.round((np.sum(
            levene.loc[indicator] < a) / levene.loc[indicator].count() * 100), 1)
        table['Fligner-Killeen'][indicator] = np.round((np.sum(
            fligner_killeen.loc[indicator] < a) / fligner_killeen.loc[indicator].count() * 100), 1)

    table.T.to_pickle('table_2_full')

    return table.T

def table_2_b(a):
    p = pd.read_pickle('p_values')
    tests = ['Brown-Forsythe', 'Bartlett', 'Levene', 'Fligner-Killeen']
    indicators = ['MA_a', 'MA_b', 'MA_c',
                  'TRB_a_min', 'TRB_b_min', 'TRB_c_min',
                  'TRB_a_max', 'TRB_b_max', 'TRB_c_max',
                  'ROUND_up', 'ROUND_down']

    table = pd.DataFrame(index=indicators, columns=tests)

    for test in tests:
        for indicator in indicators:
            table[test][indicator] = np.round(
                ((np.sum(p[test][indicator] < a) / p[test][indicator].count()) * 100),
                1)

    table.T.to_pickle('table_2')

    return table.T

def table_3(a):
    p = pd.read_pickle('p_values_adj')
    tests = ['Brown-Forsythe', 'Bartlett', 'Levene', 'Fligner-Killeen']
    indicators = ['MA_a', 'MA_b', 'MA_c',
                  'TRB_a_min', 'TRB_b_min', 'TRB_c_min',
                  'TRB_a_max', 'TRB_b_max', 'TRB_c_max',
                  'ROUND_up', 'ROUND_down']

    table = pd.DataFrame(index=indicators, columns=tests)

    for test in tests:
        for indicator in indicators:
            table[test][indicator] = np.round(
                ((np.sum(p[test][indicator] < a) / p[test][indicator].count()) * 100),
                1)

    table.T.to_pickle('table_3')

    return table.T

def table_4():
    global container

    indicators = ['MA_a', 'MA_b', 'MA_c',
                  'TRB_a_min', 'TRB_b_min', 'TRB_c_min',
                  'TRB_a_max', 'TRB_b_max', 'TRB_c_max',
                  'ROUND_up', 'ROUND_down']
    securities = container.index

    table = pd.DataFrame(columns=indicators)
    all = pd.Series()

    for security in securities:
        a = container[indicators[0]][security]['all']
        all = all.append(a, ignore_index=True)

    table['All'] = all

    for indicator in indicators:
        signal = pd.Series()
        print(f'starting {indicator}')

        for security in securities:
            a = container[indicator][security]['signal']
            signal = signal.append(a, ignore_index=True)

        table[indicator] = signal

    table.to_pickle('table_4')

    return table

def figure_1(save_fig):
    table = pd.read_pickle('table_4')
    map = {0: 'MA_a', 1: 'MA_b', 2: 'MA_c', 3: 'TRB_a_min', 4: 'TRB_b_min', 5: 'TRB_c_min',
           6: 'TRB_a_max', 7: 'TRB_b_max', 8: 'TRB_c_max', 9: 'ROUND_up', 10: 'ROUND_down',
           11: 'All'}
    labels = {0: '50-day MA', 1: '100-day MA', 2: '200-day MA', 3: '50-day min', 4: '100-day min', 5: '200-day min',
           6: '50-day max', 7: '100-day max', 8: '200-day max', 9: 'Round up', 10: 'Round down',
           11: 'Control'}

    fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True)

    for i in map:
        data = pd.to_numeric(table[map[i]], errors='coerce').dropna()
        fig.axes[i].hist(data, bins=25, range=(-.1, .1), density=True, color=colors[1], histtype='stepfilled')
        fig.axes[i].set_title(labels[i], fontdict={
        'fontname': 'Source Sans Pro',
        'weight': 'normal',
        'size': 14})

    fig.set_size_inches(11, 8)

    plt.tight_layout()

    if save_fig:
        plt.savefig('/Users/Ryan/OneDrive/thesis_new/graphics/distributions.pdf', format='pdf')

def diag():
    returns = pd.DataFrame.from_csv('/Users/Ryan/OneDrive/thesis code/trimmed/returns_out_trimmed.csv')
    count = returns.count()
    nonzeros = pd.Series(index=count.index)

    for column in returns:
        data = returns[column].dropna()
        value = np.count_nonzero(data)
        nonzeros[column] = value

    zeros = (count - nonzeros) / count