import pandas as pd
import numpy as np
from scipy import stats
import pickle
import statsmodels.stats.multitest as multitest
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker as mtick
import seaborn as sns
import brewer2mpl

projectpath = '/Users/Ryan/OneDrive/thesis code/'
containerpath = '/Users/Ryan/OneDrive/thesis code/container'
container = pd.read_pickle(containerpath)
errors = []
colors = brewer2mpl.get_map('Paired', 'Qualitative', 12).mpl_colors

indicators = {'MA_a': 'MA (50d)', 'MA_b': 'MA (100d)', 'MA_c': 'MA (200d)',
              'TRB_a_min': 'Rolling min (50d)', 'TRB_b_min': 'Rolling min (100d)',
              'TRB_c_min': 'Rolling min (200d)', 'TRB_a_max': 'Rolling max (50d)',
              'TRB_b_max': 'Rolling max (100d)', 'TRB_c_max': 'Rolling max (200d)',
              'ROUND_up': 'Round up', 'ROUND_down': 'Round down'}

def stats_tests():
    global errors
    tests = ['Brown-Forsythe', 'Bartlett', 'Levene', 'Fligner-Killeen']
    securities = list(container.index)
    indicators = list(container.columns)

    output = pd.DataFrame(index=pd.MultiIndex.from_product([securities, indicators]),
                          columns=tests)

    for security in securities:
        for indicator in indicators:
            all = pd.Series(container.loc[security][indicator]['all'])
            signal = pd.Series(container.loc[security][indicator]['signal'])
            all = pd.to_numeric(all, errors='coerce')
            signal = pd.to_numeric(signal, errors='coerce')

            try:
                output.loc[security, indicator][tests[0]] = stats.levene(
                    all, signal,
                    center='median'
                )
            except:
                errors.append([security, indicator, tests[0]])

            try:
                output.loc[security, indicator][tests[1]] = stats.bartlett(
                    all, signal
                )
            except:

                errors.append([security, indicator, tests[1]])

            try:
                output.loc[security, indicator][tests[2]] = stats.levene(
                    all, signal,
                    center='mean'
                )
            except:
                errors.append([security, indicator, tests[2]])

            try:
                output.loc[security, indicator][tests[3]] = stats.fligner(
                    all, signal
                )
            except:
                errors.append([security, indicator, tests[3]])

    p_values = output.dropna().applymap(lambda x: x.pvalue).unstack()
    p_values_container = output.dropna().applymap(lambda x: x.pvalue).unstack().melt()
    p_values.to_pickle('p_values_full')
    p_values_container.to_pickle('p_values_container_full')

def adjust_p_manual(lamb, alpha):
    tests = ['Brown-Forsythe', 'Bartlett', 'Levene', 'Fligner-Killeen']
    securities = list(container.index)
    indicators = list(container.columns)
    p_values = pd.read_pickle('p_values')
    p_values_container = pd.read_pickle('p_values_container')

    p_values_stats = pd.DataFrame(columns=p_values.columns, index=['n', 'm_0', 'q_tested',
                                                                   'q_tested_null', 'q_adj_p_null'])

    for test in tests:
        for indicator in indicators:
            n = len(p_values[test][indicator])
            s_all = pd.Series(p_values[test][indicator].dropna().values).sort_values()
            s_null = s_all[s_all<alpha]
            s_lamb = s_all[s_all>lamb]
            m = len(s_all)
            r = len(s_null)

            # number of true null tests
            m_0 = s_lamb / (m * (1 - lamb))

            q_tested = m / n
            q_tested_null = r / m

            #FDR control
            p_mult = pd.Series(((s_all.index + 1) / m) ** -1)
            adj_p = p_mult * s_all

            adj_p_null = adj_p[adj_p<alpha]
            q_adj_p_null = len(adj_p_null) / m

            p_values_stats[test][indicator]['n'] = n
            p_values_stats[test][indicator]['m_0'] = m_0
            p_values_stats[test][indicator]['q_tested'] = q_tested
            p_values_stats[test][indicator]['q_tested_null'] = q_tested_null
            p_values_stats[test][indicator]['q_adj_p_null'] = q_adj_p_null

    p_values_stats.to_pickle('p_values_stats')

def norm_test():
    global errors
    tests = ['Brown-Forsythe', 'Bartlett', 'Levene', 'Fligner-Killeen']
    securities = list(container.index)
    indicators = list(container.columns)

    output = pd.DataFrame(columns=pd.MultiIndex.from_product([securities, indicators]))

    for security in securities:
        for indicator in indicators:
            all = pd.Series(container.loc[security][indicator]['all'])
            signal = pd.Series(container.loc[security][indicator]['signal'])
            all = pd.to_numeric(all, errors='coerce')
            signal = pd.to_numeric(signal, errors='coerce')

            try:
                output[security][indicator] = stats.shapiro(all)
            except:
                errors.append([security, indicator])

            try:
                output[security][indicator] = stats.shapiro(signal)
            except:
                errors.append([security, indicator])

def adjust_p():
    tests = ['Brown-Forsythe', 'Bartlett', 'Levene', 'Fligner-Killeen']
    securities = list(container.index)
    indicators = list(container.columns)
    p_values = pd.read_pickle('p_values_full')

    p_values_adj = pd.DataFrame(columns=pd.MultiIndex.from_product([tests, indicators]), index=p_values.index)
    f1 = (lambda x: multitest.multipletests(x.values, method='fdr_bh'))

    p_values_adj_container = {test:
        {indicator: [] for indicator in indicators}
        for test in tests}

    for test in tests:
        for indicator in indicators:
           p_values_adj_container[test][indicator] = f1(p_values[test, indicator].dropna())[1]
#           p_values_adj[test, indicator] = f1(p_values[test, indicator])[1]

    with open('p_values_adj_container_full', 'wb') as handle:
        pickle.dump(p_values_adj_container, handle)

    p_values_adj.to_pickle('p_values_adj_full')

def adjusted_p_graph(test):
    sns.set_style('ticks', {'axes.labelcolor': '.0', 'font.family': [u'serif'], 'text.usetex': True})
    p_values_adj_container = pd.read_pickle('p_values_adj_container')
    data = pd.DataFrame.from_dict(p_values_adj_container[test], orient='index')
    data = data.T
    data.index.name = 'Security'
    data.columns.name = 'Indicator'
    data = pd.melt(data)
    indicators = list(container.columns)

    indicators = {'MA_a': 'MA (50d)', 'MA_b': 'MA (100d)', 'MA_c': 'MA (200d)',
                  'TRB_a_min': 'Rolling min (50d)', 'TRB_b_min': 'Rolling min (100d)',
                  'TRB_c_min': 'Rolling min (200d)', 'TRB_a_max': 'Rolling max (50d)',
                  'TRB_b_max': 'Rolling max (100d)', 'TRB_c_max': 'Rolling max (200d)',
                  'ROUND_up': 'Round up', 'ROUND_down': 'Round down'}

    bmap = brewer2mpl.get_map('Greys', 'sequential', 5)
    colors = bmap.mpl_colors

    g = sns.FacetGrid(data, col='Indicator', col_wrap=3, sharey=False)
    g.map(sns.distplot, 'value', kde=False, bins=100)
    g.set(xlim=(-.025, 1.025), xticks=[.05])

    for i in range(len(g.axes)):
        g.axes[i].set_ylim(1, 1000)
        g.axes[i].set_yscale('log')
        g.axes[i].set_xlabel(None)
        g.axes[i].spines['right'].set_visible(True)
        g.axes[i].spines['top'].set_visible(True)
    #        g.axes[i].axvline(x=alpha, color='r', linestyle='dashed', linewidth=1)

    for i in range(len(g.axes)):
        g.axes[i].patches[0].set(color=colors[4])
        g.axes[i].patches[1].set(color=colors[4])
        g.axes[i].patches[2].set(color=colors[4])
        g.axes[i].patches[3].set(color=colors[4])
        g.axes[i].patches[4].set(color=colors[4])

        g.axes[i].patches[0].set_alpha(1)
        g.axes[i].patches[1].set_alpha(1)
        g.axes[i].patches[2].set_alpha(1)
        g.axes[i].patches[3].set_alpha(1)
        g.axes[i].patches[4].set_alpha(1)

        g.axes[i].set_title(indicators[g.col_names[i]])

        for n in range(len(g.axes[i].patches) - 5):
            g.axes[i].patches[n + 5].set(color=colors[2])
            g.axes[i].patches[n + 5].set(linewidth=None)
            g.axes[i].patches[n + 5].set_alpha(1)

    plt.tight_layout()

def adjusted_p_graph_20(test, alpha):
    sns.set_style('ticks', {'axes.labelcolor': '.0', 'font.family': [u'serif'], 'text.usetex': True})
    p_values = pd.read_pickle('p_values')
    p_values_stats = pd.read_pickle('p_values_stats')
    data = pd.DataFrame(p_values[test])
    data.index.name = 'Security'
    data.columns.name = 'Indicator'
    data = pd.melt(data)
    indicators = list(container.columns)

    bmap = brewer2mpl.get_map('Greys', 'sequential', 5)
    colors = bmap.mpl_colors

    g = sns.FacetGrid(data, col='Indicator', col_wrap=3, sharey=False)
    g.map(sns.distplot, 'value', kde=False, bins=20)
    g.set(xlim=(0,1), xticks=[.05])

    for i in range(len(g.axes)):
        g.axes[i].set_ylim(1, 1000)
        g.axes[i].set_yscale('log')
        g.axes[i].set_title(indicators[i])
        g.axes[i].set_xlabel(None)
        g.axes[i].spines['right'].set_visible(True)
        g.axes[i].spines['top'].set_visible(True)
#        g.axes[i].axvline(x=alpha, color='r', linestyle='dashed', linewidth=1)

    for i in range(len(g.axes)):
        g.axes[i].patches[0].set(color=colors[4])
        g.axes[i].patches[0].set_alpha(1)

        for n in range(len(g.axes[i].patches)-1):
            g.axes[i].patches[n+1].set(color=colors[2])
            g.axes[i].patches[n+1].set_alpha(1)

def indicators_demo(a, b, c, n_years, save_fig):
    sns.set_style('ticks', {'axes.labelcolor': '.0'})

    csvpath = '/Users/Ryan/OneDrive/thesis code/trimmed'

    indicators = {'MA_a': 'MA (50d)', 'MA_b': 'MA (100d)', 'MA_c': 'MA (200d)',
                  'TRB_a_min': 'Rolling min (50d)', 'TRB_b_min': 'Rolling min (100d)',
                  'TRB_c_min': 'Rolling min (200d)', 'TRB_a_max': 'Rolling max (50d)',
                  'TRB_b_max': 'Rolling max (100d)', 'TRB_c_max': 'Rolling max (200d)',
                  'ROUND_up': 'Round up', 'ROUND_down': 'Round down'}

    securities = list(container.index)
    security = securities[np.random.randint(0, len(securities) - 1)]

    p = pd.read_csv('data_out.csv', index_col=0)[security]
    p.index = pd.to_datetime(p.index, format="%Y%m%d")
    days = len(p.dropna())
    win = 252
    range = days - win
    start = np.random.randint(0, range)
    end = start + (win * n_years)

    colors = brewer2mpl.get_map('Paired', 'Qualitative', 12).mpl_colors

    d = int(a) - 1
    e = int(b) - 1
    f = int(c) - 1

    data = pd.DataFrame.from_dict({'Price': p.dropna()[start:end],
            'MA_a': pd.DataFrame.from_csv(f'{csvpath}/{a}_MA.csv')[security].dropna()[start-d:end-d],
            'MA_b': pd.DataFrame.from_csv(f'{csvpath}/{b}_MA.csv')[security].dropna()[start-e:end-e],
            'MA_c': pd.DataFrame.from_csv(f'{csvpath}/{c}_MA.csv')[security].dropna()[start-f:end-f],
            'TRB_a_min': pd.DataFrame.from_csv(f'{csvpath}/{a}_TRB_min.csv')[security].dropna()[start-d:end-d],
            'TRB_b_min': pd.DataFrame.from_csv(f'{csvpath}/{b}_TRB_min.csv')[security].dropna()[start-e:end-e],
            'TRB_c_min': pd.DataFrame.from_csv(f'{csvpath}/{c}_TRB_min.csv')[security].dropna()[start-f:end-f],
            'TRB_a_max': pd.DataFrame.from_csv(f'{csvpath}/{a}_TRB_max.csv')[security].dropna()[start-d:end-d],
            'TRB_b_max': pd.DataFrame.from_csv(f'{csvpath}/{b}_TRB_max.csv')[security].dropna()[start-e:end-e],
            'TRB_c_max': pd.DataFrame.from_csv(f'{csvpath}/{c}_TRB_max.csv')[security].dropna()[start-f:end-f],
            'ROUND_up': pd.DataFrame.from_csv(f'{csvpath}/ROUND_up.csv')[security].dropna()[start:end],
            'ROUND_down': pd.DataFrame.from_csv(f'{csvpath}/ROUND_down.csv')[security].dropna()[start:end]})

    fig, axes = plt.subplots(ncols=1, nrows=3, sharex=True)
    fig.set_size_inches(5, 7)

    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    years_fmt = mdates.DateFormatter('%Y')

    axes[0].plot(data.index, 'Price', data=data, linewidth=2, color=colors[1])
    axes[0].plot(data.index, 'MA_a', data=data, linestyle='--', color=colors[0])
    axes[0].plot(data.index, 'MA_b', data=data, linestyle='--', color=colors[0])
    axes[0].plot(data.index, 'MA_c', data=data, linestyle='--', color=colors[0])
    axes[0].set_title('Moving average', fontdict={
        'fontname': 'Source Sans Pro',
        'weight': 'normal',
        'size': 14})
    axes[0].xaxis.set_major_locator(years)
    axes[0].xaxis.set_major_formatter(years_fmt)
    axes[0].xaxis.set_minor_locator(months)

    axes[1].plot(data.index, 'Price', data=data, linewidth=2, color=colors[1])
    axes[1].plot(data.index, 'TRB_a_min', data=data, linestyle='--', color=colors[0])
    axes[1].plot(data.index, 'TRB_b_min', data=data, linestyle='--', color=colors[0])
    axes[1].plot(data.index, 'TRB_c_min', data=data, linestyle='--', color=colors[0])
    axes[1].plot(data.index, 'TRB_a_max', data=data, linestyle=':', color=colors[0])
    axes[1].plot(data.index, 'TRB_b_max', data=data, linestyle=':', color=colors[0])
    axes[1].plot(data.index, 'TRB_c_max', data=data, linestyle=':', color=colors[0])
    axes[1].set_title('Rolling min/max', fontdict={
        'fontname': 'Source Sans Pro',
        'weight': 'normal',
        'size': 14})
    axes[1].xaxis.set_major_locator(years)
    axes[1].xaxis.set_major_formatter(years_fmt)
    axes[1].xaxis.set_minor_locator(months)

    axes[2].plot(data.index, 'Price', data=data, linewidth=2, color=colors[1])
    axes[2].plot(data.index, 'ROUND_up', data=data, linestyle=':', color=colors[0])
    axes[2].plot(data.index, 'ROUND_down', data=data, linestyle='--', color=colors[0])
    axes[2].set_title('Round up/down', fontdict={
        'fontname': 'Source Sans Pro',
        'weight': 'normal',
        'size': 14})
    axes[2].xaxis.set_major_locator(years)
    axes[2].xaxis.set_major_formatter(years_fmt)
    axes[2].xaxis.set_minor_locator(months)

    for tick in axes[0].get_xticklabels():
        tick.set_fontname('Source Sans Pro')
    for tick in axes[0].get_yticklabels():
        tick.set_fontname('Source Sans Pro')
    for tick in axes[1].get_xticklabels():
        tick.set_fontname('Source Sans Pro')
    for tick in axes[1].get_yticklabels():
        tick.set_fontname('Source Sans Pro')
    for tick in axes[2].get_xticklabels():
        tick.set_fontname('Source Sans Pro')
    for tick in axes[2].get_yticklabels():
        tick.set_fontname('Source Sans Pro')

    plt.tight_layout()

    if save_fig:
        plt.savefig('/Users/Ryan/OneDrive/thesis_new/graphics/indicatorsdemo.pdf', format='pdf')

def proportion_graph(test):

    df = pd.read_csv('data.csv')
    df.drop(columns=['PRC', 'ASK', 'BID', 'RET', 'OPENPRC', 'ASKHI', 'BIDLO', 'DATE', 'start', 'ending'], inplace=True)
    df = df.drop_duplicates(subset='PERMNO')
    df.set_index('PERMNO', inplace=True)

    p = pd.read_pickle('p_values_adj')[test]

    indicators = list(p.columns)
    securities = list(df.index)

    mask = pd.read_csv('data_out.csv')
    mask.set_index('DATE', inplace=True)
    mask = mask.loc[19570301:]
    dates = mask.index

    data = pd.DataFrame(index=dates, columns=pd.MultiIndex.from_product(
        [indicators, securities]))

    for indicator in indicators:
        print(indicator)
        for security in securities:
            mask_series = mask[str(security)]
            mask_series = mask_series.where(np.isnan(mask_series), p[indicator][str(security)])
            data.loc[:, (indicator, security)] = mask_series

    output = pd.DataFrame(index=dates, columns=indicators)

    for indicator in indicators:
        output[indicator] = data[indicator].apply(np.sum, axis=1)

    output.set_index(pd.to_datetime(output.index, format='%Y%m%d'), inplace=True)
    output.to_pickle('proportion-p')

def proportion_graph_output(save_fig):
    data = pd.read_pickle('proportion-p') / 5
#    colors = sns.cubehelix_palette(n_colors=11, start=1.5)
    colors = brewer2mpl.get_map('Paired', 'Qualitative', 12).mpl_colors
    range = 714474, 737059

    indicators = {'MA_a': 'MA (50d)', 'MA_b': 'MA (100d)', 'MA_c': 'MA (200d)',
                  'TRB_a_min': 'Rolling min (50d)', 'TRB_b_min': 'Rolling min (100d)',
                  'TRB_c_min': 'Rolling min (200d)', 'TRB_a_max': 'Rolling max (50d)',
                  'TRB_b_max': 'Rolling max (100d)', 'TRB_c_max': 'Rolling max (200d)',
                  'ROUND_up': 'Round up', 'ROUND_down': 'Round down'}

    colors = {'MA_a': colors[0], 'MA_b': colors[1], 'MA_c': colors[2],
                  'TRB_a_min': colors[3], 'TRB_b_min': colors[4],
                  'TRB_c_min': colors[5], 'TRB_a_max': colors[6],
                  'TRB_b_max': colors[7], 'TRB_c_max': colors[8],
                  'ROUND_up': colors[9], 'ROUND_down': colors[11]}

    linestyle = {'MA_a': ':', 'MA_b': ':', 'MA_c': ':',
                  'TRB_a_min': '--', 'TRB_b_min': '--',
                  'TRB_c_min': '--', 'TRB_a_max': '--',
                  'TRB_b_max': '--', 'TRB_c_max': '--',
                  'ROUND_up': '-', 'ROUND_down': '-'}

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    ax.set_xlim(range)

    for key in indicators.keys():
        ax.plot(data.index, data[key], label=indicators[key], color=colors[key], linestyle=linestyle[key])

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))

    for tick in ax.get_xticklabels():
        tick.set_fontname('Source Sans Pro')

    for tick in ax.get_yticklabels():
        tick.set_fontname('Source Sans Pro')

    ax.set_title('Proportion of statistically significant index members', fontdict={
        'fontname': 'Source Sans Pro',
        'weight': 'normal',
        'size': 14})

    ax.legend(loc=2, framealpha=1, edgecolor=None, prop={'family': 'Source Sans Pro', 'size': 'x-small'})
    plt.tight_layout()

    if save_fig:
        plt.savefig('/Users/Ryan/OneDrive/thesis_new/graphics/proportiongraph.pdf', format='pdf')

def distexample_graph(indicator, save_fig):

    securities = list(container.index)
    security = securities[np.random.randint(0, len(securities) - 1)]
    all = container[indicator][str(security)]['all']
    signal = container[indicator][str(security)]['signal']

    fig, ax = plt.subplots(nrows=1, ncols=2)

    fig.set_size_inches(5, 3)

    ax[0].hist(all, bins=np.linspace(-.1,.1,num=20), histtype='stepfilled', color=colors[1])
    ax[0].set_xlim(-.1, .1)
    ax[0].set_title('Proportion of index securities significant', fontdict={
        'fontname': 'Source Sans Pro',
        'weight': 'normal',
        'size': 14})
    ax[0].set_xticks([-.05, 0, .05])

    ax[1].hist(signal, bins=np.linspace(-.1,.1,num=20), histtype='stepfilled', color=colors[1])
    ax[1].set_xlim(-.1, .1)
    ax[1].set_title('Signal returns', fontdict={
        'fontname': 'Source Sans Pro',
        'weight': 'normal',
        'size': 14})
    ax[1].set_xticks([-.05, 0, .05])

    for tick in ax[0].get_xticklabels():
        tick.set_fontname('Source Sans Pro')
    for tick in ax[0].get_yticklabels():
        tick.set_fontname('Source Sans Pro')
    for tick in ax[1].get_xticklabels():
        tick.set_fontname('Source Sans Pro')
    for tick in ax[1].get_yticklabels():
        tick.set_fontname('Source Sans Pro')

    if save_fig:
        plt.savefig('/Users/Ryan/OneDrive/thesis_new/graphics/distexample.pdf', format='pdf')