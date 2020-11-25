#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


def getfactors(allocation):
    '''
    Takes a list of lists [[geography, allocation]], in which the allocation
    and produces a dataframe of factor returns for the given allocation
    '''
    xls_file = r'C:\Users\bmcdonald\Desktop\MSCI Factors.xlsm'
    xls = pd.ExcelFile(xls_file)

    #The following sections read each geographic factor returns and multiply them by the allocation

    Canada_factor = xls.parse(allocation[0][0], index_col=0, header=None, skiprows=1,
                              names=['MKT', 'SML', 'VAL', 'MOM', 'MVOL', 'QTY', 'RF']) * allocation[0][1]

    World_factor = xls.parse(allocation[1][0], index_col=0, header=None, skiprows=1,
                             names=['MKT', 'SML', 'VAL', 'MOM', 'MVOL', 'QTY', 'RF']) * allocation[1][1]

    USA_factor = xls.parse(allocation[2][0], index_col=0, header=None, skiprows=1,
                           names=['MKT', 'SML', 'VAL', 'MOM', 'MVOL', 'QTY', 'RF']) * allocation[2][1]

    EAFE_factor = xls.parse(allocation[3][0], index_col=0, header=None, skiprows=1,
                            names=['MKT', 'SML', 'VAL', 'MOM', 'MVOL', 'QTY', 'RF']) * allocation[3][1]

    EM_factor = xls.parse(allocation[4][0], index_col=0, header=None, skiprows=1,
                          names=['MKT', 'SML', 'VAL', 'MOM', 'MVOL', 'QTY', 'RF']) * allocation[4][1]

    factor_returns = Canada_factor + World_factor + USA_factor + EM_factor

    if allocation[2][1] != 0:  # EAFE does not go back as far, so this will prevent unnecessary Null values
        factor_returns += EAFE_factor

    factor_returns.index = pd.to_datetime(factor_returns.index).strftime('%Y-%m')
    factor_returns.index.names = ['DATE']

    return factor_returns.dropna()


def getmacros():
    '''
    Returns a dataframe of excess returns from macro factors
    '''
    xls_file = r'C:\Users\bmcdonald\Desktop\MSCI Factors.xlsm'
    xls = pd.ExcelFile(xls_file)
    macros = xls.parse('AssetClass', index_col=0, header=None, skiprows=1,
                       names='MKT INT CRD HY CUR RF'.split())
    macros['CRD'] = 0.5*macros.CRD + 0.5*macros.HY
    macros.drop('HY', axis=1, inplace=True)
    macros['RF'] = macros['RF'] / 12  #3m treasuries are annualized, so this converts to monthly
    macros.index = pd.to_datetime(macros.index).strftime('%Y-%m')
    return macros


def isolate_credit(df, rolling=120, halflife=36):
    '''
    Takes df of macro factors as well as optional rolling periods, model (wls), and halflife arguments
    Returns df of macro factors that have isolated credit returns
    '''

    weights = [1 * (0.5 ** (1 / halflife)) ** i for i in range(rolling)][::-1]

    wls = LinearRegression()

    betas = [wls.fit(df[['MKT', 'INT']].iloc[i:i + rolling],
                     df['CRD'].iloc[i:i + rolling],
                     sample_weight=weights).coef_
             for i in np.arange(df.shape[0] - rolling)]


    mkt_crd_betas = [b[0] for b in betas]
    int_crd_betas = [b[1] for b in betas]


    # This section uses betas to adjust credit and commodity returns to isolate those factor returns
    df = df.iloc[rolling:].copy()
    df.loc[:, 'CRD'] = df['CRD'] - df['MKT'] * mkt_crd_betas - df['INT'] * int_crd_betas


    return df


def macroprep(manager_rets_csv, start=None, end=None):
    # Opens managers returns and formats them into a df
    csvfile = r'C:\Users\bmcdonald\Desktop\\' + manager_rets_csv
    returns = pd.read_csv(csvfile, index_col=0)

    returns.index = pd.to_datetime(returns.index).strftime('%Y-%m')

    # Opens macro factors, concats manager returns, subs rf, drops rf

    df = pd.concat([getmacros(), returns], axis=1, sort=False)
    macros = df.sub(df['RF'], axis=0)
    macros['CUR'] = macros.CUR + macros.RF  # add back rf as not currency does not require rh deduction
    macros.drop('RF', axis=1, inplace=True)
    macros = isolate_credit(macros)
    macros.dropna(inplace=True)

    # Adjust Date range here if applicable
    macros = macros.loc[start:end]

    # Creates separate df for excess returns
    ex_returns = macros.iloc[:, -1]
    ex_returns = pd.DataFrame(ex_returns)
    assert ex_returns.shape[0] > 0, 'Manager returns df is empty'

    macros = macros.iloc[:, :4]
    assert macros.shape[0] > 0, 'Macro returns df is empty'

    return ex_returns, macros


def macroOLS(manager_return_file, start=None, end=None):
    '''
    Takes a dataframe of manager returns and a dataframe
    of macro factor returns and returns statsmodels summary of regression

    Use <summary_name>.tables[<n>] to get dataframe of different
    summary components
    '''

    # Uses factorprep to calc excess manager and factor returns
    ex_returns, factors = macroprep(manager_return_file, start, end)
    start_date = ex_returns.index[0]
    end_date = ex_returns.index[-1]
    # Adds constant and regresses manager rets against factor returns
    factors = sm.tools.add_constant(factors.to_numpy())
    model = sm.OLS(ex_returns, factors).fit(cov_type='HAC', cov_kwds={'maxlags':12})

    results_summary = model.summary2(xname=['Alpha', 'Market', 'Interest Rate', 'Credit', 'Currency'])

    print(f'Factor exposures from {start_date} to {end_date}')

    return model, results_summary, start_date, end_date


def equityprep(manager_rets_csv, allocation, start=None, end=None):
    '''
    Takes a file name of a csv of manager returns saved on desktop
    and geographic allocation of the manager and produces two dfs
    The first df includes the manager's excess returns over the risk free rate,
    and the other df includes 6 equity factor returns.

    Optional start and end date (YYYY/MM)
    '''
    # Opens benchmark factor returns and formats them in a df

    factors = getfactors(allocation)

    # Calculates excess returns of factors
    market = factors['MKT'].sub(factors['RF'], axis=0)
    rf = factors['RF']
    factors = factors.iloc[:, 1:6].sub(factors['MKT'], axis=0)
    factors = pd.concat([market, factors, rf], axis=1)
    factors.columns = 'MKT SML VAL MOM MVOL QTY RF'.split()

    # Opens managers returns and formats them into a df
    csvfile = r'C:\Users\bmcdonald\Desktop\\' + manager_rets_csv
    returns = pd.read_csv(csvfile, index_col=0)

    returns.index = pd.to_datetime(returns.index).strftime('%Y-%m')

    # Adjust Date range here if applicable
    factors = factors.loc[start:end]

    # Matches dates of factors and manager returns and drops NA
    factors = pd.concat([factors, returns], axis=1, sort=False)
    factors.dropna(inplace=True)

    # Calculates excess manager returns
    ex_returns = factors.iloc[:, -1] - factors.RF
    ex_returns = pd.DataFrame(ex_returns)
    factors = factors.iloc[:, :6]

    return ex_returns, factors


def equityOLS(manager_returns, allocation, start=None, end=None):
    '''
    Takes a dataframe of manager returns and a dataframe
    of equity factor returns and returns statsmodels summary of regression

    Use <summary_name>.tables[<n>] to get dataframe of different
    summary components
    '''

    # Uses factorprep to calc excess manager and factor returns
    ex_returns, factors = equityprep(manager_returns, allocation, start, end)
    start_date = ex_returns.index[0]
    end_date = ex_returns.index[-1]
    # Adds constant and regresses manager rets against factor returns
    factors = sm.tools.add_constant(factors.to_numpy())
    model = sm.OLS(ex_returns, factors).fit(cov_type='HAC', cov_kwds={'maxlags':12})

    results_summary = model.summary2(xname=['Alpha', 'Market', 'Size', 'Value', 'Momentum', 'Min Vol', 'Quality'])

    print(f'Factor exposures from {start_date} to {end_date}')

    return results_summary, start_date, end_date


def exportfactors(regression_summary, manager_name, start, end):
    '''
    Takes a statsmodels factor regression summary, name of manager, start and end date
    and saves an excel file with the regression model stats
    '''

    model = regression_summary.tables[0]
    coefs = regression_summary.tables[1]
    dist = regression_summary.tables[2]

    wbname = manager_name + ' Factor Exposures from ' + start + ' to ' + end + '.xlsx'
    writer = pd.ExcelWriter(wbname, engine='xlsxwriter')

    model.to_excel(writer, sheet_name='Model Stats', index=False, header=False)
    coefs.to_excel(writer, sheet_name='Coef Stats')
    dist.to_excel(writer, sheet_name='Dist Stats', index=False, header=False)

    writer.save()


def plotsummary(regression_summary, manager_name, start, end):
    coefs = regression_summary.tables[1]

    coefs_plot_data = coefs.drop('Market')
    coefs_plot_data.drop(['Std.Err.', 'P>|z|', '[0.025', '0.975]'], axis=1, inplace=True)

    ss = [np.abs(n) >= 1.96 for n in coefs_plot_data.z]

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 6)

    for i in range(len(coefs_plot_data)):
        if ss[i]:
            ax.bar(i, coefs_plot_data['Coef.'][i], align='center', color='tab:blue',
                   hatch='//', zorder=3, label='Statistically Significant')
        else:
            ax.bar(i, coefs_plot_data['Coef.'][i], align='center', color='tab:blue', zorder=3)

    ax.grid(color='silver', axis='y', zorder=0)

    ax.set_xticks(range(len(coefs_plot_data)))
    ax.set_xticklabels(coefs_plot_data.index)

    if any(ss):
        ax.legend(['Statistically Significant'])

    plt.savefig(manager_name + ' Factor Exposures from ' + start + ' to ' + end)

    plt.show()


def rollingsummary(manager_file, allocation, rolling_period):
    manager_er, factor_er = equityprep(manager_file, allocation)

    rolling_factors = {}
    rolling_factors_ss = {}

    for i in np.arange(manager_er.shape[0] - rolling_period + 1):
        excess_rolling = manager_er.iloc[i:rolling_period + i]  # Narrows data to defined rolling period
        factors_rolling = factor_er[i:rolling_period + i]

        date = excess_rolling.index[-1]

        # Adds constant and regresses manager rets against factor returns
        factors_rolling = sm.tools.add_constant(factors_rolling)
        model = sm.OLS(excess_rolling, factors_rolling).fit(cov_type='HAC', cov_kwds={'maxlags':12})

        results_summary = model.summary2(xname=['Alpha', 'Market', 'Size', 'Value', 'Momentum', 'Min Vol', 'Quality'])

        factor_coefs = results_summary.tables[1]

        betas = [factor_coefs.iloc[i][0] for i in np.arange(factor_coefs.shape[0])]

        betas_ss = [float(np.where(factor_coefs.iloc[i][2] >= 1.96, factor_coefs.iloc[i][0], np.nan))
                    for i in np.arange(factor_coefs.shape[0])]  # Puts betas in list, 0 if not significant

        rolling_factors[date] = betas
        rolling_factors_ss[date] = betas_ss

    rolling_df = pd.DataFrame.from_dict(rolling_factors, orient='index',
                                        columns=['Alpha', 'Market', 'Size', 'Value', 'Momentum', 'Min Vol', 'Quality'])

    rolling_ss_df = pd.DataFrame.from_dict(rolling_factors_ss, orient='index',
                                           columns=['Alpha', 'Market', 'Size', 'Value', 'Momentum', 'Min Vol',
                                                    'Quality'])

    start = rolling_df.index[0]
    end = rolling_df.index[-1]

    return rolling_df, rolling_ss_df, start, end


def plotrollingsummary(manager_file, manager_name, allocation, rolling_period=36):
    '''
    Takes the name of a csv of manager returns, the name of that manager,
    the geographic allocation of that manager, and returns a plot of the
    rolling factor betas
    '''

    rolling_df, rolling_ss_df, start, end = rollingsummary(manager_file, allocation, rolling_period)

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 6)

    colors = ['tab:blue', 'gold', 'gray', 'seagreen', 'firebrick', 'plum', 'tab:orange']

    for col, color in zip(rolling_df.columns, colors):
        ax.plot(rolling_df[col], linestyle='dashed', dashes=(5, 1), color=color)
    for col, color in zip(rolling_ss_df.columns, colors):
        ax.plot(rolling_ss_df[col], label=col, color=color)

    ax.legend(loc='lower left')

    ax.set_xticks(np.arange(1, rolling_df.shape[0], 12))
    ax.set_xlabel('Dashed line = not statistically significant', ha='right')

    plt.savefig(manager_name + ' ' + str(rolling_period) + ' Month Rolling Factor Exposures from ' + start + ' to ' + end)

    plt.show()
