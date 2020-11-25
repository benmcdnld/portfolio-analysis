#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def getmacros():
    '''
    Returns a dataframe of excess returns from macro factors
    '''
    xls_file = r'C:\Users\bmcdonald\Desktop\MSCI Factors.xlsm'
    xls = pd.ExcelFile(xls_file)
    macros = xls.parse('AssetClass', index_col=0, header=None, skiprows=1,
                       names='MKT INT CRD HY CUR RF'.split())
    macros['CRD'] = 0.5*macros.CRD + 0.5*macros.HY  #TurnsC CRD returns into half IG half HY
    macros.drop('HY', axis=1, inplace=True)
    macros['RF'] = macros['RF'] / 12  #3m treasuries are annualized, so this converts to monthly
    macros.index = pd.to_datetime(macros.index).strftime('%Y-%m')
    macros = isolate_credit(macros)  #Residualizes credit returns
    macros.dropna(inplace=True)
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


def macro_prep(return_series, macros, start=None, end=None):
    returns = pd.DataFrame(return_series)

    # Opens macro factors, concats return series, subs rf, drops rf

    df = pd.concat([macros, returns], axis=1, sort=False)
    df.iloc[:, -1] = df.iloc[:, -1] - df.RF

    # Adjust Date range here if applicable
    df = df.loc[start:end]
    df.dropna(inplace=True)

    ex_macros = df.drop('RF', axis=1)

    # Creates separate df for excess returns
    ex_returns = ex_macros.iloc[:, -1]
    ex_returns = pd.DataFrame(ex_returns)
    assert ex_returns.shape[0] > 0, 'Manager returns df is empty'

    ex_macros = ex_macros.iloc[:, :4]
    assert ex_macros.shape[0] > 0, 'Macro returns df is empty'

    return ex_returns, ex_macros



# Opens Excel file with returns
xls_file = r'C:\Users\bmcdonald\Desktop\OMG Manager Returns.xlsx'
xls = pd.ExcelFile(xls_file)

# his section sets month-end date of dats
import datetime
today = datetime.date.today()
first = today.replace(day=1)
lastMonth = first - datetime.timedelta(days=1)
date = lastMonth.strftime("%Y-%m")

# Pulls macro factor data and sets ols model
macros = getmacros()
ols = LinearRegression()

# Creates an empty df for betas and errors
beta_df = pd.DataFrame(columns=['MKT INT CRD CUR'.split()])
excess_returns_df = pd.DataFrame()
predict_df = pd.DataFrame()
error_df = pd.DataFrame()

# Loops through each sheet of returns
for category in 'Index Fund Alt'.split():

    cat = xls.parse(category, index_col=0, skiprows=1)
    cat.index = pd.to_datetime(cat.index).strftime('%Y-%m')
    name = cat.columns  #Creates list of series names

    start = cat.index[120]  #Sets start date to 10 years ago

    betas = []

    #Loops through each series name
    for n in name:
        # Uses factorprep to calc excess manager and factor returns
        ex_returns, ex_macros = macro_prep(cat[n], macros, start)

        model = ols.fit(ex_macros.values,
                        ex_returns.values)

        beta = model.coef_.flatten()
        betas.append(beta)

        errors = ex_returns - model.predict(ex_macros)
        predictions = ex_returns - errors

        excess_returns_df = pd.concat([excess_returns_df, ex_returns], axis=1, sort=False)
        predict_df = pd.concat([predict_df, predictions], axis=1, sort=False)
        error_df = pd.concat([error_df, errors], axis=1, sort=False)

    df = pd.DataFrame(np.stack(betas), columns=['MKT INT CRD CUR'.split()], index=name)

    beta_df = pd.concat([beta_df, df], sort=False)

beta_df.to_excel(f'{date} - TEST Macro Betas.xlsx')

excess_returns_df.sort_index(inplace=True)
predict_df.sort_index(inplace=True)
error_df.sort_index(inplace=True)

excess_returns_df.to_excel(f'{date} - Excess Returns.xlsx')
predict_df.to_excel(f'{date} - Model Predictions.xlsx')
error_df.to_excel(f'{date} - Errors.xlsx')
# error_df.corr().to_excel(f'{date} - TEST Macro Error Correlation.xlsx')