import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def perf_summary(df, bm_col=1, rf_col=-1, save=False):
    '''
    Takes a df of returns (% form), optional benchmark column number, option risk free
    columns number, and optional boolean to save results in excel workbook.
    Returns a df which summarizes performance metrics for each series.
    '''


    df_cp = (df / 100 + 1).cumprod()

    ann_ret = [
        (np.power(df_cp[c][-1], 12 / df.shape[0]) - 1) * 100
        for c in df_cp.columns
    ]

    std_dev = [
        np.std(df[c]) * np.sqrt(12) for c in df.columns
    ]

    df_neg = df[df < 0]
    ds_dev = [
        np.std(df_neg[c]) * np.sqrt(12) for c in df_neg.columns
    ]

    max_dd = []

    for c in df.columns:

        dd = []
        v = 1
        for r in df[c]:

            v = v * (1 + r / 100)

            if v > 1:
                v = 1

            dd.append(v)

        max_dd.append((1 - min(dd)) * -100)

    ann_ex_ret_cp = (df.sub(df.iloc[:, rf_col], axis=0) / 100 + 1).cumprod()

    sharpe = [
        (np.power(ann_ex_ret_cp[c][-1], 12 / ann_ex_ret_cp.shape[0]) - 1) * 100 / sd
        for c, sd in zip(ann_ex_ret_cp.columns, std_dev)
    ]

    sortino = [
        (np.power(ann_ex_ret_cp[c][-1], 12 / ann_ex_ret_cp.shape[0]) - 1) * 100 / ddev
        for c, ddev in zip(ann_ex_ret_cp.columns, ds_dev)
    ]

    beta = [
        df.cov().iloc[i, bm_col] / df.cov().iloc[bm_col, bm_col]
        for i in np.arange(df.shape[1])
    ]

    df_up = df[df.iloc[:, bm_col] >= 0]

    up_capt = [
        (np.power((1 + df_up[c] / 100).prod(), 1 / len(df_up)) - 1) /
        (np.power((1 + df_up.iloc[:, bm_col] / 100).prod(), 1 / len(df_up)) - 1)
        for c in df_up.columns
    ]

    df_down = df[df.iloc[:, bm_col] < 0]

    down_capt = [
        (np.power((1 + df_down[c] / 100).prod(), 1 / len(df_down)) - 1) /
        (np.power((1 + df_down.iloc[:, bm_col] / 100).prod(), 1 / len(df_down)) - 1)
        for c in df_down.columns
    ]

    summary = pd.DataFrame([
        ann_ret,
        std_dev,
        ds_dev,
        max_dd,
        sharpe,
        sortino,
        beta,
        up_capt,
        down_capt],
        columns=df.columns,
        index=[
            'ANN_RET',
            'STD_DEV',
            'DS_DEV',
            'MAX_DD',
            'SHARPE',
            'SORTINO',
            'BETA',
            'UP_CAPT',
            'DOWN_CAPT']
    )

    if save:
        summary.to_excel(str(df.columns[0]) + ' Performance Summary.xlsx')

    return summary


def maxdd(df, dd_col=0):
    '''
    Takes a df of returns (% format) and optional column number inputs.
    Returns a dataframe that contains the top drawdowns, along with
    the start date, trough date, length of time from peak to trough 'DURATION',
    and the length of time it took to recover 'RECOVERY'
    '''

    price_index = (df / 100 + 1).cumprod()
    price_index = np.array(price_index.iloc[:, dd_col])

    RESULTS = []

    for i in np.arange(price_index.shape[0] - 1):

        if price_index[i] > price_index[i + 1]:  # Finds peak index

            try:
                newpeak = [n for n, p in enumerate(price_index[i + 1:]) if p >= price_index[i]][0]  # Looks for newpeak
            except IndexError:
                newpeak = len(price_index[i + 1:])

            trough = min(price_index[i: (i + newpeak)])  # Finds trough value
            troughi = price_index[i: (i + newpeak)].argmin()  # Finds trough index

            DD = round((1 - trough / price_index[i]) * -100, 2)  # Calculates DD from peak

            Peak = df.index[i]

            Trough = df.index[i + troughi]

            Duration = round((troughi / 12), 2)

            Recovery = round((newpeak - troughi) / 12, 2) if price_index[i] > price_index[i + newpeak] else None

            RESULTS.append([Peak, Trough, DD, Duration, Recovery])

    RESULTS = sorted(RESULTS, key=lambda x: x[2])  # Sorts results by DD

    TOPDD = []

    for d in RESULTS:  # Iterates through drawdown list to find
        if not any(d[1] in t for t in TOPDD):  # the largest DD at each unique trough date
            TOPDD.append(d)

    return pd.DataFrame(np.array(TOPDD),
                        columns=['PEAK', 'TROUGH', 'DRAWDOWN', 'DURATION', 'RECOVERY']).head(
        25)  # Creates a DataFrame of results



def add_dd(dd_df, ret_df, col_num):
    '''
    Takes a df of drawdowns with peak and trough in columns 0 and 1, a df of returns (% form),
    and a column number.
    Returns a drawdown df with col_num's series returns during those drawdowns
    '''


    ret_df_indexed = (ret_df / 100 + 1).cumprod()

    dd_df[ret_df_indexed.columns[col_num]] = [
        np.round(
            (ret_df_indexed[ret_df_indexed.columns[col_num]].loc[dd_df.iloc[i, 1]] /
             ret_df_indexed[ret_df_indexed.columns[col_num]].loc[dd_df.iloc[i, 0]] - 1) * 100
            , 2)
        for i in np.arange(dd_df.shape[0])
    ]

    return dd_df
