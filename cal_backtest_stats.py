''''
Backtest statistics 
'''
import pandas as pd 
import numpy as np 

def cal_backtest_stats(sr_ret: pd.Series, 
                       num_days: int=252) -> dict: 
    ''''
    Calculate backtest stats (annualised):
        - Annualised return        'ret'
        - Annualised volatility    'vol' 
        - Annualised Sharpe ratio  'sharpe'
        - Max Drawdown             'max_dd'
        - Max Drawdown / Vol ratio 'max_dd_vol'
        - Sortino ratio            'sortino'

    Args: sr_ret: pd Series with dates as index, pd.Timestamp type
                  daily percentage return as values 
    '''
    # Check if index is pd.Timestamp 
    if type(sr_ret.index[0]) is not pd.Timestamp: 
        index_type = type(sr_ret.index[0])
        raise ValueError(f'sr_ret index type is {index_type}, must be pd.Tiemstamp')
    cum_series = (1+sr_ret).cumprod()
    # number of calendar days
    num_cd = (cum_series.index[-1] - cum_series.index[0]).days 
    num_yr = num_cd / 365
    final_val = cum_series.values[-1]
    ret = final_val**(1/num_yr) - 1
    cum_ret = (1+sr_ret).cumprod().iloc[-1]
    vol = sr_ret.std() * np.sqrt(num_days)
    sharpe = ret/vol
    sr_perf = (1+sr_ret).cumprod()
    sr_dd = (sr_perf - sr_perf.cummax()) / sr_perf.cummax()
    max_dd = sr_dd.min()
    max_dd_vol = max_dd / vol
    downside_dev = sr_ret[sr_ret<0].std() * np.sqrt(num_days)
    sortino = ret/downside_dev
    sr_annual_ret = sr_ret.groupby(pd.Grouper(freq='Y')).apply(lambda x: (1+x).prod()-1)
    sr_month_ret  = sr_ret.groupby(pd.Grouper(freq='M')).apply(lambda x: (1+x).prod()-1)
    pct_positive_yr = len(sr_annual_ret[sr_annual_ret>0]) / len(sr_annual_ret)
    pct_positive_month = len(sr_month_ret[sr_month_ret>0]) / len(sr_month_ret)
    return {'cum_ret': cum_ret,
            'ret': ret,
            'vol': vol, 
            'sharpe': sharpe, 
            'max_dd': max_dd, 
            'max_dd_vol': max_dd_vol, 
            'sortino': sortino, 
            'pct_positive_month': pct_positive_month, 
            'pct_positive_yr': pct_positive_yr}

def generate_stats_table(df: pd.DataFrame):
    ''''
    Generate backtest stats table: 
    Columns are strategy names, values are percentage returns 
    '''
    all_results_dict = {}
    for col in df.columns: 
        all_results_dict[col] = cal_backtest_stats(df[col].pct_change())
    df_result = pd.DataFrame(data=all_results_dict).T.rename({'ret':           'Annualised Return', 
                                                              'vol':           'Annualised Volatility', 
                                                              'sharpe':        'Sharpe Ratio',
                                                              'max_dd':        'Max Drawdown', 
                                                              'max_dd_vol':    'Max Drawdown/Vol Ratio', 
                                                              'sortino':       'Sortino Ratio', 
                                                              'pct_positive_month': '% +ve month'}, 
                                                              axis=1)
    df_result_print = df_result[['Annualised Return', 
                                 'Annualised Volatility', 'Sharpe Ratio', 
                                 'Max Drawdown',
                                 'Max Drawdown/Vol Ratio', 'Sortino Ratio', 
                                 '% +ve month']].T
    return df_result_print


def run_multi_period_stats(sr_ret: pd.Series, 
                           num_days: int=252) -> dict:
    '''''
    Calculate the above function backtest stats for: 
        - Full history 
        - ytd 
        - 1yr 
        - 3yr 
        - 5yr
    Return a dict with above periods as keys and stats dict as values 
    '''
    today = pd.Timestamp.today() 
    this_yr_begin = str(today)[:4] + '-01-01'
    one_yr_ago = str(today - pd.Timedelta('365 days'))[:10]
    three_yr_ago = str(today - pd.Timedeltap('1095 days'))[:10]
    five_yr_ago = str(today - pd.Timedelta('1825 days'))[:10]
    stats_full = cal_backtest_stats(sr_ret, 
                                    num_days)
    stats_ytd = cal_backtest_stats(sr_ret[sr_ret.index>=this_yr_begin],
                                   num_days)
    stats_1yr = cal_backtest_stats(sr_ret[sr_ret.index>=one_yr_ago], 
                                   num_days)
    stats_3yr = cal_backtest_stats(sr_ret[sr_ret.index>=three_yr_ago], 
                                   num_days)
    stats_5yr = cal_backtest_stats(sr_ret[sr_ret.index>=five_yr_ago], 
                                  num_days)
    multi_stats_dict = {'full' :   stats_full, 
                        'ytd':     stats_ytd,
                        '1y':      stats_1yr, 
                        '3y':      stats_3yr, 
                        '5y':      stats_5yr}
    return multi_stats_dict


def expand_multi_stats_dict(multi_stats_dict: dict) -> dict: 
    ''''
    Expand the multi_stats_dict into one level dict
    Used for running parameters permutation stresstest 
    '''
    full_stats_dict = {}
    for period in multi_stats_dict: 
        for stat in multi_stats_dict[period]: 
            full_stats_dict[f'{stat}_{period}'] = multi_stats_dict[period][stat]
    return full_stats_dict



