'''''
Utile functions for mean reversion pair trading strategy
'''
import pandas as pd 
import numpy as np 
from itertools import combinations
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

def select_stock_pairs_from_correlation(df_period: pd.DataFrame, 
                                        top_corr_pairs: int=15,
                                        drop_correlation: float=0.97):
    ''''
    From the raw price dataframe (by lookback window)
    Select the top pairs (drop the pairs with too high correlation 
    as they might be different listings for the same company)
    Return a list of all the filtered pairs 
    '''
    df_stacked_corr_mtx = df_period.pct_change().corr().unstack().dropna()
    df_stacked_corr_mtx_sort = df_stacked_corr_mtx[df_stacked_corr_mtx!=1]\
                                  .sort_values(ascending=False)\
                                  .drop_duplicates()\
                                  .reset_index()
    df_top_corr = df_stacked_corr_mtx_sort[df_stacked_corr_mtx_sort[0]<drop_correlation].head(top_corr_pairs)
    # Securities list to test for cointegration
    coint_test_list = list(set(list(df_top_corr['level_0']) + list(df_top_corr['level_1'])))
    return coint_test_list


# Function to find cointegrated pairs
def find_cointegrated_pairs(df_period: pd.DataFrame,
                            coint_test_list: list, 
                            p_value_threshold: float=0.1,
                            num_pairs: int=10) -> list:
    ''''
    Give a list of tickers to test, and p_value_threshold 
    Find cointegrated pairs using Engle-Granger test
    Return a list of tuples with coinegrated pairs, and the correspoinding p-value
    Find the top number of pairs with smallest p-values
    '''
    n = len(coint_test_list)  # Number of tickers
    # Stack adjusted close prices column-wise
    adj_close_data = np.column_stack([df_period[ticker].values for ticker in coint_test_list])  
    pvalue_matrix = np.ones((n, n))  # Initialize p-value matrix with ones
    for i, j in combinations(range(n), 2):  # Iterate over all pairs of tickers
        result = coint(adj_close_data[:, i], adj_close_data[:, j])  # Perform cointegration test
        pvalue_matrix[i, j] = result[1]  # Store p-value
        pvalue_matrix[j, i] = result[1]  # Symmetric entry in the matrix
    coint_pairs_all = [(coint_test_list[i], coint_test_list[j], pvalue_matrix[i,j]) for i in range(n) for j in range(i+1, n) if pvalue_matrix[i, j] < p_value_threshold]
    # Sort by p-value, only return the pairs with smallest p value
    coint_pairs = sorted(coint_pairs_all, key=lambda x: x[2])[:num_pairs]
    return coint_pairs  # Return the p-value matrix and cointegrated pairs

def rescale_price_between_selected_pairs(df_period: pd.DataFrame, 
                                         ticker1: str, 
                                         ticker2: str):
    ''''
    Linear regression on two conintegrated stocks, then scale spread
    Return a dictionary of alpha, beta and the spread array 
    (For later step entry/exit calculation)
    '''
    # ticker1
    x = np.log(df_period[ticker1].values)
    x_const = sm.add_constant(x)
    # ticker2
    y = np.log(df_period[ticker2].values)
    linear_reg = sm.OLS(y, x_const)
    result = linear_reg.fit()
    alpha = result.params[0]
    beta = result.params[1]
    spread_z = np.log(df_period[ticker2].values) - np.log(df_period[ticker1].values)*beta - alpha
    trading_params = {'beta':   beta,
                      'alpha':  alpha, 
                      'spread_z': spread_z}
    return trading_params

def generate_signal_for_pair(trading_param: dict,
                             trigger_std: float=1.96,
                             stoploss_std: float=3.09):
    ''''
    Given two tickers, backtest paris trading strategy using the 
    trading parameters

    Generate target exposures with entry and stop loss condition
    '''
    spread_z = trading_param['spread_z']
    mean = spread_z.mean()
    std = spread_z.std()
    entry_upper_bound = mean + trigger_std * std
    stop_upper_bound = mean + stoploss_std * std
    entry_lower_bound = mean - trigger_std * std
    stop_lower_bound = mean - stoploss_std * std
    # When spread goes above upper bound, then sell. 
    # When spread goes below lower bound, then buy.
    # generate buy-close and sell-close signals
    target_exposure_list = []
    trade_rec_list = []
    target_exposure = 0 
    long_status, short_status = False, False
    mean = spread_z.mean()
    for i in range(spread_z.shape[0]):
        # Enter short when z score spread hit upper limit
        if (spread_z[i] > entry_upper_bound) & (not short_status) & (spread_z[i] < stop_upper_bound):
            target_exposure = -1
            short_status = True
            trade_rec_list.append('enter short')
        # Close short when z score spread hit mean and take profit
        elif (short_status) & (spread_z[i] < mean): 
            target_exposure = 0
            short_status = False
            trade_rec_list.append('close short with profit')
        # Stop loss short position
        elif (short_status) & (spread_z[i] > stop_upper_bound):
            target_exposure = 0
            short_status = False
            trade_rec_list.append('close short with loss')

        # Enter long when z score spread hit lower limit
        elif (spread_z[i] < entry_lower_bound) & (not long_status) & (spread_z[i] > stop_lower_bound):
            target_exposure = 1
            long_status = True
            trade_rec_list.append('enter long')
        # Close long position when z score hit mean and take profit
        elif (long_status) & (spread_z[i] > mean): 
            target_exposure = 0
            long_status = False
            trade_rec_list.append('close long with profit')
        # Stop loss long position
        elif (long_status) & (spread_z[i] < stop_lower_bound):
            target_exposure = 0
            long_status = False
            trade_rec_list.append('close long with loss')

        else:
            trade_rec_list.append('no trade')

        target_exposure_list.append(target_exposure) 

    return target_exposure_list


def calcualte_period_PnL_with_pairs(df_period: pd.DataFrame, 
                                    ticker1: str,
                                    ticker2: str,
                                    trading_param: dict, 
                                    trigger_std: float=1.96, 
                                    stoploss_std: float=3.09,
                                    transaction_cost: float=0.0005):
    '''''
    Give a pair, using the generated signal (target exposure) and hedge ratio
    to calculate PnL
    '''
    target_exposure_list = generate_signal_for_pair(trading_param=trading_param,
                                                    trigger_std=trigger_std,
                                                    stoploss_std=stoploss_std)
    df_period_pct = df_period[[ticker1, ticker2]].pct_change()
    df_period_pct['target_exposure'] = target_exposure_list
    # 1 day between observation and execution, both at close - shift 2 to reflect this logic
    df_period_pct['target_exposure'] = df_period_pct['target_exposure'].shift(2)
    df_period_pct[f'target_exposure_{ticker1}'] = df_period_pct['target_exposure'] * (-1) * trading_param['beta']
    df_period_pct[f'target_exposure_{ticker2}'] = df_period_pct['target_exposure'] 
    # Approximate daily percentage PnL, assuming constant target exposure with no drift. Add transaction cost
    sr_PnL_pct = (((df_period_pct[ticker1]  * df_period_pct[f'target_exposure_{ticker1}']\
                  + df_period_pct[ticker2] * df_period_pct[f'target_exposure_{ticker2}']))\
                  - ((df_period_pct[f'target_exposure_{ticker1}'].diff().abs())* (-1) * transaction_cost)) 
    return sr_PnL_pct.fillna(0)

def run_equal_weight_pairs_portfolio(df_period: pd.DataFrame, 
                                     coint_pairs: list,
                                     trigger_std: float=1.96,
                                     stoploss_std: float=3.09,
                                     num_pairs: int=10,
                                     transaction_cost: float=0.0010):
    ''''
    Run an equal weight portfolio
    '''
    df_portfolio = pd.DataFrame(index=df_period.index)
    for i in range(num_pairs):
        ticker1, ticker2 = coint_pairs[i][0], coint_pairs[i][1]
        trading_params = rescale_price_between_selected_pairs(df_period=df_period,
                                                            ticker1=coint_pairs[i][0],
                                                            ticker2=coint_pairs[i][1])
        sr_PnL_pct = calcualte_period_PnL_with_pairs(df_period=df_period, 
                                                    ticker1=ticker1,
                                                    ticker2=ticker2, 
                                                    trading_param=trading_params,
                                                    trigger_std=trigger_std,
                                                    stoploss_std=stoploss_std,
                                                    transaction_cost=transaction_cost)
        df_portfolio[f'PnL_{ticker1}_{ticker2}'] = sr_PnL_pct * (1/num_pairs)
    df_portfolio['agg_pct_ret'] = df_portfolio.sum(axis=1)
    df_portfolio['NAV'] = (1+df_portfolio.sum(axis=1)).cumprod()
    return df_portfolio
        

if __name__ == '__main__':
    df = pd.read_csv('SPX_components_close_data.csv').set_index('Date')
    df.index = pd.to_datetime(df.index)
    df_last_yr = df[df.index>='2022-01-01']
    coint_test_list = select_stock_pairs_from_correlation(df_period=df_last_yr, 
                                                          top_corr_pairs=15,
                                                          drop_correlation=0.97)


