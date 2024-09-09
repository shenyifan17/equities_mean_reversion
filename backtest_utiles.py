'''''
Utile functions for mean reversion pair trading strategy
'''
import pandas as pd 
import numpy as np 
from itertools import combinations
from statsmodels.tsa.stattools import coint

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
                            p_value_threshold: float=0.01):
    ''''
    Give a list of tickers to test, and p_value_threshold 
    Find cointegrated pairs using Engle-Granger test
    Return a list of tuples with coinegrated pairs, and the correspoinding p-value
    '''
    n = len(coint_test_list)  # Number of tickers
    # Stack adjusted close prices column-wise
    adj_close_data = np.column_stack([df_period[ticker].values for ticker in coint_test_list])  
    pvalue_matrix = np.ones((n, n))  # Initialize p-value matrix with ones
    for i, j in combinations(range(n), 2):  # Iterate over all pairs of tickers
        result = coint(adj_close_data[:, i], adj_close_data[:, j])  # Perform cointegration test
        pvalue_matrix[i, j] = result[1]  # Store p-value
        pvalue_matrix[j, i] = result[1]  # Symmetric entry in the matrix
    coint_pairs = [(coint_test_list[i], coint_test_list[j], pvalue_matrix[i,j]) for i in range(n) for j in range(i+1, n) if pvalue_matrix[i, j] < p_value_threshold]
    return coint_pairs  # Return the p-value matrix and cointegrated pairs


if __name__ == '__main__':
    df = pd.read_csv('SPX_components_close_data.csv').set_index('Date')
    df.index = pd.to_datetime(df.index)
    df_last_yr = df[df.index>='2022-01-01']
    coint_test_list = select_stock_pairs_from_correlation(df_period=df_last_yr, 
                                                          top_corr_pairs=15,
                                                          drop_correlation=0.97)


