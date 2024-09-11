''''
Main backtest structure to run backtest given a set of parameters
'''
import pandas as pd 
import numpy as np 
from backtest_utiles import * 
from cal_backtest_stats import * 

df = pd.read_csv('SPX_components_close_data.csv').set_index('Date')
df.index = df.index.map(lambda x: pd.to_datetime(str(x)[:10]))

def run_full_backtest(start_year=2010, 
                      end_year=2024, 
                      transaction_cost=0.0020, 
                      top_corr_pairs=20,
                      num_pairs=10,
                      trigger_std=1.96, 
                      stoploss_std=3.09):
    ''''
    Run paris trading backtest: 
        1. Use last year's data to detect co-integration between a basket of stocks
        2. Select a number of pairs to form a mean reversion portfolio
        3. Run an equal weight portfolio every year
    '''
    year_list = list(range(start_year, end_year))
    schedule_list = [{'train': [f'{year}-01-01', f'{year}-12-31'], 'trade': [f'{year+1}-01-01', f'{year+1}-12-31']} for year in year_list]
    # list of portfolio pct returns per year
    LT_agg_pnl_list = []
    for schedule in schedule_list: 
        df_train = df[(df.index>=schedule['train'][0]) & (df.index<=schedule['train'][1])]
        df_trade = df[(df.index>=schedule['trade'][0]) & (df.index<=schedule['trade'][1])]
        df_train = df_train.dropna(axis=1, how='any')
        df_trade = df_trade.dropna(axis=1, how='all')
        # Form a list of potential co-inegrated pairs with high correlation, to test later
        coint_test_list = select_stock_pairs_from_correlation(df_period=df_train, 
                                                            top_corr_pairs=top_corr_pairs,
                                                            drop_correlation=0.97)
        # Find integrated pairs for last year
        coint_pairs = find_cointegrated_pairs(df_period=df_train, 
                                              coint_test_list=coint_test_list,
                                              p_value_threshold=0.2,
                                              num_pairs=num_pairs) 
        # Trade pairs this year, equal weighted portfolio
        df_portfolio = run_equal_weight_pairs_portfolio(df_period=df_trade,
                                                        coint_pairs=coint_pairs, 
                                                        trigger_std=trigger_std,
                                                        stoploss_std=stoploss_std,
                                                        num_pairs=num_pairs,
                                                        transaction_cost=transaction_cost)
        LT_agg_pnl_list.append(df_portfolio['agg_pct_ret'])
    sr_LT_pct_ret = pd.Series()
    # Long term pct return, of aggrated portfolios per year
    for sr_year in LT_agg_pnl_list: 
        sr_LT_pct_ret = pd.concat([sr_LT_pct_ret, sr_year])
    return sr_LT_pct_ret
        

if __name__=='__main__':
    params = {
        'start_year': 2010, 
        'end_year': 2024, 
        'top_corr_pairs': 20,
        'num_pairs': 10,
        'trigger_std': 1.96,
        'stoploss_std': 3.00
    }
    sr_LT_pct_ret = run_full_backtest(**params)
    sr_nav = (100*(1+sr_LT_pct_ret).cumprod())
