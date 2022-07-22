"""
A bi-directional LSTM RNN algorithm that uses the past 10 days values
of momentum, beta, value, quality, 3-day stock performance, 3-day industry-adjusted
stock performance, 3-day industry performance, and 3-day returns for the S&P to
predict the next 3-day return for each stock. All values are normalized relative
to each other, and the strategy is volatility timed to be invested in the S&P500
when S&P500 volatility is below 15% and invested in a market neutral strategy with 15 longs
and 15 shorts when S&P500 volatility is above 15%.
"""

# To Do List
# 0) Design better workflow between local and Heroku runs using Git
# 1) Trading Execution: Build limit order trading execution system
# 2) Auto-Retrain Model
# 3) Build web app with database to monitor performance and positions
# 4) Incorporate small cap factor into model (deciles)
# 5) Build trading cost monitor
# 6) Simplify format pipeline function
# 7) Iron out kinks in order_target_pct function
# 8) Monitor 'listen for' trade executions
# 9) Rebuild algorithm with a single higher-quality data source

import os
import pickle

import alpaca_trade_api as tradeapi
from keras.models import load_model
import logbook
import numpy as np
import pandas as pd
from pipeline_live.data.alpaca.pricing import USEquityPricing
from pipeline_live.data.iex.fundamentals import IEXCompany
from pipeline_live.data.iex.fundamentals import IEXKeyStats
from pipeline_live.data.polygon.filters import IsPrimaryShareEmulation
from pylivetrader.api import (get_open_orders, date_rules, time_rules, schedule_function,
                              symbol, attach_pipeline, pipeline_output)
import redis
from zipline.pipeline import Pipeline

import custom_factors as cf
from custom_orders import order_target_pct
from algo_functions import optimize_portfolio, get_rets

LOG = logbook.Logger('ALGO')

APCA_API_KEY_ID = os.environ.get('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.environ.get('APCA_API_SECRET_KEY')
APCA_API_BASE_URL = os.environ.get('APCA_API_BASE_URL')
API = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL)


def initialize(context):

    """Initialize context to store data for the algorithm."""

    LOG.info('Initializing Algorithm')

    # Adjustable variables
    context.avoid_trades = []
    context.trade_restrictions = ['ANDV', 'DDAIF', 'DHR', 'HL', 'FTV', 'LPX', 'DDAIF',
                                  'NNHE', 'NVST', 'PNM', 'SKY', 'XSAU']

    rebalance_hours = 0.0
    rebalance_minutes = 8.0
    context.long_exposure = 0.85
    context.short_exposure = -0.85
    context.num_longs = 15
    context.num_shorts = 15
    context.spyleverage = 1.25
    context.std_cutoff = 0.15
    context.idays = 3

    # Fixed variables
    rebalance_start_time = rebalance_hours*60 + rebalance_minutes
    context.combined_restrictions = context.avoid_trades + context.trade_restrictions
    context.rebalance_complete = False
    context.trade_queue = {}
    context.rolling_portfolios = []
    context.clear_queue_run = 0
    context.long_weight = context.long_exposure / context.num_longs
    context.short_weight = context.short_exposure / context.num_shorts
    context.SPY = symbol('VOO')

    redis_data = redis.from_url(os.environ.get("REDIS_URL"))
    try:
        loaded_state = pickle.loads(redis_data.get('pylivetrader_redis_state'))
        loaded_ndays = loaded_state['ndays']
        LOG.info('Loaded ndays = {}'.format(loaded_ndays))
    except:
        loaded_ndays = 0
        LOG.info('No state has been loaded: ndays = {}'.format(loaded_ndays))

    context.ndays = loaded_ndays

    attach_pipeline(make_pipeline(), 'pipeline')

    schedule_function(calculate_weights, date_rules.every_day(),
                      time_rules.market_open(minutes=(rebalance_start_time-7)))

    schedule_function(rebalance, date_rules.every_day(),
                      time_rules.market_open(minutes=rebalance_start_time))

    clear_queue_frequency = 1
    clear_queue_duration = 56
    clear_queue_start = int(rebalance_start_time) + 4
    for minutez in range(clear_queue_start,
                         clear_queue_start + clear_queue_duration,
                         clear_queue_frequency):
        schedule_function(clear_queue,
                          date_rules.every_day(),
                          time_rules.market_open(minutes=minutez))

    check_order_frequency = 10
    check_order_duration = 50
    check_order_start = int(rebalance_start_time) + 10
    for minutez in range(check_order_start,
                         check_order_start + check_order_duration,
                         check_order_frequency):
        schedule_function(check_order_status,
                          date_rules.every_day(),
                          time_rules.market_open(minutes=minutez))

    eod_operations_start_time = int(rebalance_start_time) + 70
    schedule_function(eod_operations, date_rules.every_day(),
                      time_rules.market_open(minutes=eod_operations_start_time))

def make_pipeline():

    """Build the pipeline to pull daily data."""

    LOG.info('Making Pipeline')

    mktcap = IEXKeyStats.marketcap.latest
    portfolio_value = float(API.get_account().portfolio_value)
    price = USEquityPricing.close.latest
    max_price = portfolio_value * 0.85 / 15
    price_filter = (price > 5.00) & (price < max_price)
    primary_share = IsPrimaryShareEmulation()
    universe_mask = (primary_share & price_filter)

    universe = mktcap.top(2000, mask=universe_mask)

    pipe = Pipeline({
        'ind': IEXCompany.industry.latest,
        'type': IEXCompany.issueType.latest,
        'symbol': IEXCompany.symbol.latest,

        'sstr_t01': cf.ComputeSTR_t1(),
        'sstr_t02': cf.ComputeSTR_t2(),
        'sstr_t03': cf.ComputeSTR_t3(),
        'sstr_t04': cf.ComputeSTR_t4(),
        'sstr_t05': cf.ComputeSTR_t5(),
        'sstr_t06': cf.ComputeSTR_t6(),
        'sstr_t07': cf.ComputeSTR_t7(),
        'sstr_t08': cf.ComputeSTR_t8(),
        'sstr_t09': cf.ComputeSTR_t9(),
        'sstr_t10': cf.ComputeSTR_t10(),

        'mom12_t01': cf.ComputeMOM12_t1(),
        'mom12_t02': cf.ComputeMOM12_t2(),
        'mom12_t03': cf.ComputeMOM12_t3(),
        'mom12_t04': cf.ComputeMOM12_t4(),
        'mom12_t05': cf.ComputeMOM12_t5(),
        'mom12_t06': cf.ComputeMOM12_t6(),
        'mom12_t07': cf.ComputeMOM12_t7(),
        'mom12_t08': cf.ComputeMOM12_t8(),
        'mom12_t09': cf.ComputeMOM12_t9(),
        'mom12_t10': cf.ComputeMOM12_t10(),

        'bab_t01': cf.ComputeBAB_t1(),
        'bab_t02': cf.ComputeBAB_t2(),
        'bab_t03': cf.ComputeBAB_t3(),
        'bab_t04': cf.ComputeBAB_t4(),
        'bab_t05': cf.ComputeBAB_t5(),
        'bab_t06': cf.ComputeBAB_t6(),
        'bab_t07': cf.ComputeBAB_t7(),
        'bab_t08': cf.ComputeBAB_t8(),
        'bab_t09': cf.ComputeBAB_t9(),
        'bab_t10': cf.ComputeBAB_t10(),

    }, screen=universe)

    return pipe

def format_pipeline():

    """Restructure pipeline data and add factors that could not
       be calculated within the pipeline."""

    pipe_out = pipeline_output('pipeline')
    pipe_out = pipe_out[(pipe_out.type == 'cs') | (pipe_out.type == 'ad')]
    pipe_format = pipe_out
    pipe_format = pipe_format.dropna()
    #pipe_format = pipe_format[pipe_format['bab_t01'] > 0]
    
    # Fill in columns that are missing momentum
    momcols = ['mom12_t01', 'mom12_t02', 'mom12_t03', 'mom12_t04', 'mom12_t05', 
               'mom12_t06', 'mom12_t07', 'mom12_t08', 'mom12_t09', 'mom12_t10']
    momcoldf = pipe_format[momcols]
    momcoldf = momcoldf.apply(lambda x: x.fillna(x.mean()),axis=1)
    pipe_format.drop(momcols, axis='columns', inplace=True)
    pipe_format[momcols] = momcoldf[momcols]
    
    # Remove securities that are part of industry with <6 constituents
    ind_count = pipe_format.groupby(['ind']).ind.transform('size').values
    pipe_format = pipe_format.assign(ind_count = ind_count)
    pipe_format = pipe_format[pipe_format.ind_count > 5]
    pipe_format = pipe_format.drop(['ind_count'], axis=1)   
    
    pipe_format = pipe_format.assign(iret_t01=cf.compute_iret(pipe_format, 'sstr_t01'),
                                     iret_t02=cf.compute_iret(pipe_format, 'sstr_t02'),
                                     iret_t03=cf.compute_iret(pipe_format, 'sstr_t03'),
                                     iret_t04=cf.compute_iret(pipe_format, 'sstr_t04'),
                                     iret_t05=cf.compute_iret(pipe_format, 'sstr_t05'),
                                     iret_t06=cf.compute_iret(pipe_format, 'sstr_t06'),
                                     iret_t07=cf.compute_iret(pipe_format, 'sstr_t07'),
                                     iret_t08=cf.compute_iret(pipe_format, 'sstr_t08'),
                                     iret_t09=cf.compute_iret(pipe_format, 'sstr_t09'),
                                     iret_t10=cf.compute_iret(pipe_format, 'sstr_t10'))
    
    pipe_format = pipe_format.assign(istr_t01=cf.compute_istr(pipe_format, 'sstr_t01'),
                                     istr_t02=cf.compute_istr(pipe_format, 'sstr_t02'),
                                     istr_t03=cf.compute_istr(pipe_format, 'sstr_t03'),
                                     istr_t04=cf.compute_istr(pipe_format, 'sstr_t04'),
                                     istr_t05=cf.compute_istr(pipe_format, 'sstr_t05'),
                                     istr_t06=cf.compute_istr(pipe_format, 'sstr_t06'),
                                     istr_t07=cf.compute_istr(pipe_format, 'sstr_t07'),
                                     istr_t08=cf.compute_istr(pipe_format, 'sstr_t08'),
                                     istr_t09=cf.compute_istr(pipe_format, 'sstr_t09'),
                                     istr_t10=cf.compute_istr(pipe_format, 'sstr_t10'))

    frame = pipe_format
    pipe_format = pipe_format.drop(['ind', 'type', 'symbol'], axis=1)

    # Standardize Columns
    pipe_format = pipe_format.transform(lambda x: (x - np.nanmean(x)) / (np.nanstd(x)))
    
    # Clip Columns
    pipe_format = pipe_format.transform(lambda x: x.clip(np.percentile(x, 0.5), 
                                                         np.percentile(x, 99.5)))

    spy_rets = get_rets('SPY', 30)
    pipe_format = pipe_format.assign(std_t01=np.log(spy_rets[-20:].std()*np.sqrt(252))+2,
                                     std_t02=np.log(spy_rets[-21:-1].std()*np.sqrt(252))+2,
                                     std_t03=np.log(spy_rets[-22:-2].std()*np.sqrt(252))+2,
                                     std_t04=np.log(spy_rets[-23:-3].std()*np.sqrt(252))+2,
                                     std_t05=np.log(spy_rets[-24:-4].std()*np.sqrt(252))+2,
                                     std_t06=np.log(spy_rets[-25:-5].std()*np.sqrt(252))+2,
                                     std_t07=np.log(spy_rets[-26:-6].std()*np.sqrt(252))+2,
                                     std_t08=np.log(spy_rets[-27:-7].std()*np.sqrt(252))+2,
                                     std_t09=np.log(spy_rets[-28:-8].std()*np.sqrt(252))+2,
                                     std_t10=np.log(spy_rets[-29:-9].std()*np.sqrt(252))+2)

    col_names = ['bab_t01', 'bab_t02', 'bab_t03', 'bab_t04', 'bab_t05', 
                 'bab_t06', 'bab_t07', 'bab_t08', 'bab_t09', 'bab_t10', 
                 'mom12_t01', 'mom12_t02', 'mom12_t03', 'mom12_t04', 'mom12_t05', 
                 'mom12_t06', 'mom12_t07', 'mom12_t08', 'mom12_t09', 'mom12_t10', 
                 'sstr_t01', 'sstr_t02', 'sstr_t03', 'sstr_t04', 'sstr_t05', 
                 'sstr_t06', 'sstr_t07', 'sstr_t08', 'sstr_t09', 'sstr_t10', 
                 'std_t01', 'std_t02', 'std_t03', 'std_t04', 'std_t05', 
                 'std_t06', 'std_t07', 'std_t08', 'std_t09', 'std_t10', 
                 'iret_t01', 'iret_t02', 'iret_t03', 'iret_t04', 'iret_t05',
                 'iret_t06', 'iret_t07', 'iret_t08', 'iret_t09', 'iret_t10', 
                 'istr_t01', 'istr_t02', 'istr_t03', 'istr_t04', 'istr_t05', 
                 'istr_t06', 'istr_t07', 'istr_t08', 'istr_t09', 'istr_t10']

    pipe_format = pipe_format.reindex(columns=col_names)
    pipe_final = pipe_format

    return pipe_final, frame

def construct_tensor(df_in):

    """Takes data frame and turns it into a tensor that can be fed into the
    prediction algorithm."""

    df_copy = df_in.copy()
    df_copy.columns = df_copy.columns.str.split("_", expand=True)
    df_copy.reset_index(level=[0], drop=True, inplace=True)
    df_copy = df_copy.stack(dropna=False)
    samples = len(df_copy.index.levels[0])
    time = len(df_copy.index.levels[1])
    features = len(df_copy.columns)
    data = df_copy.values.reshape(samples, time, features)
    return data

def calculate_vol(context, data):

    """Calculates volatility that is then used to time the algorithm."""

    prices = data.history(context.SPY, 'price', 21, '1d')
    returns = prices.pct_change()
    std_20 = round(returns.std()*np.sqrt(252), 2)
    std_cutoff = context.std_cutoff
    str_weight = np.where(std_20 > std_cutoff, 1, 0)

    try:
        if context.str_weight < str_weight:
            vol_threshold_hit = True
            LOG.info('Volatility threshold hit. Algorithm will trade today.')
        else:
            vol_threshold_hit = False
    except:
        vol_threshold_hit = False

    context.vol_threshold_hit = vol_threshold_hit
    context.str_weight = str_weight

    if str_weight == 0:
        LOG.info('Vol {} is below {} threshold. Algorithm will hold: \
                 S&P500'.format(std_20, std_cutoff))
    else:
        LOG.info('Vol {} is above {} threshold. Algorithm will trade: \
                 MARKET NEUTRAL PORTFOLIO'.format(std_20, std_cutoff))

    return context.str_weight


def before_trading_start(context, data):

    """
    Begins 45 minutes before market open or whenever algorithm starts up and is
    used to load models, run pipeline, and make predictions.
    """

    pipe_data, frame = format_pipeline()
    context.initial_pipeline_output = frame
    context.final_pipeline_output = pipe_data
    
    tensor = construct_tensor(pipe_data)

    model1_name = 'LSTMl8bde6vol_f1.h5'
    model2_name = 'LSTMl8bde6vol_f2.h5'
    model3_name = 'LSTMl8bde6vol_f3.h5'

    LOG.info('Making predictions')
    preds1 = load_model(model1_name).predict(tensor)
    preds2 = load_model(model2_name).predict(tensor)
    preds3 = load_model(model3_name).predict(tensor)
    preds_mean = (preds1 + preds2 + preds3) / 3
    preds = preds_mean.reshape(len(preds_mean))
    frame['preds'] = preds
    preds_df = frame[['preds', 'ind', 'symbol']]
    LOG.info('Successfully made preds_df')

    context.preds_df = preds_df
    context.str_weight = calculate_vol(context, data)

    if context.ndays % context.idays == 0:
        LOG.info('Algorithm will rebalance today')
        LOG.info('Running Pipeline')
        LOG.info('Pipeline successfully run')
        LOG.info('Algorithm ready to trade')

    else:
        LOG.info('Algorithm last run {} days ago and will not run today'.format(context.ndays))

def calculate_weights(context, data):

    """
    Separates predictions into the 40 top and bottom stocks and calculates
    an optimized equal weighted portfolio comprised of 15 longs and 15 shorts.
    """

    LOG.info('Calculating weights')

    pdf = context.preds_df
    assets = API.list_assets()
    asset_dict = {}
    for i in range(len(assets)):
        asset_dict.update({assets[i].symbol: assets[i].easy_to_borrow})

    pdf['etb'] = pdf['symbol'].map(asset_dict)
    pdf = pdf[~pdf.index.duplicated()]
    pdf = pdf[~pdf['symbol'].isin(context.combined_restrictions)]
    pdf_short = pdf[pdf.etb]

    top_40 = pdf['preds'].nlargest(40)
    bottom_40 = pdf_short['preds'].nsmallest(40)
    ret_preds = pd.DataFrame(top_40.append(bottom_40))

    rets_df = pd.DataFrame()
    for security in ret_preds.index:
        try:
            rets_df[security] = get_rets(security.symbol, 25)
        except:
            LOG.warning('Could not get rets for {}'.format(security.symbol))

    ret_preds_trunc = ret_preds[ret_preds.index.isin(rets_df.columns)]
    LOG.info('ret_pred_longs: {}'.format(ret_preds_trunc['preds'].nlargest(25)))
    LOG.info('ret_pred_shorts: {}'.format(ret_preds_trunc['preds'].nsmallest(25)))
    
    longs, shorts = optimize_portfolio(rets_df, ret_preds_trunc)
    context.longs = longs.index.tolist()
    context.shorts = shorts.index.tolist()
    LOG.info('longs: {}'.format(context.longs))
    LOG.info('Shorts: {}'.format(context.shorts))

def rebalance(context, data):

    """
    The execution module that checks the status of volatility to determine what
    to trade and then places orders. Includes a feature to determine whether a
    circuit breaker has been hit.
    """

    if context.ndays % context.idays == 0 or context.vol_threshold_hit:

        if context.str_weight == 1:

            LOG.info('Rebalancing existing positions')
            not_tradeable_count = 0

            for security in context.portfolio.positions:
                if security not in context.longs and security not in context.shorts:
                    if data.can_trade(security):
                        order_target_pct(security.symbol, 0.0, 'market')
                    else:
                        not_tradeable_count += 1
                        LOG.warning('{} is not able to trade'.format(security.symbol))
                        context.trade_queue[security] = 0.0
            LOG.info('Rebalancing existing positions complete!')

            # Check circuit breaker
            if not_tradeable_count/30 > 0.75:
                LOG.info('Circuit breaker may have been hit')

            LOG.info('Buying new long positions')
            for security in context.longs:
                if data.can_trade(security):
                    order_target_pct(security.symbol, context.long_weight, 'market')
                else:
                    LOG.warning('{} is not able to trade'.format(security.symbol))
                    context.trade_queue[security] = context.long_weight
            LOG.info('Buying new long positions complete!')

            LOG.info('Selling new short positions')
            for security in context.shorts:
                if data.can_trade(security):
                    order_target_pct(security.symbol, context.short_weight, 'market')
                else:
                    LOG.warning('{} is not able to trade'.format(security.symbol))
                    context.trade_queue[security] = context.short_weight
            LOG.info('Selling new short positions complete!')
            LOG.info('Trading complete!')

            context.rebalance_complete = True

        else:
            for security in context.portfolio.positions:
                if security.symbol not in context.SPY.symbol:
                    if data.can_trade(security):
                        order_target_pct(security.symbol, 0.0, 'market')
                    else:
                        LOG.warning('{} is not able to trade'.format(security.symbol))
                        context.trade_queue[symbol] = 0.0

            order_target_pct(context.SPY, context.spyleverage, 'market')

    context.ndays += 1
    LOG.info('Ndays = {}'.format(context.ndays))

def clear_queue(context, data):

    """
    Every minute, attempts to take any stocks that were not able to trade and
    tries again. The first time it will show details about stock, and every 5
    minutes thereafter it will log what is left in the trade queue.
    """

    if context.rebalance_complete:
        if bool(context.trade_queue):
            LOG.info('Attempting to clear trading queue')
            remove_queue_list = []
            for security, amount in context.trade_queue.items():
                if data.can_trade(security):
                    order_target_pct(security.symbol, amount, 'market')
                    remove_queue_list.append(security)
                else:
                    if context.clear_queue_run == 0:
                        LOG.warning('{} is not able to trade'.format(security.symbol))
                        LOG.info('Asset info: {}'.format(API.get_asset(security.symbol)))
            for security in remove_queue_list:
                del context.trade_queue[security]
            context.clear_queue_run += 1
            if bool(context.trade_queue):
                if context.clear_queue_run % 5 == 0:
                    LOG.info('Items remaining in trade queue: {}'.format(context.trade_queue))
            else:
                LOG.info('Trade queue is now empty')

def check_order_status(context, data):

    """Logs any open orders every 10 minutes."""

    open_orders = get_open_orders()
    if open_orders:
        LOG.info('Open Orders: {}'.format(open_orders))

def eod_operations(context, data):

    """
    End of day operations designed to reset context variables for the next
    trading day and log any important messages about the day's activity.
    """

    context.rebalance_complete = False
    if not bool(context.trade_queue):
        LOG.info('All trades were executed for the day')
    else:
        LOG.info('Trading ended without executing the following \
                 trades: {}'.format(context.trade_queue))
    context.clear_queue_run = 0
