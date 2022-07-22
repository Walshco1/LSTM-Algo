import pandas as pd
import cvxpy as cvx
import numpy as np
#import matplotlib.pyplot as plt

from datetime import datetime
from datetime import timedelta
import alpaca_trade_api as tradeapi
import os

def optimize_portfolio(rdf, rpreds):
    
    # Define data
    mu = rpreds.values.reshape(len(rpreds), 1)
    sigma = rdf.cov().values
    n = len(sigma)
    
    # Define parameters
    w = cvx.Variable(n)
    gamma = cvx.Parameter(nonneg=True)
    
    # Define problem
    ret = mu.T*w
    risk = cvx.quad_form(w, sigma)
    objective = cvx.Maximize(ret - gamma*risk)
    constraints=[cvx.sum(w) == 0,
                 cvx.norm(w, 1) <= 1.0,
                 w >= -0.05,
                 w <= 0.05]
    prob = cvx.Problem(objective, constraints)
    
    # Run optimization
    SAMPLES = 100
    risk_data = np.zeros(SAMPLES)
    ret_data = np.zeros(SAMPLES)
    gamma_vals = np.logspace(-2, 3, num=SAMPLES)
    
    w_list = []
    for i in range(SAMPLES):
        gamma.value = gamma_vals[i]
        prob.solve(solver=cvx.SCS)
        w_list.append(w.value)
        risk_data[i] = cvx.sqrt(risk).value
        ret_data[i] = ret.value
    
    # Get optimal Portfolio
    sharpe = (ret_data-ret_data[-1])/risk_data
    #plt.plot(sharpe)
    wtidx = sharpe.argmax()
    pweights = w_list[wtidx]
    
    #Plot efficient frontier
    #markers_on = [wtidx]
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plt.plot(risk_data, ret_data, 'g-')
    #for marker in markers_on:
    #    plt.plot(risk_data[marker], ret_data[marker], 'bs')
    #    ax.annotate(r"$\gamma = %.2f$" % gamma_vals[marker], 
    #                xy=(risk_data[marker]+0.0002, 
    #                    ret_data[marker]-0.0005))
    #plt.xlabel('Risk')
    #plt.ylabel('Return')
    #plt.show()
    
    odf = pd.DataFrame(rpreds)
    odf['wts'] = np.round(pweights,3)

    # Output longs and shorts
    longs = odf.wts.nlargest(15)
    shorts = odf.wts.nsmallest(15)
    return longs, shorts


def get_rets(symbol, ndays):
    
    APCA_API_KEY_ID = os.environ.get('APCA_API_KEY_ID')
    APCA_API_SECRET_KEY = os.environ.get('APCA_API_SECRET_KEY')
    APCA_API_BASE_URL = os.environ.get('APCA_API_BASE_URL')
    api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL)
    
    today = datetime.today()
    current = today.strftime("%Y-%m-%d")
    cal_days = ndays*2
    past = (today - timedelta(days=cal_days)).strftime("%Y-%m-%d")
    price_df = api.polygon.historic_agg_v2(symbol=symbol, 
                                 multiplier=1, 
                                 timespan='day',
                                 _from=past, 
                                 to=current).df
    price_rets = price_df.close.pct_change()
    price_rets = price_rets[-ndays:]
    return price_rets
