A bi-directional LSTM RNN algorithm that uses the past 10 days values
of momentum, beta, value, quality, 3-day stock performance, 3-day industry-adjusted
stock performance, 3-day industry performance, and 3-day returns for the S&P to
predict the next 3-day return for each stock. All values are normalized relative
to each other, and the strategy is volatility timed to be invested in the S&P500
when S&P500 volatility is below 15% and invested in a market neutral strategy with 15 longs
and 15 shorts when S&P500 volatility is above 15%.

Three LSTM models are trained to elimate look-ahead bias in the training process. The
predictions of the three models are then averaged before being fed into the portfolio
optimizer which generates a the long and short positions.

The algorithm is configured to run off the Alpaca Markets brokerage platform, although
API keys have been removed to protect privacy of the accounts. 
