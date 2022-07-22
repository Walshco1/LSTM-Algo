
import alpaca_trade_api as tradeapi
import numpy as np
import logbook
import os
import time

log = logbook.Logger('ALGO')

APCA_API_KEY_ID = os.environ.get('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.environ.get('APCA_API_SECRET_KEY')
APCA_API_BASE_URL = os.environ.get('APCA_API_BASE_URL')

api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL)

"""
conn = tradeapi.stream2.StreamConn()
# Handle updates on an order you've given a Client Order ID.
# The r indicates that we're listening for a regex pattern.
client_order_id = r'my_client_order_id'
@conn.on(client_order_id)
async def on_msg(conn, channel, data):
    # Print the update to the console.
    print("Update for {}. Event: {}.".format(client_order_id, data['event']))

# Start listening for updates.
conn.run(['trade_updates'])
"""

def order_target_pct(asset, target_pct, order_type):
    
    try:
        current_value = float(api.get_position(asset).market_value)
    except:
        current_value = 0.0
    try:
        current_num_shares = float(api.get_position(asset).qty)
    except:
        current_num_shares = 0.0
        
    portfolio_value = float(api.get_account().portfolio_value)
    target_value = target_pct * portfolio_value
    order_value = target_value - current_value
    last_price = api.polygon.last_trade(asset).price
    num_shares = round(order_value / last_price)
    side = 'buy'
    if num_shares <0:
        side = 'sell'
    
    if num_shares==0:
        log.debug('Number of shares requested for {} = 0'.format(asset))
    
    else:
        if order_type == 'market':
            if(current_value != 0 and int(current_value) ^ int(target_value) < 0):
                try:
                    api.submit_order(
                        symbol=asset,
                        qty=abs(current_num_shares),
                        side=side,
                        type='market',
                        time_in_force='day')
                        #client_order_id='flatten_position')
                    #order_id = api.get_order_by_client_order_id('flatten_position')
                    log.debug('Submitted market order for {} shares of {}'.format(num_shares, asset))
                except:
                    log.debug('Could not place order for {}'.format(asset)) 
                
                time.sleep(1)
                
                try:
                    api.submit_order(
                        symbol=asset,
                        qty=(abs(num_shares) - abs(current_num_shares)),
                        side=side,
                        type='market',
                        time_in_force='day')
                    log.debug('Submitted market order for {} shares of {}'.format(num_shares, asset))
                except:
                    log.debug('Could not place order for {}'.format(asset))
            else:
                try:
                    api.submit_order(
                        symbol=asset,
                        qty=abs(num_shares),
                        side=side,
                        type='market',
                        time_in_force='day')
                    log.debug('Submitted market order for {} shares of {}'.format(num_shares, asset))
                except:
                    log.debug('Could not place order for {}'.format(asset))

            
        elif order_type == 'limit':
            quote = api.polygon.last_quote(asset)
            mid_price = np.mean([quote.askprice, quote.bidprice])
            api.submit_order(
                symbol=asset,
                qty=abs(num_shares),
                side=side,
                type='limit',
                time_in_force='day',
                limit_price=mid_price)
            log.debug('Submitted limit order for {} shares of {} at {}'.format(num_shares, asset, mid_price))
            
        else:
            pass
            log.debug('order_type not recognized for {}'.format(asset))
