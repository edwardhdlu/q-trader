from keras.models import load_model

from ai_agent import Agent
from utils import *
import sys
import market_env as env





def bt():
    state = env.get_state(data, 0, window_size + 1)
    total_profit = 0
    trade_count  = 0
    reward       = 0
    for t in range(l):

        action = agent.predict(state)

        if action == 1:  # buy
            agent.open_orders.append(data[t])
            print("Buy  @ " + formatPrice(data[t]))
            reward     = 0
            trade_count += 1

        elif action == 2 and len(agent.open_orders) > 0:  # sell
            bought_price = agent.open_orders.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print("Sell @ " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
        else:# hold
            #print ('hold')
            reward     = 0

        done = True if t == l - 1 else False
        next_state = env.get_state(data, t + 1, window_size + 1)
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            print("-----------------------------------------")
            print(f'Total Profit: {formatPrice(total_profit)} , Total trades: {trade_count}')
            print("-----------------------------------------")





stock_name    = '^GSPC_2011'#^GSPC  ^GSPC_2011
model_name    = 'model_ep0'#model_ep0, model_ep10, model_ep20, model_ep30
model         = load_model("files/output/" + model_name)
window_size   = model.layers[0].input.shape.as_list()[1]
use_existing_model = True
agent = Agent(window_size, use_existing_model, model_name)
data = getStockDataVec(stock_name)
l = len(data) - 1
trading_fee=0
agent.open_orders = []
print(f'starting back-testing model {model_name} on {stock_name} (has {l} bars), window of {window_size} bars')
bt()
print(f'finished back-testing model {model_name} on {stock_name} (has {l} bars), window of {window_size} bars')
