from keras.models import load_model

from ai_agent import Agent
from dqn import Dqn
from utils import *
import sys






def bt(window_size, use_existing_model, model_name):
    dqn          = Dqn()
    agent        = Agent(window_size, use_existing_model, model_name)
    state        = dqn.get_state(data, 0, (window_size + 1))
    total_profit = 0
    total_buys  = 0
    trade_sells  = 0
    total_holds   = 0

    for t in range(l):

        action = agent.choose_best_action(state)#it will always predict

        if action == 1:  # buy
            dqn.open_orders.append(data[t])
            print("Buy  @ " + formatPrice(data[t]))
            reward     = 0
            total_buys += 1

        elif action == 2 and len(dqn.open_orders) > 0:  # sell
            bought_price = dqn.open_orders.pop(0)
            return_rate = data[t] / bought_price
            log_return = np.log(return_rate)#for normal distribution
            total_profit += log_return - trading_fee
            trade_sells += 1
            print("Sell @ " + formatPrice(data[t]) + " | Profit: " + formatPrice(log_return))

        else:# hold
            #print ('Hold')
            total_holds += 1


        done = True if t == l - 1 else False
        next_state = dqn.get_state(data, t + 1, window_size + 1)
        #agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            print("-----------------------------------------")
            print(f'Total Profit: {formatPrice(total_profit)} , Total hold/buy/exit trades: {total_holds} / {total_buys} / {trade_sells}')
            print("-----------------------------------------")






stock_name    = '^GSPC_1970_2018'#^GSPC_2011  GSPC_2019 GSPC_1970_2019 GSPC_1970_2018
model_name    = 'model_ep100'#model_ep0, model_ep10, model_ep20, model_ep30
model_inst    = load_model("files/output/" + model_name)
window_size   = model_inst.layers[0].input.shape.as_list()[1]
use_existing_model = True
data = getStockDataVec(stock_name)
l = len(data) - 1
trading_fee=0
print(f'starting back-testing model {model_name} on {stock_name} (file has {l} rows), window = {window_size} ')
bt(window_size, use_existing_model, model_name)
print(f'finished back-testing model {model_name} on {stock_name} (file has {l} rows), window = {window_size} ')
