from keras.models import load_model

from ai_agent import Agent
from dqn import Dqn
from utils import *
import sys






def bt(num_features, use_existing_model, model_name):
    dqn          = Dqn()
    agent        = Agent(num_features, use_existing_model, model_name)
    state            = dqn.get_state(data, num_features, num_features)
    total_profits    = 0
    total_holds      = 0
    total_buys       = 1
    total_sells      = 0


    for t in range(num_features,l):

        action = agent.choose_best_action(state)#it will always predict

        reward, total_profits, total_holds, total_buys, total_sells = dqn.execute_action (action, data[t], t, total_profits, total_holds, total_buys, total_sells)

        done = True if t == l - 1 else False

        next_state = dqn.get_state(data, t + 1, num_features)
        print(f'row #{t} {agent.actions[action]} @{data[t]}, state1={state}, state2={next_state}, reward={reward}')
        state = next_state

        if done:
            print("-----------------------------------------")
            print(f'Total Profit: {formatPrice(total_profits)} , Total hold/buy/exit trades: {total_holds} / {total_buys} / {total_sells}')
            print("-----------------------------------------")





stock_name    = '^GSPC_1970_2018'#^GSPC_2011  GSPC_2019 GSPC_1970_2019 GSPC_1970_2018
model_name    = 'model_ep1800'#model_ep0, model_ep10, model_ep20, model_ep30
model_inst    = load_model("files/output/" + model_name)
num_features   = model_inst.layers[0].input.shape.as_list()[1]
use_existing_model = True
data = getStockDataVec(stock_name)
l = len(data) - 1
trading_fee=0
print(f'starting back-testing model {model_name} on {stock_name} (file has {l} rows), features = {num_features} ')
bt   (num_features, use_existing_model, model_name)
print(f'finished back-testing model {model_name} on {stock_name} (file has {l} rows), features = {num_features} ')
