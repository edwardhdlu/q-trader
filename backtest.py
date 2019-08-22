from keras.models import load_model

from ai_agent import Agent
from utils import *
import sys






def bt():
    state = get_state(data, 0, window_size + 1)
    total_profit = 0
    trade_count  = 0
    reward       = 0
    for t in range(l):

        action = agent.choose_best_action(state)

        if action == 1:  # buy
            agent.open_orders.append(data[t])
            print("Buy  @ " + formatPrice(data[t]))
            reward     = 0
            trade_count += 1

        elif action == 2 and len(agent.open_orders) > 0:  # sell
            bought_price = agent.open_orders.pop(0)


            return_rate = data[t] / bought_price
            log_return = np.log(return_rate)#for normal distribution
            reward = log_return#max(data[t] - bought_price, 0)

            total_profit += log_return - trading_fee
            print("Sell @ " + formatPrice(data[t]) + " | Profit: " + formatPrice(log_return))
        else:# hold
            #print ('Hold')
            reward     = 0

        done = True if t == l - 1 else False
        next_state = get_state(data, t + 1, window_size + 1)
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            print("-----------------------------------------")
            print(f'Total Profit: {formatPrice(total_profit)} , Total trades: {trade_count}')
            print("-----------------------------------------")



# returns an an n-day state representation ending at time t of difference bw close prices. ex. [0.5,0.5,0.5,0.4,0.3,0.2,0.5,0.4,0.3,0.2]
def get_state(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    res = []
    for i in range(n - 1):
        #res.append(sigmoid(block[i + 1] - block[i]))
        res.append(np.log(block[i + 1] / block[i]))
        #res.append(np.log(block[i + 1])-np.log(block[i]))
    #add features
    #add cyclic feature(sin, cos)
    #add tech. indicators
    #add screen image
    #add economic data
    #https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114
    return np.array([res])



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
