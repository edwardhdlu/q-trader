from keras.models import load_model

from ai_agent import Agent
from utils import *
import sys
import market_env as env





def bt():
    state = env.get_state(data, 0, window_size + 1)
    total_profit = 0
    trade_count   = 0
    for t in range(l):
        action = agent.act(state)

        # sit
        next_state = env.get_state(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:  # buy
            agent.open_orders.append(data[t])
            print("Buy  @ " + formatPrice(data[t]))
            trade_count += 1

        elif action == 2 and len(agent.open_orders) > 0:  # sell
            bought_price = agent.open_orders.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print("Sell @ " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("-----------------------------------------")
            print(f'Total Profit: {formatPrice(total_profit)} , Total trades: {trade_count}')
            print("-----------------------------------------")





stock_name    = '^GSPC_2011'#^GSPC  ^GSPC_2011
model_name    = 'model_ep10'

model = load_model("files/output/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True, model_name)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size        = 32
agent.open_orders = []
bt()
