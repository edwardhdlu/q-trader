import time
from datetime import datetime

from ai_agent import Agent
from utils import *
import numpy as np



def learn():
    profit_vs_episode = []
    trades_vs_episode = []
    for e in range(episode_count + 1):
        #print("Episode " + str(e) + "/" + str(episode_count))
        state = get_state(data, 0, window_size + 1)
        total_profit = 0
        trade_count = 0
        agent.open_orders = []

        for t in range(l):

            action = agent.choose_best_action(state)#tradeoff bw predict and random

            reward, total_profit, trade_count = execute_decision(action, t, total_profit, trade_count)

            done = True if t == l - 1 else False
            next_state = get_state(data, t + 1, window_size + 1)
            #print(f'next_state={next_state}')
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("---------------------------------------")
                print(f'Episode {e}/{episode_count} Total Profit: {formatPrice(total_profit)} , Total trades: {trade_count}, epsilon: {agent.epsilon}')
                print("---------------------------------------")
                profit_vs_episode.append(total_profit)
                trades_vs_episode.append(trade_count)

            if len(agent.memory) > batch_size:
                agent.experience_replay(batch_size)#fit

        if e % 10 == 0:
            agent.model.save("files/output/model_ep" + str(e))
            print(f'saved model at files/output/model_ep{str(e)}')

    return profit_vs_episode, trades_vs_episode


def execute_decision(action, t, total_profit, trade_count):
    if action == 1:  # buy
        agent.open_orders.append(data[t])
        # print(f'row #{t} Buy  @ ' + formatPrice(data[t]))
        reward = 0
        trade_count += 1

    elif action == 2 and len(agent.open_orders) > 0:  # sell (or exiting trade)

        bought_price = agent.open_orders.pop(0)
        profit = data[t] - bought_price
        reward = get_reward(profit)
        total_profit += profit
        #print(f'row #{t} exit @ ' + formatPrice(data[t]) + " | Profit: " + formatPrice(profit))
    else:  # hold
        # print (f'row #{t} Hold')
        reward = 0
    return reward, total_profit, trade_count


# returns an an n-day state representation ending at time t
def get_state(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return np.array([res])

def get_reward(profit):
    #todo follw DeepMind suggestion to clip the reward between [-1,+1](normalize) to improve the stability over other data
    reward       = max(profit, 0)
    return reward




print('time is')
print(datetime.now().strftime('%H:%M:%S'))
start_time = time.time()
np.random.seed(7)
stock_name    = '^GSPC_2011'#^GSPC  ^GSPC_2011
window_size   = 10# (t) 10 days
episode_count = 100# minimum 200 episodes for results. episode represent trade and learn on all data.
batch_size    = 15# learn  model every bar start from bar # batch_size
use_existing_model = False
agent         = Agent(window_size, use_existing_model, '')
data          = getStockDataVec(stock_name)
l             = len(data) - 1

print(f'Running {episode_count} episodes, on {stock_name} has {l} bars, window of {window_size}, batch of {batch_size}')
profit_vs_episode, trades_vs_episode = learn()
print(f'finished learning the model. now u can backtest the model created in files/output/ on any stock')
print('python backtest.py ')

print(f'see plot of profit_vs_episode = {profit_vs_episode}')
plot_barchart(profit_vs_episode	  ,  "profit per episode" ,  "total profit", "episode", 'green')

print(f'see plot of trades_vs_episode = {trades_vs_episode}')
plot_barchart(trades_vs_episode	  ,  "trades per episode" ,  "total trades", "episode", 'blue')


print('time is')
print(datetime.now().strftime('%H:%M:%S'))
print("------------------------program ran %s seconds -----------------------------------" % (time.time() - start_time))
