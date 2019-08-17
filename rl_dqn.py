import time
from datetime import datetime

from ai_agent import Agent
from utils import *
import numpy as np



def learn():
    profit_vs_episode = []
    trades_vs_episode = []

    for episode in range(episodes + 1):
        #print("Episode " + str(e) + "/" + str(episode_count))
        state = get_state(data, 0, features + 1)
        total_profits = 0
        total_trades  = 0
        agent.open_orders = []

        for t in range(l):

            action = agent.choose_best_action(state)#tradeoff bw predict and random

            reward, total_profits, total_trades = execute_action (action, t, total_profits, total_trades)

            done = True if t == l - 1 else False
            next_state = get_state(data, t + 1, features + 1)
            #print(f'next_state={next_state}')
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f'Episode {episode}/{episodes} Total Profit: {formatPrice(total_profits)} , Total trades: {total_trades}, epsilon: {agent.epsilon}')
                print("---------------------------------------")
                profit_vs_episode.append(total_profits)
                trades_vs_episode.append(total_trades)

            if len(agent.memory) > batch_size:
                agent.experience_replay(batch_size)#fit

        if episode % 10 == 0:
            agent.model.save("files/output/model_ep" + str(episode))
            print(f'files/output/model_ep{str(episode)} saved')

    return profit_vs_episode, trades_vs_episode


def execute_action(action, t, total_profits, total_trades):
    if action == 1:  # buy
        agent.open_orders.append(data[t])
        # print(f'row #{t} Buy  @ ' + formatPrice(data[t]))
        total_trades += 1
        reward = 0


    elif action == 2 and len(agent.open_orders) > 0:  # sell (or exiting trade)

        bought_price = agent.open_orders.pop(0)
        profit = data[t] - bought_price
        total_profits += profit
        reward = get_reward(profit, total_profits)

        #print(f'row #{t} exit @ ' + formatPrice(data[t]) + " | Profit: " + formatPrice(profit))
    else:  # hold
        # print (f'row #{t} Hold')
        reward = 0
    return reward, total_profits, total_trades


# returns an an n-day state representation ending at time t of difference bw close prices. ex. [0.5,0.5,0.5,0.4,0.3,0.2,0.5,0.4,0.3,0.2]
def get_state(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    #add features
    #add cyclic feature(sin, cos)
    #add tech. indicators
    #add screen image
    #add economic data
    #https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114
    return np.array([res])

def get_reward(profit, total_profits):
    #todo follw DeepMind suggestion to clip the reward between [-1,+1](normalize) to improve the stability over other data
    reward       = max(profit, 0)
    #reward      = total_profits#https://stats.stackexchange.com/questions/220508/q-learning-is-the-reward-cumulative-or-the-delta-between-the-previous-and-last/220552
    return reward




print('time is')
print(datetime.now().strftime('%H:%M:%S'))
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True) #prevent numpy exponential #notation on print, default False

start_time = time.time()
seed()
stock_name    = '^GSPC_2011'#^GSPC  ^GSPC_2011
features      = 10# (t) 10 super simple features
episodes      = 100# minimum 200 episodes for results. episode represent trade and learn on all data.
batch_size    = 15# learn  model on  batch_size
use_existing_model = False
agent         = Agent(features, use_existing_model, '')
data          = getStockDataVec(stock_name)
l             = len(data) - 1

print(f'Running {episodes} episodes, on {stock_name} has {l} bars, window of {features}, batch of {batch_size}')
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
