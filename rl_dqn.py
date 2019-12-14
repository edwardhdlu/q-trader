import time
from datetime import datetime
import numpy as np
from dqn import Dqn
from utils import *

print('time is')  # episodes=2 +features=252 takes 6 minutes, 1000 episodes 16f  ~1.15h
print(datetime.now().strftime('%H:%M:%S'))
start_time = time.time()

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)  # prevent numpy exponential #notation on print, default False

# change these params
stock_name = 'test'  # ^GSPC_2001_2010  ^GSPC_1970_2018  ^GSPC_2011
num_features = 8  # (int) > 0
num_neurons = 64  # (int) > 0
episodes = 200  # (int) > 0 ,minimum 200 episodes for results. episode represent trade and learn on all data.
batch_size = 8  # (int) > 0 size of a batched sampled from replay buffer for training
random_action_decay = 0.9999  # (float) 0-1
future_reward_importance = 0.95  # (float) 0-1 aka decay or discount rate, determines the importance of future rewards
# If=0 then agent will only learn to consider current rewards. if=1 it will make it strive for a long-term high reward.
random_action_min = 0.05  # (float) 0-1
use_existing_model = False  # (bool)      do not touch this


data = getStockDataVec(stock_name)  # https://www.kaggle.com/camnugent/sandp500
l = len(data) - 1
print(f'Running {episodes} episodes, on {stock_name} (has {l} rows), features={num_features},'
      f' batch={batch_size}, random_action_decay={random_action_decay}')

dqn = Dqn()
profit_vs_episode, trades_vs_episode, epsilon_vs_episode, model_name, num_trains, eps = \
    dqn.learn(data, episodes, num_features, batch_size, use_existing_model, random_action_min, random_action_decay,
              num_neurons, future_reward_importance)

print(f'Learning completed. Backtest the model {model_name} on any stock')
print('python backtest.py ')
minutes = np.round((time.time() - start_time) / 60, 1)  # minutes
text = f'{stock_name} ({l}),t={minutes}, features={num_features}, nn={num_neurons},batch={batch_size},' \
       f' epi={episodes}({num_trains}), eps={np.round(eps, 2)}({np.round(random_action_decay, 5)})'

print(f'see plot of profit_vs_episode = {profit_vs_episode[:10]}')
plot_barchart(profit_vs_episode, "episode vs profit", "episode vs profit", "total profit", "episode", 'green')

print(f'see plot of trades_vs_episode = {trades_vs_episode[:10]}')
plot_barchart(trades_vs_episode, "episode vs trades", "episode vs trades", "total trades", "episode", 'blue')

print(f'see plot of epsilon_vs_episode = {epsilon_vs_episode[:10]}')
plot_barchart(epsilon_vs_episode, "episode vs epsilon", "episode vs epsilon", "epsilon(probability of random action)",
              text, 'red')

print('time is')
print(datetime.now().strftime('%H:%M:%S'))
print(f'finished run')
print(text)
