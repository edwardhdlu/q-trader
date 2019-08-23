import time
from datetime import datetime
import numpy as np
from dqn import Dqn
from utils import *

print('time is') #episodes=2 +window=252 takes 354 sec
print(datetime.now().strftime('%H:%M:%S'))
start_time = time.time()
seed()
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True) #prevent numpy exponential #notation on print, default False

stock_name          = '^GSPC_20'#^GSPC_2001_2010  ^GSPC_1970_2018  ^GSPC_2011
window_size         = 12# (t)   super simple features
episodes            = 11# minimum 200 episodes for results. episode represent trade and learn on all data.
batch_size          = 15#  (int) size of a batched sampled from replay buffer for training
random_action_decay = 0.99995
use_existing_model  = False
data                = getStockDataVec(stock_name)#https://www.kaggle.com/camnugent/sandp500
l                   = len(data) - 1
print(f'Running {episodes} episodes, on {stock_name} (has {l} rows), window={window_size}, batch={batch_size}, random_action_decay={random_action_decay}')

dqn=Dqn()
rewards_vs_episode, profit_vs_episode, trades_vs_episode, epsilon_vs_episode, model_name = \
    dqn.learn\
    (data, episodes,  window_size, batch_size, use_existing_model, random_action_decay)

print(f'finished learning the model. now u can backtest the model {model_name} on any stock')
print('python backtest.py ')
min = np.round((time.time() - start_time)/60,1)
text = f'{stock_name} ({l}),min={min}, window={window_size}, batch={batch_size}, epi={episodes}, eps={np.round(random_action_decay,1)}'




print(f'see plot of profit_vs_episode = {profit_vs_episode}')
plot_barchart(profit_vs_episode	,  "episode vs profit", "episode vs profit", "total profit", "episode", 'green')

print(f'see plot of trades_vs_episode = {trades_vs_episode}')
plot_barchart(trades_vs_episode	,  "episode vs trades", "episode vs trades", "total trades", "episode", 'blue')

print(f'see plot of epsilon_vs_episode = {trades_vs_episode}')
plot_barchart(epsilon_vs_episode	,  "episode vs epsilon", "episode vs epsilon", "epsilon(probability of random action)", text, 'red')

print('time is')
print(datetime.now().strftime('%H:%M:%S'))
print(f'finished run')
print(text)

#Total Profit:  %0.141 , Total trades: 158, hold_count: 0
#0.95        0
#0.9995     16
#0.99995     0
#0.999995   81
#0.9999995   0
#0.99999995  0
#0.999999995 5