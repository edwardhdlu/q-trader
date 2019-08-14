from ai_agent import Agent
from utils import *
import market_env as env
from numpy import array
import matplotlib.pyplot as plt
from datetime import datetime
import time

def learn():
	profit_vs_episode = []
	trades_vs_episode = []
	for e in range(episode_count + 1):
		print("Episode " + str(e) + "/" + str(episode_count))
		state = env.get_state(data, 0, window_size + 1)
		print(f'state={state}')
		total_profit = 0
		trade_count = 0
		agent.open_orders = []

		for t in range(l):

			action = agent.predict(state)


			if action == 1:  # buy
				agent.open_orders.append(data[t])
				print(f'{t} Buy  @ ' + formatPrice(data[t]))
				reward = 0
				trade_count += 1

			elif action == 2 and len(agent.open_orders) > 0:  # sell (or exiting trade)

				bought_price = agent.open_orders.pop(0)
				profit = data[t] - bought_price
				reward = env.get_reward(profit)
				total_profit += profit
				print(f'{t} Sell @ ' + formatPrice(data[t]) + " | Profit: " + formatPrice(profit))
			else:# hold
				print (f'{t} Hold')
				reward     = 0

			done = True if t == l - 1 else False
			next_state = env.get_state(data, t + 1, window_size + 1)
			#print(f'next_state={next_state}')
			agent.remember(state, action, reward, next_state, done)
			state = next_state

			if done:
				print("---------------------------------------")
				print(f'Total Profit: {formatPrice(total_profit)} , Total trades: {trade_count}')
				print("---------------------------------------")
				profit_vs_episode.append(total_profit)
				trades_vs_episode.append(trade_count)

			if len(agent.memory) > batch_size:
				agent.learn(batch_size)

		if e % 10 == 0:
			agent.model.save("files/output/model_ep" + str(e))
			print(f'saved model at files/output/model_ep{str(e)}')

	return profit_vs_episode, trades_vs_episode


print('time is')
print(datetime.now().strftime('%H:%M:%S'))
start_time = time.time()

stock_name    = '^GSPC_2011'#^GSPC  ^GSPC_2011
window_size   = 10# (t) 10 days
episode_count = 10# minimum 200 episodes for results. episode represent trade and learn on all data.
batch_size    = 10# learn  model every bar start from bar # batch_size
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
