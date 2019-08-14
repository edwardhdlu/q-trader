from agent.agent import Agent
from functions import *
import sys

stock_name    = '^GSPC_20'#^GSPC  ^GSPC_2011
window_size   = 10
episode_count = 10#minimum 200 episodes for results. a number of games we want the agent to play.
batch_size    = 32
reward        = 0
agent         = Agent(window_size)
data          = getStockDataVec(stock_name)
l             = len(data) - 1

print(f'Running {episode_count} episodes, on {stock_name} has {l} bars, window of {window_size}, batch of {batch_size}')

for e in range(episode_count + 1):
	print( "Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)
	print(f'state={state}')
	total_profit    = 0
	trade_count     = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t])
			print ("Buy  @ " + formatPrice(data[t]))
			trade_count +=1

		elif action == 2 and len(agent.inventory) > 0: # sell (or exiting trade)
			bought_price = agent.inventory.pop(0)
			reward       = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			print ("Sell @ " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
		#else:
		#	print ("sit")

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print( "-------------------------------------")
			print( f'Total Profit: {formatPrice(total_profit)} , Total trades: {trade_count}')
			print( "-------------------------------------")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.model.save("files/output/model_ep" + str(e))
