from ai_agent import Agent
from utils import *
import market_env as env



def learn():

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
				print("Buy  @ " + formatPrice(data[t]))
				reward = 0
				trade_count += 1

			elif action == 2 and len(agent.open_orders) > 0:  # sell (or exiting trade)

				bought_price = agent.open_orders.pop(0)
				profit = data[t] - bought_price
				reward = env.get_reward(profit)
				total_profit += profit
				print("Sell @ " + formatPrice(data[t]) + " | Profit: " + formatPrice(profit))
			else:# hold
			#	print ("hold")
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

			if len(agent.memory) > batch_size:
				agent.learn(batch_size)

		if e % 10 == 0:
			agent.model.save("files/output/model_ep" + str(e))
			print(f'saved model at files/output/model_ep{str(e)}')


stock_name    = '^GSPC_20'#^GSPC  ^GSPC_2011
window_size   = 10# (t) 10 days
episode_count = 31# minimum 200 episodes for results. episode represent trade and learn on all data.
batch_size    = 10# learn  model every bar start from bar # batch_size
use_existing_model = False
agent         = Agent(window_size, use_existing_model, '')
data          = getStockDataVec(stock_name)
l             = len(data) - 1

print(f'Running {episode_count} episodes, on {stock_name} has {l} bars, window of {window_size}, batch of {batch_size}')
learn()
print(f'finished learning the model. now u can backtest the model created in files/output/ on any stock')
print('python backtest.py ')
