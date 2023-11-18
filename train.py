from agent.agent2 import Agent
from functions import *
import sys

if len(sys.argv) != 4:
	print( "Usage: python train.py [stock] [window] [episodes]")
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

agent = Agent(window_size) #10
data = getStockDataVec(stock_name)
l = len(data) - 1 # len(data) : 2515/2

batch_size = 32
print( f"Version  : 0.2.0" )
print( f"Description  : dqn torch implementation " )
print( f"Number of Data  : {l}" )

for e in range(episode_count + 1):
	print( "Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1) #[[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]

	total_profit = 0
	agent.inventory = []
	#agent.memory.clear()

	for t in range(l): #[0,2514/2)
		print(f"#{t} ")
		action = agent.act(state) #[[0.3323, 0.3001, 0.3677]]

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t])
			print ("Buy: " + formatPrice(data[t]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

		done = 1 if t == l - 1 else 0 # True, False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print ("--------------------------------")
			print ("Total Profit: " + formatPrice(total_profit))
			print ("--------------------------------")
			break 

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

		if (t+1) % agent.target_update_period == 0:
			agent.update_target()

	if (e+1) % 10 == 0:
		agent.save(e+1)
