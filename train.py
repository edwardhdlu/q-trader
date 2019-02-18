from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 4:
	print "Usage: python train.py [stock] [window] [episodes]"
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 64

for e in xrange(episode_count + 1):
	print "Episode " + str(e) + "/" + str(episode_count)
	state = getState(data, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []
	b=0
	a=0

	for t in xrange(l):
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0
		a+=1

		if action == 1: # buy
			agent.inventory.append(data[t])
			print "Buy: " + formatPrice(data[t])
			b=b+1
			a=0

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward =(data[t] - bought_price)*100
			total_profit += data[t] - bought_price
			b=0
			a=0
			print "Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price)

		if(b>10 or a>20):
			reward+=-200
		done = True if t == l - 1 else False
		
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print "--------------------------------"
			print "Total Profit: " + formatPrice(total_profit)
			print "--------------------------------"

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.model.save("models/model_ep" + str(e))
