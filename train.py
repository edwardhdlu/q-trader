from agent.agent import Agent
from functions import getStockDataVec, genStatePartition, formatPrice

EPISODES = 1000
WINDOW_SIZE = 7 # one week
STOCK = "^GSPC_2014"

agent = Agent(WINDOW_SIZE)
data = getStockDataVec(STOCK)
l = len(data) - 1
batch_size = 32

for e in xrange(EPISODES):
	print "Episode " + str(e) + "/" + str(EPISODES)
	state = genStatePartition(data, 0, WINDOW_SIZE + 1)

	total_profit = 0
	agent.inventory = []

	for t in xrange(l):
		action = agent.act(state)

		# sit
		next_state = genStatePartition(data, t + 1, WINDOW_SIZE + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t])
			print "Buy: $" + str(agent.inventory[-1])

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			print "Sell: $" + str(data[t])
			print "Profit: $" + str(data[t] - bought_price)

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print "---------- Total Profit: " + formatPrice(total_profit)

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.model.save("models/model_ep" + str(e))
