import numpy as np
from agent.agent import Agent



# returns the vector containing stock data from a fixed file
def stockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))

	return vec

# returns a normalized np array containing n stock values from time t - n + 1 to t
def normalizeData(data, t, n):
	d = t - n + 1
	res = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	m = max(res)

	return np.array([map(lambda x: 1.0 * x / m, res)])



EPISODES = 1000
WINDOW_SIZE = 7
STOCK = "BABA_2014"

agent = Agent(WINDOW_SIZE)
data = stockDataVec(STOCK)
l = len(data) - 1

for e in xrange(EPISODES):
	print "Episode " + str(e) + "/" + str(EPISODES)
	state = normalizeData(data, 0, WINDOW_SIZE)

	total_profit = 0
	agent.inventory = []

	for t in xrange(l):
		action = agent.act(state)

		done = True if t == l - 1 else False
		batch_size = 32

		# sit
		next_state = normalizeData(data, t + 1, WINDOW_SIZE)
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

		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print "---------- Total Profit: $" + str(total_profit)

		if len(agent.memory) > batch_size:
			agent.exp_replay(batch_size)

	if e % 10 == 0:
		agent.model.save("models/model_ep" + str(e))
