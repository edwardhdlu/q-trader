import keras
from keras.models import load_model

import numpy as np
from agent.agent import Agent
import matplotlib.pyplot as plt



class AgentEval(Agent):
	def __init__(self, state_size):
		Agent.__init__(self, state_size)

	def _model(self):
		return load_model("models/model_ep200")

	def act(self, state):
		options = self.model.predict(state)
		return np.argmax(options[0])



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



WINDOW_SIZE = 7 # one week

agent = AgentEval(WINDOW_SIZE)
data = stockDataVec("BABA_2015")
l = len(data) - 1

state = normalizeData(data, 0, WINDOW_SIZE) # normalized extended
total_profit = 0
agent.inventory = []

# plot data
time_series = []
buy_set = [[], []]
sell_set = [[], []]

for t in xrange(l):
	time_series.append((t, data[t]))
	action = agent.act(state)

	done = True if t == l - 1 else False
	batch_size = 32

	# sit
	next_state = normalizeData(data, t + 1, WINDOW_SIZE)
	reward = 0

	if action == 1: # buy
		buy_set[0].append(t)
		buy_set[1].append(data[t])

		agent.inventory.append(data[t])
		#print "Buy: $" + str(agent.inventory[-1])
	elif action == 2 and len(agent.inventory) > 0: # sell
		sell_set[0].append(t)
		sell_set[1].append(data[t])

		bought_price = agent.inventory.pop(0)
		reward = max(data[t] - bought_price, 0)
		total_profit += data[t] - bought_price

		#print "Sell: $" + str(data[t])
		#print "Profit: $" + str(data[t] - bought_price)


	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	if done:
		print "---------- Total Profit: $" + str(total_profit)

	if len(agent.memory) > batch_size:
		agent.exp_replay(batch_size)

plt.plot(data)
plt.scatter(buy_set[0], buy_set[1], color="green")
plt.scatter(sell_set[0], sell_set[1], color="red")
plt.show()
