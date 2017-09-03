import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
from collections import deque
import random



class Trader:
	def __init__(self):
		self.state_size = 7 # normalized previous week
		self.action_size = 3 # sit, buy, sell
		self.memory = deque(maxlen=1000)

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995

		self.model = self._model()

	def _model(self):
		model = Sequential()
		model.add(Dense(units=16, input_dim=self.state_size, activation="relu"))
		model.add(Dense(units=16, activation="relu"))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(lr=0.001))

		return model

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)

		options = self.model.predict(state)
		return np.argmax(options[0])

	def exp_replay(self, batch_size):
		mini_batch = random.sample(self.memory, batch_size)

		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay 



# returns the vector containing stock data from a fixed file
def stockDataVec():
	vec = []

	lines = open("data/^GSPC_2015.csv", "r").read().splitlines()
	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))

	return vec

# prints the formatted values
def printFormatted(price, owned, balance):
	print "$" + "{:20,.2f}".format(price) + " | $" + "{:20,.2f}".format(balance) + " | " + str(int(owned))

# returns an np array containing n stock values from time t - n + 1 to t
def normalizeVec(data, t, n):
	d = t - n + 1
	res = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	m = max(res)
	return np.array([map(lambda x: 1.0 * x / m, res)])



EPISODES = 1000
WINDOW_SIZE = 7 # one week

owned_quantity = 0
amount_spent = 0
amount_earned = 0

# BALANCE = 100000

agent = Trader()
data = stockDataVec()

for e in xrange(EPISODES):
	print "Episode " + str(e) + "/" + str(EPISODES)
	state = normalizeVec(data, 0, WINDOW_SIZE) # normalized extended

	l = len(data) - 1
	for t in xrange(l):
		action = agent.act(state)

		done = True if t == l - 1 else False
		batch_size = 32

		# sit
		next_state = normalizeVec(data, t + 1, WINDOW_SIZE)
		reward = 0

		if action == 1: # buy
			owned_quantity += 1
			amount_spent += data[t]
		elif action == 2: # sell
			if owned_quantity == 0:
				action = 0 # can't sell, force sit
			else:
				owned_quantity -= 1
				amount_earned += data[t]
				reward = data[t]

		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print "Profit earned: $" + str(amount_earned - amount_spent)

		if len(agent.memory) > batch_size:
			agent.exp_replay(batch_size)

	if e % 5 == 0:
		agent.model.save("models/model_ep" + str(e))
