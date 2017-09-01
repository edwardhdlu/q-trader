import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
from collections import deque
import random



class Trader:
	def __init__(self):
		self.state_size = 3 # current price, amount of stocks owned, balance
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



def stockDataVec():
	vec = []

	lines = open("data/^GSPC.csv", "r").read().splitlines()
	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))

	return vec

def printFormatted(price, owned, balance):
	print "$" + "{:20,.2f}".format(price) + " | $" + "{:20,.2f}".format(balance) + " | " + str(int(owned))



EPISODES = 1000
BALANCE = 100000

agent = Trader()
data = stockDataVec()

for e in xrange(EPISODES):
	print "Episode " + str(e) + "/" + str(EPISODES)
	state = np.array([[data[0], 0, BALANCE]])

	l = len(data) - 1
	for t in xrange(l):
		action = agent.act(state)

		done = True if t == l - 1 else False
		batch_size = 32

		price = state[0][0]
		owned_quantity = state[0][1]
		balance = state[0][2]

		if t % 50 == 0:
			printFormatted(price, owned_quantity, balance)

		# sit
		next_state = np.array([[data[t + 1], owned_quantity, balance]])
		reward = 0

		if action == 1: # buy
			if balance < data[t]:
				action = 0 # can't buy, force sit
			else:
				next_state = np.array([[data[t + 1], owned_quantity + 1, balance - data[t]]])
		elif action == 2: # sell
			if owned_quantity == 0:
				action = 0 # can't sell, force sit
			else:
				next_state = np.array([[data[t + 1], owned_quantity - 1, balance + data[t]]])
				reward = data[t]

		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print "Profit earned: $" + str(balance - BALANCE)

		if len(agent.memory) > batch_size:
			agent.exp_replay(batch_size)

	if e % 5 == 0:
		agent.model.save("models/model_ep" + str(e))
