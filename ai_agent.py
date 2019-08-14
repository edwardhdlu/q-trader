import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque

class Agent:
	def __init__(self, state_size, use_existing_model=False, model_name=""):
		self.state_size    = state_size # normalized previous days
		self.memory        = deque(maxlen=1000)
		self.open_orders   = []
		self.model_name    = model_name
		self.is_eval       = use_existing_model
		self.actions       = ['hold', 'buy', 'sell']
		self.action_size   = len(self.actions)
		self.gamma         = 0.95 #aka decay or discount rate, to calculate the future discounted reward.
		self.epsilon       = 1.0  #aka exploration rate, this is the rate in which an agent randomly decides its action rather than prediction.
		self.epsilon_min   = 0.01 #we want the agent to explore at least this amount.
		self.epsilon_decay = 0.995#we want to decrease the number of explorations as it gets good at trading.

		self.model         = load_model("files/output/" + model_name) if use_existing_model else self._build_net()

	def _build_net(self):
		model = Sequential()
		model.add(Dense(units=64		, activation="relu",  input_dim=self.state_size))
		model.add(Dense(units=32		, activation="relu"))
		model.add(Dense(units=8 		, activation="relu"))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile  (loss="mse"      , optimizer=Adam(lr=0.001))
		return model

	#action is done by predict
	def act(self, state):
		if not self.is_eval and np.random.rand() <= self.epsilon:
			random_action = random.randrange(self.action_size)
			return random_action

		pred_prob = self.model.predict(state)
		pred = np.argmax(pred_prob[0])
		return pred

	#fit model based on data x,y:  y=reward, x = state, action
	def expReplay(self, batch_size):
		mini_batch = []
		l = len(self.memory)
		for i in range(l - batch_size + 1, l):
			mini_batch.append(self.memory[i])

		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model\
				      .fit(state, target_f, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay 
