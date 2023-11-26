import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
try:
    from keras.optimizers import Adam
except ImportError:
    from tensorflow.keras.optimizers import Adam

from os import path
import numpy as np
import random
from collections import deque

class Agent:
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.state_size = state_size # normalized previous days
		self.action_size = 3 # sit, buy, sell
		self.memory = deque(maxlen=3000)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995

		self.model=load_model("models/"+model_name) if is_eval else self._model()
		# if not is_eval and path.exists("models/model_ep40"):
		# 	self.model=load_model("models/model_ep40")
		# elif is_eval:
		# 	self.model = load_model("models/" + model_name)
		# else:
		# 	print("model not exist")
		# 	exit()
			
		
	def _model(self):
		model = Sequential()
		model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
		model.add(Dense(units=32, activation="relu"))
		model.add(Dense(units=8, activation="relu"))
		model.add(Dense(self.action_size, activation="linear")) #linear? softmax?
		model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))

		return model

	def act(self, state):
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)

		options = self.model.predict(state)
		return np.argmax(options[0]) #options.shape : (window_size, actoin_size) 

	def expReplay(self, batch_size):
		mini_batch = []
		#l = len(self.memory)
		#print(f"memory size : {l}")
		#for i in range(l - batch_size + 1, l):
		for state, action, reward, next_state, done in random.sample(self.memory, min(batch_size, len(self.memory))):
			if done:
				target = reward
			else:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
			
			target_f = self.model.predict(state) # Q(s,a)
			target_f[0][action] = target 
			self.model.fit(state, target_f, epochs=1, verbose=2) # regression with moving target.. 

		# for i in range(batch_size):
		# 	mini_batch.append(self.memory.popleft())
		
		# for state, action, reward, next_state, done in mini_batch:
		# 	# terminate state
		# 	target = reward
		# 	# not terminate state
		# 	if not done: 
		# 		target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
			
		# 	target_f = self.model.predict(state) # Q(s,a)
		# 	target_f[0][action] = target 
		# 	self.model.fit(state, target_f, epochs=1, verbose=2) # regression with moving target.. 

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay 
