import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque


class Agent:
    def __init__(self, num_features, use_existing_model=False, model_name="", random_action_min=0.1,
                 random_action_decay=0.999995, num_neurons=64, future_reward_importance=0.95):
        self.memory = deque(maxlen=100000)
        self.model_name = model_name
        self.use_existing_model = use_existing_model
        self.actions = ['hold', 'buy', 'sell']
        self.action_size = len(self.actions)
        self.gamma = future_reward_importance  # discount rate, determines the importance of future rewards.
        # gamma=0 agent learns to consider current rewards, =1 strives for a long-term high reward
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = random_action_min  # we want the agent to explore at least this amount.
        self.epsilon_decay = random_action_decay  # we want to decrease the number of explorations as it gets good
        self.num_trains = 0
        self.num_neurons = num_neurons
        self.num_features = num_features  # normalized previous days
        self.model = load_model("files/output/" + model_name) if use_existing_model else self._build_net()

    def _build_net(self):
        model = Sequential()
        model.add(Dense(units=np.maximum(int(self.num_neurons / 1), 1), activation="relu", input_dim=self.num_features))
        model.add(Dense(units=np.maximum(int(self.num_neurons / 2), 1), activation="relu"))
        model.add(Dense(units=np.maximum(int(self.num_neurons / 8), 1), activation="relu"))
        model.add(Dense(units=self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # best action is a tradeoff bw predicting based on past(exploitation) and by exploration randomly:
    # letting the model predict the action of current state based on the data you trained
    def choose_best_action(self, state):

        # exploring from time to time
        if self.use_existing_model == False:
            prob_exploit = np.random.rand()
            if prob_exploit < self.epsilon:
                random_action = random.randrange(self.action_size)
                return random_action
        # exploiting = predicting
        pred = self.model.predict(state)
        best_action = np.argmax(pred[0])
        # print(f'best_action found by predicting={self.actions[best_action]}')
        return best_action

    # fit model based on data x,y:  y=reward, x=state, action
    # This training process makes the neural net to predict the action to do based on specific state.
    # using experience replay memory.
    def experience_replay(self, batch_size):
        memory_batch = self.prepare_mem_batch(batch_size)

        for curr_state, action, reward, next_state, done in memory_batch:
            #print(f'curr_state={curr_state}, next_state={next_state}, reward={reward}, action ={action}')
            if not done:
                # predict the future discounted reward
                reward_pred = self.model.predict(next_state)  # [0, 0, 0.0029]   target=0.0036
                # maximum future reward for this state and action (s,a) is the immediate reward r plus maximum future reward for the next state
                target = reward + self.gamma * np.amax(reward_pred[0])
                # the bellman equation for discounted future rewards. https://www.youtube.com/watch?v=8vBXARV_ufk
            else:
                target = reward
            # make the agent to approximately map the current state to future discounted reward - y_f
            y_f = self.model.predict(curr_state)
            y_f[0][action] = target  # only chosen action value will change
            self.model.fit(curr_state, y_f, epochs=1, verbose=0)
            self.num_trains += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

            # print(f'epsilon={self.epsilon}')

    # increases learning speed with mini-batches
    def prepare_mem_batch(self, mini_batch_size):
        mini_batch = []
        # mini_batch = random.sample(self.memory, batch_size)#sample is not a good choice in timeseries data
        l = len(self.memory)
        for i in range(l - mini_batch_size, l):
            mini_batch.append(self.memory[i])
        return mini_batch
