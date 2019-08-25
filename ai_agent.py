import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque

class Agent:
    def __init__(self, num_features, use_existing_model=False, model_name="", random_action_min=0.1, random_action_decay = 0.999995, num_neurons=64 ,future_reward_importance=0.95):
        self.memory        = deque(maxlen=1000)
        self.model_name    = model_name
        self.use_existing_model       = use_existing_model
        self.actions       = ['hold', 'buy ', 'sell']
        self.action_size   = len(self.actions)
        self.gamma         = future_reward_importance #aka decay or discount rate, determines the importance of future rewards.If=0 then agent will only learn to consider current rewards. if=1 it will make it strive for a long-term high reward.
        self.epsilon       = 1.0  #aka exploration rate, this is the rate in which an agent randomly decides its action rather than prediction.
        self.epsilon_min   = random_action_min  #we want the agent to explore at least this amount.
        self.epsilon_decay = random_action_decay#we want to decrease the number of explorations as it gets good at trading.
        self.num_trains    = 0
        self.num_neurons   = num_neurons
        self.num_features  = num_features # normalized previous days
        self.model         = load_model("files/output/" + model_name) if use_existing_model else self._build_net()


    def _build_net(self ):
        model = Sequential()
        model.add(Dense(units=np.maximum(int(self.num_neurons/1),1) , activation="relu", input_dim=self.num_features))
        model.add(Dense(units=np.maximum(int(self.num_neurons/2),1) , activation="relu"))
        model.add(Dense(units=np.maximum(int(self.num_neurons/8),1) , activation="relu"))
        model.add(Dense(units=self.action_size             , activation="linear"))
        model.compile  (loss="mse"                         , optimizer=Adam(lr=0.001))
        model.summary()
        return model






    def remember(self ,state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    #best action is a tradeoff bw predicting based on past(exploitation) and by exploration randomly: letting the model predict the action of current state based on the data you trained
    def choose_best_action(self, state):

        #exploring from time to time
        if self.use_existing_model == False :
            prob_exploit = np.random.rand()
            if prob_exploit < self.epsilon:
                random_action = random.randrange(self.action_size)
                return random_action
        #exploiting = predicting
        pred = self.model.predict(state)
        best_action = np.argmax(pred[0])
        #print(f'best_action found by predicting={self.actions[best_action]}')
        return best_action

    #fit model based on data x,y:  y=reward, x=state, action
    #This training process makes the neural net to predict the action to do based on specific state.
    #using experience replay memory.
    def experience_replay(self, batch_size):
        memory_batch = self.prepare_mem_batch(batch_size)

        for curr_state, action, reward, next_state, done in memory_batch:


            if not done:
                # predict the future discounted reward
                reward_pred = self.model.predict(next_state)#[0, 0, 0.0029]   target=0.0036
                #In plain English, it means maximum future reward for this state and action (s,a) is the immediate reward r plus maximum future reward for the next state
                target = reward + self.gamma * np.amax(reward_pred[0])#the bellman equation for discounted future rewards. https://www.youtube.com/watch?v=8vBXARV_ufk
            else:
                target = reward #1sell>2sell reward=0.0007, pred=[0,0,0.03], target=0.04  , y_f=[0,0,0.004] > y_f= [0,0,0.004]
                                #1hold>2sell reward=0.0094, pred=[        ], target=0.0094, y_f=[0,0,0.005] > y_f= [0,0,0.0094]
            # make the agent to approximately map
            # the current state to future discounted reward
            # We'll call that y_f  = (0.25,0.25,0.25)
            y_f = self.model.predict(curr_state)
            y_f[0][action] = target #  // only action 2 value will change
            self.model.fit \
                (curr_state
                 , y_f
                 , epochs=1
                 , verbose=0)
            self.num_trains += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

            #print(f'epsilon={self.epsilon}')
        #else:
        #    print(f'warn!!! epsilon={self.epsilon} is too low. agent will exploit too much and not explore. u may have to increase epsilon_decay up to 1')

    #increases learning speed with mini-batches
    def prepare_mem_batch(self, mini_batch_size):
        mini_batch = []
        #mini_batch = random.sample(self.memory, batch_size)#sample is not a good choice in timeseries data
        l = len(self.memory)
        for i in range(l - mini_batch_size , l):
            mini_batch.append(self.memory[i])
        return mini_batch
'''
state, action, reward, next_state, done
[(array([[0.5   , 0.5   , 0.5   , 0.5   , 0.5   , 0.5   , 0.5   , 0.5   ,  0.1584, 0.9983]]) , 2, 0
, array([[0.5   , 0.5   , 0.5   , 0.5   , 0.5   , 0.5   , 0.5   , 0.1584,  0.9983, 0.0624]]) , False)


, (array([[0.5   , 0.5   , 0.5   , 0.5   , 0.5   , 0.5   , 0.5   , 0.1584,  0.9983, 0.0624]]), 0, 0, 
   array([[0.5   , 0.5   , 0.5   , 0.5   , 0.5   , 0.5   , 0.1584, 0.9983,  0.0624, 0.0871]]), False)
   
, (array([[0.5   , 0.5   , 0.5   , 0.5   , 0.5   , 0.5   , 0.1584, 0.9983,  0.0624, 0.0871]]), 0, 0, 
   array([[0.5   , 0.5   , 0.5   , 0.5   , 0.5   , 0.1584, 0.9983, 0.0624,  0.0871, 0.148 ]]), False)
   
, (array([[0.5   , 0.5   , 0.5   , 0.5   , 0.5   , 0.1584, 0.9983, 0.0624,  0.0871, 0.148 ]]), 2, 0,
   array([[0.5   , 0.5   , 0.5   , 0.5   , 0.1584, 0.9983, 0.0624, 0.0871,  0.148 , 0.9913]]), False)
   
, (array([[0.5   , 0.5   , 0.5   , 0.5   , 0.1584, 0.9983, 0.0624, 0.0871, 0.148 , 0.9913]]), 1, 0, 
   array([[0.5   , 0.5   , 0.5   , 0.1584, 0.9983, 0.0624, 0.0871, 0.148 , 0.9913, 1.    ]]), False)
   
, (array([[0.5   , 0.5   , 0.5   , 0.1584, 0.9983, 0.0624, 0.0871, 0.148 ,
        0.9913, 1.    ]]), 1, 0, array([[0.5   , 0.5   , 0.1584, 0.9983, 0.0624, 0.0871, 0.148 , 0.9913,
        1.    , 0.0998]]), False), (array([[0.5   , 0.5   , 0.1584, 0.9983, 0.0624, 0.0871, 0.148 , 0.9913,
        1.    , 0.0998]]), 0, 0, array([[0.5   , 0.1584, 0.9983, 0.0624, 0.0871, 0.148 , 0.9913, 1.    ,
        0.0998, 0.9999]]), False), (array([[0.5   , 0.1584, 0.9983, 0.0624, 0.0871, 0.148 , 0.9913, 1.    ,
        0.0998, 0.9999]]), 1, 0, array([[0.1584, 0.9983, 0.0624, 0.0871, 0.148 , 0.9913, 1.    , 0.0998,
        0.9999, 0.8557]]), False), (array([[0.1584, 0.9983, 0.0624, 0.0871, 0.148 , 0.9913, 1.    , 0.0998,
        0.9999, 0.8557]]), 1, 0, array([[0.9983, 0.0624, 0.0871, 0.148 , 0.9913, 1.    , 0.0998, 0.9999,
        0.8557, 0.    ]]), False), (array([[0.9983, 0.0624, 0.0871, 0.148 , 0.9913, 1.    , 0.0998, 0.9999,
        0.8557, 0.    ]]), 0, 0, array([[0.0624, 0.0871, 0.148 , 0.9913, 1.    , 0.0998, 0.9999, 0.8557,
        0.    , 0.1598]]), False), (array([[0.0624, 0.0871, 0.148 , 0.9913, 1.    , 0.0998, 0.9999, 0.8557,
        0.    , 0.1598]]), 0, 0, array([[0.0871, 0.148 , 0.9913, 1.    , 0.0998, 0.9999, 0.8557, 0.    ,
        0.1598, 0.9565]]), False), (array([[0.0871, 0.148 , 0.9913, 1.    , 0.0998, 0.9999, 0.8557, 0.    ,
        0.1598, 0.9565]]), 0, 0, array([[0.148 , 0.9913, 1.    , 0.0998, 0.9999, 0.8557, 0.    , 0.1598,
        0.9565, 0.9994]]), False), (array([[0.148 , 0.9913, 1.    , 0.0998, 0.9999, 0.8557, 0.    , 0.1598,
        0.9565, 0.9994]]), 1, 0, array([[0.9913, 1.    , 0.0998, 0.9999, 0.8557, 0.    , 0.1598, 0.9565,
        0.9994, 0.5842]]), False), (array([[0.9913, 1.    , 0.0998, 0.9999, 0.8557, 0.    , 0.1598, 0.9565,
        0.9994, 0.5842]]), 0, 0, array([[1.    , 0.0998, 0.9999, 0.8557, 0.    , 0.1598, 0.9565, 0.9994,
        0.5842, 0.9957]]), False)]
'''