import time
from datetime import datetime

from ai_agent import Agent
from utils import *
import numpy as np

class Dqn:
    def __init__(self):
        self.open_orders =[]
        self.model_name = ''


    def learn(self, data, episodes, num_features, batch_size, use_existing_model, random_action_min=0.1, random_action_decay=0.99995, num_neurons=64):
        agent              = Agent(num_features, use_existing_model, '', random_action_min, random_action_decay, num_neurons)
        l                  = len(data) - 1
        rewards_vs_episode = []
        profit_vs_episode  = []
        trades_vs_episode  = []
        epsilon_vs_episode = []
        for episode in range(1,episodes + 1):
            #print("Episode " + str(e) + "/" + str(episode_count))
            state            = self.get_state(data, num_features, num_features)
            total_profits    = 0
            total_trades     = 1
            #total_rewards    = 0
            self.open_orders = [data[0]]

            for t in range(num_features,l):

                action = agent.choose_best_action(state)#tradeoff bw predict and random
                #print(f'state={state}')
                reward, total_profits, total_trades = self.execute_action (action, data[t], t, total_profits, total_trades)

                done = True if t == l - 1 else False
                next_state = self.get_state(data, t + 1, num_features)
                print(f'row #{t} {agent.actions[action]} @{data[t]}, state1={state}, state2={next_state}, reward={reward}')
                agent.remember(state, action, reward, next_state, done)#store contents of memory in buffer for future learning
                state = next_state

                if done:
                    eps = np.round(agent.epsilon,3)
                    print(f'Episode {episode}/{episodes} Total Profit: {formatPrice(total_profits)} , Total trades: {total_trades}, probability of random action: {eps}')
                    print("---------------------------------------")
                    #rewards_vs_episode.append(total_rewards)
                    profit_vs_episode.append(np.round(total_profits,4))
                    trades_vs_episode.append(total_trades)
                    epsilon_vs_episode.append(eps)

                if len(agent.memory) > batch_size:#if memory of agent gets full:
                    agent.experience_replay(batch_size)#fit
                #clean memory ?
            # if episode % 1000 == 0:
            #     model_name = "files/output/model_ep" + str(episode)
            #     agent.model.save(model_name)
            #     print(f'{model_name} saved')


        model_name = "files/output/model_ep" + str(episodes)
        agent.model.save(model_name)
        print(f'{model_name} saved')
        return  profit_vs_episode, trades_vs_episode, epsilon_vs_episode, model_name, agent.num_trains, agent.epsilon


    def execute_action(self, action, close_price, t, total_profits, total_trades):

        if action == 0:  # hold
            reward = 0
            #print(f'row #{t} Hold')

        elif action == 1:  # buy
            self.open_orders.append(close_price)
            total_trades += 1
            reward = 0
            #print(f'row #{t} Buy  @ ' + formatPrice(close_price))

        elif action == 2 and len(self.open_orders) > 0:  # sell (or exiting trade)

            bought_price = self.open_orders.pop(0)
            return_rate = close_price / bought_price
            log_return = np.log(return_rate)#for normal distribution
            total_profits += log_return
            reward = log_return#get_reward(return_rate, total_profits)
            #print(f'row #{t} exit @ ' + formatPrice(close_price) + " | return_rate: " + formatPrice(reward))
        else:
            reward = 0
            #print('no open orders')

        #total_rewards += reward
        return reward, total_profits, total_trades


    # returns an an n-day state representation ending at time t of difference bw close prices. ex. [0.5,0.5,0.5,0.4,0.3,0.2,0.5,0.4,0.3,0.2]
    def get_state(self, data, to_ix, num_features):
        from_ix = to_ix - num_features
        if from_ix >= 0:
            data_block = data[from_ix:to_ix + 1]
        else:
            data_block = -from_ix * [data[0]] + data[0:to_ix + 1] # pad with t0
        res = []
        for i in range(num_features):
            #res.append(sigmoid(block[i + 1] - block[i]))
            res.append(np.log(data_block[i + 1] / data_block[i]))
            #res.append(np.log(block[i + 1])-np.log(block[i]))
        #add features
        #add cyclic feature(sin, cos)
        #add tech. indicators
        #add screen image
        #add economic data
        #https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114
        return np.array([res])

