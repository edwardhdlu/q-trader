import time
from datetime import datetime

from ai_agent import Agent
from utils import *
import numpy as np

class Dqn:
    def __init__(self):
        self.open_orders =[]
        self.model_name = ''


    def learn(self, data, episodes,  window_size, batch_size, use_existing_model, probability_of_random_action):
        agent              = Agent(window_size, use_existing_model, '', probability_of_random_action)
        l                  = len(data) - 1
        rewards_vs_episode = []
        profit_vs_episode  = []
        trades_vs_episode  = []
        epsilon_vs_episode = []
        for episode in range(episodes + 1):
            #print("Episode " + str(e) + "/" + str(episode_count))
            state            = self.get_state(data, 0, window_size + 1)
            total_profits    = 0
            total_trades     = 0
            total_rewards    = 0
            self.open_orders = []

            for t in range(l):

                action = agent.choose_best_action(state)#tradeoff bw predict and random
                #print(f'state={state}')
                reward, total_rewards, total_profits, total_trades = self.execute_action (action, data[t], t, total_rewards, total_profits, total_trades)

                done = True if t == l - 1 else False
                next_state = self.get_state(data, t + 1, window_size + 1)
                #print(f'next_state={next_state}')
                agent.remember(state, action, reward, next_state, done)#store contents of memory in buffer for future learning
                state = next_state

                if done:
                    print(f'Episode {episode}/{episodes} Total Profit: {formatPrice(total_profits)} , Total trades: {total_trades}, probability of random action: {np.round(agent.epsilon,3)}')
                    print("---------------------------------------")
                    rewards_vs_episode.append(total_rewards)
                    profit_vs_episode.append(np.round(total_profits,4))
                    trades_vs_episode.append(total_trades)
                    epsilon_vs_episode.append(agent.epsilon)

                if len(agent.memory) > batch_size:#if memory of agent gets full:
                    agent.experience_replay(batch_size)#fit
                #clean memory ?
            if episode % 30 == 0:
                model_name = "files/output/model_ep" + str(episode)
                agent.model.save(model_name)
                print(f'{model_name} saved')


        model_name = "files/output/model_ep" + str(episodes)
        agent.model.save(model_name)
        print(f'{model_name} saved')
        return rewards_vs_episode, profit_vs_episode, trades_vs_episode, epsilon_vs_episode, model_name


    def execute_action(self, action, close_price, t, total_rewards, total_profits, total_trades):

        if action == 1:  # buy
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
            #print(f'row #{t} exit @ ' + formatPrice(close_price) + " | return_rate: " + formatPrice(return_rate))

        else:  # hold
            reward = 0
            #print(f'row #{t} Hold')
        total_rewards += reward
        return reward, total_rewards, total_profits, total_trades


    # returns an an n-day state representation ending at time t of difference bw close prices. ex. [0.5,0.5,0.5,0.4,0.3,0.2,0.5,0.4,0.3,0.2]
    def get_state(self, data, t, n):
        d = t - n + 1
        block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
        res = []
        for i in range(n - 1):
            #res.append(sigmoid(block[i + 1] - block[i]))
            res.append(np.log(block[i + 1] / block[i]))
            #res.append(np.log(block[i + 1])-np.log(block[i]))
        #add features
        #add cyclic feature(sin, cos)
        #add tech. indicators
        #add screen image
        #add economic data
        #https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114
        return np.array([res])

