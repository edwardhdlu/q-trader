from typing import Tuple
from ai_agent import Agent
from utils import *
import numpy as np


class Dqn:
    def __init__(self):
        self.open_orders = []
        self.model_name = ''

    def learn(self, data, episodes, num_features, batch_size, use_existing_model, random_action_min=0.1,
              random_action_decay=0.99995, num_neurons=64, future_reward_importance=0.95):

        agent = Agent(num_features, use_existing_model, '', random_action_min, random_action_decay, num_neurons,
                      future_reward_importance)
        l = len(data) - 1
        rewards_vs_episode = []
        profit_vs_episode = []
        trades_vs_episode = []
        epsilon_vs_episode = []
        for episode in range(1, episodes + 1):
            state = self.get_state(data, num_features, num_features)
            total_profits = 0
            total_holds = 0
            total_buys = 1
            total_sells = 0
            total_notvalid = 0  # add-on buys or sells without previous buy
            # total_rewards    = 0
            self.open_orders = [data[0]]

            for t in range(num_features, l):

                action = agent.choose_best_action(state)  # tradeoff bw predict and random
                # print(f'state={state}')
                reward, total_profits, total_holds, total_buys, total_sells, total_notvalid = \
                    self.execute_action(action, data[t], t, total_profits, total_holds, total_buys, total_sells,
                                        total_notvalid)

                done = True if t == l - 1 else False

                next_state = self.get_state(data, t + 1, num_features)

                #if len(self.open_orders) > 0:  # if long add next state return as reward
                #print(action, agent.actions[action])
                if agent.actions[action] == 'buy':
                    immediate_reward = next_state[0][-1]
                elif agent.actions[action] == 'sell':
                    immediate_reward = -next_state[0][-1]
                else:
                    immediate_reward = 0
                #print("Immediate reward:{0:.5f} Reward:{1:.5f} Time:{2} Price:{3} Action:{4}".
                #      format(immediate_reward, reward, t, data[t], agent.actions[action]))
                #reward = reward + immediate_reward
                reward = immediate_reward


                #print(f'row #{t} {agent.actions[action]} @{data[t]}, state1={state}, state2={next_state}, reward={reward}')

                agent.remember(state, action, reward, next_state,
                               done)  # store contents of memory in buffer for future learning
                state = next_state

                if done:
                    # sell position at end of episode
                    reward, total_profits, total_holds, total_buys, total_sells, total_notvalid = \
                        self.execute_action(2, data[t+1], t+1, total_profits, total_holds, total_buys, total_sells,
                                            total_notvalid)
                    eps = np.round(agent.epsilon, 3)
                    print(f'Episode {episode}/{episodes} Total Profit: {formatPrice(total_profits*100)},'
                          f' Total hold/buy/sell/notvalid trades: {total_holds} / {total_buys} / {total_sells} / {total_notvalid},'
                          f' probability of random action: {eps}')
                    print("---------------------------------------")
                    # rewards_vs_episode.append(total_rewards)
                    profit_vs_episode.append(np.round(total_profits, 4))
                    trades_vs_episode.append(total_buys)
                    epsilon_vs_episode.append(eps)

                if len(agent.memory) >= batch_size:     # if enough recorded memory available
                   agent.experience_replay(batch_size)  # fit
                # clean memory ?

        model_name = "files/output/model_ep" + str(episodes)
        agent.model.save(model_name)
        print(f'{model_name} saved')
        return profit_vs_episode, trades_vs_episode, epsilon_vs_episode, model_name, agent.num_trains, agent.epsilon

    def execute_action(self, action, close_price, t, total_profits, total_holds, total_buys, total_sells,
                       total_notvalid) -> Tuple[float, float, int, int, int, int]:

        if action == 0:  # hold
            reward = 0
            total_holds += 1
            # print(f'row #{t} Hold')

        elif action == 1 and len(self.open_orders) == 0:  # buy only if not long already
            self.open_orders.append(close_price)
            total_buys += 1
            reward = 0
            #print(f'row #{t} buy  @ ' + formatPrice(close_price))

        elif action == 2 and len(self.open_orders) > 0:  # sell (or exiting trade)

            bought_price = self.open_orders.pop(0)
            # https://towardsdatascience.com/using-reinforcement-learning-to-trade-bitcoin-for-massive-profit-b69d0e8f583b
            return_rate = close_price / bought_price
            log_return = np.log(return_rate)  # return % in continuous compounded format
            total_profits += log_return
            reward = log_return  # get_reward(return_rate, total_profits)
            total_sells += 1
            #print(f'row #{t} sell @ ' + formatPrice(close_price) + " | return_rate: " + formatPrice(reward))
        else:
            reward = 0
            total_notvalid += 1
            #print('Action:{} Row:{} Price:{:.2f} repeat buy or sell with no previous buy'.format(action, t, close_price))

        # total_rewards += reward
        return reward, total_profits, total_holds, total_buys, total_sells, total_notvalid

    # returns a n-day state representation ending at time t of difference bw close prices. ex. [0.5,0.4,0.3,0.2]
    def get_state(self, data, to_ix, num_features):
        from_ix = to_ix - num_features
        data_block = data[from_ix:to_ix + 1]
        res = []
        for i in range(num_features):
            # res.append(sigmoid(block[i + 1] - block[i]))
            res.append(np.log(data_block[i + 1] / data_block[i]))
        # add features
        # add cyclic feature(sin, cos)
        # add tech. indicators
        # add screen image
        # add economic data
        # https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114
        return np.array([res])
