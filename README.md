
# Data
from https://finance.yahoo.com/quote/%5EGSPC?p=^GSPC
or from https://www.kaggle.com/janiobachmann/s-p-500-time-series-forecasting-with-prophet/data


# Value based network approach(actor ai_agent.py)
we use value-based RL (  ddpg algo. has also a  Policy netork model ) - the goal of the agent is to optimize the value function V(s) which is defined as a function that tells us the maximum expected future reward the agent shall get at each state.
The value of each state is the total amount of the reward an RL agent can expect to collect over the future, from a particular state.
we use TD (Temporal Difference) method to calculate value (probability of action) base on current and next state: 
TD In plain English, 
 means maximum future reward for this state and action (s,a) 
is the immediate reward r plus maximum future reward for the next state
                 
![max future reward](https://github.com/loliksamuel/py-ML-rl-trade/blob/master/files/output/max_future_reward.png)

the model get updated every few days.


As illustrated in below figure
The agent will use the  above value function to select which state to choose at each step. The agent will always take the state with the biggest value.
model input :
 1. historical stock data 
 2. historicsl market data 
 3. investment status, and reward and value function
 
model output(action prediction):
1. hold
2. buy
3. sell


![nn](https://github.com/loliksamuel/py-ML-rl-trade/blob/master/files/output/nn.png)





# Environment
 market env is essentially a time-series data-frame (RNNs work well with time-series data)
 
 
The policy network outputs an action daily 
the market returns the rewards of such actions (the profit)
and all this data ( status,   amount of money gain or lost), sent to policy network to train
 
![rl](https://github.com/loliksamuel/py-ML-rl-trade/blob/master/files/output/rl.png)

# Action
there are 3 possible actions that the agent can take, hold, buy, sell
there is a zero-market impact hypothesis, which essentially
states that the agent’s action can never be significant enough to affect the market env.
and the ai_agent will assign score for each action and state
  
# State
State : Current situation returned by the environment.

# Features

this Q-learning implementation applied to (short-term) stock trading. 
The model uses t-day windows of close prices as features 
to determine if the best action to take at a given time is to buy, sell or hold.


# Reward

Reward shaping is a technique inspired by animal training where supplemental rewards are provided to make a problem easier to learn. 
There is usually an obvious natural reward for any problem. 
For games, this is usually a win or loss. For financial problems, 
the reward is usually profit. Reward shaping augments the natural reward signal by adding additional rewards for making progress toward a good solution.

# Gamma (discount rate)

learning is based on immediate and long-term reward
To make the model perform well in long-term, 
we need to take into account not only the immediate rewards but also the future rewards we are going to get. 

In order to do this, we are going to have a ‘discount rate’ or ‘gamma’. 
If gamma=0 then agent will only learn to consider current rewards. 
if gamma=1 then agent will make it strive for a long-term high reward.
This way the agent will learn to maximize the discounted future reward based on the given state.

the model is not very good at making decisions in long-term , but is quite good at predicting peaks and troughs.

# Epsilon (exploration rate,)
Epsilon, aka exploration rate, this is the rate in which an agent randomly decides its action rather than prediction.
epsilon_min, used to let the agent  explore a minimum percent of randomness
epsilon_decay- used to decrease the number of explorations as it gets good at trading.




# How to Run

- Install python 3.7
	- Anaconda, Python, IPython, and Jupyter notebooks
	- Installing packages
	- `conda` environments

- Download data
	- training and test csv files from [Yahoo! Finance](https://ca.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC) 
	- put in `files/input/`

- Bring other features if u have

- Train model. for good results run:
	- with minimum 200 episodes 
	- on all data (not just 2011) 
	- with GPU  https://www.paperspace.com 
```
python rl_dqn.py
```

- See 3 plots generated 
	- episode vs profits
	- episode vs trades
	- episode vs epsilon

- Back-test last model created in files/output/ on any stock
```
python backtest.py
```


# Results

Some examples of results on test sets:

![^GSPC 2015](https://github.com/edwardhdlu/q-trader/blob/master/images/^GSPC_2015.png)
S&P 500, 2015. Profit of $431.04.

![BABA_2015](https://github.com/edwardhdlu/q-trader/blob/master/images/BABA_2015.png)
Alibaba Group Holding Ltd, 2015. Loss of $351.59.

![AAPL 2016](https://github.com/edwardhdlu/q-trader/blob/master/images/AAPL_2016.png)
Apple, Inc, 2016. Profit of $162.73.

![GOOG_8_2017](https://github.com/edwardhdlu/q-trader/blob/master/images/GOOG_8_2017.png)
Google, Inc, August 2017. Profit of $19.37.



# References

[Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/) - balance CartPole game with Q-learning

https://quantdare.com/deep-reinforcement-trading/

[paper of Deep Q network by deep mind](https://arxiv.org/pdf/1509.06461.pdf)

[paper Capturing Financial markets to apply Deep RL](https://arxiv.org/pdf/1907.04373.pdf)

[tensor force_bitcoin_trader](https://github.com/lefnire/tforce_btc_trader)

[gym_trading](https://github.com/AdrianP-/gym_trading)

[stable-baselines](https://stable-baselines.readthedocs.io/en/master/guide/quickstart.html)

[https://github.com/notadamking/tensortrade/tree/master/tensortrade](https://github.com/notadamking/tensortrade/tree/master/tensortrade)

[introduction-to-reinforcement-learning](https://medium.com/free-code-camp/a-brief-introduction-to-reinforcement-learning-7799af5840db)

[other python resources](https://github.com/topics/trading?l=python)


