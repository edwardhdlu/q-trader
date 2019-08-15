import numpy as np
import math


# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def get_state(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])

def get_reward(profit):
	#todo follw DeepMind suggestion to clip the reward between [-1,+1](normalize) to improve the stability over other data
	reward       = max(profit, 0)
	return reward


