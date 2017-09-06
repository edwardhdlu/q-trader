import numpy as np
import math

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))

	return vec

### 2 variants for representing an n-day state:

# a) returns a normalized np array containing n stock values from time t - n + 1 to t
def genStateNormalize(data, t, n):
	d = t - n + 1
	res = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	m = max(res)

	return np.array([map(lambda x: 1.0 * x / m, res)])

# b) returns an np array containing the discrete pattern of ups(1) and downs(0)
def genStateDiscretize(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in xrange(n - 1):
		res.append(1 if block[i] < block[i + 1] else 0)

	return np.array([res])
