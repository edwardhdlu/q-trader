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

### 3 variants for representing an n-day state:

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

# c) returns an np array containing changes in bin position
def genStatePartition(data, t, n, bins=10):
	d = t - n + 1
	mx = max(data)
	mn = min(data)

	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	norm = map(lambda x: 1.0 * (x - mn) / (mx - mn), block)

	res = []
	for i in xrange(n - 1):
		b1 = math.floor(norm[i] * bins)
		b2 = math.floor(norm[i + 1] * bins)
		res.append(b2 - b1)

	return np.array([res])