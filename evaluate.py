import matplotlib.pyplot as plt
from agent.agent import Agent
from functions import getStockDataVec, genStateNormalize, formatPrice

WINDOW_SIZE = 7 # one week
STOCK = "BABA_2014"

agent = Agent(WINDOW_SIZE, True, "model_ep670") # 150 360
data = getStockDataVec(STOCK)
l = len(data) - 1
batch_size = 32

state = genStateNormalize(data, 0, WINDOW_SIZE) # normalized extended
total_profit = 0
agent.inventory = []

# plot data
stock_data = []
buy_x = []
buy_y = []
sell_x = []
sell_y = []

for t in xrange(l):
	action = agent.act(state)
	stock_data.append((t, data[t]))

	# sit
	next_state = genStateNormalize(data, t + 1, WINDOW_SIZE)
	reward = 0

	if action == 1: # buy
		buy_x.append(t)
		buy_y.append(data[t])

		agent.inventory.append(data[t])

	elif action == 2 and len(agent.inventory) > 0: # sell
		sell_x.append(t)
		sell_y.append(data[t])

		bought_price = agent.inventory.pop(0)
		reward = max(data[t] - bought_price, 0)
		total_profit += data[t] - bought_price

	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	if done:
		print STOCK + " Profit: " + formatPrice(total_profit)

plt.plot(data, label=STOCK)
plt.scatter(buy_x, buy_y, color="green", label="Buy")
plt.scatter(sell_x, sell_y, color="red", label="Sell")

plt.legend()
plt.show()
