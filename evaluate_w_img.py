from agent.agent_fusion import Agent
from functions import *
import sys
import time


def run_eval(stock_name, win, ep_load, datamode, batch_size=None):
	print("variable ep, and batch_size is not used")
	agent = Agent(
		win,
		is_eval=True,
		stock_name=stock_name,
		model_name=ep_load # ep num이 들어간다
	)

	state_func = getStateHDF if datamode == "hdf" else getStateV2
	num_df, img_fs = getStockData(stock_name, win, datamode)
	l = len(img_fs) - 1

	num_state, img_state, price = state_func(
		num_df, img_fs, 0, buy_after=win + 1
	)
	total_profit = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(num_state, img_state)

		# sit
		nxt_num_state, nxt_img_state, nxt_price = state_func(
			num_df, img_fs, t + 1, buy_after=win + 1,
		)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(price)
			print ("Buy: " + formatPrice(price))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(price - bought_price, 0)
			total_profit += price - bought_price
			print ("Sell: " + formatPrice(price) + " | Profit: " + formatPrice(price - bought_price))
		print(action)
		done = True if t == l - 1 else False
		#agent.memory.append((state, action, reward, next_state, done))
		num_state, img_state, price = nxt_num_state, nxt_img_state, nxt_price

		if done:
			print ("--------------------------------")
			print (stock_name + " Total Profit: " + formatPrice(total_profit))
			print ("--------------------------------")


if __name__ == "__main__":
	print("Usage: python train.py [stock] [window] [episodes] [datamode]")
	stock, win, ep_load, datamode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
	batch_size = int(sys.argv[5])

	print(stock, win, ep_load, datamode, batch_size)
	run_eval(stock, win, ep_load, datamode, batch_size)

"A233740_TRAIN_180_10 180 500 hdf 300" # 어허~ 이놈시끼가