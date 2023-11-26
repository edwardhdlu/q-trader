from agent.agent_fusion import Agent
from functions import *
import sys
import time

def run(stock_code, win, ep, datamode, batch_size, ep_from=0):
	agent = Agent(win, is_eval=False, stock_name=stock_code, model_name=ep_from)
	state_func = getStateHDF if datamode == "hdf" else getStateV2

	num_df, img_fs = getStockData(stock_code, win, datamode)
	l = len(img_fs) - 1

	#batch_size = 32
	print(f"Version  : 0.2.0")
	print(f"Data Mode  : {state_func.__name__}")
	print(f"Description  : dqn torch implementation ")
	print(f"Number of Data  : {l}")

	for e in range(ep_from, ep - ep_from+ 1):
		print("Episode " + str(e) + "/" + str(ep))
		num_state, img_state, price = state_func(
			num_df, img_fs, 0, buy_after=win + 1
		)
		total_profit = 0
		agent.inventory = []
		# agent.memory.clear()
		start1 = time.time()
		for t in range(l):  # [0,2514/2)
			action = agent.act(num_state, img_state)

			# sit
			nxt_num_state, nxt_img_state, nxt_price = state_func(
				num_df, img_fs, t + 1, buy_after=win + 1,
			)
			reward = 0

			if action == 1:  # buy
				agent.inventory.append(price)
				# print(f"#{t} Buy: " + formatPrice(price))

			elif action == 2 and len(agent.inventory) > 0:  # sell
				bought_price = agent.inventory.pop(0)
				reward = max(price - bought_price, 0)
				total_profit += price - bought_price
				# print(f"#{t} Sell: " + formatPrice(price) + " | Profit: " + formatPrice(price - bought_price))

			if t % 5000 == 0:
				end5000 = time.time()
				print(f"#{t} Profit: " + formatPrice(total_profit) + " | Time: " + str(end5000 - start1) + "s")
				start1 = time.time()

			done = 1 if t == l - 1 else 0  # True, False
			agent.memory.append(
				((num_state, img_state), action, reward, (nxt_num_state, nxt_img_state), done)
			)
			num_state, img_state, price = nxt_num_state, nxt_img_state, nxt_price

			if done:
				print("--------------------------------")
				print("Total Profit: " + formatPrice(total_profit))
				print("--------------------------------")
				break

			if len(agent.memory) % (batch_size // 2) == 0:
				agent.expReplay(batch_size)
				# agent.memory.clear() # hmm...
		if (e + 1) % agent.target_update_period == 0:
			agent.update_target()

		if (e + 1) % 5 == 0:
			agent.save(e, stock_code)


if __name__ == "__main__":
	# if len(sys.argv) != 5:
	# 	print("Usage: python train.py [stock] [window] [episodes] [datamode]")
	# 	exit()

	stock, win, ep, datamode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
	batch_size = int(sys.argv[5])
	ep_load = int(sys.argv[6])
	print(stock, win, ep, datamode, batch_size, ep_load)
	run(stock, win, ep, datamode, batch_size, ep_load)