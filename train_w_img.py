from agent.agent_fusion import Agent
from functions import *
import sys

def run(stock_code, win, ep):
	agent = Agent(win)

	num_df, img_fs = getStockData(stock_code, win)
	l = len(img_fs) - 1

	batch_size = 32
	print(f"Version  : 0.2.0")
	print(f"Description  : dqn torch implementation ")
	print(f"Number of Data  : {l}")

	for e in range(ep + 1):
		print("Episode " + str(e) + "/" + str(ep))
		num_state, img_state, price = getStateV2(
			num_df, img_fs, 0, win + 1
		)

		total_profit = 0
		agent.inventory = []
		# agent.memory.clear()

		for t in range(l):  # [0,2514/2)
			action = agent.act(num_state, img_state)

			# sit
			nxt_num_state, nxt_img_state, nxt_price = getStateV2(
				num_df, img_fs, t + 1, win + 1
			)
			reward = 0

			if action == 1:  # buy
				agent.inventory.append(price)
				print(f"#{t} Buy: " + formatPrice(price))

			elif action == 2 and len(agent.inventory) > 0:  # sell
				bought_price = agent.inventory.pop(0)
				reward = max(price - bought_price, 0)
				total_profit += price - bought_price
				print(f"#{t} Sell: " + formatPrice(price) + " | Profit: " + formatPrice(price - bought_price))

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

			if len(agent.memory) > batch_size:
				agent.expReplay(batch_size)
		if (e + 1) % agent.target_update_period == 0:
			agent.update_target()

		if (e + 1) % 5 == 0:
			agent.save(e, stock_code)


if __name__ == "__main__":
	if len(sys.argv) != 4:
		print("Usage: python train.py [stock] [window] [episodes]")
		exit()

	stock, win, ep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
	print(stock, win, ep)
	root_path = "/locdisk/data/hoseung2/"
	run(stock, win, ep)