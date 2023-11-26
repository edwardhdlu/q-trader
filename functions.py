import os
import numpy as np
import pandas as pd
import math
import h5py
import torch
from torchvision import transforms, io
from PIL import Image
from datetime import datetime

from functools import lru_cache

DataPath = os.environ.get("DataPath", "/content/drive/MyDrive/dfmba_img_rl_trading")
if not os.path.exists(DataPath):
	DataPath = "/locdisk/data/hoseung2"

print(f"DataPath: {DataPath}")


def get_data_path(stock_name, ext_="h5"):
	if "locdisk" in DataPath:
		return f"{DataPath}/{stock_name}.{ext_}"
	elif "MyDrive" in DataPath:
		return f"{DataPath}/data/{stock_name}.{ext_}"
	else:
		raise ValueError("DataPath not found")


# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]: # 2515
		vec.append(float(line.split(",")[4])) # Close price

	return vec

# returns the sigmoid
def sigmoid(x):
	try:
		return 1 / (1 + math.exp(-x))
	except TypeError:
		return 1 / (1 + np.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])




###################### V2 ######################
def list_files_in_directory(directory):
	file_list = []
	for root, dirs, files in os.walk(directory):
		for file in files:
			file_list.append(os.path.join(root, file))
	return file_list



# instead of getStockDataVec
@lru_cache(maxsize=None)
def getStockData(stock_name, window, datamode=None):
	assert window in [180], "window size must be in [180]"
	if datamode == "hdf":
		print("in datamode hdf, window is fixed to 180")
		filename = get_data_path(stock_name, ext_="h5")
		file = h5py.File(filename, 'r')
		return file, file.keys()
	else:
		df = pd.read_csv(
			get_data_path(stock_name, ext_="csv")
		)
		df.set_index("date", inplace=True)
		df.index = pd.to_datetime(df.index)

		if "train" in stock_name.lower():
			stock_name = stock_name.split("_")[0]
			img_files = list_files_in_directory(
				f"data/{window}/{stock_name}/train"
			)
		elif "test" in stock_name.lower():
			stock_name = stock_name.split("_")[0]
			img_files = list_files_in_directory(
				f"data/{window}/{stock_name}/test"
			)
		img_files = sorted(img_files) # 시간순으로 정렬

		return df[
			['Close', 'Open', 'Low', 'High', 'Volume', 'AdjClose']
		], img_files


# instead of getState
def getStateV2(num_df, f_list, t, buy_after, *args, **kwargs):
	def _load_num_state(start, end):
		# 이미지에 들어가는 값과 동일하게 하기 위해 인덱싱 사용
		# block = num_df[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # pad with t0
		# end 까지 가져오기 때문에 +1 안해도 됨, pad t0은 이미지와 맞추어야 할 것 같은데 어떻게 구현하면 좋을지...
		block = num_df.loc[start: end]
		state = sigmoid((block - block.shift(1))).dropna(how="all") # nan이 있으면 lstm 학습이 적용되지 않음
		return np.array([state.values])

	def _load_img_state(f):
		image = Image.open(f)
		if image.mode != 'RGB': # 이미지를 제가 4채널로 생성해서^^;;
			image = image.convert('RGB')
		transform = transforms.ToTensor()
		image = transform(image) / 255
		return image

	f_name = f_list[t]
	img_state = _load_img_state(f_name)

	conv_lamb = lambda x: datetime.strptime(x, "%Y%m%d%H%M%S")
	start, end = f_name.split("/")[-1].replace(".png", "").split("_")[1:]
	start, end = conv_lamb(start), conv_lamb(end)
	num_state = _load_num_state(start, end)

	try:
		price = num_df.loc[
			num_df.index[num_df.index.tolist().index(end) + buy_after],
			"Close"
		] # 주문이 들어가는 시점의 price (임의로 정보를 받고 10분 뒤로 설정
	except IndexError:
		price = num_df.loc[
			num_df.index[-1],
			"Close"
		]
	return num_state, img_state, price


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getStateHDF(file_hdf, f_list, t=None, *args, **kwargs):
	num_state = torch.tensor(
		file_hdf[f"{t}/num"][:], dtype=torch.float
	)
	img_state  = torch.tensor(
		file_hdf[f"{t}/img"][:], dtype=torch.float
	)
	price = torch.tensor(
		file_hdf[f"{t}/p"][()], dtype=torch.float
	)

	if DEVICE != "cpu":
		num_state = num_state.to(DEVICE)
		img_state = img_state.to(DEVICE)
		price = price.to(DEVICE)
	return num_state, img_state, price