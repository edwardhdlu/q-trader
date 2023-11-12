import sys
import h5py
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from functions import *


num_df, img_fs = None, None

def data_to_tensor(stock_code, window_size, buy_after, root_path):
    file_path = f'{stock_code}_{window_size}_{buy_after}.h5'
    h5_file = h5py.File(root_path+file_path, 'w')

    global num_df
    global img_fs
    num_df, img_fs = getStockData(stock_code, int(window_size))
    with ThreadPoolExecutor(max_workers=5) as executor:
        for i in tqdm(range(len(img_fs))):
            executor.submit(create_dataset, i, h5_file)
    h5_file.close()


def create_dataset(i, h5_file):
    global buy_after
    num, img, p = getStateV2(num_df, img_fs, i, int(buy_after))

    h5_file.create_dataset(str(i) + '/num', data=num)
    h5_file.create_dataset(str(i) + '/img', data=img)
    h5_file.create_dataset(str(i) + '/p', data=p)


def run(stock_code, window_size, buy_after, root_path):
    data_to_tensor(stock_code, window_size, buy_after, root_path)


# 데이터 로드에 속도가 많이 걸리므로 h5 생성 후 로드하는 방식으로 변경 하려면 이 코드로 텐서 이미지를 사용
if __name__ == '__main__':
    stock_code = sys.argv[1]
    window_size = sys.argv[2]
    try:
        buy_after = sys.argv[3] # window 사이즈를 보고 몇분뒤에 매매 할것인지 제약
    except:
        buy_after = 10 # 없음 그냥 10분 뒤에 살게용~
        print("Automatically set buy_after to 10 minutes")
    print(stock_code, window_size, buy_after)
    root_path = "/locdisk/data/hoseung2/"
    run(stock_code, window_size, buy_after, root_path=root_path)

"""A069500_TRAIN A122630_TRAIN A229200_TRAIN A233740_TRAIN
cd ~/dq-stock-trader; nohup python generate_tensor.py A069500_TRAIN 180 10
cd ~/dq-stock-trader; nohup python generate_tensor.py A122630_TRAIN 180 10
cd ~/dq-stock-trader; nohup python generate_tensor.py A229200_TRAIN 180 10
cd ~/dq-stock-trader; nohup python generate_tensor.py A233740_TRAIN 180 10
"""