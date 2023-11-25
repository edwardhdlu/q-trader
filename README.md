# Run cmd 
- run numeral data only (Agent using agent.agent2)
  - `python train.py A069500_TRAIN {win} {ep}`
  - `python train.py A122630_TRAIN {win} {ep}`
  - `python train.py A229200_TRAIN {win} {ep}`
  - `python train.py A233740_TRAIN {win} {ep}`
- run numeral+image data (Agent using agent.agent_fusion)
  - 이미지 사이즈가 고정되어 있어서, 현재 win은 180 가능(매 1분식 과거 3시간 이미지 관찰)
  - 60, 10 추가 예정 (데이터 생성 진행중)
    - `python train_w_img.py A069500_TRAIN {win} {ep}`
    - `python train_w_img.py A122630_TRAIN {win} {ep}`
    - `python train_w_img.py A229200_TRAIN {win} {ep}`
    - `python train_w_img.py A233740_TRAIN {win} {ep}`
- tensor_file_generate
  - `cd ~/dq-stock-trader; nohup python generate_tensor.py A069500_TRAIN {win} {ep}`
  - `cd ~/dq-stock-trader; nohup python generate_tensor.py A122630_TRAIN {win} {ep}`
  - `cd ~/dq-stock-trader; nohup python generate_tensor.py A229200_TRAIN {win} {ep}`
  - `cd ~/dq-stock-trader; nohup python generate_tensor.py A233740_TRAIN {win} {ep}`

# Q-Trader

An implementation of Q-learning applied to (short-term) stock trading. The model uses n-day windows of closing prices to determine if the best action to take at a given time is to buy, sell or sit.

As a result of the short-term state representation, the model is not very good at making decisions over long-term trends, but is quite good at predicting peaks and troughs.

## Results

Some examples of results on test sets:

![^GSPC 2015](https://github.com/edwardhdlu/q-trader/blob/master/images/^GSPC_2015.png)
S&P 500, 2015. Profit of $431.04.

![BABA_2015](https://github.com/edwardhdlu/q-trader/blob/master/images/BABA_2015.png)
Alibaba Group Holding Ltd, 2015. Loss of $351.59.

![AAPL 2016](https://github.com/edwardhdlu/q-trader/blob/master/images/AAPL_2016.png)
Apple, Inc, 2016. Profit of $162.73.

![GOOG_8_2017](https://github.com/edwardhdlu/q-trader/blob/master/images/GOOG_8_2017.png)
Google, Inc, August 2017. Profit of $19.37.

## Running the Code

To train the model, download a training and test csv files from [Yahoo! Finance](https://ca.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC) into `data/`
```
mkdir model
python train ^GSPC 10 1000
```

Then when training finishes (minimum 200 episodes for results):
```
python evaluate.py ^GSPC_2011 model_ep1000
```

## References

[Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/) - Q-learning overview and Agent skeleton code