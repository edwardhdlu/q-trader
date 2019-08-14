## References

[Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/) - Q-learning overview and Agent skeleton code

https://quantdare.com/deep-reinforcement-trading/


# Q-Trader
![rl](http://manul.io/img/gekkos/arch.png)


An implementation of Q-learning applied to (short-term) stock trading. 
The model uses n-day windows of closing prices to determine if the best action to take at a given time is to buy, sell or hold.

As a result of the short-term state representation, 
the model is not very good at making decisions over long-term trends, but is quite good at predicting peaks and troughs.

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

1. download a training and test csv files from [Yahoo! Finance](https://ca.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC) into `files/input/`

2. train model with minimum 200 episodes for good results:
```
python train
```

3. backtest model created in files/output/ on any stock
```
python backtest.py
```

