import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

data = pd.read_csv("bitcoin_ticker.csv")

data.head()
data['rpt_key'].value_counts()
df = data.loc[(data['rpt_key'] == 'btc_usd')]
df.head()

df = df.reset_index(drop=True)
df['datetime'] = pd.to_datetime(df['datetime_id'])
df = df.loc[df['datetime'] > pd.to_datetime('2017-06-28 00:00:00')]
df = df[['datetime', 'last', 'diff_24h', 'diff_per_24h', 'bid', 'ask', 'low', 'high', 'volume']]
df.head()

df = df[['last']]
dataset = df.values
dataset = dataset.astype('float32')
dataset