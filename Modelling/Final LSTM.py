# Import libraries
from cgitb import reset
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from math import sqrt
# Read in the demand data
df = pd.read_csv("totaldemand_sa.csv")
# convert datetime
df['DATETIME'] = pd.to_datetime(df.DATETIME)
# keep only demand data
df_new = df[['TOTALDEMAND']]
# split into train and test
train_df = df_new[:-48]
test_df = df_new[-48:]
# Scale data to 0-1 and create training data generator
scaler = MinMaxScaler()
df_new = pd.DataFrame(scaler.fit_transform(train_df), columns = ['TOTALDEMAND'])
X_test = pd.DataFrame(scaler.transform(test_df), columns = ['TOTALDEMAND'])
X_test = list(X_test['TOTALDEMAND'])
X_test = np.array(X_test)
# convert training data to windowed data sets
ylist = list(df_new['TOTALDEMAND'])
n_future = 48
n_past = 7*48
total_period = 8*48
idx_end = len(ylist)
idx_start = idx_end - total_period
X_new = []
y_new = []
while idx_start > 0:
  x_line = ylist[idx_start:idx_start+n_past]
  y_line = ylist[idx_start+n_past:idx_start+total_period]

  X_new.append(x_line)
  y_new.append(y_line)

  idx_start = idx_start - 1
X_train = np.array(X_new)
y_train = np.array(y_new)
# reshape data into the right format for RNNs
n_samples = X_train.shape[0]
n_timesteps = X_train.shape[1]
n_steps = y_train.shape[1]
n_features = 1
X_train_rs = X_train.reshape(n_samples, n_timesteps, n_features )
X_test_rs = X_test.reshape(X_test.shape[0], 1, n_features )
# create and compile a vanila lstm model
batch_size = 96
simple_model = Sequential([
   LSTM(32, activation='tanh',input_shape=(n_timesteps, n_features)),
  Dense(y_train.shape[1]),
])
simple_model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
  loss= tf.keras.metrics.mean_squared_error,
  metrics=['mean_squared_error'],
)
# train the simple model
simple_model.fit(X_train_rs, y_train,
          validation_split=0.2,
          epochs=10,
          batch_size=batch_size,
          shuffle = True
)
# Make predictions
preds = scaler.inverse_transform(simple_model.predict(X_train_rs[-1].reshape(1,336,1)))
# save results to a dataframe
test_df['new'] =  preds.reshape(48,-1)
# save results to csv
test_df.to_csv('predictions.csv', index=False)  
# calculate rmse
rmse = sqrt(mean_squared_error(test_df.TOTALDEMAND, test_df.new))
