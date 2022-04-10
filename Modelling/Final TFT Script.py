# Load the required libraries
import pandas as pd
from darts import TimeSeries
from darts.metrics import rmse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import torch
## Darts imports
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel
from darts.metrics import mape
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from torch.nn import MSELoss
import warnings
warnings.filterwarnings("ignore")
import logging
# Read in the Demand and Temperature data
## Read in demand data
df = pd.read_csv("totaldemand_qld.csv")
# convert datetime 
df['DATETIME'] = pd.to_datetime(df.DATETIME)
new_df = df.drop(columns='REGIONID') 
## Read in temperature data
temp_df = pd.read_csv("temprature_qld.csv")
temp_df = temp_df[['DATETIME','TEMPERATURE']]
# convert datetime
temp_df['DATETIME'] = pd.to_datetime(temp_df.DATETIME)
# Merge the two dataframes
dem_temp_df = new_df.merge(temp_df,how='left',on='DATETIME')
## There are some reading gaps in temp reading, fill in the missing values
dem_temp_df.sort_values(by='DATETIME',inplace=True)
# Create Darts Timeseries
series = TimeSeries.from_dataframe(dem_temp_df,'DATETIME',['TOTALDEMAND','TEMPERATURE'])
# covert all numbers to float32 for memory efficiency
series = series.astype(np.float32)
# define training cut-off
training_cutoff = pd.Timestamp("20210317")
# make darts timeseries
series_dem = series['TOTALDEMAND']
# create train and validation series
train_dem,val_dem = series_dem.split_after(training_cutoff)
# Scale demand data
dem_scaler = Scaler()
scaled_train = dem_scaler.fit_transform(train_dem)
scaled_val = dem_scaler.transform(val_dem)
# Create temparature as past covariate
past_cov = series['TEMPERATURE']
#split temp series data
train_temp,val_temp = past_cov.split_after(training_cutoff)
# Scale temperature data
temp_scaler = Scaler()
scaled_temp_past = temp_scaler.fit_transform(train_temp)
# Create time features and scale those
cov_t = datetime_attribute_timeseries(series,attribute='year',one_hot=False)
cov_t= cov_t.stack(datetime_attribute_timeseries(series,attribute='month',one_hot=False))
cov_t = cov_t.stack(datetime_attribute_timeseries(series,attribute='dayofyear',one_hot=False))
cov_t = cov_t.stack(datetime_attribute_timeseries(series,attribute='hour',one_hot=False))
cov_t = cov_t.stack(datetime_attribute_timeseries(series,attribute='minute',one_hot=False))
cov_t = cov_t.stack(datetime_attribute_timeseries(series,attribute='dayofweek',one_hot=False))
# convert the frame to float32 for memory efficiency
cov_t = cov_t.astype(np.float32)
time_scaler = Scaler()
scaled_cov_t = time_scaler.fit_transform(cov_t)
# Randomized gridsearch
parameters = {  "input_chunk_length":[336, 672, 1344, 2688], 
                "output_chunk_length":[48],
                "hidden_size":[32, 64, 128, 256], 
                "lstm_layers":[1,2,3], 
                "num_attention_heads":[4, 6, 8,10], 
                "dropout":[0.1,0.2], 
                "batch_size":[36, 72,96,144], 
                "n_epochs":[10,20,50,100]
                }
# Inintiate search
res = TFTModel.gridsearch(    
                            parameters=parameters,
                            series=scaled_train, 
                            past_covariates=scaled_temp_past,
                            future_covariates=cov_t, 
                            val_series=scaled_val,   
                            last_points_only=False, 
                            metric=mape, 
                            reduction=np.mean, 
                            n_jobs=12, 
                            n_random_samples=0.99,      # % of full search space to evaluate
                            verbose=True)
bestmodel, dict_bestparams = res
# training the model with best parameters
bestmodel.fit(  scaled_train, 
                future_covariates=cov_t, 
                verbose=True)
# backtesting
forecast_horizon = 48
backtest_series = bestmodel.historical_forecasts(
    scaled_train,
    future_covariates=scaled_cov_t,
    start=pd.Timestamp("20210310"), # change to start=pd.Timestamp("20200731040000")
    #num_samples=num_samples,
    forecast_horizon=forecast_horizon*7,
    stride=1,
    last_points_only=False,
    retrain=False,
    verbose=False,
)
# backtest results
backtest_series_dt = concatenate(dem_scaler.inverse_transform(backtest_series))
## print out backtest results
print(
        "RMSE in backtest: {:.2f}".format(
            rmse(
                train_dem[-336:],
                backtest_series_dt,
            )
        )
    )
    
print(
        "MAPE in backtest: {:.2f}".format(
            mape(
                train_dem[-336:],
                backtest_series_dt,
            )
        )
    )
# predictions
prediction=concatenate(dem_scaler.inverse_transform(bestmodel.predict(n=48,past_covariates= scaled_temp_past)))
# save predictions and validation
df_to_save = val_dem.pd_dataframe().reset_index()
df_to_save['TFT_Pred'] = prediction.pd_dataframe().reset_index()['TOTALDEMAND']
df_to_save.to_csv('TFT_Predictions_with_Actual.csv',index=False) ## change the csv name to add winter2020
# Create darts time series for easier plotting 
comb_val_series = TimeSeries.from_dataframe(df_to_save,'DATETIME',['TOTALDEMAND','TFT_Pred'])
# plot bactesting and the validation series
plt.figure(figsize=(15,6))
train_dem[-336:].plot(color='red',linestyle='dashed',alpha=0.6,lw=1)
val_dem.plot(color='blue',label='Val',alpha=0.6,lw=1, linestyle='dashed')
backtest_series_dt.plot(label='Backtest',lw=2)
comb_val_series['TFT_Pred'].plot(label='TFT_pred',color='green',lw=2)
## Print the validation results
print(
        "RMSE in val: {:.2f}".format(
            rmse(comb_val_series['TOTALDEMAND'],comb_val_series['TFT_Pred']
            )
        )
    )
    
print(
        "MAPE in val: {:.2f}".format(
            mape(
                comb_val_series['TOTALDEMAND'],comb_val_series['TFT_Pred']
            )
        )
    )