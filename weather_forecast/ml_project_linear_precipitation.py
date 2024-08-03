import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

from datetime import datetime

file_path=r'C:\Users\vijay\Documents\Python VS code\project_weather\weather_forecast\chennai.csv'
weather_df = pd.read_csv(file_path,parse_dates=['time'])
weather_df['time'] = pd.to_datetime(weather_df['time'], format='%d-%m-%Y')
weather_df['date_ordinal'] = weather_df['time'].apply(lambda x: x.toordinal())

file_path2=r"C:\Users\vijay\Documents\Python VS code\project_weather\weather_forecast\chennai2023.csv"
weather_df_test=pd.read_csv(file_path2,parse_dates=['time'])
weather_df_test['time'] = pd.to_datetime(weather_df_test['time'], format='%d-%m-%Y')
weather_df_test['date_ordinal'] = weather_df_test['time'].apply(lambda x: x.toordinal())

weather_train_precipitation=weather_df.pop("precipitation_sum (mm)")
weather_train=weather_df[["date_ordinal","temperature_2m_max","temperature_2m_min","precipitation_hours (h)","wind_speed_10m_max (km/h)","wind_direction_10m_dominant","et0_fao_evapotranspiration (mm)"]]

print(weather_train_precipitation)

weather_test_precipitation=weather_df_test.pop("precipitation_sum (mm)")
weather_test=weather_df_test[["date_ordinal","temperature_2m_max","temperature_2m_min","precipitation_hours (h)","wind_speed_10m_max (km/h)","wind_direction_10m_dominant","et0_fao_evapotranspiration (mm)"]]

model=LinearRegression()
model.fit(weather_train,weather_train_precipitation)

prediction=np.absolute(model.predict(weather_test))

np.mean(np.absolute(prediction-weather_test_precipitation))

print('Variance score: %.2f' % model.score(weather_test, weather_test_precipitation))

for i in range(len(prediction)):
  prediction[i]=round(prediction[i],2)
weather_ml_report=pd.DataFrame({'Date':weather_df_test['time'],'Actual':weather_test_precipitation,'Prediction':prediction,'diff':(weather_test_precipitation-prediction)})
print(weather_ml_report)