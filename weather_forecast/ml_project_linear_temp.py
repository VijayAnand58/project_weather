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
#print(weather_df)
#print(weather_df.columns)
#print(weather_df.isnull().any())
# no coulmn has null values
file_path2=r"C:\Users\vijay\Documents\Python VS code\project_weather\weather_forecast\chennai2023.csv"
weather_df_test=pd.read_csv(file_path2,parse_dates=['time'])
weather_df_test['time'] = pd.to_datetime(weather_df_test['time'], format='%d-%m-%Y')
weather_df_test['date_ordinal'] = weather_df_test['time'].apply(lambda x: x.toordinal())
#print(weather_df_test)


weather_train_temp=weather_df.pop("temperature_2m_max")
weather_train=weather_df[["date_ordinal","temperature_2m_min","precipitation_sum (mm)","precipitation_hours (h)","wind_speed_10m_max (km/h)","wind_direction_10m_dominant","et0_fao_evapotranspiration (mm)"]]

print(weather_train_temp.shape)
print(weather_train.shape)

weather_test_temp=weather_df_test.pop("temperature_2m_max")
weather_test=weather_df_test[["date_ordinal","temperature_2m_min","precipitation_sum (mm)","precipitation_hours (h)","wind_speed_10m_max (km/h)","wind_direction_10m_dominant","et0_fao_evapotranspiration (mm)"]]

print(weather_test_temp.shape)
print(weather_test.shape)

model=LinearRegression()
model.fit(weather_train,weather_train_temp)

prediction=model.predict(weather_test)

np.mean(np.absolute(prediction-weather_test_temp))

print('Variance score: %.2f' % model.score(weather_test, weather_test_temp))

for i in range(len(prediction)):
  prediction[i]=round(prediction[i],2)
weather_ml_report=pd.DataFrame({'Date':weather_df_test['time'],'Actual':weather_test_temp,'Prediction':prediction,'diff':(weather_test_temp-prediction)})
print(weather_ml_report)

future_date=datetime.strptime("03-08-2024", '%d-%m-%Y')
avg_min_temp_train=weather_train["temperature_2m_min"].mean()
avg_precipitation_sum_train=weather_train["precipitation_sum (mm)"].mean()
avg_precipitation_hours_train=weather_train['precipitation_hours (h)'].mean()
avg_wind_speed_train=weather_train["wind_speed_10m_max (km/h)"].mean()
avg_wind_direction_train=weather_train["wind_direction_10m_dominant"].mean()
avg_evaporation_train=weather_train["et0_fao_evapotranspiration (mm)"].mean()
future_prediction = np.array([[future_date.toordinal(),avg_min_temp_train,avg_precipitation_sum_train,avg_precipitation_hours_train,avg_wind_speed_train,avg_wind_direction_train,avg_evaporation_train]])
prediction=model.predict(future_prediction)
print(prediction)
