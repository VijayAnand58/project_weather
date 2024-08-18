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
import os
from dataframe_compilation import weatherdf_precip_use

file_path=r'chennai.csv'
weather_df = pd.read_csv(os.path.abspath(file_path),parse_dates=['time'])
weather_df['time'] = pd.to_datetime(weather_df['time'], format='%d-%m-%Y')
weather_df['date_ordinal'] = weather_df['time'].apply(lambda x: x.toordinal())

file_path2=r'chennai2023.csv'
weather_df_test=pd.read_csv(os.path.abspath(file_path2),parse_dates=['time'])
# print(weather_df_test)
dates_string_test=[]
weather_df_test['time'] = pd.to_datetime(weather_df_test['time'], format='%d-%m-%Y')
for i in weather_df_test['time']:
   dates_string_test.append(datetime.strftime(i,"%d-%m-%Y"))
weather_df_test['date_ordinal'] = weather_df_test['time'].apply(lambda x: x.toordinal())

weather_train_precipitation=weather_df.pop("precipitation_sum (mm)")
weather_train=weather_df[["date_ordinal","temperature_2m_max","temperature_2m_min","precipitation_hours (h)","wind_speed_10m_max (km/h)","wind_direction_10m_dominant","et0_fao_evapotranspiration (mm)"]]

# print(weather_train_precipitation)

weather_test_precipitation=weather_df_test.pop("precipitation_sum (mm)")
weather_test=weather_df_test[["date_ordinal","temperature_2m_max","temperature_2m_min","precipitation_hours (h)","wind_speed_10m_max (km/h)","wind_direction_10m_dominant","et0_fao_evapotranspiration (mm)"]]

model=LinearRegression()
model.fit(weather_train,weather_train_precipitation)

prediction=model.predict(weather_test)

np.mean(np.absolute(prediction-weather_test_precipitation))

variance_precip=round(model.score(weather_test, weather_test_precipitation),4)
# print('Variance score:',variance_precip)

for i in range(len(prediction)):
  prediction[i]=round(prediction[i],2)
weather_ml_report_precip=pd.DataFrame({'Date':dates_string_test,'Actual':weather_test_precipitation,'Prediction':prediction,'diff':(weather_test_precipitation-prediction)})
# print(weather_ml_report_precip)

future_date_precip=datetime.strptime(datetime.today().strftime('%d-%m-%Y'),'%d-%m-%Y')
avg_max_temp_train=weather_train["temperature_2m_max"].mean()
avg_min_temp_train=weather_train["temperature_2m_min"].mean()
avg_precipitation_hours_train=weather_train['precipitation_hours (h)'].mean()
avg_wind_speed_train=weather_train["wind_speed_10m_max (km/h)"].mean()
avg_wind_direction_train=weather_train["wind_direction_10m_dominant"].mean()
avg_evaporation_train=weather_train["et0_fao_evapotranspiration (mm)"].mean()
future_prediction = np.array([[future_date_precip.toordinal(),avg_max_temp_train,avg_min_temp_train,avg_precipitation_hours_train,avg_wind_speed_train,avg_wind_direction_train,avg_evaporation_train]])
prediction_precip=model.predict(future_prediction)
# print(prediction_precip)

def linear_precip_user_predict(ord_date):
  future_user_predict=np.array([[ord_date,avg_max_temp_train,avg_min_temp_train,avg_precipitation_hours_train,avg_wind_speed_train,avg_wind_direction_train,avg_evaporation_train]])
  prediction_precip_user=model.predict(future_user_predict)
  return prediction_precip_user[0]


def linear_precip_user_predict_year(year):
  import datetime
  from datetime import datetime as dt
  dates = []
  dates_string=[]
  dates_ordinal_list=[]
  # prediction_temp_user_year=[]
  for month in range(1, 13):
      for day in range(1, 32):
          try:
              date =datetime.date(year, month, day)
              dates.append(date)
              dates_string.append(date.strftime("%d-%m-%y"))
          except ValueError:
              continue
  if year%4==0:
    dates.remove(datetime.date(year,2,29))
    dates_string.remove(dt.strftime(datetime.date(year,2,29),"%d-%m-%y"))
  
  for date in dates:
     ord_date=date.toordinal()
     dates_ordinal_list.append(ord_date)

  weatherdf_precip_use_linear=weatherdf_precip_use.copy(deep=True)
  weatherdf_precip_use_linear.insert(loc=0, column='date_ordinal', value=dates_ordinal_list)
  
  prediction_precip_user_year=model.predict(weatherdf_precip_use_linear)
  weather_ml_report_user_year=pd.DataFrame({'Date':dates_string,'predicted_value':prediction_precip_user_year})
  return weather_ml_report_user_year

# print(linear_precip_user_predict_year(2025))