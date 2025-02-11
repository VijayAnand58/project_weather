import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from datetime import datetime
import os

import joblib

from dataframe_compilation import weatherdf_temp_use

file_path=r'chennai.csv'
weather_df = pd.read_csv(os.path.abspath(file_path),parse_dates=['time'])
weather_df['time'] = pd.to_datetime(weather_df['time'], format='%d-%m-%Y')
weather_df['date_ordinal'] = weather_df['time'].apply(lambda x: x.toordinal())


file_path2=r'chennai2023.csv'
weather_df_test=pd.read_csv(os.path.abspath(file_path2),parse_dates=['time'])
dates_string_test=[]
weather_df_test['time'] = pd.to_datetime(weather_df_test['time'], format='%d-%m-%Y')
for i in weather_df_test['time']:
   dates_string_test.append(datetime.strftime(i,"%d-%m-%Y"))
weather_df_test['date_ordinal'] = weather_df_test['time'].apply(lambda x: x.toordinal())


weather_train_temp=weather_df.pop("temperature_2m_max")
weather_train=weather_df[["date_ordinal","precipitation_sum (mm)","precipitation_hours (h)","wind_speed_10m_max (km/h)","wind_direction_10m_dominant","et0_fao_evapotranspiration (mm)"]]

weather_test_temp=weather_df_test.pop("temperature_2m_max")
weather_test=weather_df_test[["date_ordinal","precipitation_sum (mm)","precipitation_hours (h)","wind_speed_10m_max (km/h)","wind_direction_10m_dominant","et0_fao_evapotranspiration (mm)"]]

model=joblib.load("lineartemp.pkl")

prediction=model.predict(weather_test)
variance=round(model.score(weather_test, weather_test_temp),4)


for i in range(len(prediction)):
  prediction[i]=round(prediction[i],2)
weather_ml_report=pd.DataFrame({'Date':dates_string_test,'Actual':weather_test_temp,'Prediction':prediction,'diff':round((weather_test_temp-prediction),3)})
 


future_date=datetime.strptime(datetime.today().strftime('%d-%m-%Y'),'%d-%m-%Y')
avg_precipitation_sum_train=weather_train["precipitation_sum (mm)"].mean()
avg_precipitation_hours_train=weather_train['precipitation_hours (h)'].mean()
avg_wind_speed_train=weather_train["wind_speed_10m_max (km/h)"].mean()
avg_wind_direction_train=weather_train["wind_direction_10m_dominant"].mean()
avg_evaporation_train=weather_train["et0_fao_evapotranspiration (mm)"].mean()
future_prediction = np.array([[future_date.toordinal(),avg_precipitation_sum_train,avg_precipitation_hours_train,avg_wind_speed_train,avg_wind_direction_train,avg_evaporation_train]])

prediction_today=model.predict(future_prediction)


def linear_temp_user_predict(ord_date):
  future_user_predict=np.array([[ord_date,avg_precipitation_sum_train,avg_precipitation_hours_train,avg_wind_speed_train,avg_wind_direction_train,avg_evaporation_train]])
  prediction_temp_user=model.predict(future_user_predict)
  return prediction_temp_user[0]

def linear_temp_user_predict_year(year):
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
  weatherdf_temp_use_linear=weatherdf_temp_use.copy(deep=True)
  weatherdf_temp_use_linear.insert(loc=0, column='date_ordinal', value=dates_ordinal_list)
  prediction_temp_user_year=model.predict(weatherdf_temp_use_linear)
  weather_ml_report_user_year=pd.DataFrame({'Date':dates_string,'predicted_value':prediction_temp_user_year})
  return weather_ml_report_user_year
