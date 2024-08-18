from ml_project_linear_temp import *

from sklearn.ensemble import RandomForestRegressor
regr=RandomForestRegressor(random_state=42)
regr.fit(weather_train,weather_train_temp)

prediction_rf=regr.predict(weather_test)
diff_mean=np.mean(np.absolute(prediction_rf-weather_test_temp))
# print(diff_mean)

variance_rf=round(regr.score(weather_test, weather_test_temp),4)
# print('Variance score:',variance_rf)

for i in range(len(prediction_rf)):
  prediction_rf[i]=round(prediction_rf[i],2)
weather_ml_report_rf=pd.DataFrame({'Date':dates_string_test,'Actual':weather_test_temp,'Prediction':prediction_rf,'diff':(weather_test_temp-prediction_rf)})
# print(weather_ml_report_rf)

future_prediction = np.array([[future_date.toordinal(),avg_precipitation_sum_train,avg_precipitation_hours_train,avg_wind_speed_train,avg_wind_direction_train,avg_evaporation_train]])
prediction_rf=regr.predict(future_prediction)
# print(prediction_rf)


def rf_temp_user_predict(ord_date):
  future_user_prediction = np.array([[ord_date,avg_precipitation_sum_train,avg_precipitation_hours_train,avg_wind_speed_train,avg_wind_direction_train,avg_evaporation_train]])
  prediction_rf_temp_user=regr.predict(future_user_prediction)
  return prediction_rf_temp_user[0]

def rf_temp_user_predict_year(year):
  import datetime
  from datetime import datetime as dt
  dates = []
  dates_string=[]
  prediction_temp_rf_user_year=[]
  dates_ordinal_list=[]
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
  weatherdf_temp_use_rf=weatherdf_temp_use.copy(deep=True)
  weatherdf_temp_use_rf.insert(loc=0, column='date_ordinal', value=dates_ordinal_list)
  prediction_temp_rf_user_year=regr.predict(weatherdf_temp_use_rf)
  weather_ml_report_user_year=pd.DataFrame({'Date':dates_string,'predicted_value':prediction_temp_rf_user_year})
  return weather_ml_report_user_year

# print(rf_temp_user_predict_year(2024))