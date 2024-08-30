from ml_project_linear_precipitation import *


from sklearn.ensemble import RandomForestRegressor
regr_precip=RandomForestRegressor(random_state=42)
regr_precip.fit(weather_train,weather_train_precipitation)

prediction_rf_precip=regr_precip.predict(weather_test)
# diff_mean=np.mean(np.absolute(prediction_rf_precip-weather_test_precipitation))
# print(diff_mean)

variance_rf_precip=round(regr_precip.score(weather_test, weather_test_precipitation),4)
# print('Variance score:',variance_rf_precip)

for i in range(len(prediction_rf_precip)):
  prediction_rf_precip[i]=round(prediction_rf_precip[i],2)
  prediction_rf_precip[i]=zero_conv(prediction_rf_precip[i])
weather_ml_report_rf_precip=pd.DataFrame({'Date':dates_string_test,'Actual':weather_test_precipitation,'Prediction':prediction_rf_precip,'diff':(weather_test_precipitation-prediction_rf_precip)})
# print(weather_ml_report_rf_precip)

future_prediction = np.array([[future_date_precip.toordinal(),avg_max_temp_train,avg_min_temp_train,avg_precipitation_hours_train,avg_wind_speed_train,avg_wind_direction_train,avg_evaporation_train]])
prediction_rf_precip_today=zero_conv(regr_precip.predict(future_prediction))
# print(prediction_rf_precip)

def rf_precip_user_predict(ord_date):
  future_user_prediction=np.array([[ord_date,avg_max_temp_train,avg_min_temp_train,avg_precipitation_hours_train,avg_wind_speed_train,avg_wind_direction_train,avg_evaporation_train]])
  prediction_rf_precip_user=regr_precip.predict(future_user_prediction)
  return zero_conv(prediction_rf_precip_user[0])


def rf_precip_user_predict_year(year):
  import datetime
  from datetime import datetime as dt
  dates = []
  dates_string=[]
  prediction_precip_rf_user_year=[]
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
  weatherdf_precip_use_rf=weatherdf_precip_use.copy(deep=True)
  weatherdf_precip_use_rf.insert(loc=0, column='date_ordinal', value=dates_ordinal_list)
  prediction_precip_rf_user_year=regr_precip.predict(weatherdf_precip_use_rf)
  for i in range(len(prediction_precip_rf_user_year)):
    prediction_precip_rf_user_year[i]=round(prediction_precip_rf_user_year[i],2)
    prediction_precip_rf_user_year[i]=zero_conv(prediction_precip_rf_user_year[i])
  weather_ml_report_user_year=pd.DataFrame({'Date':dates_string,'predicted_value':prediction_precip_rf_user_year})
  return weather_ml_report_user_year

# print(rf_precip_user_predict_year(2025))
