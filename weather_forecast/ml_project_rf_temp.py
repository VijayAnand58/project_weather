from ml_project_linear_temp import *

from sklearn.ensemble import RandomForestRegressor
regr=RandomForestRegressor(random_state=0)
regr.fit(weather_train,weather_train_temp)

prediction_rf=regr.predict(weather_train)
diff_mean=np.mean(np.absolute(prediction_rf-weather_train_temp))
# print(diff_mean)

variance_rf=round(regr.score(weather_test, weather_test_temp),4)
# print('Variance score:',variance_rf)

for i in range(len(prediction_rf)):
  prediction_rf[i]=round(prediction_rf[i],2)
weather_ml_report_rf=pd.DataFrame({'Date':weather_df_test['date_ordinal'],'Actual':weather_test_temp,'Prediction':prediction_rf,'diff':(weather_test_temp-prediction_rf)})
#print(weather_ml_report_rf)

future_prediction = np.array([[future_date.toordinal(),avg_min_temp_train,avg_precipitation_sum_train,avg_precipitation_hours_train,avg_wind_speed_train,avg_wind_direction_train,avg_evaporation_train]])
prediction_rf=regr.predict(future_prediction)
# print(prediction_rf)