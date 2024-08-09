from ml_project_linear_precipitation import *


from sklearn.ensemble import RandomForestRegressor
regr_precip=RandomForestRegressor(random_state=0)
regr_precip.fit(weather_train,weather_train_precipitation)

prediction_rf_precip=regr_precip.predict(weather_train)
diff_mean=np.mean(np.absolute(prediction_rf_precip-weather_train_precipitation))
# print(diff_mean)

variance_rf_precip=round(regr_precip.score(weather_test, weather_test_precipitation),4)
# print('Variance score:',variance_rf_precip)

for i in range(len(prediction_rf_precip)):
  prediction_rf_precip[i]=round(prediction_rf_precip[i],2)
weather_ml_report_rf_precip=pd.DataFrame({'Date':weather_df_test['date_ordinal'],'Actual':weather_test_precipitation,'Prediction':prediction_rf_precip,'diff':(weather_test_precipitation-prediction_rf_precip)})
#print(weather_ml_report_rf_precip)

future_prediction = np.array([[future_date_precip.toordinal(),avg_max_temp_train,avg_min_temp_train,avg_precipitation_hours_train,avg_wind_speed_train,avg_wind_direction_train,avg_evaporation_train]])
prediction_rf_precip=regr_precip.predict(future_prediction)
# print(prediction_rf_precip)