import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression

import joblib

from ml_project_linear_temp import weather_train, weather_train_temp

# linear regression temprature


model=LinearRegression()
model.fit(weather_train,weather_train_temp)

joblib.dump(model,"lineartemp.pkl")


# Random forest temp

from sklearn.ensemble import RandomForestRegressor
regr=RandomForestRegressor(random_state=42)
regr.fit(weather_train,weather_train_temp)

joblib.dump(regr,"randomtemp.pkl")


