import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression

import joblib

#linear regresssion
from ml_project_linear_precipitation import weather_train,weather_train_precipitation

model=LinearRegression()
model.fit(weather_train,weather_train_precipitation)

joblib.dump(model,"linearprecip.pkl")


#random forest
from sklearn.ensemble import RandomForestRegressor

regr_precip=RandomForestRegressor(random_state=42)
regr_precip.fit(weather_train,weather_train_precipitation)
joblib.dump(regr_precip,"randomprecip.pkl")
