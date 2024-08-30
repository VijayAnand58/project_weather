import json
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sklearn.metrics import r2_score


# import os

from ml_project_linear_temp import variance,weather_ml_report,prediction_today,linear_temp_user_predict,linear_temp_user_predict_year,prediction,weather_test_temp
from ml_project_linear_precipitation import variance_precip,weather_ml_report_precip,prediction_precip_today,linear_precip_user_predict,linear_precip_user_predict_year,prediction_precip,weather_test_precipitation
from ml_project_rf_temp import variance_rf, weather_ml_report_rf,prediction_rf_today,rf_temp_user_predict,rf_temp_user_predict_year,prediction_rf
from ml_project_rf_precipitation import variance_rf_precip,weather_ml_report_rf_precip,prediction_rf_precip_today,rf_precip_user_predict,rf_precip_user_predict_year,prediction_rf_precip

def json_converter(dataframe):
    dataframe_json=dataframe.to_json(orient="table")
    dataframe_json_object=json.loads(dataframe_json)
    dataframe_json_data=json.loads(json.dumps(dataframe_json_object['data']))
    return dataframe_json_data

def performance_score(prediction_array,test_value):
   mean_absolute_error=round(np.mean(np.absolute(prediction_array - test_value)),3)
#    residual_sum_of_squares=round(np.mean((prediction_array - test_value) ** 2),3)
   r2_score_value=round(r2_score(test_value,prediction_array ),3)
   return{"mean_absolute_error":mean_absolute_error,"r2_score":r2_score_value}

class api_transform:
    def linear_temp(self):
        data={
        "temperature_linear":{
            'variance':variance,
            'Performace_measures': performance_score(prediction,weather_test_temp),
            # 'test_date':weather_ml_report['Date'].tolist(),
            # 'test_actual':weather_ml_report['Actual'].tolist(),
            # 'test_prediction':weather_ml_report['Prediction'].tolist(),
            # 'test_diff':weather_ml_report['diff'].tolist(),
            'prediction':prediction_today[0],
            'weather_model':json_converter(weather_ml_report)
            
        }
        }
        return data
    def linear_precip(self):
        data={
            'variance':variance_precip,
            'Performance_measures': performance_score(prediction_precip,weather_test_precipitation),
            # 'test_date':weather_ml_report_precip['Date'].tolist(),
            # 'test_actual':weather_ml_report_precip['Actual'].tolist(),
            # 'test_prediction':weather_ml_report_precip['Prediction'].tolist(),
            # 'test_diff':weather_ml_report_precip['diff'].tolist(),
            'prediction':prediction_precip_today[0],
            'weather_model':json_converter(weather_ml_report_precip)            
            }
        return data
    def rf_temp(self):
        data={
            'temperature_rf':{
            'variance':variance_rf,
            'Performance_measures': performance_score(prediction_rf,weather_test_temp),
            # 'test_date':weather_ml_report_rf['Date'].tolist(),
            # 'test_actual':weather_ml_report_rf['Actual'].tolist(),
            # 'test_prediction':weather_ml_report_rf['Prediction'].tolist(),
            # 'test_diff':weather_ml_report_rf['diff'].tolist(),
            'prediction':prediction_rf_today[0],
            'weather_model':json_converter(weather_ml_report_rf)
        }
        }
        return data
    def rf_precip(self):
        data={
            'precipitation_rf':{
            'variance':variance_rf_precip,
            'Performance_measures': performance_score(prediction_rf_precip,weather_test_precipitation),
            # 'test_date':weather_ml_report_rf_precip['Date'].tolist(),
            # 'test_actual':weather_ml_report_rf_precip['Actual'].tolist(),
            # 'test_prediction':weather_ml_report_rf_precip['Prediction'].tolist(),
            # 'test_diff':weather_ml_report_rf_precip['diff'].tolist(),
            'prediction':prediction_rf_precip_today[0],
            'weather_model':json_converter(weather_ml_report_rf_precip)
        }
        }
        return data
instance=api_transform()

from typing import Union
from fastapi import FastAPI,HTTPException,Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
# origins = [
#     "http://localhost:5173",  # Add your frontend URL here
#     "https://temp-predictor-one.vercel.app/",  # Add your production URL here
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/data/temp/linear")
async def get_liner_temp_data():
   return instance.linear_temp()
@app.get("/data/temp/rf")
async def get_rf_temp_data():
    return instance.rf_temp()
@app.get("/data/precip/linear")
async def get_liner_precip_data():
   return instance.linear_precip()
@app.get("/data/precip/rf")
async def get_rf_precip_data():
   return instance.rf_precip()

class user_date(BaseModel):
    date:str


@app.post("/submit-date/")
async def sub_date(date:user_date):
    try:
        data=json.loads(date.model_dump_json())
        if "date" not in data:
            raise HTTPException(
                status_code=422, detail="Incomplete data provided")
        ord_date=datetime.strptime(data["date"],"%d-%m-%Y").toordinal()
        return {"message": " Date submitted successfully!",
                "Date":ord_date,
                "linear_temp":linear_temp_user_predict(ord_date),
                "linear_precip":linear_precip_user_predict(ord_date),
                "rf_temp":rf_temp_user_predict(ord_date),
                "rf_precip":rf_precip_user_predict(ord_date)}
    except HTTPException as e:
        # Re-raise HTTPException to return the specified 
        # status code and detail
        raise e
    except Exception as e:
        # Handle other unexpected exceptions and return a 
        # 500 Internal Server Error
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")


class user_year(BaseModel):
    year:int

@app.post("/submit-year/")
async def sub_date(year:user_year):
    try:
        data=json.loads(year.model_dump_json())
        if "year" not in data:
            raise HTTPException(
                status_code=422, detail="Incomplete data provided")
        return {"message": " Date submitted successfully!",
                "year":data["year"],
                "linear_temp":json_converter(linear_temp_user_predict_year(data['year'])),
                "linear_precip":json_converter(linear_precip_user_predict_year(data['year'])),
                "rf_temp":json_converter(rf_temp_user_predict_year(data['year'])),
                "rf_precip":json_converter(rf_precip_user_predict_year(data['year']))}
    except HTTPException as e:
        # Re-raise HTTPException to return the specified 
        # status code and detail
        raise e
    except Exception as e:
        # Handle other unexpected exceptions and return a 
        # 500 Internal Server Error
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")

