import json
from datetime import datetime
# import os

from ml_project_linear_temp import variance,weather_ml_report,prediction,linear_temp_user_predict,linear_temp_user_predict_year
from ml_project_linear_precipitation import variance_precip,weather_ml_report_precip,prediction_precip,linear_precip_user_predict,linear_precip_user_predict_year
from ml_project_rf_temp import variance_rf, weather_ml_report_rf,prediction_rf,rf_temp_user_predict,rf_temp_user_predict_year
from ml_project_rf_precipitation import variance_rf_precip,weather_ml_report_rf_precip,prediction_rf_precip,rf_precip_user_predict,rf_precip_user_predict_year

def json_converter(dataframe):
    dataframe_json=dataframe.to_json(orient="table")
    dataframe_json_object=json.loads(dataframe_json)
    dataframe_json_data=json.loads(json.dumps(dataframe_json_object['data']))
    return dataframe_json_data


class api_transform:
    def linear_temp(self):
        data={
        "temperature_linear":{
            'variance':variance,
            # 'test_date':weather_ml_report['Date'].tolist(),
            # 'test_actual':weather_ml_report['Actual'].tolist(),
            # 'test_prediction':weather_ml_report['Prediction'].tolist(),
            # 'test_diff':weather_ml_report['diff'].tolist(),
            'prediction':prediction[0],
            'weather_model':json_converter(weather_ml_report)
            
        }
        }
        return data
    def linear_precip(self):
        data={
            'variance':variance_precip,
            # 'test_date':weather_ml_report_precip['Date'].tolist(),
            # 'test_actual':weather_ml_report_precip['Actual'].tolist(),
            # 'test_prediction':weather_ml_report_precip['Prediction'].tolist(),
            # 'test_diff':weather_ml_report_precip['diff'].tolist(),
            'prediction':prediction_precip[0],
            'weather_model':json_converter(weather_ml_report_precip)            
            }
        return data
    def rf_temp(self):
        data={
            'temperature_rf':{
            'variance':variance_rf,
            # 'test_date':weather_ml_report_rf['Date'].tolist(),
            # 'test_actual':weather_ml_report_rf['Actual'].tolist(),
            # 'test_prediction':weather_ml_report_rf['Prediction'].tolist(),
            # 'test_diff':weather_ml_report_rf['diff'].tolist(),
            'prediction':prediction_rf[0],
            'weather_model':json_converter(weather_ml_report_rf)
        }
        }
        return data
    def rf_precip(self):
        data={
            'precipitation_rf':{
            'variance':variance_rf_precip,
            # 'test_date':weather_ml_report_rf_precip['Date'].tolist(),
            # 'test_actual':weather_ml_report_rf_precip['Actual'].tolist(),
            # 'test_prediction':weather_ml_report_rf_precip['Prediction'].tolist(),
            # 'test_diff':weather_ml_report_rf_precip['diff'].tolist(),
            'prediction':prediction_rf_precip[0],
            'weather_model':json_converter(weather_ml_report_rf_precip)
        }
        }
        return data
instance=api_transform()

from typing import Union
from fastapi import FastAPI,HTTPException,Request
from pydantic import BaseModel

app = FastAPI()

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

