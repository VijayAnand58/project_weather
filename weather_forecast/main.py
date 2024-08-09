import json

from ml_project_linear_temp import variance,weather_ml_report,prediction
from ml_project_linear_precipitation import variance_precip,weather_ml_report_precip,prediction_precip
from ml_project_rf_temp import variance_rf, weather_ml_report_rf,prediction_rf
from ml_project_rf_precipitation import variance_rf_precip,weather_ml_report_rf_precip,prediction_rf_precip

# def fetch_data():
#     data={
#         "temperature_linear":{
#             'variance':variance,
#             'weather_report':weather_ml_report.to_json(orient='records'),
#             'prediction':prediction[0]

#         },
#         'temperature_rf':{
#             'variance':variance_rf,
#             'weather_report':weather_ml_report_rf.to_json(orient="records"),
#             'prediction':prediction_rf[0]
#         },
#         "precipitation_linear":{
#             'variance':variance_precip,
#             'weather_report':weather_ml_report_precip.to_json(orient='records'),
#             'prediction':prediction_precip[0]

#         },
#         'precipitation_rf':{
#             'variance':variance_rf_precip,
#             'weather_report':weather_ml_report_rf_precip.to_json(orient="records"),
#             'prediction':prediction_rf_precip[0]
#         }
#     }
class api_transform:
    def linear_temp(self):
        data={
        "temperature_linear":{
            'variance':variance,
            'weather_report':weather_ml_report.to_json(orient='records'),
            'prediction':prediction[0]
        }
        }
        return data
    def linear_precip(self):
        data={
            'variance':variance_precip,
            'weather_report':weather_ml_report_precip.to_json(orient='records'),
            'prediction':prediction_precip[0]
            
            }
        return data
    def rf_temp(self):
        data={
            'temperature_rf':{
            'variance':variance_rf,
            'weather_report':weather_ml_report_rf.to_json(orient="records"),
            'prediction':prediction_rf[0]
        }
        }
        return data
    def rf_precip(self):
        data={
            'precipitation_rf':{
            'variance':variance_rf_precip,
            'weather_report':weather_ml_report_rf_precip.to_json(orient="records"),
            'prediction':prediction_rf_precip[0]
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


@app.post("/submit-date")
async def sub_date(request:Request):
    try:
        data=await request.json()
        if "date" not in data:
            raise HTTPException(
                status_code=422, detail="Incomplete data provided")
        ord_date=data["date"]
        return {"message": " Date submitted successfully!"}
    except HTTPException as e:
        # Re-raise HTTPException to return the specified 
        # status code and detail
        raise e
    except Exception as e:
        # Handle other unexpected exceptions and return a 
        # 500 Internal Server Error
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")

    



