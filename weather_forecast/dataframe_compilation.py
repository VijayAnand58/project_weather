import numpy as np
import pandas as pd
import os
import datetime
import copy

file2020=r"chennai2020.csv"
file2021=r"chennai2021.csv"
file2022=r"chennai2022.csv"
file2023=r"chennai2023.csv"

weatherdf1=pd.read_csv(os.path.abspath(file2020),parse_dates=['time'])
weatherdf2=pd.read_csv(os.path.abspath(file2021),parse_dates=['time'])
weatherdf3=pd.read_csv(os.path.abspath(file2022),parse_dates=['time'])
weatherdf4=pd.read_csv(os.path.abspath(file2023),parse_dates=['time'])

weatherdf1.pop("time")
weatherdf2.pop("time")
weatherdf3.pop("time")
weatherdf4.pop('time')

mean_df=(weatherdf1+weatherdf2+weatherdf3+weatherdf4)/4

weatherdf_temp_use=mean_df[["precipitation_sum (mm)","precipitation_hours (h)","wind_speed_10m_max (km/h)","wind_direction_10m_dominant","et0_fao_evapotranspiration (mm)"]]

# print(weatherdf_temp_use)

weatherdf_precip_use=mean_df[["temperature_2m_max",'temperature_2m_min',"precipitation_hours (h)","wind_speed_10m_max (km/h)","wind_direction_10m_dominant","et0_fao_evapotranspiration (mm)"]]

# print(weatherdf_precip_use)