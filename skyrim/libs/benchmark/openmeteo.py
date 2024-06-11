import openmeteo_requests
import pandas as pd
import numpy as np
from datetime import date
from functools import lru_cache


@lru_cache(maxsize=1024)
def forecast(
    lat: float,
    lon: float,
    vars: tuple,
    start_date: date,
    end_date: date,
    model="ecmwf_ifs",
):
    """
    Fetch weather data from OpenMeteo API for a given location and variables. Based on OpenMeteo API Python doc.
    HRES 9km resolution, hourly data, goes back to 2017.
    :param lat: Latitude of the location
    :param lon: Longitude of the location
    :param vars: List of variables to fetch e.g. ["temperature_2m", "wind_speed_10m"] see https://open-meteo.com/en/docs/historical-weather-api
    """
    openmeteo = openmeteo_requests.Client()
    url = "https://archive-api.open-meteo.com/v1/archive"
    om_to_ecmwf = {
        "wspd10m": "wind_speed_10m",
        "wspd100m": "wind_speed_100m",
        "t2m": "temperature_2m",
        "q2m": "relative_humidity_2m",
        "wdir10m": "wind_direction_10m",
        "wdir100m": "wind_direction_100m",
    }
    available_models = ["ecmwf_ifs", "era5_seamless", "era5", "era5_land", "cerra"]
    if model not in available_models:
        raise Exception(f"Undefined model {model}, available: {available_models}")
    ecmwf_to_om = {v: k for k, v in om_to_ecmwf.items()}
    vars_ = []
    for v in vars:
        v_ = om_to_ecmwf.get(v)
        if not v_:
            raise Exception(f"Undefined var {v}. Possible : {om_to_ecmwf}")
        vars_.append(v_)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": vars_,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "models": model,
        "wind_speed_unit": "ms",
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()  # highest temporal resolution is hourly
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    for i, v_ in enumerate(vars_):
        hourly_data[v_] = hourly.Variables(i).ValuesAsNumpy()
    df = pd.DataFrame(data=hourly_data)
    df.rename(columns=ecmwf_to_om, inplace=True)
    df["date"] = df["date"].apply(lambda x: x.replace(tzinfo=None))
    return df.set_index(pd.to_datetime(df.date)).drop(columns=["date"])


@lru_cache(maxsize=1024)
def forecast_multimodel(lat, lon, vars, start_date, end_date, models):
    dfs = []
    for m in models:
        df = forecast(lat, lon, vars, start_date, end_date, model=m)
        dfs.append(
            df.rename(columns=dict(zip(df.columns, (f"{m}_{c}" for c in df.columns))))
        )
    return dfs
