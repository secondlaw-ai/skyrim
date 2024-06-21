import openmeteo_requests
import pandas as pd
import numpy as np
from datetime import date
from functools import lru_cache

om_to_ecmwf = {
    "wspd10m": "wind_speed_10m",
    "wspd100m": "wind_speed_100m",
    "t2m": "temperature_2m",
    "q2m": "relative_humidity_2m",
    "wdir10m": "wind_direction_10m",
    "wdir100m": "wind_direction_100m",
}

ecmwf_to_om = {v: k for k, v in om_to_ecmwf.items()}


def to_om_vars(vars: tuple):
    vars_ = []
    for v in vars:
        v_ = om_to_ecmwf.get(v)
        if not v_:
            raise Exception(f"Undefined var {v}. Possible : {om_to_ecmwf}")
        vars_.append(v_)
    return vars_


def send_request(url: str, params: dict, vars: tuple) -> pd.DataFrame:
    openmeteo = openmeteo_requests.Client()
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
    for i, v_ in enumerate(vars):
        hourly_data[v_] = hourly.Variables(i).ValuesAsNumpy()
    df = pd.DataFrame(data=hourly_data)
    df.rename(columns=ecmwf_to_om, inplace=True)
    df["date"] = df["date"].apply(lambda x: x.replace(tzinfo=None))
    return df.set_index(pd.to_datetime(df.date)).drop(columns=["date"])


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

    url = "https://archive-api.open-meteo.com/v1/archive"
    available_models = ["ecmwf_ifs", "era5_seamless", "era5", "era5_land", "cerra"]
    if model not in available_models:
        raise Exception(f"Undefined model {model}, available: {available_models}")
    vars_ = to_om_vars(vars)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": vars_,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "models": model,
        "wind_speed_unit": "ms",
    }
    return send_request(url, params, vars_)


@lru_cache(maxsize=1024)
def forecast_past(
    lat: float,
    lon: float,
    vars: tuple,
    past_days: tuple,
    start_date: date,
    end_date: date,
    model="ecmwf_ifs025",
):
    """
    Fetch weather data from OpenMeteo API for a given location and variables. Based on OpenMeteo API Python doc.
    HRES 9km resolution, hourly data, goes back to 2017.
    :param lat: Latitude of the location
    :param lon: Longitude of the location
    :param vars: List of variables to fetch e.g. ["temperature_2m", "wind_speed_10m"] see https://open-meteo.com/en/docs/historical-weather-api
    """
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    vars_ = to_om_vars(vars)
    past_day_vars = []
    for p in past_days:
        if p > 7:
            raise Exception("Only 7 previous days available")
        suffix = "_previous_day" + str(p)
        past_day_vars += [v + suffix for v in vars_]
    vars_ += past_day_vars
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": vars_,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "models": model,
        "wind_speed_unit": "ms",
    }
    return send_request(url, params, vars_)


@lru_cache(maxsize=1024)
def forecast_multimodel(lat, lon, vars, start_date, end_date, models):
    dfs = []
    for m in models:
        df = forecast(lat, lon, vars, start_date, end_date, model=m)
        dfs.append(
            df.rename(columns=dict(zip(df.columns, (f"{m}_{c}" for c in df.columns))))
        )
    return dfs
