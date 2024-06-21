import pandas as pd
from datetime import datetime, date
from meteostat import Point, Daily, Hourly
from meteostat import Stations
from functools import lru_cache


@lru_cache(maxsize=1024)
def get_closest_station(lat: float, lon: float) -> pd.DataFrame:
    return Stations().nearby(lat, lon).fetch(1)


@lru_cache(maxsize=1024)
def observe(lat: float, lon: float, vars: tuple, start_time: date, end_time: date):
    """
    Fetch observations for the nearest station
    """
    station = get_closest_station(lat, lon)
    df = Hourly(
        station,
        datetime(start_time.year, start_time.month, start_time.day),
        datetime(end_time.year, end_time.month, end_time.day + 1),
    ).fetch()
    df.index.rename("date", inplace=True)
    # https://dev.meteostat.net/formats.html#meteorological-parameters
    o_to_ecmwf = {
        "temp": "t2m",
        "rhum": "q",
        "wdir": "wdir",
        "wspd": "wspd",
        "prcp": "tp",  # Total precip
        "rhum": "r",  # relative humidity
        "pres": "sp",  # sea level air pressure
    }
    o_to_ecmwf = {k: "o_" + v for k, v in o_to_ecmwf.items()}
    vars = ["o_" + v for v in vars]
    df.rename(columns=o_to_ecmwf, inplace=True)
    conversions = {"o_wspd": 0.277778}  # km/h to m/s
    for c in conversions:
        if c in vars:
            df[c] = df[c] * conversions[c]
    return df.loc[:, vars]
