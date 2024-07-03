from skyrim.libs.benchmark.openmeteo import forecast, forecast_past
from skyrim.libs.benchmark.observations import observe
from datetime import date

lat, lon = 37.0557, 28.3242
start, end = date(2024, 6, 1), date(2024, 6, 3)


def test_fetch_and_combine_observations():
    df = observe(lat, lon, ("t2m", "wspd"), start, end)
    dff = forecast(lat, lon, ("t2m", "wspd10m"), start, end)
    res = dff.join(df)
    assert res.shape == (72, 4)


def test_historical_runs():
    df = forecast_past(lat, lon, ("t2m", "wspd10m"), (1, 2), start, end)
    assert df.shape == (72, 6)
    assert set(df.columns) == {
        "t2m",
        "wspd10m",
        "temperature_2m_previous_day1",
        "wind_speed_10m_previous_day1",
        "temperature_2m_previous_day2",
        "wind_speed_10m_previous_day2",
    }
