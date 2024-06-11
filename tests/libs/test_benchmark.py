from skyrim.libs.benchmark.openmeteo import forecast
from skyrim.libs.benchmark.observations import observe
from datetime import date

lat, lon = 37.0557, 28.3242
start, end = date(2024, 6, 1), date(2024, 6, 3)


def test_fetch_and_combine_observations():
    df = observe(lat, lon, ("t2m", "wspd"), start, end)
    dff = forecast(lat, lon, ("t2m", "wspd10m"), start, end)
    res = dff.join(df)
    assert res.shape == (72, 2)
