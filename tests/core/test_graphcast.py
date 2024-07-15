import pytest
import datetime
import xarray as xr
from skyrim.core import Skyrim
from skyrim.core.models.graphcast import CHANNELS

model = Skyrim("graphcast", ic_source="ifs")


@pytest.mark.integ
def test_output_mapping():
    # TODO: define a data container validation for prediction.
    pred, _ = model.predict("20240404", "0000", 6, save=False)
    assert set(pred.prediction.dims) == {"time", "channel", "lat", "lon"}


@pytest.mark.integ
def test_output_channels():
    start_time = datetime.datetime(2024, 4, 4, 0, 0)
    pred = model.forecast(start_time=start_time, n_steps=1)
    assert isinstance(pred, xr.DataArray)
    assert set(model.model.out_channel_names) == set(CHANNELS)
