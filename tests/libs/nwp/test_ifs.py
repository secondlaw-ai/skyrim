import pytest
import os
import datetime
import numpy as np
import xarray as xr
from unittest.mock import patch
from unittest.mock import MagicMock
import hashlib
from skyrim.libs.nwp.ifs import IFSModel, IFS_Vocabulary


def test_ifs_vocabulary_build_vocab():
    vocab = IFS_Vocabulary.build_vocab()
    assert isinstance(vocab, dict)
    assert "u10m" in vocab
    assert "v1000" in vocab


def test_ifs_vocabulary_get():
    ifs_id, ifs_levtype, ifs_level, modifier_func = IFS_Vocabulary.get("u10m")
    assert ifs_id == "10u"
    assert ifs_levtype == "sfc"
    assert ifs_level == ""
    assert modifier_func(1) == 1

    ifs_id, ifs_levtype, ifs_level, modifier_func = IFS_Vocabulary.get("z500")
    assert ifs_id == "gh"
    assert ifs_levtype == "pl"
    assert ifs_level == "500"
    assert modifier_func(1) == 9.81


def test_ifs_model_initialization():
    channels = ["u10m", "v10m"]
    model = IFSModel(channels=channels, cache=True, source="aws")
    assert model.channels == channels
    assert model.source == "aws"


def test_ifs_model_assure_channels_exist():
    channels = ["u10m", "v10m"]
    model = IFSModel(channels=channels, cache=True, source="aws")
    model.assure_channels_exist(channels)
    with pytest.raises(Exception):
        model.assure_channels_exist(["invalid_channel"])


def test_ifs_model_list_available_channels():
    channels = IFSModel.list_available_channels()
    assert "u10m" in channels
    assert "v1000" in channels


def test_ifs_model_cache():
    channels = ["u10m", "v10m"]
    model = IFSModel(channels=channels, cache=True, source="aws")
    cache_location = model.cache
    assert cache_location == os.path.join(
        os.path.expanduser("~"), ".cache", "skyrim", "ifs"
    )
    assert os.path.exists(cache_location)


def test_ifs_model_slice_lead_time_to_steps():
    channels = ["u10m", "v10m"]
    model = IFSModel(channels=channels, cache=True, source="aws")
    start_time = datetime.datetime.strptime("20230101 0000", "%Y%m%d %H%M")
    steps = model._slice_lead_time_to_steps(24, start_time)
    assert steps == list(range(0, 25, 3))


def test_ifs_download_ifs_channel_grib_to_cache():
    # TODO: mock the download
    pass


def test_fetch_ifs_dataarray():
    channels = ["u10m", "v10m"]
    model = IFSModel(channels=channels, cache=True, source="aws")
    start_time = datetime.datetime.strptime("20230101 0000", "%Y%m%d %H%M")
    steps = [0, 3, 6]

    with (
        patch.object(model, "_download_ifs_channel_grib_to_cache") as mock_download,
        patch("xarray.open_dataarray") as mock_open_dataarray,
    ):
        mock_da = xr.DataArray(
            data=np.random.rand(len(steps), 721, 1440),
            dims=["time", "latitude", "longitude"],
            coords={
                "time": [start_time + datetime.timedelta(hours=s) for s in steps],
                "latitude": np.linspace(90, -90, 721),
                "longitude": np.linspace(-180, 180, 1440, endpoint=False),
            },
        ).roll(longitude=720, roll_coords=True)
        mock_open_dataarray.return_value = mock_da

        dataarray = model.fetch_dataarray(start_time, steps)

        assert dataarray.shape == (len(steps), len(channels), 721, 1440)
        assert list(dataarray.coords.keys()) == ["time", "channel", "lat", "lon"]
        assert (dataarray.coords["channel"].values == channels).all()
        assert (dataarray.coords["lat"].values == model.IFS_LAT).all()
        assert (dataarray.coords["lon"].values == model.IFS_LON).all()


@pytest.mark.parametrize(
    "lead_time, start_time, expected_steps",
    [
        (
            144,
            datetime.datetime.strptime("20230101 0000", "%Y%m%d %H%M"),
            list(range(0, 145, 3)),
        ),
        (
            90,
            datetime.datetime.strptime("20230101 0600", "%Y%m%d %H%M"),
            list(range(0, 91, 3)),
        ),
    ],
)
def test_slice_lead_time_to_steps_valid_lead_time_and_start_time(
    lead_time, start_time, expected_steps
):
    model = IFSModel(channels=[], cache=True, source="aws")
    assert model._slice_lead_time_to_steps(lead_time, start_time) == expected_steps


@pytest.mark.parametrize(
    "lead_time, start_time",
    [
        (
            241,
            datetime.datetime.strptime("20230101 0000", "%Y%m%d %H%M"),
        ),
        (
            91,
            datetime.datetime.strptime("20230101 0600", "%Y%m%d %H%M"),
        ),
    ],
)
def test_slice_lead_time_to_steps_invalid_lead_time_for_00_and_12_start_times(
    lead_time, start_time
):
    model = IFSModel(channels=[], cache=True, source="aws")
    with pytest.raises(ValueError):
        model._slice_lead_time_to_steps(lead_time, start_time)


def test_slice_lead_time_to_steps_valid_lead_time_and_start_time_for_06_and_18_start_times():
    model = IFSModel(channels=[], cache=True, source="aws")
    lead_time = 90
    start_time = datetime.datetime.strptime("20230101 0600", "%Y%m%d %H%M")
    expected_steps = list(range(0, lead_time + 1, 3))
    assert model._slice_lead_time_to_steps(lead_time, start_time) == expected_steps


def test_slice_lead_time_to_steps_invalid_start_time():
    model = IFSModel(channels=[], cache=True, source="aws")
    lead_time = 144
    start_time = datetime.datetime.strptime("20230101 0300", "%Y%m%d %H%M")
    with pytest.raises(ValueError):
        model._slice_lead_time_to_steps(lead_time, start_time)


if __name__ == "__main__":
    pytest.main()
