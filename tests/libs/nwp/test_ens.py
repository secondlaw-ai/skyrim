import pytest
import datetime
from unittest.mock import patch, PropertyMock

from skyrim.libs.nwp import ENS_Vocabulary, ENSModel


# Tests for ENS_Vocabulary class
def test_ens_vocabulary_build_vocab():
    vocab = ENS_Vocabulary.build_vocab()
    assert isinstance(vocab, dict)
    # Check that certain keys are in the vocabulary
    expected_keys = [
        "t2m",
        "u10m",
        "v10m",
        "u100m",
        "v100m",
        "sp",
        "msl",
        "tcwv",
        "tp",
        "u50",
        "u200",
        "u250",
        "u300",
        "u500",
        "u700",
        "u850",
        "u925",
        "u1000",
    ]
    for key in expected_keys:
        assert key in vocab


def test_ens_vocabulary_get_variable():
    ens_vocab = ENS_Vocabulary()
    # Test a regular variable
    key = "t2m"
    ens_id, ens_levtype, ens_level, modifier_func = ens_vocab.get_variable(key)
    assert ens_id == "2t"
    assert ens_levtype == "sfc"
    assert ens_level == ""
    assert modifier_func(1) == 1  # modifier function is identity

    # Test a pressure level variable
    key = "u500"
    ens_id, ens_levtype, ens_level, modifier_func = ens_vocab.get_variable(key)
    assert ens_id == "u"
    assert ens_levtype == "pl"
    assert ens_level == "500"
    assert modifier_func(1) == 1

    # Test 'gh' variable which should have modifier function
    key = "z500"
    ens_id, ens_levtype, ens_level, modifier_func = ens_vocab.get_variable(key)
    assert ens_id == "gh"
    assert ens_levtype == "pl"
    assert ens_level == "500"
    assert modifier_func(1) == 1 * 9.81  # modifier function multiplies by 9.81


def test_ens_vocabulary_contains():
    ens_vocab = ENS_Vocabulary()
    assert "t2m" in ens_vocab
    assert "nonexistent_variable" not in ens_vocab


def test_ens_vocabulary_len():
    ens_vocab = ENS_Vocabulary()
    assert len(ens_vocab) == len(ens_vocab.VOCAB)


# Tests for ENSModel class
def test_ens_model_init_valid_channels():
    channels = ["t2m", "u10m", "v10m"]
    model = ENSModel(channels=channels)
    assert model.channels == channels


def test_ens_model_init_invalid_channels():
    channels = ["t2m", "invalid_channel"]
    with pytest.raises(Exception) as exc_info:
        model = ENSModel(channels=channels)
    assert "Channel invalid_channel does not exist in the vocabulary" in str(
        exc_info.value
    )


def test_ens_model_levelist():
    channels = ["u500", "v500", "t500", "u850"]
    model = ENSModel(channels=channels)
    assert model.levelist == [500, 850]


def test_ens_model_sl_params():
    channels = ["t2m", "u10m", "v10m", "sp"]
    model = ENSModel(channels=channels)
    expected_sl_params = ["2t", "10u", "10v", "sp"]
    assert sorted(model.sl_params) == sorted(expected_sl_params)


def test_ens_model_pl_params():
    channels = ["u500", "v500", "t500", "u850"]
    model = ENSModel(channels=channels)
    expected_pl_params = ["u", "v", "t"]
    assert sorted(model.pl_params) == sorted(expected_pl_params)


def test_ens_model_clear_cache(tmpdir):
    channels = ["t2m", "u10m"]
    model = ENSModel(channels=channels)

    # Patch the 'cache' property to return a mocked cache directory
    with patch.object(ENSModel, "cache", new_callable=PropertyMock) as mock_cache:
        mock_cache.return_value = tmpdir.mkdir("ens_cache")

        dummy_file = mock_cache.return_value.join("dummy_file.txt")
        dummy_file.write("test")
        assert dummy_file.check()

        model.clear_cache()
        assert not dummy_file.check()


def test_ens_model_slice_lead_time_to_steps():
    model = ENSModel(channels=["t2m"])
    # Test for start time at 00 and lead_time within 144 hours
    start_time = datetime.datetime(2021, 1, 1, 0)
    lead_time = 72
    steps = model._slice_lead_time_to_steps(lead_time, start_time)
    assert steps == list(range(0, 73, 3))
    # Test for start time at 12 and lead_time beyond 144 hours
    start_time = datetime.datetime(2021, 1, 1, 12)
    lead_time = 150
    steps = model._slice_lead_time_to_steps(lead_time, start_time)
    expected_steps = list(range(0, 145, 3)) + list(range(150, 151, 6))
    assert steps == expected_steps
    # Test for invalid lead_time
    with pytest.raises(ValueError):
        steps = model._slice_lead_time_to_steps(400, start_time)
    # Test for invalid start_time
    start_time = datetime.datetime(2021, 1, 1, 5)
    with pytest.raises(ValueError):
        steps = model._slice_lead_time_to_steps(72, start_time)


@patch("skyrim.libs.nwp.ens.ENSModel.fetch_dataarray")
def test_ens_model_forecast(mock_fetch_dataarray):
    import xarray as xr
    import numpy as np

    dummy_data = xr.DataArray(
        np.random.rand(1, 2, 3, 4, 5), dims=["number", "time", "channel", "lat", "lon"]
    )
    mock_fetch_dataarray.return_value = dummy_data

    channels = ["t2m"]
    model = ENSModel(channels=channels)
    start_time = datetime.datetime(2021, 1, 1, 0)
    result = model.forecast(start_time=start_time, n_steps=2, step_size=6)
    assert result is dummy_data
    mock_fetch_dataarray.assert_called_once()


@patch("skyrim.libs.nwp.ens.ENSModel.fetch_dataarray")
def test_ens_model_predict(mock_fetch_dataarray):
    import xarray as xr
    import numpy as np

    dummy_data = xr.DataArray(
        np.random.rand(1, 2, 3, 4, 5), dims=["number", "time", "channel", "lat", "lon"]
    )
    mock_fetch_dataarray.return_value = dummy_data

    channels = ["t2m"]
    model = ENSModel(channels=channels)
    result = model.predict(
        date="20220101",
        time="0000",
        lead_time=72,
        save=False,
        save_config={"some_key": "some_value"},
    )
    assert result is dummy_data
    mock_fetch_dataarray.assert_called_once()
