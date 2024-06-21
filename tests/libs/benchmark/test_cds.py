import pytest
from unittest.mock import patch
import datetime
import xarray as xr
import numpy as np
from pathlib import Path

from skyrim.libs.benchmark import CDS, CDS_Vocabulary
from skyrim.common import LOCAL_CACHE


def test_cds_vocabulary_build_vocab():
    vocab = CDS_Vocabulary.build_vocab()
    assert isinstance(vocab, dict)
    assert "u10m" in vocab
    assert vocab["u10m"] == "reanalysis-era5-single-levels::10m_u_component_of_wind::"


def test_cds_vocabulary_get():
    cds_id, cds_levtype, cds_level = CDS_Vocabulary.get("u10m")
    assert cds_levtype == "reanalysis-era5-single-levels"
    assert cds_id == "10m_u_component_of_wind"
    assert cds_level == ""


def test_cds_vocabulary_contains():
    assert "u10m" in CDS_Vocabulary.VOCAB
    assert "nonexistent" not in CDS_Vocabulary.VOCAB


# Tests for CDS
@patch("skyrim.libs.benchmark.cds.cdsapi.Client")
def test_cds_initialization(mock_cds_client):
    channels = ["u10m", "v10m"]
    cds = CDS(channels=channels)
    assert cds.channels == channels
    assert cds._cache == True
    assert cds.cds_client == mock_cds_client.return_value


def test_format_to_datetime():
    time_list = CDS(channels=["u10m"])._format_to_datetime(2023, 1, [1, 2], 0)
    assert time_list == [
        datetime.datetime(2023, 1, 1, 0, 0),
        datetime.datetime(2023, 1, 2, 0, 0),
    ]

def test_cds_assure_channels_exist():
    cds = CDS(channels=["u10m", "v10m"])
    with pytest.raises(AssertionError):
        cds.assure_channels_exist(["u10m", "nonexistent"])


@patch("skyrim.libs.benchmark.cds.cdsapi.Client")
def test_cds_cache(mock_cds_client):
    cds = CDS(channels=["u10m"])
    cache_location = Path(LOCAL_CACHE) / "cds"
    assert cds.cache == str(cache_location)


if __name__ == "__main__":
    pytest.main()
