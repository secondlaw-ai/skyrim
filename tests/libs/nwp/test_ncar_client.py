import numpy as np
import pytest
from datetime import datetime
from skyrim.utils import fast_fetch
from skyrim.libs.nwp.ifs import IFSModel, GRIDS
from skyrim.libs.nwp.ncar_client import (
    NCARClient,
    ncar_sfc_vars,
    ncar_pl_vars,
    SKYRIM_IFS_VARS,
)

GERMANY_LAT = slice(55, 47)
GERMANY_LON = slice(5, 16)

client = NCARClient()


def test_lists_files():
    files = client.list_files("pl", start_time=datetime(2024, 9, 1, 0, 0))
    assert len(files) == len(ncar_pl_vars)
    assert files[0].endswith(".nc")
    files = client.list_files("sfc", start_time=datetime(2024, 9, 1, 0, 0))
    assert len(files) == len(ncar_sfc_vars)
    assert files[0].endswith(".nc")
    files = client.list_files(
        "pl", start_time=datetime(2024, 9, 1, 0, 0), filetype="grb"
    )
    assert len(files) > 0
    files = client.list_files(
        "sfc", start_time=datetime(2024, 9, 1, 0, 0), filetype="grb"
    )
    assert len(files) > 0


class TestValues:
    da = client.fetch(datetime(2024, 9, 1, 0, 0))
    regridded_da = client.regrid(da, use_cached_weights=True)
    ifs_25 = (
        IFSModel(
            channels=["t2m", "u10m", "z500"],
            cache=True,
            source="aws",
            resolution="0p25",
        )
        .forecast(datetime(2024, 9, 1, 0, 0), 6)
        .isel(time=0)
    )

    def test_channel_values(self):
        assert set(self.da.channel.values.tolist()) == SKYRIM_IFS_VARS
        assert set(self.regridded_da.channel.values.tolist()) == SKYRIM_IFS_VARS

    def test_lat_lon_values(self):
        LAT, LON = GRIDS["0p25"]
        assert np.allclose(self.regridded_da.lat.values, LAT)
        assert np.allclose(self.regridded_da.lon.values, LON)

    def test_closeness_with_ifs(self):
        # closeness with IFS 0.25 resolution:
        ifs_25_DE = self.ifs_25.sel(lat=GERMANY_LAT, lon=GERMANY_LON)
        regridded_da_DE = self.regridded_da.sel(lat=GERMANY_LAT, lon=GERMANY_LON)
        # %1.5 difference in values
        x, y = (
            regridded_da_DE.sel(channel="t2m").values,
            ifs_25_DE.sel(channel="t2m").values,
        )
        assert np.allclose(x, y, rtol=0.015)
        x, y = (
            regridded_da_DE.sel(channel="u10m").values,
            ifs_25_DE.sel(channel="u10m").values,
        )
        assert np.allclose(x.mean(), y.mean(), rtol=0.015)

    @pytest.mark.skip(
        reason="in case different regridding method is used, conversative preserves stats the most"
    )
    def test_regridding_stats(self):
        da_DE = self.da.sel(lat=GERMANY_LAT, lon=GERMANY_LON)
        regridded_da_DE = self.regridded_da.sel(lat=GERMANY_LAT, lon=GERMANY_LON)
        assert np.allclose(
            da_DE.sel(channel="t2m").values.mean(),
            regridded_da_DE.sel(channel="t2m").values.mean(),
            rtol=0.001,
        )
        assert np.allclose(
            da_DE.sel(channel="u10m").values.mean(),
            regridded_da_DE.sel(channel="u10m").values.mean(),
            rtol=0.001,
        )
