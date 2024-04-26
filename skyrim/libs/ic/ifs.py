import dataclasses
import datetime
import json
import earth2mip.grid
import numpy as np
import xarray
import hashlib
import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from loguru import logger
from typing import List
from earth2mip.datasets.era5 import METADATA
from earth2mip.initial_conditions import base

LOCAL_CACHE = os.getenv("LOCAL_CACHE") or (os.environ["HOME"] + "/.cache/modulus")

def _get_filename(time: datetime.datetime, lead_time: str):
    date_format = f"%Y%m%d/%Hz/0p4-beta/oper/%Y%m%d%H%M%S-{lead_time}-oper-fc.grib2"
    return time.strftime(date_format)


def _get_channel(c: str, **kwargs) -> xarray.DataArray:
    """

    Parameters:
    -----------
    c: channel id
    **kwargs: variables in ecmwf data
    """
    # handle 2d inputs
    if c in kwargs:
        return kwargs[c]
    else:
        varcode, pressure_level = c[0], int(c[1:])
        return kwargs[varcode].interp(isobaricInhPa=pressure_level)



def download_cached(path):
    cached_path = get_cache_file_path(path)
    if not os.path.exists(cached_path):
        logger.debug('Downloading IFS initial condition...')
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        bucket_name = 'ecmwf-forecasts' 
        file_key = path.replace(f's3://{bucket_name}/', '')
        s3.download_file(bucket_name, file_key, cached_path)
        return cached_path
    return cached_path


def get_cache_file_path(path, local_cache_path: str = LOCAL_CACHE):
    sha = hashlib.sha256(path.encode())
    filename = sha.hexdigest()
    try:
        os.makedirs(local_cache_path, exist_ok=True)
    except PermissionError as error:
        logger.error(
            "Failed to create cache folder, check permissions or set a cache"
            + " location using the LOCAL_CACHE environment variable"
        )
        raise error
    except OSError as error:
        logger.error(
            "Failed to create cache folder, set a cache"
            + " location using the LOCAL_CACHE environment variable"
        )
        raise error
    return os.path.join(local_cache_path, filename)


def get(time: datetime.datetime, channels: List[str]):
    root = 's3://ecmwf-forecasts/'
    path = root + _get_filename(time, "0h")
    path = path.replace('00z/', '00z/ifs/')
    local_path = download_cached(path)
    dataset_0h = xarray.open_dataset(local_path, engine="cfgrib")
    # get t2m and other things from 12 hour forecast initialized 12 hours before
    # The HRES is only initialized every 12 hours
    path = root + _get_filename(time - datetime.timedelta(hours=12), "12h")
    local_path = download_cached(path)
    forecast_12h = xarray.open_dataset(local_path, engine="cfgrib")

    channel_data = [
        _get_channel(
            c,
            u10m=dataset_0h.u10,
            v10m=dataset_0h.v10,
            u100m=dataset_0h.u10,
            v100m=dataset_0h.v10,
            sp=dataset_0h.sp,
            t2m=forecast_12h.t2m,
            msl=forecast_12h.msl,
            tcwv=forecast_12h.tciwv,
            t=dataset_0h.t,
            u=dataset_0h.u,
            v=dataset_0h.v,
            r=dataset_0h.r,
            z=dataset_0h.gh * 9.81,
        )
        for c in channels
    ]

    array = np.stack([d for d in channel_data], axis=0)
    darray = xarray.DataArray(
        array,
        dims=["channel", "lat", "lon"],
        coords={
            "channel": channels,
            "lon": dataset_0h.longitude.values,
            "lat": dataset_0h.latitude.values,
            "time": time,
        },
    )
    return darray


@dataclasses.dataclass
class DataSource(base.DataSource):
    def __init__(self, channel_names: List[str]):
        self._channel_names = channel_names

    @property
    def channel_names(self) -> List[str]:
        return self._channel_names

    @property
    def grid(self) -> earth2mip.grid.LatLonGrid:
        return earth2mip.grid.equiangular_lat_lon_grid(721, 1440)

    def __getitem__(self, time: datetime.datetime) -> np.ndarray:
        ds = get(time, self.channel_names)
        ds = ds.expand_dims("time", axis=0)
        # move to earth2mip.channels

        # TODO refactor interpolation to another place
        metadata = json.loads(METADATA.read_text())
        lat = np.array(metadata["coords"]["lat"])
        lon = np.array(metadata["coords"]["lon"])
        ds = ds.roll(lon=len(ds.lon) // 2, roll_coords=True)
        ds["lon"] = ds.lon.where(ds.lon >= 0, ds.lon + 360)
        assert min(ds.lon) >= 0, min(ds.lon)  # noqa
        return ds.interp(lat=lat, lon=lon, kwargs={"fill_value": "extrapolate"})
