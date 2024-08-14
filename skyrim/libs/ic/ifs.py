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
import cfgrib
from earth2mip.datasets.era5 import METADATA
from earth2mip.initial_conditions import base

LOCAL_CACHE = os.getenv("LOCAL_CACHE") or (os.environ["HOME"] + "/.cache/modulus")


def _get_filename(time: datetime.datetime, lead_time: str, resolution: str = "0p25"):
    """
    Derive file name from AWS Open Data. See more here for naming convention:
    https://confluence.ecmwf.int/display/DAC/ECMWF+open+data%3A+real-time+forecasts+from+IFS+and+AIFS
    """
    if resolution not in {"0p4-beta", "0p25"}:
        raise Exception("Unknown resolution for IFS")

    hour = time.strftime("%Hz")
    folder_name = (
        "oper" if hour in {"00z", "12z"} else "scda"
    )  # scda=short-cut-off hi-res forecast

    date_format = f"%Y%m%d/%Hz/ifs/{resolution}/{folder_name}/%Y%m%d%H%M%S-{lead_time}-{folder_name}-fc.grib2"
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
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        bucket_name = "ecmwf-forecasts"
        file_key = path.replace(f"s3://{bucket_name}/", "")
        logger.debug(f"Downloading IFS initial condition... file_key={file_key}")
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
    root = "s3://ecmwf-forecasts/"
    path = root + _get_filename(time, "0h")
    local_path = download_cached(path)
    logger.debug(f"Local path: {local_path}")

    # NOTE: cfgrib.open_datasets returns a list of xarray datasets
    # this is a very hacky workaround, check this out
    loaded = cfgrib.open_datasets(local_path)
    if len(loaded) > 1:
        logger.debug(f"Multiple datasets loaded from ECMWF repo.")
        dataset_0h = xarray.merge(
            [
                (
                    l.drop_vars("heightAboveGround")
                    if "heightAboveGround" in l.coords
                    else (
                        l.drop_vars("depthBelowLandLayer")
                        if "depthBelowLandLayer" in l.coords
                        else l
                    )
                )
                for l in loaded
            ]
        )
    if time.date() < datetime.datetime(2024, 3, 6).date():
        raise Exception("IFS HRES 0.25 only supported after 6/3/2024")
    channel_data = [
        _get_channel(
            c,
            q=dataset_0h.q,  # pangu
            w=dataset_0h.w,  # graphcast
            u10m=dataset_0h.u10,
            v10m=dataset_0h.v10,
            u100m=dataset_0h.u100,
            v100m=dataset_0h.v100,
            sp=dataset_0h.sp,
            t2m=dataset_0h.t2m,
            msl=dataset_0h.msl,
            tcwv=dataset_0h.tcwv,
            t=dataset_0h.t,
            u=dataset_0h.u,
            v=dataset_0h.v,
            r=dataset_0h.r,
            z=dataset_0h.gh * 9.807,
        )
        for c in channels
    ]
    array = np.stack([d for d in channel_data], axis=0)
    darray = xarray.DataArray(
        array,
        dims=["channel", "lat", "lon"],
        coords={
            "channel": channels,
            "lat": dataset_0h.latitude.values,  # NOTE: double check this line
            "lon": dataset_0h.longitude.values,
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
        metadata = json.loads(METADATA.read_text())
        lat = np.array(metadata["coords"]["lat"])
        lon = np.array(metadata["coords"]["lon"])
        ds = ds.roll(lon=len(ds.lon) // 2, roll_coords=True)
        ds["lon"] = ds.lon.where(ds.lon >= 0, ds.lon + 360)
        assert min(ds.lon) >= 0, min(ds.lon)  # noqa
        r = ds.interp(lat=lat, lon=lon, kwargs={"fill_value": "extrapolate"})
        return r
