#
# Example invocation:
# from skyrim.libs.benchmark.cds import CDS
# cds = CDS(channels=[])
# fetched_data = cds(start_time, lead_time)
#
import os
import shutil
import hashlib
import datetime
from typing import Union, List, Optional, Literal
import json
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import numpy as np
import xarray as xr
import cdsapi
from ...common import LOCAL_CACHE, OUTPUT_DIR


class CDS_Vocabulary:
    """
    Vocabulary for CDS API parameters.

    Adapted from earth2studio:
    https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/lexicon/cds.py

    CDS specified <dataset>::<Variable ID>::<Pressure Level>

    Additional resources:
    single levels
        https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview
    pressure levels
        https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview

    https://codes.ecmwf.int/grib/param-db/?filter=grib2
    """

    @staticmethod
    def build_vocab():
        return {
            "u10m": "reanalysis-era5-single-levels::10m_u_component_of_wind::",
            "v10m": "reanalysis-era5-single-levels::10m_v_component_of_wind::",
            "u100m": "reanalysis-era5-single-levels::100m_u_component_of_wind::",
            "v100m": "reanalysis-era5-single-levels::100m_v_component_of_wind::",
            "t2m": "reanalysis-era5-single-levels::2m_temperature::",
            "sp": "reanalysis-era5-single-levels::surface_pressure::",
            "msl": "reanalysis-era5-single-levels::mean_sea_level_pressure::",
            "tcwv": "reanalysis-era5-single-levels::total_column_water_vapour::",
            "tp": "reanalysis-era5-single-levels::total_precipitation::",
            "fg10m": "reanalysis-era5-single-levels::10m_wind_gust_since_previous_post_processing::",
            "u50": "reanalysis-era5-pressure-levels::u_component_of_wind::50",
            "u100": "reanalysis-era5-pressure-levels::u_component_of_wind::100",
            "u150": "reanalysis-era5-pressure-levels::u_component_of_wind::150",
            "u200": "reanalysis-era5-pressure-levels::u_component_of_wind::200",
            "u250": "reanalysis-era5-pressure-levels::u_component_of_wind::250",
            "u300": "reanalysis-era5-pressure-levels::u_component_of_wind::300",
            "u400": "reanalysis-era5-pressure-levels::u_component_of_wind::400",
            "u500": "reanalysis-era5-pressure-levels::u_component_of_wind::500",
            "u600": "reanalysis-era5-pressure-levels::u_component_of_wind::600",
            "u700": "reanalysis-era5-pressure-levels::u_component_of_wind::700",
            "u850": "reanalysis-era5-pressure-levels::u_component_of_wind::850",
            "u925": "reanalysis-era5-pressure-levels::u_component_of_wind::925",
            "u1000": "reanalysis-era5-pressure-levels::u_component_of_wind::1000",
            "v50": "reanalysis-era5-pressure-levels::v_component_of_wind::50",
            "v100": "reanalysis-era5-pressure-levels::v_component_of_wind::100",
            "v150": "reanalysis-era5-pressure-levels::v_component_of_wind::150",
            "v200": "reanalysis-era5-pressure-levels::v_component_of_wind::200",
            "v250": "reanalysis-era5-pressure-levels::v_component_of_wind::250",
            "v300": "reanalysis-era5-pressure-levels::v_component_of_wind::300",
            "v400": "reanalysis-era5-pressure-levels::v_component_of_wind::400",
            "v500": "reanalysis-era5-pressure-levels::v_component_of_wind::500",
            "v600": "reanalysis-era5-pressure-levels::v_component_of_wind::600",
            "v700": "reanalysis-era5-pressure-levels::v_component_of_wind::700",
            "v850": "reanalysis-era5-pressure-levels::v_component_of_wind::850",
            "v925": "reanalysis-era5-pressure-levels::v_component_of_wind::925",
            "v1000": "reanalysis-era5-pressure-levels::v_component_of_wind::1000",
            "z50": "reanalysis-era5-pressure-levels::geopotential::50",
            "z100": "reanalysis-era5-pressure-levels::geopotential::100",
            "z150": "reanalysis-era5-pressure-levels::geopotential::150",
            "z200": "reanalysis-era5-pressure-levels::geopotential::200",
            "z250": "reanalysis-era5-pressure-levels::geopotential::250",
            "z300": "reanalysis-era5-pressure-levels::geopotential::300",
            "z400": "reanalysis-era5-pressure-levels::geopotential::400",
            "z500": "reanalysis-era5-pressure-levels::geopotential::500",
            "z600": "reanalysis-era5-pressure-levels::geopotential::600",
            "z700": "reanalysis-era5-pressure-levels::geopotential::700",
            "z850": "reanalysis-era5-pressure-levels::geopotential::850",
            "z925": "reanalysis-era5-pressure-levels::geopotential::925",
            "z1000": "reanalysis-era5-pressure-levels::geopotential::1000",
            "t50": "reanalysis-era5-pressure-levels::temperature::50",
            "t100": "reanalysis-era5-pressure-levels::temperature::100",
            "t150": "reanalysis-era5-pressure-levels::temperature::150",
            "t200": "reanalysis-era5-pressure-levels::temperature::200",
            "t250": "reanalysis-era5-pressure-levels::temperature::250",
            "t300": "reanalysis-era5-pressure-levels::temperature::300",
            "t400": "reanalysis-era5-pressure-levels::temperature::400",
            "t500": "reanalysis-era5-pressure-levels::temperature::500",
            "t600": "reanalysis-era5-pressure-levels::temperature::600",
            "t700": "reanalysis-era5-pressure-levels::temperature::700",
            "t850": "reanalysis-era5-pressure-levels::temperature::850",
            "t925": "reanalysis-era5-pressure-levels::temperature::925",
            "t1000": "reanalysis-era5-pressure-levels::temperature::1000",
            "r50": "reanalysis-era5-pressure-levels::relative_humidity::50",
            "r100": "reanalysis-era5-pressure-levels::relative_humidity::100",
            "r150": "reanalysis-era5-pressure-levels::relative_humidity::150",
            "r200": "reanalysis-era5-pressure-levels::relative_humidity::200",
            "r250": "reanalysis-era5-pressure-levels::relative_humidity::250",
            "r300": "reanalysis-era5-pressure-levels::relative_humidity::300",
            "r400": "reanalysis-era5-pressure-levels::relative_humidity::400",
            "r500": "reanalysis-era5-pressure-levels::relative_humidity::500",
            "r600": "reanalysis-era5-pressure-levels::relative_humidity::600",
            "r700": "reanalysis-era5-pressure-levels::relative_humidity::700",
            "r850": "reanalysis-era5-pressure-levels::relative_humidity::850",
            "r925": "reanalysis-era5-pressure-levels::relative_humidity::925",
            "r1000": "reanalysis-era5-pressure-levels::relative_humidity::1000",
            "q50": "reanalysis-era5-pressure-levels::specific_humidity::50",
            "q100": "reanalysis-era5-pressure-levels::specific_humidity::100",
            "q150": "reanalysis-era5-pressure-levels::specific_humidity::150",
            "q200": "reanalysis-era5-pressure-levels::specific_humidity::200",
            "q250": "reanalysis-era5-pressure-levels::specific_humidity::250",
            "q300": "reanalysis-era5-pressure-levels::specific_humidity::300",
            "q400": "reanalysis-era5-pressure-levels::specific_humidity::400",
            "q500": "reanalysis-era5-pressure-levels::specific_humidity::500",
            "q600": "reanalysis-era5-pressure-levels::specific_humidity::600",
            "q700": "reanalysis-era5-pressure-levels::specific_humidity::700",
            "q850": "reanalysis-era5-pressure-levels::specific_humidity::850",
            "q925": "reanalysis-era5-pressure-levels::specific_humidity::925",
            "q1000": "reanalysis-era5-pressure-levels::specific_humidity::1000",
        }

    VOCAB = build_vocab()

    def __getitem(self, key):
        """Allow dictionary-like access (e.g., CDS_Vocabulary['u100'])"""
        return self.VOCAB[key]

    def __contains__(self, key):
        """Allow membership testing (e.g., 'u100' in CDS_Vocabulary)"""
        return key in self.VOCAB

    @classmethod
    def get(cls, channel: str) -> str:
        """Get CDS parameter ID, level type,  and level for a given channel."""
        cds_key = cls.VOCAB[channel]
        cds_levtype, cds_id, cds_level = cds_key.split("::")
        return cds_id, cds_levtype, cds_level


class CDS:
    CDS_LAT = np.linspace(90, -90, 721)
    CDS_LON = np.linspace(0, 360, 1440, endpoint=False)

    def __init__(self, channels: list[str], cache: bool = True):
        self._cache = cache
        self.channels = channels
        self.assure_channels_exist(channels)
        self.cached_files = []
        self.cds_client = cdsapi.Client()
        logger.info(f"CDS client initialized with channels: {channels}")
        logger.debug(f"CDS cache location: {self.cache}")

    def __call__(
        self,
        year: int | list[int],
        month: int | list[int],
        day: int | list[int],
        time: int | list[int],
    ) -> xr.DataArray:

        t = self._ymdt_to_datetime(year, month, day, time)
        assert isinstance(t, list), "UPSY."
        assert isinstance(t[0], datetime.datetime), "UPSY."
        cds_dataarray = self.fetch_cds_dataarray(t)
        return cds_dataarray

    def assure_channels_exist(self, channels: list[str]):
        for channel in channels:
            assert (
                channel in CDS_Vocabulary.VOCAB
            ), f"Channel {channel} not found in CDS vocabulary."

    def assure_cdsapirc_exists(self):
        raise NotImplementedError("This method is not implemented yet.")

    @staticmethod
    def list_available_channels():
        return list(CDS_Vocabulary.VOCAB.keys())

    @property
    def in_channel_names(self):
        return list(CDS_Vocabulary.VOCAB.keys())

    @property
    def out_channel_names(self):
        return self.channels

    @property
    def cache(self) -> str:
        cache_location = os.path.join(LOCAL_CACHE, "cds")
        if not self._cache:
            cache_location = os.path.join(LOCAL_CACHE, "cds", "tmp")
            logger.debug(f"Using temporary cache location: {cache_location}")

        if not os.path.exists(cache_location):
            os.makedirs(cache_location)
            logger.info(f"Created cache directory at {cache_location}")

        return cache_location

    def clear_cached_files(self):
        """Clears the cached files from the current session."""
        for file_path in self.cached_files:
            if os.path.exists(file_path):
                logger.info(f"Deleting cached file: {file_path}")
                os.remove(file_path)
        self.cached_files = []

    def clear_cache(self):
        """Clears the entire cache directory."""
        cache_dir = self.cache
        if os.path.exists(cache_dir):
            logger.info(f"Clearing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
        else:
            logger.debug(f"Cache directory not found: {cache_dir}")

    def fetch_cds_dataarray(
        self,
        time: datetime.datetime | list[datetime.datetime],
    ):
        """Retrives CDS data array for given date time by fetching channel grib files
        using the cdsapi package and combining grib files into a single data array.

        Parameters
        ----------
        time : datetime | list[datetime]
            Date time for which to fetch the data array

        Returns
        -------
        xr.DataArray
            CDS data array for given datetime.datetime

        """
        if isinstance(time, datetime.datetime):
            time = [time]

        cds_dataarray = xr.DataArray(
            data=np.empty(
                (len(time), len(self.channels), len(self.CDS_LAT), len(self.CDS_LON))
            ),
            dims=["time", "channel", "lat", "lon"],
            coords={
                "time": time,
                "channel": self.channels,
                "lat": self.CDS_LAT,
                "lon": self.CDS_LON,
            },
        )
        logger.debug(f"Initialized cds_dataarray with shape: {cds_dataarray.shape}")
        for i, channel in enumerate(self.channels):
            cds_id, cds_levtype, cds_level = CDS_Vocabulary.get(channel)
            cache_path = self._download_cds_grib_to_cache(
                time=time, variable=cds_id, levtype=cds_levtype, level=cds_level
            )
            da = xr.open_dataarray(
                cache_path, engine="cfgrib", backend_kwargs={"indexpath": ""}
            )
            logger.debug(f"Fetched channel: {channel}\n")
            logger.debug(f"cache path: {cache_path}\n")
            logger.debug(f"Fetched dataarray's shape: {da.shape}")
            # NOTE: During fetching the request body is built using the time, variable, levtype, and level
            #       the enumeration could change the size of the time dimension, therefore the dataarray
            #       should be sliced across the time dimension.
            # Fix this if turns out to be a problem.
            da = da.sel(time=time)
            cds_dataarray[:, i, :, :] = da.values

        return cds_dataarray

    def _download_cds_grib_to_cache(
        self,
        time: list[datetime.datetime],
        variable: str,
        levtype: str,
        level: str,
    ) -> str:
        """
        Downloads a CDS GRIB file to the cache directory.
        """

        sha = hashlib.sha256(f"{variable}_{levtype}_{level}_{time}".encode())
        filename = sha.hexdigest()
        cache_path = os.path.join(self.cache, filename)

        logger.debug(
            f"Request time: {time}\n"
            f"len(time): {len(time)}\n"
            f"variable: {variable}\n"
            f"levtype: {levtype}\n"
        )
        logger.debug(f"Cache path: {cache_path}")

        if not os.path.exists(cache_path):
            request_body = self._build_request_body(
                time=time, variable=variable, levtype=levtype, level=level
            )
            self.cds_client.retrieve(
                name=levtype, request=request_body, target=cache_path
            )
            self.cached_files.append(cache_path)
        else:
            logger.debug(f"File already exists in cache: {cache_path}")

        return cache_path

    def _build_request_body(
        self,
        time: List[datetime.datetime],
        variable: str,
        levtype: str,
        level: str,
    ):
        """Builds the request body for the CDS API using the provided year, month, day, and time."""

        year = sorted(list(set(str(t.year) for t in time)))
        month = sorted(list(set(f"{t.month:02d}" for t in time)))
        day = sorted(list(set(f"{t.day:02d}" for t in time)))
        time = sorted(list(set(t.strftime("%H:%M") for t in time)))
        logger.debug(
            f"Building requiest body \nYear: {year}\nMonth: {month}\nDay: {day}\nTime: {time} \nVariable: {variable}"
        )
        # build the request body
        request_body = {
            "variable": variable,
            "product_type": "reanalysis",
            "year": year,
            "month": month,
            "day": day,
            "time": time,
            "format": "grib",
        }
        if levtype == "reanalysis-era5-pressure-levels":
            request_body["pressure_level"] = level
        logger.debug(f"Request body:\n{json.dumps(request_body, indent=4)}")
        return request_body

    @staticmethod
    def _ymdt_to_datetime(
        year: Union[int, List[int]],
        month: Union[int, List[int]],
        day: Union[int, List[int]],
        time: Union[int, List[int]],
    ) -> List[datetime.datetime]:
        """Converts year, month, day, and time to a list of datetime.datetime objects."""
        # NOTE: could be placed in common.py

        if isinstance(year, int):
            year = [year]
        if isinstance(month, int):
            month = [month]
        if isinstance(day, int):
            day = [day]
        if isinstance(time, int):
            time = [time]

        return [
            datetime.datetime(y, m, d, t, 0, 0, 0)
            for y in year
            for m in month
            for d in day
            for t in time
        ]

    @staticmethod
    def _datetime_to_ymdt(time: List[datetime.datetime]):
        """Converts a list of datetime.datetime objects to year, month, day, and time components."""
        year = sorted(list(set(t.year for t in time)))
        month = sorted(list(set(t.month for t in time)))
        day = sorted(list(set(t.day for t in time)))
        hour = sorted(list(set(t.hour for t in time)))

        return year, month, day, hour

    def benchmark(
        self,
        date: str,  # YYYMMDD, e.g. 20180101
        time: str = "0000",  # HHMM, e.g. 0300, 1400, etc
        lead_time: int = 24,  # in hours
        time_step: int = 6,  # in hours
    ) -> xr.DataArray:
        """
        Fetches cds data array for a given date and time for bechmarking purposes.
        NOTE: the inferface is similar to skyrim.skyrim.predict for quick testing.
        """
        start_time = datetime.datetime.strptime(date + time, "%Y%m%d%H%M")
        timestamps = [
            start_time + datetime.timedelta(hours=i)
            for i in range(0, lead_time + 1, time_step)
        ]
        logger.debug(
            f"Fetching CDS data array for benchmarking for {len(timestamps)} timestamps."
        )
        return self.fetch_cds_dataarray(timestamps)

    def create_dataset(
        self,
        date: str,  # YYYMMDD, e.g. 20180101
        time: str = "0000",  # HHMM, e.g. 0300, 1400, etc
        lead_time: int = 24,  # in hours
        time_step: int = 6,  # in hours
        output_dir: str = OUTPUT_DIR,
        batching: Literal["daily", "monthly"] = "daily",
    ):
        """
        Create a dataset by generating and batching timestamps, then saving to NetCDF files.

        Parameters
        ----------
        date : str
            The start date in YYYYMMDD format.
        time : str, optional
            The start time in HHMM format (default is "0000").
        lead_time : int, optional
            The lead time in hours (default is 24).
        time_step : int, optional
            The time step in hours (default is 6).
        output_dir : str, optional
            The directory to save the output files (default is OUTPUT_DIR).
        batching : {'daily', 'monthly'}, optional
            The batching method, either "daily" or "monthly" (default is "daily").

        Returns
        -------
        str
        The output directory containing the created dataset.
        """

        # create daily slices of data for the given date
        # -- output_dir f"{start_time}__LT{lead_time}h__TS{time_step}h"
        #   -- metadata.json
        #   -- 20240101T0000_20240101T1800.nc
        #   -- 20240101T0000_20240101T1800.nc
        #   -- ...

        start_time = datetime.datetime.strptime(date + time, "%Y%m%d%H%M")
        logger.debug(
            f"Creating dataset for \n"
            f"start_time: {start_time}\n"
            f"lead time {lead_time}h\n"
            f"time step {time_step}h"
        )

        output_dir = os.path.join(
            output_dir, f"{date}T{time}__LT{lead_time}h__TS{time_step}h__B{batching}"
        )
        logger.debug(f"Output directory: {output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

        metadata = {
            "channels": self.channels,
            "date": date,
            "time": time,
            "lead_time": lead_time,
            "time_step": time_step,
            "batching": batching,
        }

        # generate time stamps
        timestamps = [
            start_time + datetime.timedelta(hours=i)
            for i in range(0, lead_time + 1, time_step)
        ]

        # batch timestamps into daily or monthly slices
        if batching == "monthly":
            timestamps_batched = self._batch_timestamps_monthly(timestamps)
        elif batching == "daily":
            timestamps_batched = self._batch_timestamps_daily(timestamps)
        else:
            raise ValueError(f"Invalid batching option: {batching}")

        logger.info(f"{len(timestamps_batched)} files will be created.")
        logger.info(f"Output directory: {output_dir}")
        for i, ts_batch in enumerate(timestamps_batched):
            start_dt, end_dt = ts_batch[0], ts_batch[-1]
            start_dt_str = start_dt.strftime("%Y%m%dT%H%M")
            end_dt_str = end_dt.strftime("%Y%m%dT%H%M")
            filename = os.path.join(output_dir, f"{start_dt_str}__{end_dt_str}.nc")
            da = self.fetch_cds_dataarray(ts_batch)
            da.to_netcdf(filename)
            logger.info(f"Saved file: {filename}")
            self.clear_cached_files()

        # save metadata
        metadata_filename = os.path.join(output_dir, "metadata.json")
        logger.debug(f"Saving metadata to {metadata_filename}")
        with open(metadata_filename, "w") as f:
            json.dump(metadata, f)
        return output_dir

    def _batch_timestamps_monthly(
        self,
        timestamps: List[datetime.datetime],
    ) -> List[List[datetime.datetime]]:
        """
        Group timestamps into monthly slices, considering both year and month.
        """
        unique_year_months = sorted(list(set((t.year, t.month) for t in timestamps)))
        print(unique_year_months)
        monthly_slices = []
        for year, month in unique_year_months:
            monthly_slice = [
                t for t in timestamps if t.year == year and t.month == month
            ]
            monthly_slices.append(monthly_slice)
        return monthly_slices

    def _batch_timestamps_daily(
        self,
        timestamps: List[datetime.datetime],
    ) -> List[List[datetime.datetime]]:
        """
        Group timestamps into daily slices.
        """
        unique_dates = sorted(list(set([t.date() for t in timestamps])))
        daily_slices = []
        for date in unique_dates:
            daily_slice = [t for t in timestamps if t.date() == date]
            daily_slices.append(daily_slice)
        return daily_slices

    def create_regional_dataset(
        self,
        start_time: datetime.datetime,
        lead_time: int,
        lat: slice,
        lon: slice,
        time_step: int = 1,
        **kwargs,
    ):
        pass


if __name__ == "__main__":
    start_time = datetime.datetime(2024, 1, 1, 12, 0, 0, 0)
    time = [start_time + datetime.timedelta(days=i) for i in range(0, 3)]
    cds = CDS(channels=["u10m", "u1000"])
    cds.clear_cache()
    da = cds(time)
    print(da.shape)
