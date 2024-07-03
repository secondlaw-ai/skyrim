import argparse
import datetime
import hashlib
import shutil
import os
from typing import Literal
from loguru import logger
import xarray as xr
import numpy as np
import ecmwf.opendata
from tqdm import tqdm
import boto3
import botocore

from ...common import LOCAL_CACHE, save_forecast
from ...utils import ensure_ecmwf_loaded

# example invocation:
# python -m skyrim.libs.nwp.ifs --date 20230101 --time 1200 --lead_time 36


# skyrim to ifs mapping
class IFS_Vocabulary:
    """
    Vocabulary for IFS model.

    NOTE:
    >> list(IFS_Vocabulary.VOCAB.keys()).__len__()
    >> 87

    Adapted from (huge shout out to NVIDIA/earth2studio devs):
    https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/data/ifs.py

    """

    @staticmethod
    def build_vocab() -> dict[str, str]:
        """Create HRES vocab dictionary"""
        sfc_variables = {
            "u10m": "10u::sfc::",
            "v10m": "10v::sfc::",
            "u100m": "100u::sfc::",
            "v100m": "100v::sfc::",
            "t2m": "2t::sfc::",
            "sp": "sp::sfc::",
            "msl": "msl::sfc::",
            "tcwv": "tcwv::sfc::",
            "tp": "tp::sfc::",
        }
        prs_levels = [
            50,
            100,
            150,
            200,
            250,
            300,
            400,
            500,
            600,
            700,
            850,
            925,
            1000,
        ]
        prs_names = ["u", "v", "gh", "t", "r", "q"]
        skyrim_id = ["u", "v", "z", "t", "r", "q"]
        prs_variables = {}
        for id, variable in zip(skyrim_id, prs_names):
            for level in prs_levels:
                prs_variables[f"{id}{level:d}"] = f"{variable}::pl::{level}"

        return {**sfc_variables, **prs_variables}

    VOCAB = build_vocab()

    def __getitem__(self, key):
        """Allow dictionary-like access (e.g., IFS_Vocabulary['u100'])"""
        return self.VOCAB[key]

    def __contains__(self, key):
        """Allow membership testing (e.g., 'u100' in IFS_Vocabulary)"""
        return key in self.VOCAB

    @classmethod
    def get(cls, channel: str) -> str:
        ifs_key = cls.VOCAB[channel]
        ifs_id, ifs_levtype, ifs_level = ifs_key.split("::")

        if ifs_id == "gh":
            modifier_func = lambda x: x * 9.81
            return ifs_id, ifs_levtype, ifs_level, modifier_func
        return ifs_id, ifs_levtype, ifs_level, lambda x: x


# adapted from https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/data/ifs.py
class IFSModel:
    """
    Additional resources:
        Known IFS forecasting issues:
        https://confluence.ecmwf.int/display/FCST/Known+IFS+forecasting+issues
    """

    IFS_LAT = np.linspace(90, -90, 721)
    IFS_LON = np.linspace(0, 360, 1440, endpoint=False)
    IFS_BUCKET_NAME = "ecmwf-forecasts"

    def __init__(
        self,
        channels: list[str],
        cache: bool = True,
        source: Literal["aws", "ecmwf", "azure"] = "aws",
    ):
        self._cache = cache
        self.source = source
        self.client = ecmwf.opendata.Client(source=source)
        self.model_name = "HRES"
        self.cached_files = []

        self.assure_channels_exist(channels)
        self.channels = channels
        ensure_ecmwf_loaded()
        logger.info(f"IFS model initialized with channels: {channels}")
        logger.debug(f"IFS cache location: {self.cache}")

    def assure_channels_exist(self, channels):
        for channel in channels:
            assert channel in IFS_Vocabulary.VOCAB.keys()

    @staticmethod
    def list_available_channels():
        return list(IFS_Vocabulary.VOCAB.keys())

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(LOCAL_CACHE, "ifs")
        if not self._cache:
            cache_location = os.path.join(LOCAL_CACHE, "ifs", "tmp")
        if not os.path.exists(cache_location):
            os.makedirs(cache_location, exist_ok=True)
        return cache_location

    @property
    def time_step(self):
        # TODO: implement time step similar to GlobalModels
        pass

    @property
    def in_channel_names(self):
        return list(IFS_Vocabulary.VOCAB.keys())

    @property
    def out_channel_names(self):
        return self.channels

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

    def available_start_time(self, start_time: datetime) -> bool:
        """Checks if the given date and time are available in the IFS AWS data store.

        Parameters
        ----------
        start_time : datetime
            The date and time to check availability for.

        Returns
        -------
        bool
            True if data for the specified date and time is available, False otherwise.
        """

        s3 = boto3.client(
            "s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)
        )
        file_prefix = f"{start_time.strftime('%Y%m%d')}/{start_time.hour:02d}z/"
        logger.debug(f"Checking for data at prefix: {file_prefix}")
        try:
            response = s3.list_objects_v2(
                Bucket=self.IFS_BUCKET_NAME,
                Prefix=file_prefix,
                Delimiter="/",
                MaxKeys=1,
            )
        except botocore.exceptions.ClientError as e:
            logger.error("Failed to access data from the IFS S3 bucket: {e}")
            return False

        return "KeyCount" in response and response["KeyCount"] > 0

    def predict(
        self,
        date: str,  # YYYMMDD, e.g. 20180101
        time: str,  # HHMM, e.g. 0300, 1400, etc
        lead_time: int = 240,  # in hours 0-240,
        save: bool = False,
        save_config: dict = {},
    ) -> xr.DataArray:
        """
        Predict the weather using the IFS model.

        Parameters
        ----------
        date : str
            The date in the format YYYMMDD, e.g. 20180101.
        time : str
            The time in the format HHMM, e.g. 0300, 1400, etc.
        lead_time : int, optional
            The lead time in hours 0-240, by default 240.
        save : bool, optional
            Whether to save the prediction, by default False.
        save_config : dict, optional
            The save configuration, by default {}. see `skyrim.common.save_forecast`

        Returns
        -------
        xr.DataArray
            The prediction as a DataArray.
        """
        start_time = datetime.datetime.strptime(f"{date} {time}", "%Y%m%d %H%M")
        steps = self._slice_lead_time_to_steps(lead_time, start_time)
        logger.debug(f"Forecast start time: {start_time}")
        logger.debug(f"Forecast steps: {steps}")
        logger.debug(f"len(steps): {len(steps)}")
        darray = self.fetch_dataarray(start_time, steps)
        if save:
            save_forecast(
                pred=darray,
                model_name=self.model_name,
                start_time=start_time,
                pred_time=start_time + datetime.timedelta(hours=lead_time),
                source="ifs",
                config=save_config,
            )
        if not self._cache:
            logger.debug("Clearing cached files downloaded during the session")
            self.clear_cached_files()

        return darray

    def snipe(
        self,
        target_date: str,  # YYYMMDD, e.g. 20180101
        target_time: str,  # HHMM, e.g. 1200, 1800, etc
        max_hours_back: int = 0,
    ) -> dict[np.datetime64, xr.DataArray]:
        """
        Retrieves the forecast for a specific target datetime, considering multiple possible start times.

        Parameters
        ----------
        target_date : str, optional
            The target date in YYYMMDD format, e.g., 20180101.
        target_time : str, optional
            The target time in HHMM format, e.g., 1200, 1800, etc.
        max_hours_back : int
            The maximum number of hours prior to the target datetime to consider for forecast initiation.

        Returns
        -------
        dict
            A dictionary mapping each start time to the corresponding forecast data for the target datetime.
        """

        target_datetime = datetime.datetime.strptime(
            f"{target_date} {target_time}", "%Y%m%d %H%M"
        )
        forecasts = {}

        for hours_back in range(0, max_hours_back + 1, 6):
            start_time = target_datetime - datetime.timedelta(hours=hours_back)
            logger.info(f"Fetching for start_time: {start_time}")
            lead_time = int((target_datetime - start_time).total_seconds() / 3600)

            if not (
                self.available_start_time(start_time)
                and self._validate_lead_time(start_time, lead_time)
            ):
                logger.warning(
                    f"Invalid or unavailable forecast for start time: {start_time}, lead time: {lead_time}"
                )
                continue

            forecast_data = self.fetch_dataarray(start_time, [lead_time])
            forecasts[np.datetime64(start_time)] = forecast_data

        return forecasts

    def _download_ifs_channel_grib_to_cache(
        self,
        channel: str,
        levtype: str,
        level: str,
        start_time: datetime,
        step: int | list[int] = 240,
    ) -> str | None:
        """
        Download IFS channel data in GRIB format and cache it locally.

        Parameters
        ----------
        channel : str
            The meteorological parameter to download (e.g., wind, pressure, humidity).
        levtype : str
            The level type (e.g., 'sfc' for surface or 'pl' for pressure level).
        level : str
            Specific levels to download if relevant, depending on the `levtype`.
        start_time : datetime
            The starting date and time of the forecast, expressed in UTC.
        step : int or list of int, optional
            The forecast time step(s) in hours, default is 1. For seasonal forecasts,
            steps can be months, and default is set to 1 month.

        Returns
        -------
        str
            The path to the cached file containing the downloaded data.

        Notes
        -----
        - The `type` parameter defaults to 'fc' (forecast).
        - The `stream` parameter may be required depending on the system and is
          set based on the forecasting system and the ambiguity of the request.
        - Data is downloaded and stored locally as GRIB files. If the file already
          exists, the download step is skipped.

        Forecasting systems and their typical time steps:
        - HRES at 00 and 12 UTC: Steps from 0 to 144 by 3 hours, then 144 to 240 by 6 hours.
        - HRES at 06 and 18 UTC: Steps from 0 to 90 by 3 hours.
        """

        sha = hashlib.sha256(
            f"{channel}_{levtype}_{'_'.join(level)}_{start_time}_{step}".encode()
        )
        filename = sha.hexdigest()
        cache_path = os.path.join(self.cache, filename)

        logger.debug(
            f"Request: datetime: {start_time}, channel: {channel}, levtype: {levtype}, step: {step}"
        )
        if not os.path.exists(cache_path):
            try:
                request = {
                    "date": start_time.strftime("%Y%m%d"),
                    "time": start_time.strftime("%H%M"),
                    "type": "fc",
                    "param": channel,
                    "levtype": levtype,
                    "step": step,
                    "target": cache_path,
                }
                if levtype == "pl":
                    if isinstance(level, str):
                        request["levelist"] = level
                    else:
                        raise ValueError(f"Invalid level type: {type(level)}")
                result = self.client.retrieve(**request)
                logger.debug(f"Request: datetime: {start_time}")
                logger.debug(f"Result: datetime: {result.datetime}")
                assert (
                    result.datetime == start_time
                ), f"Mismatched datetime\nresult datetime: {result.datetime}\nrequest datetime: {start_time}"
                self.cached_files.append(cache_path)

            except Exception as e:
                logger.error(
                    f"Failed to download data for {channel} at {start_time}: {e}"
                )
                return None
        return cache_path

    def _validate_start_time(self, start_time: datetime) -> bool:
        """Check if the specified time is valid based on forecast issuance schedule."""
        valid_hours = {0, 6, 12, 18}
        return start_time.hour in valid_hours

    def _validate_lead_time(self, start_time: datetime, lead_time: int) -> bool:
        """Check if the lead time is valid based on the forecast issuance schedule."""
        if start_time.hour in [0, 12]:
            return (lead_time <= 144 and lead_time % 3 == 0) or (
                144 < lead_time <= 240 and lead_time % 6 == 0
            )
        elif start_time.hour in [6, 18]:
            return lead_time <= 90 and lead_time % 3 == 0
        return False

    def _slice_lead_time_to_steps(
        self, lead_time: int, start_time: datetime
    ) -> list[int]:
        """
        Slice the lead time into forecast time steps based on the start time of the forecast.

        Parameters
        ----------
        lead_time : int
            The lead time in hours.
        start_time : datetime
            The start time of the forecast.

        Returns
        -------
        list[int]
            A list of time steps.

        Raises
        ------
        ValueError
            If the lead time or start time is invalid for the HRES forecast.

        Notes
        -----
        The time steps should be arranged depending on the start time of the forecast.

        Forecasting system    Time        List of time steps
        HRES                  00 and 12   0 to 144 by 3, 144 to 240 by 6
        HRES                  06 and 18   0 to 90 by 3
        """

        if not (
            self._validate_start_time(start_time)
            or self._validate_lead_time(start_time, lead_time)
        ):
            raise ValueError(
                f"No valid forecast available for start time {start_time} and lead time {lead_time}."
            )

        # infer the time of the forecast
        if start_time.hour in [0, 12]:
            if lead_time <= 144:
                return list(range(0, lead_time + 1, 3))
            elif lead_time <= 240:
                return list(range(0, 145, 3)) + list(range(150, lead_time + 1, 6))
            else:
                raise ValueError(
                    "Invalid lead time for HRES forecast, must be less than 240 hours for 00 and 12 start times"
                )
        elif start_time.hour in [6, 18]:
            if lead_time <= 90:
                return list(range(0, lead_time + 1, 3))
            else:
                raise ValueError(
                    "Invalid lead time for HRES forecast, must be less than 90 hours for 06 and 18 start times"
                )
        else:
            raise ValueError(
                "Invalid start time for HRES forecast, must be 00, 06, 12 or 18"
            )

    def fetch_dataarray(
        self,
        start_time: datetime.datetime,
        steps: list[int] = [0, 3, 6],
    ) -> xr.DataArray:
        """
        NOTE: The IFS dataarray structure loaded from a GRIB file:
            - Dimensions ('time', 'latitude', 'longitude') represent the axes of the dataarray.
            - Longitude values range from -180 to +179.75 degrees, wrapping around the globe.
            - Latitude values range from 90 (North Pole) to -90 (South Pole) degrees.
            - The shape of 'da.values' is (time steps, 721 latitudes, 1440 longitudes)
        """

        ifs_dataarray = xr.DataArray(
            data=np.empty(
                (len(steps), len(self.channels), len(self.IFS_LAT), len(self.IFS_LON))
            ),
            dims=["time", "channel", "lat", "lon"],
            coords={
                "time": start_time
                + np.array(
                    [datetime.timedelta(hours=s) for s in steps]
                ),  # format switches to datetime64
                "channel": self.channels,
                "lat": self.IFS_LAT,
                "lon": self.IFS_LON,
            },
        )

        if not self.available_start_time(start_time):
            logger.error(f"No IFS data available for {start_time}")
            return ifs_dataarray
        logger.debug(f"Initialized ifs_dataarray with shape: {ifs_dataarray.shape}")
        for i, channel in tqdm(
            enumerate(self.channels), desc=f"Fetching IFS for start_time: {start_time}"
        ):
            ifs_id, ifs_levtype, ifs_level, modifier_func = IFS_Vocabulary.get(channel)
            cache_path = self._download_ifs_channel_grib_to_cache(
                ifs_id, ifs_levtype, ifs_level, start_time, steps
            )
            if cache_path is None:
                logger.warning(f"Skipping {channel} due to failed download.")
                continue
            logger.debug(f"Loaded {channel} from {cache_path}")

            # IFS default coordsda [-180, 180], roll to [0, 360]
            da = xr.open_dataarray(
                cache_path, engine="cfgrib", backend_kwargs={"indexpath": ""}
            ).roll(longitude=len(self.IFS_LON) // 2, roll_coords=True)

            logger.debug(f"Fetched channel: {channel}\n")
            logger.debug(f"cache path: {cache_path}\n")
            logger.debug(f"Fetched dataarray's shape: {da.shape}")

            # to properly roll the dataarray along its longitude is as follows
            # da.coords['longitude'] = (da.coords['longitude'] + 360) % 360

            data = modifier_func(da.values)
            ifs_dataarray[:, i, :, :] = data

        return ifs_dataarray


if __name__ == "__main__":

    # Ensure that the forecast start time is rounded down to the closest multiple of 6 hours,
    # at least 12 hours ago
    now = datetime.datetime.now()
    start_time = now.replace(minute=0, second=0, microsecond=0) - datetime.timedelta(
        hours=(now.hour % 6 + 12)
    )
    default_date = start_time.strftime("%Y%m%d")
    default_time = start_time.strftime("%H%M")
    default_lead_time = 36

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run IFS/HRES weather predictions.")
    parser.add_argument(
        "--date",
        help="The date in YYYMMDD format, e.g., 20230101",
        default=default_date,
    )
    parser.add_argument(
        "--time",
        help="The time in HHMM format, e.g., 1200",
        default=default_time,
    )
    parser.add_argument(
        "--lead_time",
        type=int,
        help="The lead time in hours from 0 to 240",
        default=default_lead_time,
    )
    args = parser.parse_args()

    logger.info(f"date (str) set to {args.date}")
    logger.info(f"time (str) set to {args.time}")
    logger.info(f"lead_time (int) set to {args.lead_time} hours")

    model = IFSModel(channels=["u10m", "v10m", "t2m"], cache=False)
    forecast = model.predict(
        date=args.date,
        time=args.time,
        lead_time=args.lead_time,
        save=True,
    )
    print(f"forecast.shape: {forecast.shape}")
    print(f"model.cached_files: {model.cached_files}")
