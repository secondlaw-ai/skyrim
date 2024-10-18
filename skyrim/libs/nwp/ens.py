"""
TODO:
- [ ] Add "snipe" method to fetch all available forecasts for a given date and time. 

NOTE: 
When downloaded from opendata: 
    ENS pressure levels looks as follows:
        array([1000.,  925.,  850.,  700.,  500.,  300.,  250.,  200.,   50.])

    HRES pressure levels looks as follows:
        array([1000.,  925.,  850.,  700.,  600.,  500.,  400.,  300.,  250.,
            200.,  150.,  100.,   50.])
    SO THEY ARE NOT EXACTLY THE SAME!
    
    According to: https://www.ecmwf.int/en/forecasts/datasets/open-data
        ENS product should also have 13 pressure levels. Weird.
        Steps:
        For times 00z &12z: 0 to 144 by 3, 150 to 360 by 6.
        For times 06z & 18z: 0 to 144 by 3
        Single and Pressure Levels (hPa): 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50   

Example invocation:
python -m skyrim.libs.nwp.ens 

"""

import argparse
import datetime
import hashlib
import shutil
import os
import time
import json

from typing import Literal
from loguru import logger
import xarray as xr
import numpy as np
import boto3
import botocore
import ecmwf.opendata
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from ...common import LOCAL_CACHE, save_forecast
from ...utils import ensure_ecmwf_loaded
from .utils import load_ifs_grib_data


class ENS_Vocabulary:
    """
    Vocabulary for ENS model.

    When fetched from ECMWF using ecmwf.opendata, the variables are named as follows:
    [t2m, d2m] -> {"typeOfLevel": "heightAboveGround", "level": 2}
    [v10, u10] -> {"typeOfLevel": "heightAboveGround", "level": 10}
    ['v100', 'u100'] -> {"typeOfLevel":"heightAboveGround", "level": 100}
    ['tp', 'lsm', 'ssrd', 'sp', 'ssr', 'skt', 'strd', 'asn', 'str', 'ro'] -> {"typeOfLevel": "surface"}
    ['cape', 'tcwv'] -> {"typeOfLevel": "entireAtmosphere"}
    ['gh', 't', 'u', 'v', 'r', 'w', 'q', 'vo', 'd'] -> {"typeOfLevel": "isobaricInhPa"}
    ['msl'] -> {"typeOfLevel": "meanSea"}
    """

    @staticmethod
    def build_vocab():
        """Create ENS vocabulary dictionary"""
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
        # NOTE: some pressure levels from HRES are  missing
        prs_levels = [
            50,
            200,
            250,
            300,
            500,
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
        return self.VOCAB[key]

    def __contains__(self, key):
        return key in self.VOCAB

    def __len__(self):
        return len(self.VOCAB)

    @classmethod
    def get_variable(cls, key):
        ens_key = cls.VOCAB[key]
        ens_id, ens_levtype, ens_level = ens_key.split("::")
        if ens_id == "gh":
            modifier_func = lambda x: x * 9.81
            return ens_id, ens_levtype, ens_level, modifier_func
        return ens_id, ens_levtype, ens_level, lambda x: x


class ENSModel:
    """
    ENS is an ensemble of 51 forecasts with a horizontal resolution of around 9 km.
    It comprises one control forecast (CNTL) plus 50 forecasts each with slightly altered
    initial conditions and slightly altered model physics.
    Forecast lead time is 15 days.

    ENS	00 and 12	0 to 144 by 3, 144 to 360 by 6
    ENS	06 and 18	0 to 144 by 3

    Additional information:
        https://confluence.ecmwf.int/display/FUG/Section+2.1.2.1+ENS+-+Ensemble+Forecasts
        https://www.ecmwf.int/en/forecasts/documentation-and-support/medium-range-forecasts

    """

    MODEL_NAME = "ENS"
    VOCAB = ENS_Vocabulary.VOCAB
    BUCKET = "ecmwf-forecasts"

    def __init__(
        self,
        channels: list[str],
        numbers: list[int] = list(range(0, 51)),
        source: Literal["aws", "ecmwf", "azure"] = "aws",
        cache: bool = True,
        multithread: bool = False,
        max_workers: int = 4,
        resolution: Literal["0p25", "0p4-beta"] = "0p25",
    ):
        # TODO: add docstring!
        # TODO: check when the resolution of this product became 0.25

        self._cache = cache
        self.assure_channels_exist(channels)
        self.channels = channels
        self.cached_files = []
        self.multithread = multithread
        self.max_workers = max_workers
        self.numbers = numbers
        self.model_name = "ENS"

        ensure_ecmwf_loaded()
        self.client = ecmwf.opendata.Client(source=source, resol=resolution)
        logger.info(f"ENS model initialized with channels: {channels}")
        logger.info(f"member numbers: {numbers}")
        logger.debug(f"ENS Cache location: {self.cache}")

        if resolution == "0p25":
            self.LAT = np.linspace(90, -90, 721)
            self.LON = np.linspace(0, 360, 1440, endpoint=False)
        else:
            self.LAT = np.linspace(90, -90, 451)
            self.LON = np.linspace(0, 360, 900, endpoint=False)

    def assure_channels_exist(self, channels):
        for channel in channels:
            if channel not in self.VOCAB:
                raise Exception(f"Channel {channel} does not exist in the vocabulary.")

    @classmethod
    def list_available_channels(cls):
        return list(cls.VOCAB.keys())

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(LOCAL_CACHE, "ens")
        if not self._cache:
            cache_location = os.path.join(LOCAL_CACHE, "ens", "tmp")
        if not os.path.exists(cache_location):
            os.makedirs(cache_location, exist_ok=True)
        return cache_location

    @property
    def sl_params(self) -> list[str]:
        """Get surface level parameters, in ENS lexicon"""
        return sorted(
            set(
                ens_id
                for channel in self.channels
                for ens_id, ens_levtype, _ in [self.VOCAB[channel].split("::")]
                if ens_levtype == "sfc"
            )
        )

    @property
    def pl_params(self) -> list[str]:
        """Get pressure level parameters, in ENS lexicon"""
        return sorted(
            set(
                ens_id
                for channel in self.channels
                for ens_id, ens_levtype, level in [self.VOCAB[channel].split("::")]
                if ens_levtype == "pl"
            )
        )

    @property
    def levelist(self) -> list[int]:
        """Get pressure levels"""
        return sorted(
            set(
                [
                    int(level)
                    for channel in self.channels
                    for ens_id, ens_levtype, level in [self.VOCAB[channel].split("::")]
                    if ens_levtype == "pl"
                ]
            )
        )

    @property
    def stream(self) -> str:
        return "enfo"

    @property
    def in_channel_names(self):
        return list(self.VOCAB.keys())

    @property
    def out_channel_names(self):
        return self.channels

    def clear_cache(self):
        """Clears the entire cache directory."""
        if os.path.exists(self.cache):
            logger.info(f"Clearing cache directory: {self.cache}")
            shutil.rmtree(self.cache)
        else:
            logger.debug(f"Cache directory not found: {self.cache}")

    def clear_cached_files(self):
        """Clears the cached files from the current session."""
        for file_path in self.cached_files:
            if os.path.exists(file_path):
                logger.info(f"Deleting cached file: {file_path}")
                os.remove(file_path)
        self.cached_files = []

    def fetch_dataarray(
        self, start_time: datetime.datetime, steps: list[int]
    ) -> xr.DataArray:
        """
        Fetch DataArray for given start time and steps.
        Steps are hours from start_time.

        """
        das = []
        # Use multithreading if enabled
        if self.multithread:
            # NOTE: GRIB reading library, isn’t thread-safe
            # with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for number in self.numbers:
                    forecast_type = "cf" if number == 0 else "pf"
                    futures.append(
                        executor.submit(
                            self._fetch_member_dataarray,
                            start_time,
                            steps,
                            forecast_type,
                            number,
                        )
                    )
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Downloading members",
                ):
                    try:
                        da_member = future.result()
                        das.append(da_member)
                    except Exception as e:
                        logger.error(f"Error downloading data: {str(e)}")
                        continue
        else:
            for number in tqdm(self.numbers, desc="Downloading members"):
                forecast_type = "cf" if number == 0 else "pf"
                da_member = self._fetch_member_dataarray(
                    start_time, steps, forecast_type, number
                )
                das.append(da_member)

        # concatenate along 'number' dimension
        da = xr.concat(das, dim="number")
        da = da.assign_coords({"lat": self.LAT, "lon": self.LON})
        da = da.sel(channel=self.channels)
        return da

    def _fetch_member_dataarray(
        self,
        start_time: datetime.datetime,
        steps: list[int],
        forecast_type: Literal["cf", "pf"],
        number: int,
    ) -> xr.DataArray:
        """Download grib files for a single member and return as xr.DataArray"""
        das = []
        if self.sl_params:
            logger.debug(
                f"Downloading {forecast_type} surface variables for member {number}"
            )
            s_path = self._download_levels_to_cahce(
                start_time,
                steps,
                forecast_type=forecast_type,
                lvtype="sfc",
                number=number,
            )
            logger.debug(f"Loading data from {s_path}")
            da_s = load_ifs_grib_data(
                s_path, filter_by_keys={"dataType": forecast_type, "number": number}
            )
            das.append(da_s)

        if self.pl_params:
            logger.debug(
                f"Downloading {forecast_type} pressure level variables for member {number}"
            )
            p_path = self._download_levels_to_cahce(
                start_time,
                steps,
                forecast_type=forecast_type,
                lvtype="pl",
                number=number,
            )
            logger.debug(f"Loading data from {p_path}")
            da_p = load_ifs_grib_data(
                p_path, filter_by_keys={"dataType": forecast_type, "number": number}
            )
            das.append(da_p)

        da = xr.concat(das, dim="variable").roll(
            longitude=len(self.LON) // 2, roll_coords=True
        )
        da = da.rename({"latitude": "lat", "longitude": "lon", "variable": "channel"})
        if "number" not in da.dims:
            da = da.expand_dims("number", axis=0)
        da = da.assign_coords({"number": [number]})
        return da.transpose("number", "time", "channel", "lat", "lon")

    def _download_levels_to_cahce(
        self,
        start_time: datetime.datetime,
        steps: list[int],
        forecast_type: Literal["cf", "pf"],
        lvtype: Literal["sfc", "pl"],
        number: int = None,
    ) -> str:

        request_body = {
            "date": start_time.strftime("%Y%m%d"),
            "time": start_time.strftime("%H%M"),
            "type": forecast_type,
            "step": steps,
            "stream": self.stream,
            "param": self.sl_params if lvtype == "sfc" else self.pl_params,
        }

        if forecast_type == "pf":
            request_body["number"] = number

        if lvtype == "pl":
            request_body["levelist"] = self.levelist
        elif lvtype != "sfc":
            raise ValueError(f"Invalid level type: {lvtype}. Should be 'sfc' or 'pl'")

        # Create filename from the request body, i.e. unique hash
        filename = hashlib.sha256(
            json.dumps(request_body, sort_keys=True).encode()
        ).hexdigest()
        target = os.path.join(self.cache, filename)

        if os.path.exists(target):
            logger.info(f"File already exists: {target}")
            return target

        request_body["target"] = target

        try:
            logger.debug(f"Request body: {request_body}")
            self.client.retrieve(**request_body)
            self.cached_files.append(target)
            logger.success(f"Downloaded data to {target}")
            return target
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            logger.error(f"Request body: {request_body}")
            os.remove(target) if os.path.exists(target) else None
            raise

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
                Bucket=self.BUCKET,
                Prefix=file_prefix,
                Delimiter="/",
                MaxKeys=1,
            )
        except botocore.exceptions.ClientError as e:
            logger.error("Failed to access data from the IFS S3 bucket: {e}")
            return False

        return "KeyCount" in response and response["KeyCount"] > 0

    def forecast(
        self,
        start_time: datetime.datetime,
        n_steps: int = 3,
        step_size: int = 6,  # hours
        **kwargs,
    ) -> xr.DataArray:
        # TODO: too brittle, as it assummes that the forecast is available
        # include the forecast start_time
        steps = [step_size * i for i in range(n_steps + 1)]

        logger.debug(f"Forecast start time: {start_time}")
        logger.debug(f"Forecast steps: {steps}")
        logger.debug(f"len(steps): {len(steps)}")

        if not self.available_start_time(start_time):
            logger.error(f"No ENS data available for {start_time}")
            raise Exception(f"Data not available for {start_time}")

        darray = self.fetch_dataarray(start_time, steps)

        if not self._cache:
            logger.debug("Clearing cached files downloaded during the session")
            self.clear_cached_files()
        return darray

    def _slice_lead_time_to_steps(
        self, lead_time: int, start_time: datetime.datetime
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
            If the lead time or start time is invalid for the ENS forecast.

        ENS	00 and 12	0 to 144 by 3, 144 to 360 by 6
        ENS	06 and 18	0 to 144 by 3

        """

        # infer the time of the forecast
        if start_time.hour in [0, 12]:
            if lead_time <= 144:
                return list(range(0, lead_time + 1, 3))
            elif lead_time <= 360:
                return list(range(0, 145, 3)) + list(range(150, lead_time + 1, 6))
            else:
                raise ValueError(
                    "Invalid lead time for ENS forecast, must be less than or equal to 360 hours for 00 and 12 start times"
                )
        elif start_time.hour in [6, 18]:
            if lead_time <= 144:
                return list(range(0, lead_time + 1, 3))
            else:
                raise ValueError(
                    "Invalid lead time for ENS forecast, must be less than or equal to 144 hours for 06 and 18 start times"
                )
        else:
            raise ValueError(
                "Invalid start time for ENS forecast, must be 00, 06, 12 or 18"
            )

    def predict(
        self,
        date: str,  # YYYMMDD, e.g. 20180101
        time: str,  # HHMM, e.g. 0000, 1200, etc
        lead_time: int = 240,  # in hours 0-360,
        save: bool = False,
        save_config: dict = {},
    ) -> xr.DataArray:
        """
        Predict the weather using the ENS model.

        Parameters
        ----------
        date : str
            The date in the format YYYMMDD, e.g. 20180101.
        time : str
            The time in the format HHMM, e.g. 0000, 1200, etc.
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
                source="ens",
                config=save_config,
            )
        if not self._cache:
            logger.debug("Clearing cached files downloaded during the session")
            self.clear_cached_files()

        return darray

    def snipe(self):
        raise NotImplementedError("Snipe method not implemented for ENS model.")


def parse_arguments():

    # Ensure that the forecast start time is:
    #   rounded down to the closest multiple of 6 hours
    #   at least 12 hours ago from the current time
    now = datetime.datetime.now()
    latest_start_time = now.replace(
        minute=0, second=0, microsecond=0
    ) - datetime.timedelta(hours=(now.hour % 6 + 12))

    parser = argparse.ArgumentParser(description="Fetch ENS forecast.")
    parser.add_argument(
        "--date",
        type=str,
        help="The date in the format YYYMMDD, e.g. 20240401.",
        default=latest_start_time.strftime("%Y%m%d"),
    )
    parser.add_argument(
        "--time",
        type=str,
        help="The time in the format HHMM, e.g. 0000, 1200, etc.",
        default=latest_start_time.strftime("%H%M"),
    )
    parser.add_argument(
        "--lead_time",
        type=int,
        help="The lead time in hours 0-360.",
        default=36,
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        type=str,
        default=["t2m", "u10m", "v10m"],
        help="The channels to fetch.",
    )
    parser.add_argument(
        "--numbers",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        help="The member numbers to fetch. '0' is the control forecast.",
    )
    parser.add_argument(
        "--multithread",
        action="store_true",
        help="Whether to fetch the forecasts in parallel.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="The maximum number of workers to use for fetching forecasts.",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    logger.info(f"date (str) set to {args.date}")
    logger.info(f"time (str) set to {args.time}")
    logger.info(f"lead_time (int) set to {args.lead_time}")
    logger.info(f"channels (list[str]) set to {args.channels}")
    logger.info(f"numbers (list[int]) set to {args.numbers}")
    logger.info(f"multithread (bool) set to {args.multithread}")

    t = time.time()
    model = ENSModel(
        channels=args.channels,
        numbers=args.numbers,
        cache=False,
        multithread=args.multithread,
        max_workers=args.max_workers,
    )
    model.clear_cache()

    forecast = model.predict(
        date=args.date,
        time=args.time,
        lead_time=args.lead_time,
        save=True,
    )

    logger.success(f"Forecast fetched in {time.time() - t:.2f} seconds.")
    logger.info(f"forecast.shape: {forecast.shape}")
    logger.info(f"forecast.nbytes: {forecast.nbytes/1024/1024:.2f} MB")
