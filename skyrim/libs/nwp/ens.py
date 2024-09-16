"""
TODO:
- [ ] Add support to check if the forecast is available before downloading.
- [ ] Add support for downloading multiple forecasts at once, e.g. for each
member in parallel.
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
import ecmwf.opendata
from tqdm import tqdm

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
    LAT = np.linspace(90, -90, 721)
    LON = np.linspace(0, 360, 1440, endpoint=False)
    MODEL_NAME = "ENS"
    VOCAB = ENS_Vocabulary.VOCAB

    def __init__(
        self,
        channels: list[str],
        numbers: list[int] = list(range(0, 51)),
        source: Literal["aws", "ecmwf", "azure"] = "aws",
        cache: bool = True,
        multithread: bool = False,
    ):
        """
        ENS	00 and 12	0 to 144 by 3, 144 to 360 by 6
        ENS	06 and 18	0 to 144 by 3
        """
        # TODO: check when the resolution of this product became 0.25

        self._cache = cache
        self.assure_channels_exist(channels)
        self.channels = channels
        self.cached_files = []
        self.multithread = multithread
        self.numbers = numbers

        ensure_ecmwf_loaded()
        self.client = ecmwf.opendata.Client(source=source)
        logger.info(f"ENS model initialized with channels: {channels}")
        logger.info(f"member numbers: {numbers}")
        logger.debug(f"ENS Cache location: {self.cache}")

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
        return [
            ens_id
            for channel in self.channels
            for ens_id, ens_levtype, _ in [self.VOCAB[channel].split("::")]
            if ens_levtype == "sfc"
        ]

    @property
    def pl_params(self) -> list[str]:
        """Get pressure level parameters, in ENS lexicon"""
        return [
            ens_id
            for channel in self.channels
            for ens_id, ens_levtype, level in [self.VOCAB[channel].split("::")]
            if ens_levtype == "pl"
        ]

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
        da = xr.DataArray(
            data=np.empty(
                (
                    len(self.numbers),
                    len(steps),
                    len(self.channels),
                    len(self.LAT),
                    len(self.LON),
                )
            ),
            dims=["number", "time", "channel", "lat", "lon"],
            coords={
                "number": self.numbers,
                "time": start_time
                + np.array([datetime.timedelta(hours=step) for step in steps]),
                "channel": self.channels,
                "lat": self.LAT,
                "lon": self.LON,
            },
        )
        # fetch control forecast, member number 0
        das = []
        if 0 in self.numbers:
            logger.debug("Fetching control forecast")
            das.append(
                self._fetch_dataarray(
                    start_time=start_time,
                    steps=steps,
                    forecast_type="cf",
                )
            )

        # fetch perturbed forecast(s)
        # everything but control, i.e. members 1-50
        pf_member_numbers = [n for n in self.numbers if n != 0]
        if len(pf_member_numbers) > 0:
            logger.debug("Fetching perturbed forecast(s)")
            das.append(
                self._fetch_dataarray(
                    start_time=start_time,
                    steps=steps,
                    forecast_type="pf",
                )
            )

        # combine control and perturbed forecast
        da = xr.concat(das, dim="number")
        da = da.assign_coords({"lat": self.LAT, "lon": self.LON})
        da = da.sel(channel=self.channels)
        return da

    def _fetch_dataarray(
        self,
        start_time: datetime.datetime,
        steps: list[int],
        forecast_type: Literal["cf", "pf"],
    ) -> xr.DataArray:
        """Download grib files from ECMWF and return as xr.DataArray"""
        das = []
        if self.sl_params:
            logger.debug(f"Downloading {forecast_type} surface variables")
            s_path = self._download_levels_to_cahce(
                start_time, steps, forecast_type=forecast_type, lvtype="sfc"
            )
            logger.debug(f"Loading data from {s_path}")
            das.append(
                load_ifs_grib_data(s_path, filter_by_keys={"dataType": forecast_type})
            )

        if self.pl_params:
            logger.debug(f"Downloading {forecast_type} pressure level variables")
            p_path = self._download_levels_to_cahce(
                start_time, steps, forecast_type=forecast_type, lvtype="pl"
            )
            logger.debug(f"Loading data from {p_path}")
            das.append(
                load_ifs_grib_data(p_path, filter_by_keys={"dataType": forecast_type})
            )

        da = xr.concat(das, dim="variable").roll(
            longitude=len(self.LAT), roll_coords=True
        )
        da = da.rename({"latitude": "lat", "longitude": "lon", "variable": "channel"})
        if "number" not in da.dims:
            da = da.expand_dims("number", axis=0)

        return da.transpose("number", "time", "channel", "lat", "lon")

    def _download_levels_to_cahce(
        self,
        start_time: datetime.datetime,
        steps: list[int],
        forecast_type: Literal["cf", "pf"],
        lvtype: Literal["sfc", "pl"],
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
            request_body = request_body | {
                "number": [n for n in self.numbers if n != 0]
            }
        elif forecast_type != "cf":
            raise Exception(
                f"Invalid forecast type: {forecast_type}. Should be 'cf' or 'pf'"
            )

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

        darray = self.fetch_dataarray(start_time, steps)
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


if __name__ == "__main__":
    pass
