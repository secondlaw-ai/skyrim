import argparse
import datetime
import hashlib
import os
from typing import Literal
from loguru import logger
import xarray as xr
import numpy as np
import ecmwf.opendata
from tqdm import tqdm
from ...common import LOCAL_CACHE, save_forecast, ensure_ecmwf_loaded
from ...utils import ensure_ecmwf_loaded


# adapted from https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/data/ifs.py
# skyrim to ifs mapping
class IFS_Vocabulary:
    """
    Vocabulary for IFS model.
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
    IFS_LAT = np.linspace(90, -90, 721)
    IFS_LON = np.linspace(0, 360, 1440, endpoint=False)

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

        self.assure_channels_exist(channels)
        self.channels = channels
        ensure_ecmwf_loaded()

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
            logger.debug(f"Using temporary cache location at {cache_location}")
        if not os.path.exists(cache_location):
            os.makedirs(cache_location)
            logger.info(f"Created cache directory at {cache_location}")
        return cache_location

    @property
    def time_step(self):
        pass

    @property
    def in_channel_names(self):
        return list(IFS_Vocabulary.VOCAB.keys())

    @property
    def out_channel_names(self):
        return self.channels

    def predict(
        self,
        date: str,  # YYYMMDD, e.g. 20180101
        time: str,  # HHMM, e.g. 0300, 1400, etc
        lead_time: int = 6,  # in hours 0-240,
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
            The lead time in hours 0-240, by default 6.
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
        logger.debug(f"Using steps: {steps}")
        darray = self.fetch_ifs_dataarray(self.channels, start_time, steps)
        if save:
            save_forecast(
                pred=darray,
                model_name=self.model_name,
                start_time=start_time,
                pred_time=start_time + datetime.timedelta(hours=lead_time),
                source="ifs",
                config=save_config,
            )
        return darray

    def _download_ifs_channel_grib_to_cache(
        self,
        channel: str,
        levtype: str,
        level: str | list[str],
        start_time: datetime,
        step: int | list[int] = 1,
    ) -> str:
        """
        Download IFS channel data in GRIB format and cache it locally.

        Parameters
        ----------
        channel : str
            The meteorological parameter to download (e.g., wind, pressure, humidity).
        levtype : str
            The level type (e.g., 'sfc' for surface or 'pl' for pressure level).
        level : str or list of str
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
            f"{channel}_{levtype}_{'_'.join(level)}_{start_time}".encode()
        )
        filename = sha.hexdigest()
        cache_path = os.path.join(self.cache, filename)

        if not os.path.exists(cache_path):
            request = {
                "date": start_time,
                "type": "fc",
                "param": channel,
                "levtype": levtype,
                "step": step,
                "target": cache_path,
            }
            if levtype == "pl":  # Pressure levels
                request["levelist"] = level
            # Download
            self.client.retrieve(**request)
        return cache_path

    def _slice_lead_time_to_steps(
        self, lead_time: int, start_time: datetime
    ) -> list[int]:
        """
        Slice the lead time into time steps based on the start time of the forecast.

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
        # assert lead_time is a multiple of 3
        assert lead_time % 3 == 0, "Lead time must be a multiple of 3 hours for HRES"

        # infer the time of the forecast
        if start_time.hour in [0, 12]:
            if lead_time <= 144:
                return list(range(0, lead_time + 1, 3))
            elif lead_time <= 240:
                return list(range(0, 145, 3)) + list(range(147, lead_time + 1, 6))
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

    def fetch_ifs_dataarray(
        self,
        channels: list[str],  # skyrim channel names
        start_time: datetime,
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
                (len(steps), len(channels), len(self.IFS_LAT), len(self.IFS_LON))
            ),
            dims=["time", "channel", "lat", "lon"],
            coords={
                "time": start_time
                + np.array([datetime.timedelta(hours=s) for s in steps]),
                "channel": channels,
                "lat": self.IFS_LAT,
                "lon": self.IFS_LON,
            },
        )
        for i, channel in tqdm(enumerate(channels), desc="Downloading IFS data"):
            ifs_id, ifs_levtype, ifs_level, modifier_func = IFS_Vocabulary.get(channel)
            cache_path = self._download_ifs_channel_grib_to_cache(
                ifs_id, ifs_levtype, ifs_level, start_time, steps
            )
            # IFS default coordsda [-180, 180], roll to [0, 360]
            da = xr.open_dataarray(
                cache_path, engine="cfgrib", backend_kwargs={"indexpath": ""}
            ).roll(longitude=len(self.IFS_LON) // 2, roll_coords=True)

            # to properly roll the dataarray along its longitude is as follows
            # da.coords['longitude'] = (da.coords['longitude'] + 360) % 360

            data = modifier_func(da.values)
            ifs_dataarray[:, i, :, :] = data

        return ifs_dataarray


if __name__ == "__main__":

    # Ensure that the current time is rounded down to the closest multiple of 6 hours,
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
        help="The time in HHMM format, e.g., 1400",
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
    logger.info(f"lead_time (int) set to {args.lead_time}")

    model = IFSModel(channels=["u10m", "v10m", "msl", "u1000", "v1000"])
    forecast = model.predict(
        date=args.date,
        time=args.time,
        lead_time=args.lead_time,
        save=True,
    )
    print(f"forecast.shape: {forecast.shape}")
