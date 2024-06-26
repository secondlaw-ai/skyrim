import os
import hashlib
import argparse
import numpy as np
import xarray as xr
import datetime
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import s3fs
import shutil
from s3fs.core import S3FileSystem
from ...common import LOCAL_CACHE, save_forecast


# skyrim to gfs mapping
class GFS_Vocabulary:
    """
    Vocabulary for GFS model.

    GFS specified <Parameter ID>::<Level/ Layer>

    Additional resources:
    https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.f000.shtml

    Adapted from (huge shout out to NVIDIA/earth2studio devs):
    https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/data/gfs.py
    """

    @staticmethod
    def build_vocab():
        return {
            "u10m": "UGRD::10 m above ground",
            "v10m": "VGRD::10 m above ground",
            "u100m": "UGRD::100 m above ground",
            "v100m": "VGRD::100 m above ground",
            "t2m": "TMP::2 m above ground",
            "sp": "PRES::surface",
            "msl": "PRMSL::mean sea level",
            "tcwv": "PWAT::entire atmosphere (considered as a single layer)",
            "u50": "UGRD::50 mb",
            "u100": "UGRD::100 mb",
            "u150": "UGRD::150 mb",
            "u200": "UGRD::200 mb",
            "u250": "UGRD::250 mb",
            "u300": "UGRD::300 mb",
            "u400": "UGRD::400 mb",
            "u500": "UGRD::500 mb",
            "u600": "UGRD::600 mb",
            "u700": "UGRD::700 mb",
            "u850": "UGRD::850 mb",
            "u925": "UGRD::925 mb",
            "u1000": "UGRD::1000 mb",
            "v50": "VGRD::50 mb",
            "v100": "VGRD::100 mb",
            "v150": "VGRD::150 mb",
            "v200": "VGRD::200 mb",
            "v250": "VGRD::250 mb",
            "v300": "VGRD::300 mb",
            "v400": "VGRD::400 mb",
            "v500": "VGRD::500 mb",
            "v600": "VGRD::600 mb",
            "v700": "VGRD::700 mb",
            "v850": "VGRD::850 mb",
            "v925": "VGRD::925 mb",
            "v1000": "VGRD::1000 mb",
            "z50": "HGT::50 mb",
            "z100": "HGT::100 mb",
            "z150": "HGT::150 mb",
            "z200": "HGT::200 mb",
            "z250": "HGT::250 mb",
            "z300": "HGT::300 mb",
            "z400": "HGT::400 mb",
            "z500": "HGT::500 mb",
            "z600": "HGT::600 mb",
            "z700": "HGT::700 mb",
            "z850": "HGT::850 mb",
            "z925": "HGT::925 mb",
            "z1000": "HGT::1000 mb",
            "t50": "TMP::50 mb",
            "t100": "TMP::100 mb",
            "t150": "TMP::150 mb",
            "t200": "TMP::200 mb",
            "t250": "TMP::250 mb",
            "t300": "TMP::300 mb",
            "t400": "TMP::400 mb",
            "t500": "TMP::500 mb",
            "t600": "TMP::600 mb",
            "t700": "TMP::700 mb",
            "t850": "TMP::850 mb",
            "t925": "TMP::925 mb",
            "t1000": "TMP::1000 mb",
            "r50": "RH::50 mb",
            "r100": "RH::100 mb",
            "r150": "RH::150 mb",
            "r200": "RH::200 mb",
            "r250": "RH::250 mb",
            "r300": "RH::300 mb",
            "r400": "RH::400 mb",
            "r500": "RH::500 mb",
            "r600": "RH::600 mb",
            "r700": "RH::700 mb",
            "r850": "RH::850 mb",
            "r925": "RH::925 mb",
            "r1000": "RH::1000 mb",
            "q50": "SPFH::50 mb",
            "q100": "SPFH::100 mb",
            "q150": "SPFH::150 mb",
            "q200": "SPFH::200 mb",
            "q250": "SPFH::250 mb",
            "q300": "SPFH::300 mb",
            "q400": "SPFH::400 mb",
            "q500": "SPFH::500 mb",
            "q600": "SPFH::600 mb",
            "q700": "SPFH::700 mb",
            "q850": "SPFH::850 mb",
            "q925": "SPFH::925 mb",
            "q1000": "SPFH::1000 mb",
        }

    VOCAB = build_vocab()

    def __getitem__(self, key):
        """Allow dictionary-like access (e.g., GFS_Vocabulary['u100'])"""
        return self.VOCAB[key]

    def __contains__(self, key):
        """Allow membership testing (e.g., 'u100' in GFS_Vocabulary)"""
        return key in self.VOCAB

    @classmethod
    def get(cls, channel: str) -> str:
        """Get GFS parameter ID, level, and modifier function for a given channel."""

        gfs_key = cls.VOCAB[channel]
        gfs_id, gfs_level = gfs_key.split("::")

        if gfs_id == "HGT":
            modifier_func = lambda x: x * 9.81
            return gfs_id, gfs_level, modifier_func
        return gfs_id, gfs_level, lambda x: x


class GFSModel:
    """
    Global Forecast System (GFS)

    GFS is a global model with a base horizontal resolution of 18 miles (28 kilometers)
    between grid points.

    Temporal resolution covers analysis and forecasts out to 16 days.

    Horizontal resolution drops to 44 miles (70 kilometers) between grid points for forecasts
    between one week and two weeks.

    It produces hourly forecast output for the first 120 hours,
    then 3 hourly for days 5-16.

    Additional resources:
    https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast#:~:text=GFS%20is%20a%20global%20model,one%20week%20and%20two%20weeks.

    """

    GFS_LAT = np.linspace(90, -90, 721)
    GFS_LON = np.linspace(0, 360, 1440, endpoint=False)
    GFS_BUCKET_NAME = "noaa-gfs-bdp-pds"
    MAX_BYTE_SIZE = 5000000

    def __init__(self, channels: list[str], cache: bool = True):
        self._cache = cache
        self.model_name = "GFS"
        self.assure_channels_exist(channels)
        self.channels = channels
        self.cached_files = []
        logger.info(f"GFS model initialized with channels: {channels}")
        logger.debug(f"GFScache location: {self.cache}")

    def assure_channels_exist(self, channels: list[str]):
        for channel in channels:
            assert (
                channel in GFS_Vocabulary.VOCAB
            ), f"Channel {channel} not found in GFS vocabulary."

    @staticmethod
    def list_available_channels():
        return list(GFS_Vocabulary.VOCAB.keys())

    @property
    def cache(self):
        """Get the appropriate cache location."""
        cache_location = os.path.join(LOCAL_CACHE, "gfs")
        if not self._cache:
            cache_location = os.path.join(LOCAL_CACHE, "gfs", "tmp")
            logger.debug(f"Using temporary cache location at {cache_location}")
        if not os.path.exists(cache_location):
            os.makedirs(cache_location)
            logger.info(f"Created cache directory at {cache_location}")
        return cache_location

    @property
    def time_step(self):
        # TODO: implement time step similar to GlobalModels
        pass

    @property
    def in_channel_names(self):
        return list(GFS_Vocabulary.VOCAB.keys())

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
        """Checks if the given date and time are available in the  GFS object store.

        Parameters
        ----------
        start_time : datetime
            The date and time to check availability for.

        Returns
        -------
        bool
            True if data for the specified date and time is available, False otherwise.
        """

        fs = S3FileSystem(anon=True)

        # Object store directory for given time
        # Should contain two keys: atmos and wave
        file_name = f"gfs.{start_time.year}{start_time.month:0>2}{start_time.day:0>2}/{start_time.hour:0>2}/"
        s3_uri = f"s3://{self.GFS_BUCKET_NAME}/{file_name}"
        exists = fs.exists(s3_uri)
        (
            logger.debug(f"{s3_uri} is available")
            if exists
            else logger.debug(f"{s3_uri} is NOT available")
        )

        return exists

    def predict(
        self,
        date: str,  # YYYMMDD, e.g. 20180101
        time: str,  # HHMM, e.g. 0300, 1400, etc
        lead_time: int = 24,  # in hours 0-394,
        save: bool = False,
        save_config: dict = {},
    ) -> xr.DataArray:
        """
        Predict the weather using the GFS model.

        Parameters
        ----------
        date : str
            The date in the format YYYMMDD, e.g. 20180101.
        time : str
            The time in the format HHMM, e.g. 0300, 1400, etc.
        lead_time : int, optional
            The lead time in hours [0-394], by default 24.
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
        steps = self._slice_lead_time_to_steps(lead_time)
        logger.debug(f"Forecast start time: {start_time}")
        logger.debug(f"Forecast steps: {steps}")
        logger.debug(f"len(steps): {len(steps)}")
        darray = self.fetch_gfs_dataarray(start_time, steps)
        if save:
            save_forecast(
                pred=darray,
                model_name=self.model_name,
                start_time=start_time,
                pred_time=start_time + datetime.timedelta(hours=lead_time),
                source="gfs",
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
            logger.debug(f"Lead time is set to {lead_time}")
            if not (
                self.available_start_time(start_time)
                and self._validate_lead_time(lead_time)
            ):
                logger.warning(
                    f"Invalid or unavailable forecast for start time: {start_time}, lead time: {lead_time}"
                )
                continue

            forecast_data = self.fetch_gfs_dataarray(start_time, [lead_time])
            forecasts[np.datetime64(start_time)] = forecast_data

        return forecasts

    def _validate_start_time(self, start_time: datetime) -> bool:
        """Check if the specified time is valid based on forecast issuance schedule."""
        valid_hours = {0, 6, 12, 18}
        return start_time.hour in valid_hours

    def _validate_lead_time(self, lead_time: int) -> bool:
        return lead_time in list(range(0, 385))

    def _slice_lead_time_to_steps(self, lead_time: int) -> list[int]:
        return list(range(0, lead_time + 1))

    def _get_grib_filename(
        self, start_time: datetime, step: int, is_index: bool = False
    ):
        filename = f"gfs.{start_time.year}{start_time.month:0>2}{start_time.day:0>2}/{start_time.hour:0>2}"
        filename = os.path.join(
            filename, f"atmos/gfs.t{start_time.hour:0>2}z.pgrb2.0p25.f{step:03d}"
        )
        if is_index:
            filename += ".idx"
        return filename

    def get_grib_s3uri(self, start_time: datetime, step: int, is_index: bool = False):
        return os.path.join(
            self.GFS_BUCKET_NAME, self._get_grib_filename(start_time, step, is_index)
        )

    def fetch_gfs_dataarray(
        self,
        start_time: datetime,
        steps: list[int] = list(range(0, 385)),
    ) -> xr.DataArray:
        """
        TBA
        Additional information
        model cycle runtimes are 00, 06, 12, 18
        >> aws s3 ls --no-sign-request s3://noaa-gfs-bdp-pds/gfs.20240610/

                           PRE 00/
                           PRE 06/
                           PRE 12/
                           PRE 18/
        """

        gfs_dataarray = xr.DataArray(
            data=np.empty(
                (len(steps), len(self.channels), len(self.GFS_LAT), len(self.GFS_LON))
            ),
            dims=["time", "channel", "lat", "lon"],
            coords={
                "time": start_time
                + np.array([datetime.timedelta(hours=s) for s in steps]),
                "channel": self.channels,
                "lat": self.GFS_LAT,
                "lon": self.GFS_LON,
            },
        )
        logger.debug(f"Creating GFS dataarray with shape: {gfs_dataarray.shape}")

        for i, channel in enumerate(
            tqdm(self.channels, desc=f"Fetching GFS for {start_time}")
        ):
            gfs_id, gfs_level, modifier_func = GFS_Vocabulary.get(channel)
            gfs_name = f"{gfs_id}::{gfs_level}"
            # TODO: Check if gfs_name is in index_file
            for sidx, step in enumerate(steps):
                index_file = self._fetch_index(start_time, step)
                byte_offset, byte_length = (
                    index_file[gfs_name][0],
                    index_file[gfs_name][1],
                )
                pred_time = start_time + datetime.timedelta(hours=step)

                # Download the grib file to cache
                logger.debug(
                    f"Fetching GFS grib file for channel: {channel} at {pred_time}"
                )
                s3_uri = self.get_grib_s3uri(start_time, step)
                grib_file = self._download_s3_grib_to_cache(
                    s3_uri, byte_offset=byte_offset, byte_length=byte_length
                )
                self.cached_files.append(grib_file)
                # Open into xarray data-array
                da = xr.open_dataarray(
                    grib_file, engine="cfgrib", backend_kwargs={"indexpath": ""}
                )
                logger.debug(f"Cached data array shape: {da.shape}")
                gfs_dataarray[sidx, i, :, :] = modifier_func(da.values)

        return gfs_dataarray

    def _fetch_index(
        self, start_time: datetime, step: int
    ) -> dict[str, tuple[int, int]]:
        """Fetch GFS atmospheric index file

        Parameters
        ----------
        start_time : datetime
            The start time of the forecast.
        step : int
            The forecast step in hours.

        Returns
        -------
        dict[str, tuple[int, int]]
            Dictionary of GFS vairables (byte offset, byte length)
            Key is the GFS variable name.

        Additional information
        ----------------------
        len(index_table.keys())
        >> 695
        index_table["PRMSL::mean sea level"]
        >> (0, 1001587)
        index_table["CLMR::1 hybrid level"]
        >> (1001587, 101339)

        CC is the model cycle runtime (i.e. 00, 06, 12, 18)
        FFF is the forecast hour of product from 000 - 384
        YYYYMMDD is the Year, Month and Day
        0.25 degree resolution	gfs.tCCz.pgrb2.0p25.fFFF	ANL FH000

        """
        # https://www.nco.ncep.noaa.gov/pmb/products/gfs/
        file_name = self._get_grib_filename(start_time, step, is_index=True)
        s3_uri = os.path.join(self.GFS_BUCKET_NAME, file_name)
        # Grab index file: hold channel/variable information
        # Example:
        #   1:0:d=2024060500:PRMSL:mean sea level:anl:
        #   2:1001587:d=2024060500:CLMR:1 hybrid level:anl:
        #   3:1102926:d=2024060500:ICMR:1 hybrid level:anl:

        index_file = self._download_s3_index_to_cache(s3_uri)

        with open(index_file) as file:
            index_lines = [line.rstrip() for line in file]

        index_table = {}
        # NOTE we actually drop the last variable here (Vertical Speed Shear)
        # 696:502366026:d=2024060500:VWSH:PV=-2e-06 (Km^2/kg/s) surface:anl:
        for i, line in enumerate(index_lines[:-1]):
            lsplit = line.split(":")
            if len(lsplit) < 7:
                continue

            nlsplit = index_lines[i + 1].split(":")
            byte_length = int(nlsplit[1]) - int(lsplit[1])
            byte_offset = int(lsplit[1])
            key = f"{lsplit[3]}::{lsplit[4]}"
            if byte_length > self.MAX_BYTE_SIZE:
                raise ValueError(
                    f"Byte length, {byte_length}, of variable {key} larger than safe threshold of {self.MAX_BYTE_SIZE}"
                )

            index_table[key] = (byte_offset, byte_length)

        return index_table

    def _download_s3_index_to_cache(self, path: str) -> str:
        sha = hashlib.sha256(path.encode())
        filename = sha.hexdigest()

        cache_path = os.path.join(self.cache, filename)
        if not Path(cache_path).is_file():
            logger.debug(f"Getting GFS index file from {path}")
            fs = s3fs.S3FileSystem(anon=True, client_kwargs={})
            fs.get_file(path, cache_path)
        else:
            logger.debug(
                f"Index File {cache_path} already exists in cache, skipping download."
            )
        self.cached_files.append(cache_path)
        return cache_path

    def _download_s3_grib_to_cache(
        self, path: str, byte_offset: int = 0, byte_length: int = None
    ) -> str:
        """
        Downloads a GRIB file from an S3 bucket and caches it locally.

        Parameters
        ----------
        path : str
            The path to the GRIB file in the S3 bucket.
        byte_offset : int, optional
            The byte offset to start downloading from. Defaults to 0.
        byte_length : int, optional
            The number of bytes to download. Defaults to None, which downloads the entire file.

        Returns
        -------
        str
            The path to the locally cached file.
        """
        sha = hashlib.sha256((path + str(byte_offset)).encode())
        filename = sha.hexdigest()

        cache_path = os.path.join(self.cache, filename)

        if not Path(cache_path).is_file():
            logger.debug(f"Getting GFS grib file from {path}")
            fs = s3fs.S3FileSystem(anon=True, client_kwargs={})
            data = fs.read_block(path, offset=byte_offset, length=byte_length)
            with open(cache_path, "wb") as file:
                file.write(data)
        else:
            logger.debug(
                f"File {cache_path} already exists in cache, skipping download."
            )
        self.cached_files.append(cache_path)
        return cache_path


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

    model = GFSModel(channels=["u10m", "v10m", "t2m"], cache=False)
    forecast = model.predict(
        date=args.date,
        time=args.time,
        lead_time=args.lead_time,
        save=True,
    )
    print(f"forecast.shape: {forecast.shape}")
    print(f"model.cached_files: {model.cached_files}")
