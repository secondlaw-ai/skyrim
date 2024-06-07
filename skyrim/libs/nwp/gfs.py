import numpy as np
import xarray as xr
import datetime
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

    def __init__(self, channels: list[str], cache: bool = True, verbose: bool = True):
        self._cache = cache
        self._verbose = verbose
        self.model_name = "GFS"
        self.assure_channels_exist(channels)
        self.channels = channels

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
        pass

    @property
    def in_channel_names(self):
        return list(GFS_Vocabulary.VOCAB.keys())

    @property
    def out_channel_names(self):
        return self.channels

    def predict(
        self,
        date: str,  # YYYMMDD, e.g. 20180101
        time: str,  # HHMM, e.g. 0300, 1400, etc
        lead_time: int = 24,  # in hours 0-240,
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
            The lead time in hours 0-240, by default 24.
        save : bool, optional
            Whether to save the prediction, by default False.
        save_config : dict, optional
            The save configuration, by default {}. see `skyrim.common.save_forecast`

        Returns
        -------
        xr.DataArray
            The prediction as a DataArray.
        """
        pass

    def fetch_gfs_dataarray(
        self,
        channels: list[str],
        start_time: datetime,
    ) -> xr.DataArray:
        pass

    def _fetch_index(self, time: datetime) -> dict[str, tuple[int, int]]:
        """Fetch GFS atmospheric index file

        Parameters
        ----------
        time : datetime
            Date time to fetch

        Returns
        -------
        dict[str, tuple[int, int]]
            Dictionary of GFS vairables (byte offset, byte length)

        Additional information
        ----------------------
        len(index_table.keys())
        >> 695
        index_table["PRMSL::mean sea level"]
        >> (0, 1001587)
        index_table["CLMR::1 hybrid level"]
        >> (1001587, 101339)
        """
        # https://www.nco.ncep.noaa.gov/pmb/products/gfs/
        file_name = f"gfs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}"
        file_name = os.path.join(
            file_name, f"atmos/gfs.t{time.hour:0>2}z.pgrb2.0p25.f000.idx"
        )
        s3_uri = os.path.join(self.GFS_BUCKET_NAME, file_name)
        # Grab index file: hold channel/variable information
        # Example:
        #   1:0:d=2024060500:PRMSL:mean sea level:anl:
        #   2:1001587:d=2024060500:CLMR:1 hybrid level:anl:
        #   3:1102926:d=2024060500:ICMR:1 hybrid level:anl:

        index_file = self._download_s3_index_cached(s3_uri)
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

    def _download_s3_index_cached(self, path: str) -> str:
        sha = hashlib.sha256(path.encode())
        filename = sha.hexdigest()

        cache_path = os.path.join(self.cache, filename)
        fs = s3fs.S3FileSystem(anon=True, client_kwargs={})
        fs.get_file(path, cache_path)

        return cache_path

    def _download_s3_grib_cached(
        self, path: str, byte_offset: int = 0, byte_length: int = None
    ) -> str:
        sha = hashlib.sha256((path + str(byte_offset)).encode())
        filename = sha.hexdigest()

        cache_path = os.path.join(self.cache, filename)

        fs = s3fs.S3FileSystem(anon=True, client_kwargs={})
        if not pathlib.Path(cache_path).is_file():
            data = fs.read_block(path, offset=byte_offset, length=byte_length)
            with open(cache_path, "wb") as file:
                file.write(data)

        return cache_path


if __name__ == "__main__":
    model = GFSModel(channels=["u10m", "v10m", "msl", "u1000", "v1000"])
    forecast = model.predict(
        date="20240521",
        time="0000",
        lead_time=24,
        save=True,
    )
    print(f"forecast.shape: {forecast.shape}")
