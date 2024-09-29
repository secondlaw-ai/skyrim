import re
import requests
import xarray as xr
import xesmf as xe
from loguru import logger as log
from pathlib import Path
from datetime import datetime, date
from ...utils import fast_fetch
from .ifs import GRIDS
from pathlib import Path


REGRID_WEIGHTS_PATH = ".data/regridder"
BASE_URL = "https://rda.ucar.edu/datasets/d113001/filelist"


def _generate_filename_pattern(analysis, filetype: str = "nc"):
    if filetype not in {"nc", "grb"}:
        raise ValueError("Filetype must be either nc or grb")
    return (
        r"https\:\/\/data.rda.ucar.edu\/d113001\/ec\.oper\.an\.%s\/(?P<month>\d+)\/ec\.oper\.an\.%s\.\d{3}_\d{3}_(?P<param>.+)\.(?P<regno>.+)\.(?P<start_time>\d+)\.%s"
        % (analysis, analysis, filetype)
    )


# fmt: off
SKYRIM_IFS_VARS = { "msl", "q50", "q100", "q150", "q200", "q250", "q300", "q400", "q500", "q600", "q700", "q850", "q925", "q1000", "r50", "r100", "r150", "r200", "r250", "r300", "r400", "r500", "r600", "r700", "r850", "r925", "r1000", "sp", "t50", "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925", "t1000", "tcwv", "u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925", "u1000", "v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850", "v925", "v1000", "u100m", "v100m", "u10m", "v10m", "t2m", "z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", "z850", "z925", "z1000", }

# these are all available SFC paramaters as of 9/24
NCAR_SFC_VARS = { "swvl2", "2d", "stl3", "sdor", "skt", "swvl1", "cvl", "tcw", "sstk", "istl3", "chnk", "tcc", "sd", "slt", "tvl", "lailv", "z", "stl1", "al", "isor", "lsrh", "10v", "anor", "100v", "asn", "tcwv", "istl4", "cvh", "swvl4", "alnip", "2t", "100u", "aluvd", "sdfor", "lsm", "ci", "sr", "lcc", "istl2", "sp", "stl4", "aluvp", "swvl3", "laihv", "tsn", "hcc", "mcc", "slor", "tvh", "stl2", "src", "10u", "istl1", "tco3", "rsn", "msl", "alnid", }

ncar_pl_vars = ["u", "v", "z", "t", "r", "q"]  # same as skyrim
skyrim_pl_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
skyrim_sfc_vars = [
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "t2m",
    "sp",
    "msl",
    "tcwv",
    "tp",  # could not match this one, but is not needed
]
NCAR_TO_SKYRIM_SFC_MAPPING = {
    "10u": "u10m",
    "10v": "v10m",
    "100u": "u100m",
    "100v": "v100m",
    "2t": "t2m",
    "sp": "sp",
    "msl": "msl",
    "tcwv": "tcwv",
}
ncar_sfc_vars = [
    k for k, v in NCAR_TO_SKYRIM_SFC_MAPPING.items() if v in skyrim_sfc_vars
]
# fmt: on


def _rename(ds: xr.Dataset):
    """
    Map names in NCAR to Skyrim
    """
    sfc_mapping = {
        "VAR_10U": "u10m",
        "VAR_10V": "v10m",
        "VAR_100U": "u100m",
        "VAR_100V": "v100m",
        "VAR_2T": "t2m",
        "SP": "sp",
        "MSL": "msl",
        "TCWV": "tcwv",
    }
    pl_mapping = {"U": "u", "V": "v", "Z": "z", "T": "t", "R": "r", "Q": "q"}
    coord_mapping = {"latitude": "lat", "longitude": "lon"}
    rename_mapping = {**pl_mapping, **sfc_mapping, **coord_mapping}
    ds = ds.rename(rename_mapping)
    return ds


def _transform_dataset_to_dataarray(
    ds: xr.Dataset, start_time_step: int = 0
) -> xr.DataArray:
    ds = _rename(ds)
    data_vars = []
    channel_names = []
    for var_name, da in ds.data_vars.items():
        if var_name == "utc_date":
            continue  # Skip the utc_date variable, redundant
        if "time" in da.dims:
            da = da.isel(time=start_time_step, drop=False)
        if "level" in da.dims:
            for level in skyrim_pl_levels:
                new_name = f"{var_name}{int(level)}"
                level_data = da.sel(level=level).drop("level")
                data_vars.append(level_data)
                channel_names.append(new_name)
        else:
            data_vars.append(da)
            channel_names.append(var_name)
    stacked_data = xr.concat(data_vars, dim="channel")
    stacked_data = stacked_data.assign_coords(channel=channel_names)
    common_dims = set.intersection(*[set(da.dims) for da in data_vars])
    stacked_data = stacked_data.transpose("channel", *sorted(common_dims))
    stacked_data["channel"] = stacked_data["channel"].astype(str)
    stacked_data.name = "IFS HRES 0.1"
    stacked_data = stacked_data.chunk("auto")  # can optimize
    return stacked_data


class NCARClient:
    def __init__(self):
        pass

    def list_files(
        self,
        analysis: str,
        start_time: datetime = None,
        date: date = None,
        month: date = None,
        filetype: str = "nc",
    ) -> list[str]:
        """List files for a given date or month. Month format is the first day of the month. Analysis is either sfc or pl"""
        dt = start_time or date or month
        if not dt:
            raise ValueError("You must provide a start_time, date or month")
        endpoint_suffix = {
            "pl": "26" if filetype == "nc" else "6",
            "sfc": "29" if filetype == "nc" else "9",
        }
        params_to_filter = {"pl": ncar_pl_vars, "sfc": ncar_sfc_vars}
        endpoint = f'{BASE_URL}/{dt.strftime("%Y%m")}{endpoint_suffix[analysis]}'
        filelist_html = requests.get(endpoint)
        files = {}
        for m in re.finditer(
            _generate_filename_pattern(analysis, filetype=filetype), filelist_html.text
        ):
            res = m.groupdict()
            url = filelist_html.text[m.start() : m.end()]
            start_time_ = res["start_time"]  # e.g. 2023122600
            if not files.get(start_time_):
                files[start_time_] = []
            files[start_time_].append({"url": url, "param": res["param"]})
        if start_time:
            files_ = files[
                start_time.strftime(f"%Y%m%d{'%H' if analysis == 'pl' else ''}")
            ]
        if date:
            files_ = [
                val for key, val in files.items() if date.strftime("%Y%m%d") in key
            ]
        if month:
            files_ = [val for key, val in files.items() if date.strftime("%Y%m") in key]
        return list(
            set([f["url"] for f in files_ if f["param"] in params_to_filter[analysis]])
        )

    def download(
        self, ic_start_time: datetime, dir: str = None, cache=True, filetype="nc"
    ):
        dir = dir or f'.data/ncar_ifs_{filetype}/{ic_start_time.strftime("%Y%m%d%H")}'
        if not Path(dir).exists():
            log.debug(f"Downloading IFS HRES to {dir}...")
            urls = self.list_files("pl", start_time=ic_start_time) + self.list_files(
                "sfc", start_time=ic_start_time
            )
            fast_fetch(urls, destination_folder=dir)
            log.debug("Completed!")
        else:
            log.debug(f"Using cached IFS HRES from {dir}")
        return dir

    def fetch(
        self,
        ic_start_time: datetime.date,
    ):
        time_steps = [0, 6, 12, 18]
        if not ic_start_time.hour in time_steps:
            raise ValueError("ic_start_time must be a multiple of 6")
        dir = self.download(ic_start_time)
        start_time_step = time_steps.index(ic_start_time.hour)
        return _transform_dataset_to_dataarray(
            xr.open_mfdataset(
                str(Path(dir) / "*.nc"),
                engine="netcdf4",
                chunks="auto",
            ),
            start_time_step,
        )

    def regrid(
        self,
        da: xr.DataArray,
        target_grid: str = "0p25",
        method: str = "conservative",
        use_cached_weights: bool = True,
    ) -> xr.DataArray:
        """
        Regrid 0.1 resolution to target. Target can be either 0p25 or 0p4-beta.
        Discussion on regridding: https://climatedataguide.ucar.edu/climate-tools/regridding-overview
        For going coarse, conservative is actually better, see [here](https://xesmf.readthedocs.io/en/stable/notebooks/Compare_algorithms.html)
        For real-world data, it is generally recommended to use conservative for decreasing the resolution,
        because it takes average over small source grid boxes, while bilinear and nearest_s2d effectively
        throw away most of source grid boxes.

        To test the differences you can use the following method:
        import xesmf as xe
        xe.util.compare_xy(data, regridded_data)

        """
        target_lat, target_lon = GRIDS[target_grid]
        target_grid = xr.Dataset(
            {"lat": (["lat"], target_lat), "lon": (["lon"], target_lon)}
        )
        if use_cached_weights:
            # Default filename: conservative_2560x5120_721x1440.nc
            weights_path = f"{method}_{da.sizes['lat']}x{da.sizes['lon']}_{target_grid.sizes['lat']}x{target_grid.sizes['lon']}.nc"
            if Path(weights_path).exists():
                log.debug(f"Loading from cached regridder weights: {weights_path}")
                regridder = xe.Regridder(
                    da, target_grid, method, filename=weights_path, reuse_weights=True
                )
            else:
                regridder = xe.Regridder(da, target_grid, method)
                log.debug(f"Saving regridder weights...")
                regridder.to_netcdf()
        else:
            # super slow to generate the grid (>10min)
            regridder = xe.Regridder(
                da,
                target_grid,
                method,
            )
        return regridder(da)
