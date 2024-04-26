from typing import Literal
from enum import Enum
# sl: surface and single-level parameters
# pl: pressure level parameters

CDS_SINGLE_API_MAP = {
    "msl": "mean_sea_level_pressure",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "2t": "2m_temperature",
    "lsm": "land_sea_mask",
    # "100u": "100m_u-component_of_wind",
    #   "100v": "100m_v-component_of_wind",
    "sp": "surface_pressure",
}

CDS_PRESSURE_API_MAP = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "r": "relative_humidity",
}

CDS_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
CDS_API_MAP = CDS_SINGLE_API_MAP | CDS_PRESSURE_API_MAP

IcProvider = Enum(value='IcProvider', names=[("CDS", "cds"), ("IFS","ifs"), ("GFS","gfs"), ("HRMIP","hrmip"), ("HDF5","hdf5")])

def get_cds_api_map(level_type: Literal["single", "pressure"]):
    return CDS_SINGLE_API_MAP if level_type == "single" else CDS_PRESSURE_API_MAP
