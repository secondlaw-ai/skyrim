from typing import Literal

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

# ------------ PANGUWEATHER ------------
# input_surface.npy stores the input surface variables.
# It is a numpy array shaped (4,721,1440) where
# the first dimension represents the 4 surface variables
# (MSLP, U10, V10, T2M in the exact order).

# input_upper.npy stores the upper-air variables.
# It is a numpy array shaped (5,13,721,1440) where
# the first dimension represents the 5 surface variables
# (Z, Q, T, U and V in the exact order), and
# the second dimension represents the 13 pressure levels
# (1000hPa, 925hPa, 850hPa, 700hPa, 600hPa, 500hPa, 400hPa, 300hPa, 250hPa, 200hPa, 150hPa, 100hPa and 50hPa in the exact order).

# Note that Z represents geopotential, not geopotential height,
# so a factor of 9.80665 should be multiplied
# if the raw data contains the geopotential height.
PANGUWEATHER = {
    "single": ["msl", "10u", "10v", "2t"],
    "pressure": (
        ["z", "q", "t", "u", "v"],
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    ),
}

# ------------ GRAPHCAST ------------
GRAPHCAST = {
    "single": ["lsm", "2t", "msl", "10u", "10v", "tp", "z"],
    "pressure": (
        ["t", "z", "u", "v", "w", "q"],
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    ),
    "F": ["toa_incident_solar_radiation"],
}


# ------------ FOURCASTNETV2 ------------
FOURCASTNETV2 = {
    "single": ["10u", "10v", "2t", "sp", "msl", "tcwv", "100u", "100v"],
    "pressure": (
        ["t", "u", "v", "z", "r"],
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    ),
}

# ------------ FOURCASTNET ------------
FOURCASTNET = {
    "single": ["10u", "10v", "2t", "sp", "msl", "tcwv"],
    "pressure": (
        ["t", "u", "v", "z", "r"],
        [1000, 850, 500, 50],
    ),
}


def get_model_config(model_name: str):
    model_configs = {
        "PANGUWEATHER": PANGUWEATHER,
        "GRAPHCAST": GRAPHCAST,
        "FOURCASTNETV2": FOURCASTNETV2,
        "FOURCASTNET": FOURCASTNET,
    }
    assert model_name.upper() in model_configs, f"Model {model_name} not found."
    return model_configs.get(model_name.upper())


def get_cds_api_map(level_type: Literal["single", "pressure"]):
    return CDS_SINGLE_API_MAP if level_type == "single" else CDS_PRESSURE_API_MAP
