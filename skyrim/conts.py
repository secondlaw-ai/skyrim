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
    "single": ["msl", "u10m", "v10m", "t2m"],
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


# ------------ FOURCASTNET ------------
CHANNELS = {
    # https://github.com/NVIDIA/earth2mip/blob/86b11fe4ba2f19641802112e8b0ba6b962123130/earth2mip/networks/fcn.py#L28-L55
    "fcn": [
        "u10m",
        "v10m",
        "t2m",
        "sp",
        "msl",
        "t850",
        "u1000",
        "v1000",
        "z1000",
        "u850",
        "v850",
        "z850",
        "u500",
        "v500",
        "z500",
        "t500",
        "z50",
        "r500",
        "r850",
        "tcwv",
        "u100m",
        "v100m",
        "u250",
        "v250",
        "z250",
        "t250",
    ],
    # https://github.com/NVIDIA/earth2mip/blob/86b11fe4ba2f19641802112e8b0ba6b962123130/earth2mip/networks/fcnv2_sm.py#L37-L111
    "fcnv2_sm": [
        "u10m",
        "v10m",
        "u100m",
        "v100m",
        "t2m",
        "sp",
        "msl",
        "tcwv",
        "u50",
        "u100",
        "u150",
        "u200",
        "u250",
        "u300",
        "u400",
        "u500",
        "u600",
        "u700",
        "u850",
        "u925",
        "u1000",
        "v50",
        "v100",
        "v150",
        "v200",
        "v250",
        "v300",
        "v400",
        "v500",
        "v600",
        "v700",
        "v850",
        "v925",
        "v1000",
        "z50",
        "z100",
        "z150",
        "z200",
        "z250",
        "z300",
        "z400",
        "z500",
        "z600",
        "z700",
        "z850",
        "z925",
        "z1000",
        "t50",
        "t100",
        "t150",
        "t200",
        "t250",
        "t300",
        "t400",
        "t500",
        "t600",
        "t700",
        "t850",
        "t925",
        "t1000",
        "r50",
        "r100",
        "r150",
        "r200",
        "r250",
        "r300",
        "r400",
        "r500",
        "r600",
        "r700",
        "r850",
        "r925",
        "r1000",
    ],
    "pangu": [
        "z1000",
        "z925",
        "z850",
        "z700",
        "z600",
        "z500",
        "z400",
        "z300",
        "z250",
        "z200",
        "z150",
        "z100",
        "z50",
        "q1000",
        "q925",
        "q850",
        "q700",
        "q600",
        "q500",
        "q400",
        "q300",
        "q250",
        "q200",
        "q150",
        "q100",
        "q50",
        "t1000",
        "t925",
        "t850",
        "t700",
        "t600",
        "t500",
        "t400",
        "t300",
        "t250",
        "t200",
        "t150",
        "t100",
        "t50",
        "u1000",
        "u925",
        "u850",
        "u700",
        "u600",
        "u500",
        "u400",
        "u300",
        "u250",
        "u200",
        "u150",
        "u100",
        "u50",
        "v1000",
        "v925",
        "v850",
        "v700",
        "v600",
        "v500",
        "v400",
        "v300",
        "v250",
        "v200",
        "v150",
        "v100",
        "v50",
        "msl",
        "u10m",
        "v10m",
        "t2m",
    ],
    "graphcast": [
        "z50",
        "z100",
        "z150",
        "z200",
        "z250",
        "z300",
        "z400",
        "z500",
        "z600",
        "z700",
        "z850",
        "z925",
        "z1000",
        "q50",
        "q100",
        "q150",
        "q200",
        "q250",
        "q300",
        "q400",
        "q500",
        "q600",
        "q700",
        "q850",
        "q925",
        "q1000",
        "t50",
        "t100",
        "t150",
        "t200",
        "t250",
        "t300",
        "t400",
        "t500",
        "t600",
        "t700",
        "t850",
        "t925",
        "t1000",
        "u50",
        "u100",
        "u150",
        "u200",
        "u250",
        "u300",
        "u400",
        "u500",
        "u600",
        "u700",
        "u850",
        "u925",
        "u1000",
        "v50",
        "v100",
        "v150",
        "v200",
        "v250",
        "v300",
        "v400",
        "v500",
        "v600",
        "v700",
        "v850",
        "v925",
        "v1000",
        "w50",
        "w100",
        "w150",
        "w200",
        "w250",
        "w300",
        "w400",
        "w500",
        "w600",
        "w700",
        "w850",
        "w925",
        "w1000",
        "u10m",
        "v10m",
        "t2m",
        "msl",
        "tp06",
    ],
}


def get_model_config(model_name: str):
    model_configs = {
        "pangu": PANGUWEATHER,
        "graphcast": GRAPHCAST,
        "fcnv2_sm": FOURCASTNETV2,
        "fcn": FOURCASTNET,
    }
    assert model_name in model_configs, f"Model {model_name} not found."
    return model_configs.get(model_name)


def create_channel_names(model_name):
    config = get_model_config(model_name)
    pressure_channels = [
        f"{var}{level}"
        for var in config["pressure"][0]
        for level in config["pressure"][1]
    ]

    surface_channels = [f"{var}" for var in config["single"]]
    channel_names = pressure_channels + surface_channels
    return channel_names


def get_cds_api_map(level_type: Literal["single", "pressure"]):
    return CDS_SINGLE_API_MAP if level_type == "single" else CDS_PRESSURE_API_MAP
