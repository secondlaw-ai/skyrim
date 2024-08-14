import torch
from datetime import datetime, timedelta
from earth2mip.initial_conditions import (
    get_data_source as get_data_source_e2m,
    base,
    get_data_from_source,
)
from earth2mip import schema
from .ifs import DataSource as IfsDatasource
from earth2mip.grid import equiangular_lat_lon_grid


# fmt: off
    # https://github.com/NVIDIA/earth2mip/blob/86b11fe4ba2f19641802112e8b0ba6b962123130/earth2mip/networks/fcnv2_sm.py#L37-L111
CHANNELS = [ "u10m", "v10m", "u100m", "v100m", "t2m", "sp", "msl", "tcwv", "u50", "u100",
            "u150","u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925",
            "u1000", "v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600",
            "v700", "v850", "v925", "v1000", "z50", "z100", "z150", "z200", "z250", "z300",
            "z400", "z500", "z600", "z700", "z850", "z925", "z1000", "t50", "t100", "t150",
            "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925", "t1000",
            "r50", "r100", "r150", "r200", "r250", "r300", "r400", "r500", "r600", "r700",
            "r850", "r925","r1000",
    ]
# fmt: on
def get_data_source(
    channel_names: list[str],
    netcdf="",
    initial_condition_source=schema.InitialConditionSource.era5,
) -> base.DataSource:
    if initial_condition_source == schema.InitialConditionSource.ifs:
        return IfsDatasource(channel_names)
    return get_data_source_e2m(
        channel_names, initial_condition_source=initial_condition_source
    )


def get_ic(start_time: datetime, ic_source: str, channels=CHANNELS):
    """
    Utility method for getting IC through earth2mip data sources without initializing a model.
    """
    return get_data_from_source(
        data_source=get_data_source(
            CHANNELS, initial_condition_source=schema.InitialConditionSource("ifs")
        ),
        time=start_time,
        channel_names=CHANNELS,
        grid=equiangular_lat_lon_grid(721, 1440),
        n_history_levels=1,
        time_step=timedelta(hours=0),
        device=torch.device("cpu"),  # or torch.device('cuda', index=0)
        dtype=torch.float32,
    )
