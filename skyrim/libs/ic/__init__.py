from earth2mip.initial_conditions import get_data_source as get_data_source_e2m, base
from earth2mip import schema
from .ifs import DataSource as IfsDatasource


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
