import cdsapi
from loguru import logger
import subprocess
from conts import get_model_config, get_cds_api_map, CDS_LEVELS
from typing import Literal, Optional
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


CACHE_DIR = Path(".cache/era5_ics")
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

LEVEL_TYPE_MAP = {
    "reanalysis-era5-single-levels": "single",
    "reanalysis-era5-pressure-levels": "pressure",
}


def fetch_cds_level_data(
    level_name: Literal[
        "reanalysis-era5-single-levels", "reanalysis-era5-pressure-levels"
    ],
    year: str,  # Format YYYY
    month: str,  # Format MM
    day: str,  # Formar DD
    time: int,  # Hour in 24 hour format, e.g. 12
    output_path: str = None,
    variable: Optional[list] = None,
    pressure_level: Optional[list] = None,
    product_type: Literal["reanalysis"] = "reanalysis",
    format: Literal["grib"] = "grib",
    use_cache: bool = True,
) -> str:
    """
    Fetches data for a specified level from the CDS API and saves it to a GRIB file.

    If data already exists in the cache for the specified date and time, the download is skipped,
    and the path to the existing file is returned. Otherwise, data is fetched from the CDS API.

    Args:
        level_name: The dataset level type, either single levels or pressure levels.
        year: The year of the data to fetch, format YYYY.
        month: The month of the data to fetch, format MM.
        day: The day of the data to fetch, format DD.
        time: The hour of the day (in 24-hour format) of the data to fetch, e.g. 12.
        output_path: The path where the fetched data should be saved. If None, the data is saved to the cache directory.
        variable: A list of variables to fetch. If None, all variables for the level are fetched.
        pressure_level: Specific pressure levels to fetch for pressure level data.
        product_type: The type of product, fixed to "reanalysis".
        format: The format of the fetched data, fixed to "grib".

    Returns:
        The path to the saved GRIB file containing the fetched data.

    TODO: Add support for fetching data for multiple times in a single request.
    """

    time = f"{time:02d}:00"

    if not output_path:
        output_path = (
            CACHE_DIR
            / "raw"
            / f"date={year}{month}{day}__time={time}__level={LEVEL_TYPE_MAP[level_name]}.grib"
        )
        logger.debug(f"Output path not provided, saving to {output_path}")

    if use_cache and Path(output_path).exists():
        logger.warning(f"Data already exists in cache, skipping download")
        return output_path

    logger.info(f"Fetching data for {level_name} level type")

    params = [p for p in get_cds_api_map(LEVEL_TYPE_MAP[level_name]).values()]
    levels = CDS_LEVELS if LEVEL_TYPE_MAP[level_name] == "pressure" else None

    logger.debug(f"Params: {params}, Levels: {levels}")

    c = cdsapi.Client()

    c.retrieve(
        name=level_name,
        request={
            "product_type": product_type,
            "format": format,
            "variable": params if not variable else variable,
            "pressure_level": levels if not pressure_level else pressure_level,
            "year": year,
            "month": month,
            "day": day,
            "time": time,
        },
        target=output_path,
    )

    return output_path



def merge_files(output_file_path: str, input_file_paths: list[str]):
    """
    Merges multiple files into a single file using the `cat` command via subprocess.
    This is generally not recommended for binary files like GRIB without ensuring compatibility.
    """
    logger.debug(f"Merging {len(input_file_paths)} files into {output_file_path}")
    breakpoint()
    with output_file_path.open("wb") as merged_file:
        # Ensure the file path is a string for subprocess
        subprocess.run(["cat", *input_file_paths], stdout=merged_file, check=True)
    logger.success(
        f"Merged single-level and pressure-level data into {output_file_path}"
    )


def fetch_model_ics(model_name: str, date: str, time: int = 12):
    """
    Fetches initial condition files for a given model, date, and time.
    Downloads each dataset to a separate file and then merges all into a single final file.
    """
    model_config = get_model_config(model_name)
    time_str = f"{time:02d}:00"
    year, month, day = date[:4], date[4:6], date[6:8]

    ic_file_path = CACHE_DIR / model_name.lower() / f"date={date}_time={time_str}.grib"
    ic_file_path.parent.mkdir(parents=True, exist_ok=True)

    if ic_file_path.exists():
        logger.info(f"Initial model state already exists, skipping fetch")
        return ic_file_path

    downloaded_files = []
    try:
        for level_name, level_type in LEVEL_TYPE_MAP.items():
            temp_file_path = CACHE_DIR / f"temp_{level_name}_{date}_{time_str}.grib"
            if level_type == "pressure":
                model_vars, model_lvls = model_config.get(level_type)
            elif level_type == "single":
                model_vars = model_config.get(level_type)
                model_lvls = None
            else:
                raise ValueError(f"Invalid level type: {level_type}")

            logger.info(f"Fetching {level_type}-level data for {model_name}")

            fetch_cds_level_data(
                level_name=level_name,
                year=year,
                month=month,
                day=day,
                time=time,
                output_path=str(temp_file_path),
                variable=[get_cds_api_map(level_type).get(var) for var in model_vars],
                pressure_level=model_lvls,
                use_cache=False,
            )

            downloaded_files.append(temp_file_path)
            logger.success(f"Fetched {level_type}-level data for {model_name}")

        # Merge the downloaded files into the final IC file
        merge_files(ic_file_path, downloaded_files)

    finally:
        # Cleanup: Delete the downloaded temporary files
        for file_path in downloaded_files:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted temporary file: {file_path}")

    logger.info(f"Initial model state prepared and saved to {ic_file_path}")
    return ic_file_path


if __name__ == "__main__":
    # fetch_cds_level_data(
    #     level_type="reanalysis-era5-single-levels",
    #     year="2024",
    #     month="01",
    #     day="01",
    #     time=12,
    # )

    fetch_model_ics(model_name="PANGUWEATHER", date="20240104", time=12)
