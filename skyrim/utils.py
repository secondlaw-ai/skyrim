from dotenv import load_dotenv
from pathlib import Path
import os
from loguru import logger
import numpy as np
import datetime
from typing import List


def ensure_ecmwf_loaded():
    """
    To be able to use ecmwf's API for IFS, etc.
    we need to have the config file in /root/.ecmwfapirc
    """

    config_path = Path("~/.ecmwfapirc").expanduser()
    if config_path.exists():
        return True

    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
    ecmwf_key = os.environ.get("ECMWF_API_KEY")
    ecmwf_url = os.environ.get("ECMWF_API_URL")
    ecmwf_email = os.environ.get("ECMWF_API_EMAIL")
    logger.info(f"Gathering ECMWF API key from environment...")

    if not all([ecmwf_key, ecmwf_url, ecmwf_email]):
        raise Exception("ECMWF API config not found in the environment.")

    config_content = f'{{\n  "url": "{ecmwf_url}",\n  "key": "{ecmwf_key}",\n  "email": "{ecmwf_email}"\n}}'
    config_path.write_text(config_content)
    logger.info("ECMWF API config file created at /root/.ecmwfapirc.")


def ensure_cds_loaded():
    """Currently, earth2mip requires CDS env to be loaded in /root/.cdsapi"""
    cdsapirc_path = Path("~/.cdsapirc").expanduser()
    if cdsapirc_path.exists():
        logger.info(f"CDS API token already exists at {cdsapirc_path}")
    else:
        load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
        cds_key = os.environ.get("CDSAPI_KEY")
        cds_url = os.environ.get("CDSAPI_URL")
        logger.debug(f"cds_key: {cds_key}, cds_url: {cds_url}")
        logger.info(f"Gathering CDS API key from environment...")
        if not cds_key:
            raise Exception("CDS API config not found in the environment.")
        cdsapirc_path.write_text(f"key: {cds_key}\nurl: {cds_url}")
        logger.success(f"Successfully wrote CDS API token {cdsapirc_path}")


def convert_datetime64_to_datetime(
    datetime64_array: np.ndarray,
) -> List[datetime.datetime]:
    """
    Convert a NumPy array of datetime64[ns] objects to a list of datetime.datetime objects.

    Parameters
    ----------
    datetime64_array : np.ndarray
        A NumPy array of datetime64[ns] objects.

    Returns
    -------
    List[datetime.datetime]
        A list of datetime.datetime objects.
    """
    # Convert to an array of datetime.datetime objects
    datetime_array = datetime64_array.astype("datetime64[s]").astype(datetime.datetime)
    # Convert to a list of datetime.datetime objects
    return list(datetime_array)


def convert_datetime64_to_str(
    datetime64_array: np.ndarray, fmt: str = "%Y%m%dT%H%M"
) -> List[str]:
    """
    Convert a NumPy array of datetime64[ns] objects to a list of strings in the specified format.

    Parameters
    ----------
    datetime64_array : np.ndarray
        A NumPy array of datetime64[ns] objects.
    fmt : str, optional
        The format string to use for conversion, by default '%Y%m%dT%H%M'.

    Returns
    -------
    List[str]
        A list of strings formatted according to the specified format.
    """
    # Convert to an array of datetime.datetime objects
    datetime_array = datetime64_array.astype("datetime64[s]").astype(datetime.datetime)
    # Convert each datetime.datetime object to the desired string format
    datetime_str_list = [dt.strftime(fmt) for dt in datetime_array]
    return datetime_str_list


def convert_datetime_to_str(
    datetime_array: List[datetime.datetime], fmt: str = "%Y%m%dT%H%M"
) -> List[str]:
    """
    Convert a list of datetime.datetime objects to a list of strings in the specified format.

    Parameters
    ----------
    datetime_array : List[datetime.datetime]
        A list of datetime.datetime objects.
    fmt : str, optional
        The format string to use for conversion, by default '%Y%m%dT%H%M'.

    Returns
    -------
    List[str]
        A list of strings formatted according to the specified format.
    """
    # Convert each datetime.datetime object to the desired string format
    datetime_str_list = [dt.strftime(fmt) for dt in datetime_array]
    return datetime_str_list
