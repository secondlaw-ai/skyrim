from dotenv import load_dotenv
from pathlib import Path
import os
from loguru import logger
import numpy as np
import datetime


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
    if Path("~/.cdsapirc").expanduser().exists():
        return True
    else:
        load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
        cds_key = os.environ.get("CDSAPI_KEY")
        cds_url = os.environ.get("CDSAPI_URL")
        logger.info(f"Gathering CDS API key from environment...")
        if not cds_key:
            raise Exception("CDS API config not found in the environment.")
        Path("~/.cdsapirc").expanduser().write_text(f"key: {cds_key}\nurl: {cds_url}")
        logger.success(f"Successfully wrote CDS API key to /root/.cdsapi")


def np_datetime64_to_datetime(time):
    """
    Converts a numpy.datetime64 object or an array of numpy.datetime64 objects to datetime.datetime objects.

    Parameters:
    - time: numpy.datetime64 object or array of numpy.datetime64 objects to be converted

    Returns:
    - A datetime.datetime object or a list of datetime.datetime objects
    """

    def convert_single_time(single_time):
        _unix = np.datetime64(0, "s")  # Unix epoch start time
        _ds = np.timedelta64(1, "s")  # One second time delta
        return datetime.datetime.utcfromtimestamp((single_time - _unix) / _ds)

    if isinstance(time, np.datetime64):
        return convert_single_time(time)
    elif isinstance(time, np.ndarray) and time.dtype == "datetime64[ns]":
        return [convert_single_time(t) for t in time]
    else:
        raise TypeError(
            "The provided time must be a numpy.datetime64 object or an array of numpy.datetime64 objects"
        )


if __name__ == "__main__":
    print(Path(__file__).parent.parent / ".env")
