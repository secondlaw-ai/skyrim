import time
import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug(f"{func.__name__} executed in {end - start:.4f} seconds")
        return result

    return wrapper


def ensure_cds_loaded():
    """Currently, earth2mip requires CDS env to be loaded in /root/.cdsapi"""
    if Path("~/.cdsapirc").exists():
        return True
    else:
        load_dotenv()
        cds_key = os.environ.get("CDSAPI_KEY")
        cds_url = os.environ.get("CDSAPI_URL")
        logger.info(f"Gathering CDS API key from environment...")
        if not cds_key:
            raise Exception("CDS API config not found in the environment.")
        Path("~/.cdsapirc").write_text(f"key: {cds_key}\nurl: {cds_url}")
        logger.success(f"Successfully wrote CDS API key to /root/.cdsapi")


def ensure_ecmwf_loaded():
    """
    To be able to use ecmwf's API for IFS, etc.
    we need to have the config file in /root/.ecmwfapirc
    """

    config_path = Path("/root/.ecmwfapirc")
    if config_path.exists():
        return True

    load_dotenv()
    ecmwf_key = os.environ.get("ECMWF_API_KEY")
    ecmwf_url = os.environ.get("ECMWF_API_URL")
    ecmwf_email = os.environ.get("ECMWF_API_EMAIL")
    logger.info(f"Gathering ECMWF API key from environment...")

    if not all([ecmwf_key, ecmwf_url, ecmwf_email]):
        raise Exception("ECMWF API config not found in the environment.")

    config_content = f'{{\n  "url": "{ecmwf_url}",\n  "key": "{ecmwf_key}",\n  "email": "{ecmwf_email}"\n}}'
    config_path.write_text(config_content)
    logger.info("ECMWF API config file created at /root/.ecmwfapirc.")
