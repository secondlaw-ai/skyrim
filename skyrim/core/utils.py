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
    if Path("/root/.cdsapi").exists():
        return True
    else:
        load_dotenv()
        cds_key = os.environ.get("CDSAPI_KEY")
        cds_url = os.environ.get("CDSAPI_URL")
        if not cds_key:
            raise Exception("CDS API config not found in the environment.")
        Path("/root/.cdsapi").write_text(f"key: {cds_key}\nurl: {cds_url}")
