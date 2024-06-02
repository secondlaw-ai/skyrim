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



