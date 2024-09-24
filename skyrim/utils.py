import os
import numpy as np
import datetime
from dotenv import load_dotenv
from pathlib import Path
from loguru import logger
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


import os
import aiohttp
import aiofiles
import asyncio
from tqdm.asyncio import tqdm


async def download_file_async(
    session: aiohttp.ClientSession,
    file_url: str,
    destination_folder: str,
    chunk_size: int = 1024
    * 1024
    * 3,  # for smaller files can reduce this, or increase if you have fast connection
):
    local_filename = os.path.join(destination_folder, file_url.split("/")[-1])
    async with session.get(file_url) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        async with aiofiles.open(local_filename, "wb") as f:
            progress = tqdm(total=total_size, unit="iB", unit_scale=True)
            async for chunk in response.content.iter_chunked(chunk_size):
                if chunk:
                    await f.write(chunk)
                    progress.update(len(chunk))
            progress.close()
    return local_filename


async def fast_fetch_async(urls: list[str], destination_folder: str = ".data"):
    # Ensure destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for file_url in urls:
            tasks.append(download_file_async(session, file_url, destination_folder))
        return await asyncio.gather(*tasks)


def fast_fetch(urls: list[str], destination_folder: str = ".data"):
    """Fast fetch multiple files and download them to a destination folder."""
    return asyncio.run(fast_fetch_async(urls, destination_folder))


def large_download(url, destination, chunk_size=1024 * 1024):  # 1 MB chunk size
    """Large file download with progress bar and resume support."""
    if os.path.exists(destination):
        resume_header = {"Range": f"bytes={os.path.getsize(destination)}-"}
    else:
        resume_header = None

    with requests.get(url, headers=resume_header, stream=True) as response:
        if response.status_code == 206:
            print("Resuming download...")
        elif response.status_code != 200:
            response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        mode = "ab" if resume_header else "wb"
        with open(destination, mode) as f:
            progress = tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                initial=os.path.getsize(destination) if resume_header else 0,
            )
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filter out keep-alive new chunks
                    f.write(chunk)
                    progress.update(len(chunk))
            progress.close()
    print(f"Download completed: {destination}")
    return destination
