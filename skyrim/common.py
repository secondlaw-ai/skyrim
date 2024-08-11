import hashlib
import base58
import time
import xarray as xr
import boto3
import s3fs
import os
import zarr
from datetime import datetime
from typing import Literal, Callable
from pathlib import Path
from loguru import logger
from urllib.parse import urlparse
from io import BytesIO
from dataclasses import dataclass, field
from huggingface_hub import upload_file, login

AVAILABLE_MODELS = ["pangu", "fourcastnet", "fourcastnet_v2", "graphcast", "dlwp"]
LOCAL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "skyrim")
OUTPUT_DIR = str(Path.cwd() / "outputs")


def generate_forecast_id(length=10):
    """
    Generate a unique forecast ID using the current time.
    """
    current_time = str(time.time()).encode()
    hash_object = hashlib.sha256(current_time)
    digest = hash_object.digest()
    base58_encoded = base58.b58encode(digest)
    return base58_encoded.decode()[:length]


@dataclass
class SaveConfig:
    forecast_id: str = ""
    output_dir: str = OUTPUT_DIR
    file_type: str = "netcdf"
    filter_vars: tuple[str] = ()
    mapping_func: Callable[[xr.DataArray], xr.DataArray] = lambda x: x
    zarr_store_config: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.forecast_id:
            self.forecast_id = generate_forecast_id()


def generate_filename(
    model: str,
    start_time: datetime,
    pred_time: datetime,
    ic_source: Literal["cds", "file", "ifs", "gfs"] = "cds",
):
    """
    Generate a filename for the forecast file.
    args:
    model: str: model name
    start_time: datetime: start time of the forecast
    pred_time: datetime: prediction time of the forecast
    ic_source: Literal["cds", "file", "ifs", "gfs"]: initial condition source
    """
    return (
        f"{model}"
        + "__"
        + f"{ic_source}__"
        + f"{start_time.strftime('%Y%m%d_%H:%M')}"
        + "__"
        + f"{pred_time.strftime('%Y%m%d_%H:%M')}.nc"
    )


def remote_forecast_exists(path: str):
    if not path.endswith("/"):
        path += "/"
    s3_client = boto3.client("s3")
    p = urlparse(path)
    return "Contents" in s3_client.list_objects_v2(
        Bucket=p.netloc, Prefix=p.path[1:], MaxKeys=1
    )


def to_hf(da: xr.DataArray, repo: str, path: str):
    """
    Writes an xarray DataArray to a Hugging Face repository.

    Parameters:
    da (xr.DataArray): The DataArray to be written to the repository.
    repo (str): The name of the Hugging Face repository.
    path (str): The path within the repository where the DataArray will be stored, no suffix.

    Note:
    1/ Hugging Face automatically deduplicates identical values, so naming cannot guarantee access.
    2/ Uploading without zip is not recommended, HF rate limits when too many files are uploaded.
    3/ There is a limit of 10K files per folder as well (apparently).
    TODO:
    Add compression to the data before writing.
    """
    login(token=os.environ["HF_TOKEN"])
    Path("./temp.zarr.zip").unlink(missing_ok=True)
    try:
        with zarr.ZipStore("./temp.zarr.zip", mode="w") as store:
            da.to_zarr(store, compute=True)
        upload_file(
            path_or_fileobj="./temp.zarr.zip",
            path_in_repo=path + ".zarr.zip",
            repo_id=repo,
            repo_type="dataset",
        )
    except Exception as e:
        raise e
    finally:
        Path("./temp.zarr.zip").unlink()


def save_forecast(
    pred: xr.DataArray,
    model_name: str,
    start_time: datetime,
    pred_time: datetime,
    source: Literal["cds", "file", "ifs", "gfs"] = "cds",
    config: dict = {},
):
    requested_file_type = config.get("file_type")
    config = SaveConfig(**config)
    p = urlparse(config.output_dir)
    target = p.scheme or "local"
    if target and (not requested_file_type):
        # for all remote targets, default support zarr
        config.file_type = "zarr"

    pred = config.mapping_func(pred)
    pred = pred[config.filter_vars] if len(config.filter_vars) else pred

    if target == "local":
        if config.file_type == "netcdf":
            filename = generate_filename(model_name, start_time, pred_time, source)
            output_path = Path(config.output_dir) / config.forecast_id / filename
            logger.info(f"Saving outputs to {output_path}")
            if not output_path.parent.exists():
                logger.info(
                    f"Creating parent directory to save outputs: {output_path.parent}"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
            pred.to_netcdf(output_path, engine="scipy")
        elif config.file_type == "zarr":
            output_path = str(Path(config.output_dir) / config.forecast_id)
            if Path(output_path).exists():
                pred.to_zarr(
                    output_path,
                    append_dim="step",
                    mode="a",
                    consolidated=True,
                )
            else:
                # cant use append dim.
                pred.to_zarr(
                    output_path,
                    mode="w",
                    consolidated=True,
                )
        else:
            raise ValueError(f"Invalid file type. {config.file_type} not supported.")
    elif target == "s3":
        bucket = p.netloc
        if config.file_type == "netcdf":
            filename = generate_filename(model_name, start_time, pred_time, source)
            output_path = os.path.join(p.geturl(), config.forecast_id, filename)
            s3 = boto3.client("s3")
            buf = BytesIO()
            pred.to_netcdf(buf, engine="scipy")
            buf.seek(0)
            s3.upload_fileobj(buf, bucket, output_path)
            buf.close()
        elif config.file_type == "zarr":
            fs = s3fs.S3FileSystem(anon=False)
            output_path = os.path.join(p.geturl(), config.forecast_id)
            if remote_forecast_exists(output_path):
                pred.to_zarr(
                    fs.get_mapper(output_path),
                    append_dim="time",
                    mode="a",
                    consolidated=True,
                    **config.zarr_store_config,
                )
            else:
                # cant use append dim.
                pred.to_zarr(
                    fs.get_mapper(output_path),
                    mode="w",
                    consolidated=True,
                    **config.zarr_store_config,
                )
        else:
            raise ValueError(f"Invalid file type. {config.file_type} not supported.")
    elif target == "hf":
        # only zarr supported for now.
        organization = p.netloc
        _, repo, *parent_path = p.path.split("/")
        parent_path = "/".join(parent_path)
        output_path = os.path.join(parent_path, config.forecast_id)
        to_hf(pred, "/".join((organization, repo)), output_path)

    logger.success(f"Results saved to: {output_path}")
    return str(output_path)
