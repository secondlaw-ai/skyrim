import numpy as np
import xarray as xr
import subprocess
import pytest
from skyrim.common import save_forecast, generate_forecast_id
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfFileSystem


def mock_forecast():
    start_time = np.datetime64("2020-01-01T00:00:00")
    step = np.arange(
        start_time,
        start_time + np.timedelta64(2 * 6, "h"),
        step=np.timedelta64(6, "h"),
    )  # 2 steps in .
    time = np.array([start_time])
    lat = np.linspace(-90, 90, 181)
    lon = np.linspace(-180, 180, 361)
    temperature = 15 + 8 * np.random.randn(len(time), len(step), len(lat), len(lon))
    wind_speed = 10 * np.random.rand(len(time), len(step), len(lat), len(lon))
    return xr.Dataset(
        data_vars={
            "temperature": (["time", "step", "lat", "lon"], temperature),
            "wind_speed": (["time", "step", "lat", "lon"], wind_speed),
        },
        coords={"time": time, "lat": lat, "lon": lon, "step": step},
    )


def test_saves_forecast_locally():
    ds = mock_forecast()
    start_time = datetime.now()
    pred_time = datetime.now()
    fid = generate_forecast_id()
    output_path = save_forecast(
        ds,
        "test_model",
        start_time,
        pred_time,
        "cds",
        config={"forecast_id": fid, "file_type": "netcdf"},
    )
    assert Path(output_path).exists()
    assert fid in output_path
    assert "test_model" in output_path
    assert ".nc" in output_path
    ds = xr.open_dataset(output_path)
    assert ds.temperature.shape == (1, 2, 181, 361)
    # cleanup
    p = Path(output_path).parent
    Path(output_path).unlink()
    p.rmdir()


@pytest.mark.integ
def test_appends_forecast_in_s3():
    ds = mock_forecast()
    ds2 = ds.copy()
    ds2["step"] = ds2.step + np.timedelta64(3, "6h")  # 2 more steps
    start_time = datetime.now()
    pred_time = datetime.now()
    fid = generate_forecast_id()
    s3_path = "s3://skyrim-dev"
    # save first forecast
    assert ds.step.size == 2
    save_forecast(
        ds,
        "test_model",
        start_time,
        pred_time,
        "cds",
        config={
            "forecast_id": fid,
            "file_type": "zarr",
            "output_dir": s3_path,
        },
    )
    # append forecast
    save_forecast(
        ds2,
        "test_model",
        start_time,
        pred_time,
        "cds",
        config={
            "forecast_id": fid,
            "file_type": "zarr",
            "output_dir": s3_path,
        },
    )
    ds = xr.open_dataset(s3_path + "/" + fid, engine="zarr")
    assert ds.step.size == 4
    # cleanup
    c = subprocess.run(f"aws s3 rm s3://skyrim-dev/{fid}/ --recursive", shell=True)
    assert not c.returncode


@pytest.mark.integ
def test_saves_to_hf():
    ds = mock_forecast()
    start_time = datetime.now()
    pred_time = datetime.now()
    fid = "11"
    hf_repo = "2lw/samples"
    hf_url = "hf://" + hf_repo
    save_forecast(
        ds,
        "test_model",
        start_time,
        pred_time,
        "cds",
        config={
            "forecast_id": fid,
            "file_type": "zarr",
            "output_dir": hf_url,
        },
    )
    # how you open:
    hf_datasets_path = "hf://datasets/" + hf_repo
    filename = fid + ".zarr.zip"
    path = "zip:///::" + "/".join((hf_datasets_path, filename))
    ds = xr.open_dataset(path, engine="zarr")
    assert ds.temperature.shape == (1, 2, 181, 361)
    fs = HfFileSystem()
    fs.delete(hf_datasets_path + "/11.zarr.zip")
