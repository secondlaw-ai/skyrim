<h1 align="center">
 <a href="https://www.secondlaw.xyz">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/skyrim_banner_1.png"/>
    <img height="auto" width="90%" src="./assets/skyrim_banner_1.png"/>
  </picture>
 </a>
 <br></br>

</h1>
<div align="center">

🔥 Run state-of-the-art large weather models in less than 2 minutes.

🌪️ Ensemble and fine-tune (soon) to push the limits on forecasting.

🌎 Simulate extreme weather events!

</div>
<div>
<div align="center">

[![Static Badge](https://img.shields.io/badge/Run-On_Modal-green)](https://github.com/secondlaw-ai/skyrim?tab=readme-ov-file#forecasting-using-modal-recommended)
[![PyPI - Version](https://img.shields.io/pypi/v/skyrim)](https://pypi.org/project/Skyrim/)
[![Twitter Follow](https://img.shields.io/twitter/follow/secondlaw_ai)](https://twitter.com/secondlaw_ai)
![GitHub Repo stars](https://img.shields.io/github/stars/secondlaw-ai/skyrim)
[![GitHub License](https://img.shields.io/github/license/secondlaw-ai/skyrim)](https://github.com/secondlaw-ai/skyrim/blob/master/LICENSE)

</div>
</div>

# Getting Started

Skyrim allows you to run any large weather model with a consumer grade GPU.

Until very recently, weather forecasts were run in 100K+ CPU HPC clusters, solving massive numerical weather models (NWP). Within last 2 years, open-source foundation models trained on weather simulation datasets surpassed the skill level of these numerical models.

Our goal is to make these models accessible by providing a well maintained infrastructure.

## Installation

Clone the repo, set an env (either conda or venv) and then run

```bash
git clone https://github.com/your-repo/skyrim.git
cd skyrim
pip install .
```

Depending on your use-case (i.e. AWS storage needs or CDS initial conditions), you may need to fill in a `.env` by `cp .env.example .env`.

## Running Your First Forecast

Skyrim currently supports either running on on [modal](#forecasting-using-modal), on a container –for instance [vast.ai](#vastai-setup) or [bare metal](#bare-metal)(you will need an NVIDIA GPU with at least 24GB and installation can be long).

Modal is the fastest option, it will run forecasts "serverless" so you don't have to worry about the infrastructure.

### Forecasting using Modal (Recommended):

You will need a [modal](https://modal.com/) key. Run `modal setup` and set it up (<1 min).

Modal comes with $30 free credits and a single forecast costs about 2 cents as of May 2024.

Once you are all good to go, then run:

```bash
modal run skyrim/modal/forecast.py
```

This by default uses `pangu` model to forecast for the next 6 hours, starting from yesterday. It gets initial conditions from [NOAA GFS](https://en.wikipedia.org/wiki/Global_Forecast_System) and writes the forecast to a modal volume. You can choose different dates and weather models as shown in [here](#run-forecasts-with-different-models-initial-conditions-dates).

After you have your forecast, you can explore it by running a notebook (without GPU, so cheap) in modal:

```bash
modal run skyrim/modal/forecast.py::run_analysis
```

This will output a jupyter notebook link that you can follow and access the forecast. For instance, to read the forecast you can run from the notebook the following:

```
import xarray as xr
forecast = xr.open_dataset('/skyrim/outputs/[forecast_id]/[filename], engine='scipy')
```

Once you are done, best is to delete the volume as a daily forecast is about 2GB:

```bash
modal volume rm forecasts /[forecast_id] -r
```

If you don't want to use modal volume, and want to aggregate results in a bucket (currently only s3), you just have to run:

```bash
modal run skyrim/modal/forecast.py --output_dir s3://skyrim-dev
```

where `skyrim-dev` is the bucket that you want to aggregate the forecasts. By default, `zarr` format is used to store in AWS/GCP so you can read and move only the parts of the forecasts that you need.

See [examples](#examples) section for more.✌️

### Forecasting with your own GPUs:

If you are running on your own GPUs, installed either via [bare metal](#bare-metal) or via containers such as [vast.ai](#vast-ai-setup) then you can directly get forecasts as such:

```python
from skyrim.core import Skyrim

model = Skyrim("pangu")
final_pred, pred_paths = model.predict(
    date="20240507", # format: YYYYMMDD, start date of the forecast
    time="0000",  # format: HHMM, start time of the forecast
    lead_time=24 * 7, # in hours, next week
    save=True,
)
```

To visualise the forecast:

```python
from skyrim.libs.plotting import visualize_rollout
visualize_rollout(output_paths=pred_paths, channels=["u10m", "v10m"], output_dir=".")
```

<p align="center">
  <img src="./notebooks/pangu_20240507_00:00_to_20240514_00:00_u10m.gif" width="30%" />
  <img src="./notebooks/pangu_20240507_00:00_to_20240514_00:00_v10m.gif" width="30%" /> 
</p>

or you can still use the command line:

```bash
forecast -m graphcast --lead_time 24 --initial_conditions cds --date 20240330`
```

See [examples](#examples) section for more.✌️

#### vast.ai setup

1. Find a machine you like RTX3090 or above with at least 24GB memory. Make sure you have good bandwith (+500MB/s).
2. Select the instance template from [here](https://cloud.vast.ai/?ref_id=128656&template_id=1883215a8487ec6ea9ad68a7cdb38c5e).
3. Then clone the repo and `pip install . && pip install -r requirements.txt`

#### Bare metal

1. You will need a NVIDIA GPU with at least 24GB. We are working on quantization as well so that in the future it would be possible to run simulations with much less compute. Have an environment set with Python == 3.10, Pytorch => 2.2.2 and CUDA +12.x. Or if easier start with the docker image: `nvcr.io/nvidia/pytorch:24.01-py3`.
2. Install conda ([miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) for instance). Then run in that environment:

```bash
conda create -y -n skyenv python=3.10
conda activate skyenv
conda install eccodes python-eccodes -c conda-forge
pip install . && pip install -r requirements.txt
```

## Examples

For each run, you will first pull the initial conditions of your interest (most recent one by default), then the model will run for the desired time step. Initial conditions are pulled from GFS, ECMWF IFS (Operational) or CDS (ERA5 Reanalysis Dataset).

If you are using CDS initial conditions, then you will need a [CDS](https://cds.climate.copernicus.eu/user/login?destination=%2Fcdsapp%23!%2Fdataset%2Freanalysis-era5-single-levels) API key in your `.env` –`cp .env.example` and paste.

All examples can be run using `forecast` or `modal run skyrim/modal/forecast.py`. You just have to make snake case options kebab-case -i.e. `model_name` to `model-name`.

### Example 1: Pick models, initial conditions, lead times

Forecast using `graphcast` model, with ECMWF IFS initial conditions, starting from 2024-04-30T00:00:00 and with a lead time of a week (forecast for the next week, i.e. 168 hours):

```bash
forecast --model_name graphcast --initial_conditions ifs --date 20240403 -output_dir s3://skyrim-dev --lead_time 168
```

or in modal:

```bash
modal run skyrim/modal/forecast.py --model-name graphcast --initial-conditions ifs --date 20240403 --output-dir s3://skyrim-dev --lead-time 168
```

### Example 2: Store in AWS and then read only what you need

Say you re interested in wind at 37.0344° N, 27.4305 E to see if we can kite tomorrow. If we need wind speed, we need to pull wind vectors at about surface level, these are u10m and v10m [components](http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv) of wind. Here is how you go about it:

```bash
modal run skyrim/modal/forecast.py --output-dir s3://[your_bucket]/[optional_path]  --lead-time 24
```

Then you can read the forecast as below:

```python
import xarray as xr
import pandas as pd
zarr_store_path = "s3://[your_bucket]/[forecast_id]"
forecast = xr.open_dataset(zarr_store_path, engine='zarr') # reads the metadata
df = forecast.sel(lat=37.0344, lon=27.4305, channel=['u10m', 'v10m']).to_pandas()
```

Normally each day is about 2GB but using zarr_store you will only fetch what you need.✌️

### Example 3: Get predictions in Python

Assuming you have a local gpu set up ready to roll:

```python
from skyrim.core import Skyrim

model = Skyrim("pangu")
final_pred, pred_paths = model.predict(
    date="20240501", # format: YYYYMMDD, start date of the forecast
    time="0000",  # format: HHMM, start time of the forecast
    lead_time=12, # in hours
    save=True,
)
akyaka_coords = {"lat": 37.0557, "lon": 28.3242}
wind_speed = final_pred.wind_speed(**akyaka_coords) * 1.94384 # m/s to knots
print(f"Wind speed at Akyaka: {wind_speed:.2f} knots")

```

## Supported initial conditions and caveats

1. NOAA GFS
2. ECMWF IFS
3. ERA5 Re-analysis Dataset

## Large weather models supported

Currently supported models are:

- [x] [Graphcast](https://arxiv.org/abs/2212.12794)
- [x] [Pangu](https://arxiv.org/abs/2211.02556)
- [x] [Fourcastnet](https://arxiv.org/abs/2202.11214) (v1 & v2)
- [x] [DLWP](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020MS002109)
- [x] [(NWP) ECMWF IFS (HRES)](https://www.ecmwf.int/en/forecasts/documentation-and-support/changes-ecmwf-model) -- [notebook](https://github.com/secondlaw-ai/skyrim/blob/master/notebooks/02_ifs_model.ipynb)
- [x] [(NWP) NOAA GFS](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast) -- [notebook](https://github.com/secondlaw-ai/skyrim/blob/master/notebooks/03_gfs_model.ipynb)
- [ ] [(NWP) ICON](https://www.dwd.de/EN/ourservices/nwp_forecast_data/nwp_forecast_data.html)
- [ ] [Fuxi](https://www.nature.com/articles/s41612-023-00512-1)
- [ ] [Nano MetNet](https://arxiv.org/abs/2306.06079)

### License

For detailed information regarding licensing, please refer to the license details provided on each model's main homepage, which we link to from each of the corresponding components within our repository.

- **Pangu Weather** : [Original](https://github.com/198808xc/Pangu-Weather), [ECMWF](https://github.com/ecmwf-lab/ai-models-panguweather), [NVIDIA](https://github.com/NVIDIA/earth2mip)

- **FourcastNet** : [Original](https://github.com/NVlabs/FourCastNet), [ECMWF](https://github.com/ecmwf-lab/ai-models-fourcastnetv2),[NVIDIA](https://github.com/NVIDIA/earth2mip)

- **Graphcast** : [Original](https://github.com/google-deepmind/graphcast), [ECMWF](https://github.com/ecmwf-lab/ai-models-graphcast), [NVIDIA](https://github.com/NVIDIA/earth2mip)

- **Fuxi**: [Original](https://github.com/tpys/FuXi)

## Roadmap

- [x] ensemble prediction
- [x] interface to fetch real-time NWP-based predictions, e.g. via ECMWF API.
- [ ] global model performance comparison across various regions and parameters.
- [ ] finetuning api that trains a downstream model on top of features coming from a global/foundation model, that is optimized wrt to a specific criteria and region
- [ ] model quantization and its effect on model efficiency and accuracy.

This README will be updated regularly to reflect the progress and integration of new models or features into the library. It serves as a guide for internal development efforts and aids in prioritizing tasks and milestones.

## Development

All in [here](./CONTRIBUTING.md) ✌️

## Acknowledgements

Skyrim is built on top of NVIDIA's [earth2mip](https://github.com/NVIDIA/earth2mip) and ECMWF's [ai-models](https://github.com/ecmwf-lab/ai-models). Definitely check them out!

## Other Useful Resources

- [🌍 Awesome Large Weather Models (LWMs) | AI for Earth (AI4Earth) | AI for Science (AI4Science)](http://github.com/jaychempan/Awesome-LWMs)
- [Climate Data Store](https://cds.climate.copernicus.eu/)
- [Open Climate Fix](https://github.com/openclimatefix)
- [Herbie](https://github.com/blaylockbk/Herbie)
