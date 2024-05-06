<h1 align="center">
 <a href="https://www.secondlaw.xyz">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/skyrim_banner_1.png"/>
    <img height="auto" width="90%" src="./assets/skyrim_banner_1.png"/>
  </picture>
 </a>
 <br></br>

</h1>
<p align="center">

🔥 Run state-of-the-art large weather models in less than 2 minutes.

🌪️ Ensemble and fine-tune to push the limits on forecasting.

🌎 Simulate extreme weather events!

</p>

# Getting Started

Skyrim allows you to run any large weather model with a consumer grade GPU.

Until very recently, weather forecasts were run in 100K+ CPU HPC clusters, solving massive numerical models. Within last 2 years, open-source foundation models trained on weather simulation datasets surpassed the skill level of these numerical models.

Our goal is to make these models accessible by providing a well maintained infrastructure.

## Installation

Clone the repo, set an env (either conda or venv) and then run `pip install .`.

This will install bare-minimum to run your first forecast.

## Run your first forecast

You will need a [modal](https://modal.com/) key to run your forecast as we are loading large weather models (it requires NVIDIA GPU with at least 24GB memory). Modal comes with $30 free credits and a single forecast costs about 2 cents. Alternatively, see the [bare metal](#bare-metal) or [vast.ai](#vastai-setup) setup to run on your own GPUs.

### Forecasting using Modal:

If you are running on modal then run:

```bash
modal run modal/forecast.py
```

This by default uses `pangu` model to forecast for the next 6 hours, starting from yesterday. It gets initial conditions from NOAA GFS and writes the forecast to a modal volume. You can explore the forecast by running a notebook (without GPU) in modal:

```bash
modal run modal/forecast.py:run_analysis
```

The forecast will be at `/skyrim/outputs/` volume that you can access from the jupyter notebook.

```
import xarray as xr
forecast = xr.open_dataset('/skyrim/outputs/[forecast_id]/[filename], engine='scipy')
```

Once you are done, best is to delete the volume as a daily forecast is about 2GB:

```bash
modal volume rm forecasts /[model_name] -r
```

If you don't want to use modal volume, and want to aggregate results in cloud, we currently support s3 buckets. You just have to run:

```bash
modal run modal/forecast.py --output_dir s3://skyrim-dev
```

where `skyrim-dev` is the bucket that you want to aggregate the forecasts. By default, `zarr` format is used to store in AWS/GCP so you can read and move only the parts of the forecasts that you need.

Say interested in wind at 37.0344° N, 27.4305 E to see if we can kite. If we are interested in wind speed, we need to pull wind vectors at about surface level, these are u10m and v10m [components](http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv) of wind. Here is how you do it:

```python
import xarray as xr
import pandas as pd
zarr_store_path = "s3://skyrim-dev/[forecast_id]"
forecast = xr.open_dataset(zarr_store_path, engine='zarr') # reads the metadata
df = forecast.sel(lat=37.0344, lon=27.4305, channel=['u10m', 'v10m']).to_pandas()
```

### Forecasting with your own GPUs:

If you are running on your own GPUs, installed either via [bare metal](#bare-metal) or via [vast.ai](#vast-ai-setup) then you can just run:

`forecast`

or you can pass in options as such:

`forecast -m graphcast --lead_time 24 --initial_conditions cds --date 20240330`

#### Bare metal

1. You will need a NVIDIA GPU with at least 16GB memory, ideally 24GB. We are working on quantization as well so that in the future it would be possible to run simulations with much less compute. Have an environment set with Python +3.10, Pytorch 2.2.2 and CUDA 11.8. Or if easier start with the docker image: `pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel`.
2. Install conda (miniconda for instance). Then run in that environment:

```bash
conda create -y -n skyenv python=3.10
conda activate skyenv
./build.sh
```

#### vast.ai setup

1. Find a machine you like RTX3090 or above with at least 24GB memory. Make sure you have good bandwith (+500MB/s).
2. Select the instance template from [here](https://cloud.vast.ai/?ref_id=128656&template_id=1883215a8487ec6ea9ad68a7cdb38c5e).
3. Then clone the repo and `pip install -e . && pip install -r requirements.txt`

## Run forecasts with different models, initial conditions, dates

For each run, you will first pull the initial conditions of your interest (most recent one by default), then the model will run for the desired time step. Initial conditions are pulled from GFS, ECMWF IFS (Operational) or CDS (ERA5 Reanalysis Dataset).

If you are using CDS initial conditions, then you will need a [CDS](https://cds.climate.copernicus.eu/user/login?destination=%2Fcdsapp%23!%2Fdataset%2Freanalysis-era5-single-levels) API key in your `.env` –`cp .env.example` and paste.

### Examples

All examples are from local setup, but you can run them as it is if you just change `forecast` to `modal run modal/forecast.py` and also make snake case kebab-case -i.e. `model_name` to `model-name`.

Example 1: Forecast using `graphcast` model, with ERA5 initial conditions, starting from 2024-04-30T00:00:00 and with a lead time of a week (forecast for the next week, i.e. 168 hours):

```bash
forecast --model_name graphcast --initial_conditions cds --date 20240403 -output_dir s3://skyrim-dev --lead_time 168
```

or in modal:

```bash
modal run modal/forecast.py --model-name graphcast --initial-conditions cds --date 20240403 -output-dir s3://skyrim-dev --lead-time 168
```

## Supported initial conditions and caveats

1. GFS
2. ECMWF IFS
3. ERA5 Re-analysis Dataset

## Large weather models supported

Currently supported models are:

- [x] [Graphcast](https://arxiv.org/abs/2212.12794)
- [x] [Pangu](https://arxiv.org/abs/2211.02556)
- [x] [Fourcastnet](https://arxiv.org/abs/2202.11214) (v1 & v2)
- [x] [DLWP](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020MS002109)
- [ ] [Fuxi](https://www.nature.com/articles/s41612-023-00512-1)
- [ ] [MetNet-3](https://arxiv.org/abs/2306.06079)

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
