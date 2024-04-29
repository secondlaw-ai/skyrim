<h1 align="center">
 
 <a href="https://www.secondlaw.xyz">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/skyrim_banner_1.png"/>
    <img height=320 src="./assets/skyrim_banner_1.png"/>
  </picture>
 </a>
 <br></br>

</h1>
<p align="center">

🌎 Best-in-class weather forecasting for all.

Run masssive ensembles, simulations and fine-tuned models on top of state-of-the-art foundational weather models.

</p>

# Getting Started

Skyrim allows you to run any large weather model with a consumer grade GPU. Until very recently, weather forecasts were run in 100K+ CPU HPC clusters, solving massive numerical models. Within last 2 years, open-source foundation models trained on weather simulation datasets surpassed the skill level of these numerical models. Our goal is to make these models accessible by providing a well maintained infrastructure.

## Installation

You will need a NVIDIA GPU with at least 16GB memory, ideally 24GB. We are working on quantization as well so that in the future it would be possible to run simulations with much less compute.

### Bare metal

Currently, best is to set up your environment using Docker.

1. Start with a docker base image `pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel`.
2. Then clone the repo and run `./build.sh`.

### vast.ai setup

1. Find a machine you like RTX3090 or above with at least 24GB memory. Make sure you have good bandwith (+500MB/s).
2. Select the instance template from [here](https://cloud.vast.ai/?ref_id=128656&template_id=1883215a8487ec6ea9ad68a7cdb38c5e).
3. Then clone the repo and `pip install -r requirements.txt`

## Your first forecast

For each run, you will first pull the initial conditions of your interest (most recent one by default), then the model will run for the desired time step. Initial conditions are pulled from GFS, ECMWF IFS (Operational) or CDS (ERA5 Reanalysis Dataset).

Example: To run for the next 24h given the most recent ECWMF IFS initial conditions using `pangu` model:

`python run.py --model pangu -ic ifs`

Or in Python:

```python
from skyrim import Skyrim

# to see all the available models
print(Skyrim.list_available_models())

# initialize pangu model
model = Skyrim("pangu")

# rollout predictions from a date and time of your choice
pred, output_paths = model.predict(date="20180101", time="0000",lead_time=24, save=True)
```

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

## Acknowledgements

Skyrim is built on top of NVIDIA's [earth2mip](https://github.com/NVIDIA/earth2mip) and ECMWF's [ai-models](https://github.com/ecmwf-lab/ai-models). Definitely check them out!

## Other Useful Resources

- [🌍 Awesome Large Weather Models (LWMs) | AI for Earth (AI4Earth) | AI for Science (AI4Science)](http://github.com/jaychempan/Awesome-LWMs)
- [Climate Data Store](https://cds.climate.copernicus.eu/)
- [Open Climate Fix](https://github.com/openclimatefix)
- [Herbie](https://github.com/blaylockbk/Herbie)
