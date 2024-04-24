<h1 align="center">
 
 <a href="https://www.secondlaw.xyz">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/skyrim_22.png"/>
    <img src="./assets/skyrim_22.png"/>
  </picture>
 </a>
 <br></br>
skyrim
<br></br>

</h1>
<p align="center">

🌎🌍🌍 Best in class weather forecasting for all.

Run masssive ensembles, simulations and build fine-tuned weather models on top of state-of-the-art foundational weather models.

From SecondLaw Reseach✌️🔥.

</p>

# Getting Started

## Run your first forecast

Currently, all supported models are running on the ECMWF ERA5 initial conditions. For each run, you will first pull the initial conditions of your interest (most recent one by default), then the model will run for the desired time step.

To run for the next 24h given the most recent ERA5 initial conditions using `pangu` model:

`python run.py --model pangu --date 20180101`

See supported models section for more on the models.

Importing Skyrim in python:

```python
from skyrim import Skyrim

# to see all the available models
print(Skyrim.list_available_models())

# initialize pangu model
model = Skyrim("pangu")

# rollout predictions from a date and time of your choice
pred, output_paths = model.predict(date="20180101", time="0000",lead_time=24, save=True)

```

# Installation

You will need a NVIDIA GPU with at least 16GB memory. Recommended setup is to start with a docker base image `pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel`. Then run the `build.sh` to be fully setup.

### License

For detailed information regarding licensing, please refer to the license details provided on each model's main homepage, which we link to from each of the corresponding components within our repository.

- **Pangu Weather** : [Original](https://github.com/198808xc/Pangu-Weather), [ECMWF](https://github.com/ecmwf-lab/ai-models-panguweather), [NVIDIA](https://github.com/NVIDIA/earth2mip)

- **FourcastNet** : [Original](https://github.com/NVlabs/FourCastNet), [ECMWF](https://github.com/ecmwf-lab/ai-models-fourcastnetv2),[NVIDIA](https://github.com/NVIDIA/earth2mip)

- **Graphcast** : [Original](https://github.com/google-deepmind/graphcast), [ECMWF](https://github.com/ecmwf-lab/ai-models-graphcast), [NVIDIA](https://github.com/NVIDIA/earth2mip)

- **Fuxi**: [Original](https://github.com/tpys/FuXi)

## Roadmap
- [x] ensemble prediction
- [ ] interface to fetch real-time NWP-based predictions, e.g. via ECMWF API.
- [ ] global model performance comparison across various regions and parameters.
- [ ] finetuning api that trains a downstream model on top of features coming from a global/foundation model, that is optimized wrt to a specific criteria and region
- [ ] model quantization and its effect on model efficiency and accuracy.

This README will be updated regularly to reflect the progress and integration of new models or features into the library. It serves as a guide for internal development efforts and aids in prioritizing tasks and milestones.

## Acknowledgements
Skyrim is built on top of NVIDIA's [earth2mip](https://github.com/NVIDIA/earth2mip) and ECMWF's [ai-models](https://github.com/ecmwf-lab/ai-models). Definitely check them out!

## Other Useful Resources
* [🌍 Awesome Large Weather Models (LWMs) | AI for Earth (AI4Earth) | AI for Science (AI4Science)](http://github.com/jaychempan/Awesome-LWMs)
* [Climate Data Store](https://cds.climate.copernicus.eu/)
* 
