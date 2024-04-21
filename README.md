
<div align="center">
    <h1>Skyrim</h1>
    <img src="./assets/output_animation_ortho.gif" alt="skyrim" width="150px">
</div>


This document serves as an internal progress tracker for the development of a wrapper library that integrates several publicly available models. The goal is to streamline the usage and evaluation of these models for our projects.

## Installation

Pain is real. Suffering is optional. 
TBA.

## Models to Wrap
### License

For detailed information regarding licensing, please refer to the license details provided on each model's main homepage, which we link to from each of the corresponding components within our repository.

- **Pangu Weather** : [Original](https://github.com/198808xc/Pangu-Weather), [ECMWF](https://github.com/ecmwf-lab/ai-models-panguweather), [NVIDIA](https://github.com/NVIDIA/earth2mip)

- **FourcastNet** : [Original](https://github.com/NVlabs/FourCastNet), [ECMWF](https://github.com/ecmwf-lab/ai-models-fourcastnetv2),[NVIDIA](https://github.com/NVIDIA/earth2mip)

- **Graphcast** : [Original](https://github.com/google-deepmind/graphcast), [ECMWF](https://github.com/ecmwf-lab/ai-models-graphcast), [NVIDIA](https://github.com/NVIDIA/earth2mip)

- **Fuxi**: [Original](https://github.com/tpys/FuXi)

## Roadmap
- [ ] interface to fetch real-time NWP-based predictions, e.g. via ECMWF API.
- [ ] global model performance comparison across various regions and parameters.
- [ ] ensemble prediction
- [ ] finetuning api that trains a downstream model on top of features coming from a global/foundation model, that is optimized wrt to a specific criteria and region
- [ ] model quantization and its effect on model efficiency and accuracy.

This README will be updated regularly to reflect the progress and integration of new models or features into the library. It serves as a guide for internal development efforts and aids in prioritizing tasks and milestones.
