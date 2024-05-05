#!/bin/bash
# Create a new Conda environment with Python 3.10
# starting from the base image: pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel
git clone https://github.com/secondlaw-ai/skyrim.git && cd skyrim && pip install .
pip install ruamel.yaml
# Install Jax library with support for CUDA 11
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.16+cuda11.cudnn86-cp310-cp310-manylinux2014_x86_64.whl
# Reinstall PyTorch with a specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
# Clone and install NVIDIA Apex with custom build options
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
cd ..
rm -rf ./apex  # Clean up Apex source after installation
# finally install all other deps:
pip install -r requirements.txt