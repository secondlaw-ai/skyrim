# Base image with CUDA support
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel
WORKDIR /app
SHELL ["/bin/bash", "-c"]
RUN conda create -y -n sky python=3.10 && \
    echo "source activate sky" >> ~/.bashrc

# Clone the Earth2Mip repository and checkout a specific commit
RUN git clone https://github.com/NVIDIA/earth2mip.git && \
    cd earth2mip && \
    git checkout 86b11fe

# Activate Conda environment and install dependencies
RUN source activate sky && \
    pip install . && \
    pip install -r earth2mip/requirements.txt && \
    pip install ruamel.yaml && \
    pip install earth2mip/[pangu,graphcast]

# Install specific Jax library for CUDA 11
RUN source activate sky && \
    pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.16+cuda11.cudnn86-cp310-cp310-manylinux2014_x86_64.whl

# Reinstall PyTorch with a specific CUDA version
RUN source activate sky && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall

# Clone and install NVIDIA Apex with custom build options
RUN source activate sky && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" ./ && \
    cd .. && \
    rm -rf ./apex  # Clean up Apex source after installation

# Run command to execute script with Python in the Conda environment
CMD ["bash", "-c", "source activate sky && python run.py -m fcn"]