FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel

# Set environment variables for Git username and token
ARG GIT_USERNAME
ARG GIT_TOKEN

# Use environment variables to configure Git and clone the repository
RUN git clone https://${GIT_USERNAME}:${GIT_TOKEN}@github.com/m13uz/skyrim.git

# Clone the Earth2Mip repository and checkout a specific commit
RUN git clone https://github.com/NVIDIA/earth2mip.git && \
    cd earth2mip && \
    git checkout 86b11fe && \
    pip install . && \
    pip install -r requirements.txt && \
    pip install ruamel.yaml && \
    pip install .[pangu,graphcast]

# Install specific Jax library from a URL
RUN pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.16+cuda11.cudnn86-cp310-cp310-manylinux2014_x86_64.whl

# Install PyTorch with a specific CUDA version (2.2.2+11.8 for CUDA 11.8)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall

# Clone and install NVIDIA Apex with custom build options
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" ./ && \
    cd .. && \
    rm -rf ./apex  # Clean up Apex source after installation

# docker build --build-arg GIT_USERNAME="your_username" --build-arg GIT_TOKEN="your_token" -t your_image_name .
CMD ['python', 'run.py', '-m', 'fcn']