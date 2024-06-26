conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia \
    && conda install eccodes python-eccodes -c conda-forge \
    && pip install onnx onnxruntime-gpu==1.17.1 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
    && pip install "makani[all] @ git+https://github.com/NVIDIA/modulus-makani.git@v0.1.0"
