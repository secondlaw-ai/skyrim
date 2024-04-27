# Base image with CUDA support
FROM nvcr.io/nvidia/modulus/modulus:23.11
RUN git clone https://github.com/secondlaw-ai/skyrim
WORKDIR /skyrim
RUN pip install .
RUN pip install -r requirements.txt
# default command
CMD ["python", "run.py", "-m", "pangu", '-ic', 'ifs']