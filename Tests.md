# A minimum repro on `num_workers` using Docker

## Build docker image
First, run `nivdia-smi` to check CUDA driver version, and modify `Dockerfile` to choose the right `nvidia/cuda` base image accordingly.  The default base image is for cuda 11.2.  See all nvidia/cuda tags here: https://hub.docker.com/r/nvidia/cuda/tags

Build the image.  It will take a while depending on your system and connection speed.

    docker build -t torchrec-devel .

## Launch docker image
The following command will create an interactive session.  It's important to increase the shared memory size since the default (64MB) is way too small.

    docker run --rm -it --gpus all --shm-size 64G torchrec-devel

## Test dlrm example
First, change to the dlrm example folder
    cd torchrec/torchrec/examples/dlrm

Run the example (change `-j 1x4` to match the number of GPUs, e.g., to `-j1x2` if there are 2 GPUs)

    torchx run --scheduler local_cwd dist.ddp -j 1x4 --script dlrm_main.py

Set `num_workers` to 0 and rerun the example

    torchx run --scheduler local_cwd dist.ddp -j 1x4 --script dlrm_main.py -- --num_workers 0
