FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda install -y pytorch cudatoolkit=11.3 -c pytorch-nightly \
    && conda install -y numpy torchmetrics iopath scikit-build jinja2 ninja cmake git -c conda-forge

RUN git clone --recursive https://github.com/pytorch/FBGEMM \
    && cd FBGEMM/fbgemm_gpu/ \
    && cp /usr/include/x86_64-linux-gnu/cudnn_v8.h /usr/include/x86_64-linux-gnu/cudnn.h \
    && cp /usr/include/x86_64-linux-gnu/cudnn_version_v8.h /usr/include/x86_64-linux-gnu/cudnn_version.h \
    && python setup.py install -DCUDNN_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcudnn.so -DCUDNN_INCLUDE_PATH=/usr/include/x86_64-linux-gnu/

RUN git clone --recursive https://github.com/facebookresearch/torchrec \
    && cd torchrec/ \
    && python setup.py build develop --skip_fbgemm

RUN pip install torchx-nightly

