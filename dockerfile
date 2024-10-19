FROM nvcr.io/nvidia/pytorch:24.02-py3

ARG DEBIAN_FRONTEND=noninteractive
ARG BRANCH='r2.0.0rc0'

WORKDIR /home/

ENV PATH="${PATH}:/home/bin"

ENV FORCE_CUDA="1"

RUN pip install git+https://github.com/NVIDIA/dllogger@v1.0.0#egg=dllogger
