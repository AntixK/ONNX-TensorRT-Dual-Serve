FROM nvcr.io/nvidia/pytorch:24.08-py3
# FROM nvcr.io/nvidia/tritonserver:24.09-py3

ARG DEBIAN_FRONTEND=noninteractive
ARG BRANCH='r2.0.0rc0'

WORKDIR /home/

ENV PATH="${PATH}:/home/bin"

ENV FORCE_CUDA="1"

RUN apt-get update && \
    apt-get install -y git

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
